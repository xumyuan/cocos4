/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { cclegacy, Mat4, Vec3 } from '../../core';
import { API, Attribute, AttributeName, deviceManager, Format, PrimitiveMode } from '../../gfx';
import { Material, RenderingSubMesh } from '../../asset/assets';
import { scene } from '../../render-scene';
import { Particle } from '../particle';
import { ParticleSpace } from '../enum';
import type { ParticleSystem } from '../particle-system';
import type ParticleSystemRenderer from './particle-system-renderer-data';
import ParticleSystemRendererCPU from './particle-system-renderer-cpu';
import type { ParticleCountInfo } from './particle-system-renderer-base';
import ParticleComputeModel from '../models/particle-compute-model';
import { ParticleComputeSimulation } from '../compute/particle-compute-simulation';
import { compileParticleComputeBirth, ParticleComputeBirthBatch } from '../compute/particle-compute-birth';
import { compileParticleComputeModules } from '../compute/particle-compute-modules';
import { compileParticleComputeTrail, ParticleComputeTrail } from '../compute/particle-compute-trail';
import type { CompiledParticleComputeTrail } from '../compute/particle-compute-trail';
import { getParticleComputeModuleConfigKey, getParticleComputeTrailConfigKey } from '../compute/compute-config-key';

/** GPU time values are f32; anything that overflows or is NaN becomes 0. */
function finiteF32 (value: number): number {
    return Number.isFinite(Math.fround(value)) ? value : 0;
}

/**
 * Reuses CPU material/attribute setup while all particle state is GPU-owned.
 * The inherited CPU Particle pool remains empty; emit and update bypass it.
 */
export default class ParticleSystemRendererCompute extends ParticleSystemRendererCPU {
    private _simulation: ParticleComputeSimulation | null = null;
    private readonly _births: ParticleComputeBirthBatch[] = [];
    private _mayHaveParticles = false;
    private _lastCount = 0;
    private _countValid = true;
    private _countTick = 0;
    private _epoch = 0;
    private _birthRevision = 0;
    private _committedBirthRevision = 0;
    private _pollPending = false;
    private _disposed = false;
    private readonly _gravityVector = new Vec3();
    private readonly _gravityMatrix = new Mat4();
    private readonly _world = new Mat4();
    private _trail: ParticleComputeTrail | null = null;
    private _computeTrailModel: scene.Model | null = null;
    private _trailSubMesh: RenderingSubMesh | null = null;
    private _moduleConfigKey: string | null = null;
    private _compiledModuleData: Float32Array | null = null;
    private _trailConfigKey: string | null = null;
    private _compiledTrail: CompiledParticleComputeTrail | null = null;
    private _trailEnabled = false;

    public constructor (info: ParticleSystemRenderer) {
        super(info);
        if (deviceManager.gfxDevice.gfxAPI !== API.WEBGPU) throw new Error('Compute particles require WebGPU');
        this._useInstance = true;
    }

    public get supportsBatchEmission (): boolean { return true; }
    public get ownsTrail (): boolean { return true; }

    protected _initModel (): void {
        if (!this._model && this._particleSystem) {
            this._model = cclegacy.director.root.createModel(ParticleComputeModel);
            this._model!.setCapacity(this._particleSystem.capacity);
            this._model!.visFlags = this._particleSystem.visibility;
        }
    }

    public onInit (ps: ParticleSystem): void {
        try {
            super.onInit(ps);
            this._simulation = new ParticleComputeSimulation(deviceManager.gfxDevice, Math.max(1, ps.capacity), finiteF32(ps.time));
            this._bindState();
        } catch (error) {
            if (ps.processor === this) ps.processor = null!;
            this.onDestroy();
            throw error;
        }
    }

    public onCapacityChanged (capacity: number): void {
        const target = Math.max(1, capacity);
        const ps = this._particleSystem;
        if (ps?._trailModule?.enable) {
            ParticleComputeTrail.validateBudget(deviceManager.gfxDevice, target, ps.renderer.computeTrailMaxPoints);
        }
        const previous = this._simulation;
        if (!previous || previous.capacity === target) {
            super.onCapacityChanged(capacity);
            return;
        }
        const next = new ParticleComputeSimulation(deviceManager.gfxDevice, target, finiteF32(this._particleSystem?.time ?? 0));
        try {
            super.onCapacityChanged(capacity);
        } catch (error) {
            next.destroy();
            throw error;
        }
        // Reset GPU-owned state immediately so subsequent same-frame emit() calls
        // land on the new pool instead of being wiped in updateParticles().
        this.clear();
        previous.destroy();
        this._simulation = next;
        this._bindState();
    }

    public emitBatch (count: number, dt: number): boolean {
        if (!Number.isFinite(count) || !Number.isFinite(dt)) throw new RangeError('Particle emit count and dt must be finite');
        const ps = this._particleSystem;
        if (!ps || count <= 0 || ps.capacity === 0) return true;
        if (this._births.length >= 1024) throw new RangeError('Too many particle emission batches before a simulation update');
        this._births.push(compileParticleComputeBirth(ps, Math.ceil(count), dt));
        this._birthRevision++;
        this._mayHaveParticles = true;
        this._countValid = false;
        if (this._model) this._model.enabled = true;
        return true;
    }

    public getFreeParticle (): Particle | null {
        throw new Error('Compute particles use emitBatch; individual CPU particle allocation is unavailable');
    }

    public setNewParticle (_particle: Particle): void {
        throw new Error('Compute particles use emitBatch; individual CPU particle upload is unavailable');
    }

    public updateParticles (dt: number): number {
        const ps = this._particleSystem;
        if (!ps || !this._simulation) return 0;
        const stepDt = Math.max(0, finiteF32(dt));
        const systemTime = finiteF32(ps.time);

        if (this._simulation.capacity !== Math.max(1, ps.capacity)) {
            // Preserve births emitted earlier in this update before a direct backing-field capacity edit.
            const pendingBirths = this._births.slice();
            this.onCapacityChanged(ps.capacity);
            if (pendingBirths.length > 0 && ps.capacity > 0) {
                this._births.push(...pendingBirths);
                this._birthRevision++;
                this._mayHaveParticles = true;
                this._countValid = false;
            }
        }

        const material = ps.getMaterialInstance(0) || this.getDefaultMaterial();
        if (material) {
            this.updateScale(material.passes[0]);
            this.updateRotation(material.passes[0]);
        }

        const moduleOptions = { renderMode: ps.renderer.renderMode };
        const moduleKey = getParticleComputeModuleConfigKey(ps, moduleOptions);
        if (!this._compiledModuleData || moduleKey === null || moduleKey !== this._moduleConfigKey) {
            this._compiledModuleData = compileParticleComputeModules(ps, moduleOptions).data;
            this._moduleConfigKey = moduleKey;
        }

        this._gravityVector.set(0, -9.8, 0);
        if (ps.simulationSpace === ParticleSpace.Local) {
            if (ps.node.parent) {
                Mat4.fromQuat(this._gravityMatrix, ps.node.parent.worldRotation);
                this._gravityMatrix.transpose();
                Vec3.transformMat4(this._gravityVector, this._gravityVector, this._gravityMatrix);
            }
            Mat4.fromQuat(this._gravityMatrix, ps.node.getRotation());
            this._gravityMatrix.transpose();
            Vec3.transformMat4(this._gravityVector, this._gravityVector, this._gravityMatrix);
        }

        this._simulation.tick(
            stepDt,
            this._births,
            this._compiledModuleData,
            systemTime,
            [this._gravityVector.x, this._gravityVector.y, this._gravityVector.z],
        );
        if (this._births.length > 0) this._committedBirthRevision = this._birthRevision;
        this._births.length = 0;
        this._bindState();
        this._updateComputeTrail(stepDt, systemTime);
        if (this._model) this._model.enabled = this._mayHaveParticles;
        // No future automatic births: a delayed zero is authoritative until an
        // emit/clear/restart changes the revision or epoch. Rendering never waits.
        if (!ps.isEmitting && this._mayHaveParticles && !this._pollPending) {
            this._pollPending = true;
            const pollEpoch = this._epoch;
            const finish = (): void => { if (this._epoch === pollEpoch) this._pollPending = false; };
            // Both paths release this epoch's poll without requiring ES2018 Promise.finally.
            this.getParticleCountAsync().then(finish, finish);
        }
        return this._mayHaveParticles ? -1 : 0;
    }

    private _bindState (): void {
        if (this._simulation && this._model) {
            (this._model as ParticleComputeModel).setSimulationBuffers(
                this._simulation.stateBuffer,
                this._simulation.counterBuffer,
                this._simulation.tickId,
                this._simulation.activeBound,
            );
        }
    }

    public updateRenderData (): void { /* RenderPrepare reads GPU state in beforeRender. */ }

    public updateTrailMaterial (): void {
        super.updateTrailMaterial();
        const material = this._particleSystem?.getMaterialInstance(1) || this.getDefaultTrailMaterial();
        if (this._computeTrailModel && material) this._computeTrailModel.setSubModelMaterial(0, material);
    }

    public onRebuildPSO (index: number, material: Material): void {
        super.onRebuildPSO(index, material);
        if (index === 1 && this._computeTrailModel) this._computeTrailModel.setSubModelMaterial(0, material);
    }

    public getOwnedTrailModel (): scene.Model | null {
        return this._particleSystem?._trailModule?.enable ? this._computeTrailModel : null;
    }

    public setTrailEnabled (enabled: boolean): void {
        if (!enabled) this._disableComputeTrail();
    }

    private _disableComputeTrail (): void {
        if (this._trailEnabled) this._trail?.clearHistory();
        this._trailEnabled = false;
        if (this._computeTrailModel) {
            this._computeTrailModel.enabled = false;
            if (this._computeTrailModel.scene) this._computeTrailModel.scene.removeModel(this._computeTrailModel);
        }
    }

    public attachToScene (): void {
        super.attachToScene();
        if (this._computeTrailModel && this._particleSystem) {
            if (this._computeTrailModel.scene) this._computeTrailModel.scene.removeModel(this._computeTrailModel);
            this._particleSystem._getRenderScene().addModel(this._computeTrailModel);
        }
    }

    public detachFromScene (): void {
        super.detachFromScene();
        if (this._computeTrailModel?.scene) this._computeTrailModel.scene.removeModel(this._computeTrailModel);
    }

    public onDisable (): void {
        super.onDisable();
        this._trail?.clearHistory();
    }

    private _updateComputeTrail (dt: number, systemTime: number): void {
        const ps = this._particleSystem;
        const simulation = this._simulation;
        const module = ps?._trailModule;
        if (!ps || !simulation) {
            this._destroyComputeTrail();
            return;
        }
        if (!module?.enable) {
            this._disableComputeTrail();
            return;
        }
        this._trailEnabled = true;
        const trailKey = getParticleComputeTrailConfigKey(module);
        const recompiled = !this._compiledTrail || trailKey === null || trailKey !== this._trailConfigKey;
        if (recompiled) {
            this._compiledTrail = compileParticleComputeTrail(module);
            this._trailConfigKey = trailKey;
        }
        const compiled = this._compiledTrail!;
        const maxPoints = ps.renderer.computeTrailMaxPoints;
        if (!this._trail || this._trail.capacity !== simulation.capacity || this._trail.maxPoints !== maxPoints) {
            this._destroyComputeTrail();
            this._trail = new ParticleComputeTrail(deviceManager.gfxDevice, simulation.capacity, maxPoints, compiled);
            this.updateTrailMaterial();
            const material = ps.getMaterialInstance(1) || this.getDefaultTrailMaterial();
            if (!material) throw new Error('Compute Trail material is unavailable');
            this._computeTrailModel = cclegacy.director.root.createModel(scene.Model);
            const model = this._computeTrailModel!;
            model.node = model.transform = ps.node;
            model.visFlags = ps.visibility;
            // Buffers are borrowed from the Trail driver; its destroy owns them.
            this._trailSubMesh = new RenderingSubMesh([this._trail.vertexBuffer], [
                new Attribute(AttributeName.ATTR_POSITION, Format.RGB32F),
                new Attribute(AttributeName.ATTR_TEX_COORD, Format.RGBA32F),
                new Attribute(AttributeName.ATTR_TEX_COORD1, Format.RGB32F),
                new Attribute(AttributeName.ATTR_COLOR, Format.RGBA8, true),
            ], PrimitiveMode.TRIANGLE_LIST, this._trail.indexBuffer, this._trail.indirectBuffer, false);
            this._trailSubMesh.iaInfo.gpuIndirect = { offset: 0, count: 1, stride: 20 };
            model.initSubModel(0, this._trailSubMesh, material);
            if (this._model?.scene) ps._getRenderScene().addModel(model);
        } else if (recompiled) {
            this._trail.configure(compiled);
            this.updateTrailMaterial();
        }
        ps.node.getWorldMatrix(this._world);
        this._trail.update(
            simulation.stateBuffer,
            simulation.counterBuffer,
            dt,
            this._world,
            systemTime,
            ps.simulationSpace,
            simulation.activeBound ?? simulation.capacity,
        );
        this._computeTrailModel!.enabled = this._mayHaveParticles;
        if (this._model?.scene && !this._computeTrailModel!.scene) ps._getRenderScene().addModel(this._computeTrailModel!);
    }

    private _destroyComputeTrail (): void {
        if (this._computeTrailModel) {
            if (this._computeTrailModel.scene) this._computeTrailModel.scene.removeModel(this._computeTrailModel);
            cclegacy.director.root.destroyModel(this._computeTrailModel);
            this._computeTrailModel = null;
        }
        // RenderingSubMesh.destroy would destroy borrowed buffers a second time.
        this._trailSubMesh = null;
        this._trail?.destroy();
        this._trail = null;
    }

    public beforeRender (): void {
        if (!this._model || !this._simulation || !this.hasAliveParticles()) return;
        (this._model as ParticleComputeModel).setTextureAnimationEnabled(!!this._particleSystem?._textureAnimationModule?.enable);
        this._bindState();
        this._model.updateIA(0);
    }

    public getParticleCount (): number { return this._lastCount; }
    public hasAliveParticles (): boolean { return this._mayHaveParticles || this._births.length > 0; }
    public getParticleCountInfo (): ParticleCountInfo {
        return { value: this._lastCount,
            valid: this._countValid,
            stale: !this._countValid || this._births.length > 0 || this._countTick !== (this._simulation?.tickId ?? 0),
            tickId: this._countTick,
            epoch: this._epoch };
    }

    public async getParticleCountAsync (): Promise<number> {
        const simulation = this._simulation;
        if (!simulation) return 0;
        const epoch = this._epoch;
        const revision = this._birthRevision;
        const committedRevision = this._committedBirthRevision;
        const hadPendingBirths = this._births.length > 0;
        const tick = simulation.tickId;
        const count = await simulation.readCount();
        if (!this._disposed && this._simulation === simulation && epoch === this._epoch && tick >= this._countTick) {
            if (revision === this._birthRevision && committedRevision === this._committedBirthRevision) {
                this._lastCount = count;
                this._countValid = true;
                this._countTick = tick;
                if (count === 0 && !hadPendingBirths && this._births.length === 0) this._mayHaveParticles = false;
            } else if (tick > this._countTick || !this._countValid) {
                // New births arrived or were committed while reading back: update the latest GPU count
                // snapshot without marking it authoritative for drain termination.
                this._lastCount = count;
                this._countTick = tick;
            }
        }
        return count;
    }

    public clear (): void {
        this._epoch++;
        this._pollPending = false;
        this._birthRevision++;
        this._committedBirthRevision = this._birthRevision;
        this._births.length = 0;
        this._simulation?.clear();
        this._trail?.clear();
        this._model?.clear();
        if (this._model) this._model.enabled = false;
        if (this._computeTrailModel) this._computeTrailModel.enabled = false;
        this._lastCount = 0;
        this._countValid = true;
        this._countTick = 0;
        this._mayHaveParticles = false;
    }

    public setUseInstance (value: boolean): void {
        if (!value) throw new Error('Compute particle materials require instanced vertex attributes');
    }

    public onDestroy (): void {
        if (this._disposed) return;
        this._disposed = true;
        this._epoch++;
        this._births.length = 0;
        this._destroyComputeTrail();
        this._simulation?.destroy();
        this._simulation = null;
        super.onDestroy();
    }
}
