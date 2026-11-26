/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { EDITOR_NOT_IN_PREVIEW } from 'internal:constants';
import { cclegacy, Mat4, Quat, v3, Vec2, Vec3, Vec4, warn } from '../../core';
import { API, Attribute, AttributeName, deviceManager, Format, FormatInfos, PrimitiveMode } from '../../gfx';
import { builtinResMgr } from '../../asset/asset-manager';
import { Material, RenderingSubMesh, type Texture2D } from '../../asset/assets';
import { type Pass, scene } from '../../render-scene';
import type { Camera } from '../../render-scene/scene/camera';
import type { MacroRecord } from '../../render-scene/core/pass-utils';
import { type IMaterialInstanceInfo, MaterialInstance } from '../../render-scene/core/material-instance';
import { type IParticleModule, type Particle, PARTICLE_MODULE_NAME } from '../particle';
import { ParticleAlignmentSpace, ParticleRenderMode, ParticleSpace } from '../enum';
import type { ParticleSystem } from '../particle-system';
import type ParticleSystemRenderer from './particle-system-renderer-data';
import { type ParticleCountInfo, ParticleSystemRendererBase } from './particle-system-renderer-base';
import ParticleComputeModel from '../models/particle-compute-model';
import { ParticleComputeSimulation } from '../compute/particle-compute-simulation';
import { compileParticleComputeBirth, ParticleComputeBirthBatch } from '../compute/particle-compute-birth';
import { compileParticleComputeModules } from '../compute/particle-compute-modules';
import { compileParticleComputeTrail, ParticleComputeTrail } from '../compute/particle-compute-trail';
import type { CompiledParticleComputeTrail } from '../compute/particle-compute-trail';
import { ParticleComputeModuleConfigTracker, ParticleComputeTrailConfigTracker } from '../compute/compute-config-key';

const _gravityVec = new Vec3();
const _gravityMatrix = new Mat4();

/** GPU time values are f32; anything that overflows or is NaN becomes 0. */
function finiteF32 (value: number): number {
    return Number.isFinite(Math.fround(value)) ? value : 0;
}

function sameModuleData (previous: Float32Array, next: Float32Array): boolean {
    if (previous.length !== next.length) return false;

    // The compiler rejects non-finite values. Object.is also distinguishes signed zero.
    for (let i = 0; i < next.length; ++i) {
        if (!Object.is(previous[i], next[i])) return false;
    }
    return true;
}

// TODO: Consider extracting the rendering setup shared by the CPU and Compute particle renderers.
// BEGIN verbatim copy from particle-system-renderer-cpu.ts
const _tempNodeScale = new Vec4();
const _node_rot = new Quat();

const CC_USE_WORLD_SPACE = 'CC_USE_WORLD_SPACE';
const CC_USE_EMBEDDED_ALPHA = 'CC_USE_EMBEDDED_ALPHA';

const CC_RENDER_MODE = 'CC_RENDER_MODE';
const ROTATION_OVER_TIME_MODULE_ENABLE = 'ROTATION_OVER_TIME_MODULE_ENABLE';
const USE_NORMAL_MAP = 'USE_NORMAL_MAP';
const INSTANCE_PARTICLE = 'CC_INSTANCE_PARTICLE';
const RENDER_MODE_BILLBOARD = 0;
const RENDER_MODE_STRETCHED_BILLBOARD = 1;
const RENDER_MODE_HORIZONTAL_BILLBOARD = 2;
const RENDER_MODE_VERTICAL_BILLBOARD = 3;
const RENDER_MODE_MESH = 4;

const ATTR_NORMAL = AttributeName.ATTR_NORMAL;
const ATTR_COLOR = AttributeName.ATTR_COLOR;
const ATTR_COLOR1 = AttributeName.ATTR_COLOR1;
const ATTR_TEX_COORD = AttributeName.ATTR_TEX_COORD;
const ATTR_TEX_COORD1 = AttributeName.ATTR_TEX_COORD1;
const ATTR_TEX_COORD2 = AttributeName.ATTR_TEX_COORD2;
const ATTR_TEX_COORD3 = AttributeName.ATTR_TEX_COORD3;
const ATTR_TEX_COORD4 = AttributeName.ATTR_TEX_COORD4;

function createAttribute (name: AttributeName, format: Format, isNormalized = false, stream = 0, isInstanced = false, location = 0): Attribute {
    return new Attribute(name, format, isNormalized, stream, isInstanced, location);
}

const _vertex_attrs_ins = [
    createAttribute(ATTR_TEX_COORD4, Format.RGBA32F, false, 0, true),    // position,frame idx
    createAttribute(ATTR_TEX_COORD1, Format.RGB32F, false, 0, true),     // size
    createAttribute(ATTR_TEX_COORD2, Format.RGB32F, false, 0, true),     // rotation
    createAttribute(ATTR_COLOR, Format.RGBA8, true, 0, true),            // color
    createAttribute(ATTR_TEX_COORD, Format.RGB32F, false, 1),            // uv
];

const _vertex_attrs_stretch_ins = [
    createAttribute(ATTR_TEX_COORD4, Format.RGBA32F, false, 0, true),    // position,frame idx
    createAttribute(ATTR_TEX_COORD1, Format.RGB32F, false, 0, true),     // size
    createAttribute(ATTR_TEX_COORD2, Format.RGB32F, false, 0, true),     // rotation
    createAttribute(ATTR_COLOR, Format.RGBA8, true, 0, true),            // color
    createAttribute(ATTR_COLOR1, Format.RGB32F, false, 0, true),         // particle velocity
    createAttribute(ATTR_TEX_COORD, Format.RGB32F, false, 1),            // uv
];

const _vertex_attrs_mesh_ins = [
    createAttribute(ATTR_TEX_COORD4, Format.RGBA32F, false, 0, true),    // particle position,frame idx
    createAttribute(ATTR_TEX_COORD1, Format.RGB32F, false, 0, true),     // size
    createAttribute(ATTR_TEX_COORD2, Format.RGB32F, false, 0, true),     // rotation
    createAttribute(ATTR_COLOR, Format.RGBA8, true, 0, true),            // particle color
    createAttribute(ATTR_TEX_COORD, Format.RGB32F, false, 1),            // mesh uv
    createAttribute(ATTR_TEX_COORD3, Format.RGB32F, false, 1),           // mesh position
    createAttribute(ATTR_NORMAL, Format.RGB32F, false, 1),               // mesh normal
    createAttribute(ATTR_COLOR1, Format.RGBA8, true, 1),                 // mesh color
];

const _matInsInfo: IMaterialInstanceInfo = {
    parent: null!,
    owner: null!,
    subModelIdx: 0,
};
// END verbatim copy from particle-system-renderer-cpu.ts

export default class ParticleSystemRendererCompute extends ParticleSystemRendererBase {
    // BEGIN verbatim copy from particle-system-renderer-cpu.ts
    private _defines: MacroRecord;
    private _frameTile_velLenScale: Vec4;
    private _tmp_velLenScale: Vec4;
    private _defaultMat: Material | null = null;
    private _node_scale: Vec3;
    private _defaultTrailMat: Material | null = null;
    private _uScaleHandle = 0;
    private _uLenHandle = 0;
    private _uNodeRotHandle = 0;
    private _alignSpace = ParticleAlignmentSpace.View;
    private _inited = false;

    public getDefaultTrailMaterial (): Material | null {
        return this._defaultTrailMat;
    }

    public updateAlignSpace (space: number): void {
        this._alignSpace = space;
    }

    public getDefaultMaterial (): Material | null {
        return this._defaultMat;
    }

    public updateRotation (pass: Pass | null): void {
        if (pass) {
            this.doUpdateRotation(pass);
        }
    }

    private doUpdateRotation (pass: Pass): void {
        const mode = this._renderInfo!.renderMode;
        if (mode !== ParticleRenderMode.Mesh && this._alignSpace === ParticleAlignmentSpace.View) {
            return;
        }

        const ps = this._particleSystem;
        if (mode === ParticleRenderMode.Mesh && ps?._shapeModule?.enable && ps._shapeModule.alignToDirection) {
            // Aligned particle rotations are already in simulation space. The mesh
            // shader only needs the remaining simulation-to-world rotation.
            if (ps.simulationSpace === ParticleSpace.Local) {
                ps.node.getWorldRotation(_node_rot);
            } else {
                _node_rot.set(0, 0, 0, 1);
            }
        } else if (this._alignSpace === ParticleAlignmentSpace.Local) {
            this._particleSystem?.node.getRotation(_node_rot);
        } else if (this._alignSpace === ParticleAlignmentSpace.World) {
            this._particleSystem?.node.getWorldRotation(_node_rot);
        } else if (this._alignSpace === ParticleAlignmentSpace.View) {
            // Quat.fromEuler(_node_rot, 0.0, 0.0, 0.0);
            _node_rot.set(0.0, 0.0, 0.0, 1.0);
            const cameraLst: Camera[] | undefined = this._particleSystem?.node.scene.renderScene?.cameras;
            if (cameraLst !== undefined) {
                for (let i = 0; i < cameraLst?.length; ++i) {
                    const camera: Camera = cameraLst[i];
                    // eslint-disable-next-line max-len
                    const checkCamera: boolean = !EDITOR_NOT_IN_PREVIEW ? (camera.visibility & this._particleSystem!.node.layer) === this._particleSystem!.node.layer : camera.name === 'Editor Camera';
                    if (checkCamera) {
                        Quat.fromViewUp(_node_rot, camera.forward);
                        break;
                    }
                }
            }
        } else {
            _node_rot.set(0.0, 0.0, 0.0, 1.0);
        }
        pass.setUniform(this._uNodeRotHandle, _node_rot);
    }

    public updateScale (pass: Pass | null): void {
        if (pass) {
            this.doUpdateScale(pass);
        }
    }

    private doUpdateScale (pass): void {
        const nodeScale = this._node_scale;
        switch (this._particleSystem?.scaleSpace) {
        case ParticleSpace.Local:
            this._particleSystem?.node.getScale(nodeScale);
            break;
        case ParticleSpace.World:
            this._particleSystem?.node.getWorldScale(nodeScale);
            break;
        default:
            break;
        }
        // NOTE: the `_node_scale` should be a Vec3, but we implement `scale` uniform property as a Vec4,
        // here we pass a temperate Vec4 object to prevent creating Vec4 object every time we set uniform.
        pass.setUniform(this._uScaleHandle, _tempNodeScale.set(nodeScale.x, nodeScale.y, nodeScale.z));
    }

    public onMaterialModified (index: number, material: Material): void {
        if (!this._inited) {
            return;
        }

        if (index === 0) {
            this.updateMaterialParams();
        } else {
            this.updateTrailMaterial();
        }
    }

    public updateVertexAttrib (): void {
        if (this._renderInfo!.renderMode !== ParticleRenderMode.Mesh) {
            return;
        }
        if (this._renderInfo!.mesh) {
            const format = this._renderInfo!.mesh.readAttributeFormat(0, AttributeName.ATTR_COLOR);
            if (format) {
                let type = Format.RGBA8;
                for (let i = 0; i < FormatInfos.length; ++i) {
                    if (FormatInfos[i].name === format.name) {
                        type = i;
                        break;
                    }
                }
                this._vertAttrs[7] = createAttribute(ATTR_COLOR1, type, true, !this._useInstance ? 0 : 1);
            } else { // mesh without vertex color
                const type = Format.RGBA8;
                this._vertAttrs[7] = createAttribute(ATTR_COLOR1, type, true, !this._useInstance ? 0 : 1);
            }
        }
    }

    private _setVertexAttribIns (): void {
        switch (this._renderInfo!.renderMode) {
        case ParticleRenderMode.StrecthedBillboard:
            this._vertAttrs = _vertex_attrs_stretch_ins.slice();
            break;
        case ParticleRenderMode.Mesh:
            this._vertAttrs = _vertex_attrs_mesh_ins.slice();
            break;
        default:
            this._vertAttrs = _vertex_attrs_ins.slice();
        }
    }

    public updateMaterialParams (): void {
        if (!this._particleSystem) {
            return;
        }

        const ps = this._particleSystem;
        const shareMaterial = ps.sharedMaterial;
        if (shareMaterial != null) {
            this._renderInfo!.mainTexture = shareMaterial.getProperty('mainTexture', 0) as Texture2D;
        }

        // Renderer can destroy the cached instance when replacing a preview material.
        if (ps.sharedMaterial == null && (!this._defaultMat || !this._defaultMat.passes.length)) {
            _matInsInfo.parent = builtinResMgr.get<Material>('default-particle-material');
            _matInsInfo.owner = this._particleSystem;
            _matInsInfo.subModelIdx = 0;
            this._defaultMat = new MaterialInstance(_matInsInfo);
            _matInsInfo.parent = null!;
            _matInsInfo.owner = null!;
            _matInsInfo.subModelIdx = 0;
            if (this._renderInfo!.mainTexture !== null) {
                this._defaultMat.setProperty('mainTexture', this._renderInfo!.mainTexture);
            }
        }
        const mat: Material = ps.getMaterialInstance(0) || this._defaultMat!;
        // A custom material may still have an invalid or unavailable effect.
        if (!mat || !mat.passes.length) {
            warn('Particle material has no render passes. Check its effect import and selected technique.');
            return;
        }
        if (ps.simulationSpace === ParticleSpace.World) {
            this._defines[CC_USE_WORLD_SPACE] = true;
        } else {
            this._defines[CC_USE_WORLD_SPACE] = false;
        }

        const pass = mat.passes[0];
        this._uScaleHandle = pass.getHandle('scale');
        this._uLenHandle = pass.getHandle('frameTile_velLenScale');
        this._uNodeRotHandle = pass.getHandle('nodeRotation');

        const renderMode = this._renderInfo!.renderMode;
        const vlenScale = this._frameTile_velLenScale;
        if (renderMode === ParticleRenderMode.Billboard) {
            this._defines[CC_RENDER_MODE] = RENDER_MODE_BILLBOARD;
        } else if (renderMode === ParticleRenderMode.StrecthedBillboard) {
            this._defines[CC_RENDER_MODE] = RENDER_MODE_STRETCHED_BILLBOARD;
            vlenScale.z = this._renderInfo!.velocityScale;
            vlenScale.w = this._renderInfo!.lengthScale;
        } else if (renderMode === ParticleRenderMode.HorizontalBillboard) {
            this._defines[CC_RENDER_MODE] = RENDER_MODE_HORIZONTAL_BILLBOARD;
        } else if (renderMode === ParticleRenderMode.VerticalBillboard) {
            this._defines[CC_RENDER_MODE] = RENDER_MODE_VERTICAL_BILLBOARD;
        } else if (renderMode === ParticleRenderMode.Mesh) {
            this._defines[CC_RENDER_MODE] = RENDER_MODE_MESH;
        } else {
            warn(`particle system renderMode ${renderMode} not support.`);
        }
        const textureModule = ps._textureAnimationModule;
        if (textureModule && textureModule.enable) {
            const texture = mat.getProperty('mainTexture', 0) as Texture2D | null;
            if (texture && texture.isAlphaAtlas) {
                textureModule.scaleNumTilesXY(2);
                this._defines[CC_USE_EMBEDDED_ALPHA] = true;
            }
            Vec4.copy(this._tmp_velLenScale, vlenScale); // fix textureModule switch bug
            Vec2.set(this._tmp_velLenScale, textureModule.numTilesX, textureModule.numTilesY);
            pass.setUniform(this._uLenHandle, this._tmp_velLenScale);
        } else {
            pass.setUniform(this._uLenHandle, vlenScale);
        }

        let enable = false;
        const roationModule = this._particleSystem._rotationOvertimeModule;
        enable = roationModule ? roationModule.enable : false;
        this._defines[ROTATION_OVER_TIME_MODULE_ENABLE] = enable;
        // Explicitly clear the override when the material omits the disabled macro.
        const materialDefines = (shareMaterial || mat.parent || mat).passes[0]?.defines;
        this._defines[USE_NORMAL_MAP] = !!materialDefines?.[USE_NORMAL_MAP];

        this._defines[INSTANCE_PARTICLE] = this._useInstance;

        const matIns: MaterialInstance | null = ps.getMaterialInstance(0);
        if (matIns === null || matIns === undefined) {
            ps.setSharedMaterial(mat.parent, 0); // set material[0] as material for ui-mesh-renderer to use
            ps.setMaterialInstance(mat, 0);
        }

        mat.recompileShaders(this._defines);
        if (this._model) {
            this._model.updateMaterial(mat);
        }
    }
    // END verbatim copy from particle-system-renderer-cpu.ts

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
    private _trail: ParticleComputeTrail | null = null;
    private _computeTrailModel: scene.Model | null = null;
    private _trailSubMesh: RenderingSubMesh | null = null;
    private _moduleVersion = 0;
    private readonly _gravityComponents: [number, number, number] = [0, -9.8, 0];
    private readonly _moduleTracker = new ParticleComputeModuleConfigTracker();
    private readonly _moduleOptions = { renderMode: ParticleRenderMode.Billboard };
    private _compiledModuleData: Float32Array | null = null;
    private _trailTracker: ParticleComputeTrailConfigTracker | null = null;
    private _compiledTrail: CompiledParticleComputeTrail | null = null;
    private _trailEnabled = false;

    public constructor (info: ParticleSystemRenderer) {
        super(info);
        if (deviceManager.gfxDevice.gfxAPI !== API.WEBGPU) {
            throw new Error('Compute particles require WebGPU');
        }
        this._useInstance = true;

        // BEGIN verbatim copy from particle-system-renderer-cpu.ts
        this._frameTile_velLenScale = new Vec4(1, 1, 0, 0);
        this._tmp_velLenScale = this._frameTile_velLenScale.clone();
        this._node_scale = v3();
        this._defines = {
            CC_USE_WORLD_SPACE: true,
            CC_USE_BILLBOARD: true,
            CC_USE_STRETCHED_BILLBOARD: false,
            CC_USE_HORIZONTAL_BILLBOARD: false,
            CC_USE_VERTICAL_BILLBOARD: false,
        };
        // END verbatim copy from particle-system-renderer-cpu.ts
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
            this._initModel();
            this.updateMaterialParams();
            this.updateTrailMaterial();
            this.setVertexAttributes();
            this._simulation = new ParticleComputeSimulation(deviceManager.gfxDevice, Math.max(1, ps.capacity), finiteF32(ps.time));
            this._bindState();
            this._inited = true;
        } catch (error) {
            if (ps.processor === this) {
                ps.processor = null!;
            }
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
        // Reset now so births emitted later in this frame belong to the new simulation.
        this._resetRendererState();
        previous.destroy();
        this._simulation = next;
        this._bindState();
    }

    public emitBatch (count: number, dt: number): boolean {
        if (!Number.isFinite(count) || !Number.isFinite(dt)) {
            throw new RangeError('Particle emit count and dt must be finite');
        }
        const ps = this._particleSystem;
        if (!ps || count <= 0 || ps.capacity === 0) return true;
        if (this._births.length >= 1024) {
            throw new RangeError('Too many particle emission batches before a simulation update');
        }
        this._births.push(compileParticleComputeBirth(ps, Math.ceil(count), dt));
        this._birthRevision++;
        this._mayHaveParticles = true;
        this._countValid = false;
        if (this._model) {
            this._model.enabled = true;
        }
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

        const moduleData = this._getModuleData(ps);

        _gravityVec.set(0, -9.8, 0);
        if (ps.simulationSpace === ParticleSpace.Local) {
            // Apply parent world rotation then local rotation to match ParticleSystemRendererCPU
            // even under non-uniform parent scale, where node.worldRotation is matrix-decomposed.
            if (ps.node.parent) {
                Mat4.fromQuat(_gravityMatrix, ps.node.parent.worldRotation);
                _gravityMatrix.transpose();
                Vec3.transformMat4(_gravityVec, _gravityVec, _gravityMatrix);
            }
            Mat4.fromQuat(_gravityMatrix, ps.node.getRotation());
            _gravityMatrix.transpose();
            Vec3.transformMat4(_gravityVec, _gravityVec, _gravityMatrix);
        }

        this._gravityComponents[0] = _gravityVec.x;
        this._gravityComponents[1] = _gravityVec.y;
        this._gravityComponents[2] = _gravityVec.z;
        this._simulation.tick(
            stepDt,
            this._births,
            moduleData,
            systemTime,
            this._gravityComponents,
            this._moduleVersion,
        );
        if (this._births.length > 0) {
            this._committedBirthRevision = this._birthRevision;
        }
        this._births.length = 0;
        this._bindState();
        this._updateComputeTrail(stepDt, systemTime);
        if (this._model) {
            this._model.enabled = this._mayHaveParticles;
        }

        // No future automatic births: a delayed zero is authoritative until an
        // emit/clear/restart changes the revision or epoch. Rendering never waits.
        if (!ps.isEmitting && this._mayHaveParticles && !this._pollPending) {
            this._pollPending = true;
            const pollEpoch = this._epoch;
            const finish = (): void => {
                if (this._epoch === pollEpoch) {
                    this._pollPending = false;
                }
            };
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

    public setVertexAttributes (): void {
        this._setVertexAttribIns();
        super.setVertexAttributes();
    }

    public updateRenderMode (): void {
        this.updateMaterialParams();
        this.setVertexAttributes();
    }

    public updateTrailMaterial (): Material | null {
        const ps = this._particleSystem;
        if (!ps?._trailModule?.enable) return null;

        let material: Material | null = ps.getMaterialInstance(1);
        if (!material) {
            if (!this._defaultTrailMat?.passes.length) {
                const parent = builtinResMgr.get<Material>('default-trail-material');
                this._defaultTrailMat = new MaterialInstance({ parent, owner: ps, subModelIdx: 1 });
            }
            material = this._defaultTrailMat;
            ps.setSharedMaterial(material.parent, 1);
            ps.setMaterialInstance(material, 1);
        }

        if (!material.passes.length) {
            this._destroyComputeTrail();
            return null;
        }

        material.recompileShaders({
            CC_USE_WORLD_SPACE: ps.simulationSpace === ParticleSpace.World || ps._trailModule.space === ParticleSpace.World,
        });
        this._computeTrailModel?.setSubModelMaterial(0, material);
        return material;
    }

    public onRebuildPSO (index: number, material: Material): void {
        if (index === 0) {
            this._model?.setSubModelMaterial(0, material);
        } else if (index === 1) {
            this._computeTrailModel?.setSubModelMaterial(0, material);
        }
    }

    public enableModule (name: string, _enabled: boolean, _module: IParticleModule): void {
        // TextureAnimationModule already refreshes the material in its enable setter.
        if (name === PARTICLE_MODULE_NAME.ROTATION) {
            this.updateMaterialParams();
        }
    }

    public getNoisePreview (out: number[], width: number, height: number): void {
        const ps = this._particleSystem;
        if (!ps) return;

        const noise = ps.noiseModule;
        if (noise?.enable) {
            noise.getNoisePreview(out, ps, width, height);
        }
    }

    public getOwnedTrailModel (): scene.Model | null {
        return this._particleSystem?._trailModule?.enable ? this._computeTrailModel : null;
    }

    public setTrailEnabled (enabled: boolean): void {
        if (!enabled) {
            this._disableComputeTrail();
        }
    }

    private _disableComputeTrail (): void {
        if (this._trailEnabled) {
            this._trail?.clearHistory();
        }
        this._trailEnabled = false;
        if (this._computeTrailModel) {
            this._computeTrailModel.enabled = false;
            if (this._computeTrailModel.scene) {
                this._computeTrailModel.scene.removeModel(this._computeTrailModel);
            }
        }
    }

    public attachToScene (): void {
        super.attachToScene();
        if (this._computeTrailModel && this._particleSystem) {
            if (this._computeTrailModel.scene) {
                this._computeTrailModel.scene.removeModel(this._computeTrailModel);
            }
            this._particleSystem._getRenderScene().addModel(this._computeTrailModel);
        }
    }

    public detachFromScene (): void {
        super.detachFromScene();
        if (this._computeTrailModel?.scene) {
            this._computeTrailModel.scene.removeModel(this._computeTrailModel);
        }
    }

    public onDisable (): void {
        super.onDisable();
        this._trail?.clearHistory();
    }

    private _getModuleData (ps: ParticleSystem): Float32Array {
        this._moduleOptions.renderMode = ps.renderer.renderMode;
        const previous = this._compiledModuleData;
        if (!this._moduleTracker.check(ps, this._moduleOptions) && previous) return previous;

        try {
            const next = compileParticleComputeModules(ps, this._moduleOptions).data;
            if (previous && sameModuleData(previous, next)) return previous;

            this._compiledModuleData = next;
            ++this._moduleVersion;
            return next;
        } catch (error) {
            this._moduleTracker.reset();
            throw error;
        }
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
        const tracker = this._trailTracker ?? (this._trailTracker = new ParticleComputeTrailConfigTracker());
        const recompiled = tracker.check(module);
        if (recompiled) {
            try {
                this._compiledTrail = compileParticleComputeTrail(module);
            } catch (error) {
                tracker.reset();
                throw error;
            }
        }
        const compiled = this._compiledTrail!;
        const maxPoints = ps.renderer.computeTrailMaxPoints;
        if (!this._trail || this._trail.capacity !== simulation.capacity || this._trail.maxPoints !== maxPoints) {
            this._destroyComputeTrail();
            const material = this.updateTrailMaterial();
            if (!material) return;

            this._trail = new ParticleComputeTrail(deviceManager.gfxDevice, simulation.capacity, maxPoints, compiled);
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
        } else if (recompiled) {
            if (!this.updateTrailMaterial()) return;

            this._trail.configure(compiled);
        }
        const toWorld = ps.simulationSpace === ParticleSpace.Local && compiled.space === ParticleSpace.World;
        this._trail.update(
            simulation.stateBuffer,
            simulation.counterBuffer,
            dt,
            toWorld ? ps.node.worldMatrix : undefined,
            systemTime,
            ps.simulationSpace,
            simulation.activeBound ?? simulation.capacity,
        );
        this._computeTrailModel!.enabled = this._mayHaveParticles;
        if (this._model?.scene && !this._computeTrailModel!.scene) {
            ps._getRenderScene().addModel(this._computeTrailModel!);
        }
    }

    private _destroyComputeTrail (): void {
        if (this._computeTrailModel) {
            if (this._computeTrailModel.scene) {
                this._computeTrailModel.scene.removeModel(this._computeTrailModel);
            }
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

    /** Returns the cached GPU count. Continuous emission only refreshes it on an explicit readback. */
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
        const tick = simulation.tickId;
        const count = await simulation.readCount();
        if (!this._disposed && this._simulation === simulation && epoch === this._epoch && tick >= this._countTick) {
            if (revision === this._birthRevision && committedRevision === this._committedBirthRevision) {
                this._lastCount = count;
                this._countValid = true;
                this._countTick = tick;
                if (count === 0 && this._births.length === 0) {
                    this._mayHaveParticles = false;
                }
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
        this._simulation?.clear();
        this._resetRendererState();
    }

    private _resetRendererState (): void {
        this._epoch++;
        this._pollPending = false;
        this._births.length = 0;
        this._trail?.clear();
        if (this._model) {
            this._model.clear();
            this._model.enabled = false;
        }
        if (this._computeTrailModel) {
            this._computeTrailModel.enabled = false;
        }
        this._lastCount = 0;
        this._countValid = true;
        this._countTick = 0;
        this._mayHaveParticles = false;
    }

    public setUseInstance (value: boolean): void {
        if (!value) {
            throw new Error('Compute particle materials require instanced vertex attributes');
        }
    }

    public onDestroy (): void {
        if (this._disposed) return;
        this._disposed = true;
        this._inited = false;
        this._epoch++;
        this._births.length = 0;
        this._destroyComputeTrail();
        this._simulation?.destroy();
        this._simulation = null;
        super.onDestroy();
    }
}
