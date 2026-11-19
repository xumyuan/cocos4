/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */
import {
    API, Buffer, BufferInfo, BufferUsageBit, BufferViewInfo, DescriptorSet,
    DescriptorSetInfo, Device, DispatchInfo, MemoryUsageBit, PipelineState,
} from '../../gfx';
import { ParticleComputeBirthBatch } from './particle-compute-birth';
import {
    createParticleComputeDispatch, flushParticleComputeCommands, getParticleComputeBindingLayout,
    getParticleComputeProgram, runParticleComputePass, validateParticleComputeBuffer, validateParticleComputeCapacity,
} from './particle-compute-effect';
import type { Disposable, ParticleComputeBindingLayout } from './particle-compute-effect';
import { ComputeModuleOffset, ComputeShaderFeature } from './particle-compute-modules';

const GROUP = 64;
const MAX_BATCHES = 1024;
const STRIDE = 176;
const SCAN_PARAM_WORD_OFFSET = 20;
const FRAME_PARAM_WORDS = SCAN_PARAM_WORD_OFFSET + 4;
const FRAME_PARAM_SIZE = FRAME_PARAM_WORDS * Uint32Array.BYTES_PER_ELEMENT;
interface ScanLevel { offset: number; count: number }
interface ScanPlanStep {
    readonly pipeline: PipelineState;
    readonly dispatch: DispatchInfo;
    readonly level: number;
}
interface ScanPlan {
    readonly dispatch: DispatchInfo;
    readonly levels: readonly ScanLevel[];
    readonly steps: readonly ScanPlanStep[];
}
const SIMULATION_RESOURCES = ['FrameParams', 'InputState', 'OutputState', 'Control', 'BirthWords', 'ModuleWords'] as const;
type SimulationShared = ParticleComputeBindingLayout<typeof SIMULATION_RESOURCES[number]>;
const FIXED_KERNELS = ['initialize', 'prepare', 'scan', 'scatter', 'finalize'];
const SINGLE_DISPATCH = new DispatchInfo(1, 1, 1);
const sharedSimulationResources = new WeakMap<Device, SimulationShared>();
let nextSimulationInstanceId = 1;

function getSharedSimulationResources (device: Device): SimulationShared {
    const existing = sharedSimulationResources.get(device);
    if (existing) return existing;
    const program = getParticleComputeProgram(device, 'simulation', 'initialize');
    const shared = getParticleComputeBindingLayout(program, SIMULATION_RESOURCES);
    sharedSimulationResources.set(device, shared);
    return shared;
}

function createScanLevels (activeGroups: number): ScanLevel[] {
    const levels: ScanLevel[] = [];
    let relOffset = 0;
    for (let count = activeGroups; ; count = Math.ceil(count / GROUP)) {
        levels.push({ offset: relOffset, count });
        relOffset += count;
        if (count <= GROUP) break;
    }
    return levels;
}

/**
 * @engineInternal
 * GPU implementation of existing particle state; scene rendering consumes the final A/B buffer.
 */
export class ParticleComputeSimulation {
    private readonly _owned: Disposable[] = [];
    private readonly _states: Buffer[] = [];
    private readonly _sets: DescriptorSet[] = [];
    private readonly _corePipelines: PipelineState[] = [];
    private readonly _scanPlans = new Map<number, ScanPlan>();
    private readonly _scanSets: DescriptorSet[][] = [[], []];
    private readonly _shared: SimulationShared;
    private readonly _fullDispatch: DispatchInfo;
    private _parameters!: Buffer;
    private _control!: Buffer;
    private _births!: Buffer;
    private _modules!: Buffer;
    private _lastModuleRef: Float32Array | null = null;
    private _index = 0;
    private _tick = 0;
    private _epoch = 0;
    private _activeBound = 0;
    private _totalRequested = 0;
    private _destroyed = false;
    private readonly _paramStride: number;
    private readonly _paramWords: Uint32Array;
    private readonly _paramFloats: Float32Array;
    private readonly _birthWords = new Uint32Array(MAX_BATCHES * (4 + 64));
    private readonly _birthFloats = new Float32Array(this._birthWords.buffer);
    private readonly _groups: number;
    private readonly _seed: number;

    public constructor (private readonly _device: Device, public readonly capacity: number, initialTime = 0, seed?: number) {
        if (_device.gfxAPI !== API.WEBGPU) throw new Error('Compute particles require WebGPU');
        validateParticleComputeCapacity(capacity);
        const randomSalt = Math.floor(Math.random() * 0x100000000) >>> 0;
        const instanceMix = Math.imul(nextSimulationInstanceId++, 0x9e3779b1) >>> 0;
        // Keep exponent < 0xff so vec4 clockInfo.w passes through floatBitsToUint without NaN canonicalization.
        this._seed = ((seed === undefined ? (randomSalt ^ instanceMix) : seed) & 0xbfffffff) >>> 0;
        this._groups = Math.ceil(capacity / GROUP);
        this._fullDispatch = createParticleComputeDispatch(_device, this._groups);
        const own = <T extends Disposable>(r: T): T => { this._owned.push(r); return r; };
        const buffer = (bytes: number, usage = BufferUsageBit.STORAGE, stride = 4): Buffer => {
            if (usage & BufferUsageBit.STORAGE) validateParticleComputeBuffer(_device, bytes, 'Particle simulation');
            const bufferUsage = usage | BufferUsageBit.TRANSFER_DST | BufferUsageBit.TRANSFER_SRC;
            return own(_device.createBuffer(new BufferInfo(bufferUsage, MemoryUsageBit.DEVICE, bytes, stride)));
        };
        const flags = 16; const prefix = flags + capacity; const sums = prefix + capacity;
        const scanLevels = createScanLevels(this._groups);
        const rootLevel = scanLevels[scanLevels.length - 1];
        const free = sums + rootLevel.offset + rootLevel.count; const generations = free + capacity;
        const indices = generations + capacity; const arcs = indices + capacity;
        const rows = arcs + MAX_BATCHES;
        const alignment = _device.capabilities.uboOffsetAlignment;
        this._paramStride = Math.ceil(FRAME_PARAM_SIZE / alignment) * alignment;
        this._paramWords = new Uint32Array((scanLevels.length * this._paramStride + FRAME_PARAM_SIZE) / Uint32Array.BYTES_PER_ELEMENT);
        this._paramFloats = new Float32Array(this._paramWords.buffer);
        this._paramWords.set([capacity, 0, 0, this._groups]);
        this._paramWords.set([flags, prefix, sums, free, generations, indices, arcs, rows], 12);
        try {
            this._shared = getSharedSimulationResources(_device);
            this._states.push(buffer(capacity * STRIDE, BufferUsageBit.STORAGE, STRIDE));
            this._states.push(buffer(capacity * STRIDE, BufferUsageBit.STORAGE, STRIDE));
            this._control = buffer((rows + capacity) * 4);
            this._parameters = buffer(this._paramWords.byteLength, BufferUsageBit.UNIFORM);
            this._births = buffer(this._birthWords.byteLength);
            this._modules = buffer(4096);
            for (let role = 0; role < 2; role++) {
                this._sets.push(this._createFrameSet(role, 0));
            }
            getParticleComputeProgram(_device, 'simulation', 'spawn');
            getParticleComputeProgram(_device, 'simulation', 'update');
            for (const kernel of FIXED_KERNELS) {
                this._corePipelines.push(getParticleComputeProgram(_device, 'simulation', kernel).pipeline);
            }
            // Warm the initial scan tier so shader lookup failures surface at construction time.
            this._getScanPlan(1);
            this.clear(initialTime, false);
        } catch (error) { this.destroy(); throw error; }
    }

    private _getScanPlan (activeGroups: number): ScanPlan {
        const cached = this._scanPlans.get(activeGroups);
        if (cached) return cached;
        const levels = createScanLevels(activeGroups);
        const makeStep = (name: 'scan-sums' | 'scan-add', level: number): ScanPlanStep => ({
            pipeline: getParticleComputeProgram(this._device, 'simulation', name).pipeline,
            dispatch: createParticleComputeDispatch(this._device, Math.ceil(levels[level].count / GROUP)),
            level,
        });
        const steps = levels.map((_, index) => makeStep('scan-sums', index));
        for (let index = levels.length - 2; index >= 0; --index) {
            steps.push(makeStep('scan-add', index));
        }
        const plan: ScanPlan = {
            dispatch: activeGroups === this._groups ? this._fullDispatch : createParticleComputeDispatch(this._device, activeGroups),
            levels,
            steps,
        };
        this._scanPlans.set(activeGroups, plan);
        return plan;
    }

    public get stateBuffer (): Buffer { this._assertAlive(); return this._states[this._index]; }
    public get counterBuffer (): Buffer { this._assertAlive(); return this._control; }
    public get tickId (): number { return this._tick; }
    /** Conservative upper bound on GPU live particle count, tightened on async readback. */
    public get activeBound (): number { return this._activeBound; }

    public tick (
        dt: number,
        batches: readonly ParticleComputeBirthBatch[],
        moduleData: Float32Array,
        systemTime: number,
        gravity: readonly [number, number, number] = [0, -9.8, 0],
    ): void {
        this._assertAlive();
        if (!Number.isFinite(Math.fround(dt)) || dt < 0 || !Number.isFinite(Math.fround(systemTime))) {
            throw new RangeError('Invalid particle time');
        }
        if (gravity.length !== 3
            || !Number.isFinite(Math.fround(gravity[0]))
            || !Number.isFinite(Math.fround(gravity[1]))
            || !Number.isFinite(Math.fround(gravity[2]))) {
            throw new RangeError('Invalid particle gravity');
        }
        if (batches.length > MAX_BATCHES) {
            throw new RangeError('Too many emit batches in one particle update');
        }
        if (moduleData.length < 128) {
            throw new RangeError('Invalid particle module data');
        }

        let requested = 0;
        let offset = batches.length * 4;
        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            if (!Number.isSafeInteger(batch.count) || batch.count < 0 || batch.data.length !== 64) {
                throw new RangeError('Invalid particle birth batch');
            }
            const count = Math.min(batch.count, this.capacity - requested);
            this._birthWords[i * 4] = requested;
            requested += count;
            this._birthWords[i * 4 + 1] = requested;
            this._birthWords[i * 4 + 2] = offset;
            this._birthFloats.set(batch.data, offset);
            offset += 64;
        }

        const simBound = Math.min(this.capacity, this._activeBound + requested);
        this._activeBound = simBound;
        this._totalRequested += requested;
        if (simBound === 0) {
            this._tick++;
            return;
        }

        const activeGroups = Math.min(this._groups, 2 ** Math.ceil(Math.log2(Math.ceil(simBound / GROUP))));
        const scanPlan = this._getScanPlan(activeGroups);
        const activeDispatch = scanPlan.dispatch;
        const features = moduleData[ComputeModuleOffset.SHADER_FEATURES] | 0;
        const spawnDispatch = requested > 0 ? createParticleComputeDispatch(this._device, Math.ceil(requested / GROUP)) : null;
        const spawnPipeline = spawnDispatch ? getParticleComputeProgram(this._device, 'simulation', 'spawn', {
            CC_PARTICLE_HAS_SPLINE: Boolean(features & ComputeShaderFeature.BIRTH_SPLINE),
            CC_PARTICLE_HAS_GRADIENT: Boolean(features & ComputeShaderFeature.BIRTH_GRADIENT),
        }).pipeline : null;
        const updatePipeline = getParticleComputeProgram(this._device, 'simulation', 'update', {
            CC_PARTICLE_HAS_SPLINE: Boolean(features & ComputeShaderFeature.UPDATE_SPLINE),
            CC_PARTICLE_HAS_GRADIENT: Boolean(features & ComputeShaderFeature.UPDATE_GRADIENT),
            CC_PARTICLE_HAS_NOISE: Boolean(features & ComputeShaderFeature.NOISE),
        }).pipeline;

        this._paramWords[1] = requested;
        this._paramWords[2] = batches.length;
        this._paramWords[3] = activeGroups;
        this._paramFloats[4] = gravity[0];
        this._paramFloats[5] = gravity[1];
        this._paramFloats[6] = gravity[2];
        this._paramFloats[7] = dt;
        this._paramFloats[8] = systemTime;
        this._paramFloats[9] = this._tick;
        this._paramFloats[10] = this._epoch + 1;
        this._paramWords[11] = this._seed;

        const role = this._index;
        const set = this._sets[role];
        const scanSets = this._scanSets[role];
        const { levels, steps } = scanPlan;
        for (let level = scanSets.length; level < levels.length; ++level) {
            scanSets.push(this._createFrameSet(role, (level + 1) * this._paramStride));
        }
        runParticleComputePass(this._device, this, () => {
            const frame = this._paramWords.subarray(0, SCAN_PARAM_WORD_OFFSET);
            const sums = frame[14];
            for (let index = 0; index < levels.length; index++) {
                const level = levels[index];
                const parent = levels[index + 1];
                const slice = (index + 1) * this._paramStride / Uint32Array.BYTES_PER_ELEMENT;
                this._paramWords.set(frame, slice);
                const params = slice + SCAN_PARAM_WORD_OFFSET;
                this._paramWords[params] = level.count;
                this._paramWords[params + 1] = sums + level.offset;
                // Finalize reads the total from control[3].
                this._paramWords[params + 2] = parent ? sums + parent.offset : 3;
                this._paramWords[params + 3] = Math.ceil(level.count / GROUP);
            }
            this._parameters.update(this._paramWords.buffer, levels.length * this._paramStride + FRAME_PARAM_SIZE);
            if (batches.length) {
                this._births.update(this._birthWords.buffer, offset * 4);
            }
            this._uploadModules(moduleData);
        }, (cmd) => {
            // Fixed indices: 0: initialize, 1: prepare, 2: scan, 3: scatter, 4: finalize
            cmd.bindDescriptorSet(this._shared.setIndex, set);
            cmd.bindPipelineState(this._corePipelines[1]);
            cmd.dispatch(SINGLE_DISPATCH);
            if (spawnDispatch) {
                cmd.bindPipelineState(spawnPipeline!);
                cmd.dispatch(spawnDispatch);
            }
            cmd.bindPipelineState(updatePipeline);
            cmd.dispatch(activeDispatch);
            cmd.bindPipelineState(this._corePipelines[2]);
            cmd.dispatch(activeDispatch);
            for (let s = 0; s < steps.length; ++s) {
                cmd.bindDescriptorSet(this._shared.setIndex, scanSets[steps[s].level]);
                cmd.bindPipelineState(steps[s].pipeline);
                cmd.dispatch(steps[s].dispatch);
            }
            cmd.bindDescriptorSet(this._shared.setIndex, set);
            cmd.bindPipelineState(this._corePipelines[3]);
            cmd.dispatch(activeDispatch);
            cmd.bindPipelineState(this._corePipelines[4]);
            cmd.dispatch(SINGLE_DISPATCH);
        });
        this._index = 1 - this._index;
        this._tick++;
    }

    public async readCount (): Promise<number> {
        this._assertAlive();
        flushParticleComputeCommands(this._device);
        const epoch = this._epoch;
        const requestedAtStart = this._totalRequested;
        const bytes = await this._device.readBuffer(this._control, 0, 4);
        const count = new Uint32Array(bytes)[0];
        if (!this._destroyed && this._epoch === epoch) {
            const addedAfter = this._totalRequested - requestedAtStart;
            this._activeBound = Math.min(this._activeBound, Math.min(this.capacity, count + addedAfter));
        }
        return count;
    }

    public clear (initialTime = 0, preserveArc = true): void {
        this._assertAlive();
        if (!Number.isFinite(initialTime)) throw new RangeError('Invalid particle initial time');
        this._epoch++; this._index = 0; this._tick = 0;
        this._activeBound = 0;
        this._totalRequested = 0;
        this._paramWords[3] = this._groups;
        this._paramFloats[9] = preserveArc ? 1 : 0;
        this._paramFloats[11] = initialTime;
        const set = this._sets[0];
        const pipeline = this._corePipelines[0];
        const dispatch = this._fullDispatch;
        runParticleComputePass(this._device, this, () => {
            this._parameters.update(this._paramWords.buffer, FRAME_PARAM_SIZE);
        }, (cmd) => {
            cmd.bindDescriptorSet(this._shared.setIndex, set);
            cmd.bindPipelineState(pipeline);
            cmd.dispatch(dispatch);
        });
    }

    public destroy (): void {
        if (this._destroyed) return;
        this._destroyed = true;
        flushParticleComputeCommands(this._device);
        this._scanPlans.clear();
        for (let i = this._owned.length - 1; i >= 0; --i) this._owned[i].destroy();
        this._owned.length = 0;
        this._lastModuleRef = null;
    }

    private _createFrameSet (role: number, byteOffset: number): DescriptorSet {
        const { bindings, setLayout } = this._shared;
        const view = this._device.createBuffer(new BufferViewInfo(this._parameters, byteOffset, FRAME_PARAM_SIZE));
        let set: DescriptorSet | undefined;
        try {
            set = this._device.createDescriptorSet(new DescriptorSetInfo(setLayout));
            set.bindBuffer(bindings.FrameParams, view);
            set.bindBuffer(bindings.InputState, this._states[role]);
            set.bindBuffer(bindings.OutputState, this._states[1 - role]);
            set.bindBuffer(bindings.Control, this._control);
            set.bindBuffer(bindings.BirthWords, this._births);
            set.bindBuffer(bindings.ModuleWords, this._modules);
            set.update();
            this._owned.push(view, set);
            return set;
        } catch (error) {
            set?.destroy();
            view.destroy();
            throw error;
        }
    }

    private _uploadModules (data: Float32Array): void {
        // The renderer recompiles into a new array only when its config key changes.
        if (this._lastModuleRef === data) return;
        if (data.byteLength > this._modules.size) {
            const nextSize = 2 ** Math.ceil(Math.log2(data.byteLength));
            validateParticleComputeBuffer(this._device, nextSize, 'Particle modules');
            flushParticleComputeCommands(this._device);
            const next = this._device.createBuffer(new BufferInfo(
                BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_DST,
                MemoryUsageBit.DEVICE,
                nextSize,
                4,
            ));
            const old = this._modules;
            this._owned[this._owned.indexOf(old)] = next;
            this._modules = next;
            for (const set of this._sets.concat(...this._scanSets)) {
                set.bindBuffer(this._shared.bindings.ModuleWords, next);
                set.update();
            }
            old.destroy();
        }
        this._modules.update(data);
        this._lastModuleRef = data;
    }

    private _assertAlive (): void { if (this._destroyed) throw new Error('Particle simulation has been destroyed'); }
}
