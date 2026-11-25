/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import {
    API, Buffer, BufferInfo, BufferUsageBit, BufferViewInfo, DescriptorSet,
    DescriptorSetInfo, Device, DispatchInfo, MemoryUsageBit, PipelineState,
} from '../../gfx';
import { ComputeBirthBatchOffset, PARTICLE_COMPUTE_BIRTH_FLOATS, ParticleComputeBirthBatch } from './particle-compute-birth';
import {
    createParticleComputeDispatch, flushParticleComputeCommands, getParticleComputeBindingLayout,
    getParticleComputeProgram, PARTICLE_COMPUTE_STATE_STRIDE, PARTICLE_COMPUTE_WORKGROUP_SIZE,
    runParticleComputePass, setParticleComputeDispatch, validateParticleComputeBuffer, validateParticleComputeCapacity,
} from './particle-compute-effect';
import type { Disposable, ParticleComputeBindingLayout } from './particle-compute-effect';
import { ComputeModuleOffset, ComputeShaderFeature } from './particle-compute-modules';

export const ComputeSimControlOffset = {
    LIVE_COUNT: 0,
    ACCEPTED: 1,
    ACTIVE_COUNT: 2,
    SURVIVOR_COUNT: 3,
    FREE_COUNT: 4,
    ALIVE_OFFSET: 5,
    NEXT_ALIVE_OFFSET: 6,
    EMITTED_TOTAL: 7,
    FREE_TOP_BEFORE: 8,
    FREE_TOP_AFTER: 9,
    ARC_TOTAL: 10,
    ARC_LAST_TIME: 11,
    HEADER_WORDS: 16,
} as const;

const MAX_BATCHES = 1024;
const INITIAL_BIRTH_BATCHES = 8;
const BIRTH_BATCH_WORDS = ComputeBirthBatchOffset.HEADER_WORDS + PARTICLE_COMPUTE_BIRTH_FLOATS; // 4 + 64 = 68
const DEFAULT_GRAVITY: readonly [number, number, number] = [0, -9.8, 0];

const enum FrameParamWord {
    CAPACITY, REQUESTED, BATCH_COUNT, ACTIVE_GROUPS,
    GRAVITY_X, GRAVITY_Y, GRAVITY_Z, DT,
    SYSTEM_TIME, INITIAL_TIME, PRESERVE_ARC,
    TICK = 12, EPOCH, SEED,
    PREFIX = 16, SUMS,
    SCAN_COUNT = 24, SCAN_OFFSET, SCAN_PARENT, SCAN_GROUPS,
    LENGTH,
}

const FRAME_PARAM_SIZE = FrameParamWord.LENGTH * Uint32Array.BYTES_PER_ELEMENT; // 28 * 4 = 112
const SIMULATION_RESOURCES = ['FrameParams', 'StateBuffer', 'Control', 'BirthWords', 'ModuleWords'] as const;
type SimulationShared = ParticleComputeBindingLayout<typeof SIMULATION_RESOURCES[number]>;
const FIXED_KERNELS = ['initialize', 'prepare', 'scan-sums', 'scan-add', 'scatter', 'finalize'] as const;
const SINGLE_DISPATCH = new DispatchInfo(1, 1, 1);
const sharedSimulationResources = new WeakMap<Device, SimulationShared>();
let nextSimulationInstanceID = 1;

function getSharedSimulationResources (device: Device): SimulationShared {
    const existing = sharedSimulationResources.get(device);
    if (existing) return existing;
    const program = getParticleComputeProgram(device, 'simulation', 'initialize');
    const shared = getParticleComputeBindingLayout(program, SIMULATION_RESOURCES);
    sharedSimulationResources.set(device, shared);
    return shared;
}

interface ScanLevel {
    readonly set: DescriptorSet;
    readonly dispatch: DispatchInfo;
}

/**
 * @engineInternal
 * GPU implementation of existing particle state; consumers resolve stable handles through the current alive list.
 */
export class ParticleComputeSimulation {
    private readonly _owned: Disposable[] = [];
    private _state!: Buffer;
    private _frameSet!: DescriptorSet;
    private readonly _corePipelines = {} as Record<typeof FIXED_KERNELS[number], PipelineState>;
    private readonly _spawnPipelines = new Map<number, PipelineState>();
    private readonly _updatePipelines = new Map<number, PipelineState>();
    private readonly _scanLevels: ScanLevel[] = [];
    private readonly _spawnDispatch = new DispatchInfo(1, 1, 1);
    private readonly _activeDispatch = new DispatchInfo(1, 1, 1);
    private readonly _shared: SimulationShared;
    private readonly _fullDispatch: DispatchInfo;
    private _parameters!: Buffer;
    private _control!: Buffer;
    private _births!: Buffer;
    private _modules!: Buffer;
    private _lastModuleRef: Float32Array | null = null;
    private _lastModuleVersion?: number;
    private _tick = 0;
    private _epoch = 0;
    private _activeBound = 0;
    private _totalRequested = 0;
    private _destroyed = false;
    private readonly _paramStride: number;
    private readonly _paramWords: Uint32Array;
    private readonly _paramFloats: Float32Array;
    private _birthWords = new Uint32Array(INITIAL_BIRTH_BATCHES * BIRTH_BATCH_WORDS);
    private _birthFloats = new Float32Array(this._birthWords.buffer);
    private readonly _groups: number;
    private readonly _seed: number;

    public constructor (
        private readonly _device: Device,
        public readonly capacity: number,
        initialTime = 0,
        seed: number | undefined = undefined,
        enableStateReadback = false,
    ) {
        // Validate backend/capacity and derive per-instance seed and full dispatch size.
        if (_device.gfxAPI !== API.WEBGPU) {
            throw new Error('Compute particles require WebGPU');
        }
        validateParticleComputeCapacity(capacity);
        const randomSalt = Math.floor(Math.random() * 0x100000000) >>> 0;
        const instanceMix = Math.imul(nextSimulationInstanceID++, 0x9e3779b1) >>> 0;
        this._seed = (seed === undefined ? randomSalt ^ instanceMix : seed) >>> 0;
        this._groups = Math.ceil(capacity / PARTICLE_COMPUTE_WORKGROUP_SIZE);
        this._fullDispatch = createParticleComputeDispatch(_device, this._groups);

        const own = <T extends Disposable>(r: T): T => {
            this._owned.push(r);
            return r;
        };
        const buffer = (bytes: number, usage = BufferUsageBit.STORAGE, stride = 4): Buffer => {
            if (usage & BufferUsageBit.STORAGE) {
                validateParticleComputeBuffer(_device, bytes, 'Particle simulation');
            }
            const bufferUsage = usage | BufferUsageBit.TRANSFER_DST;
            return own(_device.createBuffer(new BufferInfo(bufferUsage, MemoryUsageBit.DEVICE, bytes, stride)));
        };

        // Lay out contiguous uint32 regions inside the Control buffer and size scan levels.
        const prefix = ComputeSimControlOffset.HEADER_WORDS;
        const sums = prefix + capacity;
        let free = sums;
        let scanLevelCount = 0;
        for (let count = this._groups; ; count = Math.ceil(count / PARTICLE_COMPUTE_WORKGROUP_SIZE)) {
            ++scanLevelCount;
            free += count;
            if (count <= PARTICLE_COMPUTE_WORKGROUP_SIZE) {
                break;
            }
        }
        const generations = free + capacity;
        const aliveA = generations + capacity;
        const aliveB = aliveA + capacity;
        const arcs = aliveB + capacity;
        const rows = arcs + MAX_BATCHES;

        // Allocate aligned UBO slices (frame + scan levels) and prefill static offsets.
        const alignment = _device.capabilities.uboOffsetAlignment;
        this._paramStride = Math.ceil(FRAME_PARAM_SIZE / alignment) * alignment;
        this._paramWords = new Uint32Array((scanLevelCount * this._paramStride + FRAME_PARAM_SIZE) / Uint32Array.BYTES_PER_ELEMENT);
        this._paramFloats = new Float32Array(this._paramWords.buffer);
        this._paramWords[FrameParamWord.CAPACITY] = capacity;
        this._paramWords.set([prefix, sums, free, aliveA, generations, aliveB, arcs, rows], FrameParamWord.PREFIX);

        // Create GPU buffers, descriptor sets, core pipelines, and initialize control state.
        try {
            this._shared = getSharedSimulationResources(_device);
            const stateUsage = enableStateReadback ? BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_SRC : BufferUsageBit.STORAGE;
            this._state = buffer(capacity * PARTICLE_COMPUTE_STATE_STRIDE, stateUsage, PARTICLE_COMPUTE_STATE_STRIDE);
            this._control = buffer((rows + capacity) * 4, BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_SRC);
            this._parameters = buffer(this._paramWords.byteLength, BufferUsageBit.UNIFORM);
            this._births = buffer(this._birthWords.byteLength);
            this._modules = buffer(4096);
            this._frameSet = this._createFrameSet(0);
            for (let level = 0; level < scanLevelCount; ++level) {
                this._scanLevels.push({
                    set: this._createFrameSet((level + 1) * this._paramStride),
                    dispatch: new DispatchInfo(1, 1, 1),
                });
            }
            for (const kernel of FIXED_KERNELS) {
                this._corePipelines[kernel] = getParticleComputeProgram(_device, 'simulation', kernel).pipeline;
            }
            this.clear(initialTime, false);
        } catch (error) {
            this.destroy();
            throw error;
        }
    }

    /** State is indexed by handle. The current draw order is stored in the counter buffer at ALIVE_OFFSET. */
    public get stateBuffer (): Buffer {
        this._assertAlive();
        return this._state;
    }

    public get counterBuffer (): Buffer {
        this._assertAlive();
        return this._control;
    }

    /** The public name matches the tickId field returned by ParticleSystem.getParticleCountInfo. */
    public get tickId (): number {
        return this._tick;
    }

    /** Conservative upper bound on GPU live particle count, tightened on async readback. */
    public get activeBound (): number {
        return this._activeBound;
    }

    public tick (
        dt: number,
        batches: readonly ParticleComputeBirthBatch[],
        moduleData: Float32Array,
        systemTime: number,
        gravity: readonly [number, number, number] = DEFAULT_GRAVITY,
        moduleVersion: number | undefined = undefined,
    ): void {
        // validate inputs before touching any GPU resources, to avoid leaving the simulation in a bad state
        this._assertAlive();
        if (batches.length > MAX_BATCHES) {
            throw new RangeError('Too many emit batches in one particle update');
        }

        // pack the batch data into a single buffer, and record the offsets in the header
        this._ensureBirthCapacity(batches.length);
        let requested = 0;
        let offset = batches.length * ComputeBirthBatchOffset.HEADER_WORDS;
        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            const count = Math.min(batch.count, this.capacity - requested);
            const header = i * ComputeBirthBatchOffset.HEADER_WORDS;
            this._birthWords[header + ComputeBirthBatchOffset.BEGIN] = requested;
            requested += count;
            this._birthWords[header + ComputeBirthBatchOffset.END] = requested;
            this._birthWords[header + ComputeBirthBatchOffset.SNAPSHOT] = offset;
            this._birthFloats.set(batch.data, offset);
            offset += PARTICLE_COMPUTE_BIRTH_FLOATS;
        }

        // estimate the scale of emitted particles
        const simBound = Math.min(this.capacity, this._activeBound + requested);
        if (simBound === 0) {
            this._tick++;
            return;
        }

        const activeGroups = Math.ceil(simBound / PARTICLE_COMPUTE_WORKGROUP_SIZE);
        const features = moduleData[ComputeModuleOffset.SHADER_FEATURES] | 0;
        const spawnPipeline = requested ? this._getPipeline('spawn', features) : null;
        const updatePipeline = this._getPipeline('update', features);
        let levelCount = 0;
        runParticleComputePass(this._device, this, () => {
            // The previous tick has now been recorded. Its dispatch objects are safe to reuse.
            setParticleComputeDispatch(this._device, this._activeDispatch, activeGroups);
            if (spawnPipeline) {
                setParticleComputeDispatch(this._device, this._spawnDispatch, Math.ceil(requested / PARTICLE_COMPUTE_WORKGROUP_SIZE));
            }
            this._paramWords[FrameParamWord.REQUESTED] = requested;
            this._paramWords[FrameParamWord.BATCH_COUNT] = batches.length;
            this._paramWords[FrameParamWord.ACTIVE_GROUPS] = activeGroups;
            this._paramFloats.set(gravity, FrameParamWord.GRAVITY_X);
            this._paramFloats[FrameParamWord.DT] = dt;
            this._paramFloats[FrameParamWord.SYSTEM_TIME] = systemTime;
            this._paramWords[FrameParamWord.TICK] = this._tick;
            this._paramWords[FrameParamWord.EPOCH] = this._epoch + 1;
            this._paramWords[FrameParamWord.SEED] = this._seed;

            let sums = this._paramWords[FrameParamWord.SUMS];
            for (let count = activeGroups; ;) {
                const groups = Math.ceil(count / PARTICLE_COMPUTE_WORKGROUP_SIZE);
                setParticleComputeDispatch(this._device, this._scanLevels[levelCount].dispatch, groups);
                ++levelCount;
                const slice = levelCount * this._paramStride / Uint32Array.BYTES_PER_ELEMENT;
                this._paramWords[slice + FrameParamWord.SCAN_COUNT] = count;
                this._paramWords[slice + FrameParamWord.SCAN_OFFSET] = sums;
                sums += count;
                this._paramWords[slice + FrameParamWord.SCAN_PARENT] = groups === 1 ? ComputeSimControlOffset.SURVIVOR_COUNT : sums;
                this._paramWords[slice + FrameParamWord.SCAN_GROUPS] = groups;
                if (groups === 1) {
                    break;
                }
                count = groups;
            }
            this._parameters.update(this._paramWords.buffer, levelCount * this._paramStride + FRAME_PARAM_SIZE);
            if (batches.length) {
                this._births.update(this._birthWords.buffer, offset * 4);
            }
            this._uploadModules(moduleData, moduleVersion);
        }, (cmd) => {
            const setIndex = this._shared.setIndex;
            cmd.bindDescriptorSet(setIndex, this._frameSet);
            cmd.bindPipelineState(this._corePipelines.prepare);
            cmd.dispatch(SINGLE_DISPATCH);
            if (spawnPipeline) {
                cmd.bindPipelineState(spawnPipeline);
                cmd.dispatch(this._spawnDispatch);
            }
            cmd.bindPipelineState(updatePipeline);
            cmd.dispatch(this._activeDispatch);

            cmd.bindPipelineState(this._corePipelines['scan-sums']);
            for (let level = 0; level < levelCount; ++level) {
                const scan = this._scanLevels[level];
                cmd.bindDescriptorSet(setIndex, scan.set);
                cmd.dispatch(scan.dispatch);
            }
            if (levelCount > 1) {
                cmd.bindPipelineState(this._corePipelines['scan-add']);
                for (let level = levelCount - 2; level >= 0; --level) {
                    const scan = this._scanLevels[level];
                    cmd.bindDescriptorSet(setIndex, scan.set);
                    cmd.dispatch(scan.dispatch);
                }
            }
            cmd.bindDescriptorSet(setIndex, this._frameSet);
            cmd.bindPipelineState(this._corePipelines.scatter);
            cmd.dispatch(this._activeDispatch);
            cmd.bindPipelineState(this._corePipelines.finalize);
            cmd.dispatch(SINGLE_DISPATCH);
        });
        this._activeBound = simBound;
        this._totalRequested += requested;
        this._tick++;
    }

    public async readCount (): Promise<number> {
        this._assertAlive();
        flushParticleComputeCommands(this._device);
        const epoch = this._epoch;
        const requestedAtStart = this._totalRequested;
        const wordBytes = Uint32Array.BYTES_PER_ELEMENT;
        const bytes = await this._device.readBuffer(this._control, ComputeSimControlOffset.LIVE_COUNT * wordBytes, wordBytes);
        const count = new Uint32Array(bytes)[0];
        if (!this._destroyed && this._epoch === epoch) {
            const addedAfter = this._totalRequested - requestedAtStart;
            this._activeBound = Math.min(this._activeBound, Math.min(this.capacity, count + addedAfter));
        }
        return count;
    }

    public clear (initialTime = 0, preserveArc = true): void {
        this._assertAlive();
        if (!Number.isFinite(initialTime)) {
            throw new RangeError('Invalid particle initial time');
        }
        this._epoch++;
        this._tick = 0;
        this._activeBound = 0;
        this._totalRequested = 0;
        this._paramWords[FrameParamWord.ACTIVE_GROUPS] = this._groups;
        this._paramFloats[FrameParamWord.PRESERVE_ARC] = preserveArc ? 1 : 0;
        this._paramFloats[FrameParamWord.INITIAL_TIME] = initialTime;

        const set = this._frameSet;
        const pipeline = this._corePipelines.initialize;
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
        this._scanLevels.length = 0;
        this._spawnPipelines.clear();
        this._updatePipelines.clear();
        for (let i = this._owned.length - 1; i >= 0; --i) {
            this._owned[i].destroy();
        }
        this._owned.length = 0;
        this._lastModuleRef = null;
    }

    private _createFrameSet (byteOffset: number): DescriptorSet {
        const { bindings, setLayout } = this._shared;
        const view = this._device.createBuffer(new BufferViewInfo(this._parameters, byteOffset, FRAME_PARAM_SIZE));
        this._owned.push(view);
        const set = this._device.createDescriptorSet(new DescriptorSetInfo(setLayout));
        this._owned.push(set);
        set.bindBuffer(bindings.FrameParams, view);
        set.bindBuffer(bindings.StateBuffer, this._state);
        set.bindBuffer(bindings.Control, this._control);
        set.bindBuffer(bindings.BirthWords, this._births);
        set.bindBuffer(bindings.ModuleWords, this._modules);
        set.update();
        return set;
    }

    private _getPipeline (kernel: 'spawn' | 'update', features: number): PipelineState {
        const spawn = kernel === 'spawn';
        const spline = spawn ? ComputeShaderFeature.BIRTH_SPLINE : ComputeShaderFeature.UPDATE_SPLINE;
        const gradient = spawn ? ComputeShaderFeature.BIRTH_GRADIENT : ComputeShaderFeature.UPDATE_GRADIENT;
        const key = features & (spline | gradient | (spawn ? 0 : ComputeShaderFeature.NOISE));
        const cache = spawn ? this._spawnPipelines : this._updatePipelines;
        let pipeline = cache.get(key);
        if (!pipeline) {
            const defines = {
                CC_PARTICLE_HAS_SPLINE: Boolean(key & spline),
                CC_PARTICLE_HAS_GRADIENT: Boolean(key & gradient),
            };
            const macros = spawn ? defines : { ...defines, CC_PARTICLE_HAS_NOISE: Boolean(key & ComputeShaderFeature.NOISE) };
            pipeline = getParticleComputeProgram(this._device, 'simulation', kernel, macros).pipeline;
            cache.set(key, pipeline);
        }
        return pipeline;
    }

    private _ensureBirthCapacity (batchCount: number): void {
        if (batchCount * BIRTH_BATCH_WORDS <= this._birthWords.length) return;
        const capacity = 2 ** Math.ceil(Math.log2(batchCount));
        const words = new Uint32Array(capacity * BIRTH_BATCH_WORDS);
        this._births = this._replaceStorage(this._births, words.byteLength, this._shared.bindings.BirthWords);
        this._birthWords = words;
        this._birthFloats = new Float32Array(words.buffer);
    }

    private _replaceStorage (old: Buffer, size: number, binding: number): Buffer {
        validateParticleComputeBuffer(this._device, size, 'Particle simulation');
        flushParticleComputeCommands(this._device);
        const next = this._device.createBuffer(new BufferInfo(BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_DST, MemoryUsageBit.DEVICE, size, 4));
        this._frameSet.bindBuffer(binding, next);
        this._frameSet.update();
        for (const { set } of this._scanLevels) {
            set.bindBuffer(binding, next);
            set.update();
        }
        this._owned[this._owned.indexOf(old)] = next;
        old.destroy();
        return next;
    }

    private _uploadModules (data: Float32Array, version?: number): void {
        if (this._lastModuleRef === data && this._lastModuleVersion === version) return;
        if (data.byteLength > this._modules.size) {
            const size = 2 ** Math.ceil(Math.log2(data.byteLength));
            this._modules = this._replaceStorage(this._modules, size, this._shared.bindings.ModuleWords);
        }
        this._modules.update(data);
        this._lastModuleRef = data;
        this._lastModuleVersion = version;
    }

    private _assertAlive (): void {
        if (this._destroyed) {
            throw new Error('Particle simulation has been destroyed');
        }
    }
}
