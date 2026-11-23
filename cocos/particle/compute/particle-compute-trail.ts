/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { Mat4 } from '../../core/math';
import {
    API, Buffer, BufferInfo, BufferUsageBit,
    DescriptorSet, DescriptorSetInfo, Device, DispatchInfo, MemoryUsageBit, PipelineState,
} from '../../gfx';
import type { MacroRecord } from '../../render-scene/core/pass-utils';
import GradientRange from '../animator/gradient-range';
import { ParticleSpace } from '../enum';
import {
    ComputeColorRange, ComputeScalar, DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES,
    PACKED_SCALAR_CURVE_COUNT_OFFSET, ParticleComputeCurvePacker,
} from './particle-compute-curves';
import {
    createParticleComputeDispatch, flushParticleComputeCommands, getParticleComputeBindingLayout,
    getParticleComputeProgram, PARTICLE_COMPUTE_STATE_STRIDE, PARTICLE_COMPUTE_WORKGROUP_SIZE,
    ParticleComputeStateSets, runParticleComputePass,
    validateParticleComputeBuffer, validateParticleComputeCapacity,
} from './particle-compute-effect';
import type { Disposable, ParticleComputeBindingLayout } from './particle-compute-effect';

const TRAIL_VERTEX_STRIDE = 44;

const ComputeTrailOffset = {
    WIDTH: 0,
    COLOR_OVERTIME: 1,
    COLOR_OVER_TRAIL: 2,
    WIDTH_FROM_PARTICLE: 3,
    COLOR_FROM_PARTICLE: 4,
    MIN_DISTANCE_SQUARED: 5,
    SHADER_FEATURES: 6,
} as const;

const ComputeTrailShaderFeature = {
    WIDTH_SPLINE: 1 << 0,
    COLOR_OVERTIME_GRADIENT: 1 << 1,
    COLOR_OVER_TRAIL_GRADIENT: 1 << 2,
    VALID: 1 << 6,
} as const;

const TRAIL_SHADERS: ReadonlyArray<{ name: string; singleGroup?: boolean }> = [
    { name: 'clear' },
    { name: 'reset-map' },
    { name: 'map-live' },
    { name: 'update-history' },
    { name: 'scan-counts' },
    { name: 'scan-sums', singleGroup: true },
    { name: 'emit-geometry' },
];

export interface ParticleComputeTrailSource {
    readonly lifeTime?: ComputeScalar;
    readonly minParticleDistance?: number;
    readonly widthRatio?: ComputeScalar;
    readonly widthFromParticle?: boolean;
    readonly colorFromParticle?: boolean;
    readonly colorOvertime?: ComputeColorRange;
    readonly colorOverTrail?: ComputeColorRange;
    readonly space?: ParticleSpace;
}

export interface CompiledParticleComputeTrail {
    readonly data: Float32Array;
    readonly lifeTime: ComputeScalar;
    readonly space: ParticleSpace;
    readonly maxSampledScalarError: number;
}

interface TrailOptions {
    samples?: number;
}

/**
 * @engineInternal
 * Compile module data once; this performs no per-particle CPU work.
 */
export function compileParticleComputeTrail (
    source: ParticleComputeTrailSource = {},
    options: TrailOptions = {},
): CompiledParticleComputeTrail {
    const minDistance = source.minParticleDistance ?? 0.1;
    if (!Number.isFinite(minDistance) || minDistance < 0 || !Number.isFinite(Math.fround(minDistance * minDistance))) {
        throw new RangeError('Trail minimum distance must be finite and nonnegative');
    }

    const packer = new ParticleComputeCurvePacker(16, options.samples ?? DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES);
    const { values } = packer;
    const offsets = ComputeTrailOffset;
    values[offsets.WIDTH] = packer.scalar(source.widthRatio, 1);
    values[offsets.COLOR_OVERTIME] = packer.color(source.colorOvertime);
    values[offsets.COLOR_OVER_TRAIL] = packer.color(source.colorOverTrail);
    values[offsets.WIDTH_FROM_PARTICLE] = source.widthFromParticle === false ? 0 : 1;
    values[offsets.COLOR_FROM_PARTICLE] = source.colorFromParticle ? 1 : 0;
    values[offsets.MIN_DISTANCE_SQUARED] = minDistance * minDistance;

    const hasSpline = values[values[offsets.WIDTH] + PACKED_SCALAR_CURVE_COUNT_OFFSET] < 0;
    const isGradient = (slot: number): boolean => {
        const mode = values[values[slot]];
        return mode !== GradientRange.Mode.Color && mode !== GradientRange.Mode.TwoColors;
    };
    const feature = ComputeTrailShaderFeature;
    values[offsets.SHADER_FEATURES] = feature.VALID
        | (hasSpline ? feature.WIDTH_SPLINE : 0)
        | (isGradient(offsets.COLOR_OVERTIME) ? feature.COLOR_OVERTIME_GRADIENT : 0)
        | (isGradient(offsets.COLOR_OVER_TRAIL) ? feature.COLOR_OVER_TRAIL_GRADIENT : 0);

    const space = source.space ?? ParticleSpace.World;
    if (space !== ParticleSpace.Local && space !== ParticleSpace.World) {
        throw new Error('Trail supports Local and World spaces');
    }
    return { data: packer.finish(), lifeTime: source.lifeTime ?? 1, space, maxSampledScalarError: packer.maxSampledScalarError };
}

const TRAIL_RESOURCES = [
    'TrailParams', 'SimulationState', 'SimulationCounters', 'TrailControl', 'TrailHistory', 'TrailModules', 'TrailVertices', 'TrailIndices',
] as const;

interface TrailShared extends ParticleComputeBindingLayout<typeof TRAIL_RESOURCES[number]> {
    readonly dummyState: Buffer;
    readonly dummyCounters: Buffer;
}

const sharedTrailResources = new WeakMap<Device, TrailShared>();

function getSharedTrailResources (device: Device): TrailShared {
    const existing = sharedTrailResources.get(device);
    if (existing) return existing;
    const program = getParticleComputeProgram(device, 'trail', 'clear');
    const layout = getParticleComputeBindingLayout(program, TRAIL_RESOURCES);
    const allocated: Disposable[] = [];
    const own = <T extends Disposable>(value: T): T => {
        allocated.push(value);
        return value;
    };
    try {
        const dummyState = own(device.createBuffer(new BufferInfo(
            BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_DST,
            MemoryUsageBit.DEVICE,
            PARTICLE_COMPUTE_STATE_STRIDE,
            4,
        )));
        const dummyCounters = own(device.createBuffer(new BufferInfo(
            BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_DST,
            MemoryUsageBit.DEVICE,
            16,
            4,
        )));
        const shared: TrailShared = { ...layout, dummyState, dummyCounters };
        for (const kernel of TRAIL_SHADERS) {
            getParticleComputeProgram(device, 'trail', kernel.name);
        }
        sharedTrailResources.set(device, shared);
        return shared;
    } catch (error) {
        for (let i = allocated.length - 1; i >= 0; --i) {
            allocated[i].destroy();
        }
        throw error;
    }
}

/**
 * @engineInternal
 * GPU history and geometry for the existing Particles/Stretch trail module.
 *
 * Main state must use the 176-byte ParticleState ABI: identity.y is a unique live
 * handle in [0,capacity), identity.z changes on handle reuse; counters[0] is live count.
 * Dead particles immediately release their trail. A full history ring drops its
 * oldest point and increments control word 5, an explicit quality-budget overflow.
 * This bounded history does not promise the CPU implementation's unbounded growth.
 */
export class ParticleComputeTrail {
    public static readonly MAX_POINTS = 4096;

    /** Validate enabled device limits for trail storage buffers without allocating GPU memory. */
    public static validateBudget (device: Device, capacity: number, maxPoints: number): {
        readonly groups: number;
        readonly groupDispatch: DispatchInfo;
        readonly sizes: readonly [number, number, number];
    } {
        validateParticleComputeCapacity(capacity);
        if (!Number.isInteger(maxPoints) || maxPoints < 1 || maxPoints > ParticleComputeTrail.MAX_POINTS) {
            throw new RangeError('Invalid particle trail capacity or history point budget');
        }
        const groups = Math.ceil(capacity / PARTICLE_COMPUTE_WORKGROUP_SIZE);
        const groupDispatch = createParticleComputeDispatch(device, groups);
        const sizes: readonly [number, number, number] = [
            capacity * maxPoints * 48,
            capacity * (maxPoints + 1) * 2 * TRAIL_VERTEX_STRIDE,
            capacity * maxPoints * 6 * 4,
        ];
        const names = ['Compute trail history', 'Compute trail vertices', 'Compute trail indices'];
        sizes.forEach((size, index) => validateParticleComputeBuffer(device, size, names[index]));
        validateParticleComputeBuffer(device, (8 + groups + capacity * 9) * 4, 'Compute trail control');
        return { groups, groupDispatch, sizes };
    }

    private readonly _owned: Disposable[] = [];
    private readonly _stateSets: ParticleComputeStateSets;
    private readonly _shared: TrailShared;
    private readonly _paramsData = new ArrayBuffer(96);
    private readonly _groups: number;
    private readonly _groupDispatch: DispatchInfo;
    private readonly _singleDispatch = new DispatchInfo(1, 1, 1);
    private _compiled: CompiledParticleComputeTrail;
    private _params!: Buffer;
    private _moduleData!: Buffer;
    private _control!: Buffer;
    private _history!: Buffer;
    private _vertices!: Buffer;
    private _indices!: Buffer;
    private _clearSet!: DescriptorSet;
    private _mayHaveHistory = false;
    private _destroyed = false;
    private _tick = 0;

    public constructor (
        private readonly _device: Device,
        public readonly capacity: number,
        public readonly maxPoints: number,
        compiled: CompiledParticleComputeTrail = compileParticleComputeTrail(),
    ) {
        const device = _device;
        if (device.gfxAPI !== API.WEBGPU) {
            throw new Error('ParticleComputeTrail requires WebGPU');
        }
        const { groups, groupDispatch, sizes } = ParticleComputeTrail.validateBudget(device, capacity, maxPoints);
        this._groups = groups;
        this._groupDispatch = groupDispatch;
        this._compiled = compiled;
        validateParticleComputeBuffer(device, compiled.data.byteLength, 'Compute trail modules');

        this._shared = getSharedTrailResources(device);
        const { bindings } = this._shared;
        this._stateSets = new ParticleComputeStateSets((state, counters) => {
            const set = this._createBaseSet(state, counters);
            this._owned.push(set);
            return set;
        }, bindings.SimulationState, bindings.SimulationCounters);

        const own = <T extends Disposable>(value: T): T => {
            this._owned.push(value);
            return value;
        };
        const buffer = (size: number, usage: BufferUsageBit, stride = 4): Buffer => {
            const flags = usage | BufferUsageBit.TRANSFER_DST | BufferUsageBit.TRANSFER_SRC;
            return own(device.createBuffer(new BufferInfo(flags, MemoryUsageBit.DEVICE, size, stride)));
        };

        try {
            this._params = buffer(96, BufferUsageBit.UNIFORM);
            this._control = buffer((8 + this._groups + capacity * 9) * 4, BufferUsageBit.STORAGE | BufferUsageBit.INDIRECT);
            this._history = buffer(sizes[0], BufferUsageBit.STORAGE, 48);
            this._vertices = buffer(sizes[1], BufferUsageBit.STORAGE | BufferUsageBit.VERTEX, TRAIL_VERTEX_STRIDE);
            this._indices = buffer(sizes[2], BufferUsageBit.STORAGE | BufferUsageBit.INDEX, 4);
            this._moduleData = buffer(compiled.data.byteLength, BufferUsageBit.STORAGE);
            this._moduleData.update(compiled.data);
            this._clearSet = own(this._createBaseSet(this._shared.dummyState, this._shared.dummyCounters));
            new Uint32Array(this._paramsData).set([capacity, maxPoints, this._groups, 0]);
            Mat4.toArray(new Float32Array(this._paramsData), Mat4.IDENTITY, 8);
            this.clear();
        } catch (error) {
            this.destroy();
            throw error;
        }
    }

    private _createBaseSet (state: Buffer, counters: Buffer): DescriptorSet {
        const set = this._device.createDescriptorSet(new DescriptorSetInfo(this._shared.setLayout));
        const buffers = [this._params, state, counters, this._control, this._history, this._moduleData, this._vertices, this._indices];
        TRAIL_RESOURCES.forEach((name, index) => set.bindBuffer(this._shared.bindings[name], buffers[index]));
        set.update();
        return set;
    }

    public get vertexBuffer (): Buffer {
        this._assertAlive();
        return this._vertices;
    }

    public get indexBuffer (): Buffer {
        this._assertAlive();
        return this._indices;
    }

    /** First 20 bytes contain indexed draw args; word 5 records this step's history overflow. */
    public get indirectBuffer (): Buffer {
        this._assertAlive();
        return this._control;
    }

    public configure (compiled: CompiledParticleComputeTrail): void {
        this._assertAlive();
        validateParticleComputeBuffer(this._device, compiled.data.byteLength, 'Compute trail modules');
        flushParticleComputeCommands(this._device);
        this._compiled = compiled;
        if (this._moduleData.size !== compiled.data.byteLength) {
            const next = this._device.createBuffer(new BufferInfo(
                BufferUsageBit.STORAGE | BufferUsageBit.TRANSFER_DST | BufferUsageBit.TRANSFER_SRC,
                MemoryUsageBit.DEVICE,
                compiled.data.byteLength,
                4,
            ));
            const old = this._moduleData;
            this._owned[this._owned.indexOf(old)] = next;
            this._moduleData = next;
            const moduleBinding = this._shared.bindings.TrailModules;
            this._clearSet.bindBuffer(moduleBinding, next);
            this._clearSet.update();
            for (const set of this._stateSets.sets) {
                set.bindBuffer(moduleBinding, next);
                set.update();
            }
            old.destroy();
        }
        this._moduleData.update(compiled.data);
    }

    public update (
        state: Buffer,
        counters: Buffer,
        dt: number,
        worldMatrix: Readonly<Mat4> = Mat4.IDENTITY,
        systemTime = 0,
        simulationSpace = ParticleSpace.World,
        activeBound = this.capacity,
    ): void {
        this._assertAlive();
        if (!Number.isFinite(dt) || dt < 0 || !Number.isFinite(systemTime)) {
            throw new RangeError('Trail time must be finite and dt nonnegative');
        }
        if (state.size < this.capacity * PARTICLE_COMPUTE_STATE_STRIDE || counters.size < 4) {
            throw new RangeError(`Trail requires the full ${PARTICLE_COMPUTE_STATE_STRIDE}-byte particle state buffer`);
        }
        if (simulationSpace !== ParticleSpace.Local && simulationSpace !== ParticleSpace.World) {
            throw new Error('Unsupported trail simulation space');
        }
        const source = this._compiled.lifeTime;
        const lifetime = typeof source === 'number' ? source : source.evaluate(systemTime, 1);
        if (!Number.isFinite(lifetime) || lifetime < 0) {
            throw new RangeError('Trail lifetime must be finite and nonnegative');
        }

        const clampedLive = Math.min(this.capacity, Math.max(0, activeBound));
        if (clampedLive === 0 && !this._mayHaveHistory) {
            this._tick++;
            return;
        }
        this._mayHaveHistory = clampedLive > 0;

        const floats = new Float32Array(this._paramsData);
        new Uint32Array(this._paramsData)[3] = simulationSpace === ParticleSpace.Local && this._compiled.space === ParticleSpace.World ? 1 : 0;
        floats.set([dt, lifetime, systemTime, this._tick++], 4);
        Mat4.toArray(floats, worldMatrix, 8);
        for (let i = 4; i < floats.length; ++i) {
            if (!Number.isFinite(floats[i])) {
                throw new RangeError('Trail parameters must fit finite f32 values');
            }
        }
        const set = this._stateSets.acquire(state, counters);
        const liveGroups = clampedLive > 0 ? Math.ceil(clampedLive / PARTICLE_COMPUTE_WORKGROUP_SIZE) : 0;
        this._submit(set, 1, TRAIL_SHADERS.length, liveGroups);
    }

    /** Reset with the particle system's clear/reset; does not upload history or wait for the GPU. */
    public clear (): void {
        this._assertAlive();
        this._tick = 0;
        this._mayHaveHistory = false;
        new Float32Array(this._paramsData)[6] = 0;
        this._submit(this._clearSet, 0, 1);
    }

    /** Disable/reenable keeps particle birth-delay progress, as CPU TrailModule.onDisable does. */
    public clearHistory (): void {
        this._assertAlive();
        this._mayHaveHistory = false;
        const floats = new Float32Array(this._paramsData);
        const time = floats[6];
        floats[6] = -1;
        this._submit(this._clearSet, 0, 1);
        floats[6] = time;
    }

    private _submit (set: DescriptorSet, start: number, end: number, liveGroups = this._groups): void {
        const mapLiveDispatch = liveGroups === 0 ? null
            : liveGroups === this._groups ? this._groupDispatch : createParticleComputeDispatch(this._device, liveGroups);
        const feature = ComputeTrailShaderFeature;
        const rawFeatures = this._compiled.data[ComputeTrailOffset.SHADER_FEATURES] | 0;
        const features = (rawFeatures & feature.VALID) !== 0
            ? rawFeatures : feature.WIDTH_SPLINE | feature.COLOR_OVERTIME_GRADIENT | feature.COLOR_OVER_TRAIL_GRADIENT;
        const pipelines: PipelineState[] = [];
        for (let i = start; i < end; ++i) {
            const kernel = TRAIL_SHADERS[i];
            const defines: MacroRecord = kernel.name === 'update-history' || kernel.name === 'emit-geometry' ? {
                CC_PARTICLE_HAS_SPLINE: Boolean(features & feature.WIDTH_SPLINE),
                CC_PARTICLE_HAS_GRADIENT: Boolean(features & (kernel.name === 'update-history'
                    ? feature.COLOR_OVERTIME_GRADIENT : feature.COLOR_OVER_TRAIL_GRADIENT)),
            } : {};
            pipelines.push(getParticleComputeProgram(this._device, 'trail', kernel.name, defines).pipeline);
        }
        runParticleComputePass(this._device, this, () => {
            this._params.update(this._paramsData);
        }, (cmd) => {
            for (let i = start; i < end; ++i) {
                const pipeline = pipelines[i - start];
                const kernel = TRAIL_SHADERS[i];
                if (kernel.name === 'map-live') {
                    if (!mapLiveDispatch) {
                        continue;
                    }
                    cmd.bindPipelineState(pipeline);
                    cmd.bindDescriptorSet(this._shared.setIndex, set);
                    cmd.dispatch(mapLiveDispatch);
                    continue;
                }
                cmd.bindPipelineState(pipeline);
                cmd.bindDescriptorSet(this._shared.setIndex, set);
                cmd.dispatch(kernel.singleGroup ? this._singleDispatch : this._groupDispatch);
            }
        });
    }

    public destroy (): void {
        if (this._destroyed) return;
        this._destroyed = true;
        flushParticleComputeCommands(this._device);
        this._stateSets.clear();
        for (let i = this._owned.length - 1; i >= 0; --i) {
            this._owned[i].destroy();
        }
        this._owned.length = 0;
    }

    private _assertAlive (): void {
        if (this._destroyed) {
            throw new Error('ParticleComputeTrail has been destroyed');
        }
    }
}
