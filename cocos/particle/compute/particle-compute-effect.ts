/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { cclegacy } from '../../core';
import {
    CommandBuffer, CommandBufferInfo, Device, DispatchInfo, PipelineBindPoint, PipelineState, PipelineStateInfo,
} from '../../gfx';
import type { Buffer, DescriptorSet, DescriptorSetLayout, Shader, ShaderInfo } from '../../gfx';
import type { MacroRecord } from '../../render-scene/core/pass-utils';
import type { ProgramLibrary } from '../../rendering/custom/private';

export const ComputeDrawOffset = {
    INDEX_COUNT: 0,
    INSTANCE_COUNT: 1,
    FIRST_INDEX: 2,
    BASE_VERTEX: 3,
    FIRST_INSTANCE: 4,
} as const;

// Match local_size_x and the ParticleState layout in the compute shaders.
export const PARTICLE_COMPUTE_WORKGROUP_SIZE = 64;
export const PARTICLE_COMPUTE_STATE_STRIDE = 176;

/** Particle handles, counters and shader array indices use unsigned 32-bit values. */
export function validateParticleComputeCapacity (capacity: number): void {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 0xFFFFFFFF) {
        throw new RangeError('Compute particle capacity must be a positive uint32 integer');
    }
}

/** Check enabled device limits before allocating GPU-owned particle storage. */
export function validateParticleComputeBuffer (device: Device, bytes: number, label: string): void {
    if (!Number.isSafeInteger(bytes) || bytes < 4 || bytes % 4 !== 0 || bytes / 4 > 0xFFFFFFFF) {
        throw new RangeError(`${label} requires ${bytes} bytes, outside the shader's uint32 word address range`);
    }
    const caps = device.capabilities;
    const limits = [
        ['maxBufferSize', caps?.maxBufferSize],
        ['maxShaderStorageBlockSize', caps?.maxShaderStorageBlockSize],
    ] as const;
    for (const [name, limit] of limits) {
        if (limit > 0 && bytes > limit) {
            throw new RangeError(`${label} requires ${bytes} bytes; the enabled device ${name} is ${limit} bytes`);
        }
    }
}

/** Spread large pools over two dispatch dimensions instead of imposing a pool-size cap. */
export function createParticleComputeDispatch (device: Device, groups: number): DispatchInfo {
    const dispatch = new DispatchInfo(1, 1, 1);
    setParticleComputeDispatch(device, dispatch, groups);
    return dispatch;
}

export function setParticleComputeDispatch (device: Device, dispatch: DispatchInfo, groups: number): void {
    if (!Number.isSafeInteger(groups) || groups < 1) {
        throw new RangeError('Invalid particle compute workgroup count');
    }
    const limits = device.capabilities?.maxComputeWorkGroupCount;
    const x = limits?.x > 0 ? Math.min(groups, limits.x) : groups;
    const y = Math.ceil(groups / x);
    if (limits?.y > 0 && y > limits.y) {
        throw new RangeError(`Particle compute requires ${groups} workgroups; the device supports ${limits.x} x ${limits.y}`);
    }
    dispatch.groupCountX = x;
    dispatch.groupCountY = y;
    dispatch.groupCountZ = 1;
}

export interface Disposable {
    destroy(): void;
}

interface StateSetEntry {
    state: Buffer;
    counters: Buffer;
    readonly set: DescriptorSet;
}

/**
 * Cache descriptor sets for recently bound simulations. When both slots
 * miss, the older slot is rebound. The create callback owns the new set.
 */
export class ParticleComputeStateSets {
    private readonly _entries: StateSetEntry[] = [];
    private _nextSlot = 0;

    public constructor (
        private readonly _create: (state: Buffer, counters: Buffer) => DescriptorSet,
        private readonly _stateBinding: number,
        private readonly _countersBinding: number,
    ) {}

    public get sets (): DescriptorSet[] {
        return this._entries.map((entry) => entry.set);
    }

    public acquire (state: Buffer, counters: Buffer): DescriptorSet {
        for (let i = 0; i < this._entries.length; ++i) {
            const entry = this._entries[i];
            if (entry.state === state && entry.counters === counters) return entry.set;
        }
        if (this._entries.length < 2) {
            const set = this._create(state, counters);
            this._entries.push({ state, counters, set });
            return set;
        }
        const entry = this._entries[this._nextSlot];
        this._nextSlot = 1 - this._nextSlot;
        entry.state = state;
        entry.counters = counters;
        entry.set.bindBuffer(this._stateBinding, state);
        entry.set.bindBuffer(this._countersBinding, counters);
        entry.set.update();
        return entry.set;
    }

    public clear (): void {
        this._entries.length = 0;
        this._nextSlot = 0;
    }
}

interface DeviceComputeBatch {
    readonly cmd: CommandBuffer;
    readonly owners: Set<object>;
    readonly tasks: Array<(cmd: CommandBuffer) => void>;
}

const deviceBatches = new WeakMap<Device, DeviceComputeBatch>();
const pendingDevices = new Set<Device>();
let hookedDirector: unknown = null;
let inDirectorFrame = false;

/**
 * Flush queued particle compute tasks in one synchronous begin/submitComputePass/end block
 * so WebGPUCommandBuffer's module-level state cannot interleave with other command buffers.
 */
export function flushParticleComputeCommands (device?: Device | null): void {
    if (!device) {
        Array.from(pendingDevices).forEach((active) => flushParticleComputeCommands(active));
        return;
    }
    const batch = deviceBatches.get(device);
    if (!batch || batch.tasks.length === 0) return;
    const tasks = batch.tasks.splice(0);
    batch.owners.clear();
    pendingDevices.delete(device);
    batch.cmd.begin();
    try {
        for (let i = 0; i < tasks.length; ++i) {
            tasks[i](batch.cmd);
        }
        batch.cmd.submitComputePass();
    } finally {
        batch.cmd.end();
    }
}

function ensureFrameHooks (): void {
    const director = cclegacy.director;
    const Director = cclegacy.Director;
    if (!director || !Director || hookedDirector === director) return;
    hookedDirector = director;
    director.on(Director.EVENT_BEGIN_FRAME, () => {
        inDirectorFrame = true;
    });
    const finish = (): void => {
        inDirectorFrame = false;
        flushParticleComputeCommands();
    };
    director.on(Director.EVENT_BEFORE_RENDER, finish);
    director.on(Director.EVENT_END_FRAME, finish);
    director.on(Director.EVENT_RESET, finish);
}

/**
 * Queue particle compute dispatches onto a per-device batch.
 * Distinct owners within a Director frame merge into one atomic submit at BEFORE_RENDER;
 * repeated dispatches on the same owner or standalone calls outside a frame flush immediately.
 */
export function runParticleComputePass (
    device: Device,
    owner: object,
    upload: (() => void) | null,
    record: (cmd: CommandBuffer) => void,
): void {
    ensureFrameHooks();
    let batch = deviceBatches.get(device);
    if (!batch) {
        batch = {
            cmd: device.createCommandBuffer(new CommandBufferInfo(device.queue)),
            owners: new Set<object>(),
            tasks: [],
        };
        deviceBatches.set(device, batch);
    }
    if (batch.owners.has(owner)) {
        flushParticleComputeCommands(device);
    }
    upload?.();
    batch.owners.add(owner);
    batch.tasks.push(record);
    pendingDevices.add(device);
    if (!inDirectorFrame) {
        flushParticleComputeCommands(device);
    }
}

interface ParticleComputeProgram {
    readonly pipeline: PipelineState;
    readonly shaderInfo: ShaderInfo;
}

export interface ParticleComputeBindingLayout<Resource extends string> {
    readonly setLayout: DescriptorSetLayout;
    readonly setIndex: number;
    readonly bindings: Record<Resource, number>;
}

const computePrograms = new WeakMap<Shader, ParticleComputeProgram>();

export function getParticleComputeProgram (
    device: Device,
    family: string,
    kernel: string,
    defines: MacroRecord = {},
): ParticleComputeProgram {
    const rendering = cclegacy.rendering;
    if (!rendering?.enableEffectImport) {
        throw new Error('Compute particles require the custom render pipeline');
    }
    const library = rendering.programLib as ProgramLibrary;
    const pass = `particle-compute-${family}`;
    const name = `particles/${pass}|particle-compute/${family}/${kernel}`;
    const invalidID = rendering.INVALID_ID as number;
    const passID = rendering.getPassID(pass) as number;
    const phaseID = passID === invalidID ? invalidID : rendering.getPhaseID(passID, kernel) as number;
    const variant = phaseID === invalidID ? null : library.getProgramVariant(device, phaseID, name, defines);
    if (!variant) {
        throw new Error(`Compute particle program ${name} is not registered; reimport the particle compute effects`);
    }
    const existing = computePrograms.get(variant.shader);
    if (existing) return existing;
    const shaderInfo = library.getShaderInfo(phaseID, name);
    const info = new PipelineStateInfo();
    info.shader = variant.shader;
    info.pipelineLayout = library.getPipelineLayout(device, phaseID, name);
    info.bindPoint = PipelineBindPoint.COMPUTE;
    const program = { pipeline: device.createPipelineState(info), shaderInfo };
    computePrograms.set(variant.shader, program);
    return program;
}

export function getParticleComputeBindingLayout<Resource extends string> (
    { pipeline, shaderInfo }: ParticleComputeProgram,
    names: readonly Resource[],
): ParticleComputeBindingLayout<Resource> {
    const resources = [...shaderInfo.blocks, ...shaderInfo.buffers];
    const bindings = {} as Record<Resource, number>;
    let setIndex = -1;
    for (const name of names) {
        const resource = resources.find((entry) => entry.name === name);
        if (!resource || (setIndex !== -1 && setIndex !== resource.set)) {
            throw new Error(`Invalid Compute particle binding ${name} in ${pipeline.shader.name}`);
        }
        setIndex = resource.set;
        bindings[name] = resource.binding;
    }
    const setLayout = pipeline.pipelineLayout.setLayouts[setIndex];
    if (!setLayout) {
        throw new Error(`Missing Compute particle descriptor set in ${pipeline.shader.name}`);
    }
    return { setLayout, setIndex, bindings };
}
