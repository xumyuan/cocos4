/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import {
    Buffer, BufferInfo, BufferUsageBit,
    DescriptorSet, DescriptorSetInfo, Device, DispatchInfo, MemoryUsageBit, PipelineState,
} from '../../gfx';
import {
    createParticleComputeDispatch, flushParticleComputeCommands, getParticleComputeBindingLayout,
    getParticleComputeProgram, PARTICLE_COMPUTE_WORKGROUP_SIZE, ParticleComputeStateSets, runParticleComputePass,
} from './particle-compute-effect';
import type { Disposable, ParticleComputeBindingLayout } from './particle-compute-effect';

export const ComputeRenderVertexOffset = {
    POSITION_X: 0,
    POSITION_Y: 1,
    POSITION_Z: 2,
    FRAME: 3,
    SIZE_X: 4,
    SIZE_Y: 5,
    SIZE_Z: 6,
    ROTATION_X: 7,
    ROTATION_Y: 8,
    ROTATION_Z: 9,
    COLOR: 10,
    WORDS: 11,
    VELOCITY_X: 11,
    VELOCITY_Y: 12,
    VELOCITY_Z: 13,
    STRETCHED_WORDS: 14,
} as const;

/** CPU particle material's instanced stream: 10 floats followed by packed RGBA8. */
const PARTICLE_COMPUTE_RENDER_STRIDE = ComputeRenderVertexOffset.WORDS * Uint32Array.BYTES_PER_ELEMENT;
/** Stretched Billboard adds ultimate velocity.xyz after packed color. */
const PARTICLE_COMPUTE_STRETCHED_RENDER_STRIDE = ComputeRenderVertexOffset.STRETCHED_WORDS * Uint32Array.BYTES_PER_ELEMENT;

const RENDER_RESOURCES = ['RenderParams', 'SimulationState', 'SimulationCounters', 'InstanceVertices', 'IndirectArguments'] as const;

interface RenderPrepareShared extends ParticleComputeBindingLayout<typeof RENDER_RESOURCES[number]> {
    readonly pipeline: PipelineState;
}

const sharedRenderPipelines = new WeakMap<Device, RenderPrepareShared>();

function getSharedRenderPipeline (device: Device): RenderPrepareShared {
    const existing = sharedRenderPipelines.get(device);
    if (existing) return existing;
    const program = getParticleComputeProgram(device, 'render', 'prepare');
    const shared = { ...getParticleComputeBindingLayout(program, RENDER_RESOURCES), pipeline: program.pipeline };
    sharedRenderPipelines.set(device, shared);
    return shared;
}

/**
 * Converts simulation state into the existing CPU particle material's vertex
 * attributes entirely on GPU. Output buffers are owned by ParticleComputeModel.
 * Submit before the scene render pass; no camera-dependent work is done here.
 */
export class ParticleComputeRenderData {
    private readonly _owned: Disposable[] = [];
    private readonly _sets: ParticleComputeStateSets;
    private readonly _shared: RenderPrepareShared;
    private readonly _dispatchInfo: DispatchInfo;
    private _parameters!: Buffer;
    private _destroyed = false;
    private readonly _parameterData = new Uint32Array(4);

    public constructor (
        private readonly _device: Device,
        private readonly _capacity: number,
        private readonly _vertexBuffer: Buffer,
        private readonly _indirectBuffer: Buffer,
        indexCount: number,
    ) {
        const device = _device;
        this._dispatchInfo = createParticleComputeDispatch(device, Math.max(1, Math.ceil(_capacity / PARTICLE_COMPUTE_WORKGROUP_SIZE)));
        if (_vertexBuffer.stride !== PARTICLE_COMPUTE_RENDER_STRIDE
            && _vertexBuffer.stride !== PARTICLE_COMPUTE_STRETCHED_RENDER_STRIDE) {
            throw new Error('Unsupported Compute particle instance stride');
        }
        this._shared = getSharedRenderPipeline(device);
        const { bindings } = this._shared;
        this._sets = new ParticleComputeStateSets(
            (state, counters) => this._createSet(state, counters),
            bindings.SimulationState,
            bindings.SimulationCounters,
        );
        const own = <T extends Disposable>(resource: T): T => {
            this._owned.push(resource);
            return resource;
        };
        try {
            this._parameters = own(device.createBuffer(new BufferInfo(
                BufferUsageBit.UNIFORM | BufferUsageBit.TRANSFER_DST,
                MemoryUsageBit.DEVICE,
                16,
                16,
            )));
            this._parameterData.set([_capacity, indexCount, _vertexBuffer.stride / 4, 0]);
            this._parameters.update(this._parameterData);
        } catch (error) {
            this.destroy();
            throw error;
        }
    }

    private _createSet (state: Buffer, counters: Buffer): DescriptorSet {
        const { bindings } = this._shared;
        const set = this._device.createDescriptorSet(new DescriptorSetInfo(this._shared.setLayout));
        this._owned.push(set);
        set.bindBuffer(bindings.RenderParams, this._parameters);
        set.bindBuffer(bindings.InstanceVertices, this._vertexBuffer);
        set.bindBuffer(bindings.IndirectArguments, this._indirectBuffer);
        set.bindBuffer(bindings.SimulationState, state);
        set.bindBuffer(bindings.SimulationCounters, counters);
        set.update();
        return set;
    }

    public prepare (state: Buffer, counters: Buffer, textureAnimation = false, activeBound = this._capacity): void {
        if (this._destroyed) {
            throw new Error('ParticleComputeRenderData has been destroyed');
        }
        const set = this._sets.acquire(state, counters);
        const textureFlag = textureAnimation ? 1 : 0;
        const clamped = Math.min(this._capacity, Math.max(0, activeBound));
        const maxGroups = Math.max(1, Math.ceil(this._capacity / PARTICLE_COMPUTE_WORKGROUP_SIZE));
        const activeGroups = Math.max(1, Math.ceil(clamped / PARTICLE_COMPUTE_WORKGROUP_SIZE));
        const dispatch = activeGroups === maxGroups ? this._dispatchInfo : createParticleComputeDispatch(this._device, activeGroups);
        runParticleComputePass(this._device, this, () => {
            if (this._parameterData[3] !== textureFlag) {
                this._parameterData[3] = textureFlag;
                this._parameters.update(this._parameterData);
            }
        }, (cmd) => {
            cmd.bindPipelineState(this._shared.pipeline);
            cmd.bindDescriptorSet(this._shared.setIndex, set);
            cmd.dispatch(dispatch);
        });
    }

    public destroy (): void {
        if (this._destroyed) return;
        this._destroyed = true;
        flushParticleComputeCommands(this._device);
        this._sets.clear();
        for (let i = this._owned.length - 1; i >= 0; --i) {
            this._owned[i].destroy();
        }
        this._owned.length = 0;
    }
}
