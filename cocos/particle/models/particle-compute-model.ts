/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { Mesh } from '../../3d/assets/mesh';
import { Material, RenderingSubMesh } from '../../asset/assets';
import {
    Attribute, AttributeName, Buffer, BufferInfo, BufferUsageBit, Format, FormatInfos,
    MemoryUsageBit, PrimitiveMode,
} from '../../gfx';
import ParticleBatchModel from './particle-batch-model';
import { ParticleComputeRenderData } from '../compute/particle-compute-render-data';
import {
    flushParticleComputeCommands, validateParticleComputeBuffer, validateParticleComputeCapacity,
} from '../compute/particle-compute-effect';

/**
 * @engineInternal
 * Scene model for GPU simulation using the CPU particle material/attribute ABI.
 * Inherits the legacy model's public shape for renderer compatibility, but never
 * allocates or uploads its CPU per-particle arrays.
 */
export default class ParticleComputeModel extends ParticleBatchModel {
    private _computeCapacity = 0;
    private _computeMesh: Mesh | null = null;
    private _computeAttributes: Attribute[] | null = null;
    private _computeMaterial: Material | null = null;
    private _computeSubMesh: RenderingSubMesh | null = null;
    private _renderData: ParticleComputeRenderData | null = null;
    private _simulationState: Buffer | null = null;
    private _simulationCounters: Buffer | null = null;
    private _simulationRevision = 0;
    private _simulationActiveBound: number | undefined = undefined;
    private _renderDirty = true;
    private _indexedArguments: Buffer | null = null;
    private _textureAnimation = false;

    public setCapacity (capacity: number): void {
        if (capacity !== 0) validateParticleComputeCapacity(capacity);
        if (capacity === this._computeCapacity) return;
        const previous = this._computeCapacity;
        this._computeCapacity = capacity;
        try {
            this._rebuildComputeGeometry();
        } catch (error) {
            this._computeCapacity = previous;
            throw error;
        }
    }

    public setVertexAttributes (mesh: Mesh | null, attributes: Attribute[]): void {
        this._computeMesh = mesh;
        this._computeAttributes = attributes;
        this._rebuildComputeGeometry();
    }

    public updateMaterial (material: Material): void {
        this._computeMaterial = material;
        if (this._computeSubMesh) this.setSubModelMaterial(0, material);
        else this._rebuildComputeGeometry();
    }

    /** Keep the previous SubModel intact if rebuilding the new one fails mid-initialization. */
    public override initSubModel (idx: number, subMeshData: RenderingSubMesh, mat: Material): void {
        const previous = this._subModels[idx] ?? null;
        if (previous) this._subModels[idx] = null!;
        try {
            super.initSubModel(idx, subMeshData, mat);
        } catch (error) {
            const partial = this._subModels[idx];
            if (previous) {
                this._subModels[idx] = previous;
            } else if (idx === this._subModels.length - 1) {
                this._subModels.pop();
            } else {
                this._subModels[idx] = null!;
            }
            if (partial && partial !== previous) {
                partial.descriptorSet?.destroy();
                partial.inputAssembler?.destroy();
                partial.worldBoundDescriptorSet?.destroy();
            }
            throw error;
        }
        previous?.destroy();
    }

    public setSimulationBuffers (state: Buffer, counters: Buffer, revision?: number, activeBound?: number): void {
        const nextRevision = revision ?? this._simulationRevision;
        if (this._simulationState !== state || this._simulationCounters !== counters || this._simulationRevision !== nextRevision) {
            this._renderDirty = true;
        }
        this._simulationState = state;
        this._simulationCounters = counters;
        this._simulationRevision = nextRevision;
        this._simulationActiveBound = activeBound;
    }

    public setTextureAnimationEnabled (enabled: boolean): void {
        if (this._textureAnimation !== enabled) {
            this._textureAnimation = enabled;
            this._renderDirty = true;
        }
    }

    /** Count is deliberately ignored: the GPU writes the real indirect count. */
    public updateIA (_count: number): void {
        if (!this._renderData || !this._simulationState || !this._simulationCounters || !this._renderDirty) return;
        if (this._simulationActiveBound === undefined) {
            this._renderData.prepare(this._simulationState, this._simulationCounters, this._textureAnimation);
        } else {
            this._renderData.prepare(this._simulationState, this._simulationCounters, this._textureAnimation, this._simulationActiveBound);
        }
        this._renderDirty = false;
    }

    public clear (): void {
        // The runtime clears its counters separately. Clear draw args immediately
        // as well, even if there is no subsequent simulation tick this frame.
        flushParticleComputeCommands(this._device);
        this._indexedArguments?.update(new Uint32Array(5));
        this._renderDirty = false;
    }

    public doDestroy (): void {
        flushParticleComputeCommands(this._device);
        this._renderData?.destroy();
        this._renderData = null;
        this._computeSubMesh?.destroy();
        this._computeSubMesh = null;
        this._indexedArguments = null;
        this._computeAttributes = null;
        this._computeMaterial = null;
        this._computeMesh = null;
        this._simulationState = null;
        this._simulationCounters = null;
        super.doDestroy();
    }

    private _rebuildComputeGeometry (): void {
        const attributes = this._computeAttributes;
        const material = this._computeMaterial;
        if (!attributes || !material) return;
        flushParticleComputeCommands(this._device);
        const prevRenderData = this._renderData;
        const prevSubMesh = this._computeSubMesh;

        let dynamicStride = 0;
        let staticStride = 0;
        const staticOffsets = new Map<string, number>();
        for (const attribute of attributes) {
            if (attribute.stream === 0 && attribute.isInstanced) {
                dynamicStride += FormatInfos[attribute.format].size;
            } else if (attribute.stream === 1 && !attribute.isInstanced) {
                staticOffsets.set(attribute.name, staticStride);
                staticStride += FormatInfos[attribute.format].size;
            } else {
                throw new Error('Compute particles require the instanced CPU particle vertex layout');
            }
        }
        const mesh = this._computeMesh;
        const primitive = mesh?.struct.primitives[0];
        const vertexCount = mesh && primitive ? mesh.struct.vertexBundles[primitive.vertexBundelIndices[0]].view.count : 4;
        const indexCount = primitive ? (primitive.indexView?.count ?? vertexCount) : 6;
        if (vertexCount < 1 || staticStride < 1) throw new Error('Compute particle geometry is empty');
        const vertices = new ArrayBuffer(staticStride * vertexCount);
        if (mesh) {
            const copy = (source: AttributeName, target: AttributeName): boolean => {
                const offset = staticOffsets.get(target);
                return offset !== undefined && mesh.copyAttribute(0, source, vertices, staticStride, offset);
            };
            copy(AttributeName.ATTR_TEX_COORD, AttributeName.ATTR_TEX_COORD);
            copy(AttributeName.ATTR_POSITION, AttributeName.ATTR_TEX_COORD3);
            copy(AttributeName.ATTR_NORMAL, AttributeName.ATTR_NORMAL);
            if (!copy(AttributeName.ATTR_COLOR, AttributeName.ATTR_COLOR1)) {
                const offset = staticOffsets.get(AttributeName.ATTR_COLOR1);
                const color = attributes.find((attribute) => attribute.name === AttributeName.ATTR_COLOR1 && attribute.stream === 1);
                if (offset !== undefined && color) {
                    const view = new DataView(vertices);
                    for (let i = 0; i < vertexCount; ++i) {
                        if (color.format === Format.RGBA8) view.setUint32(i * staticStride + offset, 0xFFFFFFFF, true);
                        else throw new Error('Missing mesh color requires the RGBA8 white fallback layout');
                    }
                }
            }
        } else {
            new Float32Array(vertices).set([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
        }
        const indexStride = vertexCount > 65535 || primitive?.indexView?.stride === 4 ? 4 : 2;
        const indices = indexStride === 4 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);
        if (mesh && primitive?.indexView) mesh.copyIndices(0, indices);
        else if (mesh) for (let i = 0; i < indexCount; ++i) indices[i] = i;
        else indices.set([0, 1, 2, 3, 2, 1]);

        const allocated: Buffer[] = [];
        let nextRenderData: ParticleComputeRenderData | null = null;
        let nextSubMesh: RenderingSubMesh | null = null;
        const create = (usage: BufferUsageBit, size: number, stride: number): Buffer => {
            if (usage & BufferUsageBit.STORAGE) validateParticleComputeBuffer(this._device, size, 'Compute particle model buffer');
            const buffer = this._device.createBuffer(new BufferInfo(usage | BufferUsageBit.TRANSFER_DST, MemoryUsageBit.DEVICE, size, stride));
            allocated.push(buffer);
            return buffer;
        };
        try {
            const dynamicBuffer = create(
                BufferUsageBit.STORAGE | BufferUsageBit.VERTEX | BufferUsageBit.TRANSFER_SRC,
                Math.max(1, this._computeCapacity) * dynamicStride,
                dynamicStride,
            );
            const staticBuffer = create(BufferUsageBit.VERTEX, vertices.byteLength, staticStride);
            staticBuffer.update(vertices);
            const indexBuffer = create(BufferUsageBit.INDEX, indices.byteLength, indexStride);
            indexBuffer.update(indices);
            const argumentsBuffer = create(BufferUsageBit.STORAGE | BufferUsageBit.INDIRECT | BufferUsageBit.TRANSFER_SRC, 20, 20);
            argumentsBuffer.update(new Uint32Array(5));
            nextRenderData = new ParticleComputeRenderData(this._device, this._computeCapacity, dynamicBuffer, argumentsBuffer, indexCount);
            nextSubMesh = new RenderingSubMesh(
                [dynamicBuffer, staticBuffer],
                attributes,
                PrimitiveMode.TRIANGLE_LIST,
                indexBuffer,
                argumentsBuffer,
            );
            nextSubMesh.iaInfo.gpuIndirect = { offset: 0, count: 1, stride: 20 };
            this.initSubModel(0, nextSubMesh, material);
            this._renderData = nextRenderData;
            this._computeSubMesh = nextSubMesh;
            this._indexedArguments = argumentsBuffer;
            this._renderDirty = true;
            prevRenderData?.destroy();
            prevSubMesh?.destroy();
        } catch (error) {
            nextRenderData?.destroy();
            if (nextSubMesh) nextSubMesh.destroy();
            else allocated.forEach((buffer) => buffer.destroy());
            throw error;
        }
    }
}

export { ParticleComputeModel };
