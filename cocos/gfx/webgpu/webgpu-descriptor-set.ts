/*
 Copyright (c) 2024 Xiamen Yaji Software Co., Ltd.

 https://www.cocos.com/

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

import { DescriptorSet } from '../base/descriptor-set';
import { WebGPUBuffer } from './webgpu-buffer';
import { IWebGPUGPUDescriptorSet, IWebGPUGPUDescriptor } from './webgpu-gpu-objects';
import { WebGPUSampler } from './webgpu-sampler';
import { WebGPUTexture } from './webgpu-texture';
import { WebGPUDescriptorSetLayout } from './webgpu-descriptor-set-layout';
import {
    DescriptorSetInfo,
    DESCRIPTOR_BUFFER_TYPE,
    DESCRIPTOR_SAMPLER_TYPE,
    DescriptorSetLayoutBinding,
    DescriptorType,
    Filter,
    ViewDimension,
    DESCRIPTOR_STORAGE_BUFFER_TYPE,
} from '../base/define';
import { WebGPUDeviceManager } from './define';
import { FormatToWGPUFormatType, SEPARATE_SAMPLER_BINDING_OFFSET } from './webgpu-commands';
import type { WebGPUShaderSetLayout } from './webgpu-shader-layout';

export class WebGPUDescriptorSet extends DescriptorSet {
    get gpuDescriptorSet (): IWebGPUGPUDescriptorSet {
        return this._gpuDescriptorSet as IWebGPUGPUDescriptorSet;
    }

    private _gpuDescriptorSet: IWebGPUGPUDescriptorSet | null = null;
    private _bindGroupEntries: Map<number, GPUBindGroupEntry> = new Map<number, GPUBindGroupEntry>();
    private _dynamicOffsets: number[] = [];
    private readonly _nativeGroups = new Map<GPUBindGroupLayout, GPUBindGroup>();

    get dynamicOffsets (): number[] {
        return this._dynamicOffsets;
    }

    get dynamicOffsetCount (): number {
        return this._dynamicOffsets.length;
    }

    public initialize (info: Readonly<DescriptorSetInfo>): void {
        const layout = this._layout = info.layout as WebGPUDescriptorSetLayout;
        const { bindings, descriptorIndices, descriptorCount } = layout.gpuDescriptorSetLayout!;

        this._buffers = Array(descriptorCount).fill(null);
        this._textures = Array(descriptorCount).fill(null);
        this._samplers = Array(descriptorCount).fill(null);

        const gpuDescriptors: IWebGPUGPUDescriptor[] = [];
        const bindGroup = null!;
        const bindGroupLayout = null!;
        this._gpuDescriptorSet = { gpuDescriptors, descriptorIndices, bindGroup, bindGroupLayout };
        this._isDirty = true;
        const bindingSize = bindings.length;
        for (let i = 0; i < bindingSize; ++i) {
            const binding = bindings[i];
            const bindCount = binding.count;
            for (let j = 0; j < bindCount; j++) {
                gpuDescriptors.push({
                    type: binding.descriptorType,
                    gpuBuffer: null,
                    gpuTexture: null,
                    gpuSampler: null,
                });
            }
        }
    }

    public destroy (): void {
        this._layout = null;
        this._gpuDescriptorSet = null;
        this._buffers.length = 0;
        this._textures.length = 0;
        this._samplers.length = 0;
        this._bindGroupEntries.clear();
        this._nativeGroups.clear();
    }

    private _bindBufferEntry (bind: DescriptorSetLayoutBinding, buffer: WebGPUBuffer): void {
        const descriptorIndex = this._gpuDescriptorSet!.descriptorIndices[bind.binding];
        const destBind = this._gpuDescriptorSet!.gpuDescriptors[descriptorIndex];
        if (destBind) {
            destBind.gpuBuffer = buffer.gpuBuffer;
        }
        const nativeBuffer = buffer.gpuBuffer.gpuBuffer;
        const bindGrpEntry: GPUBindGroupEntry = {
            binding: bind.binding,
            resource: {
                buffer: nativeBuffer!,
                offset: buffer.gpuBuffer.gpuOffset,
                size: buffer.gpuBuffer.size,
            },
        };
        this._bindGroupEntries.set(bindGrpEntry.binding, bindGrpEntry);
        buffer.resetChange();
    }

    private _bindTextureEntry (bind: DescriptorSetLayoutBinding, texture: WebGPUTexture): void {
        const descriptorIndex = this._gpuDescriptorSet!.descriptorIndices[bind.binding];
        this._gpuDescriptorSet!.gpuDescriptors[descriptorIndex].gpuTexture = texture.gpuTexture;
        const nativeTexView = texture.getNativeTextureView()!;
        const bindGrpEntry: GPUBindGroupEntry = {
            binding: bind.binding,
            resource: nativeTexView,
        };
        this._bindGroupEntries.set(bindGrpEntry.binding, bindGrpEntry);
        texture.resetChange();
    }

    private _bindSamplerEntry (bind: DescriptorSetLayoutBinding, sampler: WebGPUSampler): void {
        const samplerIdx = bind.binding + SEPARATE_SAMPLER_BINDING_OFFSET;
        const descriptorIndex = this._gpuDescriptorSet!.descriptorIndices[bind.binding];
        this._gpuDescriptorSet!.gpuDescriptors[descriptorIndex].gpuSampler = sampler.gpuSampler;
        const device = WebGPUDeviceManager.instance;
        const currTexture = (this._textures[descriptorIndex] || device.defaultResource.texture) as WebGPUTexture;
        const levelCount = currTexture.levelCount;
        const texFormat = currTexture.format;
        const isUnFilter = FormatToWGPUFormatType(texFormat) === 'unfilterable-float'
        || (FormatToWGPUFormatType(texFormat) === 'float' && !device.floatFilterable);
        if (isUnFilter) {
            sampler.gpuSampler.minFilter = Filter.POINT;
            sampler.gpuSampler.magFilter = Filter.POINT;
            sampler.gpuSampler.mipFilter = Filter.POINT;
        }
        const currGPUSampler = sampler.createGPUSampler(levelCount);
        const bindSamplerGrpEntry: GPUBindGroupEntry = {
            binding: samplerIdx,
            resource: currGPUSampler as GPUSampler,
        };
        this._bindGroupEntries.set(samplerIdx, bindSamplerGrpEntry);
        sampler.resetChange();
    }

    private _applyBindGroup (): void {
        if (this._isDirty && this._gpuDescriptorSet) {
            const layout = this._layout as WebGPUDescriptorSetLayout;
            this._bindGroupEntries.clear();
            this._nativeGroups.clear();
            this._gpuDescriptorSet.bindGroup = null!;
            this._gpuDescriptorSet.bindGroupLayout = null!;
            this._dynamicOffsets.length = 0;
            const descriptors = this._gpuDescriptorSet.gpuDescriptors;
            const bindings = layout.gpuDescriptorSetLayout!.bindings;
            const descCount = bindings.length;
            const device = WebGPUDeviceManager.instance;
            for (let i = 0; i < descCount; ++i) {
                const binding = bindings[i];
                const bindIdx = binding.binding;
                const descriptorIndex = this._gpuDescriptorSet.descriptorIndices[bindIdx];
                const descType = descriptors[descriptorIndex].type;
                if (descType & DESCRIPTOR_BUFFER_TYPE) {
                    const defaultBuffer = device.defaultResource.buffer;
                    let buffer = (this._buffers[descriptorIndex] || defaultBuffer) as WebGPUBuffer;
                    if (buffer === defaultBuffer
                        && (descType & DESCRIPTOR_STORAGE_BUFFER_TYPE)) {
                        buffer =  device.defaultResource.getStorageBuffer(bindIdx);
                    }
                    this._bindBufferEntry(binding, buffer);
                    if (descType & (DescriptorType.DYNAMIC_STORAGE_BUFFER | DescriptorType.DYNAMIC_UNIFORM_BUFFER)) {
                        this._dynamicOffsets.push(bindIdx);
                    }
                } else if (descType & (DESCRIPTOR_SAMPLER_TYPE | DescriptorType.STORAGE_IMAGE)) {
                    const isStorageImage = (descType & DescriptorType.STORAGE_IMAGE) === DescriptorType.STORAGE_IMAGE;
                    if (isStorageImage) {
                        // storage texture (compute read/write): bind the texture view, no sampler
                        let currTex = this._textures[descriptorIndex] as WebGPUTexture;
                        if (!currTex || (currTex.hasChange && !currTex.gpuTexture)) {
                            currTex = device.defaultResource.texture;
                        }
                        this._bindTextureEntry(binding, currTex);
                    } else if ((descType & DescriptorType.SAMPLER) !== DescriptorType.SAMPLER) {
                        // sampled texture
                        let currTex = this._textures[descriptorIndex] as WebGPUTexture;
                        // null or destroyed?
                        if (!currTex || (currTex.hasChange && !currTex.gpuTexture)) {
                            if (binding.viewDimension === ViewDimension.TEXCUBE) {
                                currTex = device.defaultResource.cubeTexture;
                            } else {
                                currTex = device.defaultResource.texture;
                            }
                        }
                        this._bindTextureEntry(binding, currTex);
                    }

                    if (!isStorageImage
                        && !((descType & DescriptorType.INPUT_ATTACHMENT) === DescriptorType.INPUT_ATTACHMENT
                            || (descType & DescriptorType.TEXTURE) === DescriptorType.TEXTURE)) {
                        // sampler
                        const currSampler = (this._samplers[descriptorIndex] || device.defaultResource.sampler) as WebGPUSampler;
                        this._bindSamplerEntry(binding, currSampler);
                    }
                }
            }
            this._isDirty = false;
            // Wait for a concrete shader profile. The GFX layout may contain
            // unused bindings or texture dimensions unknown to this template.
        }
    }

    private _hasResourceChange (resource: WebGPUBuffer | WebGPUTexture | WebGPUSampler): boolean {
        if (resource && resource.hasChange) {
            return true;
        }
        return false;
    }

    private _isResourceChange (): boolean {
        const layout = this._layout as WebGPUDescriptorSetLayout;
        if (!layout) {
            return false;
        }
        return layout.gpuDescriptorSetLayout!.bindings.some((bind) => {
            const binding = bind.binding;
            const index = this._gpuDescriptorSet!.descriptorIndices[binding];
            const resource = this._buffers[index] as WebGPUBuffer || this._textures[index] || this._samplers[index];
            return this._hasResourceChange(resource);
        });
    }

    public prepare (force?: boolean, profile?: WebGPUShaderSetLayout): GPUBindGroup | null {
        const gpuDescriptorSet = this._gpuDescriptorSet;
        if (!gpuDescriptorSet) {
            return null;
        }
        // Rebuild when forced, when the bind group has never been built, or when resources changed.
        if (force || this._isDirty || this._isResourceChange()) {
            this._isDirty = true;
            this._applyBindGroup();
        }
        const layout = this._layout as WebGPUDescriptorSetLayout;
        const nativeLayout = profile?.layout || layout.getNativeLayout();
        let group = this._nativeGroups.get(nativeLayout);
        if (!group) {
            group = this._createBindGroup(nativeLayout, profile?.entries || layout.gpuDescriptorSetLayout!.entries);
            this._nativeGroups.set(nativeLayout, group);
        }
        gpuDescriptorSet.bindGroupLayout = nativeLayout;
        gpuDescriptorSet.bindGroup = group;
        return group;
    }

    private _createBindGroup (nativeLayout: GPUBindGroupLayout, activeEntries: readonly GPUBindGroupLayoutEntry[]): GPUBindGroup {
        const device = WebGPUDeviceManager.instance;
        const nativeDevice = device.nativeDevice;
        const entries: GPUBindGroupEntry[] = [];
        for (const active of activeEntries) {
            const source = this._bindGroupEntries.get(active.binding);
            if (!source) throw new Error(`No GFX resource bound for shader binding ${active.binding}`);
            let entry = source;
            if (active.texture) {
                const index = this._gpuDescriptorSet!.descriptorIndices[active.binding];
                const texture = this._textures[index] as WebGPUTexture | undefined;
                if (!texture || !texture.gpuTexture?.gpuTexture) {
                    const dimension = active.texture.viewDimension || '2d';
                    if (dimension === 'cube') {
                        entry = { binding: active.binding, resource: device.defaultResource.cubeTexture.getNativeTextureView()! };
                    } else if (dimension === 'cube-array' || dimension === '2d-array') {
                        const fallback = dimension === 'cube-array' ? device.defaultResource.cubeTexture : device.defaultResource.texture;
                        entry = { binding: active.binding, resource: fallback.gpuTexture.gpuTexture!.createView({ dimension }) };
                    }
                }
            }
            entries.push(entry);
        }
        const bindGroup = nativeDevice?.createBindGroup({
            layout: nativeLayout,
            entries,
        });
        return bindGroup!;
    }

    public update (): void {
        this._applyBindGroup();
    }
}
