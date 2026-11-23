/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { ShaderStageFlagBit } from '../base/define';
import type { IWebGPUGPUShader } from './webgpu-gpu-objects';

export interface WebGPUShaderSetLayout {
    readonly layout: GPUBindGroupLayout;
    readonly entries: readonly GPUBindGroupLayoutEntry[];
    readonly dynamicBindings: readonly number[];
}

export interface WebGPUShaderPipelineLayout {
    readonly layout: GPUPipelineLayout;
    readonly sets: readonly WebGPUShaderSetLayout[];
}

/**
 * Match native layouts to translated WGSL declarations instead of the shared
 * GFX template's superset. Preserve native binding numbers and dynamic-buffer
 * semantics. This intentionally does not attempt WGSL entry-point reachability.
 */
export function getShaderLayoutEntries (
    shader: IWebGPUGPUShader,
    templates: readonly (readonly GPUBindGroupLayoutEntry[])[],
): GPUBindGroupLayoutEntry[][] {
    const sets = templates.map(() => new Map<number, GPUBindGroupLayoutEntry>());
    for (const stage of shader.gpuStages) {
        const visibility = stage.type === ShaderStageFlagBit.VERTEX ? 1 : stage.type === ShaderStageFlagBit.FRAGMENT ? 2 : 4;
        const declaration = /((?:@\w+\([^)]*\)\s*)+)var(?:\s*<\s*([^>]+)>)?\s+(\w+)\s*:\s*([^;]+);/g;
        for (let match = declaration.exec(stage.source); match; match = declaration.exec(stage.source)) {
            const groupMatch = /@group\((\d+)\)/.exec(match[1]);
            const bindingMatch = /@binding\((\d+)\)/.exec(match[1]);
            if (!groupMatch || !bindingMatch) {
                continue;
            }
            const group = Number(groupMatch[1]);
            const binding = Number(bindingMatch[1]);
            const template = templates[group]?.find((entry) => entry.binding === binding);
            if (!template) {
                throw new Error(`Shader ${shader.name} resource ${group}:${binding} is missing from its GFX template`);
            }
            const previous = sets[group].get(binding);
            if (previous) {
                previous.visibility |= visibility;
                continue;
            }
            const entry: GPUBindGroupLayoutEntry = { binding, visibility };
            const address = (match[2] || '').replace(/\s/g, '');
            const type = match[4].trim();
            if (address === 'uniform' || address.startsWith('storage')) {
                entry.buffer = { ...template.buffer,
                    type: address === 'uniform' ? 'uniform' : address.includes('read_write') ? 'storage' : 'read-only-storage' };
            } else if (type === 'sampler' || type === 'sampler_comparison') {
                entry.sampler = { ...template.sampler, type: type === 'sampler_comparison' ? 'comparison' : template.sampler?.type || 'filtering' };
            } else if (type === 'texture_external') {
                entry.externalTexture = {};
            } else if (type.startsWith('texture_')) {
                const kind = type.split('<')[0].trim();
                const suffix = kind.replace(/^texture_(?:storage_|depth_)?(?:multisampled_)?/, '');
                const dimensions: Record<string, GPUTextureViewDimension> = {
                    '1d': '1d', '2d': '2d', '2d_array': '2d-array', '3d': '3d', cube: 'cube', cube_array: 'cube-array',
                };
                const viewDimension = dimensions[suffix];
                if (!viewDimension) {
                    throw new Error(`Unsupported WGSL texture dimension: ${kind}`);
                }
                if (kind.includes('storage')) {
                    const argumentsList = /<\s*([^,>]+)\s*,\s*([^>]+)>/.exec(type);
                    if (!argumentsList) {
                        throw new Error(`Malformed WGSL storage texture: ${type}`);
                    }
                    const access = argumentsList[2].trim();
                    const nativeAccess = access === 'read' ? 'read-only' : access === 'read_write' ? 'read-write' : 'write-only';
                    entry.storageTexture = { ...template.storageTexture,
                        format: argumentsList[1].trim() as GPUTextureFormat,
                        viewDimension,
                        access: nativeAccess as GPUStorageTextureAccess };
                } else {
                    entry.texture = { ...template.texture,
                        viewDimension,
                        multisampled: kind.includes('multisampled'),
                        sampleType: kind.includes('depth') ? 'depth' : /<\s*i32\s*>/.test(type) ? 'sint'
                            : /<\s*u32\s*>/.test(type) ? 'uint' : template.texture?.sampleType || 'float' };
                }
            } else {
                throw new Error(`Unsupported WGSL resource declaration: ${type}`);
            }
            sets[group].set(binding, entry);
        }
    }
    // Creator's loose preview transform treats spread operands as arrays.
    // Convert the Map iterator explicitly so native layouts receive entries.
    return sets.map((entries) => Array.from(entries.values()).sort((a, b) => a.binding - b.binding));
}

/** Caller supplies offsets in the original GFX dynamic-binding order. */
export function getShaderDynamicOffsets (original: readonly number[], active: readonly number[], offsets: readonly number[]): number[] {
    return active.map((binding) => {
        const index = original.indexOf(binding);
        return index < 0 ? 0 : offsets[index] || 0;
    });
}
