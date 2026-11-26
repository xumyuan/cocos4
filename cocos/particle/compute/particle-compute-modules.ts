/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import GradientRange from '../animator/gradient-range';
import { ParticleRenderMode } from '../enum';
import type { ParticleSystem } from '../particle-system';
import {
    ComputeColorOffset, ComputeScalar, DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES, PACKED_SCALAR_CURVE_COUNT_OFFSET,
    PARTICLE_COMPUTE_MODULE_HEADER_FLOATS, ParticleComputeCurvePacker,
} from './particle-compute-curves';

export { ParticleComputeCurvePacker } from './particle-compute-curves';

/** Float offsets in the fixed header. Values are pointers into the same float buffer where noted. */
export const ComputeModuleOffset = {
    FLAGS: 0,
    RENDER_MODE: 1,
    SIZE_AXES: 2,
    LIMIT_AXES: 3,
    ROTATION_AXES: 4,
    DAMPEN: 5,
    TILES_X: 6,
    TILES_Y: 7,
    ANIMATION: 8,
    RANDOM_ROW: 9,
    ROW: 10,
    CYCLES: 11,
    START_SIZE_3D: 12,
    START_ROTATION_3D: 13,
    GRAVITY: 14,
    START_COLOR: 15,
    SIZE: 16,
    SIZE_X: 17,
    SIZE_Y: 18,
    SIZE_Z: 19,
    COLOR: 20,
    FORCE_X: 21,
    FORCE_Y: 22,
    FORCE_Z: 23,
    VELOCITY_X: 24,
    VELOCITY_Y: 25,
    VELOCITY_Z: 26,
    SPEED: 27,
    LIMIT: 28,
    LIMIT_X: 29,
    LIMIT_Y: 30,
    LIMIT_Z: 31,
    ROTATION_X: 32,
    ROTATION_Y: 33,
    ROTATION_Z: 34,
    START_FRAME: 35,
    FRAME: 36,
    START_SIZE_X: 37,
    START_SIZE_Y: 38,
    START_SIZE_Z: 39,
    START_ROTATION_X: 40,
    START_ROTATION_Y: 41,
    START_ROTATION_Z: 42,
    START_SPEED: 43,
    START_LIFE: 44,
    NOISE_STRENGTH: 45,
    NOISE_SPEED: 48,
    NOISE_FREQUENCY: 51,
    NOISE_OCTAVES: 52,
    NOISE_MULTIPLIER: 53,
    NOISE_SCALE: 54,
    SCALAR_END: 55,
    NOISE_PERM: 56,
    SHADER_FEATURES: 57,
    FORCE_ROTATION: 64,
    VELOCITY_ROTATION: 68,
    LIMIT_ROTATION: 72,
} as const;

export const ComputeShaderFeature = {
    VALID: 64,
    BIRTH_SPLINE: 1,
    BIRTH_GRADIENT: 2,
    UPDATE_SPLINE: 4,
    UPDATE_GRADIENT: 8,
    NOISE: 16,
} as const;

export const ComputeModuleFlag = {
    SIZE: 1 << 0,
    COLOR: 1 << 1,
    FORCE: 1 << 2,
    VELOCITY: 1 << 3,
    LIMIT: 1 << 4,
    ROTATION: 1 << 5,
    TEXTURE: 1 << 6,
    NOISE: 1 << 7,
} as const;

const NOISE_PERMUTATION = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240,
    21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88,
    237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
    111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80,
    73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182,
    189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22,
    39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210,
    144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84,
    204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78,
    66, 215, 61, 156, 180,
];

const NOISE_PERMUTATION_PAIRS = Array.from(
    { length: NOISE_PERMUTATION.length / 2 },
    (_, i) => NOISE_PERMUTATION[i * 2] | (NOISE_PERMUTATION[i * 2 + 1] << 8),
);

export type ParticleComputeModuleSource = Partial<Pick<ParticleSystem,
'gravityModifier' | 'startColor' | 'startSize3D' | 'startSizeX' | 'startSizeY' | 'startSizeZ'
| 'startRotation3D' | 'startRotationX' | 'startRotationY' | 'startRotationZ' | 'startSpeed' | 'startLifetime'
| 'sizeOvertimeModule' | 'colorOverLifetimeModule' | 'forceOvertimeModule' | 'velocityOvertimeModule'
| 'limitVelocityOvertimeModule' | 'rotationOvertimeModule' | 'textureAnimationModule' | 'noiseModule'>>;

export interface ParticleComputeModuleOptions {
    /** Sample budget for opaque/custom JS evaluators only; standard RealCurves retain keys. */
    readonly samples?: number;
    readonly renderMode?: number;
    /** Explicit transforms; identity preserves the current CPU renderer's disabled module.update calls. */
    readonly forceRotation?: readonly [number, number, number, number];
    readonly velocityRotation?: readonly [number, number, number, number];
    readonly limitRotation?: readonly [number, number, number, number];
    /** Reject custom-evaluator LUT error; does not measure direct-curve f32 roundoff. */
    readonly maxSampledScalarError?: number;
    /** Defaults to true. Providing an error threshold always enables estimation. */
    readonly estimateError?: boolean;
}

export interface CompiledParticleComputeModules {
    readonly data: Float32Array;
    readonly flags: number;
    readonly samples: number;
    readonly maxSampledScalarError: number | undefined;
}

function enabledModule<T extends { enable: boolean }> (module: T | null | undefined): T | undefined {
    return module?.enable ? module : undefined;
}

/**
 * @engineInternal
 * Compile module parameters/curves when assets change; no particle state is simulated on CPU.
 */
export function compileParticleComputeModules (
    source: ParticleComputeModuleSource = {},
    options: ParticleComputeModuleOptions = {},
): CompiledParticleComputeModules {
    const packer = new ParticleComputeCurvePacker(
        PARTICLE_COMPUTE_MODULE_HEADER_FLOATS,
        options.samples ?? DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES,
        options.estimateError !== false || options.maxSampledScalarError !== undefined,
    );
    const moduleData = packer.values;
    const offsets = ComputeModuleOffset;
    moduleData[offsets.SCALAR_END] = offsets.START_LIFE + 1;

    const scalar = (offset: number, curve: ComputeScalar | null | undefined, fallback = 0): void => {
        moduleData[offset] = packer.scalar(curve, fallback);
    };

    const size = enabledModule(source.sizeOvertimeModule);
    const color = enabledModule(source.colorOverLifetimeModule);
    const force = enabledModule(source.forceOvertimeModule);
    const velocity = enabledModule(source.velocityOvertimeModule);
    const limit = enabledModule(source.limitVelocityOvertimeModule);
    const rotation = enabledModule(source.rotationOvertimeModule);
    const texture = enabledModule(source.textureAnimationModule);
    const noise = enabledModule(source.noiseModule);

    let flags = 0;
    if (size) {
        flags |= ComputeModuleFlag.SIZE;
    }
    if (color) {
        flags |= ComputeModuleFlag.COLOR;
    }
    if (force) {
        flags |= ComputeModuleFlag.FORCE;
    }
    if (velocity) {
        flags |= ComputeModuleFlag.VELOCITY;
    }
    if (limit) {
        flags |= ComputeModuleFlag.LIMIT;
    }
    if (rotation) {
        flags |= ComputeModuleFlag.ROTATION;
    }
    if (texture) {
        flags |= ComputeModuleFlag.TEXTURE;
    }
    if (noise) {
        flags |= ComputeModuleFlag.NOISE;
    }

    if (texture && (!Number.isInteger(texture.numTilesX) || texture.numTilesX < 1
        || !Number.isInteger(texture.numTilesY) || texture.numTilesY < 1)) {
        throw new RangeError('Enabled texture animation requires positive integer tile counts');
    }

    // from offset 0 to SCALAR_END, all values are either scalars or pointers into the same float buffer
    moduleData[offsets.FLAGS] = flags;
    moduleData[offsets.RENDER_MODE] = options.renderMode ?? ParticleRenderMode.Billboard;
    moduleData[offsets.SIZE_AXES] = size?.separateAxes ? 1 : 0;
    moduleData[offsets.LIMIT_AXES] = limit?.separateAxes ? 1 : 0;
    moduleData[offsets.ROTATION_AXES] = rotation?.separateAxes ? 1 : 0;
    moduleData[offsets.DAMPEN] = limit?.dampen ?? 0;
    moduleData[offsets.TILES_X] = Math.max(1, texture?.numTilesX ?? 1);
    moduleData[offsets.TILES_Y] = Math.max(1, texture?.numTilesY ?? 1);
    moduleData[offsets.ANIMATION] = texture?.animation ?? 0;
    moduleData[offsets.RANDOM_ROW] = texture?.randomRow ? 1 : 0;
    moduleData[offsets.ROW] = texture?.rowIndex ?? 0;
    moduleData[offsets.CYCLES] = texture?.cycleCount ?? 1;
    moduleData[offsets.START_SIZE_3D] = source.startSize3D ? 1 : 0;
    moduleData[offsets.START_ROTATION_3D] = source.startRotation3D ? 1 : 0;

    scalar(offsets.GRAVITY, source.gravityModifier, 1);
    moduleData[offsets.START_COLOR] = packer.color(source.startColor);
    scalar(offsets.SIZE, size?.size, 1);
    scalar(offsets.SIZE_X, size?.x, 1);
    scalar(offsets.SIZE_Y, size?.y, 1);
    scalar(offsets.SIZE_Z, size?.z, 1);
    moduleData[offsets.COLOR] = packer.color(color?.color);
    scalar(offsets.FORCE_X, force?.x);
    scalar(offsets.FORCE_Y, force?.y);
    scalar(offsets.FORCE_Z, force?.z);
    scalar(offsets.VELOCITY_X, velocity?.x);
    scalar(offsets.VELOCITY_Y, velocity?.y);
    scalar(offsets.VELOCITY_Z, velocity?.z);
    scalar(offsets.SPEED, velocity?.speedModifier, 1);
    scalar(offsets.LIMIT, limit?.limit);
    scalar(offsets.LIMIT_X, limit?.limitX);
    scalar(offsets.LIMIT_Y, limit?.limitY);
    scalar(offsets.LIMIT_Z, limit?.limitZ);
    scalar(offsets.ROTATION_X, rotation?.x);
    scalar(offsets.ROTATION_Y, rotation?.y);
    scalar(offsets.ROTATION_Z, rotation?.z);
    scalar(offsets.START_FRAME, texture?.startFrame);
    scalar(offsets.FRAME, texture?.frameOverTime);
    scalar(offsets.START_SIZE_X, source.startSizeX, 1);
    scalar(offsets.START_SIZE_Y, source.startSizeY, 1);
    scalar(offsets.START_SIZE_Z, source.startSizeZ, 1);
    scalar(offsets.START_ROTATION_X, source.startRotationX);
    scalar(offsets.START_ROTATION_Y, source.startRotationY);
    scalar(offsets.START_ROTATION_Z, source.startRotationZ);
    scalar(offsets.START_SPEED, source.startSpeed, 1);
    scalar(offsets.START_LIFE, source.startLifetime, 1);

    moduleData[offsets.NOISE_STRENGTH] = noise?.strengthX ?? 0;
    moduleData[offsets.NOISE_STRENGTH + 1] = noise?.strengthY ?? 0;
    moduleData[offsets.NOISE_STRENGTH + 2] = noise?.strengthZ ?? 0;
    moduleData[offsets.NOISE_SPEED] = noise?.noiseSpeedX ?? 0;
    moduleData[offsets.NOISE_SPEED + 1] = noise?.noiseSpeedY ?? 0;
    moduleData[offsets.NOISE_SPEED + 2] = noise?.noiseSpeedZ ?? 0;
    moduleData[offsets.NOISE_FREQUENCY] = noise?.noiseFrequency ?? 1;
    moduleData[offsets.NOISE_OCTAVES] = 1;
    moduleData[offsets.NOISE_MULTIPLIER] = noise?.octaveMultiplier ?? 0.5;
    moduleData[offsets.NOISE_SCALE] = noise?.octaveScale ?? 2;

    if (noise) {
        const octaves = noise.octaves ?? 1;
        if (!Number.isInteger(octaves) || octaves < 1 || octaves > 4) {
            throw new RangeError('Noise octaves must be in [1, 4]');
        }

        moduleData[offsets.NOISE_OCTAVES] = octaves;
        moduleData[offsets.NOISE_PERM] = packer.append(NOISE_PERMUTATION_PAIRS);
    }

    const isSpline = (field: number): boolean => packer.values[moduleData[field] + PACKED_SCALAR_CURVE_COUNT_OFFSET] < 0;
    const isGradient = (field: number): boolean => {
        const mode = packer.values[moduleData[field] + ComputeColorOffset.MODE];
        return mode !== GradientRange.Mode.Color && mode !== GradientRange.Mode.TwoColors;
    };
    let features = ComputeShaderFeature.VALID;
    for (let field = offsets.START_SIZE_X; field <= offsets.START_LIFE; ++field) {
        if (isSpline(field)) {
            features |= ComputeShaderFeature.BIRTH_SPLINE;
            break;
        }
    }
    if (isGradient(offsets.START_COLOR)) {
        features |= ComputeShaderFeature.BIRTH_GRADIENT;
    }

    const activeUpdateRanges: Array<readonly [number, number, number]> = [
        [0, offsets.GRAVITY, offsets.GRAVITY],
        [ComputeModuleFlag.SIZE, offsets.SIZE, offsets.SIZE_Z],
        [ComputeModuleFlag.FORCE, offsets.FORCE_X, offsets.FORCE_Z],
        [ComputeModuleFlag.VELOCITY, offsets.VELOCITY_X, offsets.SPEED],
        [ComputeModuleFlag.LIMIT, offsets.LIMIT, offsets.LIMIT_Z],
        [ComputeModuleFlag.ROTATION, offsets.ROTATION_X, offsets.ROTATION_Z],
        [ComputeModuleFlag.TEXTURE, offsets.START_FRAME, offsets.FRAME],
    ];
    for (const [mask, begin, end] of activeUpdateRanges) {
        if (mask !== 0 && (flags & mask) === 0) {
            continue;
        }
        for (let field = begin; field <= end; ++field) {
            if (isSpline(field)) {
                features |= ComputeShaderFeature.UPDATE_SPLINE;
                break;
            }
        }
        if ((features & ComputeShaderFeature.UPDATE_SPLINE) !== 0) {
            break;
        }
    }
    if ((flags & ComputeModuleFlag.COLOR) !== 0 && isGradient(offsets.COLOR)) {
        features |= ComputeShaderFeature.UPDATE_GRADIENT;
    }
    if ((flags & ComputeModuleFlag.NOISE) !== 0) {
        features |= ComputeShaderFeature.NOISE;
    }
    moduleData[offsets.SHADER_FEATURES] = features;

    const identity = [0, 0, 0, 1];
    for (const [offset, value] of [
        [offsets.FORCE_ROTATION, options.forceRotation ?? identity],
        [offsets.VELOCITY_ROTATION, options.velocityRotation ?? identity],
        [offsets.LIMIT_ROTATION, options.limitRotation ?? identity],
    ] as const) {
        for (let i = 0; i < 4; ++i) {
            moduleData[offset + i] = value[i];
        }
    }

    const data = packer.finish();

    if (data.some((value) => !Number.isFinite(value))) {
        throw new RangeError('Particle module parameters must be finite');
    }

    if (options.maxSampledScalarError !== undefined && packer.maxSampledScalarError !== undefined
        && packer.maxSampledScalarError > options.maxSampledScalarError) {
        throw new RangeError(`Particle curve LUT error ${packer.maxSampledScalarError} exceeds ${options.maxSampledScalarError}`);
    }

    return { data, flags, samples: packer.samples, maxSampledScalarError: packer.maxSampledScalarError };
}
