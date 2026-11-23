/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { Mat4, Quat } from '../../core';
import { ParticleSpace } from '../enum';
import type { ParticleSystem } from '../particle-system';

export const PARTICLE_COMPUTE_BIRTH_FLOATS = 64;

export const ComputeBirthOffset = {
    EMIT_DT: 0,
    NORMALIZED_TIME: 1,
    SHAPE_ENABLED: 2,
    SHAPE_TYPE: 3,
    EMIT_FROM: 4,
    RADIUS: 5,
    RADIUS_THICKNESS: 6,
    ANGLE: 7,
    LENGTH: 8,
    ARC: 9,
    ARC_MODE: 10,
    RANDOM_POSITION_AMOUNT: 11,
    SPHERICAL_DIRECTION_AMOUNT: 12,
    ALIGN_TO_DIRECTION: 13,
    WORLD_SPACE: 14,
    RENDER_MODE: 15,
    SCALE: 16,
    TIME: 19,
    POSITION: 20,
    SHAPE_ROTATION: 24,
    WORLD_MATRIX: 28,
    WORLD_MATRIX_Y: 32,
    WORLD_MATRIX_Z: 36,
    WORLD_MATRIX_W: 40,
    WORLD_ROTATION: 44,
    BOX_THICKNESS: 48,
    ARC_SPREAD: 51,
    ARC_SPEED: 52,
} as const;

export const ComputeBirthBatchOffset = {
    BEGIN: 0,
    END: 1,
    SNAPSHOT: 2,
    HEADER_WORDS: 4,
} as const;

export interface ParticleComputeBirthBatch {
    readonly count: number;
    readonly data: Float32Array;
}

/**
 * @engineInternal
 * Snapshot one emit call, rather than constructing or uploading individual particles.
 */
export function compileParticleComputeBirth (ps: ParticleSystem, count: number, emitDt: number): ParticleComputeBirthBatch {
    const data = new Float32Array(PARTICLE_COMPUTE_BIRTH_FLOATS);
    const shape = ps._shapeModule;
    const world = ps.simulationSpace === ParticleSpace.World;
    const offsets = ComputeBirthOffset;
    data[offsets.EMIT_DT] = emitDt;
    data[offsets.NORMALIZED_TIME] = (ps.time % ps.duration) / ps.duration;
    data[offsets.SHAPE_ENABLED] = shape?.enable ? 1 : 0;
    data[offsets.WORLD_SPACE] = world ? 1 : 0;
    data[offsets.RENDER_MODE] = ps.renderer.renderMode;
    data[offsets.TIME] = ps.time;

    const shapeRotation = new Quat();
    if (shape?.enable) {
        data[offsets.SHAPE_TYPE] = shape.shapeType;
        data[offsets.EMIT_FROM] = shape.emitFrom;
        data[offsets.RADIUS] = shape.radius;
        data[offsets.RADIUS_THICKNESS] = shape.radiusThickness;
        data[offsets.ANGLE] = shape.angle * Math.PI / 180;
        data[offsets.LENGTH] = shape.length;
        data[offsets.ARC] = shape.arc * Math.PI / 180;
        data[offsets.ARC_MODE] = shape.arcMode;
        data[offsets.RANDOM_POSITION_AMOUNT] = shape.randomPositionAmount;
        data[offsets.SPHERICAL_DIRECTION_AMOUNT] = shape.sphericalDirectionAmount;
        data[offsets.ALIGN_TO_DIRECTION] = shape.alignToDirection ? 1 : 0;
        data.set([shape.scale.x, shape.scale.y, shape.scale.z], offsets.SCALE);
        data.set([shape.position.x, shape.position.y, shape.position.z], offsets.POSITION);
        Quat.fromEuler(shapeRotation, shape.rotation.x, shape.rotation.y, shape.rotation.z);
        data.set([shape.boxThickness.x, shape.boxThickness.y, shape.boxThickness.z], offsets.BOX_THICKNESS);
        data[offsets.ARC_SPREAD] = shape.arcSpread;
        data[offsets.ARC_SPEED] = shape.arcSpeed.evaluate(ps.time, 1);
    } else {
        data.set([1, 1, 1], offsets.SCALE);
    }
    Quat.toArray(data, shapeRotation, offsets.SHAPE_ROTATION);

    const matrix = new Mat4();
    const rotation = new Quat();
    if (world) {
        ps.node.getWorldMatrix(matrix);
        ps.node.getWorldRotation(rotation);
    }
    Mat4.toArray(data, matrix, offsets.WORLD_MATRIX);
    Quat.toArray(data, rotation, offsets.WORLD_ROTATION);
    return { count, data };
}
