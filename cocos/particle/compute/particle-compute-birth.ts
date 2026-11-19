/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */
import { Mat4, Quat } from '../../core';
import { ParticleSpace } from '../enum';
import type { ParticleSystem } from '../particle-system';

const PARTICLE_COMPUTE_BIRTH_FLOATS = 64;
export interface ParticleComputeBirthBatch { readonly count: number; readonly data: Float32Array }

/**
 * @engineInternal
 * Snapshot one emit call, rather than constructing or uploading individual particles.
 */
export function compileParticleComputeBirth (ps: ParticleSystem, count: number, emitDt: number): ParticleComputeBirthBatch {
    const data = new Float32Array(PARTICLE_COMPUTE_BIRTH_FLOATS);
    const shape = ps._shapeModule;
    const world = ps.simulationSpace === ParticleSpace.World;
    data[0] = emitDt;
    data[1] = (ps.time % ps.duration) / ps.duration;
    data[2] = shape?.enable ? 1 : 0;
    data[14] = world ? 1 : 0;
    data[15] = ps.renderer.renderMode;
    data[19] = ps.time;
    const shapeRotation = new Quat();
    if (shape?.enable) {
        data[3] = shape.shapeType;
        data.set([shape.emitFrom, shape.radius, shape.radiusThickness, shape.angle * Math.PI / 180], 4);
        data.set([shape.length, shape.arc * Math.PI / 180, shape.arcMode, shape.randomPositionAmount], 8);
        data[12] = shape.sphericalDirectionAmount;
        data[13] = shape.alignToDirection ? 1 : 0;
        data.set([shape.scale.x, shape.scale.y, shape.scale.z], 16);
        data.set([shape.position.x, shape.position.y, shape.position.z], 20);
        Quat.fromEuler(shapeRotation, shape.rotation.x, shape.rotation.y, shape.rotation.z);
        data.set([shape.boxThickness.x, shape.boxThickness.y, shape.boxThickness.z, shape.arcSpread], 48);
        data[52] = shape.arcSpeed.evaluate(ps.time, 1);
    } else {
        data.set([1, 1, 1], 16);
    }
    Quat.toArray(data, shapeRotation, 24);
    const matrix = new Mat4();
    const rotation = new Quat();
    if (world) {
        ps.node.getWorldMatrix(matrix);
        ps.node.getWorldRotation(rotation);
    }
    Mat4.toArray(data, matrix, 28);
    Quat.toArray(data, rotation, 44);
    return { count, data };
}
