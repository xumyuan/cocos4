/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { RealCurve } from '../../core/curves/curve';
import CurveRange from '../animator/curve-range';
import { DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES } from './particle-compute-curves';
import type { ComputeColorRange, ComputeGradient, ComputeScalar } from './particle-compute-curves';
import type { ParticleComputeModuleOptions, ParticleComputeModuleSource } from './particle-compute-modules';
import type { ParticleComputeTrailSource } from './particle-compute-trail';

const evaluateRange = CurveRange.prototype.evaluate;
const evaluateSpline = RealCurve.prototype.evaluate;
const identities = new WeakMap<object, number>();
let nextIdentity = 1;

function identity (object: object): number {
    let value = identities.get(object);
    if (value === undefined) {
        value = nextIdentity++;
        identities.set(object, value);
    }
    return value;
}

/** Content keys inspect authored fields only: no curve evaluation and no traversal of owner/target cycles. */
class ConfigKey {
    private readonly _parts: string[] = [];
    private _cacheable = true;

    public value (value: number | boolean | undefined | null): void {
        this._parts.push(
            value === undefined ? 'u' : value === null ? 'n' : Object.is(value, -0) ? '-0' : String(value),
        );
    }

    public values (values: readonly (number | boolean | undefined | null)[]): void {
        this.value(values.length);
        for (const value of values) {
            this.value(value);
        }
    }

    public scalar (value: ComputeScalar | undefined | null): void {
        if (value === undefined || value === null || typeof value === 'number') {
            this._parts.push('scalar');
            this.value(value);
            return;
        }
        // Arbitrary user evaluators can close over mutable external data. Never cache them
        // based on object identity or a few samples: the next compile must observe changes.
        if (!(value instanceof CurveRange) || value.evaluate !== evaluateRange) {
            this._cacheable = false;
            return;
        }
        this._parts.push('curve');
        this.values([identity(value), value.mode, value.constant, value.constantMin, value.constantMax, value.multiplier]);
        if (value.mode === CurveRange.Mode.Curve) {
            this._spline(value.spline);
        } else if (value.mode === CurveRange.Mode.TwoCurves) {
            this._spline(value.splineMin);
            this._spline(value.splineMax);
        }
    }

    private _spline (spline: RealCurve): void {
        if (!(spline instanceof RealCurve) || spline.evaluate !== evaluateSpline) {
            this._cacheable = false;
            return;
        }
        this.values([identity(spline), spline.preExtrapolation, spline.postExtrapolation, spline.keyFramesCount]);
        for (let i = 0; i < spline.keyFramesCount; ++i) {
            const key = spline.getKeyframeValue(i);
            this.values([
                spline.getKeyframeTime(i),
                key.value,
                key.interpolationMode,
                key.leftTangent,
                key.rightTangent,
                key.leftTangentWeight,
                key.rightTangentWeight,
                key.tangentWeightMode,
                key.easingMethod,
            ]);
        }
    }

    public color (range: ComputeColorRange | undefined | null): void {
        this._parts.push('color');
        if (!range) {
            this.value(null);
            return;
        }
        this.value(range.mode);
        for (const color of [range.color, range.colorMin, range.colorMax]) {
            this.values([color.r, color.g, color.b, color.a]);
        }
        this._gradient(range.gradient);
        this._gradient(range.gradientMin);
        this._gradient(range.gradientMax);
    }

    private _gradient (gradient: ComputeGradient): void {
        this.values([gradient.mode, gradient.colorKeys.length, gradient.alphaKeys.length]);
        for (const key of gradient.colorKeys) {
            this.values([key.time, key.color.r, key.color.g, key.color.b, key.color.a]);
        }
        for (const key of gradient.alphaKeys) {
            this.values([key.time, key.alpha]);
        }
    }

    public finish (): string | null {
        return this._cacheable ? this._parts.join('|') : null;
    }
}

/** Null means an opaque evaluator requires recompilation on this update. */
export function getParticleComputeModuleConfigKey (
    source: ParticleComputeModuleSource,
    options: ParticleComputeModuleOptions = {},
): string | null {
    const key = new ConfigKey();
    key.values([options.samples, options.renderMode, options.maxSampledScalarError, options.estimateError]);
    key.values([source.startSize3D, source.startRotation3D]);
    for (const value of [
        source.gravityModifier,
        source.startSizeX,
        source.startSizeY,
        source.startSizeZ,
        source.startRotationX,
        source.startRotationY,
        source.startRotationZ,
        source.startSpeed,
        source.startLifetime,
    ]) {
        key.scalar(value);
    }
    key.color(source.startColor);

    const size = source.sizeOvertimeModule;
    key.values([!!size, size?.enable, size?.separateAxes]);
    for (const value of [size?.size, size?.x, size?.y, size?.z]) {
        key.scalar(value);
    }

    const color = source.colorOverLifetimeModule;
    key.values([!!color, color?.enable]);
    key.color(color?.color);

    const force = source.forceOvertimeModule;
    key.values([!!force, force?.enable, force?.space]);
    for (const value of [force?.x, force?.y, force?.z]) {
        key.scalar(value);
    }

    const velocity = source.velocityOvertimeModule;
    key.values([!!velocity, velocity?.enable, velocity?.space]);
    for (const value of [velocity?.x, velocity?.y, velocity?.z, velocity?.speedModifier]) {
        key.scalar(value);
    }

    const limit = source.limitVelocityOvertimeModule;
    key.values([!!limit, limit?.enable, limit?.space, limit?.separateAxes, limit?.dampen]);
    for (const value of [limit?.limit, limit?.limitX, limit?.limitY, limit?.limitZ]) {
        key.scalar(value);
    }

    const rotation = source.rotationOvertimeModule;
    key.values([!!rotation, rotation?.enable, rotation?.separateAxes]);
    for (const value of [rotation?.x, rotation?.y, rotation?.z]) {
        key.scalar(value);
    }

    const texture = source.textureAnimationModule;
    key.values([
        !!texture,
        texture?.enable,
        texture?.numTilesX,
        texture?.numTilesY,
        texture?.animation,
        texture?.randomRow,
        texture?.rowIndex,
        texture?.cycleCount,
    ]);
    key.scalar(texture?.startFrame);
    key.scalar(texture?.frameOverTime);

    const noise = source.noiseModule;
    key.values([
        !!noise,
        noise?.enable,
        noise?.strengthX,
        noise?.strengthY,
        noise?.strengthZ,
        noise?.noiseSpeedX,
        noise?.noiseSpeedY,
        noise?.noiseSpeedZ,
        noise?.noiseFrequency,
        noise?.octaves,
        noise?.octaveMultiplier,
        noise?.octaveScale,
        noise?.remapX,
        noise?.remapY,
        noise?.remapZ,
    ]);

    for (const rotationValue of [options.forceRotation, options.velocityRotation, options.limitRotation]) {
        if (rotationValue) {
            key.values(rotationValue);
        } else {
            key.value(null);
        }
    }
    return key.finish();
}

export function getParticleComputeTrailConfigKey (
    source: ParticleComputeTrailSource,
    samples = DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES,
): string | null {
    const key = new ConfigKey();
    key.values([samples, source.minParticleDistance, source.widthFromParticle, source.colorFromParticle, source.space]);
    // Keep the lifetime object's identity as well as its authored values: compiled
    // trail data retains this evaluator for per-emitter time-dependent lifetime.
    key.scalar(source.lifeTime);
    key.scalar(source.widthRatio);
    key.color(source.colorOvertime);
    key.color(source.colorOverTrail);
    return key.finish();
}
