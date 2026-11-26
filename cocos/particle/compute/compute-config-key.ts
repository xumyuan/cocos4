/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import { RealCurve } from '../../core/curves/curve';
import CurveRange from '../animator/curve-range';
import { DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES } from './particle-compute-curves';
import type { ComputeColor, ComputeColorRange, ComputeGradient, ComputeScalar } from './particle-compute-curves';
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

enum ValueTag {
    Number,
    Boolean,
    Null,
    Undefined,
    Identity,
}

const defaultOptions: ParticleComputeModuleOptions = {};
// Stage numeric keyframe fields together to avoid boxing each float at a polymorphic call site.
const keyframeValues = new Float64Array(9);

/** Reads authored values without evaluating curves or following owner references. */
class ConfigSnapshot {
    private _values: Float64Array;
    private _tags: Uint8Array;
    private _length = -1;
    private _previousLength = -1;
    private _cursor = 0;
    private _changed = false;
    private _opaque = false;

    public constructor (capacity: number) {
        this._values = new Float64Array(capacity);
        this._tags = new Uint8Array(capacity);
    }

    protected begin (): void {
        this._previousLength = this._length;
        // An interrupted traversal must be retried even if it already overwrote part of the snapshot.
        this.reset();
        this._cursor = 0;
        this._changed = false;
        this._opaque = false;
    }

    public reset (): void {
        this._length = -1;
    }

    protected end (): boolean {
        this._length = this._opaque ? -1 : this._cursor;
        return this._changed || this._cursor !== this._previousLength || this._opaque;
    }

    private _grow (): void {
        const values = new Float64Array(this._values.length * 2);
        const tags = new Uint8Array(values.length);
        values.set(this._values);
        tags.set(this._tags);
        this._values = values;
        this._tags = tags;
    }

    private _write (value: number, tag: ValueTag): void {
        const i = this._cursor++;
        if (i === this._values.length) {
            this._grow();
        }

        if (i >= this._previousLength || this._tags[i] !== tag || !Object.is(this._values[i], value)) {
            this._changed = true;
            this._values[i] = value;
            this._tags[i] = tag;
        }
    }

    protected value (value: number | boolean | undefined | null): void {
        if (typeof value === 'number') {
            this._write(value, ValueTag.Number);
        } else if (typeof value === 'boolean') {
            this._write(+value, ValueTag.Boolean);
        } else {
            this._write(0, value === null ? ValueTag.Null : ValueTag.Undefined);
        }
    }

    protected scalar (value: ComputeScalar | undefined | null): void {
        if (value === undefined || value === null || typeof value === 'number') {
            this.value(value);
            return;
        }

        // Custom evaluators may close over mutable state.
        if (!(value instanceof CurveRange) || value.evaluate !== evaluateRange) {
            this._opaque = true;
            return;
        }

        this._write(identity(value), ValueTag.Identity);
        this.value(value.mode);
        this.value(value.constant);
        this.value(value.constantMin);
        this.value(value.constantMax);
        this.value(value.multiplier);
        if (value.mode === CurveRange.Mode.Curve) {
            this._spline(value.spline);
        } else if (value.mode === CurveRange.Mode.TwoCurves) {
            this._spline(value.splineMin);
            this._spline(value.splineMax);
        }
    }

    private _spline (spline: RealCurve): void {
        if (!(spline instanceof RealCurve) || spline.evaluate !== evaluateSpline) {
            this._opaque = true;
            return;
        }

        this._write(identity(spline), ValueTag.Identity);
        this.value(spline.preExtrapolation);
        this.value(spline.postExtrapolation);
        this.value(spline.keyFramesCount);
        for (let i = 0; i < spline.keyFramesCount; ++i) {
            const key = spline.getKeyframeValue(i);
            keyframeValues[0] = spline.getKeyframeTime(i);
            keyframeValues[1] = key.value;
            keyframeValues[2] = key.interpolationMode;
            keyframeValues[3] = key.leftTangent;
            keyframeValues[4] = key.rightTangent;
            keyframeValues[5] = key.leftTangentWeight;
            keyframeValues[6] = key.rightTangentWeight;
            keyframeValues[7] = key.tangentWeightMode;
            keyframeValues[8] = key.easingMethod;
            for (let j = 0; j < keyframeValues.length; ++j) {
                this._write(keyframeValues[j], ValueTag.Number);
            }
        }
    }

    protected color (range: ComputeColorRange | undefined | null): void {
        if (!range) {
            this.value(null);
            return;
        }

        this.value(range.mode);
        this._rgba(range.color);
        this._rgba(range.colorMin);
        this._rgba(range.colorMax);
        this._gradient(range.gradient);
        this._gradient(range.gradientMin);
        this._gradient(range.gradientMax);
    }

    private _rgba (color: ComputeColor): void {
        this.value(color.r);
        this.value(color.g);
        this.value(color.b);
        this.value(color.a);
    }

    private _gradient (gradient: ComputeGradient): void {
        this.value(gradient.mode);
        this.value(gradient.colorKeys.length);
        this.value(gradient.alphaKeys.length);
        for (let i = 0; i < gradient.colorKeys.length; ++i) {
            const key = gradient.colorKeys[i];
            this.value(key.time);
            this._rgba(key.color);
        }
        for (let i = 0; i < gradient.alphaKeys.length; ++i) {
            const key = gradient.alphaKeys[i];
            this.value(key.time);
            this.value(key.alpha);
        }
    }
}

/** @engineInternal */
export class ParticleComputeModuleConfigTracker extends ConfigSnapshot {
    public constructor () {
        super(128);
    }

    private _enabled<T extends { enable: boolean }> (module: T | null | undefined): module is T {
        const enabled = !!module?.enable;
        this.value(enabled);
        return enabled;
    }

    public check (source: ParticleComputeModuleSource, options: ParticleComputeModuleOptions = defaultOptions): boolean {
        this.begin();
        this.value(options.samples);
        this.value(options.renderMode);
        this.value(options.maxSampledScalarError);
        this.value(options.estimateError);
        this.value(source.startSize3D);
        this.value(source.startRotation3D);
        this.scalar(source.gravityModifier);
        this.scalar(source.startSizeX);
        this.scalar(source.startSizeY);
        this.scalar(source.startSizeZ);
        this.scalar(source.startRotationX);
        this.scalar(source.startRotationY);
        this.scalar(source.startRotationZ);
        this.scalar(source.startSpeed);
        this.scalar(source.startLifetime);
        this.color(source.startColor);

        const size = source.sizeOvertimeModule;
        if (this._enabled(size)) {
            this.value(size.separateAxes);
            this.scalar(size.size);
            this.scalar(size.x);
            this.scalar(size.y);
            this.scalar(size.z);
        }

        const color = source.colorOverLifetimeModule;
        if (this._enabled(color)) {
            this.color(color.color);
        }

        const force = source.forceOvertimeModule;
        if (this._enabled(force)) {
            this.value(force.space);
            this.scalar(force.x);
            this.scalar(force.y);
            this.scalar(force.z);
        }

        const velocity = source.velocityOvertimeModule;
        if (this._enabled(velocity)) {
            this.value(velocity.space);
            this.scalar(velocity.x);
            this.scalar(velocity.y);
            this.scalar(velocity.z);
            this.scalar(velocity.speedModifier);
        }

        const limit = source.limitVelocityOvertimeModule;
        if (this._enabled(limit)) {
            this.value(limit.space);
            this.value(limit.separateAxes);
            this.value(limit.dampen);
            this.scalar(limit.limit);
            this.scalar(limit.limitX);
            this.scalar(limit.limitY);
            this.scalar(limit.limitZ);
        }

        const rotation = source.rotationOvertimeModule;
        if (this._enabled(rotation)) {
            this.value(rotation.separateAxes);
            this.scalar(rotation.x);
            this.scalar(rotation.y);
            this.scalar(rotation.z);
        }

        const texture = source.textureAnimationModule;
        if (this._enabled(texture)) {
            this.value(texture.numTilesX);
            this.value(texture.numTilesY);
            this.value(texture.animation);
            this.value(texture.randomRow);
            this.value(texture.rowIndex);
            this.value(texture.cycleCount);
            this.scalar(texture.startFrame);
            this.scalar(texture.frameOverTime);
        }

        const noise = source.noiseModule;
        if (this._enabled(noise)) {
            this.value(noise.strengthX);
            this.value(noise.strengthY);
            this.value(noise.strengthZ);
            this.value(noise.noiseSpeedX);
            this.value(noise.noiseSpeedY);
            this.value(noise.noiseSpeedZ);
            this.value(noise.noiseFrequency);
            this.value(noise.octaves);
            this.value(noise.octaveMultiplier);
            this.value(noise.octaveScale);
            this.value(noise.remapX);
            this.value(noise.remapY);
            this.value(noise.remapZ);
        }

        this._rotation(options.forceRotation);
        this._rotation(options.velocityRotation);
        this._rotation(options.limitRotation);
        return this.end();
    }

    private _rotation (rotation: ParticleComputeModuleOptions['forceRotation']): void {
        if (!rotation) {
            this.value(null);
            return;
        }

        for (let i = 0; i < rotation.length; ++i) {
            this.value(rotation[i]);
        }
    }
}

/** @engineInternal */
export class ParticleComputeTrailConfigTracker extends ConfigSnapshot {
    public constructor () {
        super(64);
    }

    public check (source: ParticleComputeTrailSource, samples = DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES): boolean {
        this.begin();
        this.value(samples);
        this.value(source.minParticleDistance);
        this.value(source.widthFromParticle);
        this.value(source.colorFromParticle);
        this.value(source.space);
        // The compiled trail retains the lifetime evaluator.
        this.scalar(source.lifeTime);
        this.scalar(source.widthRatio);
        this.color(source.colorOvertime);
        this.color(source.colorOverTrail);
        return this.end();
    }
}
