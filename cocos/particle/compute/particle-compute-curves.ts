/* Copyright (c) 2026 Xiamen Yaji Software Co., Ltd. */

import CurveRange from '../animator/curve-range';
import GradientRange from '../animator/gradient-range';
import { RealCurve } from '../../core/curves/curve';

export const DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES = 256;
export const PARTICLE_COMPUTE_MODULE_HEADER_FLOATS = 128;
export const PACKED_SCALAR_CURVE_COUNT_OFFSET = 2;

const evaluateRange = CurveRange.prototype.evaluate;
const evaluateSpline = RealCurve.prototype.evaluate;

function isStandardSpline (curve: RealCurve): boolean {
    return curve instanceof RealCurve && curve.evaluate === evaluateSpline;
}

export interface ComputeScalarCurve {
    readonly mode: number;
    evaluate(time: number, randomRatio: number): number;
}

export interface ComputeColor {
    readonly r: number;
    readonly g: number;
    readonly b: number;
    readonly a: number;
}

export interface ComputeGradient {
    readonly mode: number;
    readonly colorKeys: readonly { readonly time: number; readonly color: ComputeColor }[];
    readonly alphaKeys: readonly { readonly time: number; readonly alpha: number }[];
}

export interface ComputeColorRange {
    readonly mode: number;
    readonly color: ComputeColor;
    readonly colorMin: ComputeColor;
    readonly colorMax: ComputeColor;
    readonly gradient: ComputeGradient;
    readonly gradientMin: ComputeGradient;
    readonly gradientMax: ComputeGradient;
}

export type ComputeScalar = ComputeScalarCurve | number;

function finite (value: number): number {
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
        throw new RangeError('Particle module data must be finite f32');
    }
    return value;
}

/** Build-time parameter compilation, never a per-particle simulation or upload. */
export class ParticleComputeCurvePacker {
    public readonly values: number[];
    public readonly samples: number;
    /** Custom-evaluator LUT error only; zero does not imply exact f32 arithmetic for direct curves. */
    public maxSampledScalarError = 0;

    public constructor (headerFloats = PARTICLE_COMPUTE_MODULE_HEADER_FLOATS, samples = DEFAULT_PARTICLE_COMPUTE_CURVE_SAMPLES) {
        if (!Number.isInteger(samples) || samples < 2 || samples > 4096) {
            throw new RangeError('Curve samples must be in [2, 4096]');
        }
        this.values = new Array(headerFloats).fill(0);
        this.samples = samples;
    }

    public append (values: readonly number[]): number {
        const offset = this.values.length;
        for (const value of values) {
            this.values.push(finite(value));
        }
        return offset;
    }

    public scalar (curve: ComputeScalar | null | undefined, fallback = 0): number {
        const Mode = CurveRange.Mode;
        const source = curve ?? fallback;
        const mode = typeof source === 'number' ? Mode.Constant : source.mode;

        if (source instanceof CurveRange && source.evaluate === evaluateRange
            && ((mode === Mode.Curve && isStandardSpline(source.spline))
                || (mode === Mode.TwoCurves && isStandardSpline(source.splineMin) && isStandardSpline(source.splineMax)))) {
            const low = this.spline(mode === Mode.Curve ? source.spline : source.splineMin);
            const high = mode === Mode.Curve ? low : this.spline(source.splineMax);
            // Negative count distinguishes keyframes from the legacy sampled evaluator ABI.
            return this.append([low, high, -1, source.multiplier]);
        }

        const evaluate = (t: number, random: number): number => finite(
            typeof source === 'number' ? source : source.evaluate(t, random),
        );
        // Opaque JS evaluators have no transferable keyframe representation.
        const count = mode === Mode.Constant || mode === Mode.TwoConstants ? 1 : this.samples;
        const low: number[] = [];
        const high: number[] = [];
        for (let i = 0; i < count; ++i) {
            const time = count === 1 ? 0 : i / (count - 1);
            low.push(evaluate(time, 0));
            high.push(evaluate(time, 1));
        }

        // Estimate LUT interpolation error between samples at random ratios 0 and 1.
        if (count > 1) {
            for (let i = 0; i < count - 1; ++i) {
                for (const fraction of [0.25, 0.5, 0.75]) {
                    const time = (i + fraction) / (count - 1);
                    this.maxSampledScalarError = Math.max(
                        this.maxSampledScalarError,
                        Math.abs(evaluate(time, 0) - (low[i] + (low[i + 1] - low[i]) * fraction)),
                        Math.abs(evaluate(time, 1) - (high[i] + (high[i + 1] - high[i]) * fraction)),
                    );
                }
            }
        }

        const minOffset = this.append(low);
        const maxOffset = mode === Mode.TwoCurves || mode === Mode.TwoConstants ? this.append(high) : minOffset;
        return this.append([minOffset, maxOffset, count, mode]);
    }

    private spline (curve: RealCurve): number {
        const keys: number[] = [];
        let previous = -Infinity;
        for (let i = 0; i < curve.keyFramesCount; ++i) {
            const time = Math.fround(finite(curve.getKeyframeTime(i)));
            if (time <= previous) {
                throw new RangeError('Particle curve key times must remain strictly increasing in f32');
            }
            previous = time;
            const key = curve.getKeyframeValue(i);
            keys.push(
                time,
                key.value,
                key.interpolationMode,
                key.easingMethod,
                key.leftTangent,
                key.rightTangent,
                key.leftTangentWeight,
                key.rightTangentWeight,
                key.tangentWeightMode,
            );
        }
        const offset = this.append(keys);
        return this.append([offset, curve.keyFramesCount, curve.preExtrapolation, curve.postExtrapolation]);
    }

    public gradient (gradient: ComputeGradient): number {
        const colors: number[] = [];
        const alphas: number[] = [];
        for (const key of gradient.colorKeys) {
            colors.push(key.time, key.color.r / 255, key.color.g / 255, key.color.b / 255);
        }
        for (const key of gradient.alphaKeys) {
            alphas.push(key.time, key.alpha / 255);
        }
        const colorsOffset = this.append(colors);
        const alphasOffset = this.append(alphas);
        return this.append([colorsOffset, gradient.colorKeys.length, alphasOffset, gradient.alphaKeys.length, gradient.mode]);
    }

    public color (range?: ComputeColorRange | null): number {
        const Mode = GradientRange.Mode;
        const white = [1, 1, 1, 1];
        if (!range) {
            return this.append([Mode.Color, this.append(white), 0, 0]);
        }

        const rgba = (color: ComputeColor): number[] => [color.r / 255, color.g / 255, color.b / 255, color.a / 255];
        let first = 0;
        let second = 0;
        if (range.mode === Mode.Color) {
            first = this.append(rgba(range.color));
        } else if (range.mode === Mode.TwoColors) {
            first = this.append(rgba(range.colorMin));
            second = this.append(rgba(range.colorMax));
        } else if (range.mode === Mode.TwoGradients) {
            first = this.gradient(range.gradientMin);
            second = this.gradient(range.gradientMax);
        } else {
            if (range.mode === Mode.RandomColor && (!range.gradient.colorKeys.length || !range.gradient.alphaKeys.length)) {
                throw new RangeError('RandomColor needs both color and alpha keys');
            }
            first = this.gradient(range.gradient);
        }
        return this.append([range.mode, first, second, 0]);
    }

    public finish (): Float32Array {
        return new Float32Array(this.values);
    }
}
