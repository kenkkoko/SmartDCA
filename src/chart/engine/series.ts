// Series hold data keyed by time; the chart maps them onto the main bar
// index so every series lines up with the candles.

import type { Surface } from "./surface";
import type { PriceScale } from "./price-scale";
import type { TimeScale } from "./time-scale";
import type { Bar, LineStyle, Theme, ValuePoint } from "./types";

export type MarkerShape = "arrowUp" | "arrowDown" | "triangleUp" | "triangleDown" | "circle";

export type Marker = {
  time: number;
  position: "aboveBar" | "belowBar" | "atValue";
  shape: MarkerShape;
  color: string;
  text?: string;
  /** Price for `atValue` markers. */
  value?: number;
  size?: number;
};

export type PriceLine = {
  price: number;
  color: string;
  lineStyle?: LineStyle;
  lineWidth?: number;
  title?: string;
  axisLabel?: boolean;
};

export type SeriesKind = "candles" | "line" | "histogram";

export type CommonSeriesOptions = {
  visible: boolean;
  /** Contributes to the pane's auto-scale. */
  autoscale: boolean;
  /** Show the last value on the price axis. */
  lastValueLabel: boolean;
  title?: string;
  /** "overlay": own hidden scale at the bottom of the pane (volume). */
  scale?: "main" | "overlay";
};

export type CandleOptions = CommonSeriesOptions & {
  upColor: string;
  downColor: string;
  wickUpColor: string;
  wickDownColor: string;
};

export type LineOptions = CommonSeriesOptions & {
  color: string;
  lineWidth: number;
  lineStyle: LineStyle;
};

export type HistogramOptions = CommonSeriesOptions & {
  color: string;
  /** Baseline value the bars grow from. */
  base: number;
};

type DrawEnv = {
  surface: Surface;
  time: TimeScale;
  price: PriceScale;
  /** Pane top in CSS px within the surface. */
  top: number;
  theme: Theme;
};

let seriesSeq = 0;

export abstract class Series<D extends { time: number }, O extends CommonSeriesOptions> {
  readonly id = `s${++seriesSeq}`;
  abstract readonly kind: SeriesKind;
  options: O;
  data: D[] = [];
  /** Main-index position of each data point. */
  indices: number[] = [];
  markers: Marker[] = [];
  priceLines: PriceLine[] = [];
  onChange: () => void = () => {};

  constructor(options: O) {
    this.options = options;
  }

  setData(data: D[]): void {
    this.data = data;
    this.onChange();
  }

  applyOptions(options: Partial<O>): void {
    this.options = { ...this.options, ...options };
    this.onChange();
  }

  setMarkers(markers: Marker[]): void {
    this.markers = [...markers].sort((a, b) => a.time - b.time);
    this.onChange();
  }

  setPriceLines(lines: PriceLine[]): void {
    this.priceLines = lines;
    this.onChange();
  }

  reindex(time: TimeScale): void {
    this.indices = this.data.map((point) => time.timeToIndex(point.time));
  }

  /** First/last data positions intersecting [from, to] (main indices). */
  protected window(from: number, to: number, margin = 1): [number, number] {
    const idx = this.indices;
    let lo = 0;
    let hi = idx.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (idx[mid]! < from - margin) lo = mid + 1;
      else hi = mid;
    }
    let end = lo;
    while (end < idx.length && idx[end]! <= to + margin) end++;
    return [lo, end];
  }

  abstract range(from: number, to: number): { min: number; max: number } | null;
  abstract draw(env: DrawEnv, from: number, to: number): void;
  /** Value shown in the legend / last-value label at main index `index`. */
  abstract valueAt(index: number): number | undefined;
  abstract lastValue(): { value: number; color: string } | undefined;

  /** Data position whose main index is `index` (±0.5), or -1. */
  pointIndexAt(index: number): number {
    const idx = this.indices;
    let lo = 0;
    let hi = idx.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (idx[mid]! < index - 0.5) lo = mid + 1;
      else hi = mid;
    }
    return lo < idx.length && Math.abs(idx[lo]! - index) < 0.5 ? lo : -1;
  }
}

// ---------------------------------------------------------------------------

export class CandleSeries extends Series<Bar, CandleOptions> {
  readonly kind = "candles" as const;

  range(from: number, to: number) {
    const [a, b] = this.window(from, to, 0);
    let min = Infinity;
    let max = -Infinity;
    for (let i = a; i < b; i++) {
      const bar = this.data[i]!;
      if (bar.low < min) min = bar.low;
      if (bar.high > max) max = bar.high;
    }
    return min <= max ? { min, max } : null;
  }

  draw({ surface, time, price, top }: DrawEnv, from: number, to: number): void {
    const [a, b] = this.window(from, to);
    const s = surface;
    const o = this.options;
    // body width in device px, odd/even matched to the wick so it centres
    const spacingPx = time.barSpacing * s.hpr;
    let bodyW = Math.max(1, Math.floor(spacingPx * (time.barSpacing < 4 ? 1 : 0.8)));
    const wickW = Math.max(1, Math.floor(s.hpr));
    if (bodyW > 2 && bodyW % 2 !== wickW % 2) bodyW--;
    const half = Math.floor(bodyW / 2);
    const wickHalf = Math.floor(Math.min(wickW, bodyW) / 2);
    const ctx = s.ctx;
    for (let i = a; i < b; i++) {
      const bar = this.data[i]!;
      const up = bar.close >= bar.open;
      const cx = s.bx(time.indexToX(this.indices[i]!));
      const yo = s.by(top + price.priceToY(bar.open));
      const yc = s.by(top + price.priceToY(bar.close));
      const yh = s.by(top + price.priceToY(bar.high));
      const yl = s.by(top + price.priceToY(bar.low));
      const bodyTop = Math.min(yo, yc);
      const bodyBottom = Math.max(yo, yc);
      ctx.fillStyle = up ? o.wickUpColor : o.wickDownColor;
      ctx.fillRect(cx - wickHalf, yh, Math.min(wickW, bodyW), Math.max(1, yl - yh + 1));
      ctx.fillStyle = up ? o.upColor : o.downColor;
      ctx.fillRect(cx - half, bodyTop, bodyW, Math.max(1, bodyBottom - bodyTop + 1));
    }
  }

  valueAt(index: number) {
    const i = this.pointIndexAt(index);
    return i >= 0 ? this.data[i]!.close : undefined;
  }

  barAt(index: number): Bar | undefined {
    const i = this.pointIndexAt(index);
    return i >= 0 ? this.data[i] : undefined;
  }

  lastValue() {
    const bar = this.data[this.data.length - 1];
    if (!bar) return undefined;
    return { value: bar.close, color: bar.close >= bar.open ? this.options.upColor : this.options.downColor };
  }
}

// ---------------------------------------------------------------------------

export class LineSeries extends Series<ValuePoint, LineOptions> {
  readonly kind = "line" as const;

  range(from: number, to: number) {
    const [a, b] = this.window(from, to, 0);
    let min = Infinity;
    let max = -Infinity;
    for (let i = a; i < b; i++) {
      const v = this.data[i]!.value;
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return min <= max ? { min, max } : null;
  }

  draw({ surface, time, price, top }: DrawEnv, from: number, to: number): void {
    const [a, b] = this.window(from, to);
    const o = this.options;
    // group consecutive points by color so per-segment colors work
    let run: { x: number; y: number }[] = [];
    let runColor = o.color;
    const flush = () => {
      if (run.length > 1) surface.strokePath(run, runColor, o.lineWidth, o.lineStyle);
    };
    for (let i = a; i < b; i++) {
      const p = this.data[i]!;
      if (!Number.isFinite(p.value)) {
        flush();
        run = [];
        continue;
      }
      const point = { x: time.indexToX(this.indices[i]!), y: top + price.priceToY(p.value) };
      const color = p.color ?? o.color;
      if (run.length && color !== runColor) {
        // the segment ending at this point takes this point's color
        const last = run[run.length - 1]!;
        flush();
        run = [last, point];
        runColor = color;
        continue;
      }
      if (!run.length) runColor = color;
      run.push(point);
    }
    flush();
  }

  valueAt(index: number) {
    const i = this.pointIndexAt(index);
    return i >= 0 ? this.data[i]!.value : undefined;
  }

  lastValue() {
    for (let i = this.data.length - 1; i >= 0; i--) {
      const p = this.data[i]!;
      if (Number.isFinite(p.value)) return { value: p.value, color: p.color ?? this.options.color };
    }
    return undefined;
  }
}

// ---------------------------------------------------------------------------

export class HistogramSeries extends Series<ValuePoint, HistogramOptions> {
  readonly kind = "histogram" as const;

  range(from: number, to: number) {
    const [a, b] = this.window(from, to, 0);
    let min = this.options.base;
    let max = this.options.base;
    for (let i = a; i < b; i++) {
      const v = this.data[i]!.value;
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { min, max };
  }

  draw({ surface: s, time, price, top }: DrawEnv, from: number, to: number): void {
    const [a, b] = this.window(from, to);
    const spacingPx = time.barSpacing * s.hpr;
    let w = Math.max(1, Math.floor(spacingPx * (time.barSpacing < 4 ? 1 : 0.8)));
    if (w > 2 && w % 2 !== Math.max(1, Math.floor(s.hpr)) % 2) w--;
    const half = Math.floor(w / 2);
    const base = s.by(top + price.priceToY(this.options.base));
    for (let i = a; i < b; i++) {
      const p = this.data[i]!;
      if (!Number.isFinite(p.value)) continue;
      const cx = s.bx(time.indexToX(this.indices[i]!));
      const y = s.by(top + price.priceToY(p.value));
      s.ctx.fillStyle = p.color ?? this.options.color;
      s.ctx.fillRect(cx - half, Math.min(y, base), w, Math.max(1, Math.abs(base - y)));
    }
  }

  valueAt(index: number) {
    const i = this.pointIndexAt(index);
    return i >= 0 ? this.data[i]!.value : undefined;
  }

  lastValue() {
    const p = this.data[this.data.length - 1];
    return p ? { value: p.value, color: p.color ?? this.options.color } : undefined;
  }
}

export type AnySeries = CandleSeries | LineSeries | HistogramSeries;
