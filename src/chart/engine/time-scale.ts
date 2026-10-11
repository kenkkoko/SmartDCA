// Horizontal axis: bars are laid out by index (gaps like weekends disappear),
// with a float coordinate model so nothing is rounded until it is painted.
//
//   x(i) = width - (last + rightOffset - i + 0.5) * barSpacing
//
// `rightOffset` is measured in bars (empty space after the last bar), so it
// keeps its meaning when the bar spacing changes.

import { clamp, type Time } from "./types";

export type TimeScaleOptions = {
  barSpacing: number;
  minBarSpacing: number;
  maxBarSpacing: number;
  rightOffset: number;
};

export const DEFAULT_TIME_SCALE: TimeScaleOptions = {
  barSpacing: 8,
  minBarSpacing: 0.5,
  maxBarSpacing: 60,
  rightOffset: 4,
};

export type TimeMark = { index: number; x: number; label: string; weight: number };

export class TimeScale {
  width = 0;
  barSpacing: number;
  rightOffset: number;
  readonly options: TimeScaleOptions;
  private times: Time[] = [];

  constructor(options: Partial<TimeScaleOptions> = {}) {
    this.options = { ...DEFAULT_TIME_SCALE, ...options };
    this.barSpacing = this.options.barSpacing;
    this.rightOffset = this.options.rightOffset;
  }

  setTimes(times: Time[]): void {
    this.times = times;
    this.clampOffset();
  }

  get count(): number {
    return this.times.length;
  }

  get lastIndex(): number {
    return this.times.length - 1;
  }

  indexToX(index: number): number {
    return (
      this.width - (this.lastIndex + this.rightOffset - index + 0.5) * this.barSpacing
    );
  }

  /** Float bar index under x (not rounded). */
  xToIndex(x: number): number {
    return (
      this.lastIndex + this.rightOffset + 0.5 - (this.width - x) / this.barSpacing
    );
  }

  /** Nearest bar index under x. */
  xToNearestIndex(x: number): number {
    return Math.round(this.xToIndex(x));
  }

  /** Bars that intersect the plot (clamped to data). */
  visibleRange(): { from: number; to: number } {
    if (this.count === 0) return { from: 0, to: -1 };
    const from = Math.max(0, Math.floor(this.xToIndex(0)) - 1);
    const to = Math.min(this.lastIndex, Math.ceil(this.xToIndex(this.width)) + 1);
    return { from, to };
  }

  // ---- time <-> index (supports fractional / future positions) ----

  /** Average bar interval near the end, used to extrapolate into the future. */
  private interval(): number {
    const n = this.times.length;
    if (n < 2) return 86400;
    const span = Math.min(10, n - 1);
    return (this.times[n - 1]! - this.times[n - 1 - span]!) / span;
  }

  /** Continuous index for a time: interpolates between bars, extrapolates outside. */
  timeToIndex(time: Time): number {
    const t = this.times;
    const n = t.length;
    if (n === 0) return 0;
    if (time <= t[0]!) {
      const step = n > 1 ? t[1]! - t[0]! : this.interval();
      return (time - t[0]!) / step;
    }
    if (time >= t[n - 1]!) return n - 1 + (time - t[n - 1]!) / this.interval();
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (t[mid]! <= time) lo = mid;
      else hi = mid;
    }
    return lo + (time - t[lo]!) / (t[hi]! - t[lo]!);
  }

  indexToTime(index: number): Time {
    const t = this.times;
    const n = t.length;
    if (n === 0) return 0;
    if (index <= 0) {
      const step = n > 1 ? t[1]! - t[0]! : this.interval();
      return t[0]! + index * step;
    }
    if (index >= n - 1) return t[n - 1]! + (index - (n - 1)) * this.interval();
    const lo = Math.floor(index);
    return t[lo]! + (index - lo) * (t[lo + 1]! - t[lo]!);
  }

  timeAt(index: number): Time | undefined {
    return this.times[index];
  }

  // ---- interactions ----

  /**
   * Zoom keeping the bar under `x` fixed. `scale` is in [-1, 1]; one mouse
   * wheel notch is about ±1 and changes spacing by ±10% (TradingView feel).
   */
  zoom(x: number, scale: number): void {
    const anchor = this.xToIndex(x);
    this.setBarSpacing(this.barSpacing * (1 + scale / 10));
    this.rightOffset = anchor - this.lastIndex - 0.5 + (this.width - x) / this.barSpacing;
    this.clampOffset();
  }

  /** Bar spacing for dragging the time axis: follows the distance to the right edge. */
  scaleFromDrag(startX: number, x: number, startSpacing: number): void {
    const startLen = clamp(this.width - startX, 0, this.width);
    const nowLen = clamp(this.width - x, 0, this.width);
    if (startLen === 0 || nowLen === 0) return;
    // right offset (in bars) is unchanged, so the latest bar stays put
    this.setBarSpacing((startSpacing * nowLen) / startLen);
    this.clampOffset();
  }

  /** Pan by pixels relative to a gesture start. */
  scrollFrom(startOffset: number, dx: number): void {
    this.rightOffset = startOffset - dx / this.barSpacing;
    this.clampOffset();
  }

  setBarSpacing(spacing: number): void {
    this.barSpacing = clamp(spacing, this.options.minBarSpacing, this.options.maxBarSpacing);
  }

  reset(): void {
    this.barSpacing = this.options.barSpacing;
    this.rightOffset = this.options.rightOffset;
    this.clampOffset();
  }

  /** Show every bar (used for short histories). */
  fitContent(): void {
    if (this.count === 0 || this.width <= 0) return;
    this.setBarSpacing(this.width / (this.count + this.options.rightOffset));
    this.rightOffset = this.options.rightOffset;
    this.clampOffset();
  }

  /** Keep at least a few bars on screen in either direction. */
  clampOffset(): void {
    if (this.width <= 0 || this.count === 0) return;
    const keep = Math.min(2, this.count);
    const maxOffset = this.width / this.barSpacing - keep;
    const minOffset = -(this.lastIndex - keep + 1);
    this.rightOffset = clamp(this.rightOffset, minOffset, Math.max(minOffset, maxOffset));
  }

  // ---- labels ----

  /**
   * Time axis labels at "important" bars (year > month > day boundaries),
   * spaced at least `minSpacing` px apart, preferring heavier boundaries.
   */
  marks(minSpacing = 90): TimeMark[] {
    const { from, to } = this.visibleRange();
    if (to < from) return [];
    const candidates: TimeMark[] = [];
    let prev: Date | null = from > 0 ? toDate(this.times[from - 1]!) : null;
    for (let i = from; i <= to; i++) {
      const date = toDate(this.times[i]!);
      let weight = 1;
      let label = String(date.getDate());
      if (!prev || date.getFullYear() !== prev.getFullYear()) {
        weight = 4;
        label = String(date.getFullYear());
      } else if (date.getMonth() !== prev.getMonth()) {
        weight = 3;
        label = MONTHS[date.getMonth()]!;
      } else if (date.getDate() !== prev.getDate()) {
        weight = 2;
      }
      candidates.push({ index: i, x: this.indexToX(i), label, weight });
      prev = date;
    }
    // Greedy by weight, then left-to-right, rejecting marks that crowd others.
    const chosen: TimeMark[] = [];
    const order = [...candidates].sort((a, b) => b.weight - a.weight || a.index - b.index);
    for (const mark of order) {
      if (mark.x < 0 || mark.x > this.width) continue;
      if (chosen.every((other) => Math.abs(other.x - mark.x) >= minSpacing)) chosen.push(mark);
    }
    return chosen.sort((a, b) => a.index - b.index);
  }
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function toDate(time: Time): Date {
  return new Date(time * 1000);
}

export function formatTimeLabel(time: Time, intraday = false): string {
  const d = toDate(time);
  const pad = (n: number) => String(n).padStart(2, "0");
  const day = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  return intraday ? `${day} ${pad(d.getHours())}:${pad(d.getMinutes())}` : day;
}
