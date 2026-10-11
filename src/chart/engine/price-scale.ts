// Vertical axis. The visible range is kept in *logical* units — log10(price)
// on a log scale — so zooming and panning are linear in screen space for
// both modes (equal pixels = equal % moves on log). The range maps onto the
// pane height minus the top/bottom margins.

import { clamp, type PriceScaleMode } from "./types";

export type PriceTick = { price: number; y: number; label: string };

const MIN_PRICE = 1e-12;

export class PriceScale {
  mode: PriceScaleMode;
  autoScale = true;
  margins = { top: 0.1, bottom: 0.1 };
  /** Pane height in CSS px. */
  height = 0;
  /** Visible logical range. */
  min = 0;
  max = 1;
  /** Decimals used for labels when nothing better is known. */
  precision = 2;

  constructor(mode: PriceScaleMode = "log") {
    this.mode = mode;
  }

  // ---- logical conversion ----

  toLogical(price: number): number {
    return this.mode === "log" ? Math.log10(Math.max(price, MIN_PRICE)) : price;
  }

  fromLogical(value: number): number {
    return this.mode === "log" ? Math.pow(10, value) : value;
  }

  /** Height the range is drawn into (pane minus margins). */
  get drawHeight(): number {
    return Math.max(1, this.height * (1 - this.margins.top - this.margins.bottom));
  }

  private get top(): number {
    return this.height * this.margins.top;
  }

  logicalToY(value: number): number {
    const span = this.max - this.min || 1;
    return this.top + ((this.max - value) / span) * this.drawHeight;
  }

  yToLogical(y: number): number {
    const span = this.max - this.min || 1;
    return this.max - ((y - this.top) / this.drawHeight) * span;
  }

  priceToY(price: number): number {
    return this.logicalToY(this.toLogical(price));
  }

  yToPrice(y: number): number {
    return this.fromLogical(this.yToLogical(y));
  }

  // ---- range ----

  /** Fit a price range (auto-scale). */
  setAutoRange(minPrice: number, maxPrice: number): void {
    if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice)) return;
    let lo = this.toLogical(minPrice);
    let hi = this.toLogical(maxPrice);
    if (hi - lo < 1e-9) {
      const pad = this.mode === "log" ? 0.01 : Math.max(Math.abs(hi) * 0.01, 0.01);
      lo -= pad;
      hi += pad;
    }
    this.min = lo;
    this.max = hi;
  }

  /** Switching modes keeps the visible *prices*. */
  setMode(mode: PriceScaleMode): void {
    if (mode === this.mode) return;
    const lo = this.fromLogical(this.min);
    const hi = this.fromLogical(this.max);
    this.mode = mode;
    if (mode === "log" && lo <= 0) {
      this.autoScale = true;
      return;
    }
    this.min = this.toLogical(lo);
    this.max = this.toLogical(hi);
  }

  snapshot(): { min: number; max: number } {
    return { min: this.min, max: this.max };
  }

  /**
   * Dragging the price axis (TradingView / lightweight-charts curve): measured
   * from the bottom, scale the snapshot around its centre by
   * (start + 0.2h) / (current + 0.2h). Drag up = stretch, down = compress.
   */
  scaleFromDrag(snapshot: { min: number; max: number }, startY: number, y: number): void {
    const h = Math.max(1, this.height);
    const ease = (h - 1) * 0.2;
    const startFromBottom = h - startY;
    const nowFromBottom = Math.max(0, h - y);
    const coeff = Math.max(0.1, (startFromBottom + ease) / (nowFromBottom + ease));
    const mid = (snapshot.min + snapshot.max) / 2;
    const half = ((snapshot.max - snapshot.min) * coeff) / 2;
    this.autoScale = false;
    this.min = mid - half;
    this.max = mid + half;
  }

  /** Pan 1:1 with the cursor (logical units per pixel of the drawn height). */
  scrollFrom(snapshot: { min: number; max: number }, dy: number): void {
    const shift = (dy / this.drawHeight) * (snapshot.max - snapshot.min);
    this.min = snapshot.min + shift;
    this.max = snapshot.max + shift;
  }

  // ---- ticks ----

  /** Price labels about every `spacing` px; evenly spaced on log scales too. */
  ticks(spacing = 42, format = (p: number) => this.format(p)): PriceTick[] {
    if (this.height <= 0 || !(this.max > this.min)) return [];
    const topLogical = this.yToLogical(0);
    const bottomLogical = this.yToLogical(this.height);
    const count = Math.max(2, Math.floor(this.height / spacing));
    return this.mode === "log"
      ? this.logTicks(bottomLogical, topLogical, count, format)
      : this.linearTicks(bottomLogical, topLogical, count, format);
  }

  private linearTicks(
    lo: number,
    hi: number,
    count: number,
    format: (p: number) => string,
  ): PriceTick[] {
    const step = niceStep((hi - lo) / count);
    const out: PriceTick[] = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      const price = roundTo(v, step);
      out.push({ price, y: this.priceToY(price), label: format(price) });
    }
    this.precision = decimalsFor(step);
    return out;
  }

  /**
   * Equal steps in log space anchored to absolute multiples (stable while
   * panning), each price rounded to a nice value for its local gap.
   */
  private logTicks(
    lo: number,
    hi: number,
    count: number,
    format: (p: number) => string,
  ): PriceTick[] {
    const step = (hi - lo) / count;
    const out: PriceTick[] = [];
    const minGap = (this.drawHeight / ((this.max - this.min) / step)) * 0.5;
    let prevY: number | null = null;
    let prevPrice: number | null = null;
    for (let k = Math.floor(lo / step); k * step <= hi + step; k++) {
      const raw = Math.pow(10, k * step);
      const gap = raw * (Math.pow(10, step) - 1);
      const unit = niceStep(gap / 4);
      const price = roundTo(Math.round(raw / unit) * unit, unit);
      const logical = this.toLogical(price);
      if (!(price > 0) || logical < lo || logical > hi || price === prevPrice) continue;
      const y = this.priceToY(price);
      if (prevY !== null && Math.abs(y - prevY) < minGap) continue;
      out.push({ price, y, label: format(price) });
      prevY = y;
      prevPrice = price;
      this.precision = Math.max(0, decimalsFor(unit));
    }
    return out;
  }

  /**
   * Same number of decimals for every label on the axis (TradingView shows
   * the instrument's precision, no thousands separators): 2 for prices >= 1,
   * otherwise enough for about four significant digits of the lowest price.
   */
  format(price: number): string {
    const lowest = Math.max(MIN_PRICE, Math.min(Math.abs(this.fromLogical(this.min)), Math.abs(this.fromLogical(this.max))));
    const decimals = lowest >= 1 ? 2 : clamp(Math.ceil(-Math.log10(lowest)) + 3, 2, 10);
    return price.toFixed(decimals);
  }
}

export function niceStep(raw: number): number {
  if (!Number.isFinite(raw) || raw <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / magnitude;
  const nice = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  return nice * magnitude;
}

function decimalsFor(step: number): number {
  return Math.max(0, -Math.floor(Math.log10(step)) + (step / Math.pow(10, Math.floor(Math.log10(step))) === 2.5 ? 1 : 0));
}

function roundTo(value: number, step: number): number {
  const d = decimalsFor(step);
  return Number(value.toFixed(Math.min(12, d + 2)));
}
