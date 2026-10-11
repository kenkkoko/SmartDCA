// A canvas whose backing store matches the device pixels exactly. All
// painting happens in *bitmap* space with an identity transform: callers pass
// CSS coordinates and the helpers snap edges to whole device pixels, so
// candles and 1px lines stay crisp at 125% / 150% display scaling.

import type { LineStyle } from "./types";

export class Surface {
  readonly canvas: HTMLCanvasElement;
  readonly ctx: CanvasRenderingContext2D;
  /** CSS size. */
  width = 0;
  height = 0;
  /** Device pixels per CSS pixel, per axis (can differ slightly after rounding). */
  hpr = 1;
  vpr = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2D canvas unavailable");
    this.ctx = ctx;
  }

  /** Resize to a CSS size; `bitmap` is the exact device-pixel size when known. */
  resize(width: number, height: number, bitmap?: { width: number; height: number }): boolean {
    const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
    const bw = bitmap?.width ?? Math.round(width * dpr);
    const bh = bitmap?.height ?? Math.round(height * dpr);
    const changed =
      this.canvas.width !== bw || this.canvas.height !== bh || this.width !== width || this.height !== height;
    this.width = width;
    this.height = height;
    this.canvas.width = Math.max(1, bw);
    this.canvas.height = Math.max(1, bh);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.hpr = width > 0 ? this.canvas.width / width : dpr;
    this.vpr = height > 0 ? this.canvas.height / height : dpr;
    return changed;
  }

  clear(): void {
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  // ---- coordinate snapping ----
  bx(x: number): number {
    return Math.round(x * this.hpr);
  }
  by(y: number): number {
    return Math.round(y * this.vpr);
  }
  /** Line width in device pixels for a CSS width (at least 1). */
  lw(width = 1): number {
    return Math.max(1, Math.floor(width * this.hpr));
  }

  // ---- primitives (CSS coordinates in, device pixels out) ----

  fillRect(x: number, y: number, w: number, h: number, color: string): void {
    const x0 = this.bx(x);
    const y0 = this.by(y);
    this.ctx.fillStyle = color;
    this.ctx.fillRect(x0, y0, Math.max(1, this.bx(x + w) - x0), Math.max(1, this.by(y + h) - y0));
  }

  /** Crisp horizontal line spanning [x1, x2] at y. */
  hLine(x1: number, x2: number, y: number, color: string, width = 1, style: LineStyle = "solid"): void {
    const w = this.lw(width);
    const by = this.by(y) - Math.floor(w / 2);
    const bx1 = this.bx(Math.min(x1, x2));
    const bx2 = this.bx(Math.max(x1, x2));
    this.ctx.fillStyle = color;
    if (style === "solid") {
      this.ctx.fillRect(bx1, by, Math.max(1, bx2 - bx1), w);
      return;
    }
    const [on, off] = dash(style, this.hpr);
    for (let x = bx1; x < bx2; x += on + off) this.ctx.fillRect(x, by, Math.min(on, bx2 - x), w);
  }

  /** Crisp vertical line spanning [y1, y2] at x. */
  vLine(x: number, y1: number, y2: number, color: string, width = 1, style: LineStyle = "solid"): void {
    const w = this.lw(width);
    const bx = this.bx(x) - Math.floor(w / 2);
    const by1 = this.by(Math.min(y1, y2));
    const by2 = this.by(Math.max(y1, y2));
    this.ctx.fillStyle = color;
    if (style === "solid") {
      this.ctx.fillRect(bx, by1, w, Math.max(1, by2 - by1));
      return;
    }
    const [on, off] = dash(style, this.vpr);
    for (let y = by1; y < by2; y += on + off) this.ctx.fillRect(bx, y, w, Math.min(on, by2 - y));
  }

  /** Anti-aliased polyline / segment (for sloped lines). */
  strokePath(points: { x: number; y: number }[], color: string, width = 1, style: LineStyle = "solid"): void {
    if (points.length < 2) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width * this.hpr;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash(style === "solid" ? [] : dash(style, this.hpr));
    ctx.beginPath();
    ctx.moveTo(points[0]!.x * this.hpr, points[0]!.y * this.vpr);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i]!.x * this.hpr, points[i]!.y * this.vpr);
    ctx.stroke();
    ctx.restore();
  }

  fillPolygon(points: { x: number; y: number }[], color: string): void {
    if (points.length < 3) return;
    const ctx = this.ctx;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(points[0]!.x * this.hpr, points[0]!.y * this.vpr);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i]!.x * this.hpr, points[i]!.y * this.vpr);
    ctx.closePath();
    ctx.fill();
  }

  text(
    value: string,
    x: number,
    y: number,
    color: string,
    font: { size: number; family: string; weight?: string },
    align: CanvasTextAlign = "left",
    baseline: CanvasTextBaseline = "middle",
  ): void {
    const ctx = this.ctx;
    ctx.fillStyle = color;
    ctx.font = `${font.weight ?? "normal"} ${Math.round(font.size * this.vpr)}px ${font.family}`;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(value, Math.round(x * this.hpr), Math.round(y * this.vpr));
  }

  measure(value: string, font: { size: number; family: string; weight?: string }): number {
    this.ctx.font = `${font.weight ?? "normal"} ${font.size}px ${font.family}`;
    return this.ctx.measureText(value).width;
  }

  /** Clip to a CSS rect until `restore()`. */
  clip(rect: { x: number; y: number; width: number; height: number }): void {
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.rect(this.bx(rect.x), this.by(rect.y), this.bx(rect.x + rect.width) - this.bx(rect.x), this.by(rect.y + rect.height) - this.by(rect.y));
    this.ctx.clip();
  }

  restore(): void {
    this.ctx.restore();
  }
}

function dash(style: LineStyle, ratio: number): [number, number] {
  return style === "dashed"
    ? [Math.round(6 * ratio), Math.round(4 * ratio)]
    : [Math.max(1, Math.round(1.5 * ratio)), Math.round(2.5 * ratio)];
}
