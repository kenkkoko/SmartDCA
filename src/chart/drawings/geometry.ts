// Pixel-space geometry shared by drawing tools.
import type { Point, Rect } from "../engine/types";

export function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Distance from p to segment ab. */
export function distToSegment(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return dist(p, a);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

export function distToPolyline(p: Point, points: Point[], closed = false): number {
  let best = Infinity;
  for (let i = 1; i < points.length; i++) best = Math.min(best, distToSegment(p, points[i - 1]!, points[i]!));
  if (closed && points.length > 2) best = Math.min(best, distToSegment(p, points[points.length - 1]!, points[0]!));
  return best;
}

export function insidePolygon(p: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i]!;
    const b = poly[j]!;
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/**
 * Segment a→b, optionally continued past a (left of the line's direction in
 * screen terms: towards smaller x) and/or past b to the plot edges.
 */
export function extendSegment(
  a: Point,
  b: Point,
  extendLeft: boolean,
  extendRight: boolean,
  rect: Rect,
): [Point, Point] {
  if (!extendLeft && !extendRight) return [a, b];
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0) {
    // vertical: "left/right" extend to the pane top/bottom
    const top = { x: a.x, y: rect.y };
    const bottom = { x: a.x, y: rect.y + rect.height };
    const [lo, hi] = a.y <= b.y ? [a, b] : [b, a];
    return [extendLeft ? top : lo, extendRight ? bottom : hi];
  }
  const yAt = (x: number) => a.y + ((x - a.x) * dy) / dx;
  const [left, right] = a.x <= b.x ? [a, b] : [b, a];
  const L = extendLeft ? { x: rect.x - 1, y: yAt(rect.x - 1) } : left;
  const R = extendRight ? { x: rect.x + rect.width + 1, y: yAt(rect.x + rect.width + 1) } : right;
  return [L, R];
}

/** Ray from a through b to the plot edge. */
export function ray(a: Point, b: Point, rect: Rect): [Point, Point] {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  if (dx === 0 && dy === 0) return [a, b];
  // walk far enough to leave any plot
  const scale = (rect.width + rect.height) * 4 / Math.max(1e-9, Math.hypot(dx, dy));
  return [a, { x: a.x + dx * scale, y: a.y + dy * scale }];
}

export function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

export function angleDeg(a: Point, b: Point): number {
  return (Math.atan2(-(b.y - a.y), b.x - a.x) * 180) / Math.PI;
}

/** Quadratic curve through `mid` (not a control point), sampled. */
export function curveThrough(a: Point, mid: Point, b: Point, steps = 32): Point[] {
  const c = { x: 2 * mid.x - (a.x + b.x) / 2, y: 2 * mid.y - (a.y + b.y) / 2 };
  const out: Point[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const u = 1 - t;
    out.push({ x: u * u * a.x + 2 * u * t * c.x + t * t * b.x, y: u * u * a.y + 2 * u * t * c.y + t * t * b.y });
  }
  return out;
}

export function withAlpha(color: string, alpha: number): string {
  const a = Math.max(0, Math.min(1, alpha));
  if (color.startsWith("#") && (color.length === 7 || color.length === 4)) {
    const hex = color.length === 4 ? color.slice(1).split("").map((c) => c + c).join("") : color.slice(1);
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return color;
}

/** Snap b so that a→b is horizontal, vertical or 45° (Shift while drawing). */
export function snapAngle(a: Point, b: Point): Point {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const angle = Math.atan2(dy, dx);
  const step = Math.PI / 4;
  const snapped = Math.round(angle / step) * step;
  const len = Math.hypot(dx, dy);
  return { x: a.x + Math.cos(snapped) * len, y: a.y + Math.sin(snapped) * len };
}

function luminance(color: string): number | null {
  if (!color.startsWith("#") || (color.length !== 7 && color.length !== 4)) return null;
  const hex = color.length === 4 ? color.slice(1).split("").map((c) => c + c).join("") : color.slice(1);
  const ch = [0, 2, 4].map((i) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * ch[0]! + 0.7152 * ch[1]! + 0.0722 * ch[2]!;
}

/**
 * A colour that stays visible on the background: TradingView-style black
 * levels would vanish on a dark chart, so near-invisible colours are swapped
 * for the opposite neutral. Stored settings are untouched.
 */
export function readable(color: string, background: string): string {
  const a = luminance(color);
  const b = luminance(background);
  if (a === null || b === null) return color;
  const contrast = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  if (contrast >= 1.6) return color;
  return b < 0.5 ? "#c9ced8" : "#363a45";
}

/** Black or white, whichever reads better on `background`. */
export function textOn(background: string): string {
  const l = luminance(background);
  return l !== null && l > 0.45 ? "#131722" : "#ffffff";
}
