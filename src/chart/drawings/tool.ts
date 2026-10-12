// The contract every drawing tool implements, plus painting helpers.
import type { Chart, Pane } from "../engine/chart";
import type { Surface } from "../engine/surface";
import type { Bar, LineStyle, Point, Rect, Theme } from "../engine/types";
import { withAlpha } from "./geometry";
import type { Anchor, Drawing, DrawingStyle } from "./types";

export type ToolGroup = "lines" | "channels" | "shapes" | "patterns" | "measure" | "volume" | "fibonacci" | "text" | "brush";

export type ToolContext = {
  s: Surface;
  chart: Chart;
  pane: Pane;
  /** Plot rect of the pane (CSS px). */
  rect: Rect;
  item: Drawing;
  /** Anchors projected to CSS px. */
  points: Point[];
  theme: Theme;
  selected: boolean;
  bars: Bar[];
  toX(time: number): number;
  toY(price: number): number;
  fromX(x: number): number;
  fromY(y: number): number;
  fmt(price: number): string;
};

export type ToolCaps = {
  fill?: boolean;
  text?: boolean;
  extend?: boolean;
  middleLine?: boolean;
  stats?: boolean;
  /** Line width/style controls (default true). */
  line?: boolean;
  dialog?: "fib" | "volume";
};

export type Tool = {
  type: string;
  label: string;
  group: ToolGroup;
  /** Clicks that create it; "stroke" = press-drag, "multi" = click until double-click. */
  create: number | "stroke" | "multi";
  /** Derive stored anchors from the placed ones (e.g. a position's stop). */
  complete?(placed: Anchor[]): Anchor[];
  defaults?: Partial<DrawingStyle>;
  /** Extra set-up for a brand-new drawing (e.g. the saved Fibonacci template). */
  init?(item: Drawing): void;
  caps?: ToolCaps;
  render(c: ToolContext): void;
  /** Distance from p to the drawing (px), or null when not hit. */
  hit(c: ToolContext, p: Point): number | null;
  /** Selection handles (default: the anchors). */
  handles?(c: ToolContext): Point[];
  /** New anchor points after dragging handle `index` to p (default: move that anchor). */
  dragHandle?(c: ToolContext, index: number, p: Point): Point[] | null;
  /** Prices / times tagged on the axes while selected (default: anchors). */
  axisTags?(c: ToolContext): { prices: number[]; times: number[] };
};

export const HIT = 6;

// ---- painting helpers (CSS coordinates) ----

export function stroke(c: ToolContext, points: Point[], color = c.item.style.color, width = c.item.style.width, style: LineStyle = c.item.style.lineStyle): void {
  if (points.length < 2) return;
  // exact horizontals / verticals are drawn crisp
  if (points.length === 2) {
    const [a, b] = points as [Point, Point];
    if (Math.abs(a.y - b.y) < 1e-6) return c.s.hLine(a.x, b.x, a.y, color, width, style);
    if (Math.abs(a.x - b.x) < 1e-6) return c.s.vLine(a.x, a.y, b.y, color, width, style);
  }
  c.s.strokePath(points, color, width, style);
}

export function fill(c: ToolContext, polygon: Point[], color?: string, opacity?: number): void {
  const fillColor = color ?? c.item.style.fillColor;
  if (!fillColor) return;
  c.s.fillPolygon(polygon.map((p) => ({ x: p.x, y: p.y })), withAlpha(fillColor, opacity ?? c.item.style.fillOpacity));
}

/** Text with an optional rounded background box; returns its bounds. */
export function label(
  c: ToolContext,
  text: string,
  x: number,
  y: number,
  opts: { color?: string; bg?: string; align?: CanvasTextAlign; baseline?: CanvasTextBaseline; size?: number; bold?: boolean; pad?: number } = {},
): Rect {
  const size = opts.size ?? c.item.style.fontSize;
  const font = { size, family: c.theme.fontFamily, weight: opts.bold ? "600" : "normal" };
  const lines = text.split("\n");
  const width = Math.max(...lines.map((l) => c.s.measure(l, font)));
  const lineH = size * 1.3;
  const height = lineH * lines.length;
  const pad = opts.pad ?? (opts.bg ? 5 : 0);
  const align = opts.align ?? "left";
  const baseline = opts.baseline ?? "middle";
  const left = align === "center" ? x - width / 2 : align === "right" ? x - width : x;
  const top = baseline === "top" ? y : baseline === "bottom" ? y - height : y - height / 2;
  if (opts.bg) roundRect(c.s, left - pad, top - pad, width + pad * 2, height + pad * 2, 3, opts.bg);
  lines.forEach((line, i) =>
    c.s.text(line, left, top + lineH * (i + 0.5), opts.color ?? c.item.style.textColor, font, "left", "middle"),
  );
  return { x: left - pad, y: top - pad, width: width + pad * 2, height: height + pad * 2 };
}

export function roundRect(s: Surface, x: number, y: number, w: number, h: number, r: number, color: string): void {
  const ctx = s.ctx;
  ctx.fillStyle = color;
  ctx.beginPath();
  const k = s.hpr;
  if (typeof ctx.roundRect === "function") ctx.roundRect(x * k, y * s.vpr, w * k, h * s.vpr, r * k);
  else ctx.rect(x * k, y * s.vpr, w * k, h * s.vpr);
  ctx.fill();
}

export function arrowHead(c: ToolContext, from: Point, to: Point, color = c.item.style.color, size = 8 + c.item.style.width * 2): void {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const a = { x: to.x - size * Math.cos(angle - Math.PI / 7), y: to.y - size * Math.sin(angle - Math.PI / 7) };
  const b = { x: to.x - size * Math.cos(angle + Math.PI / 7), y: to.y - size * Math.sin(angle + Math.PI / 7) };
  c.s.fillPolygon([to, a, b], color);
}

export function pctChange(from: number, to: number): string {
  if (from === 0) return "";
  const v = (to / from - 1) * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}
