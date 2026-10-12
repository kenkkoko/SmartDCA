import type { Point } from "../../engine/types";
import { distToSegment, readable, withAlpha } from "../geometry";
import { HIT, label, stroke, type Tool, type ToolContext } from "../tool";
import { defaultFibSettings, type Anchor, type FibSettings } from "../types";

// ---------------------------------------------------------------------------
// Retracement (TradingView): 1 on the first point, 0 on the second (reverse
// swaps). Levels are placed in price (or log price), lines span the two
// points unless extended, labels read "0.618(1234.56)".
// ---------------------------------------------------------------------------

/** Defaults new retracements start from ("Save as default" replaces them). */
let fibDefaults: FibSettings = defaultFibSettings();
export function getFibDefaults(): FibSettings {
  return structuredClone(fibDefaults);
}
export function setFibDefaults(settings: FibSettings | null): void {
  fibDefaults = settings ? structuredClone(settings) : defaultFibSettings();
}

export function fibSettingsOf(c: ToolContext): FibSettings {
  return c.item.style.fib ?? fibDefaults;
}

/** Enabled levels (duplicates dropped) with their price. */
export function fibLevels(anchors: Anchor[], settings: FibSettings): { value: number; color: string; price: number }[] {
  const [a, b] = anchors as [Anchor, Anchor];
  if (!a || !b) return [];
  const seen = new Set<number>();
  const out: { value: number; color: string; price: number }[] = [];
  for (const level of settings.levels) {
    if (!level.enabled || !Number.isFinite(level.value) || seen.has(level.value)) continue;
    seen.add(level.value);
    const [from, to] = settings.reverse ? [a.price, b.price] : [b.price, a.price];
    const price =
      settings.logLevels && from > 0 && to > 0
        ? Math.exp(Math.log(from) + (Math.log(to) - Math.log(from)) * level.value)
        : from + (to - from) * level.value;
    out.push({ value: level.value, color: settings.useOneColor ? settings.oneColor : level.color, price });
  }
  return out;
}

function xRange(c: ToolContext): { from: number; to: number } {
  const [a, b] = c.points as [Point, Point];
  const lo = Math.min(a.x, b.x);
  const hi = Math.max(a.x, b.x);
  return {
    from: c.item.style.extendLeft ? c.rect.x : lo,
    to: c.item.style.extendRight ? c.rect.x + c.rect.width : hi,
  };
}

function formatLevel(v: number, format: FibSettings["labelFormat"]): string {
  return format === "percent" ? `${Number((v * 100).toFixed(2))}%` : String(Number(v.toFixed(4)));
}

export const fibRetracement: Tool = {
  type: "fib_retracement",
  label: "斐波那契回撤",
  group: "fibonacci",
  create: 2,
  caps: { extend: true, dialog: "fib" },
  init(item) {
    item.style.fib = getFibDefaults();
  },
  render(c) {
    const s = fibSettingsOf(c);
    const levels = fibLevels(c.item.anchors, s).map((l) => ({ ...l, color: readable(l.color, c.theme.background), y: c.toY(l.price) }));
    const { from, to } = xRange(c);
    if (s.background) {
      const sorted = [...levels].sort((p, q) => p.value - q.value);
      for (let i = 1; i < sorted.length; i++) {
        const lower = sorted[i - 1]!;
        const upper = sorted[i]!;
        c.s.fillRect(from, Math.min(lower.y, upper.y), to - from, Math.abs(upper.y - lower.y), withAlpha(upper.color, s.backgroundOpacity));
      }
    }
    for (const l of levels) stroke(c, [{ x: from, y: l.y }, { x: to, y: l.y }], l.color);
    if (s.levelLabels || s.prices) {
      const x = s.labelsH === "left" ? from - 4 : s.labelsH === "right" ? to + 4 : (from + to) / 2;
      const align: CanvasTextAlign = s.labelsH === "left" ? "right" : s.labelsH === "right" ? "left" : "center";
      const baseline: CanvasTextBaseline = s.labelsV === "top" ? "bottom" : s.labelsV === "bottom" ? "top" : "middle";
      const dy = s.labelsV === "top" ? -2 : s.labelsV === "bottom" ? 2 : 0;
      for (const l of levels) {
        const text = `${s.levelLabels ? formatLevel(l.value, s.labelFormat) : ""}${s.prices ? `(${c.fmt(l.price)})` : ""}`;
        label(c, text, x, l.y + dy, { align, baseline, color: l.color, size: c.item.style.fontSize - 2 });
      }
    }
    if (s.trendLine) stroke(c, c.points.slice(0, 2), readable(s.trendLineColor, c.theme.background), 1, s.trendLineStyle);
  },
  hit(c, p) {
    const s = fibSettingsOf(c);
    const { from, to } = xRange(c);
    let best: number | null = null;
    if (p.x >= from - HIT && p.x <= to + HIT)
      for (const l of fibLevels(c.item.anchors, s)) {
        const d = Math.abs(p.y - c.toY(l.price));
        if (d <= HIT) best = Math.min(best ?? d, d);
      }
    if (best === null && s.trendLine) {
      const [a, b] = c.points as [Point, Point];
      const d = distToSegment(p, a, b);
      if (d <= HIT) best = d;
    }
    return best;
  },
};

// ---------------------------------------------------------------------------
// Trend-based extension: levels of the first leg projected from point 3.
// ---------------------------------------------------------------------------

const EXTENSION_LEVELS = [0, 0.618, 1, 1.272, 1.618, 2.618];
const EXTENSION_COLORS = ["#787b86", "#f23645", "#089981", "#ff9800", "#2962ff", "#9c27b0"];

function extensionLevels(c: ToolContext): { value: number; price: number; color: string }[] {
  const [a, b, d] = c.item.anchors as [Anchor, Anchor, Anchor];
  if (!a || !b || !d) return [];
  return EXTENSION_LEVELS.map((v, i) => ({ value: v, price: d.price + (b.price - a.price) * v, color: EXTENSION_COLORS[i]! }));
}

export const fibExtension: Tool = {
  type: "fib_extension",
  label: "斐波那契延伸",
  group: "fibonacci",
  create: 3,
  caps: { extend: true },
  render(c) {
    const [a, b, d] = c.points as [Point, Point, Point];
    const from = Math.min(d.x, b.x);
    const to = c.item.style.extendRight ? c.rect.x + c.rect.width : Math.max(d.x, b.x) + Math.abs(b.x - a.x);
    for (const l of extensionLevels(c)) {
      const y = c.toY(l.price);
      stroke(c, [{ x: from, y }, { x: to, y }], l.color);
      label(c, `${l.value}(${c.fmt(l.price)})`, from - 4, y, { align: "right", color: l.color, size: c.item.style.fontSize - 2 });
    }
    stroke(c, [a, b, d], "#787b86", 1, "dashed");
  },
  hit(c, p) {
    const [a, b, d] = c.points as [Point, Point, Point];
    let best: number | null = null;
    for (const l of extensionLevels(c)) {
      const dd = Math.abs(p.y - c.toY(l.price));
      if (dd <= HIT && p.x >= Math.min(d.x, b.x) - HIT) best = Math.min(best ?? dd, dd);
    }
    const seg = Math.min(distToSegment(p, a, b), distToSegment(p, b, d));
    if (best === null && seg <= HIT) best = seg;
    return best;
  },
};

// ---------------------------------------------------------------------------
// Fibonacci channel: parallel lines at Fibonacci multiples of the width.
// ---------------------------------------------------------------------------

const CHANNEL_LEVELS = [0, 0.382, 0.5, 0.618, 1, 1.618];

function channelLevels(c: ToolContext): [Point, Point, number][] {
  const [a, b, d] = c.points as [Point, Point, Point];
  if (!a || !b || !d) return [];
  const dx = b.x - a.x;
  const yAt = (x: number) => (dx === 0 ? a.y : a.y + ((x - a.x) * (b.y - a.y)) / dx);
  const offset = d.y - yAt(d.x);
  const right = c.item.style.extendRight ? c.rect.x + c.rect.width : b.x;
  return CHANNEL_LEVELS.map((v) => [
    { x: a.x, y: a.y + offset * v },
    { x: right, y: yAt(right) + offset * v },
    v,
  ]);
}

export const fibChannel: Tool = {
  type: "fib_channel",
  label: "斐波那契通道",
  group: "fibonacci",
  create: 3,
  caps: { extend: true },
  render(c) {
    channelLevels(c).forEach(([p, q, v], i) => {
      const color = EXTENSION_COLORS[i % EXTENSION_COLORS.length]!;
      stroke(c, [p, q], color);
      label(c, String(v), p.x - 4, p.y, { align: "right", color, size: c.item.style.fontSize - 2 });
    });
  },
  hit(c, p) {
    let best: number | null = null;
    for (const [a, b] of channelLevels(c)) {
      const d = distToSegment(p, a, b);
      if (d <= HIT) best = Math.min(best ?? d, d);
    }
    return best;
  },
};

export const FIB_TOOLS: Tool[] = [fibRetracement, fibExtension, fibChannel];
