import type { Point } from "../../engine/types";
import { distToPolyline, insidePolygon } from "../geometry";
import { HIT, arrowHead, fill, label, pctChange, stroke, type Tool, type ToolContext } from "../tool";
import type { Anchor } from "../types";

// ---------------------------------------------------------------------------
// Long / short position: anchors = entry, target, stop. The target's time is
// the box's right edge. Two clicks place entry and target; the stop mirrors
// the target (1:1 risk/reward), like TradingView.
// ---------------------------------------------------------------------------

function position(kind: "long" | "short"): Tool {
  const complete = (placed: Anchor[]): Anchor[] => {
    if (placed.length !== 2) return placed;
    const [entry, target] = placed as [Anchor, Anchor];
    return [entry, target, { time: target.time, price: entry.price - (target.price - entry.price) }];
  };
  const geom = (c: ToolContext) => {
    const [e, t, s] = c.points as [Point, Point, Point];
    const x0 = Math.min(e.x, t.x);
    const x1 = Math.max(e.x, t.x, x0 + 4);
    return { e, t, s, x0, x1 };
  };
  return {
    type: kind === "long" ? "long_position" : "short_position",
    label: kind === "long" ? "多單部位" : "空單部位",
    group: "measure",
    create: 2,
    complete,
    caps: { line: false },
    render(c) {
      const { e, t, s, x0, x1 } = geom(c);
      const [entry, target, stop] = c.item.anchors as [Anchor, Anchor, Anchor];
      const box = (y0: number, y1: number): Point[] => [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }];
      fill(c, box(e.y, t.y), c.theme.up, 0.22);
      fill(c, box(e.y, s.y), c.theme.down, 0.22);
      stroke(c, [{ x: x0, y: e.y }, { x: x1, y: e.y }], "#9aa4b2", 1);
      const reward = Math.abs(target.price - entry.price);
      const risk = Math.abs(entry.price - stop.price);
      const mid = (x0 + x1) / 2;
      const tUp = t.y < e.y;
      label(c, `目標 ${c.fmt(target.price)} (${pctChange(entry.price, target.price)})`, mid, t.y + (tUp ? -4 : 4), { align: "center", baseline: tUp ? "bottom" : "top", bg: c.theme.up, color: "#06281c", size: 11 });
      label(c, `停損 ${c.fmt(stop.price)} (${pctChange(entry.price, stop.price)})`, mid, s.y + (tUp ? 4 : -4), { align: "center", baseline: tUp ? "top" : "bottom", bg: c.theme.down, color: "#2d0a10", size: 11 });
      label(c, `風報比 ${risk ? (reward / risk).toFixed(2) : "∞"}`, mid, e.y, { align: "center", bg: "rgba(20,24,32,0.9)", size: 11 });
    },
    hit(c, p) {
      const { e, t, s, x0, x1 } = geom(c);
      const top = Math.min(e.y, t.y, s.y);
      const bottom = Math.max(e.y, t.y, s.y);
      return insidePolygon(p, [{ x: x0, y: top }, { x: x1, y: top }, { x: x1, y: bottom }, { x: x0, y: bottom }]) ? HIT - 1 : null;
    },
    // entry, target, stop on the left edge + the right edge on the entry line
    handles(c) {
      const { e, t, s, x0, x1 } = geom(c);
      return [{ x: x0, y: e.y }, { x: x0, y: t.y }, { x: x0, y: s.y }, { x: x1, y: e.y }];
    },
    dragHandle(c, index, p) {
      const { e, t, s } = geom(c);
      switch (index) {
        case 0: // entry moves the whole start: time and price
          return [p, { x: t.x, y: t.y }, { x: t.x, y: s.y }];
        case 1:
          return [e, { x: t.x, y: p.y }, s];
        case 2:
          return [e, t, { x: s.x, y: p.y }];
        case 3: // width
          return [e, { x: Math.max(p.x, e.x + 4), y: t.y }, { x: Math.max(p.x, e.x + 4), y: s.y }];
        default:
          return null;
      }
    },
  };
}

export const longPosition = position("long");
export const shortPosition = position("short");

// ---------------------------------------------------------------------------

function rangeBox(c: ToolContext): Point[] {
  const [a, b] = c.points as [Point, Point];
  return [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }];
}

function hitBox(c: ToolContext, p: Point): number | null {
  const box = rangeBox(c);
  const d = distToPolyline(p, box, true);
  if (d <= HIT) return d;
  return insidePolygon(p, box) ? HIT - 1 : null;
}

function barsBetween(c: ToolContext): number {
  const [a, b] = c.item.anchors as [Anchor, Anchor];
  return Math.round(c.chart.time.timeToIndex(b.time) - c.chart.time.timeToIndex(a.time));
}

function duration(c: ToolContext): string {
  const [a, b] = c.item.anchors as [Anchor, Anchor];
  const days = Math.abs(b.time - a.time) / 86400;
  return days >= 1 ? `${Math.round(days)} 天` : `${Math.round(days * 24)} 小時`;
}

export const priceRange: Tool = {
  type: "price_range",
  label: "價格區間",
  group: "measure",
  create: 2,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.15 },
  caps: { line: false },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    const [pa, pb] = c.item.anchors as [Anchor, Anchor];
    const up = pb.price >= pa.price;
    const color = up ? c.theme.up : c.theme.down;
    fill(c, rangeBox(c), color, 0.15);
    const x = (a.x + b.x) / 2;
    stroke(c, [{ x, y: a.y }, { x, y: b.y }], color, 1);
    arrowHead(c, { x, y: a.y }, { x, y: b.y }, color, 8);
    label(c, `${c.fmt(pb.price - pa.price)} (${pctChange(pa.price, pb.price)})`, x, b.y + (up ? -6 : 6), { align: "center", baseline: up ? "bottom" : "top", bg: color, color: "#0b0e14", size: 11 });
  },
  hit: hitBox,
};

export const dateRange: Tool = {
  type: "date_range",
  label: "日期區間",
  group: "measure",
  create: 2,
  caps: { line: false },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    fill(c, rangeBox(c), "#2962ff", 0.15);
    const y = (a.y + b.y) / 2;
    stroke(c, [{ x: a.x, y }, { x: b.x, y }], "#2962ff", 1);
    arrowHead(c, { x: a.x, y }, { x: b.x, y }, "#2962ff", 8);
    label(c, `${barsBetween(c)} 根 · ${duration(c)}`, (a.x + b.x) / 2, Math.max(a.y, b.y) + 6, { align: "center", baseline: "top", bg: "#2962ff", color: "#ffffff", size: 11 });
  },
  hit: hitBox,
};

export const datePriceRange: Tool = {
  type: "date_price_range",
  label: "日期與價格區間",
  group: "measure",
  create: 2,
  caps: { line: false },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    const [pa, pb] = c.item.anchors as [Anchor, Anchor];
    const up = pb.price >= pa.price;
    const color = up ? c.theme.up : c.theme.down;
    fill(c, rangeBox(c), color, 0.15);
    const x = (a.x + b.x) / 2;
    const y = (a.y + b.y) / 2;
    stroke(c, [{ x, y: a.y }, { x, y: b.y }], color, 1);
    stroke(c, [{ x: a.x, y }, { x: b.x, y }], color, 1);
    arrowHead(c, { x, y: a.y }, { x, y: b.y }, color, 8);
    arrowHead(c, { x: a.x, y }, { x: b.x, y }, color, 8);
    label(c, `${c.fmt(pb.price - pa.price)} (${pctChange(pa.price, pb.price)})\n${barsBetween(c)} 根 · ${duration(c)}`, x, Math.max(a.y, b.y) + 6, { align: "center", baseline: "top", bg: color, color: "#0b0e14", size: 11 });
  },
  hit: hitBox,
};

export const MEASURE_TOOLS: Tool[] = [longPosition, shortPosition, priceRange, dateRange, datePriceRange];
