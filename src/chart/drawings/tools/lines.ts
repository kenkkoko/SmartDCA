import { angleDeg, distToSegment, extendSegment, ray } from "../geometry";
import { HIT, arrowHead, label, lineText, pctChange, stroke, type Tool, type ToolContext } from "../tool";
import type { Point } from "../../engine/types";

function hitLine(p: Point, a: Point, b: Point): number | null {
  const d = distToSegment(p, a, b);
  return d <= HIT ? d : null;
}

function statsText(c: ToolContext): string {
  const [a, b] = c.item.anchors as [{ time: number; price: number }, { time: number; price: number }];
  const bars = Math.round(c.chart.time.timeToIndex(b.time) - c.chart.time.timeToIndex(a.time));
  const [pa, pb] = c.points as [Point, Point];
  return `${c.fmt(b.price - a.price)} (${pctChange(a.price, b.price)})\n${bars} 根 · ${angleDeg(pa, pb).toFixed(1)}°`;
}

/** Trend line with optional extensions and stats. */
function segment(c: ToolContext): [Point, Point] {
  const [a, b] = c.points as [Point, Point];
  return extendSegment(a, b, c.item.style.extendLeft, c.item.style.extendRight, c.rect);
}

export const trendLine: Tool = {
  type: "trend_line",
  label: "趨勢線",
  group: "lines",
  create: 2,
  caps: { extend: true, stats: true, text: true },
  render(c) {
    const seg = segment(c);
    stroke(c, seg);
    if (c.item.style.stats && c.points.length > 1) {
      const b = c.points[1]!;
      label(c, statsText(c), b.x + 8, b.y, { bg: "rgba(20,24,32,0.85)", size: 11 });
    }
    const [a, b] = c.points as [Point, Point];
    lineText(c, a, b);
  },
  hit(c, p) {
    const [a, b] = segment(c);
    return hitLine(p, a, b);
  },
};

export const rayTool: Tool = {
  type: "ray",
  label: "射線",
  group: "lines",
  create: 2,
  caps: { text: true },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    stroke(c, ray(a, b, c.rect));
    lineText(c, a, b);
  },
  hit(c, p) {
    const [a, b] = c.points as [Point, Point];
    const [s, e] = ray(a, b, c.rect);
    return hitLine(p, s, e);
  },
};

export const extendedLine: Tool = {
  type: "extended_line",
  label: "延伸線",
  group: "lines",
  create: 2,
  caps: { text: true },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    stroke(c, extendSegment(a, b, true, true, c.rect));
    lineText(c, a, b);
  },
  hit(c, p) {
    const [a, b] = c.points as [Point, Point];
    const [s, e] = extendSegment(a, b, true, true, c.rect);
    return hitLine(p, s, e);
  },
};

export const horizontalLine: Tool = {
  type: "horizontal_line",
  label: "水平線",
  group: "lines",
  create: 1,
  caps: { text: true },
  // TradingView: a horizontal line shows its price on the axis
  defaults: { axisLabel: true, textH: "right" },
  render(c) {
    const y = c.points[0]!.y;
    stroke(c, [{ x: c.rect.x, y }, { x: c.rect.x + c.rect.width, y }]);
    lineText(c, { x: c.rect.x + 4, y }, { x: c.rect.x + c.rect.width - 4, y });
  },
  hit(c, p) {
    const d = Math.abs(p.y - c.points[0]!.y);
    return d <= HIT ? d : null;
  },
  axisTags: (c) => ({ prices: [c.item.anchors[0]!.price], times: [] }),
};

export const horizontalRay: Tool = {
  type: "horizontal_ray",
  label: "水平射線",
  group: "lines",
  create: 1,
  caps: { text: true },
  defaults: { axisLabel: true, textH: "left" },
  render(c) {
    const a = c.points[0]!;
    stroke(c, [a, { x: c.rect.x + c.rect.width, y: a.y }]);
    lineText(c, a, { x: c.rect.x + c.rect.width - 4, y: a.y });
  },
  hit(c, p) {
    const a = c.points[0]!;
    return p.x >= a.x - HIT ? (Math.abs(p.y - a.y) <= HIT ? Math.abs(p.y - a.y) : null) : null;
  },
  axisTags: (c) => ({ prices: [c.item.anchors[0]!.price], times: [] }),
};

export const verticalLine: Tool = {
  type: "vertical_line",
  label: "垂直線",
  group: "lines",
  create: 1,
  caps: { text: true, axisLabel: false },
  render(c) {
    const x = c.points[0]!.x;
    stroke(c, [{ x, y: c.rect.y }, { x, y: c.rect.y + c.rect.height }]);
    lineText(c, { x, y: c.rect.y + c.rect.height - 4 }, { x, y: c.rect.y + 4 });
  },
  hit(c, p) {
    const d = Math.abs(p.x - c.points[0]!.x);
    return d <= HIT ? d : null;
  },
  axisTags: (c) => ({ prices: [], times: [c.item.anchors[0]!.time] }),
};

export const crossLine: Tool = {
  type: "cross_line",
  label: "十字線",
  group: "lines",
  create: 1,
  render(c) {
    const { x, y } = c.points[0]!;
    stroke(c, [{ x: c.rect.x, y }, { x: c.rect.x + c.rect.width, y }]);
    stroke(c, [{ x, y: c.rect.y }, { x, y: c.rect.y + c.rect.height }]);
  },
  hit(c, p) {
    const { x, y } = c.points[0]!;
    const d = Math.min(Math.abs(p.x - x), Math.abs(p.y - y));
    return d <= HIT ? d : null;
  },
};

export const infoLine: Tool = {
  type: "info_line",
  label: "資訊線",
  group: "lines",
  create: 2,
  caps: { extend: true, text: true },
  render(c) {
    const seg = segment(c);
    stroke(c, seg);
    lineText(c, c.points[0]!, c.points[1]!);
    const b = c.points[1]!;
    label(c, statsText(c), b.x + 8, b.y, { bg: "rgba(20,24,32,0.85)", size: 11 });
  },
  hit(c, p) {
    const [a, b] = segment(c);
    return hitLine(p, a, b);
  },
};

export const trendAngle: Tool = {
  type: "trend_angle",
  label: "趨勢角度",
  group: "lines",
  create: 2,
  render(c) {
    const [a, b] = c.points as [Point, Point];
    stroke(c, [a, b]);
    const r = 40;
    stroke(c, [a, { x: a.x + r + 10, y: a.y }], c.item.style.color, 1, "dotted");
    const angle = angleDeg(a, b);
    const ctx = c.s.ctx;
    ctx.save();
    ctx.strokeStyle = c.item.style.color;
    ctx.lineWidth = c.s.hpr;
    ctx.beginPath();
    ctx.arc(a.x * c.s.hpr, a.y * c.s.vpr, r * c.s.hpr, 0, (-angle * Math.PI) / 180, angle > 0);
    ctx.stroke();
    ctx.restore();
    label(c, `${angle.toFixed(1)}°`, a.x + r + 6, a.y - (angle > 0 ? 10 : -10), { size: 11 });
  },
  hit(c, p) {
    const [a, b] = c.points as [Point, Point];
    return hitLine(p, a, b);
  },
};

export const arrowLine: Tool = {
  type: "arrow",
  label: "箭頭",
  group: "lines",
  create: 2,
  caps: { text: true },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    stroke(c, [a, b]);
    arrowHead(c, a, b);
    lineText(c, a, b);
  },
  hit(c, p) {
    const [a, b] = c.points as [Point, Point];
    return hitLine(p, a, b);
  },
};

export const LINE_TOOLS: Tool[] = [
  trendLine,
  rayTool,
  extendedLine,
  horizontalLine,
  horizontalRay,
  verticalLine,
  crossLine,
  infoLine,
  trendAngle,
  arrowLine,
];
