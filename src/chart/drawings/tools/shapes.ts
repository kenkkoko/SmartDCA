import type { Point } from "../../engine/types";
import { curveThrough, dist, distToPolyline, insidePolygon, withAlpha } from "../geometry";
import { HIT, fill, label, stroke, type Tool, type ToolContext } from "../tool";

function rectCorners(a: Point, b: Point): Point[] {
  return [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }];
}

function textInside(c: ToolContext, box: Point[]): void {
  if (!c.item.text) return;
  const xs = box.map((p) => p.x);
  const ys = box.map((p) => p.y);
  label(c, c.item.text, (Math.min(...xs) + Math.max(...xs)) / 2, (Math.min(...ys) + Math.max(...ys)) / 2, { align: "center" });
}

export const rectangle: Tool = {
  type: "rectangle",
  label: "矩形",
  group: "shapes",
  create: 2,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.15 },
  caps: { fill: true, text: true, extend: true },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    const left = Math.min(a.x, b.x);
    const right = Math.max(a.x, b.x);
    const x0 = c.item.style.extendLeft ? c.rect.x : left;
    const x1 = c.item.style.extendRight ? c.rect.x + c.rect.width : right;
    const box = rectCorners({ x: x0, y: a.y }, { x: x1, y: b.y });
    fill(c, box);
    stroke(c, [...box, box[0]!]);
    textInside(c, box);
  },
  hit(c, p) {
    const [a, b] = c.points as [Point, Point];
    const box = rectCorners(a, b);
    const edge = distToPolyline(p, box, true);
    if (edge <= HIT) return edge;
    return c.item.style.fillColor && insidePolygon(p, box) ? HIT - 1 : null;
  },
  // TradingView: 4 corners + 4 edge midpoints
  handles(c) {
    const [a, b] = c.points as [Point, Point];
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    return [a, b, { x: b.x, y: a.y }, { x: a.x, y: b.y }, { x: mx, y: a.y }, { x: mx, y: b.y }, { x: a.x, y: my }, { x: b.x, y: my }];
  },
  dragHandle(c, index, p) {
    let [a, b] = c.points as [Point, Point];
    switch (index) {
      case 0: a = p; break;
      case 1: b = p; break;
      case 2: a = { x: a.x, y: p.y }; b = { x: p.x, y: b.y }; break;
      case 3: a = { x: p.x, y: a.y }; b = { x: b.x, y: p.y }; break;
      case 4: a = { x: a.x, y: p.y }; break;
      case 5: b = { x: b.x, y: p.y }; break;
      case 6: a = { x: p.x, y: a.y }; break;
      case 7: b = { x: p.x, y: b.y }; break;
      default: return null;
    }
    return [a, b];
  },
};

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, steps = 64): Point[] {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = (i / steps) * Math.PI * 2;
    return { x: cx + rx * Math.cos(t), y: cy + ry * Math.sin(t) };
  });
}

export const ellipse: Tool = {
  type: "ellipse",
  label: "橢圓",
  group: "shapes",
  create: 2,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.15 },
  caps: { fill: true, text: true },
  render(c) {
    const [a, b] = c.points as [Point, Point];
    const pts = ellipsePoints((a.x + b.x) / 2, (a.y + b.y) / 2, Math.abs(b.x - a.x) / 2, Math.abs(b.y - a.y) / 2);
    fill(c, pts);
    stroke(c, pts);
    textInside(c, [a, b]);
  },
  hit(c, p) {
    const [a, b] = c.points as [Point, Point];
    const pts = ellipsePoints((a.x + b.x) / 2, (a.y + b.y) / 2, Math.abs(b.x - a.x) / 2, Math.abs(b.y - a.y) / 2);
    const d = distToPolyline(p, pts);
    if (d <= HIT) return d;
    return c.item.style.fillColor && insidePolygon(p, pts) ? HIT - 1 : null;
  },
};

export const circle: Tool = {
  type: "circle",
  label: "圓形",
  group: "shapes",
  create: 2,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.15 },
  caps: { fill: true, text: true },
  render(c) {
    const [center, edge] = c.points as [Point, Point];
    const r = dist(center, edge);
    const pts = ellipsePoints(center.x, center.y, r, r);
    fill(c, pts);
    stroke(c, pts);
    if (c.item.text) label(c, c.item.text, center.x, center.y, { align: "center" });
  },
  hit(c, p) {
    const [center, edge] = c.points as [Point, Point];
    const r = dist(center, edge);
    const d = Math.abs(dist(p, center) - r);
    if (d <= HIT) return d;
    return c.item.style.fillColor && dist(p, center) < r ? HIT - 1 : null;
  },
};

export const triangle: Tool = {
  type: "triangle",
  label: "三角形",
  group: "shapes",
  create: 3,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.15 },
  caps: { fill: true },
  render(c) {
    const pts = c.points.slice(0, 3);
    fill(c, pts);
    stroke(c, [...pts, pts[0]!]);
  },
  hit(c, p) {
    const pts = c.points.slice(0, 3);
    const d = distToPolyline(p, pts, true);
    if (d <= HIT) return d;
    return c.item.style.fillColor && insidePolygon(p, pts) ? HIT - 1 : null;
  },
};

// ---- brushes ----

export const pencil: Tool = {
  type: "freehand",
  label: "畫筆",
  group: "brush",
  create: "stroke",
  render(c) {
    stroke(c, c.points);
  },
  hit(c, p) {
    const d = distToPolyline(p, c.points);
    return d <= HIT ? d : null;
  },
  handles: () => [],
};

export const highlighter: Tool = {
  type: "highlighter",
  label: "螢光筆",
  group: "brush",
  create: "stroke",
  defaults: { color: "#f5b942", width: 14 },
  caps: { line: false },
  render(c) {
    c.s.strokePath(c.points, withAlpha(c.item.style.color, 0.35), c.item.style.width);
  },
  hit(c, p) {
    const d = distToPolyline(p, c.points);
    return d <= c.item.style.width / 2 + 2 ? d : null;
  },
  handles: () => [],
};

export const polyline: Tool = {
  type: "polyline",
  label: "折線",
  group: "brush",
  create: "multi",
  caps: { fill: true },
  render(c) {
    if (c.item.style.fillColor && c.points.length > 2) fill(c, c.points);
    stroke(c, c.points);
  },
  hit(c, p) {
    const d = distToPolyline(p, c.points);
    return d <= HIT ? d : null;
  },
};

export const curvedLine: Tool = {
  type: "curved_line",
  label: "曲線",
  group: "brush",
  create: 3,
  render(c) {
    const [a, b, mid] = c.points as [Point, Point, Point]; // start, end, bend
    stroke(c, curveThrough(a, mid, b));
  },
  hit(c, p) {
    const [a, b, mid] = c.points as [Point, Point, Point]; // start, end, bend
    const d = distToPolyline(p, curveThrough(a, mid, b));
    return d <= HIT ? d : null;
  },
};

export const SHAPE_TOOLS: Tool[] = [rectangle, ellipse, circle, triangle];
export const BRUSH_TOOLS: Tool[] = [pencil, highlighter, polyline, curvedLine];
