import type { Point } from "../../engine/types";
import { distToPolyline, extendSegment, midpoint } from "../geometry";
import { HIT, fill, label, stroke, type Tool, type ToolContext } from "../tool";

/** Label each point above a local high or below a local low. */
function pointLabels(c: ToolContext, names: string[]): void {
  c.points.forEach((p, i) => {
    const name = names[i];
    if (!name) return;
    const prev = c.points[i - 1];
    const next = c.points[i + 1];
    const neighbour = prev ?? next;
    const high = neighbour ? p.y <= neighbour.y && (!next || p.y <= next.y || !prev) : true;
    label(c, name, p.x, high ? p.y - 6 : p.y + 6, { align: "center", baseline: high ? "bottom" : "top", bold: true, color: c.item.style.color, size: 12 });
  });
}

function ratio(c: ToolContext, a: number, b: number, from: number, to: number): string {
  const an = c.item.anchors;
  const leg = (i: number, j: number) => Math.abs(an[j]!.price - an[i]!.price);
  const base = leg(a, b);
  return base ? (leg(from, to) / base).toFixed(3) : "";
}

function dashedRatio(c: ToolContext, i: number, j: number, text: string): void {
  const p = c.points[i];
  const q = c.points[j];
  if (!p || !q || !text) return;
  stroke(c, [p, q], c.item.style.color, 1, "dashed");
  const m = midpoint(p, q);
  label(c, text, m.x, m.y, { align: "center", bg: "rgba(20,24,32,0.85)", size: 11 });
}

function hitPolyline(c: ToolContext, p: Point): number | null {
  const d = distToPolyline(p, c.points);
  return d <= HIT ? d : null;
}

export const xabcd: Tool = {
  type: "xabcd_pattern",
  label: "XABCD 型態",
  group: "patterns",
  create: 5,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.15, color: "#2962ff" },
  caps: { fill: true },
  render(c) {
    const [x, a, b, cc, d] = c.points;
    if (x && a && b) fill(c, [x, a, b]);
    if (b && cc && d) fill(c, [b, cc, d]);
    stroke(c, c.points);
    dashedRatio(c, 0, 2, ratio(c, 0, 1, 1, 2));
    dashedRatio(c, 1, 3, ratio(c, 1, 2, 2, 3));
    dashedRatio(c, 2, 4, ratio(c, 2, 3, 3, 4));
    dashedRatio(c, 0, 4, ratio(c, 0, 1, 1, 4));
    pointLabels(c, ["X", "A", "B", "C", "D"]);
  },
  hit: hitPolyline,
};

export const abcd: Tool = {
  type: "abcd_pattern",
  label: "ABCD 型態",
  group: "patterns",
  create: 4,
  render(c) {
    stroke(c, c.points);
    dashedRatio(c, 0, 2, ratio(c, 0, 1, 1, 2));
    dashedRatio(c, 1, 3, ratio(c, 1, 2, 2, 3));
    pointLabels(c, ["A", "B", "C", "D"]);
  },
  hit: hitPolyline,
};

export const headShoulders: Tool = {
  type: "head_shoulders",
  label: "頭肩型態",
  group: "patterns",
  create: 7,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.12 },
  caps: { fill: true },
  render(c) {
    const p = c.points;
    if (p.length >= 7) {
      fill(c, [p[0]!, p[1]!, p[2]!]);
      fill(c, [p[2]!, p[3]!, p[4]!]);
      fill(c, [p[4]!, p[5]!, p[6]!]);
      const [n1, n2] = extendSegment(p[2]!, p[4]!, true, true, c.rect);
      stroke(c, [n1, n2], c.item.style.color, 1, "dashed");
    }
    stroke(c, p);
    pointLabels(c, ["", "左肩", "", "頭", "", "右肩", ""]);
  },
  hit: hitPolyline,
};

export const elliottImpulse: Tool = {
  type: "elliott_impulse",
  label: "艾略特推動浪",
  group: "patterns",
  create: 6,
  render(c) {
    stroke(c, c.points);
    pointLabels(c, ["(0)", "(1)", "(2)", "(3)", "(4)", "(5)"]);
  },
  hit: hitPolyline,
};

export const elliottCorrection: Tool = {
  type: "elliott_correction",
  label: "艾略特修正浪",
  group: "patterns",
  create: 4,
  render(c) {
    stroke(c, c.points);
    pointLabels(c, ["(0)", "(A)", "(B)", "(C)"]);
  },
  hit: hitPolyline,
};

export const PATTERN_TOOLS: Tool[] = [xabcd, abcd, headShoulders, elliottImpulse, elliottCorrection];
