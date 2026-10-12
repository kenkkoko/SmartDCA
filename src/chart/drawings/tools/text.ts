import type { Point, Rect } from "../../engine/types";
import { distToSegment } from "../geometry";
import { HIT, label, stroke, type Tool, type ToolContext } from "../tool";

function inside(p: Point, r: Rect): boolean {
  return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height;
}

/** Measure without painting by drawing to an invisible position is overkill; estimate. */
function textBox(c: ToolContext, text: string, x: number, y: number, align: CanvasTextAlign, pad: number): Rect {
  const font = { size: c.item.style.fontSize, family: c.theme.fontFamily };
  const lines = text.split("\n");
  const w = Math.max(...lines.map((l) => c.s.measure(l, font)));
  const h = c.item.style.fontSize * 1.3 * lines.length;
  const left = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  return { x: left - pad, y: y - h / 2 - pad, width: w + pad * 2, height: h + pad * 2 };
}

export const textTool: Tool = {
  type: "text",
  label: "文字",
  group: "text",
  create: 1,
  defaults: { color: "#e6e9ef", textColor: "#e6e9ef", fontSize: 14 },
  caps: { text: true, line: false },
  render(c) {
    const p = c.points[0]!;
    label(c, c.item.text || "文字", p.x, p.y, { color: c.item.style.textColor });
  },
  hit(c, p) {
    const a = c.points[0]!;
    return inside(p, textBox(c, c.item.text || "文字", a.x, a.y, "left", 3)) ? 1 : null;
  },
};

export const callout: Tool = {
  type: "callout",
  label: "標註",
  group: "text",
  create: 2,
  defaults: { color: "#2962ff", textColor: "#ffffff", fontSize: 13 },
  caps: { text: true, line: false },
  render(c) {
    const [target, box] = c.points as [Point, Point];
    stroke(c, [target, box], c.item.style.color, 1.5);
    label(c, c.item.text || "標註", box.x, box.y, { bg: c.item.style.color, color: c.item.style.textColor, align: "center" });
  },
  hit(c, p) {
    const [target, box] = c.points as [Point, Point];
    if (inside(p, textBox(c, c.item.text || "標註", box.x, box.y, "center", 5))) return 1;
    const d = distToSegment(p, target, box);
    return d <= HIT ? d : null;
  },
};

export const priceLabel: Tool = {
  type: "price_label",
  label: "價格標籤",
  group: "text",
  create: 1,
  defaults: { color: "#2962ff", textColor: "#ffffff", fontSize: 12 },
  caps: { line: false },
  render(c) {
    const p = c.points[0]!;
    const text = c.fmt(c.item.anchors[0]!.price);
    c.s.fillPolygon([p, { x: p.x + 8, y: p.y - 10 }, { x: p.x + 8, y: p.y + 10 }], c.item.style.color);
    label(c, text, p.x + 8, p.y, { bg: c.item.style.color, color: c.item.style.textColor, pad: 4 });
  },
  hit(c, p) {
    const a = c.points[0]!;
    return inside(p, { x: a.x, y: a.y - 12, width: 90, height: 24 }) ? 1 : null;
  },
};

function mark(direction: "up" | "down"): Tool {
  return {
    type: direction === "up" ? "arrow_mark_up" : "arrow_mark_down",
    label: direction === "up" ? "向上箭頭" : "向下箭頭",
    group: "text",
    create: 1,
    defaults: { color: direction === "up" ? "#00d68f" : "#ff5b6e", textColor: "#e6e9ef" },
    caps: { text: true, line: false },
    render(c) {
      const p = c.points[0]!;
      const k = direction === "up" ? 1 : -1;
      // arrow pointing at the anchor
      const tip = p;
      const head = 9;
      c.s.fillPolygon(
        [tip, { x: tip.x + head, y: tip.y + k * head }, { x: tip.x + 4, y: tip.y + k * head }, { x: tip.x + 4, y: tip.y + k * 26 }, { x: tip.x - 4, y: tip.y + k * 26 }, { x: tip.x - 4, y: tip.y + k * head }, { x: tip.x - head, y: tip.y + k * head }],
        c.item.style.color,
      );
      if (c.item.text) label(c, c.item.text, p.x, p.y + k * 32, { align: "center", baseline: direction === "up" ? "top" : "bottom" });
    },
    hit(c, p) {
      const a = c.points[0]!;
      const k = direction === "up" ? 1 : -1;
      const top = Math.min(a.y, a.y + k * 26);
      return inside(p, { x: a.x - 10, y: top, width: 20, height: 26 }) ? 1 : null;
    },
  };
}

export const arrowMarkUp = mark("up");
export const arrowMarkDown = mark("down");

export const TEXT_TOOLS: Tool[] = [textTool, callout, priceLabel, arrowMarkUp, arrowMarkDown];
