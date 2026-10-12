import type { Point } from "../../engine/types";
import { distToSegment, extendSegment, insidePolygon, midpoint } from "../geometry";
import { HIT, fill, stroke, type Tool, type ToolContext } from "../tool";

// ---------------------------------------------------------------------------
// Parallel channel (TradingView): anchors 0-1 = base line, anchor 2 sets the
// width as a vertical offset, so both lines share the same two x positions.
// Finite unless extended; dashed middle line; six handles.
// ---------------------------------------------------------------------------

type Channel = { base: [Point, Point]; offset: number; yAt: (x: number) => number };

function channel(points: Point[]): Channel | null {
  const [a, b, c] = points;
  if (!a || !b || !c) return null;
  const dx = b.x - a.x;
  const yAt = (x: number) => (dx === 0 ? a.y : a.y + ((x - a.x) * (b.y - a.y)) / dx);
  return { base: [a, b], offset: c.y - yAt(dx === 0 ? a.x : c.x), yAt };
}

function shifted(line: [Point, Point], dy: number): [Point, Point] {
  return [{ x: line[0].x, y: line[0].y + dy }, { x: line[1].x, y: line[1].y + dy }];
}

function channelLines(c: ToolContext, ch: Channel, fraction: number): [Point, Point] {
  const line = shifted(ch.base, ch.offset * fraction);
  return extendSegment(line[0], line[1], c.item.style.extendLeft, c.item.style.extendRight, c.rect);
}

export const parallelChannel: Tool = {
  type: "parallel_channel",
  label: "平行通道",
  group: "channels",
  create: 3,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.12, middleLine: true },
  caps: { fill: true, extend: true, middleLine: true },
  render(c) {
    const ch = channel(c.points);
    if (!ch) return;
    const base = channelLines(c, ch, 0);
    const top = channelLines(c, ch, 1);
    if (ch.offset !== 0) fill(c, [base[0], base[1], top[1], top[0]]);
    stroke(c, base);
    stroke(c, top);
    if (c.item.style.middleLine) stroke(c, channelLines(c, ch, 0.5), c.item.style.color, c.item.style.width, "dashed");
  },
  hit(c, p) {
    const ch = channel(c.points);
    if (!ch) return null;
    let best: number | null = null;
    for (const f of c.item.style.middleLine ? [0, 0.5, 1] : [0, 1]) {
      const [a, b] = channelLines(c, ch, f);
      const d = distToSegment(p, a, b);
      if (d <= HIT) best = Math.min(best ?? d, d);
    }
    if (best !== null) return best;
    if (c.item.style.fillColor) {
      const base = channelLines(c, ch, 0);
      const top = channelLines(c, ch, 1);
      if (insidePolygon(p, [base[0], base[1], top[1], top[0]])) return HIT - 1;
    }
    return null;
  },
  handles(c) {
    const ch = channel(c.points);
    if (!ch) return c.points;
    const top = shifted(ch.base, ch.offset);
    return [ch.base[0], ch.base[1], top[0], top[1], midpoint(...ch.base), midpoint(...top)];
  },
  dragHandle(c, index, p) {
    const ch = channel(c.points);
    if (!ch) return null;
    let [a, b] = ch.base;
    let width = ch.offset;
    switch (index) {
      case 0: a = p; break;
      case 1: b = p; break;
      case 2: a = { x: p.x, y: p.y - width }; break; // corners keep the width
      case 3: b = { x: p.x, y: p.y - width }; break;
      case 4: { // base middle: move the base line only
        const d = p.y - (a.y + b.y) / 2;
        a = { x: a.x, y: a.y + d };
        b = { x: b.x, y: b.y + d };
        width -= d;
        break;
      }
      case 5: width = p.y - (a.y + b.y) / 2; break; // top middle: width only
      default: return null;
    }
    const m = midpoint(a, b);
    return [a, b, { x: m.x, y: m.y + width }];
  },
};

// ---------------------------------------------------------------------------
// Regression trend: least squares of closes between two times, ±2σ channel.
// ---------------------------------------------------------------------------

function regression(c: ToolContext): { lines: [Point, Point][]; ok: boolean } {
  const [a, b] = c.item.anchors;
  if (!a || !b) return { lines: [], ok: false };
  const t = c.chart.time;
  const i0 = Math.round(t.timeToIndex(Math.min(a.time, b.time)));
  const i1 = Math.round(t.timeToIndex(Math.max(a.time, b.time)));
  const bars = c.bars;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = Math.max(0, i0); i <= Math.min(bars.length - 1, i1); i++) {
    xs.push(i);
    ys.push(bars[i]!.close);
  }
  const n = xs.length;
  if (n < 2) return { lines: [], ok: false };
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let den = 0;
  for (let k = 0; k < n; k++) {
    num += (xs[k]! - mx) * (ys[k]! - my);
    den += (xs[k]! - mx) ** 2;
  }
  const slope = den ? num / den : 0;
  const icpt = my - slope * mx;
  let sd = 0;
  for (let k = 0; k < n; k++) sd += (ys[k]! - (icpt + slope * xs[k]!)) ** 2;
  sd = Math.sqrt(sd / n);
  const from = xs[0]!;
  const to = xs[n - 1]!;
  const line = (shift: number): [Point, Point] => {
    const p0 = { x: t.indexToX(from), y: c.toY(icpt + slope * from + shift) };
    const p1 = { x: t.indexToX(to), y: c.toY(icpt + slope * to + shift) };
    return extendSegment(p0, p1, false, c.item.style.extendRight, c.rect);
  };
  return { lines: [line(0), line(2 * sd), line(-2 * sd)], ok: true };
}

export const regressionTrend: Tool = {
  type: "regression_trend",
  label: "迴歸趨勢",
  group: "channels",
  create: 2,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.08 },
  caps: { fill: true, extend: true },
  render(c) {
    const { lines, ok } = regression(c);
    if (!ok) return;
    const [mid, up, down] = lines as [[Point, Point], [Point, Point], [Point, Point]];
    fill(c, [up[0], up[1], down[1], down[0]]);
    stroke(c, mid, c.item.style.color, c.item.style.width, "dashed");
    stroke(c, up);
    stroke(c, down);
  },
  hit(c, p) {
    const { lines } = regression(c);
    let best: number | null = null;
    for (const [a, b] of lines) {
      const d = distToSegment(p, a, b);
      if (d <= HIT) best = Math.min(best ?? d, d);
    }
    return best;
  },
  // handles sit on the regression line ends, at the anchors' times
  handles(c) {
    const { lines } = regression(c);
    return lines[0] ? [lines[0][0], lines[0][1]] : c.points;
  },
  dragHandle(c, index, p) {
    const pts = [...c.points];
    pts[index] = { x: p.x, y: pts[index]!.y };
    return pts;
  },
};

// ---------------------------------------------------------------------------

export const flatTopBottom: Tool = {
  type: "flat_top_bottom",
  label: "平頂/平底",
  group: "channels",
  create: 3,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.12 },
  caps: { fill: true },
  render(c) {
    const [a, b, f] = c.points as [Point, Point, Point];
    const flat: [Point, Point] = [{ x: a.x, y: f.y }, { x: b.x, y: f.y }];
    fill(c, [a, b, flat[1], flat[0]]);
    stroke(c, [a, b]);
    stroke(c, flat);
  },
  hit(c, p) {
    const [a, b, f] = c.points as [Point, Point, Point];
    const d = Math.min(distToSegment(p, a, b), distToSegment(p, { x: a.x, y: f.y }, { x: b.x, y: f.y }));
    return d <= HIT ? d : null;
  },
};

export const disjointChannel: Tool = {
  type: "disjoint_channel",
  label: "不相交通道",
  group: "channels",
  create: 3,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.12 },
  caps: { fill: true },
  render(c) {
    const [a, b, d] = c.points as [Point, Point, Point];
    // second line starts at the third point with the mirrored slope
    const e = { x: b.x, y: d.y - (b.y - a.y) };
    const second: [Point, Point] = [{ x: a.x, y: d.y }, e];
    fill(c, [a, b, second[1], second[0]]);
    stroke(c, [a, b]);
    stroke(c, second);
  },
  hit(c, p) {
    const [a, b, d] = c.points as [Point, Point, Point];
    const e = { x: b.x, y: d.y - (b.y - a.y) };
    const dd = Math.min(distToSegment(p, a, b), distToSegment(p, { x: a.x, y: d.y }, e));
    return dd <= HIT ? dd : null;
  },
};

// ---------------------------------------------------------------------------
// Andrews' pitchfork: median from the pivot through the midpoint of the other
// two points, with parallel tines through them, all rays to the right.
// ---------------------------------------------------------------------------

function pitchforkLines(c: ToolContext): [Point, Point][] {
  const [p0, p1, p2] = c.points as [Point, Point, Point];
  const m = midpoint(p1, p2);
  const dx = m.x - p0.x;
  const dy = m.y - p0.y;
  const len = Math.max(1, Math.hypot(dx, dy));
  const far = ((c.rect.width + c.rect.height) * 4) / len;
  const out = (from: Point): [Point, Point] => [from, { x: from.x + dx * far, y: from.y + dy * far }];
  return [out(p0), out(p1), out(p2), [p1, p2]];
}

export const pitchfork: Tool = {
  type: "pitchfork",
  label: "安德魯斯叉",
  group: "channels",
  create: 3,
  defaults: { fillColor: "#2962ff", fillOpacity: 0.08 },
  caps: { fill: true },
  render(c) {
    const [median, upper, lower, handle] = pitchforkLines(c) as [[Point, Point], [Point, Point], [Point, Point], [Point, Point]];
    fill(c, [upper[0], upper[1], lower[1], lower[0]]);
    stroke(c, median);
    stroke(c, upper);
    stroke(c, lower);
    stroke(c, handle);
  },
  hit(c, p) {
    let best: number | null = null;
    for (const [a, b] of pitchforkLines(c)) {
      const d = distToSegment(p, a, b);
      if (d <= HIT) best = Math.min(best ?? d, d);
    }
    return best;
  },
};

export const CHANNEL_TOOLS: Tool[] = [parallelChannel, regressionTrend, flatTopBottom, disjointChannel, pitchfork];
