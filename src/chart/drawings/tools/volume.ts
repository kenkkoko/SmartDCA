import type { Point } from "../../engine/types";
import { distToPolyline, insidePolygon, withAlpha } from "../geometry";
import { HIT, label, stroke, type Tool, type ToolContext } from "../tool";
import { DEFAULT_VP_COLORS, type Anchor } from "../types";

export type VolumeProfile = {
  rows: { lo: number; hi: number; up: number; down: number }[];
  poc: number;
  vah: number;
  val: number;
  from: number;
  to: number;
};

/** Volume per price row over bars [from, to]; each bar's volume is spread evenly over its range. */
export function computeProfile(
  bars: { high: number; low: number; open: number; close: number; volume?: number }[],
  from: number,
  to: number,
  rowCount: number,
  valueAreaPct: number,
): VolumeProfile | null {
  const i0 = Math.max(0, Math.min(from, to));
  const i1 = Math.min(bars.length - 1, Math.max(from, to));
  if (i1 < i0) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = i0; i <= i1; i++) {
    lo = Math.min(lo, bars[i]!.low);
    hi = Math.max(hi, bars[i]!.high);
  }
  if (!(hi > lo)) return null;
  const n = Math.max(1, Math.min(2000, Math.round(rowCount)));
  const h = (hi - lo) / n;
  const rows = Array.from({ length: n }, (_, k) => ({ lo: lo + k * h, hi: lo + (k + 1) * h, up: 0, down: 0 }));
  for (let i = i0; i <= i1; i++) {
    const bar = bars[i]!;
    const vol = bar.volume ?? 0;
    if (vol <= 0) continue;
    const span = Math.max(bar.high - bar.low, h * 1e-6);
    const k0 = Math.max(0, Math.floor((bar.low - lo) / h));
    const k1 = Math.min(n - 1, Math.floor((bar.high - lo) / h));
    for (let k = k0; k <= k1; k++) {
      const row = rows[k]!;
      const overlap = Math.min(row.hi, bar.high) - Math.max(row.lo, bar.low);
      const share = bar.high === bar.low ? 1 / (k1 - k0 + 1) : Math.max(0, overlap) / span;
      if (bar.close >= bar.open) row.up += vol * share;
      else row.down += vol * share;
    }
  }
  const total = rows.reduce((s, r) => s + r.up + r.down, 0);
  let poc = 0;
  rows.forEach((r, k) => {
    if (r.up + r.down > rows[poc]!.up + rows[poc]!.down) poc = k;
  });
  // value area: grow from the POC towards the heavier neighbour
  let a = poc;
  let b = poc;
  let acc = rows[poc]!.up + rows[poc]!.down;
  const target = (total * valueAreaPct) / 100;
  while (acc < target && (a > 0 || b < n - 1)) {
    const below = a > 0 ? rows[a - 1]!.up + rows[a - 1]!.down : -1;
    const above = b < n - 1 ? rows[b + 1]!.up + rows[b + 1]!.down : -1;
    if (above >= below) acc += rows[++b]!.up + rows[b]!.down;
    else acc += rows[--a]!.up + rows[a]!.down;
  }
  return { rows, poc, vah: b, val: a, from: i0, to: i1 };
}

function profileFor(c: ToolContext): VolumeProfile | null {
  const [a, b] = c.item.anchors as [Anchor, Anchor];
  if (!a || !b) return null;
  const t = c.chart.time;
  return computeProfile(c.bars, Math.round(t.timeToIndex(a.time)), Math.round(t.timeToIndex(b.time)), c.item.style.rows, c.item.style.valueArea);
}

function bounds(c: ToolContext): { x0: number; x1: number } {
  const [a, b] = c.points as [Point, Point];
  return { x0: Math.min(a.x, b.x), x1: Math.max(a.x, b.x) };
}

export const volumeProfile: Tool = {
  type: "volume_profile",
  label: "固定範圍成交量分布",
  group: "volume",
  create: 2,
  caps: { line: false, dialog: "volume" },
  render(c) {
    const vp = profileFor(c);
    const { x0, x1 } = bounds(c);
    if (!vp) return;
    const t = c.theme;
    const col = { ...DEFAULT_VP_COLORS, ...c.item.style.vp };
    const upC = col.up ?? t.up;
    const downC = col.down ?? t.down;
    const width = Math.max(10, (x1 - x0) * 0.35);
    const max = Math.max(...vp.rows.map((r) => r.up + r.down), 1);
    // range box
    const top = c.toY(vp.rows[vp.rows.length - 1]!.hi);
    const bottom = c.toY(vp.rows[0]!.lo);
    if (col.box && col.boxOpacity > 0) c.s.fillPolygon([{ x: x0, y: top }, { x: x1, y: top }, { x: x1, y: bottom }, { x: x0, y: bottom }], withAlpha(col.box, col.boxOpacity));
    vp.rows.forEach((row, k) => {
      const inVA = c.item.style.showValueArea && k >= vp.val && k <= vp.vah;
      const y0 = c.toY(row.hi);
      const y1 = c.toY(row.lo);
      const h = Math.max(0.5, y1 - y0 - (vp.rows.length < 120 ? 1 : 0));
      const wUp = (row.up / max) * width;
      const wDown = (row.down / max) * width;
      if (wUp > 0) c.s.fillRect(x0, y0, wUp, h, withAlpha(upC, inVA ? col.inOpacity : col.outOpacity));
      if (wDown > 0) c.s.fillRect(x0 + wUp, y0, wDown, h, withAlpha(downC, inVA ? col.inOpacity : col.outOpacity));
    });
    const mid = (k: number) => c.toY((vp.rows[k]!.lo + vp.rows[k]!.hi) / 2);
    if (c.item.style.showPoc) {
      stroke(c, [{ x: x0, y: mid(vp.poc) }, { x: x1, y: mid(vp.poc) }], col.poc, 1.5, "solid");
      label(c, "POC", x1 + 4, mid(vp.poc), { size: 10, color: col.poc });
    }
    if (c.item.style.showValueArea) {
      stroke(c, [{ x: x0, y: c.toY(vp.rows[vp.vah]!.hi) }, { x: x1, y: c.toY(vp.rows[vp.vah]!.hi) }], col.valueArea, 1, "dashed");
      stroke(c, [{ x: x0, y: c.toY(vp.rows[vp.val]!.lo) }, { x: x1, y: c.toY(vp.rows[vp.val]!.lo) }], col.valueArea, 1, "dashed");
      label(c, "VAH", x1 + 4, c.toY(vp.rows[vp.vah]!.hi), { size: 10, color: col.valueArea });
      label(c, "VAL", x1 + 4, c.toY(vp.rows[vp.val]!.lo), { size: 10, color: col.valueArea });
    }
  },
  hit(c, p) {
    const vp = profileFor(c);
    const { x0, x1 } = bounds(c);
    if (!vp) return null;
    const top = c.toY(vp.rows[vp.rows.length - 1]!.hi);
    const bottom = c.toY(vp.rows[0]!.lo);
    return insidePolygon(p, [{ x: x0, y: top }, { x: x1, y: top }, { x: x1, y: bottom }, { x: x0, y: bottom }]) ? HIT - 1 : null;
  },
  // handles at the range's top-left and bottom-right, prices are derived
  handles(c) {
    const vp = profileFor(c);
    const { x0, x1 } = bounds(c);
    if (!vp) return c.points;
    return [{ x: x0, y: c.toY(vp.rows[vp.rows.length - 1]!.hi) }, { x: x1, y: c.toY(vp.rows[0]!.lo) }];
  },
  dragHandle(c, index, p) {
    const pts = [...c.points];
    const [a, b] = pts as [Point, Point];
    const leftIsA = a.x <= b.x;
    const target = index === 0 ? (leftIsA ? 0 : 1) : leftIsA ? 1 : 0;
    pts[target] = { x: p.x, y: pts[target]!.y };
    return pts;
  },
  axisTags: (c) => ({ prices: [], times: c.item.anchors.map((a) => a.time) }),
};

// ---------------------------------------------------------------------------

function vwapPoints(c: ToolContext): Point[] {
  const a = c.item.anchors[0];
  if (!a) return [];
  const t = c.chart.time;
  const start = Math.max(0, Math.round(t.timeToIndex(a.time)));
  const out: Point[] = [];
  let pv = 0;
  let vol = 0;
  for (let i = start; i < c.bars.length; i++) {
    const bar = c.bars[i]!;
    const v = bar.volume ?? 0;
    pv += ((bar.high + bar.low + bar.close) / 3) * v;
    vol += v;
    if (vol > 0) out.push({ x: t.indexToX(i), y: c.toY(pv / vol) });
  }
  return out;
}

export const anchoredVwap: Tool = {
  type: "anchored_vwap",
  label: "錨定 VWAP",
  group: "volume",
  create: 1,
  defaults: { color: "#f5b942", width: 1.5 },
  render(c) {
    const pts = vwapPoints(c);
    stroke(c, pts);
    const last = pts[pts.length - 1];
    if (last) label(c, "VWAP", last.x + 4, last.y, { size: 10, color: c.item.style.color });
  },
  hit(c, p) {
    const d = distToPolyline(p, vwapPoints(c));
    return d <= HIT ? d : null;
  },
  handles(c) {
    const pts = vwapPoints(c);
    return pts[0] ? [pts[0]] : c.points;
  },
  dragHandle(_c, _i, p) {
    return [p];
  },
  axisTags: (c) => ({ prices: [], times: [c.item.anchors[0]!.time] }),
};

export const VOLUME_TOOLS: Tool[] = [volumeProfile, anchoredVwap];
