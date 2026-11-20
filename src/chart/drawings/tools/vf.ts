// V&F 量斐匯合 (Volume & Fibonacci confluence), Kai's own indicator, ported
// from his Pine Script "Combined Fib × Volume".
//
// Two anchors set a time range. Inside it:
//  1) the highest high and lowest low give the swing and its direction
//     (high before low = down leg, so the retracement levels sit above the low);
//  2) Fibonacci retracement 0-1 (with the 0.618-0.65 golden pocket) and the
//     1.272 / 1.618 extensions past the swing's end;
//  3) a volume profile of the same range: POC / VAH / VAL and HVNs
//     (local peaks with at least hvnPct of the POC's volume);
//  4) wherever a Fibonacci level and a volume level are within ±tol %, a
//     confluence zone is drawn and labelled with its members.

import type { Point } from "../../engine/types";
import { insidePolygon, readable, withAlpha } from "../geometry";
import { HIT, label, stroke, type Tool, type ToolContext } from "../tool";
import type { Anchor, VfSettings } from "../types";
import { computeProfile } from "./volume";

export type { VfSettings };

export const DEFAULT_VF: VfSettings = { rows: 200, valueArea: 70, hvnPct: 60, showHvn: true, showExt: true, tol: 0.8 };

export const VF_COLORS = {
  fib: "#9598a1",
  pocket: "#f5c518",
  volume: "#ffffff",
  hvn: "#26c6da",
  zone: "#f57c00",
  anchor: "#f5c518",
};

const RETRACE = [0, 0.382, 0.5, 0.618, 0.65, 0.786, 1];
const EXTENSIONS: [number, string][] = [
  [0.272, "1.272"],
  [0.618, "1.618"],
];

export type VfLevel = { name: string; price: number; ext?: boolean };
export type VfZone = { fib: string; price: number; hi: number; lo: number; hits: string[] };
export type VfResult = {
  from: number;
  to: number;
  swingHi: number;
  swingLo: number;
  isDown: boolean;
  fibs: VfLevel[];
  pocket: { hi: number; lo: number };
  poc: number;
  vah: number;
  val: number;
  hvns: number[];
  zones: VfZone[];
  /** Index into zones of the one with the most members (-1 = none). */
  best: number;
};

export function vfSettingsOf(style: { vf?: Partial<VfSettings> }): VfSettings {
  return { ...DEFAULT_VF, ...style.vf };
}

export function computeVf(
  bars: { high: number; low: number; open: number; close: number; volume?: number }[],
  from: number,
  to: number,
  s: VfSettings,
): VfResult | null {
  const vp = computeProfile(bars, from, to, s.rows, s.valueArea);
  if (!vp) return null;
  let hiIdx = vp.from;
  let loIdx = vp.from;
  for (let i = vp.from; i <= vp.to; i++) {
    if (bars[i]!.high > bars[hiIdx]!.high) hiIdx = i;
    if (bars[i]!.low < bars[loIdx]!.low) loIdx = i;
  }
  const swingHi = bars[hiIdx]!.high;
  const swingLo = bars[loIdx]!.low;
  const rng = swingHi - swingLo;
  const isDown = hiIdx < loIdx;

  const fibs: VfLevel[] = RETRACE.map((r) => ({ name: String(r), price: isDown ? swingLo + r * rng : swingHi - r * rng }));
  if (s.showExt) for (const [e, name] of EXTENSIONS) fibs.push({ name, price: isDown ? swingLo - e * rng : swingHi + e * rng, ext: true });
  const p618 = fibs[3]!.price;
  const p650 = fibs[4]!.price;

  const mid = (k: number) => (vp.rows[k]!.lo + vp.rows[k]!.hi) / 2;
  const vol = (k: number) => vp.rows[k]!.up + vp.rows[k]!.down;
  const poc = mid(vp.poc);
  const vah = vp.rows[vp.vah]!.hi;
  const val = vp.rows[vp.val]!.lo;
  const maxV = vol(vp.poc);
  const hvns: number[] = [];
  for (let k = 1; k < vp.rows.length - 1 && hvns.length < 8; k++) {
    const v = vol(k);
    if (v >= (maxV * s.hvnPct) / 100 && v > vol(k - 1) && v > vol(k + 1)) {
      const p = mid(k);
      if ((Math.abs(p - poc) / poc) * 100 > s.tol) hvns.push(p);
    }
  }

  const volLevels: [number, string][] = [
    [poc, "POC"],
    [vah, "VAH"],
    [val, "VAL"],
    ...hvns.map((p) => [p, "HVN"] as [number, string]),
  ];
  const zones: VfZone[] = [];
  let best = -1;
  for (const f of fibs) {
    const hits = volLevels.filter(([p]) => (Math.abs(p - f.price) / f.price) * 100 <= s.tol).map(([, n]) => n);
    if (!hits.length) continue;
    zones.push({ fib: f.name, price: f.price, hi: f.price * (1 + s.tol / 100), lo: f.price * (1 - s.tol / 100), hits });
    if (best < 0 || hits.length > zones[best]!.hits.length) best = zones.length - 1;
  }
  return { from: vp.from, to: vp.to, swingHi, swingLo, isDown, fibs, pocket: { hi: Math.max(p618, p650), lo: Math.min(p618, p650) }, poc, vah, val, hvns, zones, best };
}

function resultFor(c: ToolContext): VfResult | null {
  const [a, b] = c.item.anchors as [Anchor, Anchor];
  if (!a || !b) return null;
  const t = c.chart.time;
  return computeVf(c.bars, Math.round(t.timeToIndex(a.time)), Math.round(t.timeToIndex(b.time)), vfSettingsOf(c.item.style));
}

function xs(c: ToolContext, r: VfResult): { x0: number; x1: number; right: number } {
  const t = c.chart.time;
  const x0 = t.indexToX(r.from);
  const x1 = t.indexToX(r.to);
  // levels run 8 bars past the last candle, like the Pine version
  const right = Math.min(c.rect.x + c.rect.width - 4, Math.max(x1 + 180, t.indexToX(c.bars.length - 1 + 8)));
  return { x0, x1, right };
}

export const vfConfluence: Tool = {
  type: "vf_confluence",
  label: "V&F 量斐匯合",
  group: "volume",
  create: 2,
  caps: { line: false, dialog: "vf", axisLabel: false },
  init(item) {
    item.style.vf = { ...DEFAULT_VF };
  },
  render(c) {
    const r = resultFor(c);
    if (!r) return;
    const bg = c.theme.background;
    const { x0, x1, right } = xs(c, r);
    const top = c.rect.y;
    const bottom = c.rect.y + c.rect.height;
    const size = 10;
    const fibC = readable(VF_COLORS.fib, bg);
    const volC = readable(VF_COLORS.volume, bg);
    const hvnC = readable(VF_COLORS.hvn, bg);
    const s = vfSettingsOf(c.item.style);
    const box = (hi: number, lo: number, fillC: string, a: number, from = x1) => {
      const y0 = c.toY(hi);
      const y1 = c.toY(lo);
      c.s.fillRect(from, Math.min(y0, y1), right - from, Math.abs(y1 - y0), withAlpha(fillC, a));
    };

    // anchors
    stroke(c, [{ x: x0, y: top }, { x: x0, y: bottom }], withAlpha(VF_COLORS.anchor, 0.7), 1, "dashed");
    stroke(c, [{ x: x1, y: top }, { x: x1, y: bottom }], withAlpha(VF_COLORS.anchor, 0.7), 1, "dashed");

    // golden pocket 0.618-0.65
    box(r.pocket.hi, r.pocket.lo, VF_COLORS.pocket, 0.2);

    // confluence zones (under the lines so the levels stay visible)
    r.zones.forEach((z, i) => {
      box(z.hi, z.lo, VF_COLORS.zone, i === r.best ? 0.38 : 0.24);
      label(c, `Fib ${z.fib} ✕ ${hitsText(z.hits)}`, x1 + 6, c.toY(z.price), { size, color: "#ffffff", bg: withAlpha(VF_COLORS.zone, 0.85), bold: i === r.best });
    });

    // Fibonacci levels
    for (const f of r.fibs) {
      const y = c.toY(f.price);
      stroke(c, [{ x: x1, y }, { x: right, y }], fibC, 1, f.ext ? "dashed" : "solid");
      label(c, `${f.ext ? f.name : Number(f.name).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}  ${c.fmt(f.price)}`, right - 2, y - 1, { size, color: fibC, align: "right", baseline: "bottom" });
    }

    // volume levels
    const vline = (p: number, name: string, w: number) => {
      const y = c.toY(p);
      stroke(c, [{ x: x0, y }, { x: right, y }], volC, w, "solid");
      label(c, `${name} ${c.fmt(p)}`, right - 2, y - 2, { size, color: textFor(volC), bg: volC, align: "right", baseline: "bottom", pad: 3 });
    };
    vline(r.poc, "POC", 2);
    vline(r.vah, "VAH", 1);
    vline(r.val, "VAL", 1);
    if (s.showHvn)
      for (const p of r.hvns) {
        const y = c.toY(p);
        stroke(c, [{ x: x1, y }, { x: right, y }], hvnC, 1, "dotted");
        label(c, `HVN ${c.fmt(p)}`, right - 2, y - 1, { size: 9, color: hvnC, align: "right", baseline: "bottom" });
      }
  },
  hit(c, p) {
    const r = resultFor(c);
    if (!r) return null;
    const { x0, right } = xs(c, r);
    const yTop = c.toY(Math.max(r.swingHi, ...r.fibs.map((f) => f.price)));
    const yBot = c.toY(Math.min(r.swingLo, ...r.fibs.map((f) => f.price)));
    const poly: Point[] = [
      { x: x0, y: yTop },
      { x: right, y: yTop },
      { x: right, y: yBot },
      { x: x0, y: yBot },
    ];
    if (!insidePolygon(p, poly)) return null;
    // close to a line counts as a direct hit, anywhere else inside as a weak one
    const levels = [r.poc, r.vah, r.val, ...r.fibs.map((f) => f.price)];
    const d = Math.min(...levels.map((v) => Math.abs(c.toY(v) - p.y)));
    return d <= HIT ? d : HIT - 1;
  },
  // one handle per anchor, on the swing's high and low
  handles(c) {
    const r = resultFor(c);
    if (!r) return c.points;
    const { x0, x1 } = xs(c, r);
    return [
      { x: x0, y: c.toY(r.swingHi) },
      { x: x1, y: c.toY(r.swingLo) },
    ];
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

function textFor(bg: string): string {
  const m = /^#?([0-9a-f]{6})/i.exec(bg);
  if (!m) return "#000000";
  const n = parseInt(m[1]!, 16);
  const lum = 0.299 * (n >> 16) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255);
  return lum > 150 ? "#000000" : "#ffffff";
}

/** "POC + HVN×3" instead of repeating the same member. */
export function hitsText(hits: string[]): string {
  const counts = new Map<string, number>();
  for (const h of hits) counts.set(h, (counts.get(h) ?? 0) + 1);
  return [...counts].map(([n, k]) => (k > 1 ? `${n}×${k}` : n)).join(" + ");
}
