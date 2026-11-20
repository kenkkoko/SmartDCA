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
import { readable, withAlpha } from "../geometry";
import { HIT, label, stroke, type Tool, type ToolContext } from "../tool";
import type { Anchor, VfColorKey, VfColors, VfSettings } from "../types";
import { computeProfile } from "./volume";

export type { VfColorKey, VfColors, VfSettings };

export const DEFAULT_VF: VfSettings = { rows: 200, valueArea: 70, hvnPct: 60, showHvn: true, showExt: true, tol: 0.8 };

/** Same as the Pine inputs: color.new(c, transparency) -> opacity = 1 - transparency. */
export const DEFAULT_VF_COLORS: VfColors = {
  fib: { color: "#787b86", opacity: 0.7 },
  pocket: { color: "#ffeb3b", opacity: 0.2 },
  volume: { color: "#ffffff", opacity: 1 },
  hvn: { color: "#00bcd4", opacity: 0.6 },
  zone: { color: "#f57c00", opacity: 0.45 },
  anchor: { color: "#ffeb3b", opacity: 0.7 },
};

export const VF_COLOR_LABELS: [VfColorKey, string][] = [
  ["fib", "斐波線"],
  ["pocket", "黃金口袋底色"],
  ["volume", "POC / VAH / VAL 線"],
  ["hvn", "HVN"],
  ["zone", "匯合帶"],
  ["anchor", "錨點垂直線"],
];

export function vfColorsOf(s: VfSettings): VfColors {
  const out = {} as VfColors;
  for (const [k] of VF_COLOR_LABELS) out[k] = { ...DEFAULT_VF_COLORS[k], ...s.colors?.[k] };
  return out;
}

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
  return { ...DEFAULT_VF, ...style.vf, colors: { ...style.vf?.colors } };
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
    const s = vfSettingsOf(c.item.style);
    const col = vfColorsOf(s);
    const paint = (k: VfColorKey, readableOn = true) => withAlpha(readableOn ? readable(col[k].color, bg) : col[k].color, col[k].opacity);
    const fibC = paint("fib");
    const volC = paint("volume");
    const hvnC = paint("hvn");
    // square box with a thin border, like Pine's box.new
    const box = (hi: number, lo: number, k: VfColorKey, borderAlpha: number) => {
      const y0 = Math.min(c.toY(hi), c.toY(lo));
      const y1 = Math.max(c.toY(hi), c.toY(lo));
      c.s.fillRect(x1, y0, right - x1, y1 - y0, withAlpha(col[k].color, col[k].opacity));
      const edge = withAlpha(col[k].color, borderAlpha);
      stroke(c, [{ x: x1, y: y0 }, { x: right, y: y0 }], edge, 1, "solid");
      stroke(c, [{ x: x1, y: y1 }, { x: right, y: y1 }], edge, 1, "solid");
      stroke(c, [{ x: x1, y: y0 }, { x: x1, y: y1 }], edge, 1, "solid");
      stroke(c, [{ x: right, y: y0 }, { x: right, y: y1 }], edge, 1, "solid");
    };

    // anchors: the two lines are what you grab to select / move the tool
    const anchorC = paint("anchor", false);
    const aw = c.selected ? 1.5 : 1;
    stroke(c, [{ x: x0, y: top }, { x: x0, y: bottom }], anchorC, aw, "dashed");
    stroke(c, [{ x: x1, y: top }, { x: x1, y: bottom }], anchorC, aw, "dashed");

    // golden pocket 0.618-0.65
    box(r.pocket.hi, r.pocket.lo, "pocket", 0.6);

    // confluence zones: plain box, text inside on the left (Pine's box text)
    for (const z of r.zones) {
      box(z.hi, z.lo, "zone", 0.8);
      label(c, `Fib ${z.fib} ✕ ${hitsText(z.hits)}`, x1 + 6, c.toY(z.price), { size, color: "#ffffff" });
    }

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
      label(c, `${name} ${c.fmt(p)}`, right - 2, y - 2, { size, color: textFor(readable(col.volume.color, bg)), bg: volC, align: "right", baseline: "bottom", pad: 3 });
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
  // only the two anchor lines are grabbable, so dragging inside the range
  // still pans the chart
  hit(c, p) {
    const r = resultFor(c);
    if (!r) return null;
    const { x0, x1 } = xs(c, r);
    const d = Math.min(Math.abs(p.x - x0), Math.abs(p.x - x1));
    return d <= HIT ? d : null;
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
