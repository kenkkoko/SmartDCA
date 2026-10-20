// Numbers the AI should not have to work out itself: swing points, market
// structure, volume-profile shape, Fibonacci levels, anchored VWAPs and
// unusual volume bars. The model reads these and makes the judgement calls
// (phase, wave count, pattern); code also checks its wave count afterwards.

import { computeProfile } from "../drawings/tools/volume";
import type { Bar } from "../engine/types";
import { sma } from "../indicators";

export type Pivot = { index: number; time: number; price: number; kind: "H" | "L" };

/** ZigZag: alternating highs and lows that move at least `pct` (0.1 = 10%). */
export function zigzag(bars: Bar[], pct: number, from = 0, to = bars.length - 1): Pivot[] {
  const out: Pivot[] = [];
  if (to - from < 2) return out;
  let dir: 0 | 1 | -1 = 0;
  let hi = from;
  let lo = from;
  let ext = from;
  for (let i = from + 1; i <= to; i++) {
    const b = bars[i]!;
    if (dir === 0) {
      if (b.high > bars[hi]!.high) hi = i;
      if (b.low < bars[lo]!.low) lo = i;
      if (b.high >= bars[lo]!.low * (1 + pct) && lo < i) {
        out.push(pv(bars, lo, "L"));
        dir = 1;
        ext = i;
      } else if (b.low <= bars[hi]!.high * (1 - pct) && hi < i) {
        out.push(pv(bars, hi, "H"));
        dir = -1;
        ext = i;
      }
    } else if (dir === 1) {
      if (b.high > bars[ext]!.high) ext = i;
      else if (b.low <= bars[ext]!.high * (1 - pct)) {
        out.push(pv(bars, ext, "H"));
        dir = -1;
        ext = i;
      }
    } else {
      if (b.low < bars[ext]!.low) ext = i;
      else if (b.high >= bars[ext]!.low * (1 + pct)) {
        out.push(pv(bars, ext, "L"));
        dir = 1;
        ext = i;
      }
    }
  }
  // the swing that is still running
  if (dir === 1) out.push(pv(bars, ext, "H"));
  else if (dir === -1) out.push(pv(bars, ext, "L"));
  return out;
}

const pv = (bars: Bar[], i: number, kind: "H" | "L"): Pivot => ({ index: i, time: bars[i]!.time, price: kind === "H" ? bars[i]!.high : bars[i]!.low, kind });

/** ZigZag with the threshold picked so the range has about `target` pivots. */
export function swings(bars: Bar[], target: number, from = 0, to = bars.length - 1): { pivots: Pivot[]; pct: number } {
  let lo = 0.005;
  let hi = 0.8;
  let best = { pivots: zigzag(bars, 0.1, from, to), pct: 0.1 };
  for (let k = 0; k < 18; k++) {
    const mid = Math.sqrt(lo * hi);
    const p = zigzag(bars, mid, from, to);
    if (Math.abs(p.length - target) < Math.abs(best.pivots.length - target)) best = { pivots: p, pct: mid };
    if (p.length > target) lo = mid;
    else hi = mid;
  }
  return best;
}

/** HH / HL / LH / LL for each pivot compared with the previous one of its kind. */
export function structureLabels(pivots: Pivot[]): string[] {
  let lastH: Pivot | null = null;
  let lastL: Pivot | null = null;
  return pivots.map((p) => {
    let s = "";
    if (p.kind === "H") {
      s = lastH ? (p.price > lastH.price ? "HH" : "LH") : "H";
      lastH = p;
    } else {
      s = lastL ? (p.price > lastL.price ? "HL" : "LL") : "L";
      lastL = p;
    }
    return s;
  });
}

/** The latest close above a prior swing high (or below a prior swing low). */
export function lastBreak(bars: Bar[], pivots: Pivot[]): { index: number; dir: "up" | "down"; level: Pivot } | null {
  let found: { index: number; dir: "up" | "down"; level: Pivot } | null = null;
  for (let k = 0; k < pivots.length; k++) {
    const p = pivots[k]!;
    const next = pivots[k + 1]?.index ?? bars.length;
    for (let i = p.index + 1; i < bars.length; i++) {
      const c = bars[i]!.close;
      if ((p.kind === "H" && c > p.price) || (p.kind === "L" && c < p.price)) {
        if (i > next && (!found || i > found.index)) found = { index: i, dir: p.kind === "H" ? "up" : "down", level: p };
        break;
      }
    }
  }
  return found;
}

export type ProfileSummary = {
  poc: number;
  vah: number;
  val: number;
  high: number;
  low: number;
  /** P = volume piled at the top, b = at the bottom, D = in the middle. */
  shape: "P" | "b" | "D" | "雙分布";
  /** High-volume nodes (row mid prices), strongest first. */
  hvn: number[];
  /** Low-volume gaps between nodes. */
  lvn: number[];
};

export function profileSummary(bars: Bar[], from: number, to: number, rows = 48): ProfileSummary | null {
  const vp = computeProfile(bars, from, to, rows, 70);
  if (!vp) return null;
  const vol = vp.rows.map((r) => r.up + r.down);
  const mid = (k: number) => (vp.rows[k]!.lo + vp.rows[k]!.hi) / 2;
  const low = vp.rows[0]!.lo;
  const high = vp.rows[vp.rows.length - 1]!.hi;
  // smooth a little so single rows do not count as peaks
  const sm = vol.map((_, k) => (vol[k - 1] ?? vol[k]!) * 0.25 + vol[k]! * 0.5 + (vol[k + 1] ?? vol[k]!) * 0.25);
  const peaks: number[] = [];
  for (let k = 0; k < sm.length; k++) if (sm[k]! >= (sm[k - 1] ?? -1) && sm[k]! > (sm[k + 1] ?? -1)) peaks.push(k);
  peaks.sort((a, b) => sm[b]! - sm[a]!);
  const strong = peaks.filter((k) => sm[k]! >= sm[peaks[0]!]! * 0.5);
  const hvn: number[] = [];
  for (const k of strong) if (hvn.every((h) => Math.abs(h - k) >= rows / 8)) hvn.push(k);
  const lvn: number[] = [];
  let double = false;
  for (let i = 0; i < hvn.length; i++)
    for (let j = i + 1; j < hvn.length; j++) {
      const [a, b] = [Math.min(hvn[i]!, hvn[j]!), Math.max(hvn[i]!, hvn[j]!)];
      let v = a;
      for (let k = a; k <= b; k++) if (sm[k]! < sm[v]!) v = k;
      if (sm[v]! < Math.min(sm[a]!, sm[b]!) * 0.45) {
        lvn.push(mid(v));
        if (i === 0 && j === 1) double = true;
      }
    }
  const pos = (mid(vp.poc) - low) / (high - low);
  const shape = double ? "雙分布" : pos >= 0.62 ? "P" : pos <= 0.38 ? "b" : "D";
  return { poc: mid(vp.poc), vah: vp.rows[vp.vah]!.hi, val: vp.rows[vp.val]!.lo, high, low, shape, hvn: hvn.slice(0, 4).map(mid), lvn: [...new Set(lvn)].slice(0, 3) };
}

/** Anchored VWAP from bar `from` to the last bar. */
export function anchoredVwap(bars: Bar[], from: number, to = bars.length - 1): number | null {
  let pv = 0;
  let v = 0;
  for (let i = from; i <= to; i++) {
    const b = bars[i]!;
    pv += ((b.high + b.low + b.close) / 3) * (b.volume ?? 0);
    v += b.volume ?? 0;
  }
  return v > 0 ? pv / v : null;
}

export const FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786];
export const FIB_EXT = [1.272, 1.618];

/** Retracement levels of the move a → b (0 at b, 1 at a), plus extensions beyond b. */
export function fibFrom(a: number, b: number): { ratio: number; price: number }[] {
  return [...FIB_RATIOS.map((r) => ({ ratio: r, price: b + (a - b) * r })), ...FIB_EXT.map((r) => ({ ratio: r, price: a + (b - a) * r }))];
}

/** Merge bars into weeks (Monday start, UTC) or calendar months. */
export function aggregate(bars: Bar[], unit: "week" | "month"): Bar[] {
  const key = (t: number) => {
    const d = new Date(t * 1000);
    if (unit === "month") return d.getUTCFullYear() * 12 + d.getUTCMonth();
    return Math.floor((t / 86400 + 3) / 7); // 1970-01-01 was a Thursday
  };
  const out: Bar[] = [];
  let k: number | null = null;
  for (const b of bars) {
    const kb = key(b.time);
    const cur = out[out.length - 1];
    if (cur && kb === k) {
      cur.high = Math.max(cur.high, b.high);
      cur.low = Math.min(cur.low, b.low);
      cur.close = b.close;
      cur.volume = (cur.volume ?? 0) + (b.volume ?? 0);
    } else {
      out.push({ ...b });
      k = kb;
    }
  }
  return out;
}

export type VolumeEvent = { index: number; text: string };

/**
 * Bars worth a second look for Wyckoff events: climactic volume, a break of
 * a swing low / high that closes back inside (spring / upthrust), and quiet
 * tests near a prior climax.
 */
export function volumeEvents(bars: Bar[], pivots: Pivot[], from: number, to: number): VolumeEvent[] {
  const vols = bars.map((b) => b.volume ?? 0);
  if (!vols.some((v) => v > 0)) return [];
  const v20 = sma(vols, 20);
  const ranges = bars.map((b) => b.high - b.low);
  const r20 = sma(ranges, 20);
  const out: VolumeEvent[] = [];
  for (let i = Math.max(from, 20); i <= to; i++) {
    const b = bars[i]!;
    const av = v20[i - 1];
    const ar = r20[i - 1];
    if (!av || !ar) continue;
    const vx = vols[i]! / av;
    const span = b.high - b.low;
    const closePos = span > 0 ? (b.close - b.low) / span : 0.5;
    const notes: string[] = [];
    if (vx >= 2.2 && span >= ar * 1.4) notes.push(`爆量長K（量 ${vx.toFixed(1)} 倍、振幅 ${(span / ar).toFixed(1)} 倍），收在K棒${closePos > 0.66 ? "上緣" : closePos < 0.34 ? "下緣" : "中段"}`);
    else if (vx >= 2.5) notes.push(`爆量（${vx.toFixed(1)} 倍），振幅 ${(span / ar).toFixed(1)} 倍`);
    const prior = pivots.filter((p) => p.index < i - 2 && p.index >= i - 120);
    const lowPv = prior.filter((p) => p.kind === "L").pop();
    const highPv = prior.filter((p) => p.kind === "H").pop();
    if (lowPv && b.low < lowPv.price && b.close > lowPv.price) notes.push(`跌破前低 ${fmtP(lowPv.price)} 後收回（量 ${vx.toFixed(1)} 倍）`);
    if (highPv && b.high > highPv.price && b.close < highPv.price) notes.push(`突破前高 ${fmtP(highPv.price)} 後收回（量 ${vx.toFixed(1)} 倍）`);
    if (notes.length) out.push({ index: i, text: notes.join("；") });
  }
  // keep the strongest few, in time order
  return out.slice(-24);
}

const fmtP = (v: number) => Number(v.toPrecision(6)).toString();

/**
 * Elliott wave hard rules for an impulse 0-1-2-3-4-5 (prices of the six
 * points). Returns the broken rules in Chinese; empty = valid.
 */
export function impulseViolations(p: number[]): string[] {
  if (p.length < 6) return [];
  const up = p[1]! > p[0]!;
  const s = (x: number) => (up ? x : -x);
  const out: string[] = [];
  const w1 = s(p[1]! - p[0]!);
  const w3 = s(p[3]! - p[2]!);
  const w5 = s(p[5]! - p[4]!);
  if (s(p[2]!) <= s(p[0]!)) out.push("2 浪跌破 1 浪起點");
  if (w3 < w1 && w3 < w5) out.push("3 浪是最短的一浪");
  if (s(p[4]!) <= s(p[1]!)) out.push("4 浪進入 1 浪價格區");
  if (s(p[3]!) <= s(p[1]!)) out.push("3 浪沒有超過 1 浪終點");
  return out;
}

export type KeyLevel = {
  price: number;
  /** Where the level started (the first swing point at this price). */
  originTime: number;
  originKind: "H" | "L";
  /** Swing points (weekly + monthly, several degrees) that turned at this price. */
  touches: number;
  /** It has been both a high and a low: support / resistance flipped. */
  flipped: boolean;
  lastTime: number;
  monthly: boolean;
};

/**
 * Major support / resistance from weekly and monthly bars: swing points of
 * several sizes are clustered by price (within `tol`); a cluster scores for
 * each turn, double for monthly turns. The level is drawn from its first turn.
 */
export function keyLevels(weekly: Bar[], monthly: Bar[], tol = 0.03, perSide = 4): KeyLevel[] {
  type P = Pivot & { w: number; m: boolean };
  const pts: P[] = [];
  const add = (set: Bar[], target: number, w: number, m: boolean) => {
    if (set.length < 8) return;
    for (const p of swings(set, target).pivots) pts.push({ ...p, w, m });
  };
  add(monthly, 10, 3, true);
  add(monthly, 20, 2, true);
  add(weekly, 16, 2, false);
  add(weekly, 40, 1, false);
  if (!pts.length) return [];
  // the running swing (last pivot) is not a level yet
  const lastT = Math.max(weekly[weekly.length - 1]?.time ?? 0, monthly[monthly.length - 1]?.time ?? 0);
  pts.sort((a, b) => a.price - b.price);
  const clusters: P[][] = [];
  for (const p of pts) {
    const c = clusters[clusters.length - 1];
    if (c && p.price <= c[0]!.price * (1 + tol * 2)) c.push(p);
    else clusters.push([p]);
  }
  const levels = clusters.map((c) => {
    // the same turn seen at several sizes counts once (by time, ~a month apart)
    const turns: P[] = [];
    for (const p of [...c].sort((a, b) => b.w - a.w)) if (turns.every((q) => Math.abs(q.time - p.time) > 25 * 86400)) turns.push(p);
    const origin = [...turns].sort((a, b) => a.time - b.time)[0]!;
    const score = turns.reduce((s, p) => s + p.w, 0);
    return {
      level: {
        price: origin.price,
        originTime: origin.time,
        originKind: origin.kind,
        touches: turns.length,
        flipped: turns.some((p) => p.kind === "H") && turns.some((p) => p.kind === "L"),
        lastTime: Math.max(...turns.map((p) => p.time)),
        monthly: turns.some((p) => p.m),
      } satisfies KeyLevel,
      score: score + (turns.some((p) => p.kind === "H") && turns.some((p) => p.kind === "L") ? 2 : 0),
      onlyRunning: turns.length === 1 && Math.abs(origin.time - lastT) < 40 * 86400,
    };
  });
  // the strongest few on each side of the price; far-away levels count less
  const close = (weekly[weekly.length - 1] ?? monthly[monthly.length - 1])!.close;
  const ranked = levels
    .filter((l) => !l.onlyRunning)
    .map((l) => ({ ...l, rank: l.score - 4 * Math.abs(Math.log(l.level.price / close)) }))
    .sort((a, b) => b.rank - a.rank);
  const above = ranked.filter((l) => l.level.price > close).slice(0, perSide);
  const below = ranked.filter((l) => l.level.price <= close).slice(0, perSide);
  return [...above, ...below].map((l) => l.level).sort((a, b) => b.price - a.price);
}

// ---------------------------------------------------------------------------
// volume-price analysis and candlestick patterns
// ---------------------------------------------------------------------------

/** Candlestick patterns completed on bar i (needs a few bars before it). */
export function candlePatterns(bars: Bar[], i: number): string[] {
  const b = bars[i];
  const p = bars[i - 1];
  const q = bars[i - 2];
  if (!b || !p) return [];
  const body = (x: Bar) => Math.abs(x.close - x.open);
  const span = (x: Bar) => x.high - x.low || 1e-12;
  const upper = (x: Bar) => x.high - Math.max(x.open, x.close);
  const lower = (x: Bar) => Math.min(x.open, x.close) - x.low;
  const bull = (x: Bar) => x.close > x.open;
  const bear = (x: Bar) => x.close < x.open;
  // recent direction: close vs 5 bars ago
  const back = bars[i - 5];
  const fell = back ? p.close < back.close : false;
  const rose = back ? p.close > back.close : false;
  const out: string[] = [];
  if (lower(b) >= Math.max(body(b) * 2, span(b) * 0.6) && upper(b) <= span(b) * 0.15) out.push(fell ? "錘子線" : "上吊線");
  else if (upper(b) >= Math.max(body(b) * 2, span(b) * 0.6) && lower(b) <= span(b) * 0.15) out.push(rose ? "流星線" : "倒錘子");
  else if (body(b) <= span(b) * 0.1) out.push("十字星");
  if (bull(b) && bear(p) && b.close >= p.open && b.open <= p.close && body(b) > body(p)) out.push("看漲吞噬");
  if (bear(b) && bull(p) && b.close <= p.open && b.open >= p.close && body(b) > body(p)) out.push("看跌吞噬");
  if (b.high < p.high && b.low > p.low) out.push("孕線（內包）");
  if (q && bear(q) && body(p) <= span(p) * 0.3 && bull(b) && b.close > (q.open + q.close) / 2 && fell) out.push("晨星");
  if (q && bull(q) && body(p) <= span(p) * 0.3 && bear(b) && b.close < (q.open + q.close) / 2 && rose) out.push("夜星");
  if (bear(p) && bull(b) && b.open < p.low && b.close > (p.open + p.close) / 2 && b.close < p.open) out.push("貫穿線");
  if (bull(p) && bear(b) && b.open > p.high && b.close < (p.open + p.close) / 2 && b.close > p.open) out.push("烏雲蓋頂");
  return out;
}

export type Leg = { from: Pivot; to: Pivot; bars: number; change: number; volRatio: number; spreadRatio: number };

/** Each swing leg: how far it went, how many bars, average volume / spread vs the whole range. */
export function legs(bars: Bar[], pivots: Pivot[], from: number, to: number): Leg[] {
  let vSum = 0;
  let sSum = 0;
  for (let i = from; i <= to; i++) {
    vSum += bars[i]!.volume ?? 0;
    sSum += bars[i]!.high - bars[i]!.low;
  }
  const n = to - from + 1;
  const vAvg = vSum / n;
  const sAvg = sSum / n;
  const out: Leg[] = [];
  for (let k = 1; k < pivots.length; k++) {
    const a = pivots[k - 1]!;
    const b = pivots[k]!;
    let v = 0;
    let s = 0;
    for (let i = a.index + 1; i <= b.index; i++) {
      v += bars[i]!.volume ?? 0;
      s += bars[i]!.high - bars[i]!.low;
    }
    const m = Math.max(1, b.index - a.index);
    out.push({ from: a, to: b, bars: m, change: b.price / a.price - 1, volRatio: vAvg ? v / m / vAvg : 0, spreadRatio: sAvg ? s / m / sAvg : 0 });
  }
  return out;
}

/**
 * Volume-price notes a Wyckoff reader looks for: divergence at new highs /
 * lows, effort vs result, no demand / no supply bars near the end.
 */
export function volumePriceNotes(bars: Bar[], pivots: Pivot[], from: number, to: number): string[] {
  const ls = legs(bars, pivots, from, to);
  const out: string[] = [];
  const day = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);
  // divergence: a new extreme reached on a lighter leg than the previous push
  for (let k = 2; k < ls.length; k++) {
    const cur = ls[k]!;
    const prev = ls[k - 2]!;
    if (cur.to.kind === "H" && cur.to.price > prev.to.price && cur.volRatio < prev.volRatio * 0.8)
      out.push(`${day(cur.to.time)} 創新高 ${fmtP(cur.to.price)}，但這段上漲均量 ${cur.volRatio.toFixed(2)} 倍，低於前一段的 ${prev.volRatio.toFixed(2)} 倍（價漲量縮）`);
    if (cur.to.kind === "L" && cur.to.price < prev.to.price && cur.volRatio < prev.volRatio * 0.8)
      out.push(`${day(cur.to.time)} 創新低 ${fmtP(cur.to.price)}，但這段下跌均量 ${cur.volRatio.toFixed(2)} 倍，低於前一段的 ${prev.volRatio.toFixed(2)} 倍（賣壓減輕）`);
    // effort vs result: heavy volume, small move
    if (cur.volRatio > 1.3 && Math.abs(cur.change) < Math.abs(prev.change) * 0.5)
      out.push(`${day(cur.from.time)} ~ ${day(cur.to.time)} 量大（${cur.volRatio.toFixed(2)} 倍）但幅度小（${(cur.change * 100).toFixed(1)}%），努力與結果不一致`);
  }
  // up legs vs down legs over the range
  const up = ls.filter((l) => l.change > 0);
  const dn = ls.filter((l) => l.change < 0);
  const avg = (xs: Leg[]) => xs.reduce((s, l) => s + l.volRatio * l.bars, 0) / Math.max(1, xs.reduce((s, l) => s + l.bars, 0));
  if (up.length && dn.length) out.push(`範圍內上漲段平均量 ${avg(up).toFixed(2)} 倍、下跌段 ${avg(dn).toFixed(2)} 倍（${avg(up) > avg(dn) * 1.15 ? "漲時量大，需求較強" : avg(dn) > avg(up) * 1.15 ? "跌時量大，供給較強" : "差不多"}）`);
  // volume trend: first half vs second half of the range
  const half = Math.floor((from + to) / 2);
  const mean = (a: number, b: number) => {
    let s = 0;
    for (let i = a; i <= b; i++) s += bars[i]!.volume ?? 0;
    return s / Math.max(1, b - a + 1);
  };
  const v1 = mean(from, half);
  const v2 = mean(half + 1, to);
  if (v1 > 0) out.push(`範圍後半段均量是前半段的 ${(v2 / v1).toFixed(2)} 倍（${v2 < v1 * 0.8 ? "量能萎縮" : v2 > v1 * 1.25 ? "量能放大" : "持平"}）`);
  // no demand / no supply in the last bars: narrow up / down bar on volume lower than the two before
  for (let i = Math.max(from + 2, to - 9); i <= to; i++) {
    const b = bars[i]!;
    const v = b.volume ?? 0;
    const narrow = b.high - b.low < Math.min(bars[i - 1]!.high - bars[i - 1]!.low, bars[i - 2]!.high - bars[i - 2]!.low);
    if (!narrow || !(v < (bars[i - 1]!.volume ?? 0) && v < (bars[i - 2]!.volume ?? 0))) continue;
    if (b.close > bars[i - 1]!.close) out.push(`${day(b.time)} 窄幅上漲、量比前兩根小（可能是無需求 No Demand）`);
    else if (b.close < bars[i - 1]!.close) out.push(`${day(b.time)} 窄幅下跌、量比前兩根小（可能是無供給 No Supply）`);
  }
  return out;
}
