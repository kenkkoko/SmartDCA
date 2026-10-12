// Indicator math shared by the Smart DCA technical chart. Same definitions
// as the app's existing helpers (SMA-seeded EMA, Wilder RSI, MACD 12/26/9).

export type Num = number | null;

export function ema(values: number[], period: number): Num[] {
  const out: Num[] = new Array(values.length).fill(null);
  if (values.length < period) return out;
  const k = 2 / (period + 1);
  let sum = 0;
  for (let i = 0; i < period; i++) sum += values[i]!;
  out[period - 1] = sum / period;
  for (let i = period; i < values.length; i++) out[i] = values[i]! * k + (out[i - 1] as number) * (1 - k);
  return out;
}

export function sma(values: number[], period: number): Num[] {
  const out: Num[] = new Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i]!;
    if (i >= period) sum -= values[i - period]!;
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}

/** Wilder RSI; null until `period` bars exist. */
export function rsi(closes: number[], period = 14): Num[] {
  const out: Num[] = new Array(closes.length).fill(null);
  if (closes.length <= period) return out;
  let gains = 0;
  let losses = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i]! - closes[i - 1]!;
    if (d > 0) gains += d;
    else losses -= d;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  const value = () => (avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
  out[period] = value();
  for (let i = period + 1; i < closes.length; i++) {
    const d = closes[i]! - closes[i - 1]!;
    avgGain = (avgGain * (period - 1) + (d > 0 ? d : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (d < 0 ? -d : 0)) / period;
    out[i] = value();
  }
  return out;
}

export function macd(closes: number[], fast = 12, slow = 26, signal = 9): { dif: Num[]; dea: Num[]; histogram: Num[] } {
  const f = ema(closes, fast);
  const s = ema(closes, slow);
  const dif: Num[] = closes.map((_, i) => (f[i] != null && s[i] != null ? f[i]! - s[i]! : null));
  const first = dif.findIndex((v) => v != null);
  const dea: Num[] = new Array(closes.length).fill(null);
  if (first >= 0) {
    const d = ema(dif.slice(first) as number[], signal);
    d.forEach((v, i) => (dea[first + i] = v));
  }
  const histogram: Num[] = dif.map((v, i) => (v != null && dea[i] != null ? v - dea[i]! : null));
  return { dif, dea, histogram };
}

/** Indices where a crosses above (+1) or below (-1) b. */
export function crosses(a: Num[], b: Num[]): { index: number; dir: 1 | -1 }[] {
  const out: { index: number; dir: 1 | -1 }[] = [];
  for (let i = 1; i < a.length; i++) {
    const ap = a[i - 1];
    const ac = a[i];
    const bp = b[i - 1];
    const bc = b[i];
    if (ap == null || ac == null || bp == null || bc == null) continue;
    if (ap <= bp && ac > bc) out.push({ index: i, dir: 1 });
    else if (ap >= bp && ac < bc) out.push({ index: i, dir: -1 });
  }
  return out;
}
