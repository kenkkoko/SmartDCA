// AI analysis: talk to the local bridge (ai-bridge/server.mjs) and turn the
// selected part of the chart into a compact, number-rich prompt.

import { getTool } from "../drawings/tools";
import { DEFAULT_STYLE, newId, type Drawing } from "../drawings/types";
import type { Bar } from "../engine/types";
import { crosses, ema, macd, rsi, sma, type Num } from "../indicators";
import { aggregate, anchoredVwap, fibFrom, impulseViolations, keyLevels, lastBreak, profileSummary, structureLabels, swings, volumeEvents, type Pivot } from "./features";

// ---------------------------------------------------------------------------
// bridge client
// ---------------------------------------------------------------------------

export type AiModel = { id: string; name: string; description?: string; efforts: string[] };
export type AiProvider = {
  id: string;
  name: string;
  available: boolean;
  reason?: string;
  experimental?: boolean;
  customModel?: boolean;
  cliVersion?: string | null;
  models: AiModel[];
};
export type AiSettings = { port: number; provider: string; model: string; effort: string; drawMarks: boolean };

const SETTINGS_KEY = "smartdca.ai.settings";
export const DEFAULT_AI_SETTINGS: AiSettings = { port: 8765, provider: "claude", model: "", effort: "medium", drawMarks: true };

export function loadAiSettings(): AiSettings {
  try {
    return { ...DEFAULT_AI_SETTINGS, ...(JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") as Partial<AiSettings>) };
  } catch {
    return DEFAULT_AI_SETTINGS;
  }
}

export function saveAiSettings(s: AiSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    // private mode
  }
}

const base = (port: number) => `http://127.0.0.1:${port}`;

export async function fetchProviders(port: number, refresh = false): Promise<AiProvider[]> {
  const res = await fetch(`${base(port)}/providers${refresh ? "?refresh=1" : ""}`, { signal: AbortSignal.timeout(30000) });
  if (!res.ok) throw new Error(`橋接回應 ${res.status}`);
  return ((await res.json()) as { providers: AiProvider[] }).providers;
}

export type StreamEvent =
  | { type: "text"; text: string }
  | { type: "thinking" }
  | { type: "usage"; model?: string; costUsd?: number }
  | { type: "done"; ms: number }
  | { type: "error"; message: string };

/** POST /analyze and call `on` for each server-sent event. */
export async function streamAnalysis(
  port: number,
  body: { provider: string; model: string; effort: string; system: string; prompt: string },
  on: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${base(port)}/analyze`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let msg = `橋接回應 ${res.status}`;
    try {
      msg = ((await res.json()) as { error?: string }).error || msg;
    } catch {
      // not JSON
    }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) on(JSON.parse(line.slice(6)) as StreamEvent);
    }
  }
}

// ---------------------------------------------------------------------------
// prompt
// ---------------------------------------------------------------------------

export const SYSTEM_PROMPT = [
  "你是專業、謹慎的技術分析師，用「由大到小」的流程解讀使用者的圖表：先找出週線 / 月線級別最重要的支撐阻力，再看大週期背景，再看中期結構與型態，最後看區間細節與關鍵價位。",
  "方法依據道氏理論、威科夫（吸籌 / 派發）、艾略特波浪、K 線型態、成交量分布（VPVR）、斐波那契與錨定 VWAP。",
  "只根據使用者提供的資料分析，不要編造資料裡沒有的數字或新聞。資料中「程式計算」的擺動點、成交量分布、斐波那契與 VWAP 是精確值，價位請直接引用，不要自己重算。",
  "判斷要說出依據（哪幾根 K、哪個擺動點、量是幾倍），沒有足夠證據就說「尚不明確」，不要硬套型態或硬數浪。",
  "用繁體中文（台灣用語）回答，使用 Markdown：## 小標題、條列、**粗體**。價格要寫出具體數字。",
  "結構：",
  "## 結論（3 句以內）",
  "## 週月線關鍵支撐阻力（先找出最重要的幾條：起點、轉折次數、是否互換過，目前價格在它們之間的哪裡）",
  "## 大週期背景（以上述關鍵價位為基礎，判斷牛市或熊市、高低點結構、是否有吸籌或派發區間）",
  "## 結構轉變與波浪（是否突破週線前高 / 跌破前低造成結構改變；從轉變點起走了幾浪、現在在哪一浪）",
  "## 型態與籌碼分布（目前的型態；VPVR 是 P / b / D 型或雙分布、POC 與價值區代表什麼）",
  "## 吸籌 / 派發特徵（區間內的量價與 K 線：SC、AR、ST、Spring、SOS、LPS，或 BC、UTAD、SOW、LPSY；只標有證據的）",
  "## 關鍵價位（斐波那契、POC / VAH / VAL、錨定 VWAP、前高前低；多個工具重疊的價位優先，附理由）",
  "## 情境與失效條件（偏多與偏空兩種劇本，各自要看到什麼、什麼價位代表判斷錯誤）",
  "最後一行加上：「以上為技術面分析，不構成投資建議。」",
].join("\n");

const fmt = (v: number | null | undefined, digits?: number): string => {
  if (v == null || !Number.isFinite(v)) return "-";
  const d = digits ?? (Math.abs(v) >= 1000 ? 2 : Math.abs(v) >= 1 ? 2 : 6);
  return Number(v.toFixed(d)).toString();
};
const pct = (a: number, b: number) => (a ? `${(((b - a) / a) * 100).toFixed(2)}%` : "-");
const date = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);
const last = (arr: Num[], i: number) => arr[i] ?? null;

/** Merge consecutive bars into at most `max` rows (keeps the shape of a long range). */
function compress(bars: Bar[], max: number): Bar[] {
  if (bars.length <= max) return bars;
  const size = Math.ceil(bars.length / max);
  const out: Bar[] = [];
  for (let i = 0; i < bars.length; i += size) {
    const chunk = bars.slice(i, i + size);
    out.push({
      time: chunk[0]!.time,
      open: chunk[0]!.open,
      high: Math.max(...chunk.map((b) => b.high)),
      low: Math.min(...chunk.map((b) => b.low)),
      close: chunk[chunk.length - 1]!.close,
      volume: chunk.reduce((s, b) => s + (b.volume ?? 0), 0),
    });
  }
  return out;
}

export type AnalysisInput = {
  symbol: string;
  market: string;
  /** 15m | 1h | 4h | 1d | 1wk | 1mo */
  timeframe: string;
  /** All loaded bars (indicators need the history before the range). */
  bars: Bar[];
  /** Inclusive bar indices of the selected range. */
  from: number;
  to: number;
  /** Drawings the user made (support lines etc.). */
  drawings?: { label: string; drawing: Drawing }[];
  /** Extra context lines, e.g. "恐懼貪婪指數：25（極度恐懼）". */
  extras?: string[];
  question?: string;
};

const UNIT_NAME: Record<string, string> = { "15m": "15 分", "1h": "1 小時", "4h": "4 小時", "1d": "日", "1wk": "週", "1mo": "月" };

/** Program-computed structure the model should quote instead of re-deriving. */
function featureLines(input: AnalysisInput, date: (t: number) => string): string[] {
  const { bars, from, to } = input;
  const out: string[] = [];
  const pivotRow = (p: Pivot, tag: string) => `- ${date(p.time)} ${p.kind === "H" ? "高點" : "低點"} ${fmt(p.price)}${tag ? `（${tag}）` : ""}`;

  // 1. big-picture background from weekly / monthly bars (merged from the loaded
  //    bars up to the end of the range): key levels first, then structure
  const upto = bars.slice(0, to + 1);
  const daySpan = (upto[upto.length - 1]!.time - upto[0]!.time) / 86400;
  const tf = input.timeframe;
  if (daySpan > 400) {
    const weekly = tf === "1mo" ? [] : tf === "1wk" ? upto : aggregate(upto, "week");
    const monthly = tf === "1mo" ? upto : aggregate(upto, "month");
    const wDate = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);
    const close = upto[upto.length - 1]!.close;
    // a weekly / monthly bar carries its first day: move each level's start to
    // the actual bar of that high / low so the ray begins where the level did
    const levels = keyLevels(weekly, monthly).map((l) => {
      let best = -1;
      for (let i = 0; i < upto.length; i++) {
        const b = upto[i]!;
        if (b.time < l.originTime) continue;
        if (b.time > l.originTime + 32 * 86400) break;
        const v = l.originKind === "H" ? b.high : b.low;
        if (best < 0 || Math.abs(v - l.price) < Math.abs((l.originKind === "H" ? upto[best]!.high : upto[best]!.low) - l.price)) best = i;
      }
      return best < 0 ? l : { ...l, originTime: upto[best]!.time, lastTime: Math.max(l.lastTime, upto[best]!.time) };
    });
    if (levels.length) {
      out.push("");
      out.push(`## 週線 / 月線關鍵支撐阻力（程式計算，由高到低；目前收盤 ${fmt(close)}）`);
      for (const l of levels) {
        const role = l.price > close ? "阻力" : "支撐";
        const dist = ((l.price / close - 1) * 100).toFixed(1);
        out.push(
          `- ${fmt(l.price)}：目前為${role}（距收盤 ${Number(dist) > 0 ? "+" : ""}${dist}%）；起點 ${wDate(l.originTime)} 的${l.originKind === "H" ? "高點" : "低點"}，${l.monthly ? "月線級" : "週線級"}，轉折 ${l.touches} 次${l.flipped ? "，支撐阻力互換過" : ""}，最近一次 ${wDate(l.lastTime)}`,
        );
      }
    }
    for (const [name, set, target] of [
      ["月線", monthly, 10],
      ["週線", weekly, 14],
    ] as const) {
      if (set.length < 12) continue;
      const { pivots, pct } = swings(set, target);
      const tags = structureLabels(pivots);
      out.push("");
      out.push(`## 大週期背景：${name}擺動高低點（程式計算，擺動幅度 ≥ ${(pct * 100).toFixed(0)}%）`);
      pivots.forEach((p, k) => out.push(`- ${wDate(p.time)} ${p.kind === "H" ? "高點" : "低點"} ${fmt(p.price)}（${tags[k]}）`));
      const brk = lastBreak(set, pivots);
      if (brk) out.push(`- 最近一次結構突破：${wDate(set[brk.index]!.time)} 收盤${brk.dir === "up" ? "突破" : "跌破"} ${wDate(brk.level.time)} 的${brk.level.kind === "H" ? "高點" : "低點"} ${fmt(brk.level.price)}`);
      const last = set[set.length - 1]!.close;
      const ma = sma(set.map((b) => b.close), name === "週線" ? 50 : 20);
      const m = ma[set.length - 1];
      if (m) out.push(`- 收盤 ${fmt(last)}，${name === "週線" ? "50 週" : "20 月"}均線 ${fmt(m)}（${last >= m ? "之上" : "之下"}）`);
    }
  }

  // 2. swing points of the selected range on this timeframe
  const { pivots, pct } = swings(bars, Math.min(24, Math.max(8, Math.round((to - from) / 12))), from, to);
  const tags = structureLabels(pivots);
  out.push("");
  out.push(`## ${UNIT_NAME[tf] ?? ""}線擺動高低點（程式計算，幅度 ≥ ${(pct * 100).toFixed(1)}%；畫波浪、斐波那契、VPVR 時請用這些點）`);
  pivots.forEach((p, k) => out.push(pivotRow(p, tags[k]!)));
  const brk = lastBreak(bars, pivots);
  if (brk && brk.index >= from) out.push(`- 結構突破：${date(bars[brk.index]!.time)} 收盤${brk.dir === "up" ? "突破" : "跌破"}前${brk.level.kind === "H" ? "高" : "低"} ${fmt(brk.level.price)}`);

  // 3. Fibonacci of the largest swing in the range
  let best: [Pivot, Pivot] | null = null;
  for (let k = 1; k < pivots.length; k++) {
    const a = pivots[k - 1]!;
    const b = pivots[k]!;
    if (!best || Math.abs(Math.log(b.price / a.price)) > Math.abs(Math.log(best[1].price / best[0].price))) best = [a, b];
  }
  if (best) {
    const [a, b] = best;
    out.push("");
    out.push(`## 斐波那契（程式計算：範圍內最大的一段擺動 ${date(a.time)} ${fmt(a.price)} → ${date(b.time)} ${fmt(b.price)}）`);
    out.push(`- ${fibFrom(a.price, b.price).map((l) => `${l.ratio} = ${fmt(l.price)}`).join("、")}`);
  }

  // 4. volume profile of the range, and of the latest swing
  const vp = profileSummary(bars, from, to);
  if (vp) {
    out.push("");
    out.push("## 成交量分布 VPVR（程式計算，分析範圍）");
    out.push(`- 形狀：${vp.shape}${vp.shape === "P" ? "（量集中在上方）" : vp.shape === "b" ? "（量集中在下方）" : vp.shape === "D" ? "（量集中在中段）" : "（兩個成交密集區）"}`);
    out.push(`- POC ${fmt(vp.poc)}，價值區 VAL ${fmt(vp.val)} ~ VAH ${fmt(vp.vah)}，範圍 ${fmt(vp.low)} ~ ${fmt(vp.high)}`);
    if (vp.hvn.length > 1) out.push(`- 成交密集區（HVN）：${vp.hvn.map((v) => fmt(v)).join("、")}`);
    if (vp.lvn.length) out.push(`- 成交稀薄區（LVN）：${vp.lvn.map((v) => fmt(v)).join("、")}`);
  }
  const lastPv = pivots[pivots.length - 2];
  if (lastPv && to - lastPv.index >= 10) {
    const v2 = profileSummary(bars, lastPv.index, to);
    if (v2) out.push(`- 從 ${date(lastPv.time)} ${lastPv.kind === "H" ? "高點" : "低點"}到現在：${v2.shape} 型，POC ${fmt(v2.poc)}，VAL ${fmt(v2.val)} ~ VAH ${fmt(v2.vah)}`);
  }

  // 5. anchored VWAP from the important highs / lows
  const hiP = pivots.filter((p) => p.kind === "H").sort((x, y) => y.price - x.price)[0];
  const loP = pivots.filter((p) => p.kind === "L").sort((x, y) => x.price - y.price)[0];
  const recent = pivots.slice(-3, -1);
  const vw = [...new Map([hiP, loP, ...recent].filter((p): p is Pivot => !!p).map((p) => [p.index, p])).values()];
  const vwRows = vw.map((p) => ({ p, v: anchoredVwap(bars, p.index, to) })).filter((r) => r.v != null);
  if (vwRows.length) {
    out.push("");
    out.push(`## 錨定 VWAP（程式計算，到 ${date(bars[to]!.time)} 的值；收盤 ${fmt(bars[to]!.close)}）`);
    for (const { p, v } of vwRows) out.push(`- 從 ${date(p.time)} ${p.kind === "H" ? "高點" : "低點"} ${fmt(p.price)} 起算：${fmt(v)}（收盤在${bars[to]!.close >= v! ? "之上" : "之下"}）`);
  }

  // 6. bars worth checking for Wyckoff events
  const ev = volumeEvents(bars, pivots, from, to);
  if (ev.length) {
    out.push("");
    out.push("## 量價異常 K 棒（程式篩選的候選，是否為威科夫事件由你判斷）");
    for (const e of ev) out.push(`- ${date(bars[e.index]!.time)} 收 ${fmt(bars[e.index]!.close)}：${e.text}`);
  }
  return out;
}

export function buildPrompt(input: AnalysisInput): string {
  const { bars, from, to } = input;
  const range = bars.slice(from, to + 1);
  if (!range.length) return "";
  const closes = bars.map((b) => b.close);
  const vols = bars.map((b) => b.volume ?? 0);
  const e20 = ema(closes, 20);
  const e55 = ema(closes, 55);
  const m50 = sma(closes, 50);
  const m200 = sma(closes, 200);
  const r14 = rsi(closes, 14);
  const md = macd(closes);
  const v20 = sma(vols, 20);
  const unit = ({ "15m": "15 分", "1h": "1 小時", "4h": "4 小時", "1d": "日", "1wk": "週", "1mo": "月" } as Record<string, string>)[input.timeframe] ?? "日";
  // intraday bars need the clock time (UTC)
  const intraday = ["15m", "1h", "4h"].includes(input.timeframe);
  const date = (t: number) => (intraday ? `${new Date(t * 1000).toISOString().slice(0, 16).replace("T", " ")}` : new Date(t * 1000).toISOString().slice(0, 10));

  const hi = Math.max(...range.map((b) => b.high));
  const lo = Math.min(...range.map((b) => b.low));
  const hiBar = range.find((b) => b.high === hi)!;
  const loBar = range.find((b) => b.low === lo)!;
  let peak = range[0]!.close;
  let maxDd = 0;
  for (const b of range) {
    peak = Math.max(peak, b.close);
    maxDd = Math.min(maxDd, b.close / peak - 1);
  }
  const first = range[0]!;
  const end = range[range.length - 1]!;

  const inRange = (i: number) => i >= from && i <= to;
  const events: string[] = [];
  for (const c of crosses(e20, e55)) if (inRange(c.index)) events.push(`${date(bars[c.index]!.time)} EMA20 ${c.dir > 0 ? "上穿" : "下穿"} EMA55`);
  for (const c of crosses(md.dif, md.dea)) if (inRange(c.index)) events.push(`${date(bars[c.index]!.time)} MACD ${c.dir > 0 ? "金叉" : "死叉"}`);
  for (let i = Math.max(1, from); i <= to; i++) {
    const a = r14[i - 1];
    const b = r14[i];
    if (a == null || b == null) continue;
    if (a < 30 && b >= 30) events.push(`${date(bars[i]!.time)} RSI 上穿 30`);
    if (a > 70 && b <= 70) events.push(`${date(bars[i]!.time)} RSI 跌破 70`);
  }
  const rsiRange = r14.slice(from, to + 1).filter((v): v is number => v != null);

  const lines: string[] = [];
  lines.push(`# 標的：${input.symbol}（${input.market}）· ${unit}線`);
  lines.push(`分析範圍：${date(first.time)} ~ ${date(end.time)}，共 ${range.length} 根${unit} K；資料最後一根是 ${date(bars[bars.length - 1]!.time)}。`);
  lines.push("");
  lines.push("## 範圍摘要");
  lines.push(`- 起始收盤 ${fmt(first.close)} → 結束收盤 ${fmt(end.close)}（${pct(first.close, end.close)}）`);
  lines.push(`- 最高 ${fmt(hi)}（${date(hiBar.time)}），最低 ${fmt(lo)}（${date(loBar.time)}）`);
  lines.push(`- 範圍內最大回撤 ${(maxDd * 100).toFixed(2)}%`);
  lines.push("");
  lines.push(`## 範圍結束時（${date(end.time)}）的指標`);
  lines.push(`- EMA20 ${fmt(last(e20, to))}、EMA55 ${fmt(last(e55, to))}（${(last(e20, to) ?? 0) >= (last(e55, to) ?? 0) ? "多頭排列" : "空頭排列"}）`);
  lines.push(`- MA50 ${fmt(last(m50, to))}、MA200 ${fmt(last(m200, to))}；收盤相對 MA200：${last(m200, to) ? pct(last(m200, to)!, end.close) : "資料不足"}`);
  lines.push(`- RSI(14) ${fmt(last(r14, to), 1)}；範圍內 RSI 介於 ${rsiRange.length ? `${Math.min(...rsiRange).toFixed(1)} ~ ${Math.max(...rsiRange).toFixed(1)}` : "-"}`);
  lines.push(`- MACD(12,26,9) DIF ${fmt(last(md.dif, to))}、DEA ${fmt(last(md.dea, to))}、柱 ${fmt(last(md.histogram, to))}`);
  if (vols.some((v) => v > 0)) lines.push(`- 成交量 ${fmt(end.volume, 0)}，20 期均量 ${fmt(last(v20, to), 0)}（${last(v20, to) ? `${((end.volume ?? 0) / last(v20, to)!).toFixed(2)} 倍` : "-"}）`);
  if (events.length) {
    lines.push("");
    lines.push("## 範圍內的訊號事件");
    for (const e of events.slice(-25)) lines.push(`- ${e}`);
  }
  lines.push(...featureLines(input, date));
  if (input.extras?.length) {
    lines.push("");
    lines.push("## 其他資料");
    for (const e of input.extras) lines.push(`- ${e}`);
  }
  if (input.drawings?.length) {
    lines.push("");
    lines.push("## 使用者在圖上畫的標記（可參考，未必正確）");
    for (const { label, drawing } of input.drawings.slice(0, 20))
      lines.push(`- ${label}：${drawing.anchors.map((a) => `${date(a.time)} @ ${fmt(a.price)}`).join(" → ")}`);
  }

  // K line table: long ranges are merged so the prompt stays small
  const rows = compress(range, 150);
  const merged = rows.length < range.length;
  lines.push("");
  lines.push(`## K 線資料（${merged ? `每列合併約 ${Math.ceil(range.length / rows.length)} 根，` : ""}日期,開,高,低,收,量,RSI,EMA20,EMA55）`);
  for (const b of rows) {
    const i = bars.findIndex((x) => x.time === b.time);
    const j = merged ? Math.min(to, i + Math.ceil(range.length / rows.length) - 1) : i;
    lines.push([date(b.time), fmt(b.open), fmt(b.high), fmt(b.low), fmt(b.close), fmt(b.volume, 0), fmt(last(r14, j), 1), fmt(last(e20, j)), fmt(last(e55, j))].join(","));
  }
  if (merged) {
    lines.push("");
    lines.push("## 最近 20 根（未合併）");
    for (let i = Math.max(from, to - 19); i <= to; i++) {
      const b = bars[i]!;
      lines.push([date(b.time), fmt(b.open), fmt(b.high), fmt(b.low), fmt(b.close), fmt(b.volume, 0), fmt(last(r14, i), 1)].join(","));
    }
  }
  lines.push("");
  lines.push(input.question?.trim() ? `## 使用者的問題\n${input.question.trim()}` : "請依系統指示的結構完成分析。");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// chart marks: the AI ends its answer with a JSON block that becomes drawings
// ---------------------------------------------------------------------------

export const MARKS_FENCE = "```chart-marks";

export const MARKS_PROMPT = [
  "",
  "【在圖上標記】分析完後，在回答的最後附上一段程式碼區塊，把你提到的關鍵價位與結構畫到使用者的圖表上。",
  "格式固定為（開頭一定要是 ```chart-marks，內容是 JSON 陣列，最多 15 個）：",
  "```chart-marks",
  "[",
  '  {"type":"level","price":69000,"from":"2021-11-08","label":"週線前高"},',
  '  {"type":"support","price":58200,"label":"前波低點支撐"},',
  '  {"type":"resistance","price":73800,"label":"歷史高點壓力"},',
  '  {"type":"trend","from":["2024-08-05",49000],"to":["2024-11-04",66800],"label":"上升趨勢線"},',
  '  {"type":"zone","from":"2024-03-14","to":"2024-10-14","top":73800,"bottom":56500,"label":"盤整區"},',
  '  {"type":"buy","date":"2024-08-05","price":49000,"label":"恐慌低點"},',
  '  {"type":"sell","date":"2024-03-14","price":73800,"label":"量價背離"},',
  '  {"type":"note","date":"2024-12-17","price":108000,"text":"RSI 頂背離"},',
  '  {"type":"fib","from":["2024-08-05",49000],"to":["2025-01-20",109500],"label":"主升段回撤"},',
  '  {"type":"vpvr","from":"2024-03-14","to":"2024-10-14","label":"盤整區籌碼"},',
  '  {"type":"vwap","date":"2024-08-05","label":"恐慌低點 VWAP"},',
  '  {"type":"wave","points":[["2024-08-05",49000],["2024-08-25",65000],["2024-09-06",52500],["2024-12-17",108000],["2025-01-13",89200],["2025-01-20",109500]],"label":"推動浪"},',
  '  {"type":"wave","points":[["2025-01-20",109500],["2025-04-07",74500],["2025-05-22",111900],["2025-06-22",98200]],"label":"修正浪"},',
  '  {"type":"event","name":"Spring","date":"2024-08-05","price":49000}',
  "]",
  "```",
  "type 只能是 level / support / resistance / trend / zone / buy / sell / note / fib / vpvr / vwap / wave / event。",
  "level：週線 / 月線級的關鍵支撐阻力，畫成水平射線，from 是這條價位的起點（最早轉折的那根 K 的日期，資料裡有列出），一定要把重要的關鍵價位都用 level 畫出來；support / resistance 只用在較次要的價位。",
  "fib：from 是擺動起點、to 是終點（例如上漲段從低點到高點），用資料中程式算好的擺動高低點，畫你認為最重要的一段。",
  "vpvr：成交量分布，from / to 用重要高低點的日期（例如吸籌區間的起訖，或結構轉變點到現在）。vwap：從重要高點或低點的日期起算。",
  "wave：推動浪 6 個點（0~5）或修正浪 4 個點（0、A、B、C），每個點都要是擺動高低點；程式會檢查波浪規則。",
  "event：威科夫事件（SC、AR、ST、Spring、UT、SOS、LPS、BC、UTAD、SOW、LPSY 等），name 用英文縮寫，標在那根 K 的高點或低點。",
  "總數最多 15 個，只標最重要的。日期照資料裡的寫法（YYYY-MM-DD，盤中週期是 YYYY-MM-DD HH:MM，UTC）且必須是資料裡有的時間，價格用資料中的實際數字。",
  "label 要短（10 字以內）。這段區塊之後不要再寫任何文字；前面的分析內文裡不要提到這段區塊。",
].join("\n");

/** The analysis text without the marks block (also hides a half-streamed one). */
export function stripMarks(text: string): string {
  const i = text.indexOf(MARKS_FENCE);
  if (i < 0) {
    // a fence that is still arriving ("```chart-ma")
    const tail = text.lastIndexOf("```");
    if (tail >= 0 && MARKS_FENCE.startsWith(text.slice(tail).trim())) return text.slice(0, tail).trimEnd();
    return text;
  }
  return text.slice(0, i).trimEnd();
}

type RawMark = Record<string, unknown>;

/** Pull the JSON array out of the ```chart-marks block (null when missing / broken). */
export function extractMarks(text: string): RawMark[] | null {
  const i = text.indexOf(MARKS_FENCE);
  if (i < 0) return null;
  let body = text.slice(i + MARKS_FENCE.length);
  const end = body.indexOf("```");
  if (end >= 0) body = body.slice(0, end);
  const a = body.indexOf("[");
  const b = body.lastIndexOf("]");
  if (a < 0 || b <= a) return null;
  try {
    const parsed = JSON.parse(body.slice(a, b + 1)) as unknown;
    return Array.isArray(parsed) ? parsed.filter((m): m is RawMark => !!m && typeof m === "object") : null;
  } catch {
    return null;
  }
}

export const MARK_COLORS = { up: "#22c55e", down: "#ef4444", zone: "#2962ff", note: "#f59e0b" };
const TYPE_LABEL: Record<string, string> = {
  level: "關鍵價位",
  support: "支撐",
  resistance: "壓力",
  trend: "趨勢線",
  zone: "區間",
  buy: "買點",
  sell: "賣點",
  note: "註解",
  fib: "斐波那契",
  vpvr: "成交量分布",
  vwap: "錨定 VWAP",
  wave: "波浪",
  event: "威科夫事件",
};
const BULL_EVENTS = /^(SC|PS|AR|ST|SPRING|TEST|SOS|LPS|JAC|BU|BUEC|SHAKEOUT)$/i;

/**
 * Turn the AI's marks into drawings on the price pane. Dates snap to the
 * nearest loaded bar; anything with a missing or absurd price is dropped.
 */
export function marksToDrawings(marks: RawMark[], bars: Bar[]): Drawing[] {
  if (!bars.length) return [];
  const lo = Math.min(...bars.map((b) => b.low));
  const hi = Math.max(...bars.map((b) => b.high));
  const price = (v: unknown): number | null => {
    const n = typeof v === "string" ? Number(v.replace(/,/g, "")) : typeof v === "number" ? v : NaN;
    return Number.isFinite(n) && n > 0 && n >= lo / 3 && n <= hi * 3 ? n : null;
  };
  const time = (v: unknown): number | null => {
    if (typeof v !== "string" && typeof v !== "number") return null;
    const str = String(v).trim();
    const t =
      typeof v === "number"
        ? v > 1e11
          ? v / 1000
          : v
        : /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(str)
          ? Date.parse(`${str.slice(0, 16).replace(" ", "T")}:00Z`) / 1000
          : Date.parse(`${str.slice(0, 10)}T00:00:00Z`) / 1000;
    if (!Number.isFinite(t)) return null;
    // nearest bar (bars are sorted)
    let a = 0;
    let b = bars.length - 1;
    if (t <= bars[0]!.time) return bars[0]!.time;
    if (t >= bars[b]!.time) return bars[b]!.time;
    while (b - a > 1) {
      const m = (a + b) >> 1;
      if (bars[m]!.time <= t) a = m;
      else b = m;
    }
    return t - bars[a]!.time <= bars[b]!.time - t ? bars[a]!.time : bars[b]!.time;
  };
  const point = (v: unknown): { time: number; price: number } | null => {
    if (!Array.isArray(v) || v.length < 2) return null;
    const t = time(v[0]);
    const p = price(v[1]);
    return t == null || p == null ? null : { time: t, price: p };
  };
  const text = (m: RawMark) => {
    const s = typeof m.label === "string" ? m.label : typeof m.text === "string" ? m.text : "";
    return s.trim().slice(0, 40);
  };
  const lastTime = bars[bars.length - 1]!.time;
  const make = (type: string, anchors: { time: number; price: number }[], style: Partial<Drawing["style"]>, label: string, kind: string): Drawing => ({
    id: newId(),
    type,
    paneId: "main",
    anchors,
    style: { ...DEFAULT_STYLE, ...getTool(type)?.defaults, ...style },
    text: label || undefined,
    name: `AI ${TYPE_LABEL[kind] ?? kind}${label ? `：${label}` : ""}`,
  });

  const out: Drawing[] = [];
  const indexAt = (t: number) => {
    let a = 0;
    let b = bars.length - 1;
    while (a < b) {
      const m = (a + b) >> 1;
      if (bars[m]!.time < t) a = m + 1;
      else b = m;
    }
    return a;
  };
  /** Swing points: alternating lows / highs (direction from the first leg), snapped to that bar's low / high. */
  const swingPoints = (raw: unknown[]): ({ time: number; price: number } | null)[] => {
    const pts = raw.map(point);
    const [a, b] = pts;
    if (!a || !b) return pts;
    const upFirst = b.price > a.price;
    return pts.map((q, k) => {
      if (!q) return null;
      const bar = bars[indexAt(q.time)]!;
      const high = (k % 2 === 1) === upFirst;
      return { time: q.time, price: high ? bar.high : bar.low };
    });
  };

  for (const m of marks.slice(0, 15)) {
    const kind = String(m.type ?? "").toLowerCase();
    const label = text(m);
    if (kind === "level") {
      const p = price(m.price);
      const t = m.from == null ? null : time(m.from);
      if (p == null) continue;
      const color = p > bars[bars.length - 1]!.close ? MARK_COLORS.down : MARK_COLORS.up;
      out.push(make("horizontal_ray", [{ time: t ?? bars[0]!.time, price: p }], { color, width: 2, lineStyle: "solid", textColor: color }, label || TYPE_LABEL.level!, kind));
    } else if (kind === "support" || kind === "resistance") {
      const p = price(m.price);
      if (p == null) continue;
      const color = kind === "support" ? MARK_COLORS.up : MARK_COLORS.down;
      out.push(make("horizontal_line", [{ time: lastTime, price: p }], { color, width: 2, lineStyle: "dashed", textColor: color }, label || TYPE_LABEL[kind]!, kind));
    } else if (kind === "trend") {
      const a = point(m.from);
      const b = point(m.to);
      if (!a || !b || a.time === b.time) continue;
      out.push(make("trend_line", [a, b], { color: MARK_COLORS.zone, width: 2, extendRight: m.extend !== false }, label, kind));
    } else if (kind === "zone") {
      const a = time(m.from);
      const b = m.to == null ? lastTime : time(m.to);
      const top = price(m.top);
      const bottom = price(m.bottom);
      if (a == null || b == null || top == null || bottom == null || a === b) continue;
      out.push(make("rectangle", [{ time: a, price: top }, { time: b, price: bottom }], { color: MARK_COLORS.zone, fillColor: MARK_COLORS.zone, fillOpacity: 0.1 }, label, kind));
    } else if (kind === "buy" || kind === "sell") {
      const t = time(m.date);
      const p = price(m.price);
      if (t == null || p == null) continue;
      const up = kind === "buy";
      out.push(make(up ? "arrow_mark_up" : "arrow_mark_down", [{ time: t, price: p }], { color: up ? MARK_COLORS.up : MARK_COLORS.down }, label, kind));
    } else if (kind === "note") {
      const t = time(m.date);
      const p = price(m.price);
      if (t == null || p == null || !label) continue;
      out.push(make("text", [{ time: t, price: p }], { textColor: MARK_COLORS.note, fontSize: 13 }, label, kind));
    } else if (kind === "fib") {
      const [a, b] = swingPoints([m.from, m.to]);
      if (!a || !b || a.time === b.time || a.price === b.price) continue;
      const d = make("fib_retracement", [a, b], {}, label, kind);
      getTool("fib_retracement")?.init?.(d);
      out.push(d);
    } else if (kind === "vpvr") {
      const a = time(m.from);
      const b = m.to == null ? lastTime : time(m.to);
      if (a == null || b == null || a === b) continue;
      const [i, j] = [indexAt(Math.min(a, b)), indexAt(Math.max(a, b))];
      const seg = bars.slice(i, j + 1);
      const top = Math.max(...seg.map((x) => x.high));
      const bottom = Math.min(...seg.map((x) => x.low));
      out.push(make("volume_profile", [{ time: bars[i]!.time, price: top }, { time: bars[j]!.time, price: bottom }], {}, label, kind));
    } else if (kind === "vwap") {
      const t = time(m.date ?? m.from);
      if (t == null) continue;
      const bar = bars[indexAt(t)]!;
      out.push(make("anchored_vwap", [{ time: t, price: bar.low }], {}, label, kind));
    } else if (kind === "wave") {
      const pts = Array.isArray(m.points) ? swingPoints(m.points) : [];
      if (pts.some((q) => !q) || (pts.length !== 6 && pts.length !== 4)) continue;
      const anchors = pts as { time: number; price: number }[];
      if (anchors.some((q, k) => k > 0 && q.time <= anchors[k - 1]!.time)) continue;
      const impulse = anchors.length === 6;
      const broken = impulse ? impulseViolations(anchors.map((q) => q.price)) : [];
      const color = broken.length ? MARK_COLORS.note : MARK_COLORS.zone;
      out.push(make(impulse ? "elliott_impulse" : "elliott_correction", anchors, { color, width: 2 }, label, kind));
      if (broken.length) {
        const endPt = anchors[anchors.length - 1]!;
        out.push(make("text", [endPt], { textColor: MARK_COLORS.note, fontSize: 12 }, `⚠ 波浪規則：${broken.join("、")}`, "note"));
      }
    } else if (kind === "event") {
      const t = time(m.date);
      const name = String(m.name ?? m.label ?? "").trim().slice(0, 12);
      if (t == null || !name) continue;
      const bar = bars[indexAt(t)]!;
      const bull = BULL_EVENTS.test(name);
      out.push(make("text", [{ time: t, price: bull ? bar.low : bar.high }], { textColor: bull ? MARK_COLORS.up : MARK_COLORS.down, fontSize: 12, textBold: true }, name, kind));
    }
  }
  return out;
}
