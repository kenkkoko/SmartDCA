// AI analysis: talk to the local bridge (ai-bridge/server.mjs) and turn the
// selected part of the chart into a compact, number-rich prompt.

import { getTool } from "../drawings/tools";
import { DEFAULT_STYLE, newId, type Drawing } from "../drawings/types";
import type { Bar } from "../engine/types";
import { crosses, ema, macd, rsi, sma, type Num } from "../indicators";

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
  "你是專業、謹慎的技術分析師，替一個定期定額（DCA）投資工具 Smart DCA 的使用者解讀圖表。",
  "只根據使用者提供的 K 線、成交量與指標數據分析，不要編造資料裡沒有的數字或新聞。",
  "用繁體中文（台灣用語）回答，使用 Markdown：## 小標題、條列、**粗體**。價格要寫出具體數字。",
  "結構：",
  "## 結論（3 句以內）",
  "## 趨勢與結構（高低點、型態、趨勢強弱）",
  "## 關鍵價位（支撐 / 壓力，附價格與理由）",
  "## 指標解讀（均線、RSI、MACD，指出背離或交叉）",
  "## 量能",
  "## 風險與接下來的觀察重點（什麼情況代表判斷錯誤）",
  "## 對定期定額的含義（加碼 / 照常 / 放慢，說明理由）",
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
  timeframe: "1d" | "1wk";
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
  const unit = input.timeframe === "1wk" ? "週" : "日";

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
  "格式固定為（開頭一定要是 ```chart-marks，內容是 JSON 陣列，最多 10 個，只標最重要的）：",
  "```chart-marks",
  "[",
  '  {"type":"support","price":58200,"label":"前波低點支撐"},',
  '  {"type":"resistance","price":73800,"label":"歷史高點壓力"},',
  '  {"type":"trend","from":["2024-08-05",49000],"to":["2024-11-04",66800],"label":"上升趨勢線"},',
  '  {"type":"zone","from":"2024-03-14","to":"2024-10-14","top":73800,"bottom":56500,"label":"盤整區"},',
  '  {"type":"buy","date":"2024-08-05","price":49000,"label":"恐慌低點"},',
  '  {"type":"sell","date":"2024-03-14","price":73800,"label":"量價背離"},',
  '  {"type":"note","date":"2024-12-17","price":108000,"text":"RSI 頂背離"}',
  "]",
  "```",
  "type 只能是 support / resistance / trend / zone / buy / sell / note。日期用 YYYY-MM-DD 且必須是資料裡有的日期，價格用資料中的實際數字。",
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
const TYPE_LABEL: Record<string, string> = { support: "支撐", resistance: "壓力", trend: "趨勢線", zone: "區間", buy: "買點", sell: "賣點", note: "註解" };

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
    const t = typeof v === "number" ? (v > 1e11 ? v / 1000 : v) : Date.parse(`${String(v).slice(0, 10)}T00:00:00Z`) / 1000;
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
  for (const m of marks.slice(0, 12)) {
    const kind = String(m.type ?? "").toLowerCase();
    const label = text(m);
    if (kind === "support" || kind === "resistance") {
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
    }
  }
  return out;
}
