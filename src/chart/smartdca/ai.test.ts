import { describe, expect, it } from "vitest";
import type { Bar } from "../engine/types";
import { buildPrompt, extractMarks, MARK_COLORS, marksToDrawings, stripMarks } from "./ai";

const bars: Bar[] = Array.from({ length: 400 }, (_, i) => {
  const c = 100 + 20 * Math.sin(i / 15) + i * 0.1;
  return { time: Date.UTC(2024, 0, 1) / 1000 + i * 86400, open: c - 1, high: c + 2, low: c - 2, close: c, volume: 1000 + i };
});

describe("AI prompt", () => {
  it("summarises the selected range with indicators and the K-line table", () => {
    const p = buildPrompt({ symbol: "BTC", market: "加密", timeframe: "1d", bars, from: 300, to: 399, extras: ["恐懼貪婪指數 25"] });
    expect(p).toContain("BTC（加密）");
    expect(p).toContain("共 100 根日 K");
    expect(p).toContain("RSI(14)");
    expect(p).toContain("MACD(12,26,9)");
    expect(p).toContain("恐懼貪婪指數 25");
    // 100 bars fit without merging: one row per bar
    expect(p.split("\n").filter((l) => /^2024-|^2025-/.test(l)).length).toBe(100);
  });

  it("merges long ranges so the prompt stays small, and keeps the last 20 bars", () => {
    const p = buildPrompt({ symbol: "BTC", market: "加密", timeframe: "1d", bars, from: 0, to: 399 });
    expect(p).toContain("每列合併約");
    expect(p).toContain("最近 20 根");
    expect(p.length).toBeLessThan(20000);
  });

  it("passes the user's question and drawings", () => {
    const p = buildPrompt({
      symbol: "AAPL",
      market: "美股",
      timeframe: "1wk",
      bars,
      from: 350,
      to: 399,
      question: "會突破嗎？",
      drawings: [{ label: "趨勢線", drawing: { id: "d", type: "trend_line", paneId: "main", anchors: [{ time: bars[360]!.time, price: 110 }], style: {} as never } }],
    });
    expect(p).toContain("會突破嗎？");
    expect(p).toContain("趨勢線：");
    expect(p).toContain("週 K");
  });
});

describe("chart marks", () => {
  const day = (s: string) => Date.parse(`${s}T00:00:00Z`) / 1000;
  const bars = Array.from({ length: 30 }, (_, i) => {
    const t = day("2024-01-01") + i * 86400;
    return { time: t, open: 100 + i, high: 105 + i, low: 95 + i, close: 101 + i, volume: 1000 };
  });
  const answer = [
    "## 結論",
    "偏多。",
    "```chart-marks",
    JSON.stringify([
      { type: "support", price: 110, label: "前低" },
      { type: "resistance", price: "125", label: "壓力" },
      { type: "trend", from: ["2024-01-03", 98], to: ["2024-01-20", 115], label: "上升" },
      { type: "zone", from: "2024-01-05", to: "2024-01-15", top: 118, bottom: 104, label: "盤整" },
      { type: "buy", date: "2024-01-10", price: 104 },
      { type: "note", date: "2024-01-28", price: 130, text: "RSI 背離" },
      { type: "support", price: 999999 },
      { type: "laser", price: 100 },
    ]),
    "```",
  ].join("\n");

  it("hides the block from the text, also while it streams in", () => {
    expect(stripMarks(answer)).toBe("## 結論\n偏多。");
    expect(stripMarks("偏多。\n```chart-ma")).toBe("偏多。");
    expect(stripMarks("看 `EMA20`")).toBe("看 `EMA20`");
  });

  it("turns marks into drawings snapped to bars and drops bad ones", () => {
    const marks = extractMarks(answer)!;
    expect(marks).toHaveLength(8);
    const d = marksToDrawings(marks, bars);
    expect(d.map((x) => x.type)).toEqual(["horizontal_line", "horizontal_line", "trend_line", "rectangle", "arrow_mark_up", "text"]);
    expect(d[0]!.style.color).toBe(MARK_COLORS.up);
    expect(d[1]!.style.color).toBe(MARK_COLORS.down);
    expect(d[2]!.anchors[0]!.time).toBe(day("2024-01-03"));
    expect(d[5]!.text).toBe("RSI 背離");
    expect(d.every((x) => x.paneId === "main" && x.name?.startsWith("AI "))).toBe(true);
  });

  it("ignores a missing or broken block", () => {
    expect(extractMarks("沒有標記")).toBeNull();
    expect(extractMarks("```chart-marks\n[{oops\n```")).toBeNull();
  });
});
