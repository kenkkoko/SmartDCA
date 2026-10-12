import { describe, expect, it } from "vitest";
import type { Bar } from "../engine/types";
import { buildPrompt } from "./ai";

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
