import { describe, expect, it } from "vitest";
import type { Bar } from "../engine/types";
import { buildPrompt, marksToDrawings } from "./ai";
import { aggregate, candlePatterns, fibFrom, keyLevels, volumePriceNotes, impulseViolations, profileSummary, structureLabels, swings, zigzag } from "./features";

/** Daily bars through the given closes (high/low 1% around, volume optional). */
function series(closes: number[], vol: (i: number) => number = () => 100): Bar[] {
  const t0 = Date.UTC(2023, 0, 2) / 1000;
  return closes.map((c, i) => ({ time: t0 + i * 86400, open: closes[i - 1] ?? c, high: c * 1.01, low: c * 0.99, close: c, volume: vol(i) }));
}
const path = (...legs: [number, number][]) => {
  const out: number[] = [];
  let p = legs[0]![0];
  for (const [, to] of legs) {
    const n = 20;
    const from = p;
    for (let k = 1; k <= n; k++) out.push(from + ((to - from) * k) / n);
    p = to;
  }
  return out;
};

describe("analysis features", () => {
  it("finds alternating swing points and labels the structure", () => {
    const bars = series(path([100, 100], [100, 150], [150, 120], [120, 180], [180, 150], [150, 210]));
    const pv = zigzag(bars, 0.1);
    expect(pv.map((p) => p.kind).join("")).toBe("LHLHLH");
    expect(pv.map((p) => Math.round(p.price))).toEqual([99, 152, 119, 182, 149, 212]);
    expect(structureLabels(pv)).toEqual(["L", "H", "HL", "HH", "HL", "HH"]);
    expect(swings(bars, 6).pivots.length).toBe(6);
  });

  it("tells P, b and D volume profiles apart", () => {
    const closes = path([100, 100], [100, 200], [200, 100], [100, 200]);
    const top = profileSummary(series(closes, (i) => (closes[i]! > 170 ? 1000 : 10)), 0, closes.length - 1)!;
    const bottom = profileSummary(series(closes, (i) => (closes[i]! < 130 ? 1000 : 10)), 0, closes.length - 1)!;
    const middle = profileSummary(series(closes, (i) => (Math.abs(closes[i]! - 150) < 15 ? 1000 : 10)), 0, closes.length - 1)!;
    expect(top.shape).toBe("P");
    expect(bottom.shape).toBe("b");
    expect(middle.shape).toBe("D");
  });

  it("computes Fibonacci levels and checks Elliott rules", () => {
    const f = fibFrom(100, 200);
    expect(f.find((l) => l.ratio === 0.618)!.price).toBeCloseTo(138.2);
    expect(impulseViolations([100, 150, 120, 200, 160, 220])).toEqual([]);
    expect(impulseViolations([100, 150, 90, 200, 140, 220])).toContain("2 浪跌破 1 浪起點");
    expect(impulseViolations([100, 150, 120, 200, 145, 220])).toContain("4 浪進入 1 浪價格區");
  });

  it("finds weekly / monthly key levels and where they started", () => {
    // a ceiling near 200 hit three times, a floor near 100 twice, then a breakout
    const closes = path([150, 150], [150, 200], [200, 120], [120, 199], [199, 101], [101, 201], [201, 100], [100, 260], [260, 230]).flatMap((c) => [c, c, c, c, c, c, c]);
    const days = series(closes);
    const lv = keyLevels(aggregate(days, "week"), aggregate(days, "month"));
    const near = (x: number) => lv.find((l) => Math.abs(l.price / x - 1) < 0.04);
    const top = near(200)!;
    expect(top).toBeTruthy();
    expect(top.touches).toBeGreaterThanOrEqual(3);
    expect(top.originKind).toBe("H");
    expect(top.originTime).toBeLessThan(days[20 * 7 * 2]!.time); // starts at the first top
    expect(near(100)).toBeTruthy();
  });

  it("recognises candlestick patterns", () => {
    const bar = (o: number, h: number, l: number, c: number, i: number): Bar => ({ time: i * 86400, open: o, high: h, low: l, close: c, volume: 1 });
    const down = [110, 108, 106, 104, 102].map((c, i) => bar(c + 1, c + 2, c - 1, c, i));
    expect(candlePatterns([...down, bar(100, 100.5, 94, 100.3, 5)], 5)).toContain("錘子線");
    expect(candlePatterns([...down, bar(102, 102.5, 99, 99.5, 5), bar(99, 104, 98.8, 103.5, 6)], 6)).toContain("看漲吞噬");
    expect(candlePatterns([...down, bar(100, 103, 97, 100.1, 5)], 5)).toContain("十字星");
  });

  it("notes a new high on lighter volume", () => {
    const closes = path([100, 100], [100, 150], [150, 120], [120, 170]);
    const bars = series(closes, (i) => (i < 40 ? 300 : i < 60 ? 150 : 100));
    const notes = volumePriceNotes(bars, zigzag(bars, 0.1), 0, bars.length - 1);
    expect(notes.some((n) => n.includes("價漲量縮"))).toBe(true);
  });

  it("merges days into weeks and months", () => {
    const bars = series(Array.from({ length: 70 }, (_, i) => 100 + i));
    expect(aggregate(bars, "week").length).toBe(10);
    expect(aggregate(bars, "month").length).toBe(3);
  });

  it("adds the computed structure to the prompt", () => {
    const closes = path([100, 100], [100, 60], [60, 70], [70, 62], [62, 120], [120, 95], [95, 160], [160, 130], [130, 175], [175, 150], [150, 190], [190, 140], [140, 210], [210, 170], [170, 230], [230, 200], [230, 260], [260, 220], [220, 280], [280, 240], [240, 300], [300, 250], [250, 320]);
    const bars = series(closes, (i) => (i % 37 === 0 ? 900 : 100));
    const p = buildPrompt({ symbol: "BTC", market: "crypto", timeframe: "1d", bars, from: 0, to: bars.length - 1 });
    for (const s of ["量價分析", "週線 / 月線關鍵支撐阻力", "大週期背景：週線", "擺動高低點", "斐波那契", "成交量分布 VPVR", "錨定 VWAP", "量價異常"]) expect(p).toContain(s);
  });

  it("draws fib, VPVR, VWAP, waves and Wyckoff events from the AI's marks", () => {
    const bars = series(path([100, 100], [100, 150], [150, 120], [120, 180], [180, 160], [160, 210]));
    const d = (i: number) => new Date(bars[i]!.time * 1000).toISOString().slice(0, 10);
    const wave = [[d(19), 100], [d(39), 150], [d(59), 120], [d(79), 180], [d(99), 160], [d(119), 210]];
    const out = marksToDrawings(
      [
        { type: "level", price: 150, from: d(39) },
        { type: "fib", from: [d(59), 118], to: [d(119), 211] },
        { type: "vpvr", from: d(39), to: d(79) },
        { type: "vwap", date: d(59) },
        { type: "wave", points: wave },
        { type: "event", name: "Spring", date: d(59), price: 120 },
      ],
      bars,
    );
    expect(out.map((x) => x.type)).toEqual(["horizontal_ray", "fib_retracement", "volume_profile", "anchored_vwap", "elliott_impulse", "text"]);
    expect(out[0]!.anchors[0]!.time).toBe(bars[39]!.time);
    out.shift();
    expect(out[0]!.anchors[1]!.price).toBeCloseTo(bars[119]!.high); // snapped to the swing high
    expect(out[0]!.style.fib).toBeTruthy();
    expect(out[4]!.text).toBe("Spring");
    expect(out[4]!.anchors[0]!.price).toBeCloseTo(bars[59]!.low);
  });

  it("warns when the AI's wave count breaks a rule", () => {
    const bars = series(path([100, 100], [100, 150], [150, 90], [90, 200], [200, 160], [160, 220]));
    const d = (i: number) => new Date(bars[i]!.time * 1000).toISOString().slice(0, 10);
    const out = marksToDrawings([{ type: "wave", points: [19, 39, 59, 79, 99, 119].map((i) => [d(i), bars[i]!.close]) }], bars);
    expect(out[1]!.text).toContain("2 浪跌破 1 浪起點");
  });
});
