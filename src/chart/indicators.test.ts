import { describe, expect, it } from "vitest";
import { crosses, ema, macd, rsi, sma } from "./indicators";

describe("indicators", () => {
  const closes = Array.from({ length: 60 }, (_, i) => 100 + 10 * Math.sin(i / 5) + i * 0.3);

  it("EMA is seeded with the SMA of the first period", () => {
    const e = ema(closes, 10);
    expect(e[8]).toBeNull();
    expect(e[9]).toBeCloseTo(closes.slice(0, 10).reduce((a, b) => a + b) / 10, 10);
  });

  it("SMA matches a plain average", () => {
    expect(sma([1, 2, 3, 4], 2)).toEqual([null, 1.5, 2.5, 3.5]);
  });

  it("RSI is 100 on a straight rise and stays within 0..100", () => {
    expect(rsi(Array.from({ length: 20 }, (_, i) => i), 14)[19]).toBe(100);
    for (const v of rsi(closes)) if (v != null) expect(v >= 0 && v <= 100).toBe(true);
  });

  it("MACD histogram = DIF - DEA and starts after slow+signal-1 bars", () => {
    const m = macd(closes);
    const first = m.dea.findIndex((v) => v != null);
    expect(first).toBe(25 + 8);
    expect(m.histogram[40]).toBeCloseTo(m.dif[40]! - m.dea[40]!, 12);
  });

  it("finds crosses in both directions", () => {
    expect(crosses([1, 3, 1], [2, 2, 2])).toEqual([
      { index: 1, dir: 1 },
      { index: 2, dir: -1 },
    ]);
  });
});
