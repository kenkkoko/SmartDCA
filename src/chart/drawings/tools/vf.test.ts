import { describe, expect, it } from "vitest";
import { computeVf, DEFAULT_VF } from "./vf";

// up leg 100 -> 200, heavy volume traded around 161.8 (the 0.382 retracement of an up leg)
const bars = Array.from({ length: 50 }, (_, i) => {
  const mid = 100 + (100 * i) / 49;
  const heavy = Math.abs(mid - 162) < 3;
  return { open: mid - 0.5, close: mid + 0.5, high: mid + 1, low: mid - 1, volume: heavy ? 100 : 5 };
});

describe("V&F confluence", () => {
  it("finds the swing, its direction and the retracement levels", () => {
    const r = computeVf(bars, 0, 49, DEFAULT_VF)!;
    expect(r.isDown).toBe(false);
    expect(r.swingLo).toBe(99);
    expect(r.swingHi).toBe(201);
    const f618 = r.fibs.find((f) => f.name === "0.618")!;
    expect(f618.price).toBeCloseTo(201 - 0.618 * 102, 6);
    expect(r.fibs.find((f) => f.name === "1.618")!.price).toBeCloseTo(201 + 0.618 * 102, 6);
  });

  it("marks a zone where a Fibonacci level meets the POC", () => {
    const r = computeVf(bars, 0, 49, DEFAULT_VF)!;
    expect(Math.abs(r.poc - 162)).toBeLessThan(2);
    const z = r.zones.find((z) => z.fib === "0.382");
    expect(z?.hits).toContain("POC");
    expect(r.best).toBeGreaterThanOrEqual(0);
  });

  it("puts the retracement above the low on a down leg and the extensions below it", () => {
    const down = [...bars].reverse();
    const r = computeVf(down, 0, 49, { ...DEFAULT_VF, showExt: true })!;
    expect(r.isDown).toBe(true);
    expect(r.fibs.find((f) => f.name === "0.382")!.price).toBeCloseTo(99 + 0.382 * 102, 6);
    expect(r.fibs.find((f) => f.name === "1.272")!.price).toBeLessThan(99);
  });
});
