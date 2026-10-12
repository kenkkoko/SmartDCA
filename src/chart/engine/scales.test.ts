import { describe, expect, it } from "vitest";
import { PriceScale } from "./price-scale";
import { TimeScale } from "./time-scale";

const day = 86400;
const times = Array.from({ length: 300 }, (_, i) => 1_700_000_000 + i * day);

function timeScale(): TimeScale {
  const t = new TimeScale({ barSpacing: 8, rightOffset: 4 });
  t.width = 800;
  t.setTimes(times);
  return t;
}

describe("TimeScale", () => {
  it("keeps sub-pixel x positions (no rounding)", () => {
    const t = timeScale();
    t.setBarSpacing(6.4);
    const gaps = [10, 11, 12].map((i) => t.indexToX(i + 1) - t.indexToX(i));
    for (const g of gaps) expect(g).toBeCloseTo(6.4, 10);
  });

  it("zooms around the cursor: the bar under it stays put", () => {
    const t = timeScale();
    const x = 300;
    const before = t.xToIndex(x);
    t.zoom(x, 1);
    expect(t.barSpacing).toBeCloseTo(8.8);
    expect(t.xToIndex(x)).toBeCloseTo(before, 9);
    t.zoom(x, -0.3); // a small trackpad step zooms a little
    expect(t.barSpacing).toBeCloseTo(8.8 * 0.97);
  });

  it("time-axis drag scales spacing by distance to the right edge, latest bar pinned", () => {
    const t = timeScale();
    const lastX = t.indexToX(t.lastIndex);
    t.scaleFromDrag(600, 400, 8); // drag left: 200 -> 400 px from the edge
    expect(t.barSpacing).toBeCloseTo(16);
    expect(t.width - t.indexToX(t.lastIndex)).toBeCloseTo((t.width - lastX) * 2, 6);
  });

  it("pans 1:1 with the pointer", () => {
    const t = timeScale();
    const x0 = t.indexToX(200);
    t.scrollFrom(t.rightOffset, 37);
    expect(t.indexToX(200) - x0).toBeCloseTo(37, 9);
  });

  it("maps times between and beyond bars continuously", () => {
    const t = timeScale();
    expect(t.timeToIndex(times[10]! + day / 2)).toBeCloseTo(10.5);
    expect(t.timeToIndex(times[299]! + 3 * day)).toBeCloseTo(302);
    expect(t.indexToTime(302)).toBeCloseTo(times[299]! + 3 * day);
  });

  it("never lets every bar leave the screen", () => {
    const t = timeScale();
    t.scrollFrom(t.rightOffset, 1e6);
    expect(t.visibleRange().to).toBeGreaterThanOrEqual(t.visibleRange().from);
    t.scrollFrom(t.rightOffset, -1e6);
    expect(t.indexToX(t.lastIndex)).toBeGreaterThan(0);
  });

  it("labels prefer year and month boundaries", () => {
    const t = timeScale();
    t.setBarSpacing(2);
    const marks = t.marks();
    expect(marks.length).toBeGreaterThan(2);
    expect(marks.some((m) => /^\d{4}$/.test(m.label) || /^[A-Z][a-z]{2}$/.test(m.label))).toBe(true);
    for (let i = 1; i < marks.length; i++) expect(marks[i]!.x - marks[i - 1]!.x).toBeGreaterThanOrEqual(90);
  });
});

describe("PriceScale", () => {
  it("defaults to log", () => {
    expect(new PriceScale().mode).toBe("log");
  });

  function logScale() {
    const p = new PriceScale("log");
    p.height = 600;
    p.setAutoRange(20, 70000);
    return p;
  }

  it("maps prices into the area inside the margins", () => {
    const p = logScale();
    expect(p.priceToY(70000)).toBeCloseTo(60); // 10% top margin
    expect(p.priceToY(20)).toBeCloseTo(540); // 10% bottom margin
    expect(p.yToPrice(p.priceToY(1234))).toBeCloseTo(1234, 6);
  });

  it("log ticks are evenly spaced with round prices", () => {
    const p = logScale();
    const ticks = p.ticks(42);
    expect(ticks.length).toBeGreaterThan(8);
    const gaps = ticks.slice(1).map((t, i) => ticks[i]!.y - t.y);
    const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    for (const g of gaps) {
      expect(g).toBeGreaterThan(mean * 0.5);
      expect(g).toBeLessThan(mean * 1.8);
    }
    for (const t of ticks) {
      const digits = String(t.price).replace(/^0\.0*|\./g, "").replace(/0+$/, "");
      expect(digits.length).toBeLessThanOrEqual(2);
    }
  });

  it("price-axis drag scales around the centre in log space and turns auto off", () => {
    const p = logScale();
    const snap = p.snapshot();
    p.scaleFromDrag(snap, 300, 200); // drag up = stretch
    expect(p.autoScale).toBe(false);
    expect(p.max - p.min).toBeLessThan(snap.max - snap.min);
    expect((p.max + p.min) / 2).toBeCloseTo((snap.max + snap.min) / 2, 9);
    expect(p.yToPrice(p.height)).toBeGreaterThan(0); // never negative on log
  });

  it("vertical pan follows the cursor 1:1", () => {
    const p = logScale();
    const snap = p.snapshot();
    const y0 = p.priceToY(1000);
    p.scrollFrom(snap, 40);
    expect(p.priceToY(1000) - y0).toBeCloseTo(40, 6);
  });

  it("switching mode keeps the visible prices", () => {
    const p = logScale();
    const top = p.yToPrice(60);
    p.setMode("linear");
    expect(p.yToPrice(60)).toBeCloseTo(top, 4);
  });

  it("linear ticks use nice steps", () => {
    const p = new PriceScale("linear");
    p.height = 500;
    p.setAutoRange(100, 200);
    const ticks = p.ticks(42);
    const steps = new Set(ticks.slice(1).map((t, i) => Number((t.price - ticks[i]!.price).toFixed(6))));
    expect(steps.size).toBe(1);
    expect([1, 2, 2.5, 5, 10, 20, 25, 50]).toContain([...steps][0]);
  });
});
