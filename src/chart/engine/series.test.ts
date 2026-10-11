import { describe, expect, it } from "vitest";
import { CandleSeries, LineSeries } from "./series";
import { TimeScale } from "./time-scale";

const day = 86400;
const bars = Array.from({ length: 50 }, (_, i) => ({ time: i * day, open: 10 + i, high: 12 + i, low: 9 + i, close: 11 + i }));

function setup() {
  const time = new TimeScale();
  time.width = 500;
  time.setTimes(bars.map((b) => b.time));
  const candles = new CandleSeries({ visible: true, autoscale: true, lastValueLabel: true, upColor: "g", downColor: "r", wickUpColor: "g", wickDownColor: "r" });
  candles.setData(bars);
  candles.reindex(time);
  return { time, candles };
}

describe("series lookups", () => {
  it("finds the bar under an index (legend / crosshair)", () => {
    const { candles } = setup();
    expect(candles.barAt(0)?.open).toBe(10);
    expect(candles.barAt(17)?.open).toBe(27);
    expect(candles.barAt(49)?.open).toBe(59);
    expect(candles.barAt(50)).toBeUndefined();
  });

  it("aligns sparse series by time", () => {
    const { time } = setup();
    const line = new LineSeries({ visible: true, autoscale: true, lastValueLabel: false, color: "b", lineWidth: 1, lineStyle: "solid" });
    line.setData([{ time: 10 * day, value: 1 }, { time: 20 * day, value: 2 }]);
    line.reindex(time);
    expect(line.valueAt(10)).toBe(1);
    expect(line.valueAt(15)).toBeUndefined();
    expect(line.valueAt(20)).toBe(2);
  });

  it("auto-scale range covers visible highs and lows", () => {
    const { candles } = setup();
    expect(candles.range(5, 9)).toEqual({ min: 14, max: 21 }); // bars 5..9 only
  });
});
