import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { Chart } from "./chart";
import type { Bar } from "./types";

// jsdom has no canvas: a no-op 2D context is enough to test scale geometry.
beforeAll(() => {
  const ctx = new Proxy(
    {},
    {
      get: (_t, key) => (key === "measureText" ? (s: string) => ({ width: s.length * 7 }) : key === "canvas" ? document.createElement("canvas") : () => undefined),
      set: () => true,
    },
  );
  HTMLCanvasElement.prototype.getContext = (() => ctx) as never;
});

const start = Date.UTC(2024, 0, 1) / 1000;
const bars: Bar[] = Array.from({ length: 100 }, (_, i) => {
  const c = 100 + i;
  return { time: start + i * 86400, open: c - 1, high: c + 2, low: c - 2, close: c, volume: 1000 + i };
});

let chart: Chart;
let host: HTMLDivElement;

const frame = async () => {
  await new Promise((resolve) => requestAnimationFrame(resolve));
  await new Promise((resolve) => setTimeout(resolve, 20));
};

beforeEach(() => {
  host = document.createElement("div");
  host.getBoundingClientRect = () => ({ x: 0, y: 0, left: 0, top: 0, right: 900, bottom: 500, width: 900, height: 500, toJSON: () => ({}) });
  document.body.appendChild(host);
  chart = new Chart(host, { kinetic: false, priceScaleMode: "linear" });
});

afterEach(() => {
  chart.destroy();
  host.remove();
});

describe("overlay band scales", () => {
  it("a banded overlay series gets its own symmetric scale without touching the candles or volume", async () => {
    const pane = chart.mainPane;
    chart.addCandles(pane).setData(bars);
    const volume = chart.addHistogram(pane, { scale: "overlay" });
    volume.setData(bars.map((b) => ({ time: b.time, value: b.volume! })));
    await frame();
    const priceBefore = pane.price.snapshot();
    const volumeBefore = pane.overlay.snapshot();

    // signed flows, in very different units from volume
    const flows = chart.addHistogram(pane, { scale: "overlay", band: { top: 0.8, bottom: 0, symmetric: true } });
    flows.setData(bars.map((b, i) => ({ time: b.time, value: i % 2 ? -3e6 : 1e6 })));
    await frame();

    expect(pane.price.snapshot()).toEqual(priceBefore);
    expect(pane.overlay.snapshot()).toEqual(volumeBefore);
    const band = chart.scaleOf(pane, flows);
    expect(band).not.toBe(pane.overlay);
    expect(chart.scaleOf(pane, volume)).toBe(pane.overlay);
    const h = pane.rect.height;
    expect(band.priceToY(0)).toBeCloseTo(h * 0.9, 6); // zero line mid-band
    // the largest flow (either sign) reaches the band edge, never past it
    const visible = chart.time.visibleRange();
    expect(visible.to).toBeGreaterThan(visible.from);
    expect(band.priceToY(-3e6)).toBeCloseTo(h, 6);
    expect(band.priceToY(3e6)).toBeCloseTo(h * 0.8, 6);
  });
});
