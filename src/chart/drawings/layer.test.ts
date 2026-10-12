import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { Chart, type PointerInfo } from "../engine/chart";
import type { Bar } from "../engine/types";
import { readable } from "./geometry";
import { DrawingLayer } from "./layer";
import { loadDrawings, persistLayer, saveFibTemplate } from "./storage";
import { getTool } from "./tools";
import { fibLevels, getFibDefaults } from "./tools/fibonacci";
import { defaultFibSettings } from "./types";

// jsdom has no canvas: a context whose every method is a no-op is enough,
// the layer is tested through its geometry, not pixels.
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

function bars(n = 300): Bar[] {
  const start = Date.UTC(2024, 0, 1) / 1000;
  return Array.from({ length: n }, (_, i) => {
    const c = 100 * 1.003 ** i;
    return { time: start + i * 86400, open: c * 0.99, high: c * 1.02, low: c * 0.98, close: c, volume: 1000 };
  });
}

let chart: Chart;
let layer: DrawingLayer;
let host: HTMLDivElement;

beforeEach(async () => {
  localStorage.clear();
  saveFibTemplate(null);
  host = document.createElement("div");
  host.getBoundingClientRect = () => ({ x: 0, y: 0, left: 0, top: 0, right: 900, bottom: 500, width: 900, height: 500, toJSON: () => ({}) });
  document.body.appendChild(host);
  chart = new Chart(host, { kinetic: false });
  chart.addCandles(chart.mainPane).setData(bars());
  layer = new DrawingLayer(chart);
  // let the first frame autoscale the price axis to the data
  await new Promise((resolve) => requestAnimationFrame(resolve));
  await new Promise((resolve) => setTimeout(resolve, 20));
});

afterEach(() => {
  layer.dispose();
  chart.destroy();
  host.remove();
});

const at = (x: number, y: number, extra: Partial<PointerInfo> = {}): PointerInfo => ({
  x,
  y,
  pane: chart.mainPane,
  button: 0,
  altKey: false,
  shiftKey: false,
  event: {} as PointerEvent,
  ...extra,
});

function click(x: number, y: number, extra: Partial<PointerInfo> = {}) {
  layer.down(at(x, y, extra));
  layer.up(at(x, y, extra));
}

function drag(from: [number, number], to: [number, number]) {
  layer.down(at(...from));
  layer.move(at((from[0] + to[0]) / 2, (from[1] + to[1]) / 2));
  layer.move(at(...to));
  layer.up(at(...to));
}

describe("creating drawings", () => {
  it("places a trend line with click-click and disarms the tool", () => {
    layer.setActiveTool("trend_line");
    click(200, 300);
    layer.hover(at(400, 150));
    click(400, 150);
    expect(layer.items).toHaveLength(1);
    expect(layer.items[0]!.anchors).toHaveLength(2);
    expect(layer.activeTool).toBeNull();
    expect(layer.selectedId).toBe(layer.items[0]!.id);
  });

  it("places a two-point tool with press-drag-release", () => {
    layer.setActiveTool("rectangle");
    drag([150, 300], [350, 120]);
    expect(layer.items).toHaveLength(1);
    const [a, b] = layer.items[0]!.anchors;
    expect(a!.time).toBeLessThan(b!.time);
    expect(a!.price).toBeLessThan(b!.price);
  });

  it("ignores a stray single click (zero-size drawing)", () => {
    layer.setActiveTool("trend_line");
    click(200, 300);
    click(200, 300);
    expect(layer.items).toHaveLength(0);
  });

  it("keeps the tool armed with keep-drawing on", () => {
    layer.setKeepDrawing(true);
    layer.setActiveTool("horizontal_line");
    click(200, 200);
    click(300, 250);
    expect(layer.items).toHaveLength(2);
    expect(layer.activeTool).toBe("horizontal_line");
  });

  it("draws a parallel channel with three clicks", () => {
    layer.setActiveTool("parallel_channel");
    click(150, 300);
    click(450, 200);
    click(300, 120);
    expect(layer.items).toHaveLength(1);
    expect(layer.items[0]!.anchors).toHaveLength(3);
    expect(layer.items[0]!.style.middleLine).toBe(true);
  });

  it("finishes a polyline on double-click", () => {
    layer.setActiveTool("polyline");
    click(100, 300);
    click(200, 200);
    click(300, 260);
    click(300, 260); // second click of the double-click
    layer.dblclick(at(300, 260));
    expect(layer.items).toHaveLength(1);
    expect(layer.items[0]!.anchors).toHaveLength(3);
  });
});

describe("drawing over other drawings (TradingView rule)", () => {
  beforeEach(() => {
    layer.setActiveTool("rectangle");
    drag([100, 350], [600, 80]);
    layer.select(null);
  });

  it("an armed tool places points on top of an existing rectangle", () => {
    layer.setActiveTool("trend_line");
    click(250, 250);
    click(450, 150);
    expect(layer.items.map((d) => d.type)).toEqual(["rectangle", "trend_line"]);
  });

  it("without a tool, clicking the rectangle selects it", () => {
    click(100, 200); // on its left edge
    expect(layer.selected?.type).toBe("rectangle");
  });

  it("Alt-click selects through an armed tool", () => {
    layer.setActiveTool("trend_line");
    click(100, 200, { altKey: true });
    expect(layer.items).toHaveLength(1);
    expect(layer.selected?.type).toBe("rectangle");
  });
});

describe("editing and undo", () => {
  beforeEach(() => {
    layer.setActiveTool("trend_line");
    click(200, 300);
    click(400, 150);
  });

  it("undo removes the new drawing and redo brings it back", () => {
    layer.undo();
    expect(layer.items).toHaveLength(0);
    layer.redo();
    expect(layer.items).toHaveLength(1);
  });

  it("a body drag is one undo step; a click without moving is none", () => {
    const before = JSON.stringify(layer.items[0]!.anchors);
    click(300, 225); // select only
    expect(layer.canUndo).toBe(true);
    layer.down(at(300, 225));
    for (let i = 1; i <= 5; i++) layer.move(at(300 + i * 10, 225));
    layer.up(at(350, 225));
    expect(JSON.stringify(layer.items[0]!.anchors)).not.toBe(before);
    layer.undo();
    expect(JSON.stringify(layer.items[0]!.anchors)).toBe(before);
    layer.undo(); // the creation
    expect(layer.items).toHaveLength(0);
  });

  it("dragging a handle moves only that anchor", () => {
    const [a0, b0] = layer.items[0]!.anchors;
    layer.down(at(400, 150)); // second handle
    layer.move(at(500, 100));
    layer.up(at(500, 100));
    const [a1, b1] = layer.items[0]!.anchors;
    expect(a1).toEqual(a0);
    expect(b1!.time).toBeGreaterThan(b0!.time);
  });

  it("rapid property edits coalesce into one undo step", () => {
    const id = layer.items[0]!.id;
    layer.update(id, (d) => void (d.style.width = 2));
    layer.update(id, (d) => void (d.style.width = 3));
    layer.update(id, (d) => void (d.style.width = 4));
    layer.undo();
    expect(layer.items[0]!.style.width).toBe(1);
  });

  it("locked drawings select but do not move", () => {
    const id = layer.items[0]!.id;
    layer.update(id, (d) => void (d.locked = true));
    const before = JSON.stringify(layer.items[0]!.anchors);
    drag([300, 225], [380, 225]);
    expect(JSON.stringify(layer.items[0]!.anchors)).toBe(before);
  });

  it("remembers the last style per tool for the next drawing", () => {
    layer.update(layer.items[0]!.id, (d) => void (d.style.color = "#ff9800"));
    layer.setActiveTool("trend_line");
    click(100, 100);
    click(200, 120);
    expect(layer.items[1]!.style.color).toBe("#ff9800");
  });
});

describe("Fibonacci retracement", () => {
  it("puts 1 on the first point and 0 on the second; reverse swaps", () => {
    const anchors = [
      { time: 0, price: 100 },
      { time: 1, price: 200 },
    ];
    const s = defaultFibSettings();
    const byValue = (r: boolean) => new Map(fibLevels(anchors, { ...s, reverse: r }).map((l) => [l.value, l.price]));
    expect(byValue(false).get(1)).toBe(100);
    expect(byValue(false).get(0)).toBe(200);
    expect(byValue(false).get(0.5)).toBe(150);
    expect(byValue(true).get(1)).toBe(200);
  });

  it("log levels interpolate in log price", () => {
    const anchors = [
      { time: 0, price: 100 },
      { time: 1, price: 400 },
    ];
    const mid = fibLevels(anchors, { ...defaultFibSettings(), logLevels: true }).find((l) => l.value === 0.5)!;
    expect(mid.price).toBeCloseTo(200, 6);
  });

  it("enables Kai's 12 levels and drops duplicate values", () => {
    const levels = fibLevels([{ time: 0, price: 1 }, { time: 1, price: 2 }], defaultFibSettings());
    expect(levels.map((l) => l.value).sort((a, b) => a - b)).toEqual([0, 0.236, 0.382, 0.5, 0.618, 0.886, 1, 1.382, 1.414, 1.5, 1.618, 2]);
  });

  it("new retracements start from the saved template", () => {
    const tpl = defaultFibSettings();
    tpl.reverse = true;
    tpl.levels[0]!.color = "#123456";
    saveFibTemplate(tpl);
    layer.setActiveTool("fib_retracement");
    click(200, 300);
    click(400, 150);
    expect(layer.items[0]!.style.fib?.reverse).toBe(true);
    expect(layer.items[0]!.style.fib?.levels[0]!.color).toBe("#123456");
    // and editing the drawing does not change the template
    layer.items[0]!.style.fib!.reverse = false;
    expect(getFibDefaults().reverse).toBe(true);
  });

  it("black levels are drawn light on the dark theme", () => {
    expect(readable("#000000", "#0b0e14")).not.toBe("#000000");
    expect(readable("#9a3a35", "#0b0e14")).toBe("#9a3a35");
    expect(readable("#ffffff", "#ffffff")).not.toBe("#ffffff");
  });
});

describe("hit testing every tool", () => {
  it("each tool can be created and then selected by clicking near its first handle path", () => {
    const failures: string[] = [];
    for (const type of ["trend_line", "ray", "horizontal_line", "vertical_line", "rectangle", "ellipse", "text", "callout", "price_range", "long_position", "fib_retracement", "volume_profile"]) {
      layer.removeAll();
      layer.setActiveTool(type);
      const tool = getTool(type)!;
      const n = typeof tool.create === "number" ? tool.create : 2;
      const pts: [number, number][] = [[200, 300], [420, 160], [520, 220]];
      for (let i = 0; i < n; i++) click(...pts[i]!);
      if (layer.items.length !== 1) {
        failures.push(`${type}: not created`);
        continue;
      }
      layer.select(null);
      // the first anchor sits on (or inside) these shapes; the ellipse is
      // hit inside its fill and the volume box spans the bars' own range
      if (type === "ellipse") click(310, 230);
      else if (type === "volume_profile") click(310, chart.priceToY(chart.mainPane, chart.bars()[Math.round(chart.time.xToIndex(310))]!.close));
      else click(200, 300);
      if (layer.selected?.type !== type) failures.push(`${type}: not hit`);
    }
    expect(failures).toEqual([]);
  });
});

describe("persistence", () => {
  it("saves drawings per key and loads them back", async () => {
    const off = persistLayer(layer, "TEST");
    layer.setActiveTool("trend_line");
    click(200, 300);
    click(400, 150);
    off(); // flushes the pending save
    expect(loadDrawings("TEST")).toHaveLength(1);
    expect(loadDrawings("OTHER")).toHaveLength(0);

    const other = new DrawingLayer(chart);
    const off2 = persistLayer(other, "TEST");
    expect(other.items).toHaveLength(1);
    off2();
    other.dispose();
  });
});

describe("chart view reset (symbol switch)", () => {
  it("gives a manually scaled price axis back to auto-scale and fits all bars", () => {
    const pane = chart.mainPane;
    pane.price.scaleFromDrag({ min: pane.price.min, max: pane.price.max }, 100, 300);
    expect(pane.price.autoScale).toBe(false);
    chart.resetView();
    expect(pane.price.autoScale).toBe(true);
    expect(pane.overlay.autoScale).toBe(true);
  });
});
