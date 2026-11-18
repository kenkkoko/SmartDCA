import { describe, expect, it } from "vitest";
import type { ToolContext } from "../tool";
import { DEFAULT_STYLE, DEFAULT_VP_COLORS, resolveVpColors, type Drawing, type VpColors } from "../types";
import { volumeProfile } from "./volume";

const THEME_UP = "#00aa00";
const THEME_DOWN = "#aa0000";

/** Bars alternating up / down so both bar colours are drawn. */
const bars = Array.from({ length: 40 }, (_, i) => {
  const base = 100 + (i % 10);
  const up = i % 2 === 0;
  return { time: i, open: up ? base : base + 1, close: up ? base + 1 : base, high: base + 2, low: base - 1, volume: 100 + i };
});

type Call = { kind: string; color: string; text?: string };

function render(vp?: Partial<VpColors>): Call[] {
  const calls: Call[] = [];
  const item: Drawing = {
    id: "vp1",
    type: "volume_profile",
    anchors: [
      { time: 0, price: 0 },
      { time: 39, price: 0 },
    ],
    style: { ...DEFAULT_STYLE, vp: vp as VpColors | undefined },
  } as Drawing;
  const s = {
    fillPolygon: (_p: unknown, color: string) => calls.push({ kind: "box", color }),
    fillRect: (_x: number, _y: number, _w: number, _h: number, color: string) => calls.push({ kind: "bar", color }),
    hLine: (_a: number, _b: number, _y: number, color: string) => calls.push({ kind: "line", color }),
    vLine: () => undefined,
    strokePath: (_p: unknown, color: string) => calls.push({ kind: "line", color }),
    text: (text: string, _x: number, _y: number, color: string) => calls.push({ kind: "text", color, text }),
    measure: (t: string) => t.length * 6,
  };
  const c = {
    s,
    chart: { time: { timeToIndex: (t: number) => t } },
    item,
    points: [
      { x: 0, y: 0 },
      { x: 400, y: 0 },
    ],
    theme: { up: THEME_UP, down: THEME_DOWN, fontFamily: "sans-serif" },
    bars,
    toY: (p: number) => 1000 - p * 5,
  } as unknown as ToolContext;
  volumeProfile.render(c);
  return calls;
}

const rgba = (hex: string, a: number) => {
  const n = (k: number) => parseInt(hex.slice(1 + k * 2, 3 + k * 2), 16);
  return `rgba(${n(0)},${n(1)},${n(2)},${a})`;
};

describe("volume profile colours", () => {
  it("defaults keep the original look (theme up / down, shared VAH/VAL colour, labels follow lines)", () => {
    const r = resolveVpColors(undefined, THEME_UP, THEME_DOWN);
    expect(r).toMatchObject({
      up: THEME_UP,
      down: THEME_DOWN,
      vaUp: THEME_UP,
      vaDown: THEME_DOWN,
      vah: DEFAULT_VP_COLORS.valueArea,
      val: DEFAULT_VP_COLORS.valueArea,
      pocLabel: DEFAULT_VP_COLORS.poc,
      vahLabel: DEFAULT_VP_COLORS.valueArea,
      lineOpacity: 1,
      box: DEFAULT_VP_COLORS.box,
    });
  });

  it("drawings saved before the new fields (and explicit nulls) fall back to defaults", () => {
    const legacy = JSON.parse('{"up":"#111111","inOpacity":0.5,"outOpacity":0.2,"poc":"#222222","valueArea":"#333333","boxOpacity":0.1,"box":null}');
    const r = resolveVpColors(legacy, THEME_UP, THEME_DOWN);
    expect(r.up).toBe("#111111");
    expect(r.vaUp).toBe("#111111");
    expect(r.down).toBe(THEME_DOWN);
    expect(r.vah).toBe("#333333");
    expect(r.val).toBe("#333333");
    expect(r.vahLabel).toBe("#333333");
    expect(r.pocLabel).toBe("#222222");
    expect(r.box).toBe(DEFAULT_VP_COLORS.box);
  });

  it("an empty box colour means no box and survives a JSON round trip", () => {
    const saved = JSON.parse(JSON.stringify({ box: "" }));
    expect(resolveVpColors(saved, THEME_UP, THEME_DOWN).box).toBeNull();
    expect(render(saved).some((c) => c.kind === "box")).toBe(false);
  });

  it("renders every element with its own colour", () => {
    const calls = render({
      up: "#010101",
      down: "#020202",
      vaUp: "#030303",
      vaDown: "#040404",
      inOpacity: 0.7,
      outOpacity: 0.3,
      poc: "#050505",
      valueArea: "#060606",
      vah: "#070707",
      val: "#080808",
      labels: "#090909",
      box: "#0a0a0a",
      boxOpacity: 0.2,
    });
    const colors = new Set(calls.map((c) => c.color));
    expect(colors).toContain(rgba("#010101", 0.3));
    expect(colors).toContain(rgba("#020202", 0.3));
    expect(colors).toContain(rgba("#030303", 0.7));
    expect(colors).toContain(rgba("#040404", 0.7));
    expect(calls.find((c) => c.kind === "box")?.color).toBe(rgba("#0a0a0a", 0.2));
    const lines = calls.filter((c) => c.kind === "line").map((c) => c.color);
    expect(lines).toEqual(["#050505", "#070707", "#080808"]);
    const texts = calls.filter((c) => c.kind === "text");
    expect(texts.map((t) => t.text)).toEqual(["POC", "VAH", "VAL"]);
    expect(texts.every((t) => t.color === "#090909")).toBe(true);
  });

  it("line opacity applies to POC / VAH / VAL lines only", () => {
    const calls = render({ lineOpacity: 0.5 });
    const lines = calls.filter((c) => c.kind === "line").map((c) => c.color);
    expect(lines).toEqual([rgba(DEFAULT_VP_COLORS.poc, 0.5), rgba(DEFAULT_VP_COLORS.valueArea, 0.5), rgba(DEFAULT_VP_COLORS.valueArea, 0.5)]);
    expect(calls.find((c) => c.text === "POC")?.color).toBe(DEFAULT_VP_COLORS.poc);
  });

  it("without saved colours the bars use the chart's up / down colours", () => {
    const calls = render(undefined);
    const bars = new Set(calls.filter((c) => c.kind === "bar").map((c) => c.color));
    expect(bars).toContain(rgba(THEME_UP, DEFAULT_VP_COLORS.inOpacity));
    expect(bars).toContain(rgba(THEME_DOWN, DEFAULT_VP_COLORS.outOpacity));
  });
});

describe("VPVR up / down opacity", () => {
  it("defaults to 1 and is clamped", async () => {
    const { resolveVpColors } = await import("../types");
    const r = resolveVpColors({ upOpacity: 0.3, downOpacity: 5 } as never, "#0f0", "#f00");
    expect(r.upOpacity).toBe(0.3);
    expect(r.downOpacity).toBe(1);
    expect(resolveVpColors(undefined, "#0f0", "#f00").upOpacity).toBe(1);
  });
});
