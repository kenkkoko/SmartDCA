import { describe, expect, it } from "vitest";
import { DARK_TEXT, LIGHT_TEXT, contrastRatio, parseColor, readableTextColor, relativeLuminance } from "./color";
import { labelTextColor, type ToolContext } from "./tool";

describe("labelTextColor", () => {
  const ctx = (textColor: string, autoTextColor: boolean, background = "#0b0e14") =>
    ({ theme: { background }, item: { style: { textColor } }, autoTextColor }) as unknown as ToolContext;

  it("auto-picks against the label box on dark and light charts", () => {
    expect(labelTextColor(ctx("#e6e9ef", true), "#000000")).toBe(LIGHT_TEXT);
    expect(labelTextColor(ctx("#111111", true, "#ffffff"), "#000000")).toBe(LIGHT_TEXT);
    expect(labelTextColor(ctx("#e6e9ef", true), "#ffffff")).toBe(DARK_TEXT);
  });

  it("keeps the theme's auto colour when there is no box", () => {
    expect(labelTextColor(ctx("#e6e9ef", true))).toBe("#e6e9ef");
  });

  it("respects an explicit text colour unless it would vanish", () => {
    expect(labelTextColor(ctx("#ffeb3b", false), "#000000")).toBe("#ffeb3b");
    expect(labelTextColor(ctx("#000000", false), "#000000")).toBe(LIGHT_TEXT);
    expect(labelTextColor(ctx("#000000", false))).toBe(LIGHT_TEXT);
  });
});

describe("parseColor", () => {
  it("reads hex and rgb(a) forms", () => {
    expect(parseColor("#fff")).toEqual({ r: 255, g: 255, b: 255, a: 1 });
    expect(parseColor("#2962ff")).toEqual({ r: 41, g: 98, b: 255, a: 1 });
    expect(parseColor("#00000080")!.a).toBeCloseTo(128 / 255);
    expect(parseColor("rgba(20,24,32,0.9)")).toEqual({ r: 20, g: 24, b: 32, a: 0.9 });
    expect(parseColor("rgb(255 0 0)")).toEqual({ r: 255, g: 0, b: 0, a: 1 });
  });

  it("rejects what it cannot read", () => {
    expect(parseColor("auto")).toBeNull();
    expect(parseColor("#12")).toBeNull();
    expect(parseColor("#ggg")).toBeNull();
  });
});

describe("luminance / contrast", () => {
  it("spans black to white", () => {
    expect(relativeLuminance("#000000")).toBe(0);
    expect(relativeLuminance("#ffffff")).toBeCloseTo(1);
    expect(contrastRatio("#000", "#fff")).toBeCloseTo(21);
  });

  it("composites translucent colours over the background", () => {
    // half-transparent black over white sits in between
    const l = relativeLuminance("rgba(0,0,0,0.5)", "#ffffff")!;
    expect(l).toBeGreaterThan(0.1);
    expect(l).toBeLessThan(0.5);
  });
});

describe("readableTextColor", () => {
  it("puts light text on black / dark boxes", () => {
    expect(readableTextColor("#000000")).toBe(LIGHT_TEXT);
    expect(readableTextColor("#0b0e14")).toBe(LIGHT_TEXT);
    expect(readableTextColor("#2962ff")).toBe(LIGHT_TEXT);
    expect(readableTextColor("rgba(20,24,32,0.9)", "#ffffff")).toBe(LIGHT_TEXT);
  });

  it("puts dark text on white / bright boxes", () => {
    expect(readableTextColor("#ffffff")).toBe(DARK_TEXT);
    expect(readableTextColor("#00d68f")).toBe(DARK_TEXT);
    expect(readableTextColor("#ffeb3b")).toBe(DARK_TEXT);
    // a faint box on a light chart reads as light
    expect(readableTextColor("rgba(0,0,0,0.1)", "#ffffff")).toBe(DARK_TEXT);
  });

  it("always clears a usable contrast on the extremes", () => {
    for (const bg of ["#000000", "#ffffff", "#ff5b6e", "#00d68f", "#808080"]) {
      expect(contrastRatio(readableTextColor(bg), bg)!).toBeGreaterThanOrEqual(4.5);
    }
  });
});
