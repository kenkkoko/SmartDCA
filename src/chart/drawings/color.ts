// Colour parsing and contrast helpers shared by the drawing tools, so label
// text stays readable whatever colour the user picks for a drawing (a black
// price range label must not get near-black text).

export type Rgba = { r: number; g: number; b: number; a: number };

/** Light / dark text candidates for auto-picked label text. */
export const LIGHT_TEXT = "#ffffff";
export const DARK_TEXT = "#131722";

/** Parses #rgb, #rgba, #rrggbb, #rrggbbaa, rgb(...) and rgba(...); null otherwise. */
export function parseColor(color: string): Rgba | null {
  const s = color.trim().toLowerCase();
  if (s.startsWith("#")) {
    let hex = s.slice(1);
    if (!/^[0-9a-f]+$/.test(hex)) return null;
    if (hex.length === 3 || hex.length === 4) hex = hex.split("").map((ch) => ch + ch).join("");
    if (hex.length !== 6 && hex.length !== 8) return null;
    const n = (i: number) => parseInt(hex.slice(i, i + 2), 16);
    return { r: n(0), g: n(2), b: n(4), a: hex.length === 8 ? n(6) / 255 : 1 };
  }
  const m = /^rgba?\(([^)]+)\)$/.exec(s);
  if (!m) return null;
  const parts = m[1]!.split(/[\s,/]+/).filter(Boolean);
  if (parts.length < 3) return null;
  const ch = (v: string) => (v.endsWith("%") ? (parseFloat(v) / 100) * 255 : parseFloat(v));
  const [r, g, b] = parts.slice(0, 3).map(ch) as [number, number, number];
  const aRaw = parts[3];
  const a = aRaw === undefined ? 1 : aRaw.endsWith("%") ? parseFloat(aRaw) / 100 : parseFloat(aRaw);
  if ([r, g, b, a].some((v) => !Number.isFinite(v))) return null;
  const clamp = (v: number, max: number) => Math.max(0, Math.min(max, v));
  return { r: clamp(r, 255), g: clamp(g, 255), b: clamp(b, 255), a: clamp(a, 1) };
}

/** `top` painted over the opaque `bottom` (alpha compositing). */
function over(top: Rgba, bottom: Rgba): Rgba {
  const a = top.a;
  return { r: top.r * a + bottom.r * (1 - a), g: top.g * a + bottom.g * (1 - a), b: top.b * a + bottom.b * (1 - a), a: 1 };
}

function luminanceOf(c: Rgba): number {
  const [r, g, b] = [c.r, c.g, c.b].map((v) => {
    const x = v / 255;
    return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * WCAG relative luminance (0 = black, 1 = white). A translucent colour is
 * first composited over `under` (e.g. the chart background) when given.
 */
export function relativeLuminance(color: string, under?: string): number | null {
  const c = parseColor(color);
  if (!c) return null;
  if (c.a < 1 && under) {
    const u = parseColor(under);
    if (u) return luminanceOf(over(c, { ...u, a: 1 }));
  }
  return luminanceOf(c);
}

/** WCAG contrast ratio between two colours (1–21); null if either is unparsable. */
export function contrastRatio(a: string, b: string, under?: string): number | null {
  const la = relativeLuminance(a, under);
  const lb = relativeLuminance(b, under);
  if (la === null || lb === null) return null;
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/**
 * Light text on dark backgrounds, dark text on light ones: whichever of the
 * two candidates contrasts more with `background` (composited over `under`
 * when translucent). Unparsable backgrounds get light text.
 */
export function readableTextColor(background: string, under?: string, light = LIGHT_TEXT, dark = DARK_TEXT): string {
  if (relativeLuminance(background, under) === null) return light;
  const cLight = contrastRatio(light, background, under) ?? 0;
  const cDark = contrastRatio(dark, background, under) ?? 0;
  return cDark > cLight ? dark : light;
}

/** Minimum contrast for a user-chosen label text colour to be kept as is. */
export const MIN_TEXT_CONTRAST = 3;
