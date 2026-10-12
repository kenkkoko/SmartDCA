// Chart appearance the user picks (圖表設定): background, grid and candle
// colours, kept separately for the dark and the light theme.

import type { Theme } from "../engine/types";

export type Appearance = {
  background?: string;
  grid?: string;
  gridHidden?: boolean;
  up?: string;
  down?: string;
  /** Wick colours; undefined = same as the body. */
  wickUp?: string;
  wickDown?: string;
  /** Hide the thin current-price line. */
  priceLineHidden?: boolean;
};

type Stored = { dark?: Appearance; light?: Appearance };

const KEY = "smartdca.chart.appearance";

function readAll(): Stored {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "{}") as Stored;
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

export function loadAppearance(mode: "dark" | "light"): Appearance {
  return { ...readAll()[mode] };
}

export function saveAppearance(mode: "dark" | "light", a: Appearance): void {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...readAll(), [mode]: a }));
  } catch {
    // private mode: the change still applies until reload
  }
}

function luminance(color: string): number | null {
  const m = /^#([0-9a-f]{6})$/i.exec(color.trim());
  if (!m) return null;
  const n = parseInt(m[1]!, 16);
  const ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * ch[0]! + 0.7152 * ch[1]! + 0.0722 * ch[2]!;
}

/** The base theme with the user's colours on top. */
export function applyAppearance(base: Theme, a: Appearance): Theme {
  const t: Theme = { ...base };
  if (a.background) {
    t.background = a.background;
    // a light background on the dark theme (or the reverse) flips the
    // text and the "auto" drawing colour so both stay readable
    const l = luminance(a.background);
    if (l !== null) {
      const light = l > 0.45;
      const baseLight = (luminance(base.background) ?? 0) > 0.45;
      if (light !== baseLight) {
        t.text = light ? "#4a4a4a" : "#9a968f";
        t.drawing = light ? "#111111" : "#f1efea";
        t.grid = light ? "rgba(17,17,17,0.07)" : "rgba(241,239,234,0.06)";
        t.crosshair = light ? "#787b86" : "#9598a1";
      }
    }
  }
  if (a.grid) t.grid = a.grid;
  if (a.gridHidden) t.grid = "rgba(0,0,0,0)";
  if (a.up) t.up = a.up;
  if (a.down) t.down = a.down;
  return t;
}
