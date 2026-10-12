// Core data types for the Smart DCA chart engine.

/** Epoch seconds (UTC). */
export type Time = number;

export type Bar = {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type ValuePoint = {
  time: Time;
  value: number;
  /** Optional per-point color (line segment ending here / histogram bar). */
  color?: string;
};

export type Point = { x: number; y: number };

export type Rect = { x: number; y: number; width: number; height: number };

export type PriceScaleMode = "log" | "linear";

export type LineStyle = "solid" | "dashed" | "dotted";

export type Theme = {
  background: string;
  text: string;
  grid: string;
  border: string;
  crosshair: string;
  up: string;
  down: string;
  axisLabelBackground: string;
  axisLabelText: string;
  fontFamily: string;
  fontSize: number;
  /** Default colour of drawings ("auto"): white-ish on dark, black-ish on light. */
  drawing: string;
  /** Colours for the toolbar, properties bar and dialogs around the canvas. */
  ui: {
    /** Toolbars, menus, dialogs. */
    panel: string;
    /** Raised surface inside a panel (inputs, hovered rows). */
    raised: string;
    text: string;
    muted: string;
    line: string;
    hover: string;
    accent: string;
    accentSoft: string;
    accentText: string;
    shadow: string;
  };
};

/** Green up, red down (Kai's convention). */
export const DARK_THEME: Theme = {
  background: "#0b0e14",
  text: "#9aa4b2",
  grid: "rgba(255,255,255,0.06)",
  border: "rgba(255,255,255,0.12)",
  crosshair: "rgba(200,205,215,0.55)",
  up: "#00d68f",
  down: "#ff5b6e",
  axisLabelBackground: "#2a2f3a",
  axisLabelText: "#e6e9ef",
  fontFamily:
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans TC', sans-serif",
  fontSize: 11,
  drawing: "#e6e9ef",
  ui: {
    panel: "#161b25",
    raised: "#0e1219",
    text: "#e6e9ef",
    muted: "#8b95a5",
    line: "rgba(255,255,255,0.14)",
    hover: "rgba(255,255,255,0.10)",
    accent: "#4c7dff",
    accentSoft: "rgba(76,125,255,0.22)",
    accentText: "#ffffff",
    shadow: "0 12px 32px rgba(0,0,0,0.55)",
  },
};

/** Light chart: white paper, black drawings (TradingView light). */
export const LIGHT_THEME: Theme = {
  background: "#ffffff",
  text: "#4a4f5a",
  grid: "rgba(17,17,17,0.07)",
  border: "rgba(17,17,17,0.16)",
  crosshair: "rgba(30,34,45,0.5)",
  up: "#089981",
  down: "#f23645",
  axisLabelBackground: "#131722",
  axisLabelText: "#ffffff",
  fontFamily: DARK_THEME.fontFamily,
  fontSize: 11,
  drawing: "#131722",
  ui: {
    panel: "#ffffff",
    raised: "#f3f4f6",
    text: "#131722",
    muted: "#5d6370",
    line: "rgba(17,17,17,0.16)",
    hover: "rgba(17,17,17,0.07)",
    accent: "#1b39c4",
    accentSoft: "rgba(27,57,196,0.12)",
    accentText: "#ffffff",
    shadow: "0 12px 32px rgba(0,0,0,0.18)",
  },
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
