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
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
