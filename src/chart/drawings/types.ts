// Drawing data model. Anchors are (time, price) so drawings survive zoom,
// pan, new bars and timeframe changes; pixels are derived when painting.

import type { LineStyle } from "../engine/types";

export type Anchor = { time: number; price: number };

export type FibLevel = { value: number; color: string; enabled: boolean };

export type FibSettings = {
  levels: FibLevel[];
  trendLine: boolean;
  trendLineColor: string;
  trendLineStyle: LineStyle;
  useOneColor: boolean;
  oneColor: string;
  background: boolean;
  backgroundOpacity: number;
  /** TradingView puts 1 at the first point and 0 at the second; reverse swaps. */
  reverse: boolean;
  prices: boolean;
  levelLabels: boolean;
  labelFormat: "values" | "percent";
  labelsH: "left" | "center" | "right";
  labelsV: "top" | "middle" | "bottom";
  /** Interpolate levels in log price. */
  logLevels: boolean;
};

/** The 24-row table as Kai set it up in TradingView. */
export const DEFAULT_FIB_LEVELS: FibLevel[] = [
  { value: 0, color: "#000000", enabled: true },
  { value: 0.236, color: "#000000", enabled: true },
  { value: 0.382, color: "#18322b", enabled: true },
  { value: 0.5, color: "#18322b", enabled: true },
  { value: 0.618, color: "#9a3a35", enabled: true },
  { value: 1.382, color: "#18322b", enabled: true },
  { value: 1, color: "#000000", enabled: true },
  { value: 1.618, color: "#9a3a35", enabled: true },
  { value: 0.886, color: "#000000", enabled: true },
  { value: 1.414, color: "#db882f", enabled: true },
  { value: 2, color: "#000000", enabled: true },
  { value: 1.5, color: "#18322b", enabled: true },
  { value: 0.65, color: "#7d3733", enabled: false },
  { value: 2.272, color: "#f5af3d", enabled: false },
  { value: 2.414, color: "#81bb6b", enabled: false },
  { value: 2, color: "#65a591", enabled: false },
  { value: 1.236, color: "#7bc7e1", enabled: false },
  { value: 3.272, color: "#8f8f8f", enabled: false },
  { value: 3.414, color: "#5b71ff", enabled: false },
  { value: 4, color: "#e15f57", enabled: false },
  { value: 4.272, color: "#9947bb", enabled: false },
  { value: 4.414, color: "#d95171", enabled: false },
  { value: 4.618, color: "#f5af3d", enabled: false },
  { value: 4.764, color: "#65a591", enabled: false },
];

export function defaultFibSettings(): FibSettings {
  return {
    levels: DEFAULT_FIB_LEVELS.map((l) => ({ ...l })),
    trendLine: true,
    trendLineColor: "#787b86",
    trendLineStyle: "dashed",
    useOneColor: false,
    oneColor: "#65a591",
    background: false,
    backgroundOpacity: 0.15,
    reverse: false,
    prices: true,
    levelLabels: true,
    labelFormat: "values",
    labelsH: "left",
    labelsV: "middle",
    logLevels: false,
  };
}

export type DrawingStyle = {
  color: string;
  width: number;
  lineStyle: LineStyle;
  /** Shape / channel background; undefined = none. */
  fillColor?: string;
  fillOpacity: number;
  extendLeft: boolean;
  extendRight: boolean;
  /** Channels: dashed middle line. */
  middleLine: boolean;
  /** Lines: show price change, %, bars and angle. */
  stats: boolean;
  textColor: string;
  fontSize: number;
  /** Volume profile. */
  rows: number;
  valueArea: number;
  showPoc: boolean;
  showValueArea: boolean;
  /** Fibonacci retracement settings. */
  fib?: FibSettings;
};

/** "auto" colours follow the chart theme: white-ish on dark, black-ish on light. */
export const AUTO_COLOR = "auto";

export const DEFAULT_STYLE: DrawingStyle = {
  color: AUTO_COLOR,
  width: 1,
  lineStyle: "solid",
  fillOpacity: 0.2,
  extendLeft: false,
  extendRight: false,
  middleLine: false,
  stats: false,
  textColor: AUTO_COLOR,
  fontSize: 13,
  rows: 366,
  valueArea: 70,
  showPoc: true,
  showValueArea: true,
};

export type Drawing = {
  id: string;
  type: string;
  /** Pane the drawing lives in ("main" for the price chart). */
  paneId: string;
  anchors: Anchor[];
  style: DrawingStyle;
  text?: string;
  name?: string;
  locked?: boolean;
  hidden?: boolean;
  /** Folder in the layers panel. */
  groupId?: string;
};

/** A folder of drawings in the layers panel. */
export type DrawingGroup = {
  id: string;
  name: string;
  /** Folded in the layers panel. */
  collapsed?: boolean;
};

let seq = 0;
export function newId(): string {
  return `d${Date.now().toString(36)}${(++seq).toString(36)}`;
}
