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
  /** Volume profile colours (missing = defaults). */
  vp?: VpColors;
  /** Keep the anchors' price labels on the price axis even when not selected. */
  axisLabel?: boolean;
  /** Where line text sits: above / on / below the line, and along it. */
  textV?: "top" | "middle" | "bottom";
  textH?: "left" | "center" | "right";
  textBold?: boolean;
};

/**
 * Volume profile colours. Every optional field falls back to another one, so
 * drawings saved before a field existed keep their look:
 * up / down -> the chart's up / down colours; vaUp / vaDown -> up / down;
 * vah / val -> valueArea; labels -> the colour of the line they name.
 */
export type VpColors = {
  up?: string;
  down?: string;
  /** Bars inside the value area; undefined = same as up / down. */
  vaUp?: string;
  vaDown?: string;
  /** Bar opacity inside / outside the value area. */
  inOpacity: number;
  outOpacity: number;
  /** Extra opacity for up / down bars, multiplied with the above (missing = 1). */
  upOpacity?: number;
  downOpacity?: number;
  poc: string;
  /** VAH / VAL line colour (shared fallback for vah / val). */
  valueArea: string;
  vah?: string;
  val?: string;
  /** Opacity of the POC / VAH / VAL lines (missing = 1). */
  lineOpacity?: number;
  /** POC / VAH / VAL text; undefined = each line's own colour. */
  labels?: string;
  /** Range box fill; undefined = default colour, "" = none (survives JSON). */
  box?: string;
  boxOpacity: number;
};

export const DEFAULT_VP_COLORS: VpColors = {
  inOpacity: 0.6,
  outOpacity: 0.25,
  poc: "#ff4d4f",
  valueArea: "#5b9cff",
  lineOpacity: 1,
  box: "#2962ff",
  boxOpacity: 0.05,
};

/** Stand-ins for the chart's up / down colours where no theme is at hand (UI swatches). */
export const VP_UP_FALLBACK = "#089981";
export const VP_DOWN_FALLBACK = "#f23645";

export type ResolvedVpColors = {
  up: string;
  down: string;
  vaUp: string;
  vaDown: string;
  inOpacity: number;
  outOpacity: number;
  upOpacity: number;
  downOpacity: number;
  poc: string;
  vah: string;
  val: string;
  lineOpacity: number;
  pocLabel: string;
  vahLabel: string;
  valLabel: string;
  /** null = no box. */
  box: string | null;
  boxOpacity: number;
};

/** Saved volume profile colours over the defaults; explicit undefined / null (older saves) is ignored. */
export function mergeVpColors(vp: Partial<VpColors> | undefined): VpColors {
  const c: VpColors = { ...DEFAULT_VP_COLORS };
  for (const [k, v] of Object.entries(vp ?? {})) if (v !== undefined && v !== null) (c as Record<string, unknown>)[k] = v;
  return c;
}

/** Every volume profile colour resolved, following the fallbacks described on VpColors. */
export function resolveVpColors(vp: Partial<VpColors> | undefined, up: string, down: string): ResolvedVpColors {
  const c = mergeVpColors(vp);
  const num = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : d);
  const upC = c.up || up;
  const downC = c.down || down;
  const poc = c.poc || DEFAULT_VP_COLORS.poc;
  const valueArea = c.valueArea || DEFAULT_VP_COLORS.valueArea;
  const vah = c.vah || valueArea;
  const val = c.val || valueArea;
  return {
    up: upC,
    down: downC,
    vaUp: c.vaUp || upC,
    vaDown: c.vaDown || downC,
    inOpacity: num(c.inOpacity, DEFAULT_VP_COLORS.inOpacity),
    outOpacity: num(c.outOpacity, DEFAULT_VP_COLORS.outOpacity),
    upOpacity: num(c.upOpacity, 1),
    downOpacity: num(c.downOpacity, 1),
    poc,
    vah,
    val,
    lineOpacity: num(c.lineOpacity, 1),
    pocLabel: c.labels || poc,
    vahLabel: c.labels || vah,
    valLabel: c.labels || val,
    box: c.box ? c.box : null,
    boxOpacity: num(c.boxOpacity, DEFAULT_VP_COLORS.boxOpacity),
  };
}

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
