// Smart DCA's technical chart on the new engine: candles, EMA ribbon with
// per-segment bull/bear colour, MA200 or Realized Price, RSI and MACD panes
// with cross markers, FNG fear dots, plus the full drawing toolset.
// Data fetching and the signal panel stay in the app; this only paints.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Chart, ChartPrimitive, Pane } from "../engine/chart";
import type { CandleSeries, HistogramSeries, LineSeries, Marker } from "../engine/series";
import { DARK_THEME, LIGHT_THEME, type Bar, type Theme, type ValuePoint } from "../engine/types";

// Smart DCA's palette (the site's light / dark tokens) so the chart, its
// toolbars and side panels read as one surface with the page around them.
const SDCA_DARK: Theme = {
  ...DARK_THEME,
  background: "#121314",
  text: "#9a968f",
  grid: "rgba(241,239,234,0.06)",
  border: "rgba(241,239,234,0.14)",
  axisLabelBackground: "#2f3032",
  axisLabelText: "#f1efea",
  drawing: "#f1efea",
  ui: {
    panel: "#121314",
    raised: "#1a1b1d",
    text: "#f1efea",
    muted: "#9a968f",
    line: "#2f3032",
    hover: "rgba(241,239,234,0.08)",
    accent: "#6f8bff",
    accentSoft: "rgba(111,139,255,0.18)",
    accentText: "#0b0d1a",
    shadow: "0 12px 32px rgba(0,0,0,0.55)",
  },
};
const SDCA_LIGHT: Theme = {
  ...LIGHT_THEME,
  text: "#4a4a4a",
  drawing: "#111111",
  ui: {
    panel: "#ffffff",
    raised: "#f7f7f5",
    text: "#111111",
    muted: "#6b6b6b",
    line: "#d9d9d9",
    hover: "rgba(17,17,17,0.06)",
    accent: "#1b39c4",
    accentSoft: "rgba(27,57,196,0.10)",
    accentText: "#ffffff",
    shadow: "0 12px 32px rgba(0,0,0,0.16)",
  },
};
import { withAlpha } from "../drawings/geometry";
import { crosses, ema, macd, rsi, sma, type Num } from "../indicators";
import type { DrawingLayer } from "../drawings/layer";
import { DrawingChart } from "../react/DrawingChart";
import type { SidePanel } from "../react/SideBar";
import { applyAppearance, loadAppearance, saveAppearance, type Appearance } from "./appearance";
import { AppearancePanel } from "./AppearancePanel";

export const EMA_FAST = 20;
export const EMA_SLOW = 55;
export const EMA_PERIODS = [0, 0.2, 0.4, 0.6, 0.8, 1].map((k) => Math.round(EMA_FAST + (EMA_SLOW - EMA_FAST) * k));
const EMA_BULL = "rgba(0, 200, 83, 0.6)";
const EMA_BEAR = "rgba(255, 23, 68, 0.6)";
const VOLUME_MA = 20;

export type HoverInfo = { time: number; open: number; high: number; low: number; close: number; prevClose: number | null };

export type TechnicalChartViewProps = {
  candles: Bar[];
  /**
   * Identity of the data set (symbol, timeframe, load). The view re-fits only
   * when it changes, so live price updates keep the user's zoom and panes.
   * Without it every new `candles` array counts as a new data set.
   */
  dataKey?: string;
  showEMA: boolean;
  /** MA200 line (stocks). */
  showMA200: boolean;
  /** Realized Price line (crypto), already aligned to candle times. */
  realizedPrice: ValuePoint[] | null;
  showRSI: boolean;
  showMACD: boolean;
  /** Volume bars at the bottom of the price pane. */
  showVolume?: boolean;
  /** Extra candle markers (FNG fear dots). */
  markers: Marker[];
  /** Drawings are saved per key (e.g. "BTC.1d"). */
  storageKey: string;
  onHover?: (info: HoverInfo | null) => void;
  /** Bump to fit all bars (the "reset" button). */
  fitSignal?: number;
  className?: string;
  /** Overlays drawn over the chart area (OHLC box, signal panel). */
  children?: ReactNode;
  /** Name of the K-line layer, e.g. "BTC". */
  symbolLabel?: string;
  /** Hands the live chart + drawing layer to the page (AI range selection). */
  onChartReady?: (chart: Chart | null, layer: DrawingLayer | null) => void;
  /** Right-sidebar panels next to 圖層 (watchlist). */
  sidePanels?: SidePanel[];
  defaultSidePanel?: string | null;
  /** Follows the app's light / dark switch. */
  theme?: "light" | "dark";
  /** When the last bar closes (epoch s): countdown under the last price. */
  barCloseTime?: number | null;
};

const points = (candles: Bar[], values: Num[], color?: (i: number) => string | undefined): ValuePoint[] => {
  const out: ValuePoint[] = [];
  values.forEach((v, i) => {
    if (v != null && Number.isFinite(v)) out.push({ time: candles[i]!.time, value: v, color: color?.(i) });
  });
  return out;
};

function indexOfTime(bars: Bar[], time: number): number {
  let lo = 0;
  let hi = bars.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const t = bars[mid]!.time;
    if (t === time) return mid;
    if (t < time) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

/** Pane titles with the hovered (or last) values, painted top-left of each sub pane. */
class PaneLegend implements ChartPrimitive {
  zIndex = 50;
  entries = new Map<string, { label: string; color: string; values: Num[] }[]>();
  index: number | null = null;
  draw({ surface, chart, theme }: Parameters<ChartPrimitive["draw"]>[0]): void {
    for (const pane of chart.panes) {
      const rows = this.entries.get(pane.id);
      if (!rows) continue;
      let x = 8;
      const y = pane.rect.y + 12;
      const font = { size: 11, family: theme.fontFamily };
      for (const row of rows) {
        const i = this.index ?? row.values.length - 1;
        const v = row.values[Math.max(0, Math.min(row.values.length - 1, i))] ?? null;
        const text = v == null ? row.label : `${row.label} ${Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2)}`;
        surface.text(text, x, y, row.color, font);
        x += surface.measure(text, font) + 12;
      }
    }
  }
}

const settingsIcon = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="10" cy="10" r="2.6" />
    <path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4" />
  </svg>
);

const candleIcon = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
    <path d="M6 2v3M6 13v4M14 4v3M14 14v3" />
    <rect x="4" y="5" width="4" height="8" rx="0.5" />
    <rect x="12" y="7" width="4" height="7" rx="0.5" fill="currentColor" fillOpacity=".3" />
  </svg>
);

export function TechnicalChartView(props: TechnicalChartViewProps) {
  const [candlesHidden, setCandlesHidden] = useState(false);
  const mode = props.theme === "light" ? "light" : "dark";
  const baseTheme = mode === "light" ? SDCA_LIGHT : SDCA_DARK;
  // 圖表設定: the user's background / grid / candle colours, one set per theme
  const [appearance, setAppearanceState] = useState<Appearance>(() => loadAppearance(mode));
  const [appearanceMode, setAppearanceMode] = useState(mode);
  if (appearanceMode !== mode) {
    setAppearanceMode(mode);
    setAppearanceState(loadAppearance(mode));
  }
  const setAppearance = (a: Appearance) => {
    setAppearanceState(a);
    saveAppearance(mode, a);
  };
  const theme = useMemo(() => applyAppearance(baseTheme, appearance), [baseTheme, appearance]);
  const UP = theme.up;
  const DOWN = theme.down;
  const [chart, setChart] = useState<Chart | null>(null);
  const refs = useRef<{
    candles: CandleSeries | null;
    volume: HistogramSeries | null;
    volumeMa: LineSeries | null;
    main: LineSeries[];
    panes: Pane[];
    rsi: LineSeries | null;
    macd: { hist: HistogramSeries; dif: LineSeries; dea: LineSeries } | null;
    legend: PaneLegend;
  }>({ candles: null, volume: null, volumeMa: null, main: [], panes: [], rsi: null, macd: null, legend: new PaneLegend() });
  const fittedKey = useRef<string | Bar[] | null>(null);
  const dataKey = props.dataKey ?? props.candles;
  const hover = useRef(props.onHover);
  const lastHoverIndex = useRef<number | null>(null);
  hover.current = props.onHover;
  const candlesRef = useRef(props.candles);
  candlesRef.current = props.candles;

  const readyCb = useRef(props.onChartReady);
  readyCb.current = props.onChartReady;
  const onReady = (c: Chart, layer: DrawingLayer) => {
    // no explicit colours: candles take the chart theme's and follow theme switches
    refs.current.candles = c.addCandles(c.mainPane);
    const offLegend = c.addPrimitive(refs.current.legend);
    const off = c.onCrosshairMove((info) => {
      const legend = refs.current.legend;
      const bars = candlesRef.current;
      const index = info?.bar ? indexOfTime(bars, info.bar.time) : -1;
      if (index !== legend.index) {
        legend.index = index >= 0 ? index : null;
        c.invalidate(false);
      }
      if (index === lastHoverIndex.current) return;
      lastHoverIndex.current = index;
      if (!info?.bar || index < 0) return hover.current?.(null);
      const b = info.bar;
      hover.current?.({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close, prevClose: index > 0 ? bars[index - 1]!.close : null });
    });
    setChart(c);
    readyCb.current?.(c, layer);
    return () => {
      off();
      offLegend();
      setChart(null);
      readyCb.current?.(null, null);
    };
  };

  useEffect(() => {
    chart?.setBarCloseTime(props.barCloseTime ?? null);
  }, [chart, props.barCloseTime]);

  // candle colours (runs after DrawingChart applied the theme)
  useEffect(() => {
    refs.current.candles?.applyOptions({
      upColor: theme.up,
      downColor: theme.down,
      wickUpColor: appearance.wickUp ?? theme.up,
      wickDownColor: appearance.wickDown ?? theme.down,
    });
    chart?.invalidate();
  }, [chart, theme, appearance.wickUp, appearance.wickDown]);

  useEffect(() => {
    refs.current.candles?.applyOptions({ visible: !candlesHidden });
    chart?.invalidate();
  }, [chart, candlesHidden]);

  // candles + fit on new data
  useEffect(() => {
    const series = refs.current.candles;
    if (!chart || !series || !props.candles.length) return;
    series.setData(props.candles);
    // a live tick of the same data set: keep the view where the user left it
    if (fittedKey.current === dataKey) return;
    fittedKey.current = dataKey;
    // another symbol can live in a completely different price range: re-fit
    // both axes, otherwise a manually scaled price axis keeps the old range
    chart.resetView();
    // the OHLC box must not keep showing the previous symbol's bar
    refs.current.legend.index = null;
    lastHoverIndex.current = null;
    hover.current?.(null);
  }, [chart, props.candles, dataKey]);

  useEffect(() => {
    if (chart && props.fitSignal) chart.resetView();
  }, [chart, props.fitSignal]);

  // volume: own hidden scale in the bottom fifth of the price pane (TradingView)
  useEffect(() => {
    if (!chart) return;
    const r = refs.current;
    if (r.volume) chart.removeSeries(r.volume);
    if (r.volumeMa) chart.removeSeries(r.volumeMa);
    r.volume = null;
    r.volumeMa = null;
    const bars = props.candles;
    if (props.showVolume === false || !bars.some((b) => (b.volume ?? 0) > 0)) return chart.invalidate();
    r.volume = chart.addHistogram(chart.mainPane, { scale: "overlay", title: "Vol" });
    // paint behind the candles
    const list = chart.mainPane.series;
    list.splice(list.indexOf(r.volume), 1);
    list.unshift(r.volume);
    r.volume.setData(
      bars.map((b) => ({ time: b.time, value: b.volume ?? 0, color: withAlpha(b.close >= b.open ? UP : DOWN, 0.35) })),
    );
    // 20-bar volume moving average on the same overlay scale
    const vols = bars.map((b) => b.volume ?? 0);
    r.volumeMa = chart.addLine(chart.mainPane, { scale: "overlay", color: "rgba(91,156,255,0.9)", lineWidth: 1.5, title: "Vol MA20" });
    r.volumeMa.setData(points(bars, sma(vols, VOLUME_MA)));
    list.splice(list.indexOf(r.volumeMa), 1);
    list.splice(1, 0, r.volumeMa);
    chart.invalidate();
  }, [chart, props.candles, props.showVolume, UP, DOWN]);

  // main-pane overlays + candle markers
  useEffect(() => {
    if (!chart) return;
    const r = refs.current;
    for (const s of r.main) chart.removeSeries(s);
    r.main = [];
    const bars = props.candles;
    const closes = bars.map((b) => b.close);
    const markers: Marker[] = [...props.markers];
    if (props.showEMA && bars.length) {
      const lines = EMA_PERIODS.map((p) => ema(closes, p));
      const fast = lines[0]!;
      const slow = lines[lines.length - 1]!;
      const bull = (i: number) => (fast[i] != null && slow[i] != null ? (fast[i]! >= slow[i]! ? EMA_BULL : EMA_BEAR) : undefined);
      lines.forEach((values, k) => {
        const s = chart.addLine(chart.mainPane, { color: EMA_BULL, lineWidth: k === 0 || k === lines.length - 1 ? 2 : 1 });
        // only where both fast and slow exist, like the original ribbon
        s.setData(points(bars, values.map((v, i) => (bull(i) ? v : null)), bull));
        r.main.push(s);
      });
      for (const c of crosses(fast, slow))
        markers.push({ time: bars[c.index]!.time, position: c.dir > 0 ? "belowBar" : "aboveBar", shape: c.dir > 0 ? "arrowUp" : "arrowDown", color: c.dir > 0 ? UP : DOWN });
    }
    if (props.showMA200 && closes.length >= 200) {
      const values = sma(closes, 200);
      const last = values[values.length - 1]!;
      const s = chart.addLine(chart.mainPane, { color: closes[closes.length - 1]! >= last ? UP : DOWN, lineWidth: 2, lastValueLabel: true, title: "MA200" });
      s.setData(points(bars, values));
      r.main.push(s);
    }
    if (props.realizedPrice?.length) {
      const s = chart.addLine(chart.mainPane, { color: "#f59e0b", lineWidth: 2, lastValueLabel: true, title: "Realized Price" });
      s.setData(props.realizedPrice);
      r.main.push(s);
    }
    r.candles?.setMarkers(markers.sort((a, b) => a.time - b.time));
    chart.invalidate();
  }, [chart, props.candles, props.showEMA, props.showMA200, props.realizedPrice, props.markers, UP, DOWN]);

  // RSI / MACD panes: built when toggled or for a new data set (pane ids and
  // heights must survive live ticks), values filled in by the effect below
  const hasBars = props.candles.length > 0;
  useEffect(() => {
    if (!chart) return;
    const r = refs.current;
    for (const p of r.panes) chart.removePane(p.id);
    r.panes = [];
    r.rsi = null;
    r.macd = null;
    r.legend.entries.clear();
    if (!hasBars) return chart.invalidate();
    if (props.showRSI) {
      const pane = chart.addPane({ weight: 1 / 3 });
      r.rsi = chart.addLine(pane, { color: "#a78bfa", lineWidth: 1.5, lastValueLabel: true, title: "RSI(14)" });
      r.rsi.setPriceLines([
        { price: 70, color: withAlpha(DOWN, 0.5), lineStyle: "dashed" },
        { price: 50, color: theme.border, lineStyle: "dashed" },
        { price: 30, color: withAlpha(UP, 0.5), lineStyle: "dashed" },
      ]);
      r.panes.push(pane);
    }
    if (props.showMACD) {
      const pane = chart.addPane({ weight: 1 / 3 });
      const hist = chart.addHistogram(pane, {});
      const dif = chart.addLine(pane, { color: "#3b82f6", lineWidth: 1.5, title: "DIF" });
      const dea = chart.addLine(pane, { color: "#fbbf24", lineWidth: 1.5, title: "DEA" });
      r.macd = { hist, dif, dea };
      r.panes.push(pane);
    }
    chart.invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chart, props.showRSI, props.showMACD, hasBars, dataKey, UP, DOWN, theme.border]);

  useEffect(() => {
    if (!chart) return;
    const r = refs.current;
    const bars = props.candles;
    if (!bars.length) return;
    const closes = bars.map((b) => b.close);
    const marker = (i: number, dir: number): Marker => ({
      time: bars[i]!.time,
      position: dir > 0 ? "belowBar" : "aboveBar",
      shape: dir > 0 ? "arrowUp" : "arrowDown",
      color: dir > 0 ? UP : DOWN,
    });
    if (r.rsi && r.panes[0]) {
      const values = rsi(closes, 14);
      r.rsi.setData(points(bars, values));
      const m: Marker[] = [];
      for (let i = 1; i < values.length; i++) {
        const a = values[i - 1];
        const b = values[i];
        if (a == null || b == null) continue;
        if (a <= 30 && b > 30) m.push(marker(i, 1));
        else if (a >= 70 && b < 70) m.push(marker(i, -1));
      }
      r.rsi.setMarkers(m);
      r.legend.entries.set(r.panes[0].id, [{ label: "RSI(14)", color: "#a78bfa", values }]);
    }
    if (r.macd) {
      const pane = r.panes[r.panes.length - 1]!;
      const m = macd(closes);
      r.macd.hist.setData(points(bars, m.histogram, (i) => withAlpha(m.histogram[i]! >= 0 ? UP : DOWN, 0.75)));
      r.macd.dif.setData(points(bars, m.dif));
      r.macd.dea.setData(points(bars, m.dea));
      r.macd.dif.setMarkers(crosses(m.dif, m.dea).map((c) => marker(c.index, c.dir)));
      r.legend.entries.set(pane.id, [
        { label: "MACD 12 26 9", color: theme.text, values: [] },
        { label: "DIF", color: "#3b82f6", values: m.dif },
        { label: "DEA", color: "#fbbf24", values: m.dea },
        { label: "Hist", color: theme.text, values: m.histogram },
      ]);
    }
    chart.invalidate();
  }, [chart, props.candles, props.showRSI, props.showMACD, hasBars, dataKey, UP, DOWN, theme.text, theme.border]);

  const appearancePanel: SidePanel = {
    id: "appearance",
    title: "圖表設定（背景、K 線顏色）",
    icon: settingsIcon,
    render: (close) => <AppearancePanel value={appearance} onChange={setAppearance} base={baseTheme} onClose={close} />,
  };

  return (
    <DrawingChart className={props.className} storageKey={props.storageKey} onReady={onReady} theme={theme} sidePanels={[...(props.sidePanels ?? []), appearancePanel]} defaultSidePanel={props.defaultSidePanel}
      fixedLayers={[{ id: "candles", label: `K 線${props.symbolLabel ? ` · ${props.symbolLabel}` : ""}`, icon: candleIcon, hidden: candlesHidden, onToggle: () => setCandlesHidden((h) => !h) }]}
    >
      {props.children}
    </DrawingChart>
  );
}
