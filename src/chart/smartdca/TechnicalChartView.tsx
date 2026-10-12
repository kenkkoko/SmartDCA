// Smart DCA's technical chart on the new engine: candles, EMA ribbon with
// per-segment bull/bear colour, MA200 or Realized Price, RSI and MACD panes
// with cross markers, FNG fear dots, plus the full drawing toolset.
// Data fetching and the signal panel stay in the app; this only paints.

import { useEffect, useRef, useState } from "react";
import type { Chart, ChartPrimitive, Pane } from "../engine/chart";
import type { CandleSeries, HistogramSeries, LineSeries, Marker } from "../engine/series";
import type { Bar, ValuePoint } from "../engine/types";
import { crosses, ema, macd, rsi, sma, type Num } from "../indicators";
import type { DrawingLayer } from "../drawings/layer";
import { DrawingChart } from "../react/DrawingChart";

export const EMA_FAST = 20;
export const EMA_SLOW = 55;
export const EMA_PERIODS = [0, 0.2, 0.4, 0.6, 0.8, 1].map((k) => Math.round(EMA_FAST + (EMA_SLOW - EMA_FAST) * k));
const EMA_BULL = "rgba(0, 200, 83, 0.6)";
const EMA_BEAR = "rgba(255, 23, 68, 0.6)";
const UP = "#00d68f";
const VOLUME_MA = 20;
const DOWN = "#ff5b6e";

export type HoverInfo = { time: number; open: number; high: number; low: number; close: number; prevClose: number | null };

export type TechnicalChartViewProps = {
  candles: Bar[];
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

export function TechnicalChartView(props: TechnicalChartViewProps) {
  const [chart, setChart] = useState<Chart | null>(null);
  const refs = useRef<{
    candles: CandleSeries | null;
    volume: HistogramSeries | null;
    volumeMa: LineSeries | null;
    main: LineSeries[];
    panes: Pane[];
    legend: PaneLegend;
  }>({ candles: null, volume: null, volumeMa: null, main: [], panes: [], legend: new PaneLegend() });
  const hover = useRef(props.onHover);
  hover.current = props.onHover;
  const candlesRef = useRef(props.candles);
  candlesRef.current = props.candles;

  const onReady = (c: Chart, _layer: DrawingLayer) => {
    refs.current.candles = c.addCandles(c.mainPane, { upColor: UP, downColor: DOWN, wickUpColor: UP, wickDownColor: DOWN });
    const offLegend = c.addPrimitive(refs.current.legend);
    let lastIndex: number | null = null;
    const off = c.onCrosshairMove((info) => {
      const legend = refs.current.legend;
      const bars = candlesRef.current;
      const index = info?.bar ? indexOfTime(bars, info.bar.time) : -1;
      if (index !== legend.index) {
        legend.index = index >= 0 ? index : null;
        c.invalidate(false);
      }
      if (index === lastIndex) return;
      lastIndex = index;
      if (!info?.bar || index < 0) return hover.current?.(null);
      const b = info.bar;
      hover.current?.({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close, prevClose: index > 0 ? bars[index - 1]!.close : null });
    });
    setChart(c);
    return () => {
      off();
      offLegend();
      setChart(null);
    };
  };

  // candles + fit on new data
  useEffect(() => {
    const series = refs.current.candles;
    if (!chart || !series || !props.candles.length) return;
    series.setData(props.candles);
    chart.fitContent();
  }, [chart, props.candles]);

  useEffect(() => {
    if (chart && props.fitSignal) chart.fitContent();
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
      bars.map((b) => ({ time: b.time, value: b.volume ?? 0, color: b.close >= b.open ? "rgba(0,214,143,0.35)" : "rgba(255,91,110,0.35)" })),
    );
    // 20-bar volume moving average on the same overlay scale
    const vols = bars.map((b) => b.volume ?? 0);
    r.volumeMa = chart.addLine(chart.mainPane, { scale: "overlay", color: "rgba(91,156,255,0.9)", lineWidth: 1.5, title: "Vol MA20" });
    r.volumeMa.setData(points(bars, sma(vols, VOLUME_MA)));
    list.splice(list.indexOf(r.volumeMa), 1);
    list.splice(1, 0, r.volumeMa);
    chart.invalidate();
  }, [chart, props.candles, props.showVolume]);

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
  }, [chart, props.candles, props.showEMA, props.showMA200, props.realizedPrice, props.markers]);

  // RSI / MACD panes, rebuilt in order so toggling never stacks them
  useEffect(() => {
    if (!chart) return;
    const r = refs.current;
    for (const p of r.panes) chart.removePane(p.id);
    r.panes = [];
    r.legend.entries.clear();
    const bars = props.candles;
    if (!bars.length) return;
    const closes = bars.map((b) => b.close);
    const marker = (i: number, dir: number): Marker => ({
      time: bars[i]!.time,
      position: dir > 0 ? "belowBar" : "aboveBar",
      shape: dir > 0 ? "arrowUp" : "arrowDown",
      color: dir > 0 ? UP : DOWN,
    });

    if (props.showRSI && closes.length > 14) {
      const pane = chart.addPane({ weight: 1 / 3 });
      const values = rsi(closes, 14);
      const s = chart.addLine(pane, { color: "#a78bfa", lineWidth: 1.5, lastValueLabel: true, title: "RSI(14)" });
      s.setData(points(bars, values));
      s.setPriceLines([
        { price: 70, color: "rgba(255,91,110,0.5)", lineStyle: "dashed" },
        { price: 50, color: "rgba(255,255,255,0.18)", lineStyle: "dashed" },
        { price: 30, color: "rgba(0,214,143,0.5)", lineStyle: "dashed" },
      ]);
      const m: Marker[] = [];
      for (let i = 1; i < values.length; i++) {
        const a = values[i - 1];
        const b = values[i];
        if (a == null || b == null) continue;
        if (a <= 30 && b > 30) m.push(marker(i, 1));
        else if (a >= 70 && b < 70) m.push(marker(i, -1));
      }
      s.setMarkers(m);
      r.panes.push(pane);
      r.legend.entries.set(pane.id, [{ label: "RSI(14)", color: "#a78bfa", values }]);
    }

    if (props.showMACD) {
      const pane = chart.addPane({ weight: 1 / 3 });
      const m = macd(closes);
      chart
        .addHistogram(pane, {})
        .setData(points(bars, m.histogram, (i) => (m.histogram[i]! >= 0 ? "rgba(0,214,143,0.75)" : "rgba(255,91,110,0.75)")));
      const dif = chart.addLine(pane, { color: "#3b82f6", lineWidth: 1.5, title: "DIF" });
      dif.setData(points(bars, m.dif));
      chart.addLine(pane, { color: "#fbbf24", lineWidth: 1.5, title: "DEA" }).setData(points(bars, m.dea));
      dif.setMarkers(crosses(m.dif, m.dea).map((c) => marker(c.index, c.dir)));
      r.panes.push(pane);
      r.legend.entries.set(pane.id, [
        { label: "MACD 12 26 9", color: "#9aa4b2", values: [] },
        { label: "DIF", color: "#3b82f6", values: m.dif },
        { label: "DEA", color: "#fbbf24", values: m.dea },
        { label: "Hist", color: "#9aa4b2", values: m.histogram },
      ]);
    }
    chart.invalidate();
  }, [chart, props.candles, props.showRSI, props.showMACD]);

  return <DrawingChart className={props.className} storageKey={props.storageKey} onReady={onReady} />;
}
