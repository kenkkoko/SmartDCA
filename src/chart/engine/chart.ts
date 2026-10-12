// The chart: panes stacked vertically, one shared time axis at the bottom and
// a price axis on the right of each pane. Two full-size canvases: `base`
// (data, grid, axes, drawings) repaints on data/scale changes; `overlay`
// (crosshair) repaints on every mouse move without touching the base.

import { PriceScale } from "./price-scale";
import {
  CandleSeries,
  HistogramSeries,
  LineSeries,
  type AnySeries,
  type CandleOptions,
  type HistogramOptions,
  type LineOptions,
  type Marker,
} from "./series";
import { Surface } from "./surface";
import { TimeScale, formatTimeLabel } from "./time-scale";
import { DARK_THEME, clamp, type Bar, type PriceScaleMode, type Rect, type Theme } from "./types";

export type Pane = {
  id: string;
  weight: number;
  price: PriceScale;
  /** Hidden scale for overlay series such as volume (bottom 20%). */
  overlay: PriceScale;
  series: AnySeries[];
  rect: Rect;
};

export type CrosshairInfo = {
  x: number;
  y: number;
  index: number;
  time: number | undefined;
  paneId: string;
  price: number;
  bar: Bar | undefined;
};

/** What the chart tells primitives (drawings) when they paint. */
export type PrimitiveContext = {
  surface: Surface;
  chart: Chart;
  theme: Theme;
};

export type ChartPrimitive = {
  /** Lower paints first. Drawings use 100+. */
  zIndex?: number;
  draw(context: PrimitiveContext): void;
  /** Paints over the axes (e.g. price/time tags of a selected drawing). */
  drawOverAxes?(context: PrimitiveContext): void;
};

/** A pointer gesture the drawing layer may claim before the chart pans. */
export type PointerInfo = {
  x: number;
  y: number;
  pane: Pane;
  button: number;
  altKey: boolean;
  shiftKey: boolean;
  event: PointerEvent;
};

export type InteractionHandler = {
  /** Return true to own this gesture (no chart panning). */
  down?(info: PointerInfo): boolean;
  move?(info: PointerInfo): void;
  up?(info: PointerInfo): void;
  /** Mouse moving without a button: return a cursor to show, or undefined. */
  hover?(info: PointerInfo): string | undefined;
  dblclick?(info: PointerInfo): boolean;
};

export type ChartOptions = {
  theme: Theme;
  priceScaleMode: PriceScaleMode;
  timeAxisHeight: number;
  minPriceAxisWidth: number;
  /** Kinetic scrolling after a fling. */
  kinetic: boolean;
};

const SEPARATOR = 1;

type Gesture =
  | { kind: "pan"; startX: number; startY: number; offset: number; pane: Pane; price: { min: number; max: number } | null; samples: { x: number; t: number }[] }
  | { kind: "price"; pane: Pane; startY: number; snapshot: { min: number; max: number } }
  | { kind: "time"; startX: number; spacing: number }
  | { kind: "separator"; index: number; startY: number; weights: [number, number]; heights: [number, number] }
  | { kind: "handler"; pane: Pane }
  | { kind: "pinch"; distance: number; spacing: number; center: number };

export class Chart {
  readonly options: ChartOptions;
  readonly time = new TimeScale();
  readonly panes: Pane[] = [];
  readonly root: HTMLDivElement;
  private readonly base: Surface;
  private readonly overlay: Surface;
  private primitives: ChartPrimitive[] = [];
  private interaction: InteractionHandler | null = null;
  private crosshair: { x: number; y: number } | null = null;
  private crosshairListeners = new Set<(info: CrosshairInfo | null) => void>();
  private viewListeners = new Set<() => void>();
  private frame = 0;
  private dirtyBase = true;
  private priceAxisWidth = 64;
  private gesture: Gesture | null = null;
  private pointers = new Map<number, { x: number; y: number }>();
  private kineticFrame = 0;
  private resizeObserver: ResizeObserver | null = null;
  private dprQuery: MediaQueryList | null = null;
  private disposed = false;
  private mainSeries: CandleSeries | null = null;
  private paneSeq = 0;

  constructor(container: HTMLElement, options: Partial<ChartOptions> = {}) {
    this.options = {
      theme: DARK_THEME,
      priceScaleMode: "log",
      timeAxisHeight: 26,
      minPriceAxisWidth: 56,
      kinetic: true,
      ...options,
    };
    this.root = document.createElement("div");
    this.root.style.cssText = "position:relative;width:100%;height:100%;overflow:hidden;touch-action:none;user-select:none;outline:none;";
    this.root.tabIndex = 0; // keyboard shortcuts for drawings
    const baseCanvas = document.createElement("canvas");
    const overlayCanvas = document.createElement("canvas");
    for (const c of [baseCanvas, overlayCanvas]) c.style.cssText = "position:absolute;left:0;top:0;display:block;";
    this.root.append(baseCanvas, overlayCanvas);
    container.appendChild(this.root);
    this.base = new Surface(baseCanvas);
    this.overlay = new Surface(overlayCanvas);
    this.addPane({ weight: 1 }); // main pane
    this.attachEvents(overlayCanvas);
    this.observeSize(container);
  }

  // ======================================================================
  // Public API
  // ======================================================================

  get theme(): Theme {
    return this.options.theme;
  }

  get mainPane(): Pane {
    return this.panes[0]!;
  }

  addPane({ weight = 0.25 }: { weight?: number } = {}): Pane {
    const pane: Pane = {
      id: this.panes.length === 0 ? "main" : `pane${++this.paneSeq}`,
      weight,
      price: new PriceScale(this.panes.length === 0 ? this.options.priceScaleMode : "linear"),
      overlay: Object.assign(new PriceScale("linear"), { margins: { top: 0.8, bottom: 0 } }),
      series: [],
      rect: { x: 0, y: 0, width: 0, height: 0 },
    };
    this.panes.push(pane);
    this.layout();
    this.invalidate();
    return pane;
  }

  removePane(id: string): void {
    const i = this.panes.findIndex((p) => p.id === id);
    if (i <= 0) return; // the main pane stays
    this.panes.splice(i, 1);
    this.layout();
    this.invalidate();
  }

  addCandles(pane: Pane, options: Partial<CandleOptions> = {}): CandleSeries {
    const t = this.options.theme;
    const series = new CandleSeries({
      visible: true,
      autoscale: true,
      lastValueLabel: true,
      upColor: t.up,
      downColor: t.down,
      wickUpColor: t.up,
      wickDownColor: t.down,
      ...options,
    });
    if (!this.mainSeries) this.mainSeries = series;
    return this.attach(pane, series);
  }

  addLine(pane: Pane, options: Partial<LineOptions> = {}): LineSeries {
    return this.attach(
      pane,
      new LineSeries({ visible: true, autoscale: true, lastValueLabel: false, color: "#3b82f6", lineWidth: 1.5, lineStyle: "solid", ...options }),
    );
  }

  addHistogram(pane: Pane, options: Partial<HistogramOptions> = {}): HistogramSeries {
    return this.attach(
      pane,
      new HistogramSeries({ visible: true, autoscale: true, lastValueLabel: false, color: "#64748b", base: 0, ...options }),
    );
  }

  removeSeries(series: AnySeries): void {
    for (const pane of this.panes) pane.series = pane.series.filter((s) => s !== series);
    if (this.mainSeries === series) this.mainSeries = null;
    this.invalidate();
  }

  setPriceScaleMode(mode: PriceScaleMode, pane: Pane = this.mainPane): void {
    pane.price.setMode(mode);
    this.invalidate();
    this.emitView();
  }

  setAutoScale(on: boolean, pane: Pane = this.mainPane): void {
    pane.price.autoScale = on;
    this.invalidate();
    this.emitView();
  }

  addPrimitive(primitive: ChartPrimitive): () => void {
    this.primitives.push(primitive);
    this.primitives.sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0));
    this.invalidate();
    return () => {
      this.primitives = this.primitives.filter((p) => p !== primitive);
      this.invalidate();
    };
  }

  setInteractionHandler(handler: InteractionHandler | null): void {
    this.interaction = handler;
  }

  onCrosshairMove(listener: (info: CrosshairInfo | null) => void): () => void {
    this.crosshairListeners.add(listener);
    return () => this.crosshairListeners.delete(listener);
  }

  /** Fires after zoom/pan/scale changes (for toolbars showing state). */
  onViewChange(listener: () => void): () => void {
    this.viewListeners.add(listener);
    return () => this.viewListeners.delete(listener);
  }

  fitContent(): void {
    this.time.fitContent();
    this.invalidate();
  }

  /** New symbol / data set: fit all bars and give every price axis back to auto-scale. */
  resetView(): void {
    for (const pane of this.panes) {
      pane.price.autoScale = true;
      pane.overlay.autoScale = true;
    }
    this.time.fitContent();
    this.invalidate();
    this.emitView();
  }

  resetTimeScale(): void {
    this.time.reset();
    this.invalidate();
    this.emitView();
  }

  // ---- coordinates (CSS px relative to the chart root) ----

  timeToX(time: number): number {
    return this.time.indexToX(this.time.timeToIndex(time));
  }

  xToTime(x: number): number {
    return this.time.indexToTime(this.time.xToIndex(x));
  }

  priceToY(pane: Pane, price: number): number {
    return pane.rect.y + pane.price.priceToY(price);
  }

  yToPrice(pane: Pane, y: number): number {
    return pane.price.yToPrice(y - pane.rect.y);
  }

  paneAt(y: number): Pane | undefined {
    return this.panes.find((p) => y >= p.rect.y && y < p.rect.y + p.rect.height);
  }

  get plotWidth(): number {
    return this.time.width;
  }

  /** Bottom of the pane area (top of the time axis), CSS px. */
  get plotBottom(): number {
    return this.base.height - this.options.timeAxisHeight;
  }

  get width(): number {
    return this.base.width;
  }

  /** Candles of the main series (for tools that read prices / volume). */
  bars(): Bar[] {
    return this.mainSeries?.data ?? [];
  }

  /** Price-axis label text for a pane (same precision as the axis). */
  formatPrice(pane: Pane, price: number): string {
    return pane.price.format(price);
  }

  /** Repaint (coalesced to the next animation frame). */
  invalidate(baseToo = true): void {
    if (baseToo) this.dirtyBase = true;
    if (this.frame || this.disposed) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  }

  destroy(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    cancelAnimationFrame(this.kineticFrame);
    this.resizeObserver?.disconnect();
    this.dprQuery?.removeEventListener("change", this.onDprChange);
    this.root.remove();
  }

  // ======================================================================
  // Data plumbing
  // ======================================================================

  private attach<T extends AnySeries>(pane: Pane, series: T): T {
    series.onChange = () => {
      if (series === this.mainSeries) this.time.setTimes(this.mainSeries.data.map((b) => b.time));
      this.reindexAll();
      this.invalidate();
    };
    pane.series.push(series);
    series.onChange();
    return series;
  }

  private reindexAll(): void {
    for (const pane of this.panes) for (const s of pane.series) s.reindex(this.time);
  }

  // ======================================================================
  // Layout & size
  // ======================================================================

  private observeSize(container: HTMLElement): void {
    const apply = (width: number, height: number, bitmap?: { width: number; height: number }) => {
      this.base.resize(width, height, bitmap);
      this.overlay.resize(width, height, bitmap);
      this.layout();
      this.invalidate();
    };
    if (typeof ResizeObserver !== "undefined") {
      this.resizeObserver = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (!entry) return;
        const box = entry.contentRect;
        const device = (entry as ResizeObserverEntry & { devicePixelContentBoxSize?: ResizeObserverSize[] }).devicePixelContentBoxSize?.[0];
        apply(box.width, box.height, device ? { width: device.inlineSize, height: device.blockSize } : undefined);
      });
      try {
        this.resizeObserver.observe(container, { box: "device-pixel-content-box" });
      } catch {
        this.resizeObserver.observe(container);
      }
    }
    const rect = container.getBoundingClientRect();
    apply(rect.width, rect.height);
    this.watchDpr();
  }

  private onDprChange = () => {
    this.base.resize(this.base.width, this.base.height);
    this.overlay.resize(this.overlay.width, this.overlay.height);
    this.invalidate();
    this.watchDpr();
  };

  /** Re-render when the window moves to a screen with another pixel ratio. */
  private watchDpr(): void {
    if (typeof window === "undefined" || !window.matchMedia) return;
    this.dprQuery?.removeEventListener("change", this.onDprChange);
    this.dprQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    this.dprQuery.addEventListener("change", this.onDprChange, { once: true });
  }

  private layout(): void {
    const width = this.base.width;
    const height = this.base.height;
    const plotWidth = Math.max(0, width - this.priceAxisWidth);
    const panesHeight = Math.max(0, height - this.options.timeAxisHeight);
    const total = this.panes.reduce((sum, p) => sum + p.weight, 0) || 1;
    let y = 0;
    this.panes.forEach((pane, i) => {
      const last = i === this.panes.length - 1;
      const h = last ? panesHeight - y : Math.round((pane.weight / total) * panesHeight) - SEPARATOR;
      pane.rect = { x: 0, y, width: plotWidth, height: Math.max(0, h) };
      pane.price.height = pane.rect.height;
      pane.overlay.height = pane.rect.height;
      y += h + SEPARATOR;
    });
    const before = this.time.width;
    this.time.width = plotWidth;
    if (before === 0 && plotWidth > 0) this.time.clampOffset();
  }

  // ======================================================================
  // Rendering
  // ======================================================================

  private render(): void {
    if (this.disposed || this.base.width === 0) return;
    if (this.dirtyBase) {
      this.dirtyBase = false;
      this.autoscale();
      this.renderBase();
    }
    this.renderOverlay();
  }

  private autoscale(): void {
    const { from, to } = this.time.visibleRange();
    for (const pane of this.panes) {
      for (const overlay of [false, true]) {
        const scale = overlay ? pane.overlay : pane.price;
        if (!scale.autoScale) continue;
        let min = Infinity;
        let max = -Infinity;
        for (const s of pane.series) {
          if (!s.options.visible || !s.options.autoscale) continue;
          if ((s.options.scale === "overlay") !== overlay) continue;
          const r = s.range(from, to);
          if (!r) continue;
          min = Math.min(min, r.min);
          max = Math.max(max, r.max);
        }
        if (min <= max) scale.setAutoRange(overlay ? Math.min(0, min) : min, max);
      }
    }
  }

  private font(weight?: string) {
    return { size: this.theme.fontSize, family: this.theme.fontFamily, weight };
  }

  private renderBase(): void {
    const s = this.base;
    const t = this.theme;
    s.clear();
    s.fillRect(0, 0, s.width, s.height, t.background);

    // measure the price axis from its widest label, then re-layout if needed
    const ticksByPane = this.panes.map((pane) => pane.price.ticks());
    let widest = this.options.minPriceAxisWidth;
    for (const ticks of ticksByPane) for (const tick of ticks) widest = Math.max(widest, s.measure(tick.label, this.font()) + 16);
    for (const pane of this.panes) {
      const last = pane.series.find((x) => x.options.lastValueLabel)?.lastValue();
      if (last) widest = Math.max(widest, s.measure(pane.price.format(last.value), this.font()) + 16);
    }
    widest = Math.ceil(widest);
    if (Math.abs(widest - this.priceAxisWidth) > 2) {
      this.priceAxisWidth = widest;
      this.layout();
      this.autoscale();
      this.panes.forEach((pane, i) => (ticksByPane[i] = pane.price.ticks()));
    }

    const { from, to } = this.time.visibleRange();
    const marks = this.time.marks();
    const plotRight = this.time.width;

    this.panes.forEach((pane, i) => {
      const r = pane.rect;
      // grid on the axis labels
      s.clip(r);
      for (const m of marks) s.vLine(m.x, r.y, r.y + r.height, t.grid);
      for (const tick of ticksByPane[i]!) s.hLine(0, plotRight, r.y + tick.y, t.grid);
      // series
      const env = { surface: s, time: this.time, price: pane.price, top: r.y, theme: t };
      const overlayEnv = { ...env, price: pane.overlay };
      // overlay series (volume) paint first, underneath the price series
      for (const series of pane.series)
        if (series.options.visible && series.options.scale === "overlay") series.draw(overlayEnv, from, to);
      for (const series of pane.series)
        if (series.options.visible && series.options.scale !== "overlay") series.draw(env, from, to);
      // reference lines
      for (const series of pane.series) {
        if (!series.options.visible) continue;
        for (const line of series.priceLines)
          s.hLine(0, plotRight, r.y + pane.price.priceToY(line.price), line.color, line.lineWidth ?? 1, line.lineStyle ?? "dashed");
        this.drawMarkers(series, pane, from, to);
      }
      s.restore();
      // separator below the pane
      if (i < this.panes.length - 1) s.fillRect(0, r.y + r.height, s.width, SEPARATOR, t.border);
    });

    // primitives (drawings) clipped to the plot area of all panes
    s.clip({ x: 0, y: 0, width: plotRight, height: s.height - this.options.timeAxisHeight });
    for (const p of this.primitives) p.draw({ surface: s, chart: this, theme: t });
    s.restore();

    this.drawPriceAxes(ticksByPane);
    this.drawTimeAxis(marks);
    for (const p of this.primitives) p.drawOverAxes?.({ surface: s, chart: this, theme: t });
  }

  private drawMarkers(series: AnySeries, pane: Pane, from: number, to: number): void {
    if (!series.markers.length) return;
    const s = this.base;
    const size = Math.max(5, Math.min(10, this.time.barSpacing * 0.9));
    // candles anchor to the bar's high / low, lines and histograms to their own value
    const candle = series instanceof CandleSeries ? series : undefined;
    for (const m of series.markers) {
      const index = this.time.timeToIndex(m.time);
      if (index < from - 1 || index > to + 1) continue;
      const x = this.time.indexToX(index);
      const bar = candle?.barAt(Math.round(index));
      const value = m.value ?? series.valueAt(Math.round(index));
      const high = bar?.high ?? value;
      const low = bar?.low ?? value;
      if (high === undefined || low === undefined) continue;
      let y: number;
      if (m.position === "atValue" && m.value !== undefined) y = pane.price.priceToY(m.value);
      else if (m.position === "aboveBar") y = pane.price.priceToY(high) - size - 2;
      else y = pane.price.priceToY(low) + size + 2;
      y += pane.rect.y;
      drawMarkerShape(s, m, x, y, (m.size ?? 1) * size);
      if (m.text) {
        const below = m.position === "belowBar";
        s.text(m.text, x, below ? y + size + 2 : y - size - 2, m.color, this.font("600"), "center", below ? "top" : "bottom");
      }
    }
  }

  private drawPriceAxes(ticksByPane: ReturnType<PriceScale["ticks"]>[]): void {
    const s = this.base;
    const t = this.theme;
    const x0 = this.time.width;
    s.fillRect(x0, 0, s.width - x0, s.height, t.background);
    s.vLine(x0, 0, s.height, t.border);
    this.panes.forEach((pane, i) => {
      const r = pane.rect;
      s.clip({ x: x0, y: r.y, width: s.width - x0, height: r.height });
      // coloured labels (last value, price lines) win over tick labels
      const labels: { y: number; text: string; color: string }[] = [];
      for (const series of pane.series) {
        if (!series.options.visible) continue;
        const last = series.options.lastValueLabel ? series.lastValue() : undefined;
        if (last) labels.push({ y: r.y + pane.price.priceToY(last.value), text: pane.price.format(last.value), color: last.color });
        for (const line of series.priceLines)
          if (line.axisLabel) labels.push({ y: r.y + pane.price.priceToY(line.price), text: line.title ?? pane.price.format(line.price), color: line.color });
      }
      const half = this.theme.fontSize / 2 + 4;
      for (const tick of ticksByPane[i]!) {
        const y = r.y + tick.y;
        s.hLine(x0, x0 + 4, y, t.border);
        if (labels.some((l) => Math.abs(l.y - y) < half * 2)) continue;
        s.text(tick.label, x0 + 8, y, t.text, this.font());
      }
      for (const l of labels) this.axisLabel(s, l.y, l.text, l.color, "#ffffff");
      s.restore();
    });
  }

  private axisLabel(s: Surface, y: number, text: string, bg: string, fg: string): void {
    const x0 = this.time.width;
    const h = this.theme.fontSize + 8;
    s.fillRect(x0 + 1, y - h / 2, s.width - x0 - 1, h, bg);
    s.text(text, x0 + 8, y, fg, this.font("600"));
  }

  private drawTimeAxis(marks: ReturnType<TimeScale["marks"]>): void {
    const s = this.base;
    const t = this.theme;
    const y0 = s.height - this.options.timeAxisHeight;
    s.fillRect(0, y0, s.width, this.options.timeAxisHeight, t.background);
    s.hLine(0, s.width, y0, t.border);
    s.clip({ x: 0, y: y0, width: this.time.width, height: this.options.timeAxisHeight });
    for (const m of marks) {
      s.vLine(m.x, y0, y0 + 4, t.border);
      s.text(m.label, m.x, y0 + this.options.timeAxisHeight / 2 + 1, m.weight >= 3 ? t.axisLabelText : t.text, this.font(m.weight >= 4 ? "600" : undefined), "center");
    }
    s.restore();
  }

  private renderOverlay(): void {
    const s = this.overlay;
    s.clear();
    const c = this.crosshair;
    if (!c) return;
    const pane = this.paneAt(c.y);
    if (!pane || c.x < 0 || c.x > this.time.width) return;
    const t = this.theme;
    // vertical line snaps to the bar centre (TradingView "normal" crosshair)
    const index = clamp(this.time.xToNearestIndex(c.x), 0, Math.max(0, this.time.lastIndex + 500));
    const x = this.time.indexToX(index);
    const plotBottom = s.height - this.options.timeAxisHeight;
    s.vLine(x, 0, plotBottom, t.crosshair, 1, "dashed");
    s.hLine(0, this.time.width, c.y, t.crosshair, 1, "dashed");
    // price label
    const price = this.yToPrice(pane, c.y);
    const h = t.fontSize + 8;
    s.fillRect(this.time.width + 1, c.y - h / 2, s.width - this.time.width - 1, h, t.axisLabelBackground);
    s.text(pane.price.format(price), this.time.width + 8, c.y, t.axisLabelText, this.font());
    // time label
    const time = this.time.indexToTime(index);
    const label = formatTimeLabel(time, this.isIntraday());
    const w = s.measure(label, this.font()) + 16;
    const lx = clamp(x - w / 2, 0, this.time.width - w);
    s.fillRect(lx, plotBottom + 1, w, this.options.timeAxisHeight - 1, t.axisLabelBackground);
    s.text(label, lx + w / 2, plotBottom + this.options.timeAxisHeight / 2 + 1, t.axisLabelText, this.font(), "center");
  }

  private isIntraday(): boolean {
    const a = this.time.timeAt(0);
    const b = this.time.timeAt(1);
    return a !== undefined && b !== undefined && b - a < 86400;
  }

  // ======================================================================
  // Interaction
  // ======================================================================

  private region(x: number, y: number):
    | { kind: "plot"; pane: Pane }
    | { kind: "price"; pane: Pane }
    | { kind: "time" }
    | { kind: "separator"; index: number }
    | { kind: "none" } {
    const timeTop = this.base.height - this.options.timeAxisHeight;
    if (y >= timeTop) return x <= this.time.width ? { kind: "time" } : { kind: "none" };
    for (let i = 0; i < this.panes.length - 1; i++) {
      const edge = this.panes[i]!.rect.y + this.panes[i]!.rect.height;
      if (Math.abs(y - edge) <= 3) return { kind: "separator", index: i };
    }
    const pane = this.paneAt(y);
    if (!pane) return { kind: "none" };
    return x > this.time.width ? { kind: "price", pane } : { kind: "plot", pane };
  }

  private local(event: PointerEvent | MouseEvent | WheelEvent): { x: number; y: number } {
    const r = this.root.getBoundingClientRect();
    return { x: event.clientX - r.left, y: event.clientY - r.top };
  }

  private info(event: PointerEvent, pane: Pane): PointerInfo {
    const p = this.local(event);
    return { ...p, pane, button: event.button, altKey: event.altKey, shiftKey: event.shiftKey, event };
  }

  private attachEvents(target: HTMLCanvasElement): void {
    target.addEventListener("pointerdown", this.onPointerDown);
    target.addEventListener("pointermove", this.onPointerMove);
    target.addEventListener("pointerup", this.onPointerUp);
    target.addEventListener("pointercancel", this.onPointerUp);
    target.addEventListener("pointerleave", this.onPointerLeave);
    target.addEventListener("wheel", this.onWheel, { passive: false });
    target.addEventListener("dblclick", this.onDblClick);
    target.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  private onPointerDown = (event: PointerEvent) => {
    const target = event.currentTarget as HTMLElement;
    this.root.focus({ preventScroll: true });
    const p = this.local(event);
    this.pointers.set(event.pointerId, p);
    cancelAnimationFrame(this.kineticFrame);
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.gesture = { kind: "pinch", distance: Math.hypot(a!.x - b!.x, a!.y - b!.y), spacing: this.time.barSpacing, center: (a!.x + b!.x) / 2 };
      return;
    }
    target.setPointerCapture?.(event.pointerId);
    const region = this.region(p.x, p.y);
    switch (region.kind) {
      case "plot": {
        if (this.interaction?.down?.(this.info(event, region.pane))) {
          this.gesture = { kind: "handler", pane: region.pane };
          return;
        }
        if (event.button !== 0) return;
        this.gesture = {
          kind: "pan",
          startX: p.x,
          startY: p.y,
          offset: this.time.rightOffset,
          pane: region.pane,
          price: region.pane.price.autoScale ? null : region.pane.price.snapshot(),
          samples: [{ x: p.x, t: performance.now() }],
        };
        this.setCursor("grabbing");
        return;
      }
      case "price":
        this.gesture = { kind: "price", pane: region.pane, startY: p.y - region.pane.rect.y, snapshot: region.pane.price.snapshot() };
        return;
      case "time":
        this.gesture = { kind: "time", startX: p.x, spacing: this.time.barSpacing };
        return;
      case "separator": {
        const a = this.panes[region.index]!;
        const b = this.panes[region.index + 1]!;
        this.gesture = { kind: "separator", index: region.index, startY: p.y, weights: [a.weight, b.weight], heights: [a.rect.height, b.rect.height] };
        return;
      }
    }
  };

  private onPointerMove = (event: PointerEvent) => {
    const p = this.local(event);
    if (this.pointers.has(event.pointerId)) this.pointers.set(event.pointerId, p);
    const g = this.gesture;
    if (g?.kind === "pinch" && this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const d = Math.hypot(a!.x - b!.x, a!.y - b!.y);
      const anchor = this.time.xToIndex(g.center);
      this.time.setBarSpacing((g.spacing * d) / Math.max(1, g.distance));
      this.time.rightOffset = anchor - this.time.lastIndex - 0.5 + (this.time.width - g.center) / this.time.barSpacing;
      this.time.clampOffset();
      return this.changed();
    }
    if (g) {
      switch (g.kind) {
        case "handler":
          this.interaction?.move?.(this.info(event, g.pane));
          this.crosshair = p;
          this.invalidate(false);
          return;
        case "pan":
          this.time.scrollFrom(g.offset, p.x - g.startX);
          if (g.price) g.pane.price.scrollFrom(g.price, p.y - g.startY);
          g.samples.push({ x: p.x, t: performance.now() });
          if (g.samples.length > 6) g.samples.shift();
          this.crosshair = p;
          return this.changed();
        case "price":
          g.pane.price.scaleFromDrag(g.snapshot, g.startY, p.y - g.pane.rect.y);
          return this.changed();
        case "time":
          this.time.scaleFromDrag(g.startX, p.x, g.spacing);
          return this.changed();
        case "separator": {
          const dy = p.y - g.startY;
          const total = g.heights[0] + g.heights[1];
          const h0 = clamp(g.heights[0] + dy, 30, total - 30);
          const weightSum = g.weights[0] + g.weights[1];
          this.panes[g.index]!.weight = (h0 / total) * weightSum;
          this.panes[g.index + 1]!.weight = ((total - h0) / total) * weightSum;
          this.layout();
          return this.changed();
        }
      }
    }
    // hover
    this.crosshair = p;
    const region = this.region(p.x, p.y);
    let cursor = "crosshair";
    if (region.kind === "price") cursor = "ns-resize";
    else if (region.kind === "time") cursor = "ew-resize";
    else if (region.kind === "separator") cursor = "row-resize";
    else if (region.kind === "plot") cursor = this.interaction?.hover?.(this.info(event, region.pane)) ?? "crosshair";
    this.setCursor(cursor);
    this.invalidate(false);
    this.emitCrosshair();
  };

  private onPointerUp = (event: PointerEvent) => {
    const g = this.gesture;
    this.pointers.delete(event.pointerId);
    if (g?.kind === "handler") this.interaction?.up?.(this.info(event, g.pane));
    if (g?.kind === "pan" && this.options.kinetic) this.startKinetic(g.samples);
    if (g?.kind !== "pinch" || this.pointers.size === 0) this.gesture = null;
    this.setCursor("crosshair");
  };

  private onPointerLeave = () => {
    if (this.gesture) return;
    this.crosshair = null;
    this.invalidate(false);
    this.emitCrosshair();
  };

  private onWheel = (event: WheelEvent) => {
    event.preventDefault();
    cancelAnimationFrame(this.kineticFrame);
    const p = this.local(event);
    // Chromium on Windows scales deltas by the pixel ratio; undo that
    const speed =
      event.deltaMode === 1 ? 32 : event.deltaMode === 2 ? 120 : /Win/.test(navigator.platform) && /Chrome/.test(navigator.userAgent) ? 1 / (window.devicePixelRatio || 1) : 1;
    const dy = (-event.deltaY * speed) / 100;
    const dx = (event.deltaX * speed) / 100;
    if (dy !== 0) this.time.zoom(clamp(p.x, 0, this.time.width), Math.sign(dy) * Math.min(1, Math.abs(dy)));
    if (dx !== 0) this.time.scrollFrom(this.time.rightOffset, -dx * 80);
    this.changed();
  };

  private onDblClick = (event: MouseEvent) => {
    const p = this.local(event);
    const region = this.region(p.x, p.y);
    if (region.kind === "price") {
      region.pane.price.autoScale = true;
      return this.changed();
    }
    if (region.kind === "time") {
      this.time.reset();
      return this.changed();
    }
    if (region.kind === "plot" && this.interaction?.dblclick?.({ ...p, pane: region.pane, button: 0, altKey: event.altKey, shiftKey: event.shiftKey, event: event as PointerEvent })) return;
  };

  private startKinetic(samples: { x: number; t: number }[]): void {
    if (samples.length < 2) return;
    const a = samples[0]!;
    const b = samples[samples.length - 1]!;
    const dt = b.t - a.t;
    if (dt <= 0 || performance.now() - b.t > 60) return;
    let velocity = (b.x - a.x) / dt; // px per ms
    if (Math.abs(velocity) < 0.2) return;
    let last = performance.now();
    const step = (now: number) => {
      const elapsed = now - last;
      last = now;
      this.time.scrollFrom(this.time.rightOffset, velocity * elapsed);
      velocity *= Math.pow(0.996, elapsed); // smooth decay
      this.changed();
      if (Math.abs(velocity) > 0.02) this.kineticFrame = requestAnimationFrame(step);
    };
    this.kineticFrame = requestAnimationFrame(step);
  }

  private changed(): void {
    this.invalidate();
    this.emitView();
    this.emitCrosshair();
  }

  private setCursor(cursor: string): void {
    const el = this.overlay.canvas;
    if (el.style.cursor !== cursor) el.style.cursor = cursor;
  }

  private emitView(): void {
    for (const l of this.viewListeners) l();
  }

  private emitCrosshair(): void {
    if (!this.crosshairListeners.size) return;
    const c = this.crosshair;
    const pane = c ? this.paneAt(c.y) : undefined;
    if (!c || !pane || c.x > this.time.width) {
      for (const l of this.crosshairListeners) l(null);
      return;
    }
    const index = this.time.xToNearestIndex(c.x);
    const info: CrosshairInfo = {
      x: c.x,
      y: c.y,
      index,
      time: this.time.timeAt(index),
      paneId: pane.id,
      price: this.yToPrice(pane, c.y),
      bar: this.mainSeries?.barAt(index),
    };
    for (const l of this.crosshairListeners) l(info);
  }
}

function drawMarkerShape(s: Surface, m: Marker, x: number, y: number, size: number): void {
  const half = size / 2;
  switch (m.shape) {
    case "circle": {
      const ctx = s.ctx;
      ctx.fillStyle = m.color;
      ctx.beginPath();
      ctx.arc(x * s.hpr, y * s.vpr, half * s.hpr * 0.8, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    case "arrowUp":
    case "triangleUp":
      s.fillPolygon([{ x, y: y - half }, { x: x + half, y: y + half }, { x: x - half, y: y + half }], m.color);
      return;
    case "arrowDown":
    case "triangleDown":
      s.fillPolygon([{ x, y: y + half }, { x: x + half, y: y - half }, { x: x - half, y: y - half }], m.color);
      return;
  }
}
