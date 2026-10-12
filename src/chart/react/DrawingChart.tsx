// Chart + drawing tools in one component: left toolbar, floating properties
// bar, dialogs, and localStorage persistence per `storageKey` (e.g. symbol).
import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { DrawingLayer } from "../drawings/layer";
import { persistLayer } from "../drawings/storage";
import { getTool } from "../drawings/tools";
import type { Drawing } from "../drawings/types";
import type { Chart, ChartOptions } from "../engine/chart";
import { DARK_THEME, type Theme } from "../engine/types";
import { DrawingToolbar, FibSettingsDialog, PropertiesBar, TextDialog, VolumeSettingsDialog } from "./drawing-ui";
import { LayersPanel } from "./LayersPanel";
import { TradingChart } from "./TradingChart";

/** CSS variables the toolbar / panels / dialogs are styled with. */
export function themeVars(t: Theme): CSSProperties {
  return {
    "--c-bg": t.background,
    "--c-panel": t.ui.panel,
    "--c-raised": t.ui.raised,
    "--c-text": t.ui.text,
    "--c-muted": t.ui.muted,
    "--c-line": t.ui.line,
    "--c-hover": t.ui.hover,
    "--c-accent": t.ui.accent,
    "--c-accent-soft": t.ui.accentSoft,
    "--c-accent-text": t.ui.accentText,
    "--c-danger": t.down,
    "--c-drawing": t.drawing,
    "--c-shadow": t.ui.shadow,
  } as CSSProperties;
}

const LAYERS_PREF = "smartdca.chart.layersOpen";

export type DrawingChartProps = {
  onReady: (chart: Chart, layer: DrawingLayer) => void | (() => void);
  /** Drawings are saved per key; omit to keep them in memory only. */
  storageKey?: string;
  options?: Partial<ChartOptions>;
  className?: string;
  style?: CSSProperties;
  /** Hide the left toolbar (e.g. on phones). */
  toolbar?: boolean;
  /** Light / dark; switching repaints the chart, drawings and all panels. */
  theme?: Theme;
  /** Overlays (legends, info panels) positioned over the chart area only. */
  children?: ReactNode;
};

type DialogState = { kind: "fib" | "volume" | "text"; id: string } | null;

export function DrawingChart({ onReady, storageKey, options, className, style, toolbar = true, theme = DARK_THEME, children }: DrawingChartProps) {
  const [layer, setLayer] = useState<DrawingLayer | null>(null);
  const [dialog, setDialog] = useState<DialogState>(null);
  const [layersOpen, setLayersOpen] = useState(() => {
    try {
      return localStorage.getItem(LAYERS_PREF) === "1";
    } catch {
      return false;
    }
  });
  const toggleLayers = useCallback(() => {
    setLayersOpen((open) => {
      try {
        localStorage.setItem(LAYERS_PREF, open ? "0" : "1");
      } catch {
        // private mode: just not remembered
      }
      return !open;
    });
  }, []);
  const themeRef = useRef(theme);
  themeRef.current = theme;
  const ready = useRef(onReady);
  ready.current = onReady;

  const handleReady = useCallback((chart: Chart) => {
    chart.setTheme(themeRef.current);
    const l = new DrawingLayer(chart);
    const offEvent = l.onEvent((e) => {
      // text-type tools ask for their text right away (TradingView)
      if (e.type === "editText" || (e.type === "created" && (e.item.type === "text" || e.item.type === "callout")))
        setDialog({ kind: "text", id: e.item.id });
    });
    setLayer(l);
    const cleanup = ready.current(chart, l);
    return () => {
      offEvent();
      if (typeof cleanup === "function") cleanup();
      l.dispose();
      setLayer(null);
    };
  }, []);

  useEffect(() => {
    layer?.chart.setTheme(theme);
  }, [layer, theme]);

  // (re)load drawings when the key changes
  useEffect(() => {
    if (!layer || !storageKey) return;
    return persistLayer(layer, storageKey);
  }, [layer, storageKey]);

  const item: Drawing | undefined = dialog ? layer?.items.find((d) => d.id === dialog.id) : undefined;
  const close = useCallback(() => {
    setDialog(null);
    layer?.chart.root.focus();
  }, [layer]);

  return (
    <div className={`flex min-h-0 ${className ?? ""}`} style={{ ...themeVars(theme), background: theme.background, ...style }}>
      {toolbar && layer ? <DrawingToolbar layer={layer} layersOpen={layersOpen} onToggleLayers={toggleLayers} /> : null}
      <div className="relative min-h-0 min-w-0 flex-1" data-testid="chart-area">
        <TradingChart className="h-full w-full" options={options} onReady={handleReady} />
        {children}
        {layer ? <PropertiesBar layer={layer} onOpenDialog={(kind, d) => setDialog({ kind, id: d.id })} /> : null}
      </div>
      {layer && layersOpen ? <LayersPanel layer={layer} onClose={toggleLayers} /> : null}
      {layer && item && dialog?.kind === "fib" && getTool(item.type)?.caps?.dialog === "fib" ? <FibSettingsDialog layer={layer} item={item} onClose={close} /> : null}
      {layer && item && dialog?.kind === "volume" ? <VolumeSettingsDialog layer={layer} item={item} onClose={close} /> : null}
      {layer && item && dialog?.kind === "text" ? <TextDialog layer={layer} item={item} onClose={close} /> : null}
    </div>
  );
}
