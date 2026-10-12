// Chart + drawing tools in one component: left toolbar, floating properties
// bar, dialogs, and localStorage persistence per `storageKey` (e.g. symbol).
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { DrawingLayer } from "../drawings/layer";
import { persistLayer } from "../drawings/storage";
import { getTool } from "../drawings/tools";
import type { Drawing } from "../drawings/types";
import type { Chart, ChartOptions } from "../engine/chart";
import { DrawingToolbar, FibSettingsDialog, PropertiesBar, TextDialog, VolumeSettingsDialog } from "./drawing-ui";
import { TradingChart } from "./TradingChart";

export type DrawingChartProps = {
  onReady: (chart: Chart, layer: DrawingLayer) => void | (() => void);
  /** Drawings are saved per key; omit to keep them in memory only. */
  storageKey?: string;
  options?: Partial<ChartOptions>;
  className?: string;
  style?: CSSProperties;
  /** Hide the left toolbar (e.g. on phones). */
  toolbar?: boolean;
};

type DialogState = { kind: "fib" | "volume" | "text"; id: string } | null;

export function DrawingChart({ onReady, storageKey, options, className, style, toolbar = true }: DrawingChartProps) {
  const [layer, setLayer] = useState<DrawingLayer | null>(null);
  const [dialog, setDialog] = useState<DialogState>(null);
  const ready = useRef(onReady);
  ready.current = onReady;

  const handleReady = useCallback((chart: Chart) => {
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
    <div className={`flex min-h-0 ${className ?? ""}`} style={style}>
      {toolbar && layer ? <DrawingToolbar layer={layer} /> : null}
      <div className="relative min-h-0 min-w-0 flex-1">
        <TradingChart className="h-full w-full" options={options} onReady={handleReady} />
        {layer ? <PropertiesBar layer={layer} onOpenDialog={(kind, d) => setDialog({ kind, id: d.id })} /> : null}
      </div>
      {layer && item && dialog?.kind === "fib" && getTool(item.type)?.caps?.dialog === "fib" ? <FibSettingsDialog layer={layer} item={item} onClose={close} /> : null}
      {layer && item && dialog?.kind === "volume" ? <VolumeSettingsDialog layer={layer} item={item} onClose={close} /> : null}
      {layer && item && dialog?.kind === "text" ? <TextDialog layer={layer} item={item} onClose={close} /> : null}
    </div>
  );
}
