// Chart + drawing tools in one component: left toolbar, floating properties
// bar, dialogs, and localStorage persistence per `storageKey` (e.g. symbol).
import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { DrawingLayer } from "../drawings/layer";
import { persistLayer } from "../drawings/storage";
import { getTool } from "../drawings/tools";
import type { Drawing } from "../drawings/types";
import type { Chart, ChartOptions } from "../engine/chart";
import { DARK_THEME, type Theme } from "../engine/types";
import { btn, DrawingToolbar, FibSettingsDialog, PropertiesBar, TextDialog, VolumeSettingsDialog } from "./drawing-ui";
import { Icon } from "./icons";
import { LayersPanel, type FixedLayer } from "./LayersPanel";
import { SideBar, usePhone, useSidePanel, type SidePanel } from "./SideBar";
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
  /** Extra right-sidebar panels (e.g. a watchlist); 圖層 is always added. */
  sidePanels?: SidePanel[];
  /** Panel open the first time (before the user picks one). */
  defaultSidePanel?: string | null;
  /** Chart series listed as permanent layers under the drawings (K線). */
  fixedLayers?: FixedLayer[];
};

type DialogState = { kind: "fib" | "volume" | "text"; id: string } | null;

export function DrawingChart({ onReady, storageKey, options, className, style, toolbar = true, theme = DARK_THEME, children, sidePanels = [], defaultSidePanel = null, fixedLayers = [] }: DrawingChartProps) {
  const [layer, setLayer] = useState<DrawingLayer | null>(null);
  const [dialog, setDialog] = useState<DialogState>(null);
  const panels: SidePanel[] = [
    ...sidePanels,
    {
      id: "layers",
      title: "圖層",
      icon: <Icon name="layers" />,
      render: (close) => (layer ? <LayersPanel layer={layer} onClose={close} fixed={fixedLayers} /> : null),
    },
  ];
  const phone = usePhone();
  const [sidePanel, setSidePanel] = useSidePanel(panels, defaultSidePanel, phone);
  // phone: the drawing toolbar and the panel buttons are floating, opened on demand
  const [toolsOpen, setToolsOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    if (!phone || !layer) return;
    // picking a tool closes the toolbar so the chart is free to draw on
    return layer.subscribe(() => {
      if (layer.activeTool) setToolsOpen(false);
    });
  }, [phone, layer]);
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
    // root keeps the caller's positioning (the chart page passes "absolute inset-0");
    // phone panels are absolutely positioned against it
    <div className={`chart-no-select flex min-h-0 ${className?.includes("absolute") || className?.includes("fixed") ? "" : "relative"} ${className ?? ""}`} style={{ ...themeVars(theme), background: theme.background, ...style }}>
      {toolbar && layer && !phone ? (
        <DrawingToolbar
          layer={layer}
          footer={panels
            .filter((p) => p.inToolbar)
            .map((p) => (
              <button
                key={p.id}
                className={btn(p.id === sidePanel)}
                title={p.id === sidePanel ? `關閉${p.title}` : p.title}
                aria-pressed={p.id === sidePanel}
                data-testid={`toggle-${p.id}`}
                onClick={() => setSidePanel(p.id === sidePanel ? null : p.id)}
              >
                {p.icon}
              </button>
            ))}
        />
      ) : null}
      <div className="relative min-h-0 min-w-0 flex-1" data-testid="chart-area">
        <TradingChart className="h-full w-full" options={options} onReady={handleReady} />
        {children}
        {layer ? <PropertiesBar layer={layer} onOpenDialog={(kind, d) => setDialog({ kind, id: d.id })} /> : null}
        {phone && layer ? (
          <>
            {toolbar && toolsOpen ? (
              <div className="absolute inset-y-0 left-0 z-40 shadow-2xl" data-testid="phone-tools">
                <DrawingToolbar layer={layer} />
              </div>
            ) : null}
            {toolbar ? (
              <button
                className={`absolute bottom-9 left-2 z-40 flex h-10 w-10 items-center justify-center rounded-full border border-[var(--c-line)] shadow-lg ${toolsOpen || layer.activeTool ? "bg-[var(--c-accent)] text-[var(--c-accent-text)]" : "bg-[var(--c-panel)] text-[var(--c-text)]"}`}
                title={toolsOpen ? "收起畫線工具" : "畫線工具"}
                data-testid="phone-tools-toggle"
                onClick={() => setToolsOpen(!toolsOpen)}
              >
                <Icon name="trend_line" />
              </button>
            ) : null}
            {panels.length ? (
              <div className="absolute bottom-9 right-[72px] z-40 flex flex-col items-end gap-1.5">
                {menuOpen
                  ? panels.map((p) => (
                      <button
                        key={p.id}
                        className="flex h-9 items-center gap-2 rounded-full border border-[var(--c-line)] bg-[var(--c-panel)] pl-3 pr-2 text-[12.5px] text-[var(--c-text)] shadow-lg"
                        data-testid={`toggle-${p.id}`}
                        onClick={() => {
                          setMenuOpen(false);
                          setSidePanel(p.id);
                        }}
                      >
                        {p.title.replace(/（.*）/, "")}
                        {p.icon}
                      </button>
                    ))
                  : null}
                <button
                  className={`flex h-10 w-10 items-center justify-center rounded-full border border-[var(--c-line)] shadow-lg ${menuOpen ? "bg-[var(--c-accent)] text-[var(--c-accent-text)]" : "bg-[var(--c-panel)] text-[var(--c-text)]"}`}
                  title="觀察清單、圖層、圖表設定…"
                  data-testid="phone-menu"
                  onClick={() => setMenuOpen(!menuOpen)}
                >
                  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M3 5h12M3 9h12M3 13h12" /></svg>
                </button>
              </div>
            ) : null}
          </>
        ) : null}
      </div>
      <SideBar panels={panels} active={sidePanel} onSelect={setSidePanel} phone={phone} />
      {layer && item && dialog?.kind === "fib" && getTool(item.type)?.caps?.dialog === "fib" ? <FibSettingsDialog layer={layer} item={item} onClose={close} /> : null}
      {layer && item && dialog?.kind === "volume" ? <VolumeSettingsDialog layer={layer} item={item} onClose={close} /> : null}
      {layer && item && dialog?.kind === "text" ? <TextDialog layer={layer} item={item} onClose={close} /> : null}
    </div>
  );
}
