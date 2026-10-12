// Right sidebar (TradingView): a toolbar of panel buttons on the far right
// and one panel open at a time next to it (watchlist, layers…), so opening
// panels never squeezes the chart twice. Width is dragged from the panel's
// left edge and remembered.

import { useEffect, useRef, useState, type ReactNode } from "react";
import { btn } from "./drawing-ui";

export type SidePanel = {
  id: string;
  title: string;
  icon: ReactNode;
  /** Content; `close` hides the panel. */
  render: (close: () => void) => ReactNode;
  /** Button sits at the bottom of the right toolbar. */
  bottom?: boolean;
  /** Button sits at the bottom of the left drawing toolbar instead (圖表設定). */
  inToolbar?: boolean;
};

const ACTIVE_PREF = "smartdca.chart.sidePanel";
const WIDTH_PREF = "smartdca.chart.sideWidth";
const MIN_W = 220;
const MAX_W = 520;

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function write(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // private mode: not remembered
  }
}

export function useSidePanel(panels: SidePanel[], fallback: string | null): [string | null, (id: string | null) => void] {
  const [active, setActive] = useState<string | null>(() => {
    const saved = read(ACTIVE_PREF);
    if (saved === "none") return null;
    if (saved && panels.some((p) => p.id === saved)) return saved;
    return fallback;
  });
  const set = (id: string | null) => {
    setActive(id);
    write(ACTIVE_PREF, id ?? "none");
  };
  return [active && panels.some((p) => p.id === active) ? active : null, set];
}

export function SideBar({ panels, active, onSelect }: { panels: SidePanel[]; active: string | null; onSelect: (id: string | null) => void }) {
  const [width, setWidth] = useState(() => {
    const v = Number(read(WIDTH_PREF));
    return v >= MIN_W && v <= MAX_W ? v : 340;
  });
  const drag = useRef<{ x: number; w: number } | null>(null);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (!drag.current) return;
      setWidth(Math.max(MIN_W, Math.min(MAX_W, drag.current.w + (drag.current.x - e.clientX))));
    };
    const up = () => {
      if (!drag.current) return;
      drag.current = null;
      document.body.style.cursor = "";
      setWidth((w) => {
        write(WIDTH_PREF, String(Math.round(w)));
        return w;
      });
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, []);

  const panel = panels.find((p) => p.id === active);
  return (
    <>
      {panel ? (
        <aside
          className="relative flex h-full min-h-0 max-w-[85vw] shrink-0 flex-col border-l border-[var(--c-line)] bg-[var(--c-panel)] text-[var(--c-text)]"
          style={{ width }}
          data-testid={`side-${panel.id}`}
        >
          <div
            className="absolute -left-1 top-0 z-10 h-full w-2 cursor-col-resize hover:bg-[var(--c-accent-soft)]"
            title="拖曳調整寬度"
            data-testid="side-resize"
            onPointerDown={(e) => {
              e.preventDefault();
              drag.current = { x: e.clientX, w: width };
              document.body.style.cursor = "col-resize";
            }}
          />
          {panel.render(() => onSelect(null))}
        </aside>
      ) : null}
      <nav className="flex h-full w-11 shrink-0 flex-col items-center gap-0.5 border-l border-[var(--c-line)] bg-[var(--c-panel)] py-1.5" data-testid="right-toolbar">
        {[...panels.filter((p) => !p.bottom && !p.inToolbar), ...panels.filter((p) => p.bottom && !p.inToolbar)].map((p, i, list) => (
          <button
            key={p.id}
            className={`${btn(p.id === active)} ${p.bottom && !list[i - 1]?.bottom ? "mt-auto" : ""}`}
            title={p.id === active ? `關閉${p.title}` : p.title}
            aria-pressed={p.id === active}
            data-testid={`toggle-${p.id}`}
            onClick={() => onSelect(p.id === active ? null : p.id)}
          >
            {p.icon}
          </button>
        ))}
      </nav>
    </>
  );
}

/** Shared header of a sidebar panel: title, count, actions, close. */
export function PanelHeader({ title, count, children, onClose }: { title: string; count?: number; children?: ReactNode; onClose: () => void }) {
  return (
    <div className="flex h-11 shrink-0 items-center gap-1 border-b border-[var(--c-line)] pl-3 pr-1.5">
      <span className="text-[13px] font-semibold">{title}</span>
      {count !== undefined ? <span className="ml-1 text-[11px] tabular-nums text-[var(--c-muted)]">{count}</span> : null}
      <div className="ml-auto flex items-center gap-0.5">{children}</div>
      <button className={panelIconBtn()} title="關閉" onClick={onClose}>
        <svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
          <path d="M3 3l8 8M11 3l-8 8" />
        </svg>
      </button>
    </div>
  );
}

/** Small square icon button used inside panels (rows, headers). */
export const panelIconBtn = (on = false) =>
  `flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md ring-1 ring-inset transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--c-accent)] disabled:cursor-not-allowed disabled:opacity-30 [&_svg]:h-4 [&_svg]:w-4 ${
    on ? "bg-[var(--c-accent-soft)] text-[var(--c-accent)] ring-[var(--c-accent)]" : "text-[var(--c-muted)] ring-transparent hover:bg-[var(--c-hover)] hover:text-[var(--c-text)] hover:ring-[var(--c-line)]"
  }`;
