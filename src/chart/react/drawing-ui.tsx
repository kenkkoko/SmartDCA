// Drawing UI: left toolbar, floating properties bar, colour picker and the
// settings dialogs (Fibonacci, volume profile, text). All state lives in the
// DrawingLayer; React only subscribes to its version counter.

import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import type { DrawingLayer } from "../drawings/layer";
import { saveFibTemplate } from "../drawings/storage";
import type { Tool } from "../drawings/tool";
import { getTool, TOOL_GROUPS } from "../drawings/tools";
import { getFibDefaults } from "../drawings/tools/fibonacci";
import { defaultFibSettings, type Drawing, type FibSettings } from "../drawings/types";
import type { LineStyle } from "../engine/types";
import { Icon } from "./icons";

export function useLayer(layer: DrawingLayer | null): number {
  return useSyncExternalStore(
    (cb) => (layer ? layer.subscribe(cb) : () => undefined),
    () => layer?.version ?? 0,
  );
}

function useOutside(open: boolean, close: () => void): React.RefObject<HTMLDivElement> {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown, true);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, close]);
  return ref;
}

const btn = (active = false) =>
  `flex h-9 w-9 items-center justify-center rounded-md transition-colors ${
    active ? "bg-[#2962ff]/25 text-[#7aa2ff]" : "text-slate-300 hover:bg-white/10 hover:text-white"
  }`;

// ===========================================================================
// Left toolbar
// ===========================================================================

export function DrawingToolbar({ layer }: { layer: DrawingLayer }) {
  useLayer(layer);
  // each group button shows the tool last picked from it (TradingView)
  const [lastInGroup, setLastInGroup] = useState<Record<string, string>>({});
  const [openGroup, setOpenGroup] = useState<{ group: string; top: number; left: number } | null>(null);
  const close = useMemo(() => () => setOpenGroup(null), []);
  const menuRef = useOutside(openGroup !== null, close);
  const allHidden = layer.items.length > 0 && layer.items.every((d) => d.hidden);

  const pick = (group: string, tool: Tool) => {
    setLastInGroup((m) => ({ ...m, [group]: tool.type }));
    layer.setActiveTool(tool.type);
    setOpenGroup(null);
    layer.chart.root.focus();
  };

  return (
    <div className="flex h-full w-11 shrink-0 flex-col items-center gap-0.5 overflow-y-auto border-r border-white/10 bg-[#0e1219] py-1.5" data-testid="drawing-toolbar">
      <button className={btn(!layer.activeTool)} title="游標" onClick={() => layer.setActiveTool(null)}>
        <Icon name="cursor" />
      </button>
      <div className="my-1 h-px w-6 bg-white/10" />
      {TOOL_GROUPS.map((g) => {
        const current = getTool(lastInGroup[g.group] ?? "") ?? g.tools[0]!;
        const active = g.tools.some((t) => t.type === layer.activeTool);
        const shown = active ? getTool(layer.activeTool!)! : current;
        return (
          <div key={g.group} className="group relative">
            <button
              className={btn(active)}
              title={shown.label}
              data-tool={shown.type}
              onClick={() => (active ? layer.setActiveTool(null) : pick(g.group, shown))}
            >
              <Icon name={shown.type} />
            </button>
            {g.tools.length > 1 ? (
              <button
                className="absolute -right-1 bottom-0 flex h-4 w-3.5 items-end justify-center text-[9px] leading-none text-slate-500 hover:text-white"
                title={`${g.label}…`}
                data-group={g.group}
                onClick={(e) => {
                  // the toolbar scrolls, so the flyout is placed with fixed coordinates
                  const r = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
                  setOpenGroup(openGroup?.group === g.group ? null : { group: g.group, top: r.top, left: r.right + 4 });
                }}
              >
                ▸
              </button>
            ) : null}
            {openGroup?.group === g.group ? (
              <div ref={menuRef} style={{ top: Math.min(openGroup.top, window.innerHeight - 40 - g.tools.length * 34), left: openGroup.left }} className="fixed z-30 w-52 rounded-lg border border-white/10 bg-[#161b25] py-1 shadow-2xl">
                <div className="px-3 pb-1 pt-1.5 text-[11px] font-medium text-slate-500">{g.label}</div>
                {g.tools.map((t) => (
                  <button
                    key={t.type}
                    className={`flex w-full items-center gap-2.5 px-3 py-1.5 text-left text-[13px] ${
                      t.type === layer.activeTool ? "text-[#7aa2ff]" : "text-slate-200 hover:bg-white/10"
                    }`}
                    data-tool-item={t.type}
                    onClick={() => pick(g.group, t)}
                  >
                    <Icon name={t.type} />
                    {t.label}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
      <div className="my-1 h-px w-6 bg-white/10" />
      <button className={btn(layer.magnet)} title="磁鐵模式（吸附開高低收）" onClick={() => layer.setMagnet(!layer.magnet)}>
        <Icon name="magnet" />
      </button>
      <button className={btn(layer.keepDrawing)} title="保持繪圖模式" onClick={() => layer.setKeepDrawing(!layer.keepDrawing)}>
        <Icon name="keep" />
      </button>
      <button className={btn(allHidden)} title={allHidden ? "顯示所有繪圖" : "隱藏所有繪圖"} onClick={() => layer.setAllHidden(!allHidden)}>
        <Icon name={allHidden ? "eyeOff" : "eye"} />
      </button>
      <button className={`${btn()} disabled:opacity-30`} title="復原 (Ctrl+Z)" disabled={!layer.canUndo} onClick={() => layer.undo()}>
        <Icon name="undo" />
      </button>
      <button className={`${btn()} disabled:opacity-30`} title="重做 (Ctrl+Y)" disabled={!layer.canRedo} onClick={() => layer.redo()}>
        <Icon name="redo" />
      </button>
      <button
        className={`${btn()} disabled:opacity-30`}
        title="移除所有繪圖"
        disabled={!layer.items.length}
        onClick={() => {
          if (window.confirm(`移除全部 ${layer.items.length} 個繪圖？（可用 Ctrl+Z 復原）`)) layer.removeAll();
        }}
      >
        <Icon name="trash" />
      </button>
    </div>
  );
}

// ===========================================================================
// Colour picker
// ===========================================================================

const PALETTE = [
  ["#ffffff", "#d1d4dc", "#b2b5be", "#9598a1", "#787b86", "#5d606b", "#434651", "#2a2e39", "#131722", "#000000"],
  ["#f23645", "#ff9800", "#ffeb3b", "#4caf50", "#089981", "#00bcd4", "#2962ff", "#673ab7", "#9c27b0", "#e91e63"],
  ["#fccbcd", "#ffe0b2", "#fff9c4", "#c8e6c9", "#ace5dc", "#b2ebf2", "#bbd9fb", "#d1c4e9", "#e1bee7", "#f8bbd0"],
  ["#faa1a4", "#ffcc80", "#fff59d", "#a5d6a7", "#70ccbd", "#80deea", "#90bff9", "#b39ddb", "#ce93d8", "#f48fb1"],
  ["#f7525f", "#ffb74d", "#fff176", "#81c784", "#42bda8", "#4dd0e1", "#5b9cf6", "#9575cd", "#ba68c8", "#f06292"],
  ["#b22833", "#f57c00", "#fbc02d", "#388e3c", "#056656", "#0097a7", "#1848cc", "#512da8", "#7b1fa2", "#c2185b"],
  ["#801922", "#e65100", "#f57f17", "#1b5e20", "#00332a", "#006064", "#0c3299", "#311b92", "#4a148c", "#880e4f"],
];

export function ColorButton({
  color,
  onChange,
  opacity,
  onOpacity,
  title,
  allowNone,
  children,
}: {
  color: string | undefined;
  onChange: (c: string | undefined) => void;
  opacity?: number;
  onOpacity?: (v: number) => void;
  title?: string;
  allowNone?: boolean;
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const close = useMemo(() => () => setOpen(false), []);
  const ref = useOutside(open, close);
  return (
    <div className="relative" ref={ref}>
      <button className="flex h-7 min-w-7 items-center justify-center gap-1 rounded-md px-1.5 hover:bg-white/10" title={title} onClick={() => setOpen(!open)}>
        {children}
        <span
          className="h-4 w-4 rounded-sm border border-white/30"
          style={{ background: color ?? "transparent", backgroundImage: color ? undefined : "linear-gradient(135deg,transparent 45%,#f23645 45%,#f23645 55%,transparent 55%)" }}
        />
      </button>
      {open ? (
        <div className="absolute left-0 top-9 z-40 w-[232px] rounded-lg border border-white/10 bg-[#161b25] p-2.5 shadow-2xl">
          <div className="grid grid-cols-10 gap-[3px]">
            {PALETTE.flat().map((c) => (
              <button
                key={c}
                className={`h-[19px] w-[19px] rounded-sm ${c === color ? "ring-2 ring-[#2962ff] ring-offset-1 ring-offset-[#161b25]" : ""}`}
                style={{ background: c }}
                onClick={() => onChange(c)}
              />
            ))}
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs text-slate-400">
            <label className="flex items-center gap-1.5">
              自訂
              <input type="color" className="h-6 w-8 cursor-pointer bg-transparent" value={color?.startsWith("#") && color.length === 7 ? color : "#2962ff"} onChange={(e) => onChange(e.target.value)} />
            </label>
            {allowNone ? (
              <button className="ml-auto rounded px-2 py-0.5 hover:bg-white/10" onClick={() => onChange(undefined)}>
                無
              </button>
            ) : null}
          </div>
          {onOpacity ? (
            <label className="mt-2 flex items-center gap-2 text-xs text-slate-400">
              透明度
              <input type="range" min={0} max={100} value={Math.round((opacity ?? 1) * 100)} onChange={(e) => onOpacity(Number(e.target.value) / 100)} className="flex-1" />
              <span className="w-8 text-right tabular-nums">{Math.round((opacity ?? 1) * 100)}%</span>
            </label>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function Select<T extends string | number>({ value, options, onChange, title }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; title?: string }) {
  const [open, setOpen] = useState(false);
  const close = useMemo(() => () => setOpen(false), []);
  const ref = useOutside(open, close);
  const current = options.find((o) => o.value === value);
  return (
    <div className="relative" ref={ref}>
      <button className="flex h-8 items-center gap-1 rounded-md px-2 text-xs text-slate-200 hover:bg-white/10" title={title} onClick={() => setOpen(!open)}>
        {current?.label}
        <span className="text-[9px] text-slate-500">▾</span>
      </button>
      {open ? (
        <div className="absolute left-0 top-9 z-40 min-w-[120px] rounded-lg border border-white/10 bg-[#161b25] py-1 shadow-2xl">
          {options.map((o) => (
            <button
              key={String(o.value)}
              className={`flex w-full items-center px-3 py-1.5 text-left text-xs ${o.value === value ? "text-[#7aa2ff]" : "text-slate-200 hover:bg-white/10"}`}
              onClick={() => {
                onChange(o.value);
                setOpen(false);
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

const lineSample = (style: LineStyle, width = 1) => (
  <svg width="28" height="10">
    <line x1="1" y1="5" x2="27" y2="5" stroke="currentColor" strokeWidth={width} strokeDasharray={style === "dashed" ? "5 3" : style === "dotted" ? "1.5 2.5" : undefined} />
  </svg>
);

// ===========================================================================
// Floating properties bar
// ===========================================================================

export function PropertiesBar({ layer, onOpenDialog }: { layer: DrawingLayer; onOpenDialog: (kind: "fib" | "volume" | "text", item: Drawing) => void }) {
  useLayer(layer);
  const item = layer.selected;
  if (!item) return null;
  const tool = getTool(item.type);
  if (!tool) return null;
  const caps = tool.caps ?? {};
  const set = (change: (d: Drawing) => void) => layer.update(item.id, change);
  const st = item.style;
  const sep = <div className="mx-0.5 h-5 w-px bg-white/10" />;

  return (
    <div
      className="absolute left-1/2 top-2 z-20 flex -translate-x-1/2 items-center gap-0.5 rounded-lg border border-white/10 bg-[#161b25]/95 px-1.5 py-1 text-slate-200 shadow-xl backdrop-blur"
      data-testid="properties-bar"
      onPointerDown={(e) => e.stopPropagation()}
    >
      <span className="whitespace-nowrap px-1.5 text-xs text-slate-400">{tool.label}</span>
      {sep}
      {caps.line !== false ? (
        <>
          <ColorButton title="線條顏色" color={st.color} onChange={(c) => c && set((d) => void (d.style.color = c))}>
            <svg width="14" height="14" viewBox="0 0 14 14"><path d="M2 12L10 4l2 2-8 8H2z" fill="none" stroke="currentColor" strokeWidth="1.2" /></svg>
          </ColorButton>
          <Select
            title="線寬"
            value={st.width}
            options={[1, 2, 3, 4].map((w) => ({ value: w, label: <span className="flex items-center gap-1.5">{lineSample("solid", w)}{w}px</span> }))}
            onChange={(w) => set((d) => void (d.style.width = w))}
          />
          <Select
            title="線條樣式"
            value={st.lineStyle}
            options={(["solid", "dashed", "dotted"] as LineStyle[]).map((s) => ({ value: s, label: lineSample(s) }))}
            onChange={(s) => set((d) => void (d.style.lineStyle = s))}
          />
        </>
      ) : (
        <ColorButton title="顏色" color={st.color} onChange={(c) => c && set((d) => void (d.style.color = c))} />
      )}
      {caps.fill ? (
        <ColorButton
          title="背景填色"
          allowNone
          color={st.fillColor}
          opacity={st.fillOpacity}
          onChange={(c) => set((d) => void (d.style.fillColor = c))}
          onOpacity={(v) => set((d) => void (d.style.fillOpacity = v))}
        >
          <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 7l4-4 4 4-4 4z" fill="currentColor" fillOpacity=".5" stroke="currentColor" /></svg>
        </ColorButton>
      ) : null}
      {caps.text ? (
        <>
          <ColorButton title="文字顏色" color={st.textColor} onChange={(c) => c && set((d) => void (d.style.textColor = c))}>
            <span className="text-xs font-semibold">T</span>
          </ColorButton>
          <button className={btn()} title="編輯文字" onClick={() => onOpenDialog("text", item)}>
            <Icon name="textEdit" />
          </button>
        </>
      ) : null}
      {caps.extend || caps.middleLine || caps.stats ? sep : null}
      {caps.extend ? (
        <>
          <button className={btn(st.extendLeft)} title="向左延伸" onClick={() => set((d) => void (d.style.extendLeft = !d.style.extendLeft))}>
            <Icon name="extendLeft" />
          </button>
          <button className={btn(st.extendRight)} title="向右延伸" onClick={() => set((d) => void (d.style.extendRight = !d.style.extendRight))}>
            <Icon name="extendRight" />
          </button>
        </>
      ) : null}
      {caps.middleLine ? (
        <button className={btn(st.middleLine)} title="中線" onClick={() => set((d) => void (d.style.middleLine = !d.style.middleLine))}>
          <Icon name="middle" />
        </button>
      ) : null}
      {caps.stats ? (
        <button className={btn(st.stats)} title="顯示統計（價差、%、K 棒數）" onClick={() => set((d) => void (d.style.stats = !d.style.stats))}>
          <Icon name="stats" />
        </button>
      ) : null}
      {caps.dialog ? (
        <button className={btn()} title="設定" data-testid="open-settings" onClick={() => onOpenDialog(caps.dialog!, item)}>
          <Icon name="settings" />
        </button>
      ) : null}
      {sep}
      <button className={btn(!!item.locked)} title={item.locked ? "解除鎖定" : "鎖定"} onClick={() => set((d) => void (d.locked = !d.locked))}>
        <Icon name={item.locked ? "lock" : "unlock"} />
      </button>
      <button className={btn()} title="複製" onClick={() => layer.clone(item.id)}>
        <Icon name="clone" />
      </button>
      <button className={btn()} title="移到最上層" onClick={() => layer.bringToFront(item.id)}>
        <Icon name="front" />
      </button>
      <button className={btn()} title="移到最下層" onClick={() => layer.sendToBack(item.id)}>
        <Icon name="back" />
      </button>
      <button className={btn()} title="隱藏" onClick={() => set((d) => void (d.hidden = true))}>
        <Icon name="eyeOff" />
      </button>
      <button className={`${btn()} hover:!text-[#ff5b6e]`} title="刪除 (Delete)" onClick={() => layer.remove(item.id)}>
        <Icon name="trash" />
      </button>
    </div>
  );
}

// ===========================================================================
// Dialog shell
// ===========================================================================

function Dialog({ title, onCancel, onOk, footerLeft, children, width = 560 }: { title: string; onCancel: () => void; onOk: () => void; footerLeft?: ReactNode; children: ReactNode; width?: number }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onPointerDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="flex max-h-[90vh] flex-col rounded-xl border border-white/10 bg-[#161b25] text-slate-200 shadow-2xl" style={{ width }} role="dialog" aria-label={title}>
        <div className="flex items-center border-b border-white/10 px-5 py-3">
          <span className="text-[15px] font-semibold">{title}</span>
          <button className="ml-auto text-xl leading-none text-slate-400 hover:text-white" onClick={onCancel}>
            ×
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
        <div className="flex items-center gap-2 border-t border-white/10 px-5 py-3">
          {footerLeft}
          <button className="ml-auto rounded-md border border-white/15 px-4 py-1.5 text-sm hover:bg-white/10" onClick={onCancel}>
            取消
          </button>
          <button className="rounded-md bg-[#2962ff] px-5 py-1.5 text-sm font-medium text-white hover:bg-[#1e53e5]" onClick={onOk} data-testid="dialog-ok">
            確定
          </button>
        </div>
      </div>
    </div>
  );
}

const Check = ({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children?: ReactNode }) => (
  <label className="flex cursor-pointer select-none items-center gap-2 text-sm">
    <input type="checkbox" className="h-4 w-4 accent-[#2962ff]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    {children}
  </label>
);

/** Edits a drawing with live preview; OK commits one undo step, Cancel restores. */
function useDraft<T>(layer: DrawingLayer, item: Drawing, read: (d: Drawing) => T, write: (d: Drawing, v: T) => void) {
  const original = useRef(structuredClone(read(item)));
  const [draft, setDraft] = useState<T>(() => structuredClone(read(item)));
  useEffect(() => {
    layer.preview(item.id, (d) => write(d, structuredClone(draft)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);
  const commit = () => {
    layer.preview(item.id, (d) => write(d, structuredClone(original.current)));
    layer.update(item.id, (d) => write(d, structuredClone(draft)));
  };
  const revert = () => layer.preview(item.id, (d) => write(d, structuredClone(original.current)));
  return { draft, setDraft, commit, revert };
}

// ===========================================================================
// Fibonacci settings (TradingView layout: two columns of level rows)
// ===========================================================================

type FibDraft = { fib: FibSettings; extendLeft: boolean; extendRight: boolean };

export function FibSettingsDialog({ layer, item, onClose }: { layer: DrawingLayer; item: Drawing; onClose: () => void }) {
  const { draft, setDraft, commit, revert } = useDraft<FibDraft>(
    layer,
    item,
    (d) => ({ fib: d.style.fib ?? getFibDefaults(), extendLeft: d.style.extendLeft, extendRight: d.style.extendRight }),
    (d, v) => {
      d.style.fib = v.fib;
      d.style.extendLeft = v.extendLeft;
      d.style.extendRight = v.extendRight;
    },
  );
  const [menu, setMenu] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const closeMenu = useMemo(() => () => setMenu(false), []);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2200);
    return () => clearTimeout(t);
  }, [toast]);
  const menuRef = useOutside(menu, closeMenu);
  const f = draft.fib;
  const patch = (p: Partial<FibSettings>) => setDraft((d) => ({ ...d, fib: { ...d.fib, ...p } }));
  const setLevel = (i: number, p: Partial<FibSettings["levels"][number]>) =>
    patch({ levels: f.levels.map((l, k) => (k === i ? { ...l, ...p } : l)) });
  const half = Math.ceil(f.levels.length / 2);
  const columns = [f.levels.slice(0, half), f.levels.slice(half)];

  const row = (i: number) => {
    const l = f.levels[i]!;
    return (
      <div key={i} className="flex items-center gap-2" data-fib-row={i}>
        <input type="checkbox" className="h-4 w-4 accent-[#2962ff]" checked={l.enabled} onChange={(e) => setLevel(i, { enabled: e.target.checked })} />
        <input
          type="number"
          step="0.001"
          className="h-7 w-[84px] rounded-md border border-white/15 bg-[#0e1219] px-2 text-sm tabular-nums outline-none focus:border-[#2962ff]"
          value={Number.isFinite(l.value) ? l.value : ""}
          onChange={(e) => setLevel(i, { value: e.target.value === "" ? NaN : Number(e.target.value) })}
        />
        <ColorButton color={l.color} onChange={(c) => c && setLevel(i, { color: c })} />
      </div>
    );
  };

  return (
    <Dialog
      title="斐波那契回撤 · 設定"
      width={600}
      onCancel={() => {
        revert();
        onClose();
      }}
      onOk={() => {
        commit();
        onClose();
      }}
      footerLeft={
        <div className="relative" ref={menuRef}>
          <button className="rounded-md border border-white/15 px-3 py-1.5 text-sm hover:bg-white/10" onClick={() => setMenu(!menu)}>
            範本 ▾
          </button>
          {menu ? (
            <div className="absolute bottom-10 left-0 z-10 w-48 rounded-lg border border-white/10 bg-[#1c2230] py-1 shadow-2xl">
              <button
                className="block w-full px-3 py-1.5 text-left text-sm hover:bg-white/10"
                data-testid="fib-save-default"
                onClick={() => {
                  saveFibTemplate(structuredClone(f));
                  setMenu(false);
                  setToast("已儲存為預設，新的回撤會使用這組設定");
                }}
              >
                儲存為預設
              </button>
              <button
                className="block w-full px-3 py-1.5 text-left text-sm hover:bg-white/10"
                onClick={() => {
                  patch(getFibDefaults());
                  setMenu(false);
                }}
              >
                套用預設
              </button>
              <button
                className="block w-full px-3 py-1.5 text-left text-sm text-slate-400 hover:bg-white/10"
                onClick={() => {
                  patch(defaultFibSettings());
                  saveFibTemplate(null);
                  setMenu(false);
                  setToast("已還原原廠設定");
                }}
              >
                還原原廠設定
              </button>
            </div>
          ) : null}
          {toast ? <span className="absolute bottom-10 left-0 w-max rounded bg-[#2962ff] px-2 py-1 text-xs text-white">{toast}</span> : null}
        </div>
      }
    >
      <div className="grid grid-cols-2 gap-x-8">
        {columns.map((col, c) => (
          <div key={c} className="flex flex-col gap-1">
            {col.map((_, k) => row(c * half + k))}
          </div>
        ))}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-x-8 gap-y-3 border-t border-white/10 pt-4">
        <div className="flex items-center gap-2">
          <Check checked={f.trendLine} onChange={(v) => patch({ trendLine: v })}>
            趨勢線
          </Check>
          <ColorButton color={f.trendLineColor} onChange={(c) => c && patch({ trendLineColor: c })} />
          <Select value={f.trendLineStyle} options={(["solid", "dashed", "dotted"] as LineStyle[]).map((s) => ({ value: s, label: lineSample(s) }))} onChange={(s) => patch({ trendLineStyle: s })} />
        </div>
        <div className="flex items-center gap-2">
          <Check checked={f.useOneColor} onChange={(v) => patch({ useOneColor: v })}>
            使用單一顏色
          </Check>
          <ColorButton color={f.oneColor} onChange={(c) => c && patch({ oneColor: c })} />
        </div>
        <div className="flex items-center gap-2">
          <Check checked={f.background} onChange={(v) => patch({ background: v })}>
            背景
          </Check>
          <input type="range" min={0} max={100} value={Math.round(f.backgroundOpacity * 100)} onChange={(e) => patch({ backgroundOpacity: Number(e.target.value) / 100 })} className="w-24" />
          <span className="text-xs tabular-nums text-slate-400">{Math.round(f.backgroundOpacity * 100)}%</span>
        </div>
        <Check checked={f.logLevels} onChange={(v) => patch({ logLevels: v })}>
          依對數座標計算水平
        </Check>
        <Check checked={draft.extendLeft} onChange={(v) => setDraft((d) => ({ ...d, extendLeft: v }))}>
          向左延伸
        </Check>
        <Check checked={draft.extendRight} onChange={(v) => setDraft((d) => ({ ...d, extendRight: v }))}>
          向右延伸
        </Check>
        <Check checked={f.reverse} onChange={(v) => patch({ reverse: v })}>
          反轉
        </Check>
        <Check checked={f.prices} onChange={(v) => patch({ prices: v })}>
          價格
        </Check>
        <div className="flex items-center gap-2">
          <Check checked={f.levelLabels} onChange={(v) => patch({ levelLabels: v })}>
            水平標籤
          </Check>
          <Select value={f.labelFormat} options={[{ value: "values", label: "數值" }, { value: "percent", label: "百分比" }]} onChange={(v) => patch({ labelFormat: v })} />
        </div>
        <div className="flex items-center gap-2 text-sm">
          標籤位置
          <Select value={f.labelsH} options={[{ value: "left", label: "左" }, { value: "center", label: "中" }, { value: "right", label: "右" }]} onChange={(v) => patch({ labelsH: v })} />
          <Select value={f.labelsV} options={[{ value: "top", label: "上" }, { value: "middle", label: "中" }, { value: "bottom", label: "下" }]} onChange={(v) => patch({ labelsV: v })} />
        </div>
      </div>
    </Dialog>
  );
}

// ===========================================================================
// Volume profile & text
// ===========================================================================

type VolumeDraft = { rows: number; valueArea: number; showPoc: boolean; showValueArea: boolean };

export function VolumeSettingsDialog({ layer, item, onClose }: { layer: DrawingLayer; item: Drawing; onClose: () => void }) {
  const { draft, setDraft, commit, revert } = useDraft<VolumeDraft>(
    layer,
    item,
    (d) => ({ rows: d.style.rows, valueArea: d.style.valueArea, showPoc: d.style.showPoc, showValueArea: d.style.showValueArea }),
    (d, v) => Object.assign(d.style, v),
  );
  const num = (key: "rows" | "valueArea", min: number, max: number) => (
    <input
      type="number"
      min={min}
      max={max}
      className="w-24 rounded-md border border-white/15 bg-[#0e1219] px-2 py-1 text-sm tabular-nums outline-none focus:border-[#2962ff]"
      value={draft[key]}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v)) setDraft((d) => ({ ...d, [key]: Math.max(min, Math.min(max, v)) }));
      }}
    />
  );
  return (
    <Dialog
      title="固定範圍成交量分布 · 設定"
      width={380}
      onCancel={() => {
        revert();
        onClose();
      }}
      onOk={() => {
        commit();
        onClose();
      }}
    >
      <div className="grid grid-cols-[1fr_auto] items-center gap-3 text-sm">
        <span>列數</span>
        {num("rows", 10, 2000)}
        <span>價值區 %</span>
        {num("valueArea", 1, 100)}
      </div>
      <div className="mt-4 flex flex-col gap-2.5">
        <Check checked={draft.showPoc} onChange={(v) => setDraft((d) => ({ ...d, showPoc: v }))}>
          顯示 POC
        </Check>
        <Check checked={draft.showValueArea} onChange={(v) => setDraft((d) => ({ ...d, showValueArea: v }))}>
          顯示價值區（VAH / VAL）
        </Check>
      </div>
    </Dialog>
  );
}

type TextDraft = { text: string; fontSize: number };

export function TextDialog({ layer, item, onClose }: { layer: DrawingLayer; item: Drawing; onClose: () => void }) {
  const { draft, setDraft, commit, revert } = useDraft<TextDraft>(
    layer,
    item,
    (d) => ({ text: d.text ?? "", fontSize: d.style.fontSize }),
    (d, v) => {
      d.text = v.text;
      d.style.fontSize = v.fontSize;
    },
  );
  return (
    <Dialog
      title="編輯文字"
      width={420}
      onCancel={() => {
        revert();
        onClose();
      }}
      onOk={() => {
        commit();
        onClose();
      }}
    >
      <textarea
        autoFocus
        className="h-28 w-full resize-none rounded-md border border-white/15 bg-[#0e1219] p-2 text-sm outline-none focus:border-[#2962ff]"
        value={draft.text}
        placeholder="輸入文字…"
        onChange={(e) => setDraft((d) => ({ ...d, text: e.target.value }))}
      />
      <div className="mt-3 flex items-center gap-2 text-sm">
        字級
        <Select value={draft.fontSize} options={[10, 11, 12, 13, 14, 16, 20, 24, 28, 32].map((v) => ({ value: v, label: `${v}` }))} onChange={(v) => setDraft((d) => ({ ...d, fontSize: v }))} />
      </div>
    </Dialog>
  );
}
