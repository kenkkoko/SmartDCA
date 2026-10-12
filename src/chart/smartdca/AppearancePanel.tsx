// 圖表設定 side panel: background, grid and candle colours with live preview.

import type { ReactNode } from "react";
import { ColorButton } from "../react/drawing-ui";
import { PanelHeader } from "../react/SideBar";
import type { Theme } from "../engine/types";
import type { Appearance } from "./appearance";

const PRESETS: { name: string; a: Appearance }[] = [
  { name: "預設", a: {} },
  { name: "TradingView", a: { up: "#089981", down: "#f23645" } },
  { name: "純黑", a: { background: "#000000", gridHidden: true } },
  { name: "深藍", a: { background: "#0f1729", grid: "rgba(148,163,184,0.08)" } },
  { name: "灰影線", a: { up: "#26a69a", down: "#ef5350", wickUp: "#999999", wickDown: "#999999" } },
];

export function AppearancePanel({
  value,
  onChange,
  base,
  onClose,
}: {
  value: Appearance;
  onChange: (a: Appearance) => void;
  /** Theme without the user's colours (shows the defaults). */
  base: Theme;
  onClose: () => void;
}) {
  const set = (patch: Partial<Appearance>) => onChange({ ...value, ...patch });
  const reset = (key: keyof Appearance) => (
    <button
      className={`h-6 cursor-pointer rounded px-1.5 text-[11px] ${value[key] !== undefined ? "text-[var(--c-accent)] hover:underline" : "invisible"}`}
      onClick={() => {
        const next = { ...value };
        delete next[key];
        onChange(next);
      }}
      title="恢復預設"
    >
      預設
    </button>
  );
  const row = (label: string, control: ReactNode, key?: keyof Appearance) => (
    <div className="flex h-9 items-center gap-2">
      <span className="min-w-0 flex-1 text-[12.5px] text-[var(--c-text)]">{label}</span>
      {key ? reset(key) : null}
      {control}
    </div>
  );
  const section = (title: string) => <p className="mb-1 mt-4 text-[11px] font-semibold text-[var(--c-muted)] first:mt-0">{title}</p>;

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="appearance-panel">
      <PanelHeader title="圖表設定" onClose={onClose} />
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-3">
        {section("快速套用")}
        <div className="flex flex-wrap gap-1">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              className="h-7 cursor-pointer rounded-md px-2.5 text-[12px] text-[var(--c-text)] ring-1 ring-inset ring-[var(--c-line)] hover:bg-[var(--c-hover)]"
              onClick={() => onChange({ ...p.a })}
            >
              {p.name}
            </button>
          ))}
        </div>

        {section("畫布")}
        {row("背景", <ColorButton color={value.background ?? base.background} onChange={(c) => c && set({ background: c })} title="背景顏色" />, "background")}
        {row(
          "格線",
          <div className="flex items-center gap-1">
            <label className="flex cursor-pointer items-center gap-1 text-[11.5px] text-[var(--c-muted)]">
              <input type="checkbox" className="accent-[var(--c-accent)]" checked={!value.gridHidden} onChange={(e) => set({ gridHidden: !e.target.checked })} data-testid="appearance-grid" />
              顯示
            </label>
            <ColorButton color={value.grid ?? base.grid} onChange={(c) => c && set({ grid: c })} title="格線顏色" />
          </div>,
          "grid",
        )}

        {row("座標軸文字", <ColorButton color={value.axisText ?? base.text} onChange={(c) => c && set({ axisText: c })} title="價格軸 / 時間軸的文字顏色" />, "axisText")}

        {section("K 線")}
        {row("上漲實體", <ColorButton color={value.up ?? base.up} onChange={(c) => c && set({ up: c })} title="上漲 K 線" />, "up")}
        {row("下跌實體", <ColorButton color={value.down ?? base.down} onChange={(c) => c && set({ down: c })} title="下跌 K 線" />, "down")}
        {row("上漲影線", <ColorButton color={value.wickUp ?? value.up ?? base.up} onChange={(c) => c && set({ wickUp: c })} title="上漲影線（預設跟實體同色）" />, "wickUp")}
        {row("下跌影線", <ColorButton color={value.wickDown ?? value.down ?? base.down} onChange={(c) => c && set({ wickDown: c })} title="下跌影線（預設跟實體同色）" />, "wickDown")}
        {row(
          "現價線",
          <label className="flex cursor-pointer items-center gap-1 text-[11.5px] text-[var(--c-muted)]">
            <input type="checkbox" className="accent-[var(--c-accent)]" checked={!value.priceLineHidden} onChange={(e) => set({ priceLineHidden: !e.target.checked })} data-testid="appearance-priceline" />
            顯示
          </label>,
        )}
        {section("價格標籤")}
        {row("標籤底色", <ColorButton color={value.labelBg ?? base.up} onChange={(c) => c && set({ labelBg: c })} title="現價標籤底色（預設跟最後一根 K 線同色）" />, "labelBg")}
        {row("標籤文字", <ColorButton color={value.labelText ?? "#ffffff"} onChange={(c) => c && set({ labelText: c })} title="現價標籤與倒數的文字顏色" />, "labelText")}
        <p className="mt-1 text-[11px] leading-relaxed text-[var(--c-muted)]">成交量柱也會跟著漲跌顏色。深色和淺色主題各自記住一組設定。</p>

        <button
          className="mt-4 h-8 w-full cursor-pointer rounded-md text-[12px] text-[var(--c-text)] ring-1 ring-inset ring-[var(--c-line)] hover:bg-[var(--c-hover)]"
          onClick={() => onChange({})}
          data-testid="appearance-reset"
        >
          全部恢復預設
        </button>
      </div>
    </div>
  );
}
