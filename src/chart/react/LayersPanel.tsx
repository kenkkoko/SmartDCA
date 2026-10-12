// 圖層 panel (TradingView's object tree): every drawing, top-most first.
// Click selects, double-click renames, drag reorders; each row can be
// shown / hidden, locked and deleted.

import { useState, type DragEvent } from "react";
import type { DrawingLayer } from "../drawings/layer";
import { getTool } from "../drawings/tools";
import type { Drawing } from "../drawings/types";
import { useLayer } from "./drawing-ui";
import { Icon } from "./icons";

const iconBtn = (on = false) =>
  `flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-md ring-1 ring-inset transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--c-accent)] [&_svg]:h-4 [&_svg]:w-4 ${
    on ? "bg-[var(--c-accent-soft)] text-[var(--c-accent)] ring-[var(--c-accent)]" : "text-[var(--c-muted)] ring-transparent hover:bg-[var(--c-hover)] hover:text-[var(--c-text)] hover:ring-[var(--c-line)]"
  }`;

export function layerName(layer: DrawingLayer, item: Drawing): string {
  if (item.name) return item.name;
  const label = getTool(item.type)?.label ?? item.type;
  // number drawings of the same kind in creation order: 趨勢線 1, 趨勢線 2…
  const same = layer.items.filter((d) => d.type === item.type);
  return same.length > 1 ? `${label} ${same.indexOf(item) + 1}` : label;
}

export function LayersPanel({ layer, onClose }: { layer: DrawingLayer; onClose: () => void }) {
  useLayer(layer);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [dragId, setDragId] = useState<string | null>(null);
  const [dropAt, setDropAt] = useState<{ id: string; above: boolean } | null>(null);
  const rows = [...layer.items].reverse(); // top-most first
  const allHidden = rows.length > 0 && rows.every((d) => d.hidden);

  const commitName = (id: string) => {
    const name = draft.trim();
    layer.update(id, (d) => {
      d.name = name || undefined;
    });
    setEditing(null);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    if (!dragId || !dropAt || dragId === dropAt.id) return reset();
    const target = layer.items.findIndex((d) => d.id === dropAt.id);
    const from = layer.items.findIndex((d) => d.id === dragId);
    // rows are reversed: "above" in the list = later in paint order
    let index = dropAt.above ? target + 1 : target;
    if (from < index) index -= 1;
    layer.moveTo(dragId, index);
    reset();
  };
  const reset = () => {
    setDragId(null);
    setDropAt(null);
  };

  return (
    <aside
      className="flex h-full w-[248px] shrink-0 flex-col border-l border-[var(--c-line)] bg-[var(--c-panel)] text-[var(--c-text)]"
      data-testid="layers-panel"
      onKeyDown={(e) => e.stopPropagation()}
    >
      <div className="flex items-center gap-1 border-b border-[var(--c-line)] px-3 py-2">
        <Icon name="layers" />
        <span className="ml-1 text-sm font-semibold">圖層</span>
        <span className="text-xs text-[var(--c-muted)]">{rows.length}</span>
        <button className={`${iconBtn()} ml-auto`} title={allHidden ? "全部顯示" : "全部隱藏"} disabled={!rows.length} onClick={() => layer.setAllHidden(!allHidden)}>
          <Icon name={allHidden ? "eyeOff" : "eye"} />
        </button>
        <button className={iconBtn()} title="關閉圖層面板" onClick={onClose}>
          <span className="text-base leading-none">×</span>
        </button>
      </div>

      {rows.length === 0 ? (
        <p className="px-4 py-6 text-center text-xs leading-relaxed text-[var(--c-muted)]">
          還沒有繪圖。
          <br />
          從左側工具列選一個工具開始畫，畫好的東西會列在這裡。
        </p>
      ) : (
        <ul className="min-h-0 flex-1 overflow-y-auto py-1" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
          {rows.map((item) => {
            const selected = item.id === layer.selectedId;
            const marker = dropAt?.id === item.id && dragId && dragId !== item.id;
            return (
              <li
                key={item.id}
                draggable={editing !== item.id}
                onDragStart={(e) => {
                  setDragId(item.id);
                  e.dataTransfer.effectAllowed = "move";
                  e.dataTransfer.setData("text/plain", item.id);
                }}
                onDragEnd={reset}
                onDragOver={(e) => {
                  e.preventDefault();
                  const r = e.currentTarget.getBoundingClientRect();
                  setDropAt({ id: item.id, above: e.clientY < r.top + r.height / 2 });
                }}
                onClick={() => layer.select(selected ? null : item.id)}
                onDoubleClick={() => {
                  setEditing(item.id);
                  setDraft(item.name ?? layerName(layer, item));
                }}
                data-layer-row={item.id}
                className={`group relative mx-1.5 my-0.5 flex cursor-pointer items-center gap-1.5 rounded-md px-1.5 py-1 ring-1 ring-inset ${
                  selected ? "bg-[var(--c-accent-soft)] ring-[var(--c-accent)]" : "ring-transparent hover:bg-[var(--c-hover)] hover:ring-[var(--c-line)]"
                } ${dragId === item.id ? "opacity-40" : ""}`}
              >
                {marker ? <span className={`pointer-events-none absolute inset-x-1 h-0.5 rounded bg-[var(--c-accent)] ${dropAt!.above ? "-top-0.5" : "-bottom-0.5"}`} /> : null}
                <span className="cursor-grab text-[var(--c-muted)] active:cursor-grabbing [&_svg]:h-3.5 [&_svg]:w-3.5" title="拖曳調整上下順序">
                  <Icon name="grip" />
                </span>
                <span className={`[&_svg]:h-4 [&_svg]:w-4 ${item.hidden ? "opacity-40" : ""}`}>
                  <Icon name={item.type} />
                </span>
                {editing === item.id ? (
                  <input
                    autoFocus
                    onFocus={(e) => e.target.select()}
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    onBlur={() => commitName(item.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitName(item.id);
                      if (e.key === "Escape") setEditing(null);
                    }}
                    className="h-6 min-w-0 flex-1 rounded border border-[var(--c-accent)] bg-[var(--c-raised)] px-1.5 text-xs outline-none"
                  />
                ) : (
                  <span className={`min-w-0 flex-1 truncate text-xs ${item.hidden ? "text-[var(--c-muted)] line-through" : ""}`} title="雙擊重新命名">
                    {layerName(layer, item)}
                  </span>
                )}
                <button
                  className={iconBtn(!!item.hidden)}
                  title={item.hidden ? "顯示" : "隱藏"}
                  onClick={(e) => {
                    e.stopPropagation();
                    layer.update(item.id, (d) => void (d.hidden = !d.hidden));
                  }}
                >
                  <Icon name={item.hidden ? "eyeOff" : "eye"} />
                </button>
                <button
                  className={iconBtn(!!item.locked)}
                  title={item.locked ? "解除鎖定" : "鎖定（不能拖動）"}
                  onClick={(e) => {
                    e.stopPropagation();
                    layer.update(item.id, (d) => void (d.locked = !d.locked));
                  }}
                >
                  <Icon name={item.locked ? "lock" : "unlock"} />
                </button>
                <button
                  className={`${iconBtn()} hover:!text-[var(--c-danger)]`}
                  title="刪除"
                  onClick={(e) => {
                    e.stopPropagation();
                    layer.remove(item.id);
                  }}
                >
                  <Icon name="trash" />
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {rows.length > 1 ? <p className="border-t border-[var(--c-line)] px-3 py-2 text-[11px] text-[var(--c-muted)]">最上面的圖層畫在最上方。拖曳列可調整順序，雙擊可改名。</p> : null}
    </aside>
  );
}
