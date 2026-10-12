// 圖層 panel (TradingView's object tree): every drawing, top-most first, with
// folders. Click selects, double-click renames, drag reorders or moves a
// drawing into / out of a folder; rows and folders can be shown / hidden,
// locked and deleted. Lives in the right sidebar (SideBar.tsx).

import { useState, type DragEvent, type ReactNode } from "react";
import type { DrawingLayer } from "../drawings/layer";
import { getTool } from "../drawings/tools";
import type { Drawing, DrawingGroup } from "../drawings/types";
import { useLayer } from "./drawing-ui";
import { Icon } from "./icons";
import { PanelHeader, panelIconBtn } from "./SideBar";

const iconBtn = panelIconBtn;

export function layerName(layer: DrawingLayer, item: Drawing): string {
  if (item.name) return item.name;
  const label = getTool(item.type)?.label ?? item.type;
  // number drawings of the same kind in creation order: 趨勢線 1, 趨勢線 2…
  const same = layer.items.filter((d) => d.type === item.type);
  return same.length > 1 ? `${label} ${same.indexOf(item) + 1}` : label;
}

type Entry = { kind: "item"; item: Drawing } | { kind: "group"; group: DrawingGroup; members: Drawing[] };

/** Top-level rows, top-most first; a folder sits where its top-most drawing is. */
function entries(layer: DrawingLayer): Entry[] {
  const out: Entry[] = [];
  const seen = new Set<string>();
  // empty folders first, so a new folder shows up at the top
  for (const g of layer.groups) if (!layer.items.some((d) => d.groupId === g.id)) out.push({ kind: "group", group: g, members: [] });
  for (let i = layer.items.length - 1; i >= 0; i--) {
    const item = layer.items[i]!;
    const group = item.groupId ? layer.groupOf(item.groupId) : undefined;
    if (!group) {
      out.push({ kind: "item", item });
      continue;
    }
    if (seen.has(group.id)) continue;
    seen.add(group.id);
    out.push({ kind: "group", group, members: [...layer.membersOf(group.id)].reverse() });
  }
  return out;
}

type Drop = { kind: "item"; id: string; above: boolean } | { kind: "group"; id: string };

export function LayersPanel({ layer, onClose }: { layer: DrawingLayer; onClose: () => void }) {
  useLayer(layer);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [dragId, setDragId] = useState<string | null>(null);
  const [drop, setDrop] = useState<Drop | null>(null);
  const list = entries(layer);
  const allHidden = layer.items.length > 0 && layer.items.every((d) => d.hidden);

  const startEdit = (id: string, name: string) => {
    setEditing(id);
    setDraft(name);
  };
  const commitName = (id: string, isGroup: boolean) => {
    const name = draft.trim();
    if (isGroup) layer.renameGroup(id, name);
    else
      layer.update(id, (d) => {
        d.name = name || undefined;
      });
    setEditing(null);
  };

  const reset = () => {
    setDragId(null);
    setDrop(null);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    if (!dragId || !drop) return reset();
    const from = layer.items.findIndex((d) => d.id === dragId);
    if (drop.kind === "group") {
      // into a folder: just above its current top-most drawing
      const members = layer.membersOf(drop.id);
      const top = members.length ? layer.items.indexOf(members[members.length - 1]!) : from;
      layer.place(dragId, from <= top ? top : top + 1, drop.id);
    } else if (drop.id !== dragId) {
      const target = layer.items.find((d) => d.id === drop.id)!;
      let index = layer.items.indexOf(target) + (drop.above ? 1 : 0);
      if (from < index) index -= 1;
      layer.place(dragId, index, target.groupId ?? null);
    }
    reset();
  };

  const nameField = (id: string, isGroup: boolean, label: string, muted: boolean) =>
    editing === id ? (
      <input
        autoFocus
        onFocus={(e) => e.target.select()}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        onBlur={() => commitName(id, isGroup)}
        onKeyDown={(e) => {
          if (e.key === "Enter") commitName(id, isGroup);
          if (e.key === "Escape") setEditing(null);
        }}
        className="h-6 min-w-0 flex-1 rounded border border-[var(--c-accent)] bg-[var(--c-raised)] px-1.5 text-xs outline-none"
      />
    ) : (
      <span className={`min-w-0 flex-1 truncate text-xs ${isGroup ? "font-semibold" : ""} ${muted ? "text-[var(--c-muted)] line-through" : ""}`} title="雙擊重新命名">
        {label}
      </span>
    );

  const itemRow = (item: Drawing, nested: boolean): ReactNode => {
    const selected = item.id === layer.selectedId;
    const marker = drop?.kind === "item" && drop.id === item.id && dragId && dragId !== item.id;
    return (
      <li
        key={item.id}
        draggable={editing !== item.id}
        onDragStart={(e) => {
          e.stopPropagation();
          setDragId(item.id);
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", item.id);
        }}
        onDragEnd={reset}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          const r = e.currentTarget.getBoundingClientRect();
          setDrop({ kind: "item", id: item.id, above: e.clientY < r.top + r.height / 2 });
        }}
        onClick={() => layer.select(selected ? null : item.id)}
        onDoubleClick={() => startEdit(item.id, item.name ?? layerName(layer, item))}
        data-layer-row={item.id}
        className={`group relative my-0.5 flex cursor-pointer items-center gap-1.5 rounded-md py-1 pr-1.5 ring-1 ring-inset ${nested ? "ml-5 mr-1.5 pl-1.5" : "mx-1.5 pl-1.5"} ${
          selected ? "bg-[var(--c-accent-soft)] ring-[var(--c-accent)]" : "ring-transparent hover:bg-[var(--c-hover)] hover:ring-[var(--c-line)]"
        } ${dragId === item.id ? "opacity-40" : ""}`}
      >
        {marker ? <span className={`pointer-events-none absolute inset-x-1 h-0.5 rounded bg-[var(--c-accent)] ${drop.above ? "-top-0.5" : "-bottom-0.5"}`} /> : null}
        <span className="cursor-grab text-[var(--c-muted)] active:cursor-grabbing [&_svg]:h-3.5 [&_svg]:w-3.5" title="拖曳調整順序，拖到群組上可放進群組">
          <Icon name="grip" />
        </span>
        <span className={`[&_svg]:h-4 [&_svg]:w-4 ${item.hidden ? "opacity-40" : ""}`}>
          <Icon name={item.type} />
        </span>
        {nameField(item.id, false, layerName(layer, item), !!item.hidden)}
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
  };

  const groupBlock = (group: DrawingGroup, members: Drawing[]): ReactNode => {
    const hidden = members.length > 0 && members.every((d) => d.hidden);
    const locked = members.length > 0 && members.every((d) => d.locked);
    const target = drop?.kind === "group" && drop.id === group.id && dragId;
    return (
      <li key={group.id} data-layer-group={group.id}>
        <div
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDrop({ kind: "group", id: group.id });
          }}
          onClick={() => layer.setGroupCollapsed(group.id, !group.collapsed)}
          onDoubleClick={(e) => {
            e.stopPropagation();
            startEdit(group.id, group.name);
          }}
          className={`mx-1.5 my-0.5 flex cursor-pointer items-center gap-1 rounded-md py-1 pl-1 pr-1.5 ring-1 ring-inset ${
            target ? "bg-[var(--c-accent-soft)] ring-[var(--c-accent)]" : "ring-transparent hover:bg-[var(--c-hover)] hover:ring-[var(--c-line)]"
          }`}
          title={group.collapsed ? "展開群組" : "收合群組"}
        >
          <span className={`text-[var(--c-muted)] transition-transform [&_svg]:h-3.5 [&_svg]:w-3.5 ${group.collapsed ? "" : "rotate-90"}`}>
            <Icon name="chevron" />
          </span>
          <span className="text-[var(--c-accent)] [&_svg]:h-4 [&_svg]:w-4">
            <Icon name="folder" />
          </span>
          {nameField(group.id, true, group.name, hidden)}
          <span className="px-1 text-[11px] tabular-nums text-[var(--c-muted)]">{members.length}</span>
          <button
            className={iconBtn(hidden)}
            disabled={!members.length}
            title={hidden ? "顯示整個群組" : "隱藏整個群組"}
            onClick={(e) => {
              e.stopPropagation();
              layer.setGroupHidden(group.id, !hidden);
            }}
          >
            <Icon name={hidden ? "eyeOff" : "eye"} />
          </button>
          <button
            className={iconBtn(locked)}
            disabled={!members.length}
            title={locked ? "解除鎖定整個群組" : "鎖定整個群組"}
            onClick={(e) => {
              e.stopPropagation();
              layer.setGroupLocked(group.id, !locked);
            }}
          >
            <Icon name={locked ? "lock" : "unlock"} />
          </button>
          <button
            className={iconBtn()}
            title="解散群組（保留裡面的繪圖）"
            onClick={(e) => {
              e.stopPropagation();
              layer.removeGroup(group.id, true);
            }}
          >
            <Icon name="ungroup" />
          </button>
          <button
            className={`${iconBtn()} hover:!text-[var(--c-danger)]`}
            title="刪除群組與裡面的繪圖"
            onClick={(e) => {
              e.stopPropagation();
              if (!members.length || window.confirm(`刪除「${group.name}」和裡面 ${members.length} 個繪圖？（可用 Ctrl+Z 復原）`)) layer.removeGroup(group.id);
            }}
          >
            <Icon name="trash" />
          </button>
        </div>
        {!group.collapsed ? (
          members.length ? (
            <ul>{members.map((m) => itemRow(m, true))}</ul>
          ) : (
            <p className="ml-9 py-1 text-[11px] text-[var(--c-muted)]">空群組，把繪圖拖到這裡</p>
          )
        ) : null}
      </li>
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="layers-panel" onKeyDown={(e) => e.stopPropagation()}>
      <PanelHeader title="圖層" count={layer.items.length} onClose={onClose}>
        <button
          className={iconBtn()}
          title={layer.selectedId ? "新增群組（把選取的繪圖放進去）" : "新增群組"}
          data-testid="add-group"
          onClick={() => layer.addGroup(undefined, layer.selectedId ? [layer.selectedId] : [])}
        >
          <Icon name="folderPlus" />
        </button>
        <button className={iconBtn()} title={allHidden ? "全部顯示" : "全部隱藏"} disabled={!layer.items.length} onClick={() => layer.setAllHidden(!allHidden)}>
          <Icon name={allHidden ? "eyeOff" : "eye"} />
        </button>
      </PanelHeader>

      {list.length === 0 ? (
        <p className="px-4 py-6 text-center text-xs leading-relaxed text-[var(--c-muted)]">
          還沒有繪圖。
          <br />
          從左側工具列選一個工具開始畫，畫好的東西會列在這裡。
        </p>
      ) : (
        <ul className="min-h-0 flex-1 overflow-y-auto py-1" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
          {list.map((e) => (e.kind === "item" ? itemRow(e.item, false) : groupBlock(e.group, e.members)))}
        </ul>
      )}
    </div>
  );
}
