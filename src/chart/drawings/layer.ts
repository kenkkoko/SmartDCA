// The drawing layer: owns the drawings, paints them (as a chart primitive)
// and handles creating / selecting / editing them (as the chart's
// interaction handler).
//
// TradingView rules implemented here:
//  - while a tool is armed, a click always places a point — even on top of
//    another drawing (hold Alt to select what is underneath instead);
//  - two-point tools can be drawn with click-click or press-drag-release;
//  - Shift keeps lines horizontal / vertical / 45°;
//  - after finishing, the tool disarms unless "keep drawing" is on.

import type { Chart, ChartPrimitive, InteractionHandler, Pane, PointerInfo, PrimitiveContext } from "../engine/chart";
import { formatTimeLabel } from "../engine/time-scale";
import type { Surface } from "../engine/surface";
import type { Point } from "../engine/types";
import { dist, readable, snapAngle, textOn } from "./geometry";
import type { Tool, ToolContext } from "./tool";
import { roundRect } from "./tool";
import { getTool } from "./tools";
import { AUTO_COLOR, DEFAULT_STYLE, newId, type Anchor, type Drawing, type DrawingGroup, type DrawingStyle } from "./types";

type Creating = {
  tool: Tool;
  item: Drawing;
  /** Points fixed so far (the rest repeat the preview). */
  placed: number;
  downAt: Point | null;
  dragged: boolean;
};

type Drag =
  | { kind: "handle"; id: string; index: number; startAnchors: Anchor[]; before: string; moved: boolean }
  | { kind: "body"; id: string; start: Point; startPoints: Point[]; startAnchors: Anchor[]; before: string; moved: boolean }
  | { kind: "stroke"; last: Point };

const HANDLE_RADIUS = 5;

export type LayerEvent = { type: "created"; item: Drawing } | { type: "editText"; item: Drawing };

export class DrawingLayer implements ChartPrimitive, InteractionHandler {
  readonly zIndex = 100;
  readonly chart: Chart;
  items: Drawing[] = [];
  /** Layers-panel folders; drawings point at them with `groupId`. */
  groups: DrawingGroup[] = [];
  selectedId: string | null = null;
  hoveredId: string | null = null;
  activeTool: string | null = null;
  /** Keep the tool armed after finishing a drawing. */
  keepDrawing = false;
  /** Snap prices to the bar's open / high / low / close. */
  magnet = false;
  /** Style overrides remembered per tool (last used colours etc.). */
  toolStyles: Record<string, Partial<DrawingStyle>> = {};

  private creating: Creating | null = null;
  private drag: Drag | null = null;
  private past: string[] = [];
  private future: string[] = [];
  private lastCoalesce: { id: string; at: number } | null = null;
  private listeners = new Set<() => void>();
  private eventListeners = new Set<(e: LayerEvent) => void>();
  private changeListeners = new Set<(items: Drawing[]) => void>();
  private detach: (() => void)[] = [];
  /** Bumps on every change; React reads it via useSyncExternalStore. */
  version = 0;

  constructor(chart: Chart) {
    this.chart = chart;
    this.detach.push(chart.addPrimitive(this));
    chart.setInteractionHandler(this);
    const keydown = (e: KeyboardEvent) => this.onKey(e);
    chart.root.addEventListener("keydown", keydown);
    this.detach.push(() => chart.root.removeEventListener("keydown", keydown));
  }

  dispose(): void {
    for (const d of this.detach) d();
    this.chart.setInteractionHandler(null);
  }

  // ======================================================================
  // State API (used by the React toolbar / properties bar)
  // ======================================================================

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Persist whenever the drawings themselves change. */
  onItemsChange(listener: (items: Drawing[]) => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }

  onEvent(listener: (e: LayerEvent) => void): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  get selected(): Drawing | undefined {
    return this.items.find((d) => d.id === this.selectedId);
  }

  setItems(items: Drawing[], groups: DrawingGroup[] = []): void {
    this.groups = groups.map((g) => ({ ...g }));
    const known = new Set(this.groups.map((g) => g.id));
    this.items = items.map((d) => ({ ...d, groupId: d.groupId && known.has(d.groupId) ? d.groupId : undefined, style: { ...DEFAULT_STYLE, ...d.style } }));
    this.past = [];
    this.future = [];
    this.selectedId = null;
    this.notify(false);
  }

  /**
   * One-shot range pick (AI analysis): the next click-click or drag on the
   * chart reports its two times instead of creating a drawing.
   */
  captureRange(onPick: (from: number, to: number) => void): void {
    this.setActiveTool("date_range");
    this.capture = onPick;
    this.notify(false);
  }

  get capturing(): boolean {
    return this.capture !== null;
  }

  private capture: ((from: number, to: number) => void) | null = null;

  setActiveTool(type: string | null): void {
    this.cancelCreating();
    this.capture = null;
    this.activeTool = type;
    if (type) this.selectedId = null;
    this.notify(false);
  }

  setKeepDrawing(on: boolean): void {
    this.keepDrawing = on;
    this.notify(false);
  }

  setMagnet(on: boolean): void {
    this.magnet = on;
    this.notify(false);
  }

  /** Hide / show every drawing at once (one undo step). */
  setAllHidden(hidden: boolean): void {
    if (!this.items.length) return;
    this.checkpoint();
    for (const d of this.items) d.hidden = hidden;
    if (hidden) this.selectedId = null;
    this.notify(true);
  }

  select(id: string | null): void {
    this.selectedId = id;
    this.notify(false);
  }

  /** Edit a drawing; rapid edits of the same drawing (sliders) share one undo step. */
  update(id: string, change: (d: Drawing) => void): void {
    const item = this.items.find((d) => d.id === id);
    if (!item) return;
    const now = Date.now();
    if (!this.lastCoalesce || this.lastCoalesce.id !== id || now - this.lastCoalesce.at > 600) this.checkpoint();
    this.lastCoalesce = { id, at: now };
    change(item);
    this.toolStyles[item.type] = pickStyle(item.style);
    this.notify(true);
  }

  /** Live preview while a dialog is open: no undo step, not persisted. */
  preview(id: string, change: (d: Drawing) => void): void {
    const item = this.items.find((d) => d.id === id);
    if (!item) return;
    change(item);
    this.notify(false);
  }

  remove(id: string): void {
    if (!this.items.some((d) => d.id === id)) return;
    this.checkpoint();
    this.items = this.items.filter((d) => d.id !== id);
    if (this.selectedId === id) this.selectedId = null;
    this.notify(true);
  }

  removeAll(): void {
    if (!this.items.length && !this.groups.length) return;
    this.checkpoint();
    this.items = [];
    this.groups = [];
    this.selectedId = null;
    this.notify(true);
  }

  clone(id: string): void {
    const item = this.items.find((d) => d.id === id);
    if (!item) return;
    this.checkpoint();
    const copy: Drawing = structuredClone(item);
    copy.id = newId();
    // nudge so the copy is visible next to the original
    const shift = (this.chart.time.indexToTime(1) - this.chart.time.indexToTime(0)) * 3;
    copy.anchors = copy.anchors.map((a) => ({ ...a, time: a.time + shift }));
    this.items.push(copy);
    this.selectedId = copy.id;
    this.notify(true);
  }

  /** Move a drawing to paint position `index` (0 = bottom), one undo step. */
  moveTo(id: string, index: number): void {
    const from = this.items.findIndex((d) => d.id === id);
    const to = Math.max(0, Math.min(this.items.length - 1, index));
    if (from < 0 || from === to) return;
    this.checkpoint();
    const [item] = this.items.splice(from, 1);
    this.items.splice(to, 0, item!);
    this.notify(true);
  }

  /** Drop from the layers panel: paint position and folder in one undo step. */
  place(id: string, index: number, groupId: string | null): void {
    const from = this.items.findIndex((d) => d.id === id);
    if (from < 0) return;
    const to = Math.max(0, Math.min(this.items.length - 1, index));
    const item = this.items[from]!;
    if (from === to && (item.groupId ?? null) === groupId) return;
    this.checkpoint();
    this.items.splice(from, 1);
    this.items.splice(to, 0, item);
    item.groupId = groupId ?? undefined;
    this.notify(true);
  }

  /** Paint order: later items paint on top. */
  bringToFront(id: string): void {
    const i = this.items.findIndex((d) => d.id === id);
    if (i < 0 || i === this.items.length - 1) return;
    this.checkpoint();
    this.items.push(...this.items.splice(i, 1));
    this.notify(true);
  }

  sendToBack(id: string): void {
    const i = this.items.findIndex((d) => d.id === id);
    if (i <= 0) return;
    this.checkpoint();
    this.items.unshift(...this.items.splice(i, 1));
    this.notify(true);
  }

  undo(): void {
    const prev = this.past.pop();
    if (prev === undefined) return;
    this.future.push(this.snapshot());
    this.restore(prev);
    if (!this.items.some((d) => d.id === this.selectedId)) this.selectedId = null;
    this.notify(true);
  }

  redo(): void {
    const next = this.future.pop();
    if (next === undefined) return;
    this.past.push(this.snapshot());
    this.restore(next);
    this.notify(true);
  }

  get canUndo(): boolean {
    return this.past.length > 0;
  }

  get canRedo(): boolean {
    return this.future.length > 0;
  }

  private snapshot(): string {
    return JSON.stringify({ items: this.items, groups: this.groups });
  }

  private restore(json: string): void {
    const state = JSON.parse(json) as { items: Drawing[]; groups: DrawingGroup[] };
    this.items = state.items;
    this.groups = state.groups;
  }

  // ---- groups (layers-panel folders) ----

  groupOf(id: string): DrawingGroup | undefined {
    return this.groups.find((g) => g.id === id);
  }

  membersOf(groupId: string): Drawing[] {
    return this.items.filter((d) => d.groupId === groupId);
  }

  /** New folder (optionally holding some drawings); returns its id. */
  addGroup(name?: string, memberIds: string[] = []): string {
    this.checkpoint();
    const id = newId();
    let n = this.groups.length + 1;
    while (this.groups.some((g) => g.name === `群組 ${n}`)) n++;
    this.groups.unshift({ id, name: name ?? `群組 ${n}` });
    for (const d of this.items) if (memberIds.includes(d.id)) d.groupId = id;
    this.notify(true);
    return id;
  }

  renameGroup(id: string, name: string): void {
    const g = this.groupOf(id);
    if (!g || !name.trim() || g.name === name.trim()) return;
    this.checkpoint();
    g.name = name.trim();
    this.notify(true);
  }

  /** Fold / unfold in the panel (not an undo step). */
  setGroupCollapsed(id: string, collapsed: boolean): void {
    const g = this.groupOf(id);
    if (!g) return;
    g.collapsed = collapsed;
    this.notify(true);
  }

  /** Move a drawing into a folder (null = out of any folder). */
  setGroup(itemId: string, groupId: string | null): void {
    const d = this.items.find((x) => x.id === itemId);
    if (!d || (d.groupId ?? null) === groupId) return;
    this.checkpoint();
    d.groupId = groupId ?? undefined;
    this.notify(true);
  }

  setGroupHidden(id: string, hidden: boolean): void {
    const members = this.membersOf(id);
    if (!members.length) return;
    this.checkpoint();
    for (const d of members) d.hidden = hidden;
    if (hidden && members.some((d) => d.id === this.selectedId)) this.selectedId = null;
    this.notify(true);
  }

  setGroupLocked(id: string, locked: boolean): void {
    const members = this.membersOf(id);
    if (!members.length) return;
    this.checkpoint();
    for (const d of members) d.locked = locked;
    this.notify(true);
  }

  /** Delete a folder; its drawings go with it unless `keepDrawings`. */
  removeGroup(id: string, keepDrawings = false): void {
    if (!this.groupOf(id)) return;
    this.checkpoint();
    this.groups = this.groups.filter((g) => g.id !== id);
    if (keepDrawings) {
      for (const d of this.items) if (d.groupId === id) d.groupId = undefined;
    } else {
      this.items = this.items.filter((d) => d.groupId !== id);
    }
    if (!this.items.some((d) => d.id === this.selectedId)) this.selectedId = null;
    this.notify(true);
  }

  /**
   * Replace everything in folder `id` (created on top if missing) with
   * `drawings`, as one undo step. Used for the AI's chart marks.
   */
  replaceGroup(id: string, name: string, drawings: Drawing[]): void {
    this.checkpoint();
    const g = this.groupOf(id);
    if (g) g.name = name;
    else this.groups.unshift({ id, name });
    this.items = this.items.filter((d) => d.groupId !== id);
    for (const d of drawings) this.items.push({ ...d, groupId: id, style: { ...DEFAULT_STYLE, ...d.style } });
    if (!this.items.some((d) => d.id === this.selectedId)) this.selectedId = null;
    this.notify(true);
  }

  private checkpoint(): void {
    this.past.push(this.snapshot());
    if (this.past.length > 100) this.past.shift();
    this.future = [];
  }

  private notify(itemsChanged: boolean): void {
    this.version++;
    this.chart.invalidate();
    for (const l of this.listeners) l();
    if (itemsChanged) for (const l of this.changeListeners) l(this.items);
  }

  // ======================================================================
  // Coordinates
  // ======================================================================

  private paneOf(item: Drawing): Pane {
    return this.chart.panes.find((p) => p.id === item.paneId) ?? this.chart.mainPane;
  }

  /** The drawing as painted: "auto" colours resolved for the current theme. */
  private themed(item: Drawing): Drawing {
    const t = this.chart.theme;
    const color = item.style.color === AUTO_COLOR ? t.drawing : readable(item.style.color, t.background);
    const textColor = item.style.textColor === AUTO_COLOR ? t.drawing : item.style.textColor;
    if (color === item.style.color && textColor === item.style.textColor) return item;
    return { ...item, style: { ...item.style, color, textColor } };
  }

  private context(raw: Drawing, c: PrimitiveContext | null = null): ToolContext {
    const chart = this.chart;
    const item = this.themed(raw);
    const pane = this.paneOf(item);
    const rect = { x: 0, y: pane.rect.y, width: chart.plotWidth, height: pane.rect.height };
    return {
      s: c?.surface ?? measureOnly(),
      chart,
      pane,
      rect,
      item,
      points: item.anchors.map((a) => ({ x: chart.timeToX(a.time), y: chart.priceToY(pane, a.price) })),
      theme: chart.theme,
      selected: raw.id === this.selectedId,
      bars: chart.bars(),
      toX: (time) => chart.timeToX(time),
      toY: (price) => chart.priceToY(pane, price),
      fromX: (x) => chart.xToTime(x),
      fromY: (y) => chart.yToPrice(pane, y),
      fmt: (price) => chart.formatPrice(pane, price),
    };
  }

  /** Pixel → anchor: snaps to bar times (not for brushes), optional magnet. */
  private toAnchor(p: Point, pane: Pane, tool: Tool | undefined): Anchor {
    const t = this.chart.time;
    const brush = tool?.group === "brush";
    const index = brush ? t.xToIndex(p.x) : Math.round(t.xToIndex(p.x));
    let price = this.chart.yToPrice(pane, p.y);
    if (this.magnet && !brush) {
      const bar = this.chart.bars()[index];
      if (bar) {
        let best: number | null = null;
        for (const v of [bar.open, bar.high, bar.low, bar.close]) {
          const d = Math.abs(this.chart.priceToY(pane, v) - p.y);
          if (d < 30 && (best === null || d < Math.abs(this.chart.priceToY(pane, best) - p.y))) best = v;
        }
        if (best !== null) price = best;
      }
    }
    return { time: t.indexToTime(index), price };
  }

  private handlesOf(ctx: ToolContext, tool: Tool): Point[] {
    return tool.handles ? tool.handles(ctx) : ctx.points;
  }

  /** Topmost drawing (or a handle of the selected one) under p. */
  private hitTest(p: Point): { item: Drawing; handle: number | null } | null {
    const selected = this.selected;
    if (selected && !selected.hidden) {
      const tool = getTool(selected.type);
      if (tool) {
        const handles = safe(() => this.handlesOf(this.context(selected, null), tool), []);
        const i = handles.findIndex((h) => dist(h, p) <= HANDLE_RADIUS + 3);
        if (i >= 0) return { item: selected, handle: i };
      }
    }
    for (let i = this.items.length - 1; i >= 0; i--) {
      const item = this.items[i]!;
      if (item.hidden) continue;
      const tool = getTool(item.type);
      if (!tool) continue;
      const ctx = this.context(item, null);
      if (p.y < ctx.rect.y || p.y > ctx.rect.y + ctx.rect.height) continue;
      if (safe(() => tool.hit(ctx, p), null) !== null) return { item, handle: null };
    }
    return null;
  }

  // ======================================================================
  // Painting
  // ======================================================================

  draw(c: PrimitiveContext): void {
    const all = this.creating ? [...this.items, this.creating.item] : this.items;
    for (const item of all) {
      if (item.hidden) continue;
      const tool = getTool(item.type);
      if (!tool) continue;
      const ctx = this.context(item, c);
      c.surface.clip(ctx.rect);
      const creating = item === this.creating?.item;
      try {
        tool.render(ctx);
        // while placing, show the raw points (derived handles need all anchors)
        const handles = creating ? ctx.points.slice(0, this.creating!.placed + 1) : item.id === this.selectedId ? this.handlesOf(ctx, tool) : [];
        for (const h of handles) drawHandle(c, h, ctx.item.style.color);
      } catch {
        // a half-placed drawing may not have enough points yet
      }
      c.surface.restore();
    }
  }

  drawOverAxes(c: PrimitiveContext): void {
    const item = this.creating?.item ?? this.selected;
    if (!item || item.hidden) return;
    const tool = getTool(item.type);
    if (!tool) return;
    const ctx = this.context(item, c);
    const tagColor = ctx.item.style.color;
    const tags = tool.axisTags?.(ctx) ?? { prices: item.anchors.map((a) => a.price), times: item.anchors.map((a) => a.time) };
    const s = c.surface;
    const t = c.theme;
    const font = { size: t.fontSize, family: t.fontFamily };
    const axisX = this.chart.plotWidth;
    const h = t.fontSize + 8;
    for (const price of tags.prices) {
      const y = ctx.toY(price);
      if (y < ctx.rect.y || y > ctx.rect.y + ctx.rect.height) continue;
      roundRect(s, axisX + 1, y - h / 2, this.chart.width - axisX - 2, h, 2, tagColor);
      s.text(ctx.fmt(price), axisX + 8, y, textOn(tagColor), font);
    }
    const timeTop = this.chart.plotBottom;
    for (const time of tags.times) {
      const x = ctx.toX(time);
      if (x < 0 || x > axisX) continue;
      const text = formatTimeLabel(time);
      const w = s.measure(text, font) + 14;
      roundRect(s, x - w / 2, timeTop + 2, w, h, 2, tagColor);
      s.text(text, x, timeTop + 2 + h / 2, textOn(tagColor), font, "center");
    }
  }

  // ======================================================================
  // Interaction
  // ======================================================================

  private newItem(tool: Tool, pane: Pane, anchor: Anchor): Drawing {
    const item: Drawing = {
      id: newId(),
      type: tool.type,
      paneId: pane.id,
      anchors: [anchor],
      style: { ...DEFAULT_STYLE, ...tool.defaults, ...this.toolStyles[tool.type] },
    };
    tool.init?.(item);
    return item;
  }

  private previewAnchors(cr: Creating, anchor: Anchor): void {
    const count = typeof cr.tool.create === "number" ? cr.tool.create : cr.placed + 1;
    const fixed = cr.item.anchors.slice(0, cr.placed);
    cr.item.anchors = [...fixed, ...Array.from({ length: Math.max(0, count - cr.placed) }, () => ({ ...anchor }))];
  }

  private pointerAnchor(info: PointerInfo, cr: Creating | null): Anchor {
    const pane = cr ? this.paneOf(cr.item) : info.pane;
    let p: Point = { x: info.x, y: info.y };
    // Shift: snap 2-point lines to 0 / 45 / 90 degrees
    if (info.shiftKey && cr && cr.placed >= 1 && cr.tool.group === "lines") {
      const a = cr.item.anchors[0]!;
      p = snapAngle({ x: this.chart.timeToX(a.time), y: this.chart.priceToY(pane, a.price) }, p);
    }
    return this.toAnchor(p, pane, cr?.tool ?? (this.activeTool ? getTool(this.activeTool) : undefined));
  }

  down(info: PointerInfo): boolean {
    if (info.button !== 0) return false;
    const toolType = this.activeTool;
    // An armed tool always places a point (Alt selects through instead).
    if (toolType && !info.altKey) {
      const tool = getTool(toolType);
      if (!tool) return false;
      const cr = this.creating;
      if (cr) {
        // click-mode: fix the next point
        const anchor = this.pointerAnchor(info, cr);
        this.previewAnchors(cr, anchor);
        cr.placed += 1;
        cr.downAt = { x: info.x, y: info.y };
        cr.dragged = false;
        if (cr.tool.create === "multi") {
          this.previewAnchors(cr, anchor);
        } else if (cr.placed >= (cr.tool.create as number)) {
          this.finish();
        }
        this.chart.invalidate();
        return true;
      }
      const anchor = this.toAnchor({ x: info.x, y: info.y }, info.pane, tool);
      const item = this.newItem(tool, info.pane, anchor);
      const next: Creating = { tool, item, placed: 1, downAt: { x: info.x, y: info.y }, dragged: false };
      this.creating = next;
      if (tool.create === "stroke") {
        this.drag = { kind: "stroke", last: { x: info.x, y: info.y } };
      } else if (tool.create === 1) {
        this.finish();
      } else {
        this.previewAnchors(next, anchor);
      }
      this.notify(false);
      return true;
    }

    const hit = this.hitTest({ x: info.x, y: info.y });
    if (!hit) {
      if (this.selectedId) this.select(null);
      return false;
    }
    this.selectedId = hit.item.id;
    if (!hit.item.locked) {
      const startAnchors = hit.item.anchors.map((a) => ({ ...a }));
      const before = this.snapshot();
      if (hit.handle !== null) this.drag = { kind: "handle", id: hit.item.id, index: hit.handle, startAnchors, before, moved: false };
      else this.drag = { kind: "body", id: hit.item.id, start: { x: info.x, y: info.y }, startPoints: this.context(hit.item).points, startAnchors, before, moved: false };
    }
    this.notify(false);
    return true;
  }

  move(info: PointerInfo): void {
    const cr = this.creating;
    if (cr && this.drag?.kind === "stroke") {
      if (dist(this.drag.last, info) < 3) return;
      this.drag.last = { x: info.x, y: info.y };
      cr.item.anchors.push(this.toAnchor(info, this.paneOf(cr.item), cr.tool));
      this.chart.invalidate();
      return;
    }
    if (cr) {
      if (cr.downAt && dist(cr.downAt, info) > 3) cr.dragged = true;
      this.previewAnchors(cr, this.pointerAnchor(info, cr));
      this.chart.invalidate();
      return;
    }
    const drag = this.drag;
    if (!drag || drag.kind === "stroke") return;
    const item = this.items.find((d) => d.id === drag.id);
    if (!item) return;
    const tool = getTool(item.type);
    const pane = this.paneOf(item);
    if (!drag.moved) {
      // one undo step per drag, recorded only once something actually moves
      drag.moved = true;
      this.past.push(drag.before);
      if (this.past.length > 100) this.past.shift();
      this.future = [];
      this.lastCoalesce = null;
    }
    if (drag.kind === "handle") {
      const ctx = this.context({ ...item, anchors: drag.startAnchors });
      const next = tool?.dragHandle ? tool.dragHandle(ctx, drag.index, { x: info.x, y: info.y }) : replaceAt(ctx.points, drag.index, { x: info.x, y: info.y });
      if (!next) return;
      item.anchors = next.map((p) => this.toAnchor(p, pane, tool));
    } else {
      const dx = info.x - drag.start.x;
      const dy = info.y - drag.start.y;
      item.anchors = drag.startPoints.map((p) => this.toAnchor({ x: p.x + dx, y: p.y + dy }, pane, tool));
    }
    this.notify(true);
  }

  up(info: PointerInfo): void {
    const cr = this.creating;
    if (cr && this.drag?.kind === "stroke") {
      this.drag = null;
      this.finish();
      return;
    }
    if (cr) {
      // press-drag-release places the second point
      if (cr.dragged && cr.placed === 1 && typeof cr.tool.create === "number" && cr.tool.create >= 2) {
        this.previewAnchors(cr, this.pointerAnchor(info, cr));
        cr.placed = 2;
        if (cr.placed >= cr.tool.create) this.finish();
      }
      cr.downAt = null;
      return;
    }
    if (this.drag) {
      const moved = this.drag.kind !== "stroke" && this.drag.moved;
      this.drag = null;
      this.notify(moved);
    }
  }

  hover(info: PointerInfo): string | undefined {
    const cr = this.creating;
    if (cr) {
      this.previewAnchors(cr, this.pointerAnchor(info, cr));
      this.chart.invalidate();
      return "crosshair";
    }
    if (this.activeTool && !info.altKey) return "crosshair";
    const hit = this.hitTest({ x: info.x, y: info.y });
    const id = hit?.item.id ?? null;
    if (id !== this.hoveredId) {
      this.hoveredId = id;
      this.chart.invalidate();
    }
    if (!hit) return undefined;
    return hit.handle !== null ? "pointer" : hit.item.locked ? "default" : "grab";
  }

  dblclick(info: PointerInfo): boolean {
    const cr = this.creating;
    if (cr && cr.tool.create === "multi") {
      // the double-click's own clicks added duplicate points
      cr.item.anchors = cr.item.anchors.slice(0, Math.max(2, cr.placed - 1));
      cr.placed = cr.item.anchors.length;
      this.finish();
      return true;
    }
    const hit = this.hitTest({ x: info.x, y: info.y });
    if (hit && getTool(hit.item.type)?.caps?.text) {
      this.emit({ type: "editText", item: hit.item });
      return true;
    }
    return false;
  }

  private finish(): void {
    const cr = this.creating;
    if (!cr) return;
    this.creating = null;
    const tool = cr.tool;
    let anchors = cr.item.anchors.slice(0, typeof tool.create === "number" ? tool.create : cr.item.anchors.length);
    if (tool.complete) anchors = tool.complete(anchors);
    // drop a zero-size two-point drawing (a stray click)
    if (anchors.length >= 2 && tool.create !== 1) {
      const pane = this.paneOf(cr.item);
      const pts = anchors.map((a) => ({ x: this.chart.timeToX(a.time), y: this.chart.priceToY(pane, a.price) }));
      if (pts.every((p) => dist(p, pts[0]!) < 2)) {
        this.notify(false);
        return;
      }
    }
    if (this.capture) {
      const pick = this.capture;
      this.capture = null;
      this.activeTool = null;
      this.notify(false);
      const [a, b] = anchors;
      pick(Math.min(a!.time, b!.time), Math.max(a!.time, b!.time));
      return;
    }
    cr.item.anchors = anchors;
    this.checkpoint();
    this.items.push(cr.item);
    this.selectedId = cr.item.id;
    if (!this.keepDrawing) this.activeTool = null;
    this.notify(true);
    this.emit({ type: "created", item: cr.item });
  }

  private cancelCreating(): void {
    this.creating = null;
    this.drag = null;
  }

  private emit(e: LayerEvent): void {
    for (const l of this.eventListeners) l(e);
  }

  private onKey(e: KeyboardEvent): void {
    const target = e.target as HTMLElement | null;
    if (target && /INPUT|TEXTAREA|SELECT/.test(target.tagName)) return;
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Escape") {
      // tell outer handlers (e.g. a modal closing on Esc) when the chart used it
      if (this.creating || this.activeTool || this.selectedId) e.preventDefault();
      this.capture = null;
      if (this.creating) {
        if (this.creating.tool.create === "multi" && this.creating.placed >= 2) {
          this.creating.item.anchors = this.creating.item.anchors.slice(0, this.creating.placed);
          this.finish();
        } else this.cancelCreating();
      } else if (this.activeTool) this.activeTool = null;
      else this.selectedId = null;
      this.notify(false);
    } else if (e.key === "Enter" && this.creating?.tool.create === "multi") {
      this.creating.item.anchors = this.creating.item.anchors.slice(0, this.creating.placed);
      this.finish();
    } else if ((e.key === "Delete" || e.key === "Backspace") && this.selectedId) {
      e.preventDefault();
      this.remove(this.selectedId);
    } else if (mod && e.key.toLowerCase() === "z" && !e.shiftKey) {
      e.preventDefault();
      this.undo();
    } else if (mod && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
      e.preventDefault();
      this.redo();
    }
  }
}

let measurer: CanvasRenderingContext2D | null | undefined;
/** Hit tests only need text measurement; paint calls are no-ops. */
function measureOnly(): Surface {
  if (measurer === undefined) {
    try {
      measurer = typeof document !== "undefined" ? document.createElement("canvas").getContext("2d") : null;
    } catch {
      measurer = null;
    }
  }
  const noop = () => undefined;
  return new Proxy({} as Surface, {
    get(_t, key) {
      if (key === "measure")
        return (text: string, font: { size: number; family: string; weight?: string }) => {
          if (!measurer) return text.length * font.size * 0.6;
          measurer.font = `${font.weight ?? "normal"} ${font.size}px ${font.family}`;
          return measurer.measureText(text).width;
        };
      if (key === "hpr" || key === "vpr") return 1;
      return noop;
    },
  });
}

/** A malformed (e.g. hand-edited, stored) drawing must not break the chart. */
function safe<T>(f: () => T, fallback: T): T {
  try {
    return f();
  } catch {
    return fallback;
  }
}

function replaceAt(points: Point[], index: number, p: Point): Point[] {
  const out = [...points];
  out[index] = p;
  return out;
}

function pickStyle(style: DrawingStyle): Partial<DrawingStyle> {
  const { color, width, lineStyle, fillColor, fillOpacity, textColor, fontSize } = style;
  return { color, width, lineStyle, fillColor, fillOpacity, textColor, fontSize };
}

function drawHandle(c: PrimitiveContext, p: Point, color: string): void {
  const s = c.surface;
  const ctx = s.ctx;
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x * s.hpr, p.y * s.vpr, HANDLE_RADIUS * s.hpr, 0, Math.PI * 2);
  ctx.fillStyle = c.theme.background;
  ctx.fill();
  ctx.lineWidth = 1.5 * s.hpr;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.restore();
}
