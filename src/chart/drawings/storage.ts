// localStorage persistence: drawings per symbol, Fibonacci template and the
// last style used per tool. Every access is guarded (private mode, quota).

import { onSyncedKeys } from "../sync";
import { getFibDefaults, setFibDefaults } from "./tools/fibonacci";
import type { DrawingLayer } from "./layer";
import type { Drawing, DrawingGroup, DrawingStyle, FibSettings } from "./types";

const PREFIX = "smartdca.chart.";

function read<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: unknown): void {
  try {
    if (value === null) localStorage.removeItem(PREFIX + key);
    else localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // storage full or blocked: drawings still live in memory
  }
}

const validDrawing = (d: Drawing) => d && typeof d.type === "string" && Array.isArray(d.anchors);

/** Drawings (and layers-panel folders) for one symbol. Older saves are a bare array. */
export function loadLayerState(symbol: string): { items: Drawing[]; groups: DrawingGroup[] } {
  const raw = read<Drawing[] | { items: Drawing[]; groups?: DrawingGroup[] }>(`drawings.${symbol}`);
  if (Array.isArray(raw)) return { items: raw.filter(validDrawing), groups: [] };
  if (raw && Array.isArray(raw.items))
    return { items: raw.items.filter(validDrawing), groups: Array.isArray(raw.groups) ? raw.groups.filter((g) => g && typeof g.id === "string") : [] };
  return { items: [], groups: [] };
}

export function loadDrawings(symbol: string): Drawing[] {
  return loadLayerState(symbol).items;
}

export function saveDrawings(symbol: string, items: Drawing[], groups: DrawingGroup[] = []): void {
  if (!items.length && !groups.length) return write(`drawings.${symbol}`, null);
  write(`drawings.${symbol}`, groups.length ? { items, groups } : items);
}

/** Restore the saved Fibonacci template into the tool defaults. */
export function loadFibTemplate(): void {
  const saved = read<FibSettings>("fibDefaults");
  if (saved && Array.isArray(saved.levels)) setFibDefaults({ ...getFibDefaults(), ...saved });
}

export function saveFibTemplate(settings: FibSettings | null): void {
  setFibDefaults(settings);
  write("fibDefaults", settings);
}

/**
 * Wire a layer to storage for `symbol`: loads its drawings, the per-tool
 * styles and toggles, and saves on every change. Returns the unsubscribe.
 */
export function persistLayer(layer: DrawingLayer, symbol: string): () => void {
  loadFibTemplate();
  layer.toolStyles = read<Record<string, Partial<DrawingStyle>>>("toolStyles") ?? {};
  const prefs = read<{ keepDrawing?: boolean; magnet?: boolean }>("drawingPrefs") ?? {};
  layer.keepDrawing = !!prefs.keepDrawing;
  layer.magnet = !!prefs.magnet;
  const state = loadLayerState(symbol);
  layer.setItems(state.items, state.groups);
  let timer: ReturnType<typeof setTimeout> | null = null;
  const offItems = layer.onItemsChange(() => {
    // drags fire many changes; write at most every 250 ms
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      saveDrawings(symbol, layer.items, layer.groups);
      write("toolStyles", layer.toolStyles);
    }, 250);
  });
  let lastPrefs = JSON.stringify(prefs);
  const offState = layer.subscribe(() => {
    const next = JSON.stringify({ keepDrawing: layer.keepDrawing, magnet: layer.magnet });
    if (next !== lastPrefs) {
      lastPrefs = next;
      write("drawingPrefs", JSON.parse(next));
    }
  });
  // drawings / styles changed on another device (cross-device sync)
  const offSync = onSyncedKeys((keys) => {
    if (keys.includes(PREFIX + `drawings.${symbol}`) && !timer) {
      const next = loadLayerState(symbol);
      layer.setItems(next.items, next.groups);
    }
    if (keys.includes(PREFIX + "toolStyles")) layer.toolStyles = read<Record<string, Partial<DrawingStyle>>>("toolStyles") ?? {};
    if (keys.includes(PREFIX + "fibDefaults")) loadFibTemplate();
  });
  return () => {
    if (timer) {
      clearTimeout(timer);
      saveDrawings(symbol, layer.items, layer.groups);
      write("toolStyles", layer.toolStyles);
    }
    offItems();
    offState();
    offSync();
  };
}
