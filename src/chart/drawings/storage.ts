// localStorage persistence: drawings per symbol, Fibonacci template and the
// last style used per tool. Every access is guarded (private mode, quota).

import { getFibDefaults, setFibDefaults } from "./tools/fibonacci";
import type { DrawingLayer } from "./layer";
import type { Drawing, DrawingStyle, FibSettings } from "./types";

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

export function loadDrawings(symbol: string): Drawing[] {
  const items = read<Drawing[]>(`drawings.${symbol}`);
  return Array.isArray(items) ? items.filter((d) => d && typeof d.type === "string" && Array.isArray(d.anchors)) : [];
}

export function saveDrawings(symbol: string, items: Drawing[]): void {
  write(`drawings.${symbol}`, items.length ? items : null);
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
  layer.setItems(loadDrawings(symbol));
  let timer: ReturnType<typeof setTimeout> | null = null;
  const offItems = layer.onItemsChange((items) => {
    // drags fire many changes; write at most every 250 ms
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      saveDrawings(symbol, items);
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
  return () => {
    if (timer) {
      clearTimeout(timer);
      saveDrawings(symbol, layer.items);
      write("toolStyles", layer.toolStyles);
    }
    offItems();
    offState();
  };
}
