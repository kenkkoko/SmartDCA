// Cross-device sync of chart data (drawings, chart settings, watchlist,
// indicator toggles) through Supabase table `chart_sync`
// (supabase/chart_sync.sql).
//
// localStorage stays the working copy, so charts work offline and logged out.
// While signed in:
//   - every write to a synced key is uploaded ~1.5 s later (last write wins
//     per key; times come from the server, so device clocks do not matter)
//   - rows changed on other devices are pulled on start, when the tab comes
//     back into view and every 30 s, written to localStorage, and announced
//     with a "smartdca:sync" event so open charts / lists reload them

type Row = { key: string; data: string | null; updated_at: string };

/* eslint-disable @typescript-eslint/no-explicit-any */
type Client = any;

export const SYNC_EVENT = "smartdca:sync";

export function isSyncedKey(key: string): boolean {
  return (
    key.startsWith("smartdca.chart.drawings.") ||
    key === "smartdca.chart.toolStyles" ||
    key === "smartdca.chart.fibDefaults" ||
    key === "smartdca.chart.drawingPrefs" ||
    key === "smartdca.chart.appearance" ||
    key === "chart-watchlist-v1" ||
    key === "tech-timeframe" ||
    key.startsWith("tech2-")
  );
}

const META_KEY = "smartdca.sync.meta";
type Meta = { user: string; lastPull: string | null; keys: Record<string, string> };

type State = {
  client: Client;
  user: string;
  meta: Meta;
  dirty: Set<string>;
  timer: ReturnType<typeof setTimeout> | null;
  disabled: boolean;
  pulling: boolean;
};

let state: State | null = null;
/** True while writing rows that came from the server (not echoed back). */
let applying = false;
let patched = false;

const proto = typeof Storage !== "undefined" ? Storage.prototype : null;

function readMeta(user: string): Meta {
  try {
    const m = JSON.parse(localStorage.getItem(META_KEY) || "null") as Meta | null;
    if (m && m.user === user && m.keys) return m;
  } catch {
    // corrupt: start over
  }
  return { user, lastPull: null, keys: {} };
}

function saveMeta(): void {
  if (!state) return;
  try {
    applying = true;
    localStorage.setItem(META_KEY, JSON.stringify(state.meta));
  } catch {
    // quota: sync still works this session
  } finally {
    applying = false;
  }
}

/** Watch writes to localStorage (the chart code and React state write there directly). */
function patchStorage(): void {
  if (patched || !proto) return;
  patched = true;
  const setItem = proto.setItem;
  const removeItem = proto.removeItem;
  proto.setItem = function (this: Storage, key: string, value: string) {
    const before = this === window.localStorage && state && !applying ? this.getItem(key) : undefined;
    setItem.call(this, key, value);
    if (before !== undefined && before !== String(value) && isSyncedKey(key)) markDirty(key);
  };
  proto.removeItem = function (this: Storage, key: string) {
    const had = this === window.localStorage && state && !applying ? this.getItem(key) !== null : false;
    removeItem.call(this, key);
    if (had && isSyncedKey(key)) markDirty(key);
  };
}

function markDirty(key: string): void {
  if (!state || state.disabled) return;
  state.dirty.add(key);
  if (state.timer) clearTimeout(state.timer);
  state.timer = setTimeout(() => void flush(), 1500);
}

function missingTable(error: { code?: string; message?: string } | null): boolean {
  return !!error && (error.code === "PGRST205" || error.code === "42P01" || /chart_sync/.test(error.message ?? ""));
}

function disable(error: { message?: string }): void {
  if (!state) return;
  state.disabled = true;
  console.info("[chart sync] 停用跨裝置同步：", error.message ?? error, "（請在 Supabase 執行 supabase/chart_sync.sql）");
}

async function flush(): Promise<void> {
  const s = state;
  if (!s || s.disabled || !s.dirty.size) return;
  s.timer = null;
  const keys = [...s.dirty];
  s.dirty.clear();
  const rows = keys.map((key) => ({ user_id: s.user, key, data: localStorage.getItem(key) }));
  const { data, error } = await s.client.from("chart_sync").upsert(rows, { onConflict: "user_id,key" }).select("key,updated_at");
  if (state !== s) return;
  if (error) {
    if (missingTable(error)) return disable(error);
    // offline etc.: try again later
    for (const k of keys) s.dirty.add(k);
    s.timer = setTimeout(() => void flush(), 15000);
    return;
  }
  for (const row of (data ?? []) as Row[]) s.meta.keys[row.key] = row.updated_at;
  saveMeta();
}

async function pull(): Promise<void> {
  const s = state;
  if (!s || s.disabled || s.pulling) return;
  s.pulling = true;
  try {
    let query = s.client.from("chart_sync").select("key,data,updated_at").order("updated_at", { ascending: true }).limit(2000);
    if (s.meta.lastPull) query = query.gt("updated_at", s.meta.lastPull);
    const { data, error } = await query;
    if (state !== s) return;
    if (error) {
      if (missingTable(error)) disable(error);
      return;
    }
    const changed: string[] = [];
    for (const row of (data ?? []) as Row[]) {
      if (row.updated_at > (s.meta.lastPull ?? "")) s.meta.lastPull = row.updated_at;
      if (!isSyncedKey(row.key) || s.dirty.has(row.key)) continue; // a pending local edit wins
      const known = s.meta.keys[row.key];
      if (known && known >= row.updated_at) continue; // our own upload coming back
      s.meta.keys[row.key] = row.updated_at;
      if (localStorage.getItem(row.key) === row.data) continue;
      applying = true;
      try {
        if (row.data === null) localStorage.removeItem(row.key);
        else localStorage.setItem(row.key, row.data);
      } finally {
        applying = false;
      }
      changed.push(row.key);
    }
    saveMeta();
    if (changed.length) window.dispatchEvent(new CustomEvent(SYNC_EVENT, { detail: { keys: changed } }));
  } finally {
    s.pulling = false;
  }
}

/**
 * Start syncing for a signed-in user; returns a stop function. The first run
 * on a device downloads the account's data, then uploads what only this
 * device has (e.g. the drawings made before signing in).
 */
export function startChartSync(client: Client, userId: string): () => void {
  if (!client || !userId || typeof window === "undefined") return () => {};
  stopChartSync();
  patchStorage();
  state = { client, user: userId, meta: readMeta(userId), dirty: new Set(), timer: null, disabled: false, pulling: false };
  const s = state;
  void (async () => {
    await pull();
    if (state !== s || s.disabled) return;
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && isSyncedKey(key) && !s.meta.keys[key]) markDirty(key);
    }
  })();
  const onVisible = () => {
    if (document.visibilityState === "visible") void pull();
    else void flush();
  };
  const timer = setInterval(() => document.visibilityState === "visible" && void pull(), 30000);
  document.addEventListener("visibilitychange", onVisible);
  window.addEventListener("focus", onVisible);
  window.addEventListener("pagehide", onVisible);
  cleanup = () => {
    clearInterval(timer);
    document.removeEventListener("visibilitychange", onVisible);
    window.removeEventListener("focus", onVisible);
    window.removeEventListener("pagehide", onVisible);
  };
  return stopChartSync;
}

let cleanup: (() => void) | null = null;

export function stopChartSync(): void {
  cleanup?.();
  cleanup = null;
  if (state?.timer) {
    clearTimeout(state.timer);
    void flush();
  }
  state = null;
}

/** Run `onChange` when synced keys arrive from another device. */
export function onSyncedKeys(onChange: (keys: string[]) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (e: Event) => onChange(((e as CustomEvent).detail?.keys as string[]) ?? []);
  window.addEventListener(SYNC_EVENT, handler);
  return () => window.removeEventListener(SYNC_EVENT, handler);
}
