import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { onSyncedKeys, startChartSync, stopChartSync } from "./sync";

/** In-memory stand-in for the Supabase table (server clock = a counter). */
function fakeServer() {
  const rows = new Map<string, { key: string; data: string | null; updated_at: string }>();
  let clock = 0;
  const stamp = () => `2026-01-01T00:00:${String(++clock).padStart(6, "0")}Z`;
  const client = {
    from() {
      return {
        upsert(list: { key: string; data: string | null }[]) {
          const saved = list.map((r) => {
            const row = { key: r.key, data: r.data, updated_at: stamp() };
            rows.set(r.key, row);
            return { key: row.key, updated_at: row.updated_at };
          });
          return { select: async () => ({ data: saved, error: null }) };
        },
        select() {
          let since: string | null = null;
          const q = {
            order: () => q,
            limit: () => q,
            gt: (_: string, v: string) => ((since = v), q),
            then: (resolve: (r: unknown) => void) =>
              resolve({ data: [...rows.values()].filter((r) => !since || r.updated_at > since).sort((a, b) => a.updated_at.localeCompare(b.updated_at)), error: null }),
          };
          return q;
        },
      };
    },
  };
  return { client, rows };
}

const flushTimers = async () => {
  await vi.advanceTimersByTimeAsync(2000);
};

describe("chart sync", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
  });
  afterEach(() => {
    stopChartSync();
    vi.useRealTimers();
  });

  it("uploads this device's drawings and brings them to another device", async () => {
    const server = fakeServer();
    const key = "smartdca.chart.drawings.tech.crypto.BTC";

    // device A: drew before signing in, then signs in
    localStorage.setItem(key, '[{"id":"a"}]');
    localStorage.setItem("smartdca.ai.settings", '{"port":1}'); // device-only, never synced
    startChartSync(server.client, "u1");
    await flushTimers();
    expect(server.rows.get(key)?.data).toBe('[{"id":"a"}]');
    expect(server.rows.has("smartdca.ai.settings")).toBe(false);

    // edits upload a moment later
    localStorage.setItem(key, '[{"id":"a"},{"id":"b"}]');
    await flushTimers();
    expect(server.rows.get(key)?.data).toBe('[{"id":"a"},{"id":"b"}]');
    stopChartSync();

    // device B: empty storage, same account
    localStorage.clear();
    const seen: string[][] = [];
    const off = onSyncedKeys((keys) => seen.push(keys));
    startChartSync(server.client, "u1");
    await flushTimers();
    expect(localStorage.getItem(key)).toBe('[{"id":"a"},{"id":"b"}]');
    expect(seen.flat()).toContain(key);

    // B deletes the drawings; the deletion syncs too
    localStorage.removeItem(key);
    await flushTimers();
    expect(server.rows.get(key)?.data).toBeNull();
    off();
  });

  it("applies another device's newer change and does not echo it back", async () => {
    const server = fakeServer();
    const key = "chart-watchlist-v1";
    startChartSync(server.client, "u1");
    localStorage.setItem(key, '{"groups":[]}');
    await flushTimers();
    const uploaded = server.rows.get(key)!.updated_at;

    // another device writes a newer value
    server.rows.set(key, { key, data: '{"groups":[{"id":"x"}]}', updated_at: "2099-01-01T00:00:00Z" });
    window.dispatchEvent(new Event("focus"));
    await flushTimers();
    expect(localStorage.getItem(key)).toBe('{"groups":[{"id":"x"}]}');
    // nothing re-uploaded: the server row is still the other device's
    expect(server.rows.get(key)!.updated_at).toBe("2099-01-01T00:00:00Z");
    expect(uploaded).not.toBe("2099-01-01T00:00:00Z");
  });
});
