// AI 分析 panel (right sidebar). Picks a range on the chart (the visible
// bars or a dragged range), packs the K-lines + indicators of that range into
// a prompt and streams the answer from the AI on this computer through the
// local bridge. Settings choose the AI, the model and the effort level.

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { DrawingLayer } from "../drawings/layer";
import { getTool } from "../drawings/tools";
import type { Chart, ChartPrimitive } from "../engine/chart";
import type { Bar } from "../engine/types";
import { PanelHeader, panelIconBtn } from "../react/SideBar";
import {
  buildPrompt,
  extractMarks,
  fetchProviders,
  MARKS_PROMPT,
  marksToDrawings,
  stripMarks,
  loadAiSettings,
  saveAiSettings,
  streamAnalysis,
  SYSTEM_PROMPT,
  type AiProvider,
  type AiSettings,
} from "./ai";

const EFFORT_LABEL: Record<string, string> = { low: "低", medium: "中", high: "高", xhigh: "極高", max: "最大" };
const day = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);

type Status = "checking" | "offline" | "online";
type Run = {
  text: string;
  state: "running" | "done" | "error";
  error?: string;
  meta: string;
  thinking: boolean;
  costUsd?: number;
  ms?: number;
  /** Asked the AI to mark the chart. */
  marks: boolean;
  /** Drawings placed from the answer (null = not yet / none found). */
  placed?: number | null;
};

/** Folder in the layers panel that holds the AI's marks (one per symbol). */
const AI_GROUP = "ai-marks";

/** Shaded band over the picked range while the panel is open. */
class RangeBand implements ChartPrimitive {
  zIndex = 5;
  constructor(public from: number, public to: number) {}
  draw({ surface, chart, theme }: Parameters<ChartPrimitive["draw"]>[0]): void {
    const x0 = chart.timeToX(this.from);
    const x1 = chart.timeToX(this.to);
    const y0 = 0;
    const h = chart.plotBottom;
    const left = Math.max(0, Math.min(x0, x1));
    const right = Math.min(chart.plotWidth, Math.max(x0, x1));
    if (right <= left) return;
    surface.fillRect(left, y0, right - left, h, `${theme.ui.accent}1f`);
    surface.vLine(left, y0, h, theme.ui.accent, 1, "dashed");
    surface.vLine(right, y0, h, theme.ui.accent, 1, "dashed");
    surface.text("AI 分析範圍", left + 6, 14, theme.ui.accent, { size: 11, family: theme.fontFamily, weight: "600" });
  }
}

/** Tiny Markdown → React (headings, bullets, numbered lists, bold, code). */
function Markdown({ text }: { text: string }) {
  const inline = (s: string, key: number): ReactNode[] =>
    s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) =>
      part.startsWith("**") && part.endsWith("**") ? (
        <strong key={`${key}-${i}`}>{part.slice(2, -2)}</strong>
      ) : part.startsWith("`") && part.endsWith("`") ? (
        <code key={`${key}-${i}`} className="rounded bg-[var(--c-raised)] px-1 text-[0.92em]">
          {part.slice(1, -1)}
        </code>
      ) : (
        part
      ),
    );
  const out: ReactNode[] = [];
  let list: { ordered: boolean; items: ReactNode[] } | null = null;
  const flush = () => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    out.push(
      <Tag key={`l${out.length}`} className={`my-1 space-y-0.5 pl-5 ${list.ordered ? "list-decimal" : "list-disc"}`}>
        {list.items}
      </Tag>,
    );
    list = null;
  };
  text.split("\n").forEach((raw, n) => {
    const line = raw.trimEnd();
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    const ul = /^\s*[-*•]\s+(.*)$/.exec(line);
    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (h) {
      flush();
      out.push(
        <h4 key={n} className="mb-1 mt-3 text-[13px] font-bold text-[var(--c-text)] first:mt-0">
          {inline(h[2]!, n)}
        </h4>,
      );
    } else if (ul || ol) {
      const ordered = !!ol;
      if (!list || list.ordered !== ordered) {
        flush();
        list = { ordered, items: [] };
      }
      list.items.push(<li key={n}>{inline((ul ?? ol)![1]!, n)}</li>);
    } else if (!line.trim()) {
      flush();
    } else {
      flush();
      out.push(
        <p key={n} className="my-1">
          {inline(line, n)}
        </p>,
      );
    }
  });
  flush();
  return <div className="text-[12.5px] leading-relaxed text-[var(--c-text)]">{out}</div>;
}

export type AiPanelProps = {
  chart: Chart | null;
  layer: DrawingLayer | null;
  bars: Bar[];
  symbol: string;
  market: string;
  timeframe: string;
  extras?: string[];
  onClose: () => void;
};

export function AiPanel({ chart, layer, bars, symbol, market, timeframe, extras, onClose }: AiPanelProps) {
  const [settings, setSettings] = useState<AiSettings>(loadAiSettings);
  const [view, setView] = useState<"analysis" | "settings">("analysis");
  const [status, setStatus] = useState<Status>("checking");
  const [providers, setProviders] = useState<AiProvider[]>([]);
  const [detectError, setDetectError] = useState<string | null>(null);
  const [mode, setMode] = useState<"visible" | "custom">("visible");
  const [custom, setCustom] = useState<{ from: number; to: number } | null>(null);
  const [picking, setPicking] = useState(false);
  const [question, setQuestion] = useState("");
  const [includeDrawings, setIncludeDrawings] = useState(true);
  const [run, setRun] = useState<Run | null>(null);
  const [, setTick] = useState(0);
  const abort = useRef<AbortController | null>(null);

  const update = (patch: Partial<AiSettings>) => {
    setSettings((s) => {
      const next = { ...s, ...patch };
      saveAiSettings(next);
      return next;
    });
  };

  const detect = async (refresh = false) => {
    setStatus("checking");
    setDetectError(null);
    try {
      const list = await fetchProviders(settings.port, refresh);
      setProviders(list);
      setStatus("online");
      // pick a working AI / model the first time
      setSettings((s) => {
        let next = s;
        const p = list.find((x) => x.id === s.provider && x.available) ?? list.find((x) => x.available);
        if (p && p.id !== s.provider) next = { ...next, provider: p.id, model: "" };
        if (p && !p.models.some((m) => m.id === next.model)) next = { ...next, model: p.models[0]?.id ?? "" };
        if (next !== s) saveAiSettings(next);
        return next;
      });
    } catch (e) {
      setStatus("offline");
      setDetectError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    void detect();
    // re-check when the port changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.port]);

  // the visible range label follows panning / zooming
  useEffect(() => (chart ? chart.onViewChange(() => setTick((n) => n + 1)) : undefined), [chart]);

  // shaded band for a picked range
  useEffect(() => {
    if (!chart || mode !== "custom" || !custom) return;
    return chart.addPrimitive(new RangeBand(custom.from, custom.to));
  }, [chart, mode, custom]);

  useEffect(() => () => abort.current?.abort(), []);
  // a new symbol: the old range means nothing
  useEffect(() => {
    setCustom(null);
    setMode("visible");
  }, [symbol]);

  const provider = providers.find((p) => p.id === settings.provider);
  const model = provider?.models.find((m) => m.id === settings.model);
  const efforts = model?.efforts ?? [];

  const indices = (() => {
    if (!bars.length) return null;
    const indexOf = (t: number) => {
      let i = bars.findIndex((b) => b.time >= t);
      if (i < 0) i = bars.length - 1;
      return i;
    };
    if (mode === "custom" && custom) {
      const a = indexOf(custom.from);
      let b = indexOf(custom.to);
      if (bars[b]!.time > custom.to && b > a) b -= 1;
      return { from: a, to: b };
    }
    if (!chart) return { from: 0, to: bars.length - 1 };
    const v = chart.time.visibleRange();
    return { from: Math.max(0, Math.ceil(v.from)), to: Math.min(bars.length - 1, Math.floor(v.to)) };
  })(); // recomputed every render: the visible range follows panning

  const rangeLabel = indices && bars[indices.from] && bars[indices.to] ? `${day(bars[indices.from]!.time)} ~ ${day(bars[indices.to]!.time)}（${indices.to - indices.from + 1} 根）` : "—";

  const pick = () => {
    if (!layer) return;
    setPicking(true);
    layer.captureRange((from, to) => {
      setCustom({ from, to });
      setMode("custom");
      setPicking(false);
    });
    layer.chart.root.focus();
  };
  // stop showing "框選中" if the user pressed Esc
  // (and refresh the "已在圖上標記" note when the AI folder is deleted)
  useEffect(
    () =>
      layer
        ? layer.subscribe(() => {
            if (!layer.capturing) setPicking(false);
            setTick((n) => n + 1);
          })
        : undefined,
    [layer],
  );

  const start = async () => {
    if (!indices || status !== "online" || !provider) return;
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    const drawings =
      includeDrawings && layer
        ? layer.items
            .filter((d) => !d.hidden && d.anchors.some((a) => a.time >= bars[indices.from]!.time && a.time <= bars[indices.to]!.time))
            .map((d) => ({ label: d.name || getTool(d.type)?.label || d.type, drawing: d }))
        : [];
    const prompt = buildPrompt({ symbol, market, timeframe, bars, from: indices.from, to: indices.to, drawings, extras, question });
    const meta = `${provider.name} · ${model?.name ?? (settings.model || "預設模型")}${efforts.includes(settings.effort) ? ` · 努力 ${EFFORT_LABEL[settings.effort] ?? settings.effort}` : ""} · ${rangeLabel}`;
    const marks = settings.drawMarks && !!layer;
    const snapshot = bars;
    setRun({ text: "", state: "running", meta, thinking: false, marks });
    let full = "";
    try {
      await streamAnalysis(
        settings.port,
        {
          provider: provider.id,
          model: settings.model,
          effort: efforts.includes(settings.effort) ? settings.effort : "",
          system: marks ? SYSTEM_PROMPT + MARKS_PROMPT : SYSTEM_PROMPT,
          prompt,
        },
        (e) => {
          if (e.type === "text") full += e.text;
          setRun((r) => {
            if (!r) return r;
            if (e.type === "text") return { ...r, text: r.text + e.text, thinking: false };
            if (e.type === "thinking") return { ...r, thinking: true };
            if (e.type === "usage") return { ...r, costUsd: e.costUsd };
            if (e.type === "done") return { ...r, state: "done", ms: e.ms };
            if (e.type === "error") return { ...r, state: "error", error: e.message };
            return r;
          });
        },
        ac.signal,
      );
      setRun((r) => (r && r.state === "running" ? { ...r, state: "done" } : r));
      if (marks && layer) {
        const found = extractMarks(full);
        const drawings = found ? marksToDrawings(found, snapshot) : [];
        if (drawings.length) {
          const d = new Date();
          layer.replaceGroup(AI_GROUP, `AI 標記 · ${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`, drawings);
        }
        setRun((r) => (r ? { ...r, placed: drawings.length } : r));
      }
    } catch (e) {
      if (ac.signal.aborted) setRun((r) => (r ? { ...r, state: "done", error: "已停止" } : r));
      else setRun((r) => (r ? { ...r, state: "error", error: e instanceof Error ? e.message : String(e) } : r));
    }
  };

  const select = "h-8 w-full cursor-pointer rounded-md bg-[var(--c-raised)] px-2 text-[12px] text-[var(--c-text)] outline-none ring-1 ring-inset ring-[var(--c-line)] focus:ring-[var(--c-accent)]";
  const label = "mb-1 block text-[11px] font-semibold text-[var(--c-muted)]";
  const gear = (
    <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
      <circle cx="10" cy="10" r="2.5" />
      <path d="M10 2v3M10 15v3M2 10h3M15 10h3M4.3 4.3l2.1 2.1M13.6 13.6l2.1 2.1M4.3 15.7l2.1-2.1M13.6 6.4l2.1-2.1" />
    </svg>
  );

  const statusPill =
    status === "online" ? (
      <span className="flex items-center gap-1 text-[11px] text-[var(--c-muted)]">
        <span className="h-2 w-2 rounded-full" style={{ background: "#22c55e" }} />
        已連線本機 AI
      </span>
    ) : status === "checking" ? (
      <span className="text-[11px] text-[var(--c-muted)]">偵測中…</span>
    ) : (
      <span className="flex items-center gap-1 text-[11px] text-[var(--c-danger)]">
        <span className="h-2 w-2 rounded-full bg-[var(--c-danger)]" />
        未連線
      </span>
    );

  // Start the bridge through the smartdca-ai:// link the installer registers, then wait for it
  const [launching, setLaunching] = useState(false);
  const launch = () => {
    setLaunching(true);
    window.location.href = "smartdca-ai://start";
    let tries = 0;
    const poll = async () => {
      tries++;
      try {
        await fetchProviders(settings.port);
        setLaunching(false);
        void detect();
        return;
      } catch {
        if (tries < 20) setTimeout(poll, 1500);
        else setLaunching(false);
      }
    };
    setTimeout(poll, 1500);
  };

  const step = "flex gap-2";
  const num = (n: number) => <span className="mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-[var(--c-accent)] text-[10px] font-bold text-[var(--c-accent-text)]">{n}</span>;
  const offlineHelp = (
    <div className="space-y-3 rounded-md bg-[var(--c-raised)] p-3 text-[12px] leading-relaxed text-[var(--c-text)] ring-1 ring-inset ring-[var(--c-line)]" data-testid="ai-offline">
      <div>
        <p className="font-semibold">連接你電腦上的 AI</p>
        <p className="text-[11.5px] text-[var(--c-muted)]">
          AI 分析用的是你自己電腦裡的 Claude Code、Codex 或 Ollama。網頁和 App 都不能直接開啟電腦裡的程式，所以需要裝一個小小的「本機 AI 橋接」（只要裝一次）。
        </p>
      </div>
      <button
        className="flex h-9 w-full cursor-pointer items-center justify-center gap-1.5 rounded-md bg-[var(--c-accent)] text-[12.5px] font-semibold text-[var(--c-accent-text)] hover:brightness-110 disabled:opacity-60"
        onClick={launch}
        disabled={launching}
        data-testid="ai-launch"
      >
        {launching ? "啟動中，等待連線…" : "已安裝：啟動本機 AI"}
      </button>
      <div className="space-y-1.5 border-t border-[var(--c-line)] pt-3">
        <p className="text-[11px] font-semibold text-[var(--c-muted)]">還沒安裝？（Windows）</p>
        <p className={step}>
          {num(1)}
          <span>
            先裝{" "}
            <a className="text-[var(--c-accent)] underline" href="https://nodejs.org/" target="_blank" rel="noreferrer">
              Node.js（LTS）
            </a>
            ，以及{" "}
            <a className="text-[var(--c-accent)] underline" href="https://claude.com/claude-code" target="_blank" rel="noreferrer">
              Claude Code
            </a>
            （並登入）。
          </span>
        </p>
        <p className={step}>
          {num(2)}
          <span>
            <a className="font-semibold text-[var(--c-accent)] underline" href="/downloads/smartdca-ai-bridge.zip" download data-testid="ai-download">
              下載本機 AI 橋接（zip）
            </a>
            ，解壓縮後雙擊 <code className="rounded bg-[var(--c-panel)] px-1">Install-AI-Bridge.bat</code>。
          </span>
        </p>
        <p className={step}>
          {num(3)}
          <span>之後按上面的「啟動本機 AI」就會自動開啟（瀏覽器第一次會問要不要開啟，選允許）。</span>
        </p>
        <p className="pt-1 text-[11px] text-[var(--c-muted)]">macOS / Linux：解壓縮後在資料夾執行 npm install 與 node server.mjs。</p>
      </div>
      <div className="flex items-center gap-2">
        <button className="cursor-pointer text-[11.5px] text-[var(--c-accent)] hover:underline" onClick={() => void detect(true)}>
          重新偵測
        </button>
        {detectError ? <span className="truncate text-[10.5px] text-[var(--c-muted)]">（{detectError}）</span> : null}
      </div>
    </div>
  );

  const available = providers.filter((p) => p.available);
  /** AI / model / effort pickers (shown right above the analyse button and in settings). */
  const modelControls = provider?.available ? (
    <div className="space-y-2" data-testid="ai-model-controls">
      {available.length > 1 ? (
        <div>
          <label className={label} htmlFor="ai-provider">
            AI
          </label>
          <select id="ai-provider" className={select} value={settings.provider} onChange={(e) => update({ provider: e.target.value, model: available.find((p) => p.id === e.target.value)?.models[0]?.id ?? "" })}>
            {available.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      <div>
        <label className={label} htmlFor="ai-model">
          模型{available.length <= 1 ? `（${provider.name}）` : ""}
        </label>
        <select id="ai-model" className={select} value={settings.model} onChange={(e) => update({ model: e.target.value })} data-testid="ai-model">
          {provider.models.map((m) => (
            <option key={m.id || "default"} value={m.id}>
              {m.name}
              {m.description ? ` — ${m.description}` : ""}
            </option>
          ))}
        </select>
        {provider.customModel ? (
          <input
            className={`${select} mt-1.5 cursor-text`}
            placeholder="或輸入模型名稱"
            value={provider.models.some((m) => m.id === settings.model) ? "" : settings.model}
            onChange={(e) => update({ model: e.target.value })}
          />
        ) : null}
      </div>
      <div>
        <span className={label}>努力程度（越高越仔細，也越慢、越耗用量）</span>
        {efforts.length ? (
          <div className="flex flex-wrap gap-1" data-testid="ai-effort">
            {efforts.map((e) => (
              <button
                key={e}
                onClick={() => update({ effort: e })}
                className={`h-7 cursor-pointer rounded-md px-2.5 text-[12px] ring-1 ring-inset ${
                  settings.effort === e ? "bg-[var(--c-accent-soft)] font-semibold text-[var(--c-accent)] ring-[var(--c-accent)]" : "text-[var(--c-text)] ring-[var(--c-line)] hover:bg-[var(--c-hover)]"
                }`}
              >
                {EFFORT_LABEL[e] ?? e}
              </button>
            ))}
          </div>
        ) : (
          <p className="text-[11px] text-[var(--c-muted)]">這個模型沒有努力程度設定。</p>
        )}
      </div>
    </div>
  ) : null;

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="ai-panel" onKeyDown={(e) => e.stopPropagation()}>
      <PanelHeader title="AI 分析" onClose={onClose}>
        {statusPill}
        <button className={panelIconBtn(view === "settings")} title="AI 設定：選擇電腦上的 AI、模型與努力程度" data-testid="ai-settings" onClick={() => setView(view === "settings" ? "analysis" : "settings")}>
          {gear}
        </button>
      </PanelHeader>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {view === "settings" ? (
          <div className="space-y-4">
            {status === "offline" ? offlineHelp : null}
            <div>
              <span className={label}>電腦上偵測到的 AI</span>
              <div className="space-y-1.5">
                {providers.map((p) => (
                  <button
                    key={p.id}
                    disabled={!p.available}
                    onClick={() => update({ provider: p.id, model: p.models[0]?.id ?? "" })}
                    className={`flex w-full cursor-pointer items-start gap-2 rounded-md px-2.5 py-2 text-left ring-1 ring-inset disabled:cursor-not-allowed ${
                      p.id === settings.provider && p.available ? "bg-[var(--c-accent-soft)] ring-[var(--c-accent)]" : "ring-[var(--c-line)] hover:bg-[var(--c-hover)]"
                    }`}
                    data-testid={`ai-provider-${p.id}`}
                  >
                    <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: p.available ? "#22c55e" : "var(--c-line)" }} />
                    <span className="min-w-0 flex-1">
                      <span className="block text-[12.5px] font-semibold text-[var(--c-text)]">
                        {p.name}
                        {p.experimental ? <span className="ml-1 text-[10px] font-normal text-[var(--c-muted)]">實驗性</span> : null}
                      </span>
                      <span className="block text-[11px] text-[var(--c-muted)]">{p.available ? `${p.models.length} 個模型${p.cliVersion ? ` · ${p.cliVersion}` : ""}` : p.reason}</span>
                    </span>
                  </button>
                ))}
                {status === "online" && providers.length === 0 ? <p className="text-[12px] text-[var(--c-muted)]">沒有偵測到任何 AI。</p> : null}
              </div>
              {status === "online" ? (
                <button className="mt-2 cursor-pointer text-[11px] text-[var(--c-accent)] hover:underline" onClick={() => void detect(true)}>
                  重新偵測
                </button>
              ) : null}
            </div>
            {modelControls}
            <div>
              <label className={label} htmlFor="ai-port">
                橋接連接埠
              </label>
              <input id="ai-port" className={`${select} w-28 cursor-text`} type="number" value={settings.port} onChange={(e) => update({ port: Number(e.target.value) || 8765 })} />
            </div>
            <button className="h-8 w-full cursor-pointer rounded-md bg-[var(--c-accent)] text-[12px] font-semibold text-[var(--c-accent-text)] hover:brightness-110" onClick={() => setView("analysis")}>
              完成
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {status === "offline" ? offlineHelp : null}
            <div>
              <span className={label}>分析範圍</span>
              <div className="grid grid-cols-2 gap-1">
                {(["visible", "custom"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => (m === "custom" && !custom ? pick() : setMode(m))}
                    className={`h-8 cursor-pointer rounded-md text-[12px] ring-1 ring-inset ${
                      mode === m ? "bg-[var(--c-accent-soft)] font-semibold text-[var(--c-accent)] ring-[var(--c-accent)]" : "text-[var(--c-text)] ring-[var(--c-line)] hover:bg-[var(--c-hover)]"
                    }`}
                    data-testid={`ai-range-${m}`}
                  >
                    {m === "visible" ? "目前畫面" : "自選範圍"}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-[11.5px] tabular-nums text-[var(--c-text)]">{rangeLabel}</p>
              <button
                className={`mt-1 cursor-pointer text-[11.5px] hover:underline ${picking ? "font-semibold text-[var(--c-danger)]" : "text-[var(--c-accent)]"}`}
                onClick={() => (picking ? layer?.setActiveTool(null) : pick())}
                data-testid="ai-pick"
              >
                {picking ? "在圖上點兩下或拖曳選出範圍…（Esc 取消）" : custom ? "重新在圖上框選範圍" : "在圖上框選範圍"}
              </button>
            </div>

            <div>
              <label className={label} htmlFor="ai-question">
                想問的事（可留空）
              </label>
              <textarea
                id="ai-question"
                className="h-16 w-full resize-none rounded-md bg-[var(--c-raised)] p-2 text-[12px] text-[var(--c-text)] outline-none ring-1 ring-inset ring-[var(--c-line)] focus:ring-[var(--c-accent)]"
                placeholder="例：現在適合加碼嗎？這個三角收斂往哪邊突破機率高？"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />
              <label className="mt-1 flex cursor-pointer items-center gap-1.5 text-[11.5px] text-[var(--c-muted)]">
                <input type="checkbox" className="accent-[var(--c-accent)]" checked={includeDrawings} onChange={(e) => setIncludeDrawings(e.target.checked)} />
                一併參考我在範圍內畫的線
              </label>
              <label className="mt-1 flex cursor-pointer items-center gap-1.5 text-[11.5px] text-[var(--c-muted)]">
                <input type="checkbox" className="accent-[var(--c-accent)]" checked={settings.drawMarks} onChange={(e) => update({ drawMarks: e.target.checked })} data-testid="ai-draw-marks" />
                讓 AI 在圖上標記支撐、壓力、趨勢線與重點
              </label>
            </div>

            {status === "online" ? <div className="border-t border-[var(--c-line)] pt-3">{modelControls}</div> : null}

            {run?.state === "running" ? (
              <button
                className="h-9 w-full cursor-pointer rounded-md text-[13px] font-semibold text-[var(--c-text)] ring-1 ring-inset ring-[var(--c-line)] hover:bg-[var(--c-hover)]"
                onClick={() => abort.current?.abort()}
              >
                停止
              </button>
            ) : (
              <button
                className="h-9 w-full cursor-pointer rounded-md bg-[var(--c-accent)] text-[13px] font-semibold text-[var(--c-accent-text)] hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
                disabled={status !== "online" || !provider?.available || !indices}
                onClick={() => void start()}
                data-testid="ai-run"
              >
                分析 {symbol}
              </button>
            )}

            {run ? (
              <div className="rounded-md p-3 ring-1 ring-inset ring-[var(--c-line)]" data-testid="ai-result">
                <div className="mb-2 flex items-center gap-2 border-b border-[var(--c-line)] pb-2">
                  <span className="min-w-0 flex-1 truncate text-[10.5px] text-[var(--c-muted)]" title={run.meta}>
                    {run.meta}
                  </span>
                  {run.text ? (
                    <button className={`${panelIconBtn()} !h-6 !w-auto px-2 text-[11px]`} onClick={() => void navigator.clipboard?.writeText(stripMarks(run.text))} title="複製分析內容">
                      複製
                    </button>
                  ) : null}
                </div>
                {run.text ? <Markdown text={stripMarks(run.text)} /> : null}
                {run.state === "running" ? (
                  <p className="mt-2 animate-pulse text-[11.5px] text-[var(--c-muted)]">
                    {run.thinking ? "AI 思考中…" : run.text.includes("```chart-marks") ? "在圖上標記中…" : run.text ? "撰寫中…" : "傳送資料給 AI…"}
                  </p>
                ) : null}
                {run.marks && run.placed != null ? (
                  run.placed > 0 && layer?.groupOf(AI_GROUP) ? (
                    <div className="mt-2 flex items-center gap-2 rounded-md bg-[var(--c-accent-soft)] px-2 py-1.5 text-[11.5px] text-[var(--c-text)]" data-testid="ai-marks-placed">
                      <span className="min-w-0 flex-1">已在圖上標記 {run.placed} 處（圖層「AI 標記」，可隱藏或刪除）</span>
                      <button className="shrink-0 cursor-pointer text-[var(--c-accent)] hover:underline" onClick={() => layer.removeGroup(AI_GROUP)} data-testid="ai-marks-clear">
                        清除標記
                      </button>
                    </div>
                  ) : run.placed === 0 ? (
                    <p className="mt-2 text-[11px] text-[var(--c-muted)]">這次 AI 沒有給出可畫在圖上的標記。</p>
                  ) : null
                ) : null}
                {run.error ? <p className="mt-2 text-[11.5px] text-[var(--c-danger)]">{run.error}</p> : null}
                {run.state === "done" && !run.error ? (
                  <p className="mt-2 text-[10.5px] text-[var(--c-muted)]">
                    完成{run.ms ? ` · ${(run.ms / 1000).toFixed(1)} 秒` : ""}
                    {run.costUsd ? ` · 約 US$${run.costUsd.toFixed(3)}（計入你的 AI 用量）` : ""}
                  </p>
                ) : null}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
