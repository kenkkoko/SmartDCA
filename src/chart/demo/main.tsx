// Playground (stages 1 + 3: engine and drawing tools) for the new chart engine: /src/chart/demo/ (npm run dev)
// ?symbol=BTCUSDT (Binance daily) or ?data=synthetic for offline testing.
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "../../tailwind.css";
import type { Chart, CrosshairInfo } from "../engine/chart";
import type { Bar } from "../engine/types";
import { DrawingChart } from "../react/DrawingChart";

function synthetic(n = 1200): Bar[] {
  const bars: Bar[] = [];
  let price = 200;
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const start = Date.UTC(2021, 0, 1) / 1000;
  for (let i = 0; i < n; i++) {
    const drift = 0.0025 + 0.03 * Math.sin(i / 90);
    const open = price;
    const close = Math.max(1, open * (1 + drift * 0.3 + (rand() - 0.5) * 0.06));
    const high = Math.max(open, close) * (1 + rand() * 0.02);
    const low = Math.min(open, close) * (1 - rand() * 0.02);
    bars.push({ time: start + i * 86400, open, high, low, close, volume: 1000 + rand() * 9000 });
    price = close;
  }
  return bars;
}

async function load(): Promise<Bar[]> {
  const params = new URLSearchParams(location.search);
  if (params.get("data") === "synthetic") return synthetic();
  const symbol = params.get("symbol") ?? "BTCUSDT";
  try {
    const res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=1d&limit=1000`);
    const rows = (await res.json()) as string[][];
    return rows.map((r) => ({
      time: Number(r[0]) / 1000,
      open: Number(r[1]),
      high: Number(r[2]),
      low: Number(r[3]),
      close: Number(r[4]),
      volume: Number(r[5]),
    }));
  } catch {
    return synthetic();
  }
}

function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  values.forEach((v, i) => out.push(i === 0 ? v : v * k + out[i - 1]! * (1 - k)));
  return out;
}

const symbol = new URLSearchParams(location.search).get("data") === "synthetic" ? "SYNTH" : (new URLSearchParams(location.search).get("symbol") ?? "BTCUSDT");

function App() {
  const [chart, setChart] = useState<Chart | null>(null);
  const [bars, setBars] = useState<Bar[]>([]);
  const [hover, setHover] = useState<CrosshairInfo | null>(null);
  const [, force] = useState(0);

  useEffect(() => {
    void load().then(setBars);
  }, []);

  useEffect(() => {
    if (!chart || bars.length === 0) return;
    const main = chart.mainPane;
    const candles = chart.addCandles(main);
    candles.setData(bars);
    const t = chart.theme;
    const volume = chart.addHistogram(main, { scale: "overlay", autoscale: true });
    volume.setData(bars.map((b) => ({ time: b.time, value: b.volume ?? 0, color: b.close >= b.open ? `${t.up}55` : `${t.down}55` })));
    const closes = bars.map((b) => b.close);
    const fast = ema(closes, 20);
    const slow = ema(closes, 55);
    chart.addLine(main, { color: "#f5b942", lineWidth: 1.5 }).setData(bars.map((b, i) => ({ time: b.time, value: fast[i]! })));
    chart.addLine(main, { color: "#8b5cf6", lineWidth: 1.5 }).setData(bars.map((b, i) => ({ time: b.time, value: slow[i]! })));
    const unsubscribe = [chart.onCrosshairMove(setHover), chart.onViewChange(() => force((n) => n + 1))];
    return () => unsubscribe.forEach((u) => u());
  }, [chart, bars]);

  const price = chart?.mainPane.price;
  const bar = hover?.bar ?? bars[bars.length - 1];
  const prev = bar ? bars[bars.indexOf(bar) - 1] : undefined;
  const change = bar && prev ? (bar.close / prev.close - 1) * 100 : 0;
  const up = change >= 0;

  const button = (active: boolean) =>
    `px-2.5 py-1 rounded text-xs ${active ? "bg-slate-200 text-slate-900" : "text-slate-300 hover:bg-slate-700"}`;

  return (
    <div className="flex h-screen flex-col bg-[#0b0e14] text-slate-200">
      <div className="flex items-center gap-2 border-b border-white/10 px-3 py-2 text-sm">
        <span className="font-semibold">{symbol} · 新圖表引擎</span>
        {bar ? (
          <span className={`ml-3 font-mono text-xs ${up ? "text-[#00d68f]" : "text-[#ff5b6e]"}`}>
            開 {bar.open.toFixed(2)} 高 {bar.high.toFixed(2)} 低 {bar.low.toFixed(2)} 收 {bar.close.toFixed(2)} ({up ? "+" : ""}
            {change.toFixed(2)}%)
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-1">
          <button className={button(price?.mode === "log")} onClick={() => chart?.setPriceScaleMode("log")}>
            對數
          </button>
          <button className={button(price?.mode === "linear")} onClick={() => chart?.setPriceScaleMode("linear")}>
            線性
          </button>
          <button className={button(!!price?.autoScale)} onClick={() => chart?.setAutoScale(true)}>
            自動
          </button>
        </div>
      </div>
      <DrawingChart className="min-h-0 flex-1" storageKey={`demo.${symbol}`} onReady={(c) => setChart(c)} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
