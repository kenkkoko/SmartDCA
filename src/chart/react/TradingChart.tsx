// React wrapper: mounts a Chart into a div and hands it to the parent once.
import { useEffect, useRef, type CSSProperties } from "react";
import { Chart, type ChartOptions } from "../engine/chart";

export type TradingChartProps = {
  /** Called once with the live chart; return a cleanup to run on unmount. */
  onReady: (chart: Chart) => void | (() => void);
  options?: Partial<ChartOptions>;
  className?: string;
  style?: CSSProperties;
};

export function TradingChart({ onReady, options, className, style }: TradingChartProps) {
  const ref = useRef<HTMLDivElement>(null);
  // keep the latest callback without re-creating the chart
  const ready = useRef(onReady);
  ready.current = onReady;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const chart = new Chart(el, options);
    const cleanup = ready.current(chart);
    return () => {
      if (typeof cleanup === "function") cleanup();
      chart.destroy();
    };
    // options are read once at mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return <div ref={ref} className={className} style={{ position: "relative", ...style }} />;
}
