// Small 20px stroke icons for the drawing toolbar (hand-made, no icon pack).
import type { ReactNode } from "react";

const svg = (children: ReactNode) => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
    {children}
  </svg>
);
const dot = (x: number, y: number) => <circle cx={x} cy={y} r="1.8" fill="currentColor" stroke="none" />;

const ICONS: Record<string, ReactNode> = {
  cursor: svg(<path d="M5 3l10 7-4.5 1 2.5 5-2 1-2.5-5L5 15z" />),
  trend_line: svg(<>{dot(4, 16)}{dot(16, 4)}<path d="M5 15L15 5" /></>),
  ray: svg(<>{dot(4, 15)}<path d="M5 14.3L18 5" /></>),
  extended_line: svg(<path d="M1 17L19 3" />),
  horizontal_line: svg(<>{dot(10, 10)}<path d="M1 10h18" /></>),
  horizontal_ray: svg(<>{dot(5, 10)}<path d="M5 10h14" /></>),
  vertical_line: svg(<>{dot(10, 10)}<path d="M10 1v18" /></>),
  cross_line: svg(<><path d="M1 10h18M10 1v18" /></>),
  info_line: svg(<>{dot(4, 16)}{dot(16, 6)}<path d="M5 15L15 7" /><rect x="10" y="12" width="8" height="5" rx="1" /></>),
  trend_angle: svg(<>{dot(4, 15)}<path d="M4 15L16 5M4 15h13M11 15a7 7 0 0 0-1.5-4.5" /></>),
  arrow: svg(<><path d="M4 16L16 4M10 4h6v6" /></>),
  parallel_channel: svg(<><path d="M2 12L13 4M7 17L18 9" />{dot(2, 12)}{dot(13, 4)}{dot(18, 9)}</>),
  regression_trend: svg(<><path d="M2 13L18 5M2 9L18 1.5M2 17L18 9" strokeDasharray="0" /></>),
  flat_top_bottom: svg(<><path d="M2 5h16M2 16L18 10" /></>),
  disjoint_channel: svg(<><path d="M2 6L18 3M2 12L18 17" /></>),
  pitchfork: svg(<><path d="M3 10L9 10M9 4L18 4M9 10L18 10M9 16L18 16M9 4v12" /></>),
  fib_retracement: svg(<><path d="M2 3h16M2 7.5h16M2 11h16M2 17h16" /><path d="M4 17L16 3" strokeDasharray="2 2" /></>),
  fib_extension: svg(<><path d="M2 4h16M2 9h16M2 14h16" /><path d="M3 17L8 11L11 15" /></>),
  fib_channel: svg(<><path d="M2 8L18 2M2 13L18 7M2 18L18 12" /></>),
  xabcd_pattern: svg(<path d="M2 15L6 5L10 12L14 3L18 14" />),
  abcd_pattern: svg(<path d="M3 15L8 5L12 11L17 3" />),
  head_shoulders: svg(<path d="M1 15L4 9L7 13L10 4L13 13L16 9L19 15" />),
  elliott_impulse: svg(<path d="M2 16L6 9L8 12L13 3L15 7L18 2" />),
  elliott_correction: svg(<path d="M3 4L9 14L12 9L17 17" />),
  long_position: svg(<><rect x="3" y="3" width="14" height="8" fill="currentColor" fillOpacity=".25" /><rect x="3" y="11" width="14" height="5" /></>),
  short_position: svg(<><rect x="3" y="4" width="14" height="5" /><rect x="3" y="9" width="14" height="8" fill="currentColor" fillOpacity=".25" /></>),
  price_range: svg(<><path d="M3 3h14M3 17h14M10 4v12M7 7l3-3 3 3" /></>),
  date_range: svg(<><path d="M3 3v14M17 3v14M4 10h12M13 7l3 3-3 3" /></>),
  date_price_range: svg(<><rect x="3" y="3" width="14" height="14" /><path d="M6 14L14 6" /></>),
  volume_profile: svg(<><path d="M3 3v14" /><path d="M3 5h6M3 8h11M3 11h14M3 14h8" strokeWidth="2" /></>),
  anchored_vwap: svg(<>{dot(3, 15)}<path d="M3 15C7 11 9 13 12 9S16 6 18 6" /></>),
  rectangle: svg(<rect x="3" y="5" width="14" height="10" />),
  ellipse: svg(<ellipse cx="10" cy="10" rx="8" ry="5.5" />),
  circle: svg(<circle cx="10" cy="10" r="7" />),
  triangle: svg(<path d="M10 3L18 16H2z" />),
  freehand: svg(<path d="M2 14c3-6 5 2 8-3s4-6 8-2" />),
  highlighter: svg(<path d="M2 14c3-6 5 2 8-3s4-6 8-2" strokeWidth="4" strokeOpacity=".5" />),
  polyline: svg(<><path d="M2 15L7 6L12 12L18 4" />{dot(7, 6)}{dot(12, 12)}</>),
  curved_line: svg(<path d="M2 16Q10 -2 18 16" />),
  text: svg(<path d="M4 4h12M10 4v13M7 17h6" />),
  callout: svg(<><rect x="5" y="2" width="13" height="8" rx="1.5" /><path d="M8 10L3 17" /></>),
  price_label: svg(<><path d="M2 10l4-4h12v8H6z" /></>),
  arrow_mark_up: svg(<path d="M10 3l6 7h-4v7H8v-7H4z" />),
  arrow_mark_down: svg(<path d="M10 17l6-7h-4V3H8v7H4z" />),
  magnet: svg(<path d="M5 3v7a5 5 0 0 0 10 0V3h-3v7a2 2 0 0 1-4 0V3zM5 6h3M12 6h3" />),
  keep: svg(<><path d="M6 14L14 4" /><path d="M4 10l-2 2 6 6 2-2" /><path d="M13 16h5M15.5 13.5v5" /></>),
  eye: svg(<><path d="M1.5 10S5 4 10 4s8.5 6 8.5 6-3.5 6-8.5 6-8.5-6-8.5-6z" /><circle cx="10" cy="10" r="2.5" /></>),
  eyeOff: svg(<><path d="M1.5 10S5 4 10 4s8.5 6 8.5 6-3.5 6-8.5 6-8.5-6-8.5-6z" /><path d="M3 17L17 3" /></>),
  trash: svg(<path d="M3 5h14M8 5V3h4v2M5 5l1 12h8l1-12" />),
  undo: svg(<path d="M7 4L3 8l4 4M3 8h9a5 5 0 0 1 0 10H8" />),
  redo: svg(<path d="M13 4l4 4-4 4M17 8H8a5 5 0 0 0 0 10h4" />),
  lock: svg(<><rect x="4" y="9" width="12" height="9" rx="1.5" /><path d="M7 9V6a3 3 0 0 1 6 0v3" /></>),
  unlock: svg(<><rect x="4" y="9" width="12" height="9" rx="1.5" /><path d="M7 9V6a3 3 0 0 1 6 0" /></>),
  clone: svg(<><rect x="6" y="6" width="11" height="11" rx="1.5" /><path d="M3 13V4a1 1 0 0 1 1-1h9" /></>),
  front: svg(<><rect x="7" y="7" width="10" height="10" fill="currentColor" fillOpacity=".35" /><path d="M3 13V3h10" /></>),
  back: svg(<><rect x="3" y="3" width="10" height="10" fill="currentColor" fillOpacity=".35" /><path d="M7 17h10V7" /></>),
  settings: svg(<><circle cx="10" cy="10" r="2.5" /><path d="M10 2v3M10 15v3M2 10h3M15 10h3M4.3 4.3l2.1 2.1M13.6 13.6l2.1 2.1M4.3 15.7l2.1-2.1M13.6 6.4l2.1-2.1" /></>),
  textEdit: svg(<><path d="M3 5h9M7.5 5v11" /><path d="M12 16l5-5-2-2-5 5v2z" /></>),
  extendLeft: svg(<><path d="M18 10H2M6 6l-4 4 4 4" /></>),
  extendRight: svg(<><path d="M2 10h16M14 6l4 4-4 4" /></>),
  middle: svg(<><path d="M2 5h16M2 15h16" /><path d="M2 10h16" strokeDasharray="2 2" /></>),
  stats: svg(<><path d="M3 17V9M8 17V4M13 17v-6M18 17V7" /></>),
  layers: svg(<><path d="M10 2.5l8 4-8 4-8-4z" /><path d="M2 10.5l8 4 8-4" /><path d="M2 14l8 4 8-4" /></>),
  grip: svg(<><circle cx="7.5" cy="5" r="1" fill="currentColor" /><circle cx="12.5" cy="5" r="1" fill="currentColor" /><circle cx="7.5" cy="10" r="1" fill="currentColor" /><circle cx="12.5" cy="10" r="1" fill="currentColor" /><circle cx="7.5" cy="15" r="1" fill="currentColor" /><circle cx="12.5" cy="15" r="1" fill="currentColor" /></>),
  more: svg(<><circle cx="4" cy="10" r="1" fill="currentColor" /><circle cx="10" cy="10" r="1" fill="currentColor" /><circle cx="16" cy="10" r="1" fill="currentColor" /></>),
};

export function Icon({ name }: { name: string }) {
  return <>{ICONS[name] ?? ICONS.trend_line}</>;
}
