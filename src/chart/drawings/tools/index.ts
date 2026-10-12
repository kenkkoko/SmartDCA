// All drawing tools, grouped like TradingView's left toolbar.
import type { Tool, ToolGroup } from "../tool";
import { CHANNEL_TOOLS } from "./channels";
import { FIB_TOOLS } from "./fibonacci";
import { LINE_TOOLS } from "./lines";
import { MEASURE_TOOLS } from "./measure";
import { PATTERN_TOOLS } from "./patterns";
import { BRUSH_TOOLS, SHAPE_TOOLS } from "./shapes";
import { TEXT_TOOLS } from "./text";
import { VOLUME_TOOLS } from "./volume";

export const TOOL_GROUPS: { group: ToolGroup; label: string; tools: Tool[] }[] = [
  { group: "lines", label: "線條", tools: LINE_TOOLS },
  { group: "channels", label: "通道", tools: CHANNEL_TOOLS },
  { group: "fibonacci", label: "斐波那契", tools: FIB_TOOLS },
  { group: "patterns", label: "型態", tools: PATTERN_TOOLS },
  { group: "measure", label: "測量與部位", tools: MEASURE_TOOLS },
  { group: "volume", label: "成交量", tools: VOLUME_TOOLS },
  { group: "shapes", label: "形狀", tools: SHAPE_TOOLS },
  { group: "brush", label: "畫筆", tools: BRUSH_TOOLS },
  { group: "text", label: "文字與標記", tools: TEXT_TOOLS },
];

const BY_TYPE = new Map<string, Tool>(TOOL_GROUPS.flatMap((g) => g.tools).map((t) => [t.type, t]));

export function getTool(type: string): Tool | undefined {
  return BY_TYPE.get(type);
}

export const ALL_TOOLS: Tool[] = [...BY_TYPE.values()];
