// Smart DCA · 本機 AI 橋接
//
// 網頁不能直接啟動你電腦上的程式，所以由這個小服務在 127.0.0.1 上替網頁
// 呼叫本機的 AI：
//   - Claude Code（透過 Claude Agent SDK，使用你已登入的 Claude 帳號）
//   - Codex CLI（`codex exec`，使用你已登入的 OpenAI 帳號）
//   - Ollama（http://127.0.0.1:11434 的本機模型）
//
// 只聽 127.0.0.1，只接受 Smart DCA 網站與 localhost 的請求，不讓 AI 使用任何
// 工具（不能讀寫檔案、不能執行指令），只回答文字。
//
// 啟動：在專案資料夾執行 `npm run ai-bridge`（或雙擊 Start-AI-Bridge.bat）。

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const VERSION = "1.0.0";
const PORT = Number(process.env.SMARTDCA_AI_PORT || 8765);
const ALLOWED = [
  /^https:\/\/dca\.hellokai07\.com$/,
  // only this project's own Pages site and its preview builds (<hash>.smartdca.pages.dev);
  // a pattern like smartdca*.pages.dev would let anyone's Pages project named smartdca-xxx in
  /^https:\/\/([a-z0-9-]+\.)?smartdca\.pages\.dev$/,
  /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/,
  ...(process.env.SMARTDCA_AI_ORIGINS || "")
    .split(",")
    .filter(Boolean)
    .map((o) => new RegExp(`^${o.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`)),
];

const EFFORTS = ["low", "medium", "high", "xhigh", "max"];
const IS_WIN = process.platform === "win32";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** Run a short command; resolves { ok, out } without throwing. */
function run(cmd, args, { timeout = 8000, input } = {}) {
  return new Promise((resolve) => {
    let out = "";
    let done = false;
    let child;
    try {
      child = IS_WIN ? spawn([cmd, ...args].join(" "), { shell: true, windowsHide: true }) : spawn(cmd, args, { windowsHide: true });
    } catch {
      return resolve({ ok: false, out: "" });
    }
    const finish = (ok) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ ok, out: out.trim() });
    };
    const timer = setTimeout(() => {
      child.kill();
      finish(false);
    }, timeout);
    child.stdout?.on("data", (d) => (out += d));
    child.stderr?.on("data", (d) => (out += d));
    child.on("error", () => finish(false));
    child.on("close", (code) => finish(code === 0));
    if (input !== undefined) child.stdin?.end(input);
  });
}

async function loadClaudeSdk() {
  try {
    return await import("@anthropic-ai/claude-agent-sdk");
  } catch {
    return null;
  }
}

/** A prompt stream that never sends a message: lets us ask the CLI for its models. */
function idlePrompt(signal) {
  return {
    async *[Symbol.asyncIterator]() {
      await new Promise((resolve) => signal.addEventListener("abort", resolve, { once: true }));
    },
  };
}

function withTimeout(promise, ms, label) {
  return Promise.race([promise, new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} 逾時`)), ms))]);
}

// ---------------------------------------------------------------------------
// providers
// ---------------------------------------------------------------------------

const CLAUDE_FALLBACK = [
  { id: "opus", name: "Claude Opus（最新）", efforts: EFFORTS },
  { id: "sonnet", name: "Claude Sonnet（最新）", efforts: EFFORTS },
  { id: "haiku", name: "Claude Haiku（最新）", efforts: [] },
];

async function detectClaude() {
  const sdk = await loadClaudeSdk();
  const cli = await run("claude", ["--version"]);
  const base = { id: "claude", name: "Claude Code", cliVersion: cli.ok ? cli.out.split("\n")[0] : null };
  if (!sdk)
    return {
      ...base,
      available: false,
      reason: cli.ok
        ? "找到 Claude Code，但橋接還沒安裝 Claude Agent SDK：在專案資料夾執行 npm run ai-bridge:install"
        : "沒有找到 Claude Code（https://claude.com/claude-code）",
      models: [],
    };
  const ac = new AbortController();
  try {
    const q = sdk.query({ prompt: idlePrompt(ac.signal), options: { tools: [], settingSources: [], persistSession: false, abortController: ac } });
    const list = await withTimeout(q.supportedModels(), 20000, "讀取 Claude 模型");
    ac.abort();
    try {
      q.close();
    } catch {
      // already closed
    }
    const models = list.map((m) => ({
      id: m.value,
      name: m.displayName || m.value,
      description: m.description || "",
      efforts: m.supportsEffort ? m.supportedEffortLevels || EFFORTS : [],
    }));
    return { ...base, available: models.length > 0, models: models.length ? models : CLAUDE_FALLBACK };
  } catch (e) {
    ac.abort();
    return {
      ...base,
      available: false,
      reason: `Claude Code 無法啟動：${e?.message || e}。請先在終端機執行 claude 並登入。`,
      models: CLAUDE_FALLBACK,
    };
  }
}

async function readCodexDefaultModel() {
  try {
    const text = await readFile(path.join(os.homedir(), ".codex", "config.toml"), "utf8");
    return /^\s*model\s*=\s*"([^"]+)"/m.exec(text)?.[1] ?? null;
  } catch {
    return null;
  }
}

async function detectCodex() {
  const cli = await run("codex", ["--version"]);
  if (!cli.ok) return { id: "codex", name: "Codex CLI", available: false, reason: "沒有找到 Codex CLI", models: [] };
  const configured = await readCodexDefaultModel();
  return {
    id: "codex",
    name: "Codex CLI",
    cliVersion: cli.out.split("\n")[0],
    available: true,
    experimental: true,
    models: [
      { id: "", name: configured ? `預設（${configured}）` : "Codex 預設模型", efforts: ["low", "medium", "high"] },
      ...(configured ? [{ id: configured, name: configured, efforts: ["low", "medium", "high"] }] : []),
    ],
    customModel: true,
  };
}

const OLLAMA = process.env.OLLAMA_HOST || "http://127.0.0.1:11434";

async function detectOllama() {
  try {
    const res = await withTimeout(fetch(`${OLLAMA}/api/tags`), 3000, "Ollama");
    const json = await res.json();
    const models = (json.models || []).map((m) => ({ id: m.name, name: m.name, efforts: [] }));
    return { id: "ollama", name: "Ollama（本機模型）", available: models.length > 0, reason: models.length ? undefined : "Ollama 沒有已下載的模型", models };
  } catch {
    return { id: "ollama", name: "Ollama（本機模型）", available: false, reason: "沒有偵測到 Ollama（127.0.0.1:11434）", models: [] };
  }
}

let cache = null;
async function providers(refresh) {
  if (!cache || refresh || Date.now() - cache.at > 5 * 60_000) {
    const list = await Promise.all([detectClaude(), detectCodex(), detectOllama()]);
    if (process.env.SMARTDCA_AI_TEST)
      list.unshift({ id: "echo", name: "測試用 AI", available: true, models: [{ id: "echo-1", name: "Echo 1", efforts: ["low", "medium", "high"] }] });
    cache = { at: Date.now(), list };
  }
  return cache.list;
}

// ---------------------------------------------------------------------------
// analysis (streams text chunks through `emit`)
// ---------------------------------------------------------------------------

async function analyzeClaude({ model, effort, system, prompt }, emit, signal) {
  const sdk = await loadClaudeSdk();
  if (!sdk) throw new Error("橋接沒有安裝 Claude Agent SDK（npm run ai-bridge:install）");
  const ac = new AbortController();
  signal.addEventListener("abort", () => ac.abort(), { once: true });
  const q = sdk.query({
    prompt,
    options: {
      ...(model ? { model } : {}),
      ...(effort && EFFORTS.includes(effort) ? { effort } : {}),
      systemPrompt: system,
      tools: [], // 只回答文字，不使用任何工具
      maxTurns: 1,
      settingSources: [],
      persistSession: false,
      includePartialMessages: true,
      abortController: ac,
    },
  });
  let streamed = false;
  for await (const m of q) {
    if (m.type === "stream_event" && !m.parent_tool_use_id) {
      const ev = m.event;
      if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta") {
        streamed = true;
        emit({ type: "text", text: ev.delta.text });
      } else if (ev.type === "content_block_delta" && ev.delta?.type === "thinking_delta") {
        emit({ type: "thinking" });
      }
    } else if (m.type === "assistant" && !streamed) {
      for (const block of m.message?.content || []) if (block.type === "text") emit({ type: "text", text: block.text });
    } else if (m.type === "result") {
      if (m.subtype !== "success") throw new Error(m.errors?.join("; ") || m.subtype);
      emit({ type: "usage", model: m.modelUsage ? Object.keys(m.modelUsage)[0] : model, costUsd: m.total_cost_usd });
    }
  }
}

function analyzeCodex({ model, effort, system, prompt }, emit, signal) {
  return new Promise((resolve, reject) => {
    const args = ["exec", "--skip-git-repo-check"];
    if (model) args.push("-m", model);
    if (effort) args.push("-c", `model_reasoning_effort=${effort}`);
    args.push("-"); // 從 stdin 讀題目（避免命令列長度限制）
    // args are validated (model / effort are plain names), so joining them for the Windows shell is safe
    const child = IS_WIN
      ? spawn(["codex", ...args].join(" "), { shell: true, windowsHide: true, cwd: os.tmpdir() })
      : spawn("codex", args, { windowsHide: true, cwd: os.tmpdir() });
    signal.addEventListener("abort", () => child.kill(), { once: true });
    let err = "";
    child.stdout.on("data", (d) => emit({ type: "text", text: d.toString() }));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", reject);
    child.on("close", (code) => (code === 0 || signal.aborted ? resolve() : reject(new Error(err.trim().slice(-400) || `codex 結束碼 ${code}`))));
    child.stdin.end(`${system}\n\n${prompt}`);
  });
}

// Ollama reads only num_ctx tokens (default 2048–4096) and silently drops the
// start of a longer conversation, i.e. the system prompt and most of the chart
// data. Size the window to the prompt (~1 token per 2 characters for this
// mix of Chinese and numbers, plus room for the answer).
export function ollamaContext(system, prompt) {
  const need = Math.ceil((system.length + prompt.length) / 2) + 4096;
  return Math.min(32768, Math.max(8192, Math.ceil(need / 1024) * 1024));
}

async function analyzeOllama({ model, system, prompt }, emit, signal) {
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: "POST",
    signal,
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      stream: true,
      messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
      options: { num_ctx: ollamaContext(system, prompt), temperature: 0.2 },
    }),
  });
  if (!res.ok || !res.body) throw new Error(`Ollama 回應 ${res.status}`);
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of res.body) {
    buf += decoder.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.message?.content) emit({ type: "text", text: msg.message.content });
      if (msg.error) throw new Error(msg.error);
    }
  }
}

const RUNNERS = { claude: analyzeClaude, codex: analyzeCodex, ollama: analyzeOllama };

// SMARTDCA_AI_TEST=1: a fake provider that echoes a canned analysis (for testing the UI)
if (process.env.SMARTDCA_AI_TEST) {
  RUNNERS.echo = async ({ model, effort, prompt }, emit, signal) => {
    const text = `## 測試回覆\n\n模型 **${model}**，努力程度 ${effort}。收到 ${prompt.length} 字的資料。\n\n- 趨勢：測試\n- 支撐：100\n- 壓力：200\n`;
    for (const part of text.match(/.{1,12}/gs) || []) {
      if (signal.aborted) return;
      emit({ type: "text", text: part });
      await new Promise((r) => setTimeout(r, 20));
    }
  };
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

// Turn the providers' own errors into what the user has to do. Each provider
// fails on its own: an expired Claude login says nothing about Ollama.
export function friendlyError(provider, msg) {
  if (provider === "claude" && /OAuth|authenticat|log ?in|401|invalid api key|credential/i.test(msg))
    return `Claude Code 的登入過期了（${msg}）。請在電腦上開終端機執行 claude，輸入 /login 重新登入，完成後回到這裡再按一次分析（不用重開橋接）。`;
  if (provider === "claude" && /rate limit|usage limit|429|quota/i.test(msg))
    return `Claude 的用量額度暫時用完了（${msg}）。請稍後再試，或改用其他模型。`;
  if (provider === "codex" && /login|auth|401/i.test(msg)) return `Codex 的登入過期了（${msg}）。請在終端機執行 codex login 重新登入。`;
  if (provider === "ollama" && /ECONNREFUSED|fetch failed/i.test(msg)) return "連不到 Ollama：請確認 Ollama 有在執行（工作列的羊駝圖示）。";
  return msg;
}

function cors(req, res) {
  const origin = req.headers.origin;
  if (!origin) return true; // curl / same machine tools
  if (!ALLOWED.some((re) => re.test(origin))) return false;
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Vary", "Origin");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "content-type");
  // Chrome's Private Network Access: a public site calling 127.0.0.1
  res.setHeader("Access-Control-Allow-Private-Network", "true");
  return true;
}

function json(res, status, body) {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

async function readBody(req, limit = 2_000_000) {
  let size = 0;
  const chunks = [];
  for await (const c of req) {
    size += c.length;
    if (size > limit) throw new Error("請求太大");
    chunks.push(c);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

const server = createServer(async (req, res) => {
  if (!cors(req, res)) return json(res, 403, { error: "這個網站沒有被允許使用本機 AI 橋接" });
  if (req.method === "OPTIONS") return res.writeHead(204).end();
  const url = new URL(req.url || "/", "http://127.0.0.1");
  try {
    if (req.method === "GET" && url.pathname === "/health") return json(res, 200, { ok: true, name: "smartdca-ai-bridge", version: VERSION });
    if (req.method === "GET" && url.pathname === "/providers") return json(res, 200, { providers: await providers(url.searchParams.has("refresh")) });
    if (req.method === "POST" && url.pathname === "/analyze") {
      const body = await readBody(req);
      const runner = RUNNERS[body.provider];
      if (!runner) return json(res, 400, { error: `不認得的 AI：${body.provider}` });
      if (typeof body.prompt !== "string" || !body.prompt) return json(res, 400, { error: "缺少分析內容" });
      // model / effort reach a command line (codex on Windows runs through the shell): allow plain names only
      if (body.model && !/^[\w.:\/-]{1,80}$/.test(String(body.model))) return json(res, 400, { error: "模型名稱只能包含英數字與 . : / - _" });
      if (body.effort && !["minimal", ...EFFORTS].includes(String(body.effort))) return json(res, 400, { error: "不認得的努力程度" });
      res.writeHead(200, { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" });
      const ac = new AbortController();
      res.on("close", () => ac.abort());
      const emit = (ev) => {
        if (!res.writableEnded) res.write(`data: ${JSON.stringify(ev)}\n\n`);
      };
      const started = Date.now();
      console.log(`[analyze] ${body.provider} ${body.model || "(預設)"} effort=${body.effort || "-"} ${body.prompt.length} 字`);
      try {
        await runner({ model: body.model, effort: body.effort, system: String(body.system || ""), prompt: body.prompt }, emit, ac.signal);
        emit({ type: "done", ms: Date.now() - started });
      } catch (e) {
        if (!ac.signal.aborted) emit({ type: "error", message: friendlyError(body.provider, String(e?.message || e)) });
      }
      return res.end();
    }
    json(res, 404, { error: "not found" });
  } catch (e) {
    if (!res.headersSent) json(res, 500, { error: String(e?.message || e) });
    else res.end();
  }
});

server.on("error", (e) => {
  if (e.code === "EADDRINUSE") {
    console.log(`連接埠 ${PORT} 已被佔用：橋接可能已經在執行（看看工作列有沒有另一個橋接視窗），不用再開一次。`);
    process.exit(0);
  }
  console.error("橋接無法啟動：", e.message);
  process.exit(1);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`Smart DCA 本機 AI 橋接 v${VERSION}：http://127.0.0.1:${PORT}`);
  console.log("保持這個視窗開著，回到 Smart DCA 圖表的「AI 分析」就能使用。按 Ctrl+C 關閉。");
  providers(true).then((list) => {
    for (const p of list) console.log(`  ${p.available ? "✓" : "✗"} ${p.name}${p.available ? `：${p.models.length} 個模型` : `（${p.reason}）`}`);
  });
});
