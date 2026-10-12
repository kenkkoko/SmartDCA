# Smart DCA 本機 AI 橋接

讓 Smart DCA 圖表的「AI 分析」使用你電腦上的 AI。網頁本身不能啟動電腦裡的程式，所以由這個小服務在 `127.0.0.1:8765` 替它呼叫：

| AI | 需要 | 模型 / 努力程度 |
| --- | --- | --- |
| Claude Code | 已安裝並登入 `claude` | 由 Claude Agent SDK 讀取你帳號可用的模型；努力程度 低 / 中 / 高 / 極高 / 最大 |
| Codex CLI（實驗性） | 已安裝並登入 `codex` | `~/.codex/config.toml` 的預設模型或自行輸入；努力程度 低 / 中 / 高 |
| Ollama | `ollama serve` 且有下載模型 | 本機已下載的模型 |

## 使用

1. Windows：雙擊 `ai-bridge/windows/Start-AI-Bridge.bat`（第一次會自動安裝）。
   其他系統：`npm run ai-bridge:install` 一次，之後 `npm run ai-bridge`。
2. 視窗顯示偵測到的 AI 後保持開著。
3. 在圖表右側工具列打開「AI 分析」，齒輪裡選 AI、模型與努力程度。

## 安全

- 只監聽 `127.0.0.1`，只接受 `dca.hellokai07.com` 與 localhost 的網頁請求（`SMARTDCA_AI_ORIGINS` 可再加網址，逗號分隔）。
- AI 不能使用任何工具：不能讀寫檔案、不能執行指令，只回答文字。
- 分析會計入你自己的 Claude / OpenAI 用量。

環境變數：`SMARTDCA_AI_PORT`（預設 8765）、`OLLAMA_HOST`（預設 http://127.0.0.1:11434）。
