Smart DCA 本機 AI 橋接
======================

讓 Smart DCA 圖表的「AI 分析」使用這台電腦上的 AI：
  - Claude Code（需已安裝並登入 claude）
  - Codex CLI（需已安裝並登入 codex，實驗性）
  - Ollama（需執行 ollama 並下載模型）

安裝（Windows）
  1. 先安裝 Node.js LTS：https://nodejs.org/
  2. 雙擊 Install-AI-Bridge.bat
     - 檔案會放到 %LOCALAPPDATA%\SmartDCA\ai-bridge
     - 會註冊 smartdca-ai:// 連結，讓 Smart DCA 可以一鍵啟動橋接
  3. 回到 Smart DCA → 圖表 → 右側 ✦ AI 分析 →「啟動本機 AI」

macOS / Linux
  在這個資料夾執行：
    npm install --omit=dev
    node server.mjs
  然後保持視窗開著。

安全
  - 只監聽 127.0.0.1，只接受 dca.hellokai07.com 與 localhost 網頁的請求
  - AI 不能使用任何工具（不能讀寫檔案或執行指令），只回答文字
  - 分析會計入你自己的 Claude / OpenAI 用量

移除
  刪除 %LOCALAPPDATA%\SmartDCA\ai-bridge，並刪除登錄機碼
  HKEY_CURRENT_USER\Software\Classes\smartdca-ai
