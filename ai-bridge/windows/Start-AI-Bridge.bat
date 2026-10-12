@echo off
chcp 65001 >nul
rem Starts the Smart DCA AI bridge from this folder (also opened by smartdca-ai:// links).
cd /d "%~dp0"
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 (
  echo Smart DCA AI 橋接已經在執行中。
  timeout /t 2 >nul
  exit /b 0
)
if not exist "node_modules\@anthropic-ai\claude-agent-sdk" (
  echo 第一次使用：安裝 Claude Agent SDK ...
  call npm install --omit=dev --no-audit --no-fund
)
start "Smart DCA AI 橋接（關掉這個視窗就會停止）" /min cmd /k node server.mjs
