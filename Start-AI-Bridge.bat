@echo off
rem Dev starter for the repo: runs the bridge in this window.
cd /d "%~dp0ai-bridge"
where node >nul 2>nul
if errorlevel 1 (
  echo [!] Node.js is not installed: https://nodejs.org/
  pause
  exit /b 1
)
if not exist "node_modules\@anthropic-ai\claude-agent-sdk" (
  echo First run: installing Claude Agent SDK ...
  call npm install --no-audit --no-fund
)
node server.mjs
pause
