@echo off
chcp 65001 >nul
title Smart DCA AI Bridge
cd /d "%~dp0"
if not exist "ai-bridge\node_modules\@anthropic-ai\claude-agent-sdk" (
  echo 第一次使用：安裝 Claude Agent SDK ...
  call npm install --prefix ai-bridge --no-audit --no-fund
)
node ai-bridge\server.mjs
pause
