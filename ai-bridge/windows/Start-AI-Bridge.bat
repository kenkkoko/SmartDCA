@echo off
rem Smart DCA AI bridge starter. ASCII only on purpose: cmd misreads UTF-8
rem batch files on some Windows setups. Also opened by smartdca-ai:// links.
setlocal
title Smart DCA AI Bridge

rem -- already running? ------------------------------------------------------
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 (
  echo Smart DCA AI bridge is already running.
  ping -n 3 127.0.0.1 >nul
  exit /b 0
)

rem -- find server.mjs: installed folder, the zip folder, or the repo ---------
set "DIR="
if exist "%~dp0server.mjs" set "DIR=%~dp0"
if not defined DIR if exist "%~dp0ai-bridge\server.mjs" set "DIR=%~dp0ai-bridge\"
if not defined DIR if exist "%~dp0..\server.mjs" set "DIR=%~dp0..\"
if not defined DIR if exist "%LOCALAPPDATA%\SmartDCA\ai-bridge\server.mjs" set "DIR=%LOCALAPPDATA%\SmartDCA\ai-bridge\"
if not defined DIR (
  echo [!] server.mjs not found next to this file.
  echo     Unzip smartdca-ai-bridge.zip first, then run Install-AI-Bridge.bat.
  pause
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo [!] Node.js is not installed. Get the LTS version from https://nodejs.org/
  start "" https://nodejs.org/
  pause
  exit /b 1
)

cd /d "%DIR%"
if not exist "node_modules\@anthropic-ai\claude-agent-sdk" (
  echo First run: installing Claude Agent SDK ...
  call npm install --omit=dev --no-audit --no-fund
  if errorlevel 1 (
    echo [!] npm install failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)

rem a minimised window that keeps the bridge running; it pauses on errors
start "Smart DCA AI Bridge - close this window to stop" /min cmd /c "node server.mjs || pause"
endlocal
