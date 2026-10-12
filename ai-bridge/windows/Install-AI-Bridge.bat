@echo off
rem Smart DCA AI bridge installer. ASCII only on purpose (see Start-AI-Bridge.bat).
setlocal
title Smart DCA AI Bridge - Install
echo.
echo  Smart DCA AI bridge - install
echo  ------------------------------------------
echo  Lets the chart's AI analysis use Claude Code / Codex / Ollama on this PC.
echo.

if not exist "%~dp0server.mjs" (
  echo  [!] server.mjs not found. Unzip smartdca-ai-bridge.zip first
  echo      ^(right click - Extract All^), then run this file from that folder.
  pause
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo  [!] Node.js is not installed. Install the LTS version from https://nodejs.org/
  echo      then run this installer again.
  start "" https://nodejs.org/
  pause
  exit /b 1
)

set "TARGET=%LOCALAPPDATA%\SmartDCA\ai-bridge"
echo  1/3 Copying files to %TARGET%
if not exist "%TARGET%" mkdir "%TARGET%"
for %%F in (server.mjs package.json package-lock.json Start-AI-Bridge.bat README.txt) do (
  if exist "%~dp0%%F" copy /y "%~dp0%%F" "%TARGET%\" >nul
)

echo  2/3 Installing Claude Agent SDK (takes a minute the first time)
pushd "%TARGET%"
call npm install --omit=dev --no-audit --no-fund
if errorlevel 1 (
  popd
  echo  [!] npm install failed. Check your internet connection and try again.
  pause
  exit /b 1
)
popd

echo  3/3 Registering smartdca-ai:// so Smart DCA can start the bridge
reg add "HKCU\Software\Classes\smartdca-ai" /ve /d "URL:Smart DCA AI Bridge" /f >nul
reg add "HKCU\Software\Classes\smartdca-ai" /v "URL Protocol" /d "" /f >nul
reg add "HKCU\Software\Classes\smartdca-ai\shell\open\command" /ve /d "\"%TARGET%\Start-AI-Bridge.bat\" \"%%1\"" /f >nul

echo.
echo  Done. In Smart DCA: Chart - AI analysis - "Start local AI".
echo  The browser asks once whether to open the link: allow it.
echo.
echo  Starting the bridge now ...
call "%TARGET%\Start-AI-Bridge.bat"
endlocal
