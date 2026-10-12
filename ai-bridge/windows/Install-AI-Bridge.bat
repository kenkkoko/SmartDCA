@echo off
chcp 65001 >nul
setlocal
title Smart DCA AI 橋接 - 安裝
echo.
echo  Smart DCA 本機 AI 橋接 - 安裝
echo  ------------------------------------------
echo  讓 Smart DCA 圖表的「AI 分析」使用這台電腦上的 Claude Code / Codex / Ollama。
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo  [!] 這台電腦還沒有 Node.js。
  echo      請先安裝 Node.js（LTS 版）：https://nodejs.org/
  echo      裝好後重新執行這個安裝檔。
  start "" https://nodejs.org/
  pause
  exit /b 1
)

set "TARGET=%LOCALAPPDATA%\SmartDCA\ai-bridge"
echo  1/3 複製檔案到 %TARGET%
if not exist "%TARGET%" mkdir "%TARGET%"
copy /y "%~dp0server.mjs" "%TARGET%\" >nul
copy /y "%~dp0package.json" "%TARGET%\" >nul
if exist "%~dp0package-lock.json" copy /y "%~dp0package-lock.json" "%TARGET%\" >nul
copy /y "%~dp0Start-AI-Bridge.bat" "%TARGET%\" >nul
copy /y "%~dp0README.txt" "%TARGET%\" >nul

echo  2/3 安裝 Claude Agent SDK（第一次需要一點時間）
pushd "%TARGET%"
call npm install --omit=dev --no-audit --no-fund
if errorlevel 1 (
  popd
  echo  [!] npm install 失敗，請確認網路後重試。
  pause
  exit /b 1
)
popd

echo  3/3 讓 Smart DCA 可以一鍵啟動橋接（smartdca-ai:// 連結）
reg add "HKCU\Software\Classes\smartdca-ai" /ve /d "URL:Smart DCA AI Bridge" /f >nul
reg add "HKCU\Software\Classes\smartdca-ai" /v "URL Protocol" /d "" /f >nul
reg add "HKCU\Software\Classes\smartdca-ai\shell\open\command" /ve /d "\"%TARGET%\Start-AI-Bridge.bat\" \"%%1\"" /f >nul

echo.
echo  完成！之後在 Smart DCA 圖表的「AI 分析」按「啟動本機 AI」就會自動開啟橋接。
echo  （瀏覽器第一次會問要不要開啟，選「開啟」並勾選一律允許。）
echo.
echo  現在先幫你啟動一次 ...
call "%TARGET%\Start-AI-Bridge.bat"
endlocal
