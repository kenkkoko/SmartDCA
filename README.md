# Smart DCA — 智慧定期定額投資追蹤

> 🌐 線上網站：**https://dca.hellokai07.com/**

一個支援**加密貨幣、美股、台股**的智慧 DCA（Dollar-Cost Averaging，定期定額）投資工具。
結合市場情緒指標與技術指標，告訴你「現在是不是歷史上值得加碼的時機」。

## ✨ 功能特色

- **恐懼與貪婪指數儀表板** — 即時 Fear & Greed Index，搭配過去 365 天情緒分佈，並給出對應的 DCA 操作建議（強力買入 / 定期買入 / 觀望）
- **歷史走勢 + 恐慌買點回放** — 在價格圖上標出歷史上的「極度恐懼」買入訊號，驗證恐慌加碼策略
- **投資組合追蹤** — 跨市場（加密貨幣 / 美股 / 台股）持股管理、成本與未實現損益、總市值走勢圖
- **觀察清單 + 訊號燈** — 每檔標的依 RSI 與 1Y 區間位置給出「加碼 / 持有 / 觀望 / 警示」建議
- **AI 投資顧問** — 透過 Gemini 依即時情緒與價格走勢產出個人化建議（Premium 或自帶 API Key）
- **技術分析論壇** — Markdown 編輯器（支援程式碼高亮、LaTeX 公式、貼圖）、標籤分類、草稿
- **PWA** — 可安裝到手機主畫面，支援推播通知（新文章通知）

## 🏗️ 架構

| 層 | 技術 |
|---|---|
| 前端 | React 18 + Vite + Tailwind CSS，單頁應用，部署於 GitHub Pages |
| 圖表 | Chart.js 4（走勢 / 縮放）+ 自製 React 圖表引擎 `src/chart`（K 線、43 種繪圖工具、圖層、即時報價、AI 分析） |
| 後端 | Supabase（Auth / Postgres + RLS / Storage / Edge Functions） |
| Edge Functions | `gemini-proxy`（AI 分析，伺服器端驗證 Premium）、`price-proxy`、`cmc-proxy` |
| 排程 | GitHub Actions：每日訊號檢查、新文章推播通知 |

## 📁 專案結構

```
├─ index.html              網站入口(Vite)
├─ src/
│  ├─ app.jsx              主程式(各頁面、圖表頁、觀察清單…)
│  ├─ forum.jsx            論壇
│  └─ chart/               自製圖表引擎
│     ├─ engine/           畫布、時間軸 / 價格軸、K 線與指標系列
│     ├─ drawings/         繪圖工具、圖層、存檔
│     ├─ react/            工具列、側邊面板、對話框
│     ├─ smartdca/         Smart DCA 專用:技術圖表、AI 分析、圖表設定
│     └─ demo/             引擎試玩頁(npm run dev 後開 /src/chart/demo/)
├─ public/                 靜態檔(圖示、manifest、service worker、下載檔)
├─ data/                   economic_calendar.json(由 scripts/gen_calendar.py 產生)
├─ backend/                GitHub Actions 跑的 Python:每日訊號、論壇推播、判讀驗證
├─ ai-bridge/              本機 AI 橋接(Node):讓圖表的 AI 分析用你電腦上的 Claude / Codex / Ollama
├─ scripts/                建置與產生資料的小工具(AI 橋接 zip、經濟日曆、圖示)
├─ supabase/               資料表 SQL 與 Edge Functions
└─ .github/workflows/      部署與排程
```

## 🔐 金鑰管理

- Supabase URL / anon key 由 GitHub Actions 建置時以 Vite 環境變數注入（`VITE_SUPABASE_URL`）
- Gemini API Key 只存在 Edge Function 環境變數，前端拿不到；使用者也可自帶金鑰（BYOK）
- Premium 權限於伺服器端（service role + `user_profiles`）驗證，不信任前端旗標

## 🧑‍💻 本地開發

```bash
git clone https://github.com/kenkkoko/SmartDCA.git
cd SmartDCA
npm install
# 建立 local-config.js（已被 .gitignore 忽略）：
#   window.__LOCAL_CONFIG__ = { SUPABASE_URL: '...', SUPABASE_ANON_KEY: '...' };
npm run dev      # 開發模式(HMR),開啟 http://localhost:5173
npm run build    # 產出 dist/
```

## 📄 License

見 [LICENSE](LICENSE)。
