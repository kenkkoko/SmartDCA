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
| 前端 | React 18（UMD）+ Tailwind CSS，單頁應用，部署於 GitHub Pages |
| 圖表 | Chart.js 4（走勢 / 縮放）+ TradingView Lightweight Charts 5（K 線 + 繪圖工具） |
| 後端 | Supabase（Auth / Postgres + RLS / Storage / Edge Functions） |
| Edge Functions | `gemini-proxy`（AI 分析，伺服器端驗證 Premium）、`price-proxy`、`cmc-proxy` |
| 排程 | GitHub Actions：每日訊號檢查、新文章推播通知 |

## 🔐 金鑰管理

- Supabase URL / anon key 由 GitHub Actions 於部署時注入（`__SUPABASE_URL_PLACEHOLDER__`）
- Gemini API Key 只存在 Edge Function 環境變數，前端拿不到；使用者也可自帶金鑰（BYOK）
- Premium 權限於伺服器端（service role + `user_profiles`）驗證，不信任前端旗標

## 🧑‍💻 本地開發

```bash
git clone https://github.com/kenkkoko/SmartDCA.git
cd SmartDCA
# 建立 local-config.js（已被 .gitignore 忽略）：
#   window.__LOCAL_CONFIG__ = { SUPABASE_URL: '...', SUPABASE_ANON_KEY: '...' };
python -m http.server 8000
# 開啟 http://localhost:8000
```

## 📄 License

見 [LICENSE](LICENSE)。
