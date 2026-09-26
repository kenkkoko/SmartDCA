---
version: 1
slug: "src-app-jsx"
primary_target: "src/app.jsx"
related_targets: ["index.html"]
---

# Surface: 首頁／加密貨幣分頁 (App shell + CryptoDashboard)

Mode: Operate. Scope of this round: global tokens + light/dark theme + app shell (header, market nav, theme switch) + crypto tab, including the shared FearGreedGauge and WeeklyMacdPanel. Other tabs inherit tokens through the compatibility shim only; their per-component rebuild is a later round.

Audience/job: DCA investor checking daily whether to add; phone and desktop equally. Constraints: no functional changes, all copy verbatim (user-approved additions: MACD divergence annotations "價格 ↘ 新低 / DIF ↗ 沒破底" and their bearish mirror), green = up / red = down in every market. Theme: follows system, header switch overrides and is remembered.

History: the nixie direction was built and rejected by the user ("設計太分離，沒有一體性"). Previews in preview/ (b-home.html is the approved reference: light + dark).

## Direction contract

THESIS: The dashboard is a fund factsheet: one continuous typeset page on a strict grid, sections opened by heavy black rules, never a stack of floating cards. Refuses the category default of glowing gradient cards on navy.

OWN-WORLD: Paper white #ffffff / ink #111 (dark: #121314 / #f1efea), hairline rules #d9d9d9 (dark #2f3032), 3px ink rules open sections. One ultramarine accent (#1b39c4, dark #6f8bff) used only for the current action. Schibsted Grotesk + Noto Sans TC, tabular figures, heavy 900 headings. Zone colours only inside the dial and distribution bar.

STORY: Visitor reads the dial (where 72 sits between 極度恐懼 and 極度貪婪), sees the blue 當前操作 block, then the weekly MACD divergence drawn as two connectors, then the price history.

FIRST VIEWPORT: Header rule; text nav with underline; search + sub-tabs meta row; two tickers side by side on hairlines; the FNG section: dial (7 cols) beside DCA strategy, blue action block, five-step action scale, 365-day distribution (5 cols).

FORM: Grounded candidate "基金月報 factsheet", presented as IMPECCABLE'S PICK in re-roll round 1 and chosen by the user; seed key 81c70793 (reroll 1). Signature: the dial needle eases to the value; MACD divergence connectors A→B on price and DIF panels.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
