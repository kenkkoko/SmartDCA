# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primarily the owner and a small circle of friends using it as a personal DCA tool, with an ambition to grow into a public tool for Taiwanese retail investors (register, upgrade to Pro). Usage is split roughly evenly between phone (installed PWA, quick daily check of sentiment and signals before deciding whether to add to a position) and desktop (K-line analysis, writing forum posts, managing the portfolio).

## Product Purpose

Smart DCA answers one recurring question for a dollar-cost-averaging investor: "Is now historically a good time to add more?" It combines market sentiment (Fear & Greed Index) with technical indicators (RSI, 1Y range position, weekly MACD divergence) across crypto, US stocks, and Taiwan stocks, and turns them into a plain recommendation (強力買入 / 定期買入 / 觀望). Success is a user who opens it, understands the market mood in seconds, and makes a calmer, rules-based buying decision.

## Positioning

Sentiment-driven DCA timing, not trading: it replays historical "extreme fear" buy points on the price chart to prove the panic-buying strategy, and applies the same signal logic across three markets (crypto / US / TW) in one place, in Traditional Chinese.

## Operating Context

- Daily ritual: check Fear & Greed gauge and 365-day sentiment distribution, glance at watchlist signal lights, decide whether to buy this period.
- Deeper sessions: historical chart with fear buy points, TradingView-style K-line modal with drawing tools, trade journal (Wyckoff structure + portfolio-wide risk check), portfolio P&L and total value chart.
- Community: technical-analysis forum with Markdown, code highlighting, LaTeX, images, tags, drafts; push notifications for new posts.
- AI advisor via Gemini (Premium, or bring-your-own API key).
- Scheduled GitHub Actions for daily signal checks and forum notifications.

## Capabilities and Constraints

- Stack: React 18 + Vite + Tailwind CSS 3, single-page app; almost all UI lives in `src/app.jsx` (~8,900 lines) and `src/forum.jsx`; design tokens and component CSS live in a `<style>` block in `index.html`.
- Charts: Chart.js 4 (history / zoom) and TradingView Lightweight Charts 5 (K-line + drawing tools), loaded from CDN.
- Backend: Supabase (Auth, Postgres + RLS, Storage, Edge Functions `gemini-proxy`, `price-proxy`, `cmc-proxy`).
- Deployed on GitHub Pages at https://dca.hellokai07.com/ ; installable PWA with push notifications.
- Tabs: 加密貨幣, 美股市場, 台股市場, 投資組合, 論壇.
- Redesign constraint (confirmed): no functional changes and no copy changes. Visual layer only.
- Price color convention (confirmed): green = up, red = down in every market, including Taiwan stocks.

## Brand Commitments

- Name: Smart DCA (tagline "Intelligent Investing").
- Existing logo and app icons may be replaced; they are not binding.
- All existing Traditional Chinese copy and terminology stay verbatim.

## Evidence on Hand

- Live data only: real prices, Fear & Greed values, and user portfolios. No testimonials, user counts, performance claims, or press exist; none may be invented.
- Pro tier exists (「升級 Pro」); pricing is not documented in the repo and must not be invented.

## Product Principles

1. The answer before the analysis: the mood and the recommended action must be legible at a glance, detail on demand.
2. Calm over hype: the product exists to counter panic and FOMO, so the interface must never amplify them.
3. One signal language across three markets: crypto, US, and TW read the same way.
4. Works as well on a phone in one hand as on a desktop with charts open.
