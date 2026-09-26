---
name: Smart DCA
description: Sentiment-driven DCA timing, typeset as a fund factsheet.
colors:
  paper: "#ffffff"
  surface: "#f7f7f5"
  surface-2: "#efefec"
  ink: "#111111"
  ink-2: "#4a4a4a"
  ink-3: "#6b6b6b"
  rule: "#d9d9d9"
  faint: "#ececec"
  accent: "#1b39c4"
  accent-ink: "#ffffff"
  accent-sub: "#c9d1ff"
  up: "#13804a"
  down: "#c62f2f"
  amber: "#96600c"
  zone-extreme-fear: "#c62f2f"
  zone-fear: "#e39a3b"
  zone-neutral: "#b9b9b9"
  zone-greed: "#5aa37a"
  zone-extreme-greed: "#13804a"
  paper-dark: "#121314"
  surface-dark: "#1a1b1d"
  surface-2-dark: "#222326"
  ink-dark: "#f1efea"
  ink-2-dark: "#bdbab3"
  ink-3-dark: "#9a968f"
  rule-dark: "#2f3032"
  faint-dark: "#252628"
  accent-dark: "#6f8bff"
  accent-ink-dark: "#0b0d1a"
  accent-sub-dark: "#141d4f"
  up-dark: "#45c07e"
  down-dark: "#f0655f"
  amber-dark: "#e7a948"
  zone-extreme-fear-dark: "#f0655f"
  zone-fear-dark: "#e7a948"
  zone-neutral-dark: "#6d6c69"
  zone-greed-dark: "#6fbf8e"
  zone-extreme-greed-dark: "#45c07e"
typography:
  display:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "min(20% of dial width, 118px)"
    fontWeight: 900
    lineHeight: 1
    letterSpacing: "-0.04em"
    fontFeature: "tnum"
  headline:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "22px"
    fontWeight: 900
    letterSpacing: "-0.01em"
  verdict:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "24px"
    fontWeight: 900
  title:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 900
  figure:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "26px"
    fontWeight: 700
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  body:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.625
  nav:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 500
  label:
    fontFamily: "Schibsted Grotesk, Noto Sans TC, ui-sans-serif, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 500
rounded:
  none: "0px"
spacing:
  hair: "4px"
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "20px"
  xl: "24px"
  section: "40px"
components:
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "6px 14px"
  button-outline-sm:
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "4px 10px"
  button-solid:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "6px 14px"
  button-solid-hover:
    backgroundColor: "{colors.ink-2}"
  button-icon:
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    size: "36px"
  action-block:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.accent-ink}"
    typography: "{typography.verdict}"
    rounded: "{rounded.none}"
    padding: "14px 18px"
  action-step-current:
    textColor: "{colors.accent}"
    typography: "{typography.label}"
    padding: "8px 4px 0 0"
  toggle-on:
    textColor: "{colors.ink}"
    padding: "2px 0"
  toggle-off:
    textColor: "{colors.ink-2}"
    padding: "2px 0"
  nav-tab-active:
    textColor: "{colors.ink}"
    typography: "{typography.nav}"
    padding: "10px 0"
  ticker:
    textColor: "{colors.ink}"
    typography: "{typography.figure}"
    padding: "12px 0 14px"
  chip:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.none}"
    padding: "0 5px"
  input-underline:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "6px 2px"
---

# Design System: Smart DCA

## Overview

**Creative North Star: "The Fund Factsheet"**

The dashboard is one continuous typeset page on a strict grid, read top to bottom like a monthly fund report. Sections are opened by heavy 3px ink rules, divided by hairlines, and never lifted into floating cards. Ink on paper carries almost everything; colour is reserved for meaning (price direction, sentiment zones) and a single ultramarine for the one thing the reader should do now.

Density is that of a printed factsheet: tight, tabular, labelled. Numbers are set in tabular figures in the same grotesk as the prose, so the page reads as a single typeset object rather than a UI assembled from widgets. Calm is structural: no glow, no gradient, no motion beyond the dial needle settling on its value.

The world exists in two papers, light (white paper, near-black ink) and dark (warm charcoal paper, warm off-white ink). Theme follows the system until the header switch writes an explicit choice to `localStorage('theme')`; after that the saved choice wins. The theme is set on `<html data-theme>` before first paint, and charts read their colours from the same CSS variables at build time.

The category default this world refuses: glowing gradient cards on navy.

**Key Characteristics:**
- Heavy ink rules open sections; hairlines divide within them.
- Sharp corners everywhere; zero radius is the system.
- One ultramarine accent, used only for the current action.
- Green = up, red = down, in every market including Taiwan.
- Tabular figures in the text face; no separate mono face for numbers.
- Flat paper; depth comes from rules and weight, not shadow.

## Colors

Ink on paper with one ultramarine, plus a disciplined set of semantic colours that appear only where they carry data.

### Primary
- **Factsheet Ultramarine** (`accent`; dark `accent-dark`): fills the 當前操作 block and marks the current step of the five-step action scale (text colour plus a 3px top bar). Also the focus ring, caret, native control accent and text selection. The dark value is lifted to keep the block legible on charcoal. Text on it uses `accent-ink`; its small label uses `accent-sub`.

### Neutral
- **Paper** (`paper` / `paper-dark`): the page. Everything sits directly on it.
- **Surface** and **Surface 2** (`surface`, `surface-2` and dark variants): quiet tonal fills for image placeholders, inline code and the legacy shim; not a card background in the built world.
- **Ink** (`ink` / `ink-dark`): headings, figures, the section-opening rules, outline button strokes, the dial ticks and needle, the chart price line.
- **Ink 2** (`ink-2`): body copy, descriptions, unselected toggles and nav tabs.
- **Ink 3** (`ink-3`): labels, axis ticks, dates, meta text.
- **Rule** (`rule`): 1px hairlines between rows, tickers and key figures; disabled button strokes.
- **Faint** (`faint`): chart grid lines and the "other days" segment of the distribution bar.

### Semantic
- **Up Green** (`up`) and **Down Red** (`down`): price change, P&L, bullish/bearish divergence connectors, MACD histogram bars (35% opacity). Direction only.
- **Amber** (`amber`): the 恐懼 (fear) count and warning chips.
- **Sentiment zones** (`zone-extreme-fear` through `zone-extreme-greed`): the five arcs of the Fear & Greed dial and the segments of the 365-day distribution bar. Inactive arcs sit at 38% opacity; the current zone is drawn at full strength and 1.9x band width.

### Named Rules
**The One Accent Rule.** Ultramarine marks the current action and nothing else: the action block and the current step of the action scale. Selection, navigation and emphasis elsewhere are done in ink (weight and underline), never in blue.

**The Market-Neutral Direction Rule.** Green is up and red is down in every market, Taiwan stocks included. Never invert for local convention; one signal language across three markets.

**The Zones Stay Home Rule.** The five sentiment zone colours live only inside the dial and the distribution bar. They are not reused as decoration, badges or backgrounds.

## Typography

**Display Font:** Schibsted Grotesk (with Noto Sans TC for Chinese, then ui-sans-serif / system-ui)
**Body Font:** the same stack
**Label/Mono Font:** none; numbers use the same stack with `font-variant-numeric: tabular-nums`

**Character:** A single hard-working grotesk at two extremes: 900 black for headings and the verdict, 400 to 500 for prose and labels. Noto Sans TC matches weight step for step, so Chinese headings carry the same heaviness as Latin.

### Hierarchy
- **Display** (900, up to 118px at 20% of dial width, -0.04em, tabular): the Fear & Greed reading at the centre of the dial. One per page.
- **Verdict** (900, 24px): the DCA strategy title, the action inside the ultramarine block, the MACD status title (20px on mobile).
- **Headline** (900, 22px, -0.01em; 18px under 640px): section titles directly under the heavy rule.
- **Title** (900, 17px): minor section titles (市場頭條, 關於數據來源, error titles).
- **Figure** (700, 26px, -0.01em, tabular; 19px under 640px): ticker prices. Key figures in the MACD row use 700 at 22px (18px mobile).
- **Body** (400, 15px, line-height 1.625): descriptions, news headlines. Notes cap at 80ch.
- **Nav** (15px; 500 unselected, 700 selected): market tabs.
- **Label** (500, 12px, ink-3): field labels, legends, axis text, meta. Chart text is 11 to 12px, 500.

### Named Rules
**The Tabular Rule.** Every number that can change is set in tabular figures in the text face. Do not reintroduce a monospace face for prices.

**The Two-Weight Rule.** Headings are 900 or they are not headings; figures are 700; everything else is 400 to 500. No 600s, no light weights.

## Layout

A single centred column, max 1024px, with 16px side gutters (24px from 640px). Content is stacked sections, each opened by a 3px ink rule with 14px above its head and 18px (12px on mobile) between head and content; sections are separated by 40px (24 to 32px for the first). The header ends in its own 3px ink rule; the market nav and the search/sub-tab meta row each sit on a 1px hairline.

Inside a section the grid is explicit. The Fear & Greed section splits 7fr / 5fr (dial beside strategy) with a 40px gap on desktop and stacks with 20px on mobile. Tickers sit side by side on a hairline row, divided by a 1px vertical rule with 20px (12px mobile) inset; on mobile the name and price wrap to their own lines. Key figures form a 4-column row divided by vertical hairlines that folds to 2x2 under 768px. News lists run in two columns from 768px with a 32px column gap, each row closed by a hairline.

Breakpoints are 640px (type and inset step-down) and 768px (column changes). Safe-area insets pad the body for the installed PWA.

## Elevation & Depth

Flat. Surfaces do not cast shadows; depth is expressed by rule weight (3px ink opens, 1.5px ink outlines, 1px hairline divides) and by type weight. Selected states are drawn as inset bars (`box-shadow: inset 0 -2px 0` or `inset 0 3px 0`), which are rules, not elevation.

### Named Rules
**The Rules-Not-Shadows Rule.** If something needs separating, give it a rule. Charts draw on plain paper: no glow under lines, no dot grid, no gradient beyond a 6% ink fill under the price line.

## Shapes

Sharp. Every built element has square corners: buttons, chips, the action block, inputs, tooltips (Chart.js `cornerRadius: 0`). Strokes come in three weights: 3px ink (section opener, active nav underline, current-step bar), 1.5px ink (buttons, chips, selected time range, input underline), 1px hairline. The only curves are the dial arcs, the round needle cap and the MACD pivot circles, which are data geometry.

## Components

### Buttons
Printed and plain: a box drawn with a pen.
- **Shape:** square (0px), 1.5px ink stroke.
- **Outline (default):** transparent fill, ink text, 500 at 14px, padding 6px 14px; small variant 13px at 4px 10px; icon variant 36px square.
- **Solid:** ink fill, paper text. Used for the primary action in a row (登入, 分析 →, active 編輯 state).
- **Hover / Focus:** outline gains an 8% ink wash; solid shifts to ink-2. 150ms colour transition on the system ease-out. Focus is a 2px ultramarine outline, 2px offset.
- **Disabled:** ink-3 text, rule-coloured stroke, no fill.

### Chips
- **Style:** 11px, 700, 1.5px border in `currentColor`, padding 0 5px, square. Colour is set by meaning: ink for PRO/VIP, ink-2 for GEMINI, up/down for divergence strength.

### Text Toggles
- **Style:** a row of plain text buttons, 14px (13px mobile), 16px apart. Unselected ink-2; selected ink, 700, with a 2px ink underline drawn as an inset bar. Used for sub-tabs (行情 / 鏈上估值 / 合約數據) and view switches (恐懼貪婪 / 週線 MACD).
- **Time range:** the one boxed variant: the selected range gets a 1.5px ink box and 700; others have a transparent box of the same size so nothing shifts.

### Inputs / Fields
- **Style:** no box, transparent fill, a 1.5px ink underline, 15px text, ink-3 placeholder. Paired with a small outline button.
- **Disabled/locked:** text drops to ink-3 with a lock icon at the right end.

### Navigation
- **Market tabs:** text only, 15px, gaps across a hairline row. Active tab is ink 700 with a 3px ink underline; others ink-2 500. Full names from 768px, short names below.
- **Header:** wordmark left, outline icon buttons and a solid 登入 right, closed by a 3px ink rule. The theme switch is an outline icon button that toggles light/dark and remembers the choice.

### Tickers
Prices sit on hairlines, not in cards. Name 13px 700 ink-2 (ink with a 2px underline when selected), price in the Figure role, change 13px 500 in up/down. Adjacent tickers are divided by a vertical hairline.

### The Current Action Block (signature)
The only ultramarine surface. Full-width flat block, padding 14px 18px, the 當前操作 label left in `accent-sub`, the action right in the Verdict role. Directly beneath, the five-step action scale (強力買入 → 止盈/減倉) sits on a 1px ink line in five equal columns; the current step turns ultramarine, 900, with a 3px ultramarine bar on top. Followed by the 365-day distribution: an 8px bar of zone segments with 2px gaps.

### The Sentiment Dial (signature)
A 180-degree dial drawn in real pixels (280 to 620px). Five zone arcs, 51 ink ticks (every tenth longer and numbered; every twentieth on narrow screens), zone names inside the arcs with the current one in ink 900. The needle is a short ink bar on the outer ring only, so it never crosses the reading, and eases to the value over 900ms (`cubic-bezier(0.16, 1, 0.3, 1)`); no motion under reduced-motion. The reading sits at the centre in the Display role with the classification below it.

### The Divergence Diagram (signature)
Weekly close above, MACD (12, 26, 9) below, drawn in real pixels. Price and DIF in ink, DEA in ink-3, histogram in up/down at 35%. A detected divergence is drawn as a 2.5px connector A→B offset from each curve, with hollow pivot circles, dashed guides through both panels and a short annotation in the divergence colour with a paper halo. Bullish is up green, bearish is down red; older divergences draw at 55% opacity without labels. The status head uses a 44px ring glyph (down-chevron, up-chevron or flat bar) in the status colour.

### Charts (Chart.js)
Defaults come from the tokens at build time: ink line at 1.6px, 0.25 tension, 6% ink fill fading to 0, faint grid, ink-3 ticks at 11px 500, tooltip on paper with a 1px ink border, square corners, no colour swatches. Fear buy points are down-red (extreme fear) and zone-fear amber. Charts rebuild when the theme changes.

## Do's and Don'ts

### Do:
- **Do** open every top-level section with a 3px ink rule and put its title (Headline, 900) directly beneath.
- **Do** keep ultramarine to the current action block and the current step of the action scale.
- **Do** colour price direction green up / red down in every market, Taiwan included.
- **Do** set every changing number in tabular figures in Schibsted Grotesk / Noto Sans TC.
- **Do** mark selection with ink weight plus an underline bar (2px toggles, 3px nav), not with fills or colour.
- **Do** read chart and SVG colours from the CSS variables so both papers render correctly.
- **Do** keep corners square (0px) on every new element.

### Don't:
- **Don't** build glowing gradient cards on navy, or any card stack; this page is one continuous sheet.
- **Don't** use ultramarine for links, tabs, icons, chips or hover states.
- **Don't** use the sentiment zone colours outside the dial and the distribution bar.
- **Don't** add shadows, glows or dot grids to surfaces or chart lines.
- **Don't** invert red/green for Taiwan stocks.
- **Don't** reintroduce a monospace face for prices.

## Pending migration

Only the app shell (header, market nav, theme switch) and the crypto tab, with the shared dial, DCA strategy column and weekly MACD diagram, are built in this world. 美股市場, 台股市場, 投資組合, 論壇, all modals and popovers, and TechnicalChartModal still run on the compatibility shim in `index.html`, which remaps legacy dark Tailwind classes onto these tokens and clamps radii to 2 to 4px. Shim behaviour (soft-shadowed popovers, residual 2 to 4px radii, filled blue buttons remapped to ink, tinted status backgrounds, forum code blocks with rounded corners and JetBrains Mono) is a stopgap, not part of this system. Rebuild those surfaces with the components above rather than copying shim output.
