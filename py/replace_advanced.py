"""Replace the TechnicalChartModal block with the next iteration:

  - K-line + indicator colors aligned with design tokens (--up #00d68f / --down #ff5b6e)
  - EMA Ribbon palette switches green ↔ red based on latest fast-vs-slow trend
  - 200MA overlay (auto-extends history range so the SMA has enough bars)
  - RSI: 50 mid-line in addition to 70 / 30 reference lines
  - MACD: histogram tinted per bar by sign (same as before but with site tokens)
  - "Composite" watchlist-style signal (RSI + 1Y position) as a toggleable indicator
  - Click-to-draw uses logical-coordinate interpolation so anchors land at the
    exact tap position instead of snapping to the nearest bar
  - Touch → mouse polyfill on the chart container so anchor drag works on phones
  - Tool-category popup repositions to viewport edges on narrow screens
  - Compact signal summary panel pinned top-right of the chart area
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "index.html"

START_MARKER = "        // ─────────────────────────────────────────────\n        // TechnicalChartModal — full-screen analysis built on\n"
END_MARKER = "\n        const WatchlistDashboard = ({ watchlist, onUpdateWatchlist, isPremium, user }) => {\n"

NEW_BLOCK = r'''        // ─────────────────────────────────────────────
        // TechnicalChartModal — full-screen analysis built on
        // TradingView Lightweight Charts v5 + deepentropy/lightweight-charts-drawing
        // (68 community-built drawing tools).
        // ─────────────────────────────────────────────
        // EMA Ribbon — mirrors the Pine "BTC EMA Ribbon Reversal" indicator
        //   fast = 20, slow = 55, four mid lines at round(fast + (slow-fast)*0.2/0.4/0.6/0.8)
        const EMA_FAST = 20;
        const EMA_SLOW = 55;
        const EMA_PERIODS = (() => {
            const f = EMA_FAST, s = EMA_SLOW;
            return [f,
                Math.round(f + (s - f) * 0.2),
                Math.round(f + (s - f) * 0.4),
                Math.round(f + (s - f) * 0.6),
                Math.round(f + (s - f) * 0.8),
                s];
        })();
        // Single semi-transparent color flips with the latest fast>slow trend (matches Pine color.new(..., 40))
        const EMA_BULL_COLOR = 'rgba(0, 200, 83, 0.6)';
        const EMA_BEAR_COLOR = 'rgba(255, 23, 68, 0.6)';
        const UP_COLOR = '#00d68f';
        const DOWN_COLOR = '#ff5b6e';

        // 68 tools grouped into 8 categories. Anchor counts come from the library's registry.
        const DRAWING_TOOL_CATEGORIES = [
            { key: 'line', label: '線', tools: [
                { type: 'trend-line', name: '趨勢線', a: 2 },
                { type: 'ray', name: '射線', a: 2 },
                { type: 'extended-line', name: '延伸線', a: 2 },
                { type: 'horizontal-line', name: '水平線', a: 1 },
                { type: 'horizontal-ray', name: '水平射線', a: 1 },
                { type: 'vertical-line', name: '垂直線', a: 1 },
                { type: 'cross-line', name: '十字線', a: 1 },
                { type: 'info-line', name: '訊息線', a: 2 },
                { type: 'trend-angle', name: '趨勢角度', a: 2 },
                { type: 'arrow', name: '箭頭', a: 2 },
            ]},
            { key: 'channel', label: '通道', tools: [
                { type: 'parallel-channel', name: '平行通道', a: 3 },
                { type: 'regression-trend', name: '迴歸趨勢', a: 2 },
                { type: 'flat-top-bottom', name: '平頂 / 底', a: 3 },
                { type: 'disjoint-channel', name: '斷開通道', a: 4 },
            ]},
            { key: 'pitchfork', label: '叉形', tools: [
                { type: 'andrews-pitchfork', name: 'Andrews 叉形', a: 3 },
                { type: 'schiff-pitchfork', name: 'Schiff 叉形', a: 3 },
                { type: 'modified-schiff-pitchfork', name: 'Modified Schiff', a: 3 },
                { type: 'inside-pitchfork', name: 'Inside 叉形', a: 3 },
            ]},
            { key: 'fibonacci', label: '黃金分割', tools: [
                { type: 'fib-retracement', name: 'Fib 回撤', a: 2 },
                { type: 'fib-extension', name: 'Fib 延伸', a: 3 },
                { type: 'fib-channel', name: 'Fib 通道', a: 3 },
                { type: 'fib-time-zone', name: 'Fib 時區', a: 2 },
                { type: 'fib-speed-fan', name: 'Fib 速度扇形', a: 2 },
                { type: 'fib-time-extension', name: 'Fib 時間延伸', a: 3 },
                { type: 'fib-circles', name: 'Fib 圓', a: 2 },
                { type: 'fib-spiral', name: 'Fib 螺旋', a: 2 },
                { type: 'fib-arcs', name: 'Fib 弧', a: 2 },
                { type: 'fib-wedge', name: 'Fib 楔形', a: 3 },
                { type: 'pitchfan', name: 'Pitchfan', a: 3 },
            ]},
            { key: 'gann', label: '江恩', tools: [
                { type: 'gann-box', name: '江恩盒', a: 2 },
                { type: 'gann-fan', name: '江恩扇形', a: 2 },
                { type: 'gann-square-fixed', name: '江恩方形（固定）', a: 1 },
                { type: 'gann-square', name: '江恩方形', a: 2 },
            ]},
            { key: 'forecasting', label: '預測', tools: [
                { type: 'long-position', name: '多單部位', a: 3 },
                { type: 'short-position', name: '空單部位', a: 3 },
                { type: 'forecast', name: '預測', a: 2 },
                { type: 'bars-pattern', name: 'K 棒模式', a: 3 },
                { type: 'projection', name: '投影', a: 3 },
                { type: 'price-range', name: '價格區間', a: 2 },
                { type: 'date-range', name: '日期區間', a: 2 },
                { type: 'date-price-range', name: '日期 + 價格區間', a: 2 },
            ]},
            { key: 'shape', label: '形狀', tools: [
                { type: 'rectangle', name: '矩形', a: 2 },
                { type: 'rotated-rectangle', name: '旋轉矩形', a: 3 },
                { type: 'circle', name: '圓形', a: 2 },
                { type: 'triangle', name: '三角形', a: 3 },
                { type: 'ellipse', name: '橢圓', a: 2 },
                { type: 'arc', name: '弧', a: 3 },
                { type: 'path', name: '路徑', a: 2 },
                { type: 'polyline', name: '折線', a: 2 },
                { type: 'curve', name: '曲線', a: 4 },
                { type: 'double-curve', name: '雙曲線', a: 3 },
            ]},
            { key: 'annotation', label: '標註', tools: [
                { type: 'text-annotation', name: '文字', a: 1 },
                { type: 'callout', name: '標註框', a: 2 },
                { type: 'anchored-text', name: '錨定文字', a: 2 },
                { type: 'note', name: '便籤', a: 1 },
                { type: 'price-note', name: '價格便籤', a: 1 },
                { type: 'price-label', name: '價格標籤', a: 1 },
                { type: 'flag-mark', name: '旗幟', a: 1 },
                { type: 'pin', name: '釘子', a: 1 },
                { type: 'comment', name: '註解', a: 1 },
                { type: 'signpost', name: '路標', a: 1 },
                { type: 'table', name: '表格', a: 1 },
                { type: 'brush', name: '筆刷', a: 2 },
                { type: 'highlighter', name: '螢光筆', a: 2 },
                { type: 'arrow-marker', name: '箭頭記號', a: 1 },
                { type: 'arrow-mark-up', name: '向上箭頭', a: 1 },
                { type: 'arrow-mark-down', name: '向下箭頭', a: 1 },
            ]},
        ];

        // FNG → bull/bear/neutral classification (mirrors the dashboard's DCA suggestion)
        const fngClassify = (v) => {
            if (v == null || isNaN(v)) return null;
            if (v <= 25) return { tone: 'bull', label: '極度恐懼' };
            if (v <= 45) return { tone: 'bull', label: '恐懼' };
            if (v <= 55) return { tone: 'neutral', label: '中立' };
            if (v <= 74) return { tone: 'bear', label: '貪婪' };
            return { tone: 'bear', label: '極度貪婪' };
        };

        const TechnicalChartModal = ({ symbol, type, onClose }) => {
            const containerRef = useRef(null);
            const chartRef = useRef(null);
            const candleSeriesRef = useRef(null);
            const emaSeriesRef = useRef([]);
            const ma200SeriesRef = useRef(null);
            const rsiSeriesRef = useRef(null);
            const macdRefs = useRef({ dif: null, dea: null, hist: null });
            const markersPluginRef = useRef(null);     // candle pane: holds FNG dots + EMA LONG/SHORT
            const rsiMarkersRef = useRef(null);        // RSI pane: 30 / 70 crossover markers
            const macdMarkersRef = useRef(null);       // MACD pane: DIF×DEA golden / death cross markers
            const emaCrossesRef = useRef([]);          // cached so the FNG-markers effect can merge them
            const drawingManagerRef = useRef(null);
            const drawingsRef = useRef([]);
            const activeToolRef = useRef(null);
            const pendingAnchorsRef = useRef([]);
            const selectedDrawingIdRef = useRef(null);
            const candleDataRef = useRef([]);

            const [showHelp, setShowHelp] = useState(false);
            const [activeTool, setActiveTool] = useState(null); // { type, name, requiredAnchors, collected }
            const [openCategory, setOpenCategory] = useState(null);
            const [hasSelection, setHasSelection] = useState(false);

            const [showEMA, setShowEMA] = useLocalState('tech-show-ema', true);
            const [showMA200, setShowMA200] = useLocalState('tech-show-ma200', true);
            const [showRSI, setShowRSI] = useLocalState('tech-show-rsi', true);
            const [showMACD, setShowMACD] = useLocalState('tech-show-macd', true);
            const [showFearSignal, setShowFearSignal] = useLocalState('tech-show-fear', true);
            const [showComposite, setShowComposite] = useLocalState('tech-show-composite', true);
            const [showSignalPanel, setShowSignalPanel] = useLocalState('tech-show-panel', true);
            const [timeframe, setTimeframe] = useLocalState('tech-timeframe', '1d'); // '1d' | '1wk'

            const signalSource = (type === 'CRYPTO' || type === 'crypto' || type === 'US') ? 'fng' : 'rsi';
            const [fngHistory, setFngHistory] = useState([]);
            const [timeRange, setTimeRange] = useState('1y');
            const [history, setHistory] = useState([]);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);

            // Auto-extend the underlying fetch range so indicators have enough bars to compute
            const effectiveHistoryRange = useMemo(() => {
                let r = timeRange;
                if (timeframe === '1wk' && ['1mo', '3mo', '6mo', '1y'].includes(r)) r = '2y';
                if (showMA200) {
                    if (timeframe === '1d' && ['1mo', '3mo', '6mo'].includes(r)) r = '1y';
                    if (timeframe === '1wk' && ['1y', '2y'].includes(r)) r = '5y';
                }
                return r;
            }, [timeframe, timeRange, showMA200]);

            // ─── Fetch price history (Binance for crypto, Yahoo via cors proxy for stocks) ───
            useEffect(() => {
                if (!symbol) return;
                let cancelled = false;
                (async () => {
                    setLoading(true);
                    setError(null);
                    try {
                        const apiType = (type === 'CRYPTO' || type === 'crypto') ? 'crypto' : 'stock';
                        if (apiType === 'crypto') {
                            const binSym = getBinanceSymbol(symbol);
                            if (!binSym) { setError(`不支援的幣種：${symbol}（不在 Binance 上市）`); setHistory([]); return; }
                            const days = rangeToDays(effectiveHistoryRange);
                            const limit = Math.min(1000, days + 5);
                            const binRes = await fetch(`https://api.binance.com/api/v3/klines?symbol=${binSym}&interval=1d&limit=${limit}`);
                            if (cancelled) return;
                            if (!binRes.ok) { setError(`Binance returned ${binRes.status}`); setHistory([]); return; }
                            const klines = await binRes.json();
                            setHistory(klines.map(k => ({
                                date: new Date(k[0]).toISOString().slice(0, 10),
                                price: +k[4], open: +k[1], high: +k[2], low: +k[3],
                            })));
                            return;
                        }
                        const range = ['1mo','3mo','6mo','1y','2y','5y','max'].includes(effectiveHistoryRange) ? effectiveHistoryRange : 'max';
                        const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1d&range=${range}`;
                        const stockRes = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yahooUrl));
                        if (cancelled) return;
                        if (!stockRes.ok) { setError(`Yahoo returned ${stockRes.status}`); setHistory([]); return; }
                        const yfJson = await stockRes.json();
                        const result = yfJson?.chart?.result?.[0];
                        if (!result) { setError(yfJson?.chart?.error?.description || '查無此代號'); setHistory([]); return; }
                        const ts = result.timestamp || [];
                        const q = result.indicators?.quote?.[0] || {};
                        const opens = q.open || [], highs = q.high || [], lows = q.low || [], closes = q.close || [];
                        const hist = [];
                        for (let i = 0; i < ts.length; i++) {
                            const c = closes[i];
                            if (c == null || isNaN(c)) continue;
                            const entry = { date: new Date(ts[i] * 1000).toISOString().slice(0, 10), price: c };
                            if (opens[i] != null && !isNaN(opens[i])) entry.open = opens[i];
                            if (highs[i] != null && !isNaN(highs[i])) entry.high = highs[i];
                            if (lows[i] != null && !isNaN(lows[i])) entry.low = lows[i];
                            hist.push(entry);
                        }
                        setHistory(hist);
                    } catch (e) {
                        if (!cancelled) setError(String(e));
                    } finally {
                        if (!cancelled) setLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, [symbol, type, effectiveHistoryRange]);

            // ─── Fetch FNG history (crypto / US only) ───
            useEffect(() => {
                if (signalSource !== 'fng') { setFngHistory([]); return; }
                let cancelled = false;
                (async () => {
                    try {
                        const r = effectiveHistoryRange;
                        const days = r === '1mo' ? 30 : r === '3mo' ? 90 : r === '6mo' ? 180 :
                                     r === '1y' ? 365 : r === '2y' ? 730 : r === 'max' ? 1825 : 365;
                        if (type === 'CRYPTO' || type === 'crypto') {
                            const res = await fetch(`https://api.alternative.me/fng/?limit=${days}&format=json`);
                            const json = await res.json();
                            if (cancelled) return;
                            const arr = (json.data || []).map(d => ({
                                date: new Date(Number(d.timestamp) * 1000).toISOString().slice(0, 10),
                                fng: parseInt(d.value, 10),
                            })).reverse();
                            setFngHistory(arr);
                        } else if (type === 'US') {
                            const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent('https://production.dataviz.cnn.io/index/fearandgreed/graphdata'));
                            const json = await res.json();
                            if (cancelled) return;
                            const raw = json?.fear_and_greed_historical?.data || [];
                            setFngHistory(raw.map(d => ({ date: new Date(d.x).toISOString().slice(0, 10), fng: Math.round(d.y) })));
                        }
                    } catch (e) {
                        console.warn('FNG fetch failed:', e);
                        if (!cancelled) setFngHistory([]);
                    }
                })();
                return () => { cancelled = true; };
            }, [type, effectiveHistoryRange, signalSource]);

            const displayHistory = useMemo(
                () => (timeframe === '1wk' ? aggregateToWeekly(history) : history),
                [history, timeframe]
            );

            const candleData = useMemo(() => {
                const seen = new Set();
                const arr = displayHistory
                    .filter(p => { if (seen.has(p.date)) return false; seen.add(p.date); return true; })
                    .map(p => ({
                        time: Math.floor(new Date(p.date + 'T00:00:00Z').getTime() / 1000),
                        open: +(p.open ?? p.price),
                        high: +(p.high ?? p.price),
                        low: +(p.low ?? p.price),
                        close: +p.price,
                    }));
                arr.sort((a, b) => a.time - b.time);
                return arr;
            }, [displayHistory]);

            // Click handler reads through this ref so it always sees fresh data
            useEffect(() => { candleDataRef.current = candleData; }, [candleData]);

            const fngByDate = useMemo(() => {
                const m = {};
                for (const d of fngHistory) m[d.date] = d.fng;
                return m;
            }, [fngHistory]);

            // ─── Init chart (once) ───
            useEffect(() => {
                if (!containerRef.current || !window.LightweightCharts) return;
                const LWC = window.LightweightCharts;
                const chart = LWC.createChart(containerRef.current, {
                    layout: {
                        background: { type: (LWC.ColorType && LWC.ColorType.Solid) || 'solid', color: '#07080c' },
                        textColor: '#b8bcc8',
                        panes: { separatorColor: 'rgba(255,255,255,0.10)', separatorHoverColor: 'rgba(255,255,255,0.20)', enableResize: true },
                    },
                    grid: {
                        vertLines: { color: 'rgba(255,255,255,0.04)' },
                        horzLines: { color: 'rgba(255,255,255,0.04)' },
                    },
                    timeScale: { borderColor: 'rgba(255,255,255,0.1)', timeVisible: true, secondsVisible: false, rightOffset: 6 },
                    rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
                    crosshair: { mode: 1 },
                    autoSize: true,
                });
                chartRef.current = chart;

                const candleSeries = chart.addSeries(LWC.CandlestickSeries, {
                    upColor: UP_COLOR, downColor: DOWN_COLOR,
                    borderUpColor: UP_COLOR, borderDownColor: DOWN_COLOR,
                    wickUpColor: UP_COLOR, wickDownColor: DOWN_COLOR,
                }, 0);
                candleSeriesRef.current = candleSeries;

                if (typeof LWC.createSeriesMarkers === 'function') {
                    markersPluginRef.current = LWC.createSeriesMarkers(candleSeries, []);
                }

                if (window.LightweightChartsDrawing && window.LightweightChartsDrawing.DrawingManager) {
                    const manager = new window.LightweightChartsDrawing.DrawingManager();
                    manager.attach(chart, candleSeries, containerRef.current);
                    drawingManagerRef.current = manager;
                    manager.on && manager.on('drawing:selected', (e) => {
                        selectedDrawingIdRef.current = e.drawingId || null;
                        setHasSelection(true);
                    });
                    manager.on && manager.on('drawing:deselected', () => {
                        selectedDrawingIdRef.current = null;
                        setHasSelection(false);
                    });
                }

                // ─── Free-form click → anchor: interpolate sub-bar time so the anchor
                //     lands exactly where the user tapped (no bar snap). ───
                const onClick = (param) => {
                    const tool = activeToolRef.current;
                    if (!tool || !param || !param.point) return;
                    const data = candleDataRef.current;
                    let time = null;
                    try {
                        const logical = chart.timeScale().coordinateToLogical(param.point.x);
                        if (logical != null && data.length > 0) {
                            const idx = Math.floor(logical);
                            const frac = logical - idx;
                            if (idx >= 0 && idx < data.length - 1) {
                                time = data[idx].time + (data[idx + 1].time - data[idx].time) * frac;
                            } else if (idx >= data.length - 1) {
                                const n = data.length;
                                const spacing = n >= 2 ? data[n - 1].time - data[n - 2].time : 86400;
                                time = data[n - 1].time + spacing * (logical - (n - 1));
                            } else {
                                const spacing = data.length >= 2 ? data[1].time - data[0].time : 86400;
                                time = data[0].time + spacing * logical;
                            }
                        }
                    } catch {}
                    if (time == null) {
                        if (param.time == null) return;
                        time = param.time;
                    }
                    time = Math.floor(time);
                    const price = candleSeries.coordinateToPrice(param.point.y);
                    if (price == null) return;
                    pendingAnchorsRef.current.push({ time, price });
                    setActiveTool(prev => prev ? { ...prev, collected: pendingAnchorsRef.current.length } : prev);
                    if (pendingAnchorsRef.current.length >= tool.requiredAnchors) {
                        const registry = window.LightweightChartsDrawing && window.LightweightChartsDrawing.getToolRegistry && window.LightweightChartsDrawing.getToolRegistry();
                        if (registry && drawingManagerRef.current) {
                            const id = 'd-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6);
                            const drawing = registry.createDrawing(tool.type, id, pendingAnchorsRef.current.slice());
                            if (drawing) {
                                drawingManagerRef.current.addDrawing(drawing);
                                drawingsRef.current.push(id);
                            }
                        }
                        pendingAnchorsRef.current = [];
                        activeToolRef.current = null;
                        setActiveTool(null);
                        if (containerRef.current) containerRef.current.style.cursor = '';
                    }
                };
                chart.subscribeClick(onClick);

                // ─── Touch → mouse polyfill so the drawing manager's mouse-only
                //     anchor-drag handlers also respond to finger drags on mobile. ───
                const container = containerRef.current;
                const dispatchMouse = (type, t) => {
                    const ev = new MouseEvent(type, {
                        bubbles: true, cancelable: true, view: window,
                        clientX: t.clientX, clientY: t.clientY,
                        button: 0,
                    });
                    container.dispatchEvent(ev);
                };
                const onTouchStart = (e) => {
                    if (e.touches.length !== 1) return;
                    dispatchMouse('mousedown', e.touches[0]);
                };
                const onTouchMove = (e) => {
                    if (e.touches.length !== 1) return;
                    dispatchMouse('mousemove', e.touches[0]);
                };
                const onTouchEnd = (e) => {
                    const t = e.changedTouches[0];
                    if (t) dispatchMouse('mouseup', t);
                };
                container.addEventListener('touchstart', onTouchStart, { passive: true });
                container.addEventListener('touchmove', onTouchMove, { passive: true });
                container.addEventListener('touchend', onTouchEnd, { passive: true });

                const onKey = (e) => {
                    if (e.key === 'Escape') {
                        if (activeToolRef.current) {
                            activeToolRef.current = null;
                            pendingAnchorsRef.current = [];
                            setActiveTool(null);
                            if (containerRef.current) containerRef.current.style.cursor = '';
                        } else {
                            onClose && onClose();
                        }
                    } else if (e.key === 'Delete' || e.key === 'Backspace') {
                        const m = drawingManagerRef.current;
                        const id = selectedDrawingIdRef.current;
                        if (m && id) {
                            m.removeDrawing(id);
                            drawingsRef.current = drawingsRef.current.filter(x => x !== id);
                            selectedDrawingIdRef.current = null;
                            setHasSelection(false);
                        }
                    }
                };
                window.addEventListener('keydown', onKey);

                return () => {
                    window.removeEventListener('keydown', onKey);
                    container.removeEventListener('touchstart', onTouchStart);
                    container.removeEventListener('touchmove', onTouchMove);
                    container.removeEventListener('touchend', onTouchEnd);
                    try { chart.unsubscribeClick(onClick); } catch {}
                    try { drawingManagerRef.current && drawingManagerRef.current.detach(); } catch {}
                    try { chart.remove(); } catch {}
                    chartRef.current = null;
                    candleSeriesRef.current = null;
                    drawingManagerRef.current = null;
                    markersPluginRef.current = null;
                    emaSeriesRef.current = [];
                    ma200SeriesRef.current = null;
                    rsiSeriesRef.current = null;
                    macdRefs.current = { dif: null, dea: null, hist: null };
                };
            }, []);

            // ─── Apply candle data + auto-fit on first load / range change ───
            useEffect(() => {
                const series = candleSeriesRef.current;
                const chart = chartRef.current;
                if (!series || !chart || candleData.length === 0) return;
                series.setData(candleData);
                try { chart.timeScale().fitContent(); } catch {}
            }, [candleData]);

            // ─── EMA Ribbon (6 lines on main pane) — per-segment color flip ───
            // Each EMA line is split into bull / bear segments at every fast×slow
            // crossover; each segment is its own LineSeries so the ribbon recolors
            // exactly at the cross instead of being one flat color for the whole
            // history. Segments overlap by one point at the boundary so the line
            // doesn't visually break at the crossover bar.
            useEffect(() => {
                const chart = chartRef.current;
                if (!chart) return;
                emaSeriesRef.current.forEach(s => { try { chart.removeSeries(s); } catch {} });
                emaSeriesRef.current = [];
                if (!showEMA || candleData.length === 0) return;
                const closes = candleData.map(d => d.close);
                const emas = EMA_PERIODS.map(p => computeEMA(closes, p));
                const fastArr = emas[0];
                const slowArr = emas[EMA_PERIODS.length - 1];

                EMA_PERIODS.forEach((period, idx) => {
                    const ema = emas[idx];
                    const lineWidth = (idx === 0 || idx === EMA_PERIODS.length - 1) ? 2 : 1;
                    let curBull = null;
                    let curPoints = [];

                    const flushSegment = () => {
                        if (curPoints.length < 2 || curBull == null) return;
                        const s = chart.addSeries(window.LightweightCharts.LineSeries, {
                            color: curBull ? EMA_BULL_COLOR : EMA_BEAR_COLOR,
                            lineWidth,
                            priceLineVisible: false,
                            lastValueVisible: false,
                            crosshairMarkerVisible: false,
                        }, 0);
                        s.setData(curPoints);
                        emaSeriesRef.current.push(s);
                    };

                    for (let i = 0; i < ema.length; i++) {
                        if (ema[i] == null || fastArr[i] == null || slowArr[i] == null) continue;
                        const bull = fastArr[i] >= slowArr[i];
                        const point = { time: candleData[i].time, value: ema[i] };
                        if (curBull == null) {
                            curBull = bull;
                            curPoints = [point];
                        } else if (bull === curBull) {
                            curPoints.push(point);
                        } else {
                            // Regime change: include the crossover point as the end of
                            // this segment AND the start of the next so they meet visually.
                            curPoints.push(point);
                            flushSegment();
                            curBull = bull;
                            curPoints = [point];
                        }
                    }
                    flushSegment();
                });
            }, [showEMA, candleData]);

            // EMA fast×slow cross markers, computed once and reused by the combined-markers effect
            const emaCrossMarkers = useMemo(() => {
                if (!showEMA || candleData.length === 0) return [];
                const closes = candleData.map(d => d.close);
                const fast = computeEMA(closes, EMA_FAST);
                const slow = computeEMA(closes, EMA_SLOW);
                const out = [];
                for (let i = 1; i < candleData.length; i++) {
                    const fPrev = fast[i - 1], fCur = fast[i];
                    const sPrev = slow[i - 1], sCur = slow[i];
                    if (fPrev == null || fCur == null || sPrev == null || sCur == null) continue;
                    if (fPrev <= sPrev && fCur > sCur) {
                        out.push({ time: candleData[i].time, position: 'belowBar', shape: 'arrowUp', color: UP_COLOR });
                    } else if (fPrev >= sPrev && fCur < sCur) {
                        out.push({ time: candleData[i].time, position: 'aboveBar', shape: 'arrowDown', color: DOWN_COLOR });
                    }
                }
                return out;
            }, [showEMA, candleData]);

            // ─── 200-period SMA on main pane (price > MA = bull → green, else red) ───
            useEffect(() => {
                const chart = chartRef.current;
                if (!chart) return;
                if (ma200SeriesRef.current) { try { chart.removeSeries(ma200SeriesRef.current); } catch {} ma200SeriesRef.current = null; }
                if (!showMA200 || candleData.length < 50) return;
                const period = 200;
                const closes = candleData.map(d => d.close);
                if (closes.length < period) return;
                const data = [];
                let sum = 0;
                for (let i = 0; i < closes.length; i++) {
                    sum += closes[i];
                    if (i >= period) sum -= closes[i - period];
                    if (i >= period - 1) data.push({ time: candleData[i].time, value: sum / period });
                }
                if (!data.length) return;
                const lastPrice = closes[closes.length - 1];
                const lastMa = data[data.length - 1].value;
                const bull = lastPrice >= lastMa;
                const s = chart.addSeries(window.LightweightCharts.LineSeries, {
                    color: bull ? UP_COLOR : DOWN_COLOR,
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'MA200',
                }, 0);
                s.setData(data);
                ma200SeriesRef.current = s;
            }, [showMA200, candleData]);

            // ─── RSI (pane 1) with 70 / 50 / 30 reference lines + cross markers ───
            // LONG marker: RSI crosses UP through 30 (oversold rebound)
            // SHORT marker: RSI crosses DOWN through 70 (overbought reversal)
            useEffect(() => {
                const chart = chartRef.current;
                const LWC = window.LightweightCharts;
                if (!chart) return;
                if (rsiSeriesRef.current) { try { chart.removeSeries(rsiSeriesRef.current); } catch {} rsiSeriesRef.current = null; }
                rsiMarkersRef.current = null;
                if (!showRSI || candleData.length === 0) return;
                const period = 14;
                const closes = candleData.map(d => d.close);
                if (closes.length <= period) return;
                let gains = 0, losses = 0;
                for (let i = 1; i <= period; i++) {
                    const d = closes[i] - closes[i - 1];
                    if (d > 0) gains += d; else losses -= d;
                }
                let avgGain = gains / period;
                let avgLoss = losses / period;
                const out = [];
                const pushRsi = (idx) => {
                    const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
                    const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + rs);
                    out.push({ time: candleData[idx].time, value: rsi });
                };
                pushRsi(period);
                for (let i = period + 1; i < closes.length; i++) {
                    const d = closes[i] - closes[i - 1];
                    const gain = d > 0 ? d : 0;
                    const loss = d < 0 ? -d : 0;
                    avgGain = (avgGain * (period - 1) + gain) / period;
                    avgLoss = (avgLoss * (period - 1) + loss) / period;
                    pushRsi(i);
                }
                const s = chart.addSeries(LWC.LineSeries, {
                    color: '#a78bfa',  // purple — bull / bear is conveyed by the cross markers, not the line
                    lineWidth: 1.5,
                    priceLineVisible: false, lastValueVisible: true, title: 'RSI(14)',
                }, 1);
                s.setData(out);
                try { s.createPriceLine({ price: 70, color: 'rgba(255,91,110,0.5)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false }); } catch {}
                try { s.createPriceLine({ price: 50, color: 'rgba(255,255,255,0.18)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false }); } catch {}
                try { s.createPriceLine({ price: 30, color: 'rgba(0,214,143,0.5)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false }); } catch {}
                rsiSeriesRef.current = s;

                // Cross markers on the RSI line itself
                if (typeof LWC.createSeriesMarkers === 'function') {
                    const markers = [];
                    for (let i = 1; i < out.length; i++) {
                        const prev = out[i - 1].value, cur = out[i].value;
                        if (prev <= 30 && cur > 30) {
                            markers.push({ time: out[i].time, position: 'belowBar', shape: 'arrowUp', color: UP_COLOR });
                        } else if (prev >= 70 && cur < 70) {
                            markers.push({ time: out[i].time, position: 'aboveBar', shape: 'arrowDown', color: DOWN_COLOR });
                        }
                    }
                    try { rsiMarkersRef.current = LWC.createSeriesMarkers(s, markers); } catch {}
                }
            }, [showRSI, candleData]);

            // ─── MACD (pane 2) + DIF×DEA golden / death cross markers on DIF line ───
            useEffect(() => {
                const chart = chartRef.current;
                const LWC = window.LightweightCharts;
                if (!chart) return;
                ['dif', 'dea', 'hist'].forEach(k => {
                    if (macdRefs.current[k]) { try { chart.removeSeries(macdRefs.current[k]); } catch {} macdRefs.current[k] = null; }
                });
                macdMarkersRef.current = null;
                if (!showMACD || candleData.length === 0) return;
                const closes = candleData.map(d => d.close);
                const macd = computeMACD(closes);
                const histData = [];
                const difData = [];
                const deaData = [];
                for (let i = 0; i < closes.length; i++) {
                    const t = candleData[i].time;
                    if (macd.histogram[i] != null) histData.push({ time: t, value: macd.histogram[i], color: macd.histogram[i] >= 0 ? 'rgba(0,214,143,0.75)' : 'rgba(255,91,110,0.75)' });
                    if (macd.dif[i] != null) difData.push({ time: t, value: macd.dif[i] });
                    if (macd.dea[i] != null) deaData.push({ time: t, value: macd.dea[i] });
                }
                macdRefs.current.hist = chart.addSeries(LWC.HistogramSeries, { priceLineVisible: false, lastValueVisible: false }, 2);
                macdRefs.current.hist.setData(histData);
                macdRefs.current.dif = chart.addSeries(LWC.LineSeries, { color: '#3b82f6', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, title: 'DIF' }, 2);
                macdRefs.current.dif.setData(difData);
                macdRefs.current.dea = chart.addSeries(LWC.LineSeries, { color: '#fbbf24', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, title: 'DEA' }, 2);
                macdRefs.current.dea.setData(deaData);

                // Cross markers anchored on the DIF line
                if (typeof LWC.createSeriesMarkers === 'function') {
                    const markers = [];
                    for (let i = 1; i < closes.length; i++) {
                        const dPrev = macd.dif[i - 1], dCur = macd.dif[i];
                        const ePrev = macd.dea[i - 1], eCur = macd.dea[i];
                        if (dPrev == null || dCur == null || ePrev == null || eCur == null) continue;
                        if (dPrev <= ePrev && dCur > eCur) {
                            markers.push({ time: candleData[i].time, position: 'belowBar', shape: 'arrowUp', color: UP_COLOR });
                        } else if (dPrev >= ePrev && dCur < eCur) {
                            markers.push({ time: candleData[i].time, position: 'aboveBar', shape: 'arrowDown', color: DOWN_COLOR });
                        }
                    }
                    try { macdMarkersRef.current = LWC.createSeriesMarkers(macdRefs.current.dif, markers); } catch {}
                }
            }, [showMACD, candleData]);

            // FNG dot markers (crypto / US only) — combined later with EMA cross markers
            const fngMarkers = useMemo(() => {
                if (!showFearSignal || signalSource !== 'fng' || candleData.length === 0 || fngHistory.length === 0) return [];
                const sortedDates = Object.keys(fngByDate).sort();
                const markers = [];
                candleData.forEach(bar => {
                    const dateStr = new Date(bar.time * 1000).toISOString().slice(0, 10);
                    let v = fngByDate[dateStr];
                    if (v === undefined) {
                        let lo = 0, hi = sortedDates.length - 1, best = -1;
                        while (lo <= hi) {
                            const mid = (lo + hi) >> 1;
                            if (sortedDates[mid] <= dateStr) { best = mid; lo = mid + 1; } else hi = mid - 1;
                        }
                        v = best >= 0 ? fngByDate[sortedDates[best]] : null;
                    }
                    if (v == null || v > 45) return;
                    markers.push({
                        time: bar.time,
                        position: 'belowBar',
                        color: v <= 25 ? DOWN_COLOR : '#f59e0b',
                        shape: 'circle',
                        size: v <= 25 ? 1 : 0.7,
                    });
                });
                return markers;
            }, [candleData, fngByDate, fngHistory, showFearSignal, signalSource]);

            // Combined candle-pane markers: FNG dots + EMA LONG / SHORT cross arrows, sorted by time
            useEffect(() => {
                const plugin = markersPluginRef.current;
                if (!plugin) return;
                const combined = [...fngMarkers, ...emaCrossMarkers].sort((a, b) => a.time - b.time);
                try { plugin.setMarkers(combined); } catch {}
            }, [fngMarkers, emaCrossMarkers]);

            // ─── Aggregate every signal we display into the summary panel ───
            const signalSummary = useMemo(() => {
                const out = [];
                if (candleData.length === 0) return out;
                const closes = candleData.map(d => d.close);
                const lastPrice = closes[closes.length - 1];

                // FNG (only crypto / US carry a real FNG)
                if (signalSource === 'fng' && fngHistory.length) {
                    const last = fngHistory[fngHistory.length - 1];
                    const cls = fngClassify(last.fng);
                    if (cls) out.push({ key: 'fng', name: 'FNG', tone: cls.tone, label: `${last.fng} · ${cls.label}` });
                }

                // Composite — same logic as the watchlist (RSI + 1Y position)
                const rsi = computeRSI(closes, 14);
                if (showComposite && rsi != null) {
                    const high = Math.max(...closes);
                    const low = Math.min(...closes);
                    const pos = high - low > 0 ? (lastPrice - low) / (high - low) : 0.5;
                    const comp = getCompositeSignal(rsi, pos);
                    const tone = /加碼/.test(comp.label) ? 'bull' : /賣出/.test(comp.label) ? 'bear' : 'neutral';
                    out.push({ key: 'composite', name: '綜合', tone, label: `${comp.emoji || ''} ${comp.label}`.trim() });
                }

                // EMA Ribbon (fast 20 vs slow 50)
                if (showEMA) {
                    const f = computeEMA(closes, EMA_PERIODS[0]);
                    const s = computeEMA(closes, EMA_PERIODS[EMA_PERIODS.length - 1]);
                    const lf = f[f.length - 1], ls = s[s.length - 1];
                    if (lf != null && ls != null) {
                        const bull = lf >= ls;
                        out.push({ key: 'ema', name: 'EMA', tone: bull ? 'bull' : 'bear', label: bull ? '多頭排列' : '空頭排列' });
                    }
                }

                // RSI bucket
                if (showRSI && rsi != null) {
                    let tone, label;
                    if (rsi >= 70) { tone = 'bear'; label = `${Math.round(rsi)} 超買`; }
                    else if (rsi <= 30) { tone = 'bull'; label = `${Math.round(rsi)} 超賣`; }
                    else if (rsi >= 50) { tone = 'bull'; label = `${Math.round(rsi)} 偏多`; }
                    else { tone = 'bear'; label = `${Math.round(rsi)} 偏空`; }
                    out.push({ key: 'rsi', name: 'RSI', tone, label });
                }

                // MACD: above signal line + positive histogram = bull
                if (showMACD) {
                    const m = computeMACD(closes);
                    const lDif = m.dif[m.dif.length - 1];
                    const lDea = m.dea[m.dea.length - 1];
                    const lHist = m.histogram[m.histogram.length - 1];
                    if (lDif != null && lDea != null && lHist != null) {
                        const above = lDif > lDea;
                        const positive = lHist > 0;
                        const tone = above && positive ? 'bull' : (!above && !positive ? 'bear' : 'neutral');
                        const label = tone === 'bull' ? '多' : tone === 'bear' ? '空' : '震盪';
                        out.push({ key: 'macd', name: 'MACD', tone, label });
                    }
                }

                // 200MA
                if (showMA200 && closes.length >= 200) {
                    let sum = 0;
                    for (let i = closes.length - 200; i < closes.length; i++) sum += closes[i];
                    const ma200 = sum / 200;
                    const bull = lastPrice >= ma200;
                    out.push({ key: 'ma200', name: 'MA200', tone: bull ? 'bull' : 'bear', label: bull ? '價格站上' : '價格跌破' });
                }

                return out;
            }, [candleData, fngHistory, signalSource, showEMA, showRSI, showMACD, showMA200, showComposite]);

            // ─── Tool picker handlers ───
            const pickTool = (tool) => {
                pendingAnchorsRef.current = [];
                activeToolRef.current = { type: tool.type, requiredAnchors: tool.a };
                setActiveTool({ type: tool.type, name: tool.name, requiredAnchors: tool.a, collected: 0 });
                setOpenCategory(null);
                if (containerRef.current) containerRef.current.style.cursor = 'crosshair';
            };
            const cancelTool = () => {
                activeToolRef.current = null;
                pendingAnchorsRef.current = [];
                setActiveTool(null);
                if (containerRef.current) containerRef.current.style.cursor = '';
            };
            const clearAllDrawings = () => {
                const m = drawingManagerRef.current;
                if (!m) return;
                if (!confirm('清除所有畫線標註？')) return;
                m.clearAll();
                drawingsRef.current = [];
                selectedDrawingIdRef.current = null;
                setHasSelection(false);
            };
            const deleteSelected = () => {
                const m = drawingManagerRef.current;
                const id = selectedDrawingIdRef.current;
                if (!m || !id) return;
                m.removeDrawing(id);
                drawingsRef.current = drawingsRef.current.filter(x => x !== id);
                selectedDrawingIdRef.current = null;
                setHasSelection(false);
            };
            const resetZoom = () => { try { chartRef.current && chartRef.current.timeScale().fitContent(); } catch {} };

            // ─── UI atoms ───
            const EyeToggle = ({ label, checked, onChange, color }) => (
                <button
                    onClick={() => onChange(!checked)}
                    title={label}
                    className="flex items-center gap-1.5 px-2 h-8 rounded-lg hover:bg-white/[0.06] transition-colors shrink-0"
                    style={{ border: '1px solid var(--line)', background: checked ? 'rgba(255,255,255,0.04)' : 'transparent' }}
                >
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: color, opacity: checked ? 1 : 0.3 }}></span>
                    <span className="text-[11px] font-semibold whitespace-nowrap" style={{ color: checked ? 'var(--text)' : 'var(--text-3)' }}>{label}</span>
                </button>
            );

            const rangeBtn = (key, label) => (
                <button
                    key={key}
                    onClick={() => setTimeRange(key)}
                    className={`px-2.5 h-7 rounded-md text-[11px] font-bold transition-all shrink-0 ${timeRange === key ? 'pill-grad' : 'hover:bg-white/[0.08]'}`}
                    style={timeRange === key ? { color: 'var(--brand-ink)' } : { color: 'var(--text-2)' }}
                >{label}</button>
            );

            const CategoryIcon = ({ k }) => {
                const stroke = 'currentColor';
                const common = { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'none', stroke, strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round' };
                if (k === 'line') return <svg {...common}><line x1="4" y1="20" x2="20" y2="4"/></svg>;
                if (k === 'channel') return <svg {...common}><line x1="4" y1="18" x2="20" y2="6"/><line x1="4" y1="14" x2="20" y2="2"/></svg>;
                if (k === 'pitchfork') return <svg {...common}><line x1="4" y1="20" x2="20" y2="4"/><line x1="8" y1="20" x2="20" y2="8"/><line x1="12" y1="20" x2="20" y2="12"/></svg>;
                if (k === 'fibonacci') return <svg {...common}><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/><line x1="3" y1="14" x2="21" y2="14"/><line x1="3" y1="18" x2="21" y2="18"/></svg>;
                if (k === 'gann') return <svg {...common}><rect x="4" y="4" width="16" height="16"/><line x1="4" y1="4" x2="20" y2="20"/></svg>;
                if (k === 'forecasting') return <svg {...common}><path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/></svg>;
                if (k === 'shape') return <svg {...common}><rect x="4" y="4" width="16" height="16" rx="1"/></svg>;
                if (k === 'annotation') return <svg {...common}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>;
                return null;
            };

            // tone → pill colors
            const tonePill = (tone) => {
                if (tone === 'bull') return { bg: 'rgba(0,214,143,0.15)', color: UP_COLOR, border: 'rgba(0,214,143,0.4)' };
                if (tone === 'bear') return { bg: 'rgba(255,91,110,0.15)', color: DOWN_COLOR, border: 'rgba(255,91,110,0.4)' };
                return { bg: 'rgba(255,255,255,0.06)', color: 'var(--text-2)', border: 'var(--line)' };
            };

            // ─── Layout ───
            return (
                <div className="fixed inset-0 z-[110] flex flex-col" style={{ background: 'var(--bg)', height: '100dvh' }}>
                    {/* Top bar */}
                    <div className="shrink-0 flex flex-col gap-2 px-3 md:px-4 pt-3 pb-2" style={{ borderBottom: '1px solid var(--line)' }}>
                        {/* Row 1: title + status + help/reset/close */}
                        <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2 min-w-0">
                                <button onClick={onClose} className="w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 shrink-0" style={{ color: 'var(--text-2)' }} title="關閉">✕</button>
                                <div className="min-w-0">
                                    <p className="label leading-none">TECHNICAL</p>
                                    <h2 className="text-base md:text-lg font-extrabold text-white tracking-tight truncate">{symbol}</h2>
                                </div>
                                {loading && (
                                    <div className="hidden sm:flex items-center gap-1.5 text-[10px] ml-2" style={{ color: 'var(--text-3)' }}>
                                        <RefreshCw className="animate-spin" size={11} />
                                        載入中
                                    </div>
                                )}
                                {error && <div className="text-[10px] ml-2 truncate" style={{ color: '#ff7d8c' }}>{error}</div>}
                                {activeTool && (
                                    <div className="hidden md:flex items-center gap-1.5 text-[10px] ml-2 px-2 py-0.5 rounded-md" style={{ background: 'rgba(59,130,246,0.15)', color: 'var(--brand-1)', border: '1px solid var(--brand-1)' }}>
                                        繪製：{activeTool.name}（{activeTool.collected ?? 0}/{activeTool.requiredAnchors}）
                                        <button onClick={cancelTool} className="ml-1 hover:underline">取消</button>
                                    </div>
                                )}
                            </div>
                            <div className="flex items-center gap-1 shrink-0">
                                <button onClick={() => setShowSignalPanel(!showSignalPanel)}
                                    className="h-8 px-2 rounded-md text-[10px] mono hover:bg-white/[0.08] hidden sm:inline-block"
                                    style={{ color: showSignalPanel ? 'var(--brand-1)' : 'var(--text-2)', border: '1px solid ' + (showSignalPanel ? 'var(--brand-1)' : 'var(--line)') }}
                                    title="切換信號面板">
                                    📊 信號
                                </button>
                                <button onClick={resetZoom}
                                    className="h-8 px-2 rounded-md text-[10px] mono hover:bg-white/[0.08]"
                                    style={{ color: 'var(--text-2)', border: '1px solid var(--line)' }} title="重置縮放">⤾ 重置</button>
                                <button onClick={() => setShowHelp(true)}
                                    className="h-8 w-8 rounded-md flex items-center justify-center hover:bg-white/[0.08]"
                                    style={{ color: 'var(--text-2)', border: '1px solid var(--line)' }} title="說明">
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                                    </svg>
                                </button>
                            </div>
                        </div>

                        {/* Row 2: range • timeframe • indicator toggles (horizontally scrollable) */}
                        <div className="flex items-center gap-2 overflow-x-auto no-scrollbar -mx-1 px-1" style={{ scrollbarWidth: 'none' }}>
                            <div className="flex items-center gap-0.5 p-0.5 rounded-lg shrink-0" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--line)' }}>
                                {rangeBtn('1mo', '1M')}
                                {rangeBtn('3mo', '3M')}
                                {rangeBtn('6mo', '6M')}
                                {rangeBtn('1y', '1Y')}
                                {rangeBtn('max', 'ALL')}
                            </div>

                            <div className="flex items-center gap-0.5 p-0.5 rounded-lg shrink-0" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--line)' }} title="K 線週期">
                                <button onClick={() => setTimeframe('1d')}
                                    className={`px-2.5 h-7 rounded-md text-[11px] font-bold ${timeframe === '1d' ? 'pill-grad' : 'hover:bg-white/[0.06]'}`}
                                    style={timeframe === '1d' ? { color: 'var(--brand-ink)' } : { color: 'var(--text-2)' }}>日</button>
                                <button onClick={() => setTimeframe('1wk')}
                                    className={`px-2.5 h-7 rounded-md text-[11px] font-bold ${timeframe === '1wk' ? 'pill-grad' : 'hover:bg-white/[0.06]'}`}
                                    style={timeframe === '1wk' ? { color: 'var(--brand-ink)' } : { color: 'var(--text-2)' }}>週</button>
                            </div>

                            <div className="w-px h-6 shrink-0" style={{ background: 'var(--line)' }}></div>

                            {signalSource === 'fng' && (
                                <EyeToggle label="FNG 信號" checked={showFearSignal} onChange={setShowFearSignal} color={DOWN_COLOR} />
                            )}
                            <EyeToggle label="EMA Ribbon" checked={showEMA} onChange={setShowEMA} color="#3b82f6" />
                            <EyeToggle label="MA200" checked={showMA200} onChange={setShowMA200} color="#f97316" />
                            <EyeToggle label="RSI" checked={showRSI} onChange={setShowRSI} color="#a78bfa" />
                            <EyeToggle label="MACD" checked={showMACD} onChange={setShowMACD} color="#fbbf24" />
                            <EyeToggle label="綜合信號" checked={showComposite} onChange={setShowComposite} color={UP_COLOR} />
                        </div>
                    </div>

                    {/* Body: drawing toolbar + chart */}
                    <div className="flex-1 min-h-0 flex">
                        {/* Vertical toolbar (left). Categories pop a submenu of tools. */}
                        <div className="shrink-0 flex flex-col items-stretch gap-1 py-2 px-1.5" style={{ borderRight: '1px solid var(--line)', width: 48, background: 'rgba(255,255,255,0.02)' }}>
                            {DRAWING_TOOL_CATEGORIES.map(cat => (
                                <div key={cat.key} className="relative">
                                    <button
                                        onClick={() => setOpenCategory(openCategory === cat.key ? null : cat.key)}
                                        className="w-9 h-9 rounded-md flex items-center justify-center hover:bg-white/[0.08] transition-colors"
                                        style={{
                                            color: openCategory === cat.key ? 'var(--brand-1)' : 'var(--text-2)',
                                            background: openCategory === cat.key ? 'rgba(59,130,246,0.12)' : 'transparent',
                                            border: '1px solid ' + (openCategory === cat.key ? 'var(--brand-1)' : 'transparent'),
                                        }}
                                        title={cat.label}
                                    >
                                        <CategoryIcon k={cat.key} />
                                    </button>
                                </div>
                            ))}

                            <div className="my-1 h-px" style={{ background: 'var(--line)' }}></div>

                            <button
                                onClick={deleteSelected}
                                disabled={!hasSelection}
                                className="w-9 h-9 rounded-md flex items-center justify-center hover:bg-white/[0.08] transition-colors disabled:opacity-30 disabled:hover:bg-transparent"
                                style={{ color: hasSelection ? DOWN_COLOR : 'var(--text-3)' }}
                                title="刪除選取的標註（或按 Delete 鍵）"
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                </svg>
                            </button>
                            <button
                                onClick={clearAllDrawings}
                                className="w-9 h-9 rounded-md flex items-center justify-center hover:bg-white/[0.08] transition-colors"
                                style={{ color: 'var(--text-2)' }}
                                title="清除所有標註"
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>
                                </svg>
                            </button>
                        </div>

                        {/* Tool-category popup. Positioned absolute next to the sidebar but clamped
                            to the viewport so it never spills off a narrow phone screen. */}
                        {openCategory && (() => {
                            const cat = DRAWING_TOOL_CATEGORIES.find(c => c.key === openCategory);
                            if (!cat) return null;
                            return (
                                <>
                                    {/* Backdrop catches outside taps on mobile */}
                                    <div className="fixed inset-0 z-[115] md:hidden" onClick={() => setOpenCategory(null)}></div>
                                    <div
                                        className="fixed z-[116] rounded-lg shadow-2xl p-1 max-h-[60vh] overflow-y-auto"
                                        style={{
                                            background: 'var(--surface)',
                                            border: '1px solid var(--line)',
                                            top: 'calc(env(safe-area-inset-top, 0px) + 120px)',
                                            left: 56,
                                            width: 'min(220px, calc(100vw - 72px))',
                                        }}
                                    >
                                        <div className="text-[10px] uppercase tracking-wider px-2 py-1 mb-0.5 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
                                            <span>{cat.label}（{cat.tools.length}）</span>
                                            <button onClick={() => setOpenCategory(null)} className="md:hidden text-[11px] hover:text-white">✕</button>
                                        </div>
                                        {cat.tools.map(tool => (
                                            <button
                                                key={tool.type}
                                                onClick={() => pickTool(tool)}
                                                className="block w-full text-left px-2 py-1.5 rounded-md text-xs hover:bg-white/[0.06]"
                                                style={{ color: 'var(--text)' }}
                                            >
                                                {tool.name}
                                                <span className="ml-2 text-[10px]" style={{ color: 'var(--text-3)' }}>{tool.a} 點</span>
                                            </button>
                                        ))}
                                    </div>
                                </>
                            );
                        })()}

                        {/* Chart container */}
                        <div className="flex-1 min-w-0 p-1 md:p-2 relative">
                            <div ref={containerRef} className="w-full h-full" style={{ minHeight: '400px' }}></div>

                            {/* Signal summary panel — top-right, semi-transparent, collapsible */}
                            {showSignalPanel && signalSummary.length > 0 && (
                                <div
                                    className="absolute top-2 right-2 rounded-lg p-2 z-20 backdrop-blur"
                                    style={{
                                        background: 'rgba(18,20,28,0.85)',
                                        border: '1px solid var(--line)',
                                        width: 'min(180px, calc(100vw - 80px))',
                                        maxHeight: 'calc(100% - 16px)',
                                        overflowY: 'auto',
                                    }}
                                >
                                    <div className="flex items-center justify-between mb-1.5">
                                        <span className="text-[10px] font-bold tracking-wider uppercase" style={{ color: 'var(--text-3)' }}>信號總覽</span>
                                        <button onClick={() => setShowSignalPanel(false)} className="w-5 h-5 rounded hover:bg-white/[0.08] text-[11px]" style={{ color: 'var(--text-3)' }} title="收起">−</button>
                                    </div>
                                    <div className="space-y-1">
                                        {signalSummary.map(it => {
                                            const p = tonePill(it.tone);
                                            return (
                                                <div key={it.key} className="flex items-center justify-between text-[10px] gap-2">
                                                    <span style={{ color: 'var(--text-3)' }}>{it.name}</span>
                                                    <span className="px-1.5 py-0.5 rounded font-bold truncate" style={{ background: p.bg, color: p.color, border: '1px solid ' + p.border, maxWidth: 110 }}>{it.label}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                            {!showSignalPanel && (
                                <button
                                    onClick={() => setShowSignalPanel(true)}
                                    className="absolute top-2 right-2 px-2 py-1 rounded-lg text-[10px] font-bold z-20 backdrop-blur hover:bg-white/[0.08]"
                                    style={{ background: 'rgba(18,20,28,0.85)', border: '1px solid var(--line)', color: 'var(--text-2)' }}
                                    title="展開信號面板"
                                >📊 信號</button>
                            )}

                            {activeTool && (
                                <div className="md:hidden absolute top-2 left-2 right-[150px] px-3 py-2 rounded-lg flex items-center justify-between text-[11px] z-20" style={{ background: 'rgba(59,130,246,0.15)', color: 'var(--brand-1)', border: '1px solid var(--brand-1)' }}>
                                    <span className="truncate">繪製：{activeTool.name}（{activeTool.collected ?? 0}/{activeTool.requiredAnchors}）</span>
                                    <button onClick={cancelTool} className="hover:underline ml-2 shrink-0">取消</button>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Help overlay */}
                    {showHelp && (
                        <div className="fixed inset-0 z-[120] flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.6)' }} onClick={() => setShowHelp(false)}>
                            <div className="w-full max-w-sm rounded-2xl p-5 space-y-2" style={{ background: 'var(--surface)', border: '1px solid var(--line)' }} onClick={(e) => e.stopPropagation()}>
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="text-sm font-bold text-white">圖表說明</h3>
                                    <button onClick={() => setShowHelp(false)} className="w-7 h-7 rounded-full hover:bg-white/10" style={{ color: 'var(--text-2)' }}>✕</button>
                                </div>
                                <div className="text-[11px] leading-relaxed space-y-1.5" style={{ color: 'var(--text-2)' }}>
                                    <p><strong className="text-white">圖表</strong>：基於 TradingView Lightweight Charts v5；滑鼠拖曳平移、滾輪縮放、雙指捏合縮放；拖曳價格軸 / 時間軸縮放單一方向。</p>
                                    <p><strong className="text-white">畫線工具</strong>：左側工具列共 8 分類、68 種工具。點分類圖示打開選單，挑選工具後在圖表上任意位置點擊指定數量的錨點即完成（自由位置，不會吸到 K 棒）。</p>
                                    <p><strong className="text-white">選取 / 編輯 / 刪除</strong>：點擊已完成的標註可選取，拖曳錨點微調；按 Delete 或左側垃圾桶刪除；Esc 取消當前繪製。手機可用單指拖曳錨點。</p>
                                    <p><strong className="text-white">EMA Ribbon</strong>：6 條均線 20 / 27 / 34 / 41 / 48 / 55（fast=20、slow=55）。線段顏色逐段切換 — fast 站上 slow 該段顯綠、跌破則紅；K 棒下方綠 △ = 金叉（多），K 棒上方紅 ▽ = 死叉（空）。</p>
                                    <p><strong className="text-white">RSI / MACD / MA200</strong>：RSI 紫線恆色，加 70 / 50 / 30 參考線，上穿 30 標 △、下穿 70 標 ▽；MACD DIF×DEA 金叉 △、死叉 ▽；MA200 線色隨價格站上 / 跌破切換。</p>
                                    <p><strong className="text-white">信號總覽</strong>：右上角面板獨立列出各指標的多 / 空狀態，不做綜合判斷；可隨時收起。</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            );
        };
'''


def main() -> int:
    src = HTML.read_text(encoding="utf-8")
    if START_MARKER not in src:
        print("start marker not found", file=sys.stderr)
        return 1
    if END_MARKER not in src:
        print("end marker not found", file=sys.stderr)
        return 1
    head, _, rest = src.partition(START_MARKER)
    block_and_tail = START_MARKER + rest
    idx = block_and_tail.index(END_MARKER)
    tail = block_and_tail[idx:]
    out = head + NEW_BLOCK + tail
    HTML.write_text(out, encoding="utf-8")
    print("ok, new file size:", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
