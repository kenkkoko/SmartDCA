import React from 'react';
import ReactDOM from 'react-dom/client';
import { createClient } from '@supabase/supabase-js';
import economicCalendar from '../economic_calendar.json';

        const { useState, useEffect, useMemo, useRef } = React;

        // ─────────────────────────────────────────────
        // Register Service Worker on page load (required for PWA install)
        // ─────────────────────────────────────────────
        if ('serviceWorker' in navigator && window.location.protocol !== 'file:') {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('./sw.js')
                    .then(reg => console.log('[SW] registered:', reg.scope))
                    .catch(err => console.warn('[SW] registration failed:', err));
            });
        }


        // ─────────────────────────────────────────────
        // Theme (light / dark). index.html sets <html data-theme> before paint;
        // chart colours are read from CSS variables at the moment a chart is built.
        // ─────────────────────────────────────────────
        const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        const hexA = (hex, a) => {
            const h = hex.replace('#', '');
            const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
            return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
        };
        const CHART = {
            get line()      { return cssVar('--ink'); },
            get fillTop()   { return hexA(cssVar('--ink'), 0.06); },
            get fillBot()   { return hexA(cssVar('--ink'), 0); },
            get brand1()    { return cssVar('--accent'); },
            get brand2()    { return cssVar('--accent'); },
            get up()        { return cssVar('--up'); },
            get down()      { return cssVar('--down'); },
            get fear()      { return cssVar('--down'); },
            get fearLight() { return cssVar('--z1'); },
            get grid()      { return cssVar('--faint'); },
            get tick()      { return cssVar('--ink-3'); },
            get text()      { return cssVar('--ink-2'); },
            get tooltipBg() { return cssVar('--paper'); },
            get border()    { return cssVar('--rule'); },
        };
        const makePriceGradient = (context) => {
            const ctx = context.chart.ctx;
            const g = ctx.createLinearGradient(0, 0, 0, 400);
            g.addColorStop(0, CHART.fillTop);
            g.addColorStop(1, CHART.fillBot);
            return g;
        };
        const applyChartDefaults = () => {
            if (typeof window.Chart === 'undefined') return;
            const C = window.Chart;
            const font = "'Schibsted Grotesk', 'Noto Sans TC', ui-sans-serif, system-ui, sans-serif";
            C.defaults.color = CHART.tick;
            C.defaults.borderColor = CHART.grid;
            C.defaults.font.family = font;
            C.defaults.font.size = 11;
            C.defaults.font.weight = '500';
            C.defaults.plugins.tooltip.backgroundColor = CHART.tooltipBg;
            C.defaults.plugins.tooltip.titleColor = cssVar('--ink');
            C.defaults.plugins.tooltip.bodyColor = CHART.text;
            C.defaults.plugins.tooltip.borderColor = cssVar('--ink');
            C.defaults.plugins.tooltip.borderWidth = 1;
            C.defaults.plugins.tooltip.padding = 10;
            C.defaults.plugins.tooltip.cornerRadius = 0;
            C.defaults.plugins.tooltip.displayColors = false;
            C.defaults.plugins.tooltip.titleFont = { family: font, size: 12, weight: '700' };
            C.defaults.plugins.tooltip.bodyFont = { family: font, size: 12 };
            C.defaults.plugins.title.color = CHART.text;
            C.defaults.plugins.title.font = { family: font, size: 12, weight: '500' };
            C.defaults.plugins.title.padding = { bottom: 12 };
            C.defaults.plugins.legend.labels.color = CHART.text;
            C.defaults.elements.line.tension = 0.25;
            C.defaults.elements.line.borderWidth = 1.6;
            C.defaults.elements.point.hoverRadius = 5;
            C.defaults.elements.point.hoverBorderWidth = 2;
            C.defaults.elements.point.hoverBackgroundColor = cssVar('--paper');
        };
        applyChartDefaults();

        // Current theme + setter. Explicit choice is remembered; otherwise follow the system.
        const THEME_EVENT = 'smartdca-themechange';
        const useTheme = () => {
            const [theme, setThemeState] = useState(() => document.documentElement.getAttribute('data-theme') || 'light');
            const setTheme = (t) => {
                document.documentElement.setAttribute('data-theme', t);
                try { localStorage.setItem('theme', t); } catch (e) {}
                applyChartDefaults();
                setThemeState(t);
                window.dispatchEvent(new Event(THEME_EVENT));
            };
            useEffect(() => {
                const mq = window.matchMedia('(prefers-color-scheme: dark)');
                const onSystem = (e) => {
                    let saved = null;
                    try { saved = localStorage.getItem('theme'); } catch (err) {}
                    if (saved) return;
                    const t = e.matches ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', t);
                    applyChartDefaults();
                    setThemeState(t);
                    window.dispatchEvent(new Event(THEME_EVENT));
                };
                mq.addEventListener ? mq.addEventListener('change', onSystem) : mq.addListener(onSystem);
                return () => { mq.removeEventListener ? mq.removeEventListener('change', onSystem) : mq.removeListener(onSystem); };
            }, []);
            return [theme, setTheme];
        };
        // Re-render trigger for components whose chart options embed theme colours
        const useThemeVersion = () => {
            const [v, setV] = useState(0);
            useEffect(() => {
                const on = () => setV(x => x + 1);
                window.addEventListener(THEME_EVENT, on);
                return () => window.removeEventListener(THEME_EVENT, on);
            }, []);
            return v;
        };
        // Measured width of an element (for SVG drawn in real pixels). Returns [ref, width].
        // Re-binds after every render so it keeps measuring when a component swaps its root element.
        const useWidth = () => {
            const ref = useRef(null);
            const [w, setW] = useState(0);
            React.useLayoutEffect(() => {
                const node = ref.current;
                if (!node) return;
                const measure = () => setW(Math.round(node.getBoundingClientRect().width));
                measure();
                const ro = new ResizeObserver(measure);
                ro.observe(node);
                return () => ro.disconnect();
            });
            return [ref, w];
        };

        // --- Supabase Configuration ---
        // Local dev: reads from local-config.js (gitignored).
        // Production: placeholders are sed-replaced by GitHub Actions.
        const LOCAL_CFG = window.__LOCAL_CONFIG__ || null;
        const SUPABASE_URL = LOCAL_CFG?.SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL || '__SUPABASE_URL_PLACEHOLDER__';
        const SUPABASE_ANON_KEY = LOCAL_CFG?.SUPABASE_ANON_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY || '__SUPABASE_ANON_KEY_PLACEHOLDER__';
        const FORUM_ENABLED = LOCAL_CFG?.FORUM_ENABLED ?? true;
        // --- Demo 展示模式偵測(作品集 / 履歷用連結 ?demo=kai2026)---
        // 只有從這個特殊連結進來才啟用 demo;正式網址(無此參數)完全不受影響、不顯示任何 demo 元素。
        const DEMO = new URLSearchParams(location.search).get('demo') === 'kai2026';
        window.__DEMO__ = DEMO;
        const supabase = (!SUPABASE_URL.includes('PLACEHOLDER'))
            // demo 模式用「純記憶體 session」(persistSession:false):展示帳號的登入完全不寫進
            // localStorage,因此不污染正式站的登入狀態 —— 離開 demo、改用正式網址時仍是使用者自己的帳號。
            ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY,
                DEMO ? { auth: { persistSession: false, autoRefreshToken: true } } : undefined)
            : null;

        // --- Demo 展示模式:自動登入唯讀展示帳號(Premium),讓訪客免註冊即可體驗完整付費功能
        //     (含需伺服器端驗證會員身分的 AI 摘要與加密行情);唯讀攔截避免訪客操作污染資料。---
        if (DEMO && supabase) {
            const DEMO_EMAIL = 'test416@gmail.com';
            const DEMO_PASSWORD = 'test4166';
            // 唯讀:攔截資料表寫入(select 等讀取不受影響)
            const fakeRes = () => {
                const p = Promise.resolve({ data: [], error: null });
                ['select','eq','neq','gt','gte','lt','lte','like','ilike','is','in','contains','order','limit','range','match','single','maybeSingle'].forEach(m => { p[m] = () => p; });
                return p;
            };
            const _from = supabase.from.bind(supabase);
            supabase.from = (t) => {
                const qb = _from(t);
                ['insert','update','upsert','delete'].forEach(m => { qb[m] = fakeRes; });
                return qb;
            };
            if (supabase.storage) {
                const _sf = supabase.storage.from.bind(supabase.storage);
                supabase.storage.from = (b) => {
                    const so = _sf(b);
                    so.upload = async () => ({ data: null, error: { message: 'Demo 展示模式為唯讀,無法上傳。' } });
                    so.remove = async () => ({ data: null, error: null });
                    return so;
                };
            }
            // 自動登入展示帳號:persistSession:false 已確保是乾淨的記憶體 session
            // (不會撿到訪客 localStorage 裡的舊 token),所以直接登入即可,不需處理殘留。
            (async () => {
                try {
                    const { error } = await supabase.auth.signInWithPassword({ email: DEMO_EMAIL, password: DEMO_PASSWORD });
                    if (error) console.warn('[demo] auto-login failed:', error.message);
                } catch (e) { console.warn('[demo] auto-login error:', e && e.message); }
            })();
            // 展示橫幅
            const _showDemoTag = () => {
                if (!document.body) { setTimeout(_showDemoTag, 50); return; }
                const tag = document.createElement('div');
                tag.textContent = '👀 展示模式 Demo · 已解鎖所有 Premium 功能(唯讀)';
                tag.style.cssText = 'position:fixed;left:50%;bottom:calc(14px + env(safe-area-inset-bottom, 0px));transform:translateX(-50%);z-index:99999;background:rgba(20,20,24,.88);color:#fff;font-size:12px;font-weight:500;padding:7px 16px;border-radius:999px;border:1px solid rgba(255,255,255,.22);backdrop-filter:blur(8px);box-shadow:0 8px 24px rgba(0,0,0,.4);pointer-events:none;white-space:nowrap';
                document.body.appendChild(tag);
            };
            _showDemoTag();
        }

        // --- 內建 Icons (取代外部 lucide-react 依賴) ---
        const IconBase = ({ children, size = 24, className = "" }) => (
            <svg
                xmlns="http://www.w3.org/2000/svg"
                width={size}
                height={size}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={className}
            >
                {children}
            </svg>
        );

        const Icons = {
            Bell: (props) => (
                <IconBase {...props}>
                    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
                    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
                </IconBase>
            ),
            TrendingUp: (props) => (
                <IconBase {...props}>
                    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                    <polyline points="17 6 23 6 23 12" />
                </IconBase>
            ),
            AlertTriangle: (props) => (
                <IconBase {...props}>
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                </IconBase>
            ),
            Info: (props) => (
                <IconBase {...props}>
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="16" x2="12" y2="12" />
                    <line x1="12" y1="8" x2="12.01" y2="8" />
                </IconBase>
            ),
            RefreshCw: (props) => (
                <IconBase {...props}>
                    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                    <path d="M21 3v5h-5" />
                    <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                    <path d="M8 16H3v5" />
                </IconBase>
            ),
            Smartphone: (props) => (
                <IconBase {...props}>
                    <rect width="14" height="20" x="5" y="2" rx="2" ry="2" />
                    <path d="M12 18h.01" />
                </IconBase>
            ),
            ArrowUp: (props) => (
                <IconBase {...props}>
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                </IconBase>
            ),
            ArrowDown: (props) => (
                <IconBase {...props}>
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <polyline points="19 12 12 19 5 12" />
                </IconBase>
            ),
            Newspaper: (props) => (
                <IconBase {...props}>
                    <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2" />
                    <path d="M18 14h-8" />
                    <path d="M15 18h-5" />
                    <path d="M10 6h8v4h-8V6Z" />
                </IconBase>
            ),
            Sparkles: (props) => (
                <IconBase {...props}>
                    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
                </IconBase>
            ),
            ExternalLink: (props) => (
                <IconBase {...props}>
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    <polyline points="15 3 21 3 21 9" />
                    <line x1="10" y1="14" x2="21" y2="3" />
                </IconBase>
            ),
            Bot: (props) => (
                <IconBase {...props}>
                    <path d="M12 8V4H8" />
                    <rect width="16" height="12" x="4" y="8" rx="2" />
                    <path d="M2 14h2" />
                    <path d="M20 14h2" />
                    <path d="M15 13v2" />
                    <path d="M9 13v2" />
                </IconBase>
            ),
            Globe: (props) => (
                <IconBase {...props}>
                    <circle cx="12" cy="12" r="10" />
                    <line x1="2" y1="12" x2="22" y2="12" />
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                </IconBase>
            ),
            Activity: (props) => (
                <IconBase {...props}>
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </IconBase>
            ),
            BellRing: (props) => (
                <IconBase {...props}>
                    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
                    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
                    <path d="M4 2C2.8 3.7 2 5.7 2 8" />
                    <path d="M22 8c0-2.3-.8-4.3-2-6" />
                </IconBase>
            ),
            BellOff: (props) => (
                <IconBase {...props}>
                    <path d="M8.7 3A6 6 0 0 1 18 8a21.3 21.3 0 0 0 .6 5" />
                    <path d="M17 17H3s3-2 3-9a4.67 4.67 0 0 1 .3-1.7" />
                    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
                    <path d="M2 2l20 20" />
                </IconBase>
            ),
            LogOut: (props) => (
                <IconBase {...props}>
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16 17 21 12 16 7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                </IconBase>
            ),
            User: (props) => (
                <IconBase {...props}>
                    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                </IconBase>
            ),
            Lock: (props) => (
                <IconBase {...props}>
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </IconBase>
            ),
            CalendarDays: (props) => (
                <IconBase {...props}>
                    <rect width="18" height="18" x="3" y="4" rx="2" />
                    <path d="M16 2v4" />
                    <path d="M8 2v4" />
                    <path d="M3 10h18" />
                    <path d="M8 14h.01" />
                    <path d="M12 14h.01" />
                    <path d="M16 14h.01" />
                    <path d="M8 18h.01" />
                    <path d="M12 18h.01" />
                </IconBase>
            )
        };

        const { Bell, TrendingUp, AlertTriangle, Info, RefreshCw, Smartphone, ArrowUp, ArrowDown, Newspaper, Sparkles, ExternalLink, Bot, Globe, Activity, BellRing, BellOff, LogOut, User, Lock, CalendarDays } = Icons;

        // --- Auth Component ---
        // Logo mark "每月一格": a 4x4 month grid. Filled dots = periods invested,
        // the ultramarine square = this period, hollow dots = periods to come.
        // Same geometry as py/gen_logo.py (the raster icons).
        const LogoMark = ({ size = 28, className = '' }) => (
            <svg width={size} height={size} viewBox="0 0 64 64" className={className} aria-hidden="true" style={{ flex: 'none' }}>
                {Array.from({ length: 16 }, (_, i) => {
                    const r = Math.floor(i / 4), c = i % 4, x = 12 + c * 13, y = 12 + r * 13;
                    if (i === 13) return <rect key={i} x={x - 6} y={y - 6} width="12" height="12" style={{ fill: 'var(--accent)' }} />;
                    return i < 13
                        ? <circle key={i} cx={x} cy={y} r="4.2" style={{ fill: 'var(--ink)' }} />
                        : <circle key={i} cx={x} cy={y} r="3.4" fill="none" style={{ stroke: 'var(--ink)', strokeWidth: 1.6 }} />;
                })}
            </svg>
        );

        const AuthComponent = ({ user, setUser, isPremium, onOpenSettings, userProfile }) => {
            const [localLoading, setLocalLoading] = useState(false);
            const [showLoginModal, setShowLoginModal] = useState(false);
            const [showMenu, setShowMenu] = useState(false);
            const [email, setEmail] = useState('');
            const [password, setPassword] = useState('');
            const [isSignUp, setIsSignUp] = useState(false);
            const [confirmPassword, setConfirmPassword] = useState('');
            const [msg, setMsg] = useState(null);

            const handleLogin = async (e) => {
                e.preventDefault();
                if (!supabase) {
                    alert("請先設定 Supabase URL 和 Key!");
                    return;
                }
                if (isSignUp && password !== confirmPassword) {
                    setMsg('密碼不一致，請重新確認。');
                    return;
                }
                setLocalLoading(true);
                setMsg(null);
                try {
                    const { data, error } = isSignUp
                        ? await supabase.auth.signUp({ email, password })
                        : await supabase.auth.signInWithPassword({ email, password });

                    if (error) throw error;

                    if (isSignUp && !data.session) {
                        setMsg("註冊成功！請檢查信箱驗證信。");
                    } else if (data.session) {
                        setUser(data.user);
                        setShowLoginModal(false);
                    }
                } catch (error) {
                    setMsg(error.message);
                } finally {
                    setLocalLoading(false);
                }
            };

            const handleGoogleLogin = async () => {
                if (!supabase) return;
                try {
                    const { data, error } = await supabase.auth.signInWithOAuth({
                        provider: 'google',
                        options: {
                            redirectTo: window.location.href.split('#')[0] // Ensure we redirect to the clean URL
                        }
                    });
                    if (error) throw error;
                } catch (error) {
                    setMsg(error.message);
                }
            };

            const handleLogout = async () => {
                if (!supabase) return;
                if (window.__DEMO__) {
                    // 展示模式:「離開展示」→ 導回正式站(去掉 ?demo 參數)。
                    // 展示帳號是 persistSession:false 的記憶體 session,離開即消失、不殘留。
                    window.location.href = window.location.origin + window.location.pathname;
                    return;
                }
                await supabase.auth.signOut();
                setUser(null);
                window.location.reload(); // Reload to clear states
            };

            if (user) {
                const avatarUrl = user.user_metadata?.avatar_url || user.user_metadata?.picture;
                const initial = (user.email || 'U').charAt(0).toUpperCase();
                const displayName = (userProfile && userProfile.display_name) ? userProfile.display_name : user.email.split('@')[0];
                return (
                    <div className="relative">
                        <button
                            onClick={() => setShowMenu(v => !v)}
                            className="glass rounded-full pl-1 pr-2 py-1 flex items-center gap-2 hover:bg-white/[0.08] transition-colors cursor-pointer"
                            title="開啟選單"
                        >
                            {avatarUrl ? (
                                <img src={avatarUrl} alt="User Avatar" className="w-7 h-7 rounded-full ring-soft" />
                            ) : (
                                <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold pill-grad" style={{ color: 'var(--brand-ink)' }}>{initial}</div>
                            )}
                            <span className="text-xs font-semibold hidden md:block" style={{ color: 'var(--text-2)' }}>{displayName}</span>
                            {/* PRO badge inside pill */}
                            {isPremium && (
                                <span className="fs-chip ml-1" style={{ color: 'var(--ink)' }}>
                                    PRO
                                </span>
                            )}
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-3)' }}>
                                <polyline points="6 9 12 15 18 9"></polyline>
                            </svg>
                        </button>

                        {/* Backdrop to catch outside clicks */}
                        {showMenu && (
                            <div onClick={() => setShowMenu(false)} className="fixed inset-0 z-[90]"></div>
                        )}

                        {/* Dropdown menu */}
                        {showMenu && (
                            <div className="absolute right-0 mt-2 w-56 glass-strong rounded-2xl shadow-2xl overflow-hidden z-[95] animate-in fade-in zoom-in-95">
                                {/* User info header */}
                                <div className="p-3 flex items-center gap-3" style={{ borderBottom: '1px solid var(--line)', background: 'var(--wash)' }}>
                                    {avatarUrl ? (
                                        <img src={avatarUrl} alt="" className="w-9 h-9 rounded-full ring-soft" />
                                    ) : (
                                        <div className="w-9 h-9 rounded-full flex items-center justify-center text-sm font-bold pill-grad" style={{ color: 'var(--brand-ink)' }}>{initial}</div>
                                    )}
                                    <div className="min-w-0 flex-1">
                                        <p className="text-sm font-bold text-white truncate">{displayName}</p>
                                        <p className="text-[10px] mono truncate" style={{ color: 'var(--text-3)' }}>{user.email}</p>
                                    </div>
                                </div>

                                {/* Subscription action */}
                                {!isPremium && (
                                    <button
                                        onClick={() => {
                                            setShowMenu(false);
                                            const checkoutUrl = `${LEMON_CHECKOUT_URL}?checkout[email]=${encodeURIComponent(user.email)}`;
                                            window.open(checkoutUrl, '_blank');
                                        }}
                                        className="w-full px-3 py-2.5 flex items-center gap-2.5 text-sm transition-colors text-left hover:bg-[color:var(--wash-2)]"
                                        style={{ color: 'var(--ink)' }}
                                    >
                                        <Sparkles size={14} style={{ color: 'var(--ink)' }} />
                                        <span className="flex-1">升級 Pro</span>
                                        <span className="fs-chip" style={{ color: 'var(--ink)' }}>VIP</span>
                                    </button>
                                )}
                                {isPremium && (
                                    <button
                                        onClick={() => { setShowMenu(false); window.open(LEMON_PORTAL_URL, '_blank'); }}
                                        className="w-full px-3 py-2.5 flex items-center gap-2.5 text-sm transition-colors text-left hover:bg-[color:var(--wash-2)]"
                                        style={{ color: 'var(--ink)' }}
                                    >
                                        <Sparkles size={14} style={{ color: 'var(--ink)' }} />
                                        <span className="flex-1">管理訂閱</span>
                                    </button>
                                )}

                                {/* Settings */}
                                {onOpenSettings && (
                                    <button
                                        onClick={() => { setShowMenu(false); onOpenSettings(); }}
                                        className="w-full px-3 py-2.5 flex items-center gap-2.5 text-sm transition-colors text-left hover:bg-[color:var(--wash-2)]"
                                        style={{ color: 'var(--text)' }}
                                    >
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <circle cx="12" cy="12" r="3"></circle>
                                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                                        </svg>
                                        <span className="flex-1">設定</span>
                                    </button>
                                )}

                                {/* Logout */}
                                <button
                                    onClick={() => { setShowMenu(false); handleLogout(); }}
                                    className="w-full px-3 py-2.5 flex items-center gap-2.5 text-sm transition-colors text-left hover:bg-red-500/10"
                                    style={{ color: 'var(--down)', borderTop: '1px solid var(--line)' }}
                                >
                                    <LogOut size={14} />
                                    <span className="flex-1">{window.__DEMO__ ? '離開展示' : '登出'}</span>
                                </button>
                            </div>
                        )}
                    </div>
                );
            }

            return (
                <>
                    <button
                        onClick={() => setShowLoginModal(true)}
                        className="fs-btn solid sm h-9"
                    >
                        <User size={14} /> 登入
                    </button>

                    {showLoginModal && (
                        <div className="fixed inset-0 flex items-center justify-center z-[100] p-4" style={{ background: 'rgba(7,8,12,0.78)', backdropFilter: 'blur(8px)' }}>
                            <div className="glass-strong p-7 rounded-3xl shadow-2xl max-w-sm w-full relative">
                                <button
                                    onClick={() => setShowLoginModal(false)}
                                    className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors"
                                    style={{ color: 'var(--text-3)' }}
                                >
                                    ✕
                                </button>

                                <div className="flex items-center gap-3 mb-6">
                                    <LogoMark size={36} />
                                    <div>
                                        <h3 className="text-lg font-extrabold text-white tracking-tight">
                                            {isSignUp ? '建立帳號' : '歡迎回來'}
                                        </h3>
                                        <p className="label mt-0.5">Smart DCA</p>
                                    </div>
                                </div>

                                <form onSubmit={handleLogin} className="space-y-3">
                                    <div>
                                        <label className="label block mb-1.5">Email</label>
                                        <input
                                            type="email"
                                            required
                                            value={email}
                                            onChange={e => setEmail(e.target.value)}
                                            className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
                                            style={{ background: 'var(--wash)', border: '1px solid var(--line)' }}
                                            onFocus={e => e.target.style.borderColor = 'rgba(139,92,246,0.5)'}
                                            onBlur={e => e.target.style.borderColor = 'var(--line)'}
                                        />
                                    </div>
                                    <div>
                                        <label className="label block mb-1.5">Password</label>
                                        <input
                                            type="password"
                                            required
                                            minLength={6}
                                            value={password}
                                            onChange={e => setPassword(e.target.value)}
                                            className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
                                            style={{ background: 'var(--wash)', border: '1px solid var(--line)' }}
                                            onFocus={e => e.target.style.borderColor = 'rgba(139,92,246,0.5)'}
                                            onBlur={e => e.target.style.borderColor = 'var(--line)'}
                                        />
                                    </div>
                                    {isSignUp && (
                                        <div>
                                            <label className="label block mb-1.5">Confirm Password</label>
                                            <input
                                                type="password"
                                                required
                                                minLength={6}
                                                value={confirmPassword}
                                                onChange={e => setConfirmPassword(e.target.value)}
                                                className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
                                                style={{ background: 'var(--wash)', border: '1px solid var(--line)' }}
                                                onFocus={e => e.target.style.borderColor = 'rgba(139,92,246,0.5)'}
                                                onBlur={e => e.target.style.borderColor = 'var(--line)'}
                                            />
                                        </div>
                                    )}

                                    {msg && (
                                        <div className="text-xs text-center px-3 py-2 rounded-lg" style={{ background: 'rgba(255,91,110,0.08)', color: 'var(--down)', border: '1px solid rgba(255,91,110,0.2)' }}>
                                            {msg}
                                        </div>
                                    )}

                                    <button
                                        type="submit"
                                        disabled={localLoading}
                                        className="w-full py-2.5 pill-grad text-white rounded-xl text-sm font-bold transition-all glow-brand disabled:opacity-50"
                                    >
                                        {localLoading ? '處理中...' : (isSignUp ? '註冊' : '登入')}
                                    </button>
                                </form>

                                <div className="my-4 flex items-center gap-3">
                                    <div className="h-px flex-1" style={{ background: 'var(--line)' }}></div>
                                    <span className="text-[10px] mono" style={{ color: 'var(--text-3)' }}>OR</span>
                                    <div className="h-px flex-1" style={{ background: 'var(--line)' }}></div>
                                </div>

                                <button
                                    type="button"
                                    onClick={handleGoogleLogin}
                                    className="w-full py-2.5 bg-white text-slate-900 rounded-xl text-sm font-bold hover:bg-slate-100 transition-colors flex items-center justify-center gap-2"
                                >
                                    <svg className="w-4 h-4" viewBox="0 0 24 24"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" /><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" /><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" /><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" /></svg>
                                    使用 Google 繼續
                                </button>

                                <div className="mt-5 text-center">
                                    <button
                                        onClick={() => { setIsSignUp(!isSignUp); setMsg(null); setConfirmPassword(''); }}
                                        className="text-xs underline transition-colors"
                                        style={{ color: 'var(--text-3)' }}
                                    >
                                        {isSignUp ? '已有帳號？登入' : '沒有帳號？立即註冊'}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </>
            );
        };

        // --- 恐懼貪婪指數儀表板元件 ---
        console.log("SmartDCA Version: 1.1 (Debug Mode)");

        // API Key Placeholder for GitHub Actions Injection
        // API Key Placeholder for GitHub Actions Injection
        // Helper: Format Price (8 decimals for < 1, 2 for >= 1)
        const formatPrice = (price) => {
            if (price === null || price === undefined) return 'N/A';
            if (price < 1) return price.toLocaleString(undefined, { maximumFractionDigits: 8 });
            return price.toLocaleString(undefined, { maximumFractionDigits: 2 });
        };

        // --- Gemini API Helper (via Supabase Edge Function) ---
        // opts.macro: 讓 gemini-proxy 在伺服器端附上最新總經數據與經濟日曆
        const fetchGeminiAdvice = async (prompt, opts = {}) => {
            if (!supabase || SUPABASE_URL.includes('PLACEHOLDER')) {
                return 'WARNING: AI not available (Supabase not configured)';
            }
            try {
                const sessionRes = await supabase.auth.getSession();
                const token = (sessionRes.data && sessionRes.data.session && sessionRes.data.session.access_token) || SUPABASE_ANON_KEY;
                const hdrs = { 'Content-Type': 'application/json', 'apikey': SUPABASE_ANON_KEY, 'Authorization': 'Bearer ' + token };
                const userKey = (typeof window !== 'undefined' && window.__USER_GEMINI_KEY) || undefined;
                const res = await fetch(SUPABASE_URL + '/functions/v1/gemini-proxy', {
                    method: 'POST',
                    headers: hdrs,
                    body: JSON.stringify({ prompt, apiKey: userKey, macro: !!opts.macro })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Proxy error');
                return data.candidates[0].content.parts[0].text;
            } catch (err) {
                console.error('Gemini Proxy Error:', err);
                return 'WARNING: AI request failed, please try again.';
            }
                };

        // --- AI Advice Component ---
        const AIAdviceBlock = ({ marketData, assetName, priceStats, isLocked }) => {
            const [advice, setAdvice] = useState(null);
            const [loading, setLoading] = useState(false);
            const [started, setStarted] = useState(false);

            const handleAnalyze = () => {
                if (!marketData) return;
                setStarted(true);
                setLoading(true);
                const priceInfo = priceStats ? `
                資產數據:
                - 當前價格: $${formatPrice(priceStats.current)}
                - 近一年最高: $${formatPrice(priceStats.high)}
                - 近一年最低: $${formatPrice(priceStats.low)}
                ` : '';

                const prompt = `
                你是一位專業的投資顧問。根據以下 ${assetName} 的市場數據，提供一個簡短的 DCA (平均成本法) 操作建議 (50字以內)。
                **核心任務：**
                1. 分析當前的 FNG/RSI 數值所代表的市場情緒強度。
                2. 根據情緒強度，結合資產名稱和當前價格，**相較於最近一年的價格波動 (參考最高/最低價)**，判斷現在的價格是否具有吸引力？並分析歷史高點與當前價格相差幾%。
                3. 根據以下行動邏輯，生成一段富有洞察力和鼓勵性的建議。
                **行動邏輯：**
                - 極度恐懼 (<= 25): 立即建議「強力分批買入」或「執行最大額度投入」。
                - 恐懼 (26 - 44): 建議「小額分批買入」，鼓勵保持紀律。
                - 中立 (45 - 55): 建議「維持觀望，不買也不賣」。
                - 貪婪 (56 - 74) 極度貪婪 (>= 75):: 建議「停止買入，開始小額分批賣出 (止盈)」。

                市場數據:
                ${marketData}
                ${priceInfo}

                建議風格: 理性、穩健、鼓勵分批進場。請用繁體中文回答。
                `;
                fetchGeminiAdvice(prompt, { macro: true }).then(text => {
                    setAdvice(text);
                    setLoading(false);
                });
            };

            if (!started) return (
                <div className="fs-rule" style={{ borderBottom: '1px solid var(--rule)' }}>
                    <button
                        onClick={isLocked ? () => { } : handleAnalyze}
                        disabled={isLocked}
                        className={`w-full py-3.5 flex items-center justify-between gap-3 text-left ${isLocked ? 'cursor-not-allowed' : ''}`}
                    >
                        <div className="flex items-center gap-3 min-w-0">
                            {isLocked ? <Lock size={18} style={{ color: 'var(--ink-3)' }} /> : <Bot size={20} style={{ color: 'var(--ink)' }} />}
                            <div className="min-w-0">
                                <div className="text-[15px] font-bold flex items-center gap-2" style={{ color: 'var(--ink)' }}>
                                    AI 投資顧問
                                    {isLocked ? <span className="fs-chip" style={{ color: 'var(--ink)' }}>PRO</span> : <span className="fs-chip" style={{ color: 'var(--ink-2)' }}>GEMINI</span>}
                                </div>
                                <div className="text-[13px] mt-0.5 truncate" style={{ color: 'var(--ink-2)' }}>
                                    {isLocked ? '升級 Pro 解鎖 Gemini 深度分析' : '根據情緒指數 + 價格走勢產出個人化建議'}
                                </div>
                            </div>
                        </div>
                        <span className={`fs-btn sm shrink-0 ${isLocked ? '' : 'solid'}`}>{isLocked ? '升級' : '分析 →'}</span>
                    </button>
                </div>
            );

            if (loading) return (
                <div className="py-4 flex items-center gap-3 text-sm fs-rule" style={{ color: 'var(--ink-2)', borderBottom: '1px solid var(--rule)' }}>
                    <RefreshCw className="animate-spin" size={16} />
                    <span>AI 正在分析市場數據...</span>
                </div>
            );

            if (!advice) return null;

            return (
                <div className="fs-rule pt-4 pb-5" style={{ borderBottom: '1px solid var(--rule)' }}>
                    <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-2">
                            <Bot size={18} style={{ color: 'var(--ink)' }} />
                            <h4 className="font-bold text-[15px]" style={{ color: 'var(--ink)' }}>AI 投資顧問建議</h4>
                            <span className="fs-chip" style={{ color: 'var(--ink-2)' }}>GEMINI</span>
                        </div>
                        <button onClick={() => { setAdvice(null); setStarted(false); }} className="p-1.5 transition-colors" style={{ color: 'var(--ink-3)' }} aria-label="重新分析">
                            <RefreshCw size={14} />
                        </button>
                    </div>

                    {priceStats && (
                        <div className="grid grid-cols-3 mb-4" style={{ borderTop: '1px solid var(--ink)', borderBottom: '1px solid var(--rule)' }}>
                            <div className="py-2">
                                <div className="fs-lbl">當前</div>
                                <div className="text-[15px] font-bold num" style={{ color: 'var(--ink)' }}>${formatPrice(priceStats.current)}</div>
                            </div>
                            <div className="py-2 pl-3" style={{ borderLeft: '1px solid var(--rule)' }}>
                                <div className="fs-lbl">1Y 高</div>
                                <div className="text-[15px] font-bold num" style={{ color: 'var(--up)' }}>${formatPrice(priceStats.high)}</div>
                            </div>
                            <div className="py-2 pl-3" style={{ borderLeft: '1px solid var(--rule)' }}>
                                <div className="fs-lbl">1Y 低</div>
                                <div className="text-[15px] font-bold num" style={{ color: 'var(--down)' }}>${formatPrice(priceStats.low)}</div>
                            </div>
                        </div>
                    )}

                    <p className="text-[15px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>
                        {advice}
                    </p>
                </div>
            );
        };

        // --- 經濟日曆 (讀 repo 根目錄 economic_calendar.json,由 py/gen_calendar.py 產生) ---
        // JSON 只存美東時間;這裡依美國日光節約時間換算成台灣時間
        const NY_PARTS = new Intl.DateTimeFormat('en-US', {
            timeZone: 'America/New_York', hourCycle: 'h23',
            year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
        });
        const etToDate = (dateStr, timeStr) => {
            const [y, m, d] = dateStr.split('-').map(Number);
            const [hh, mm] = timeStr.split(':').map(Number);
            const guess = Date.UTC(y, m - 1, d, hh, mm);
            const p = Object.fromEntries(NY_PARTS.formatToParts(new Date(guess)).map(x => [x.type, x.value]));
            const nyAsUtc = Date.UTC(+p.year, +p.month - 1, +p.day, +p.hour, +p.minute);
            return new Date(guess - (nyAsUtc - guess));
        };
        const TW_DAY = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', month: 'numeric', day: 'numeric', weekday: 'short' });
        const TW_TIME = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
        const TW_KEY = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Taipei' });
        const CALENDAR_EVENTS = (economicCalendar.events || [])
            .map(e => ({ ...e, at: etToDate(e.date, e.time_et) }))
            .sort((a, b) => a.at - b.at);
        const IMPACT_STYLE = {
            high: { color: 'var(--down)', label: '高' },
            medium: { color: 'var(--amber)', label: '中' },
            low: { color: 'var(--ink-3)', label: '低' },
        };
        const hasHighImpactSoon = (now = Date.now()) =>
            CALENDAR_EVENTS.some(e => e.impact === 'high' && e.at > now && e.at - now < 24 * 3600 * 1000);

        const relativeLabel = (at, now) => {
            const hours = (at - now) / 3600000;
            if (hours < 1) return '即將公布';
            if (hours < 24) return `${Math.round(hours)} 小時後`;
            return `${Math.round(hours / 24)} 天後`;
        };

        // Month grid in the logo's language: today = the ultramarine square,
        // high impact = filled dot, medium = hollow dot, released = greyed.
        const WEEKDAYS = ['日', '一', '二', '三', '四', '五', '六'];
        const TW_MONTH = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric', month: 'long' });
        const keyToParts = (key) => key.split('-').map(Number); // 'YYYY-MM-DD' → [y, m, d]
        const monthKey = (y, m) => `${y}-${String(m).padStart(2, '0')}`;

        const ImpactDot = ({ impact, past, size = 7 }) => {
            const color = past ? 'var(--ink-3)' : 'var(--ink)';
            return impact === 'high'
                ? <span aria-hidden="true" style={{ width: size, height: size, borderRadius: '50%', background: color, display: 'inline-block', flex: 'none' }} />
                : <span aria-hidden="true" style={{ width: size, height: size, borderRadius: '50%', border: `1.5px solid ${color}`, display: 'inline-block', flex: 'none' }} />;
        };

        const EconomicCalendarModal = ({ onClose }) => {
            const now = Date.now();
            const todayKey = TW_KEY.format(now);
            const [onlyHigh, setOnlyHigh] = useState(false);

            const events = CALENDAR_EVENTS.filter(e => !onlyHigh || e.impact === 'high');
            const byDay = new Map();
            events.forEach(e => {
                const k = TW_KEY.format(e.at);
                if (!byDay.has(k)) byDay.set(k, []);
                byDay.get(k).push(e);
            });
            const nextHigh = CALENDAR_EVENTS.find(e => e.impact === 'high' && e.at > now);

            // Months the data covers (in Taipei dates), bounded for navigation
            const firstKey = CALENDAR_EVENTS.length ? TW_KEY.format(CALENDAR_EVENTS[0].at) : todayKey;
            const lastKey = CALENDAR_EVENTS.length ? TW_KEY.format(CALENDAR_EVENTS[CALENDAR_EVENTS.length - 1].at) : todayKey;
            const [ty, tm] = keyToParts(todayKey);
            const [view, setView] = useState({ y: ty, m: tm });
            const [selected, setSelected] = useState(() => {
                if (byDay.has(todayKey)) return todayKey;
                const next = CALENDAR_EVENTS.find(e => e.at > now);
                const k = next ? TW_KEY.format(next.at) : todayKey;
                return k.slice(0, 7) === monthKey(ty, tm) ? k : todayKey;
            });

            const vKey = monthKey(view.y, view.m);
            const canPrev = vKey > firstKey.slice(0, 7);
            const canNext = vKey < lastKey.slice(0, 7);
            const shift = (delta) => setView(({ y, m }) => {
                const d = new Date(Date.UTC(y, m - 1 + delta, 1));
                return { y: d.getUTCFullYear(), m: d.getUTCMonth() + 1 };
            });
            const goToday = () => { setView({ y: ty, m: tm }); setSelected(todayKey); };

            // Grid cells: leading blanks, days of month, trailing blanks to complete the last week
            const firstDow = new Date(Date.UTC(view.y, view.m - 1, 1)).getUTCDay();
            const daysInMonth = new Date(Date.UTC(view.y, view.m, 0)).getUTCDate();
            const cells = [];
            for (let i = 0; i < firstDow; i++) cells.push(null);
            for (let d = 1; d <= daysInMonth; d++) cells.push(`${vKey}-${String(d).padStart(2, '0')}`);
            while (cells.length % 7) cells.push(null);

            const monthEvents = events.filter(e => TW_KEY.format(e.at).startsWith(vKey));
            const selectedEvents = byDay.get(selected) || [];
            const selDate = new Date(`${selected}T12:00:00+08:00`);

            return (
                <div className="fixed inset-0 flex items-end sm:items-center justify-center z-[110] sm:p-4" style={{ background: 'rgba(0,0,0,0.55)' }} onClick={onClose}>
                    <div className="w-full sm:max-w-2xl max-h-[92vh] sm:max-h-[88vh] overflow-hidden flex flex-col relative"
                        style={{ background: 'var(--paper)', borderTop: '3px solid var(--ink)', boxShadow: '0 24px 48px -16px rgba(0,0,0,0.45)' }}
                        onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="cal-title">

                        <div className="px-5 pt-4 pb-3" style={{ borderBottom: '1px solid var(--rule)' }}>
                            <button onClick={onClose} className="absolute top-3 right-3 fs-btn icon" aria-label="關閉">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
                            </button>
                            <div className="flex items-baseline gap-3 flex-wrap pr-12">
                                <h3 id="cal-title" className="fs-title">美國經濟日曆</h3>
                                <span className="fs-lbl">ECONOMIC CALENDAR</span>
                            </div>
                            <p className="text-[13px] mt-1" style={{ color: 'var(--ink-3)' }}>台灣時間 · 已自動換算美國日光節約時間</p>

                            {nextHigh && (
                                <button
                                    onClick={() => { const k = TW_KEY.format(nextHigh.at); const [y, m] = keyToParts(k); setView({ y, m }); setSelected(k); }}
                                    className="w-full mt-3 flex items-center justify-between gap-3 py-2.5 px-3 text-left"
                                    style={{ background: 'var(--accent)', color: 'var(--accent-ink)' }}
                                >
                                    <span className="min-w-0">
                                        <span className="block text-[12px] opacity-80">下一個高影響事件</span>
                                        <span className="block text-[15px] font-bold truncate">{nextHigh.title}</span>
                                    </span>
                                    <span className="text-right shrink-0">
                                        <span className="block text-[12px] opacity-80 num">{TW_DAY.format(nextHigh.at)} {TW_TIME.format(nextHigh.at)}</span>
                                        <span className="block text-[15px] font-bold">{relativeLabel(nextHigh.at, now)}</span>
                                    </span>
                                </button>
                            )}

                            <div className="flex items-center justify-between gap-3 mt-3 flex-wrap">
                                <div className="flex items-center gap-1">
                                    <button onClick={() => canPrev && shift(-1)} disabled={!canPrev} className="fs-btn icon" aria-label="上個月">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M15 6l-6 6 6 6" /></svg>
                                    </button>
                                    <span className="text-[17px] font-black num min-w-[7.5em] text-center" style={{ color: 'var(--ink)' }}>
                                        {TW_MONTH.format(new Date(Date.UTC(view.y, view.m - 1, 15)))}
                                    </span>
                                    <button onClick={() => canNext && shift(1)} disabled={!canNext} className="fs-btn icon" aria-label="下個月">
                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
                                    </button>
                                    <button onClick={goToday} className="fs-btn sm ml-1">今天</button>
                                </div>
                                <div className="fs-toggle">
                                    {[[false, '全部'], [true, '僅高影響']].map(([val, label]) => (
                                        <button key={label} onClick={() => setOnlyHigh(val)} className={onlyHigh === val ? 'on' : ''}>{label}</button>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar">
                            {/* Month grid */}
                            <div className="px-5 pt-3">
                                <div className="grid grid-cols-7" style={{ borderBottom: '1px solid var(--ink)' }}>
                                    {WEEKDAYS.map(w => (
                                        <div key={w} className="text-center text-[12px] py-1.5" style={{ color: 'var(--ink-3)' }}>{w}</div>
                                    ))}
                                </div>
                                <div className="grid grid-cols-7">
                                    {cells.map((k, i) => {
                                        if (!k) return <div key={`b${i}`} style={{ borderBottom: '1px solid var(--rule)' }} />;
                                        const list = byDay.get(k) || [];
                                        const isToday = k === todayKey;
                                        const isSel = k === selected;
                                        const isPast = k < todayKey;
                                        const day = keyToParts(k)[2];
                                        return (
                                            <button
                                                key={k}
                                                onClick={() => setSelected(k)}
                                                aria-pressed={isSel}
                                                aria-label={`${view.m} 月 ${day} 日${list.length ? `，${list.length} 個事件` : ''}`}
                                                className="relative flex flex-col items-start gap-1 p-1.5 sm:p-2 text-left min-h-[52px] sm:min-h-[76px] transition-colors hover:bg-[color:var(--wash)]"
                                                style={{
                                                    borderBottom: '1px solid var(--rule)',
                                                    boxShadow: isSel ? 'inset 0 0 0 2px var(--ink)' : 'none',
                                                    background: isSel ? 'var(--wash)' : undefined,
                                                }}
                                            >
                                                <span
                                                    className="num text-[13px] sm:text-[14px] leading-none inline-flex items-center justify-center"
                                                    style={isToday
                                                        ? { background: 'var(--accent)', color: 'var(--accent-ink)', fontWeight: 900, width: 22, height: 22 }
                                                        : { color: isPast ? 'var(--ink-3)' : 'var(--ink)', fontWeight: list.length ? 800 : 500, height: 22 }}
                                                >{day}</span>
                                                {list.length > 0 && (
                                                    <>
                                                        {/* phones: dots only */}
                                                        <span className="flex gap-1 flex-wrap sm:hidden">
                                                            {list.map(e => <ImpactDot key={e.type + e.title} impact={e.impact} past={e.at <= now} size={6} />)}
                                                        </span>
                                                        {/* desktop: dot + event code */}
                                                        <span className="hidden sm:flex flex-col gap-0.5 w-full">
                                                            {list.slice(0, 3).map(e => (
                                                                <span key={e.type + e.title} className="flex items-center gap-1 text-[11px] font-bold leading-tight num truncate"
                                                                    style={{ color: e.at <= now ? 'var(--ink-3)' : 'var(--ink)' }}>
                                                                    <ImpactDot impact={e.impact} past={e.at <= now} size={6} />{e.type}
                                                                </span>
                                                            ))}
                                                        </span>
                                                    </>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                                <div className="flex items-center gap-4 flex-wrap py-2.5 text-[12px]" style={{ color: 'var(--ink-3)' }}>
                                    <span className="inline-flex items-center gap-1.5"><ImpactDot impact="high" />高影響</span>
                                    <span className="inline-flex items-center gap-1.5"><ImpactDot impact="medium" />中影響</span>
                                    <span className="inline-flex items-center gap-1.5"><span style={{ width: 10, height: 10, background: 'var(--accent)', display: 'inline-block' }} />今天</span>
                                    <span className="ml-auto num">本月 {monthEvents.length} 個事件</span>
                                </div>
                            </div>

                            {/* Selected day */}
                            <div className="px-5 pb-5">
                                <div className="fs-section">
                                    <div className="flex items-baseline justify-between gap-3 mb-1">
                                        <h4 className="fs-title-sm">{TW_DAY.format(selDate)}{selected === todayKey ? ' · 今天' : ''}</h4>
                                        <span className="text-[12px] num" style={{ color: 'var(--ink-3)' }}>{selectedEvents.length ? `${selectedEvents.length} 個事件` : ''}</span>
                                    </div>
                                    {selectedEvents.length === 0 && (
                                        <p className="text-[14px] py-3" style={{ color: 'var(--ink-3)' }}>這天沒有符合條件的事件</p>
                                    )}
                                    {selectedEvents.map(e => {
                                        const s = IMPACT_STYLE[e.impact] || IMPACT_STYLE.low;
                                        const past = e.at <= now;
                                        return (
                                            <div key={e.date + e.type + e.title} className="flex items-center gap-3 py-3" style={{ borderBottom: '1px solid var(--rule)', opacity: past ? 0.6 : 1 }}>
                                                <ImpactDot impact={e.impact} past={past} size={9} />
                                                <span className="text-[15px] font-bold num w-12 shrink-0" style={{ color: 'var(--ink)' }}>{TW_TIME.format(e.at)}</span>
                                                <div className="min-w-0 flex-1">
                                                    <div className="text-[15px] leading-snug" style={{ color: 'var(--ink)' }}>
                                                        {e.title}
                                                        {!e.confirmed && <span className="ml-1.5 text-[11px]" style={{ color: 'var(--ink-3)' }}>暫定</span>}
                                                    </div>
                                                    <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>
                                                        {s.label}影響{e.note ? ` · ${e.note}` : ''}
                                                    </div>
                                                </div>
                                                <span className="text-[12px] num shrink-0 font-semibold" style={{ color: past ? 'var(--ink-3)' : 'var(--ink)' }}>
                                                    {past ? '已公布' : relativeLabel(e.at, now)}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>

                        <div className="px-5 py-2.5" style={{ borderTop: '1px solid var(--rule)' }}>
                            <p className="text-[11px] num" style={{ color: 'var(--ink-3)' }}>SOURCE · FED / BLS / BEA / CENSUS · 暫定=依規則推估</p>
                        </div>
                    </div>
                </div>
            );
        };

        // --- AI News Analysis Component ---
        const NewsAIAnalysis = ({ newsItems, isLocked }) => {
            const [summary, setSummary] = useState(null);
            const [loading, setLoading] = useState(false);
            const [started, setStarted] = useState(false);

            const handleAnalyze = () => {
                if (!newsItems || newsItems.length === 0) return;
                setStarted(true);
                setLoading(true);
                const titles = newsItems.map(n => n.title).join("\n");
                const prompt = `
                 請閱讀以下今日財經新聞標題，並總結成一段 50 字以內的「今日市場重點」。

                 **核心任務：**
                 1. 總結**至少兩到三個**不同的市場主題 (如：宏觀經濟影響、新技術發展、主要資產價格)。
                 2. 避免將單一資產的價格突破作為唯一的總結重點。

                 新聞標題:
                 ${titles}

                 請用繁體中文回答，語氣專業客觀。
                 `;
                fetchGeminiAdvice(prompt).then(text => {
                    setSummary(text);
                    setLoading(false);
                });
            };

            if (!started) return (
                <div className="mb-6">
                    <button
                        onClick={handleAnalyze}
                        className="fs-btn solid"
                    >
                        <Newspaper size={16} />
                        AI 分析今日新聞重點
                    </button>
                </div>
            );

            if (isLocked) return (
                <div className="mb-6">
                    <button
                        disabled
                        className="fs-btn"
                    >
                        <Lock size={16} />
                        AI 分析今日新聞重點
                    </button>
                    <p className="text-xs text-amber-500 mt-2 font-bold">升級 Premium 解鎖 AI 新聞摘要</p>
                </div>
            );

            if (loading) return (
                <div className="mb-6 p-4 bg-slate-800/50 rounded-xl border border-slate-700 animate-pulse flex items-center gap-2 text-slate-400 text-sm">
                    <RefreshCw className="animate-spin" size={16} />
                    ✨ AI 正在閱讀今日新聞並生成總結...
                </div>
            );

            if (!summary) return null;

            return (
                <div className="mb-6 p-4 bg-gradient-to-r from-slate-800 to-slate-900 rounded-xl border border-blue-500/30 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                    <div className="flex justify-between items-start mb-2">
                        <h4 className="text-blue-400 font-bold text-sm flex items-center gap-2">
                            <span className="text-lg">📰</span> AI 今日新聞重點
                        </h4>
                        <button onClick={() => { setSummary(null); setStarted(false); }} className="text-slate-600 hover:text-slate-400"><RefreshCw size={14} /></button>
                    </div>
                    <p className="text-slate-300 text-sm leading-relaxed">
                        {summary}
                    </p>
                </div>
            );
        };

        // Fear & Greed zones: [upper bound, label]; colours come from --z0..--z4
        const FNG_ZONES = [
            [25, '極度恐懼'], [44, '恐懼'], [55, '中立'], [74, '貪婪'], [100, '極度貪婪'],
        ];
        const fngZoneIndex = (v) => FNG_ZONES.findIndex(([max]) => v <= max);

        // Dial gauge drawn in real pixels so labels stay legible at every width.
        const FearGreedGauge = ({ fngValue, classification }) => {
            const [boxRef, width] = useWidth();
            const hasValue = fngValue !== null && fngValue !== undefined && fngValue !== '';
            const v = Math.min(100, Math.max(0, hasValue ? Number(fngValue) : 50));
            const zi = fngZoneIndex(v);
            const [armed, setArmed] = useState(false);
            useEffect(() => {
                // Arm after first paint; the timeout covers throttled or hidden frames where rAF never fires
                const id = requestAnimationFrame(() => requestAnimationFrame(() => setArmed(true)));
                const t = setTimeout(() => setArmed(true), 120);
                return () => { cancelAnimationFrame(id); clearTimeout(t); };
            }, []);

            const W = Math.max(280, Math.min(width || 560, 620));
            const R = W * 0.4, band = Math.max(12, W * 0.03);
            const cx = W / 2, cy = R + (W < 420 ? 34 : 44);
            const H = cy + 26;
            const ang = (val) => Math.PI * (1 - val / 100);
            const pt = (val, r) => [cx + r * Math.cos(ang(val)), cy - r * Math.sin(ang(val))];
            const arc = (a, b, r) => {
                const [x1, y1] = pt(a, r), [x2, y2] = pt(b, r);
                return `M${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2}`;
            };
            const bounds = [0, ...FNG_ZONES.map(([m]) => m)];
            const rBand = R - band;
            const small = W < 420;

            return (
                <div ref={boxRef} className="w-full max-w-[620px] mx-auto">
                    {width > 0 && (
                        <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`恐懼與貪婪指數 ${hasValue ? fngValue : '--'} ${classification ?? ''}`} style={{ display: 'block', overflow: 'visible' }}>
                            {FNG_ZONES.map(([, label], i) => (
                                <path key={label} d={arc(bounds[i] + 0.6, bounds[i + 1] - 0.6, rBand)} fill="none"
                                    style={{ stroke: `var(--z${i})`, strokeWidth: hasValue && i === zi ? band * 1.9 : band, opacity: hasValue && i === zi ? 1 : 0.38 }} />
                            ))}
                            {Array.from({ length: 51 }, (_, k) => {
                                const val = k * 2, big = val % 10 === 0;
                                const [x1, y1] = pt(val, R + 4), [x2, y2] = pt(val, R + (big ? 16 : 10));
                                const [tx, ty] = pt(val, R + 30);
                                return (
                                    <g key={val}>
                                        <line x1={x1} y1={y1} x2={x2} y2={y2} style={{ stroke: 'var(--ink)', strokeWidth: big ? 1.5 : 0.75 }} />
                                        {big && (!small || val % 20 === 0) && (
                                            <text x={tx} y={ty + 4} textAnchor="middle" style={{ fill: 'var(--ink-3)', fontSize: 12 }}>{val}</text>
                                        )}
                                    </g>
                                );
                            })}
                            {FNG_ZONES.map(([, label], i) => {
                                const [tx, ty] = pt((bounds[i] + bounds[i + 1]) / 2, rBand - band * (hasValue && i === zi ? 1.6 : 1.2) - 10);
                                const on = hasValue && i === zi;
                                return (
                                    <text key={label} x={tx} y={ty + 4} textAnchor="middle"
                                        style={{ fill: on ? 'var(--ink)' : 'var(--ink-3)', fontSize: small ? 11 : 13, fontWeight: on ? 900 : 500 }}>{label}</text>
                                );
                            })}
                            {hasValue && (
                                <g className="fs-needle" style={{ transformOrigin: `${cx}px ${cy}px`, transform: `rotate(${((armed ? v : 0) / 100) * 180 - 90}deg)` }}>
                                    {/* Pointer lives on the outer ring only, so it never crosses the reading */}
                                    <line x1={cx} y1={cy - R * 0.58} x2={cx} y2={cy - R - 3} style={{ stroke: 'var(--ink)', strokeWidth: 3.5, strokeLinecap: 'round' }} />
                                    <circle cx={cx} cy={cy - R * 0.58} r={4} style={{ fill: 'var(--ink)' }} />
                                </g>
                            )}
                            <text x={cx} y={cy - 18} textAnchor="middle" className="num"
                                style={{ fill: 'var(--ink)', fontSize: Math.round(Math.min(W * 0.2, 118)), fontWeight: 900, letterSpacing: '-0.04em' }}>
                                {hasValue ? fngValue : '--'}
                            </text>
                            <text x={cx} y={cy + 8} textAnchor="middle" style={{ fill: 'var(--ink)', fontSize: 16, fontWeight: 700 }}>{classification ?? ''}</text>
                        </svg>
                    )}
                </div>
            );
        };

        // DCA strategy column that sits beside the dial: verdict, the one accent block, action scale
        const DCA_ACTIONS = ['強力買入', '分批買入', '持有觀望', '停止買入', '止盈/減倉'];
        const DcaStrategy = ({ suggestion, children, label = 'DCA 策略', actions = DCA_ACTIONS }) => (
            <div className="min-w-0">
                <div className="flex items-baseline gap-3 flex-wrap">
                    <span className="fs-lbl">{label}</span>
                    <span className="text-2xl font-black" style={{ color: 'var(--ink)' }}>{suggestion.title}</span>
                </div>
                <p className="mt-1.5 text-[15px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>{suggestion.desc}</p>
                <div className="fs-action mt-5">
                    <span className="fs-lbl">當前操作</span>
                    <span className="text-2xl font-black">{suggestion.action}</span>
                </div>
                <div className="fs-steps mt-4">
                    {actions.map(a => <span key={a} className={a === suggestion.action ? 'on' : ''}>{a}</span>)}
                </div>
                {children}
            </div>
        );

        // Stock / Taiwan dashboards end the scale with 考慮止盈 and call the middle step 持有/觀望
        const MARKET_ACTIONS = ['強力買入', '分批買入', '持有/觀望', '停止買入', '考慮止盈'];

        // Fear buy points on a price line: extreme fear = filled triangle, fear = hollow ring, others unmarked
        const fearPointProps = (values, extreme = 25, fear = 44) => ({
            pointStyle: values.map(v => (v <= extreme ? 'triangle' : 'circle')),
            pointBackgroundColor: values.map(v => (v <= extreme ? CHART.fear : 'rgba(0,0,0,0)')),
            pointBorderColor: values.map(v => (v <= extreme ? CHART.fear : v <= fear ? CHART.fearLight : 'rgba(0,0,0,0)')),
            pointBorderWidth: values.map(v => (v <= extreme ? 0 : 1.2)),
            pointRadius: values.map(v => (v <= extreme ? 3.5 : v <= fear ? 2.2 : 0)),
            pointHoverRadius: 5,
        });
        // Printed-figure chart caption + x axis
        const figureTitle = (text) => ({ display: true, text, color: CHART.tick, align: 'start', font: { size: 12, weight: '500' } });
        const figureX = () => ({ grid: { display: false }, ticks: { color: CHART.tick, maxTicksLimit: window.innerWidth < 640 ? 4 : 8, maxRotation: 0, autoSkipPadding: 12 } });

        // Watchlist star drawn in the icon set's stroke
        const WatchStar = ({ on, onClick }) => (
            <button onClick={onClick} className="p-1 transition-colors" aria-pressed={on} aria-label="觀察清單"
                style={{ color: on ? 'var(--accent)' : 'var(--ink-3)' }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill={on ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true">
                    <path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z" />
                </svg>
            </button>
        );

        // A price-history figure: heading with watch star, 進階分析 link, time range, chart
        const HistorySection = ({ title, watch, onAdvanced, range, onRange, children, height = 'h-[300px] sm:h-[350px]' }) => (
            <section className="fs-section mt-10">
                <div className="fs-head">
                    <h2 className="fs-title flex items-center gap-2">{title}{watch}</h2>
                    <div className="flex items-center gap-4 flex-wrap">
                        <button onClick={onAdvanced} className="text-[14px] underline underline-offset-4" style={{ color: 'var(--ink)' }} title="進階技術分析">進階分析</button>
                        <TimeRangeSelector range={range} onRangeChange={onRange} />
                    </div>
                </div>
                <div className={`${height} w-full relative`}>{children}</div>
            </section>
        );

        // Headlines section: AI summary button, top-3 list, full list
        const NewsBlock = ({ title, news, loading, error, ready, isLocked }) => (
            <section className="fs-section mt-10">
                <div className="fs-head"><h2 className="fs-title">{title}</h2></div>
                {ready && news.length > 0 && <><NewsAIAnalysis newsItems={news} isLocked={isLocked} /><NewsSummary news={news} /></>}
                <NewsSection news={news} loading={loading} error={error} />
            </section>
        );

        const SourceNote = ({ children }) => (
            <p className="mt-10 pt-3 text-[12px]" style={{ color: 'var(--ink-3)', borderTop: '1px solid var(--rule)' }}>{children}</p>
        );

        // --- 新聞摘要元件 ---
        const NewsSummary = ({ news }) => {
            const topNews = news.slice(0, 3);
            return (
                <div className="mb-6 pb-2" style={{ borderBottom: '1px solid var(--ink)' }}>
                    <div className="flex items-baseline justify-between gap-3 mb-1">
                        <h3 className="fs-title-sm flex items-center gap-2"><Sparkles size={15} style={{ color: 'var(--ink)' }} />市場頭條</h3>
                        <span className="fs-lbl">今日重點</span>
                    </div>
                    <div>
                        {topNews.map((item, index) => (
                            <a key={index} href={item.link} target="_blank" rel="noopener noreferrer" className="flex gap-3 items-baseline group py-2.5" style={{ borderTop: index ? '1px solid var(--rule)' : 'none' }}>
                                <span className="num text-[13px] font-bold w-6 shrink-0" style={{ color: 'var(--ink-3)' }}>0{index + 1}</span>
                                <span className="text-[15px] leading-snug line-clamp-2 group-hover:underline underline-offset-4" style={{ color: 'var(--ink)' }}>
                                    {item.title}
                                </span>
                            </a>
                        ))}
                    </div>
                </div>
            );
        };

        // --- 新聞列表元件 ---
        const NewsSection = ({ news, loading, error }) => {
            if (loading) {
                return (
                    <div className="flex justify-center items-center py-12">
                        <RefreshCw className="animate-spin" size={24} style={{ color: 'var(--text-3)' }} />
                    </div>
                );
            }
            if (error) {
                return (
                    <div className="text-center py-8 text-sm" style={{ color: 'var(--down)' }}>
                        無法載入新聞: {error}
                    </div>
                );
            }

            const displayNews = news.slice(0, 6);

            return (
                <div className="grid md:grid-cols-2 md:gap-x-8">
                    {displayNews.map((item, index) => (
                        <a
                            key={index}
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="group flex items-stretch gap-3 py-3.5"
                            style={{ borderBottom: '1px solid var(--rule)' }}
                        >
                            <div className="flex flex-col justify-between flex-1 min-w-0">
                                <h4 className="font-medium text-[15px] line-clamp-2 leading-snug group-hover:underline underline-offset-4" style={{ color: 'var(--ink)' }}>
                                    {item.title}
                                </h4>
                                <div className="flex items-center justify-between text-xs mt-2" style={{ color: 'var(--ink-3)' }}>
                                    <span className="num">{new Date(item.pubDate).toLocaleDateString()}</span>
                                    <span className="flex items-center gap-1">
                                        閱讀 <ExternalLink size={11} />
                                    </span>
                                </div>
                            </div>
                            {item.thumbnail && (
                                <div className="w-20 sm:w-24 h-16 shrink-0 relative overflow-hidden" style={{ background: 'var(--surface-2)' }}>
                                    <img
                                        src={item.thumbnail}
                                        alt={item.title}
                                        className="w-full h-full object-cover"
                                        onError={(e) => { e.currentTarget.parentElement.style.display = 'none'; }}
                                    />
                                </div>
                            )}
                        </a>
                    ))}
                </div>
            );
        };


        // --- Crosshair Plugin ---
        const crosshairPlugin = {
            id: 'crosshair',
            afterDatasetsDraw: (chart) => {
                // Check if tooltip is active
                if (chart.tooltip?._active?.length) {
                    const activePoint = chart.tooltip._active[0];
                    const ctx = chart.ctx;
                    const x = activePoint.element.x;
                    const y = activePoint.element.y;
                    const topY = chart.scales.y.top;
                    const bottomY = chart.scales.y.bottom;
                    const leftX = chart.scales.x.left;
                    const rightX = chart.scales.x.right;

                    ctx.save();
                    ctx.beginPath();
                    ctx.setLineDash([5, 5]);
                    ctx.lineWidth = 1;
                    ctx.strokeStyle = CHART.tick;

                    // Vertical Line
                    ctx.moveTo(x, topY);
                    ctx.lineTo(x, bottomY);

                    // Horizontal Line
                    ctx.moveTo(leftX, y);
                    ctx.lineTo(rightX, y);

                    ctx.stroke();
                    ctx.restore();
                }
            }
        };

        // --- 自定義 Chart 元件 ---

        // ─────────────────────────────────────────────
        // Dot-grid background plugin (under the line)
        // ─────────────────────────────────────────────
        const dotGridPlugin = {
            id: 'dotGrid',
            beforeDatasetsDraw: (chart) => {
                const { ctx, chartArea } = chart;
                // Factsheet charts draw on plain paper; the dot grid is retired.
                if (!chartArea || true) return;
                const { left, top, right, bottom } = chartArea;
                ctx.save();
                ctx.beginPath();
                ctx.rect(left, top, chartArea.width, chartArea.height);
                ctx.clip();
                const step = 10;
                const dotRadius = 0.7;
                ctx.fillStyle = 'rgba(59, 130, 246, 0.08)';
                for (let x = left; x <= right; x += step) {
                    for (let y = top; y <= bottom; y += step) {
                        ctx.beginPath();
                        ctx.arc(x, y, dotRadius, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
                ctx.restore();
            }
        };

        // ─────────────────────────────────────────────
        // Line glow plugin (neon shadow under the line)
        // ─────────────────────────────────────────────
        const lineGlowPlugin = {
            id: 'lineGlow',
            beforeDatasetDraw: (chart, args) => {
                const { ctx } = chart;
                const dataset = chart.data.datasets[args.index];
                if (dataset.type && dataset.type !== 'line') return;
                ctx.save();
                ctx.shadowColor = 'rgba(0,0,0,0)';
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
            },
            afterDatasetDraw: (chart) => {
                chart.ctx.restore();
            }
        };

        const ChartComponent = ({ data, options, type = 'line' }) => {
            const chartRef = useRef(null);
            const chartInstance = useRef(null);
            const themeV = useThemeVersion();

            useEffect(() => {
                if (!chartRef.current) return;
                if (chartInstance.current) {
                    chartInstance.current.destroy();
                }
                const ctx = chartRef.current.getContext('2d');
                chartInstance.current = new Chart(ctx, {
                    type: type,
                    data: data,
                    options: options,
                    plugins: [crosshairPlugin, dotGridPlugin, lineGlowPlugin]
                });
                return () => {
                    if (chartInstance.current) {
                        chartInstance.current.destroy();
                    }
                };
            }, [data, options, type, themeV]);

            return <canvas ref={chartRef} />;
        };

        // --- Sentiment Scarcity Bar Component ---
        const SentimentScarcityBar = ({ data, title = "過去 365 天買入機會分佈" }) => {
            if (!data || data.length === 0) return null;

            const total = data.length;
            const extremeFearCount = data.filter(v => v <= 25).length;
            const fearCount = data.filter(v => v > 25 && v <= 44).length;
            const otherCount = total - extremeFearCount - fearCount;

            const extremeFearPercent = (extremeFearCount / total) * 100;
            const fearPercent = (fearCount / total) * 100;
            const otherPercent = (otherCount / total) * 100;

            const segments = [
                { key: 'xf', pct: extremeFearPercent, bg: 'var(--z0)', tip: `極度恐懼: ${extremeFearCount} 天 (${extremeFearPercent.toFixed(1)}%)` },
                { key: 'f',  pct: fearPercent,        bg: 'var(--z1)', tip: `恐懼: ${fearCount} 天 (${fearPercent.toFixed(1)}%)` },
                { key: 'o',  pct: otherPercent,       bg: 'var(--faint)', tip: `其他: ${otherCount} 天 (${otherPercent.toFixed(1)}%)` },
            ];

            return (
                <div className="mt-6 text-left">
                    <h4 className="text-[13px]" style={{ color: 'var(--ink-2)' }}>{title}</h4>
                    <div className="w-full h-2 flex gap-[2px] mt-2">
                        {segments.filter(s => s.pct > 0).map(s => (
                            <div key={s.key} style={{ width: `${s.pct}%`, background: s.bg }} className="h-full relative group">
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block text-[12px] px-2 py-1 whitespace-nowrap z-10 glass-strong" style={{ color: 'var(--ink)' }}>
                                    {s.tip}
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="text-[13px] mt-1.5" style={{ color: 'var(--ink-2)' }}>
                        <span className="font-bold num" style={{ color: 'var(--down)' }}>{extremeFearCount}天</span> 極度恐懼 •
                        <span className="font-bold num ml-1" style={{ color: 'var(--amber)' }}>{fearCount}天</span> 恐懼
                    </div>
                </div>
            );
        };

        // --- Index Price Card Component (Shared) ---
        const IndexPriceCard = ({ name, symbol, logo, data, active = false, editMode = false, onClick, currency = '$' }) => {
            const handleClick = () => { if (!editMode && onClick) onClick(); };
            return (
                <div onClick={handleClick} className={`${editMode || !onClick ? '' : 'cursor-pointer'} fs-ticker ${active ? 'on' : ''}`}>
                    <span className="fs-ticker-name">
                        {logo && <img src={logo} className="w-4 h-4 shrink-0" alt="" onError={(e) => { e.target.style.display = 'none'; }} />}
                        {name}
                    </span>
                    <span className="fs-ticker-price num">{data ? `${currency}${formatPrice(data.price)}` : '...'}</span>
                    {data && (
                        <span className="fs-ticker-change num" style={{ color: data.changePercent >= 0 ? 'var(--up)' : 'var(--down)' }}>
                            {data.changePercent >= 0 ? '▲' : '▼'} {data.changePercent >= 0 ? '+' : ''}{data.changePercent.toFixed(2)}%
                        </span>
                    )}
                </div>
            );
        };

        // --- 股票代號搜尋元件 ---
        // --- 股票代號搜尋元件 ---
        const SymbolSearch = ({ symbol, onSearch, placeholder = "輸入代號 (e.g. AAPL)", transformInput = (s) => s.toUpperCase(), isLocked = false }) => {
            const [input, setInput] = useState(symbol);

            // 當外部 symbol 改變時，更新內部 input
            useEffect(() => {
                setInput(symbol);
            }, [symbol]);

            return (
                <div className="flex gap-2 items-center relative">
                    <div className="relative flex-1">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(transformInput(e.target.value))}
                            onKeyDown={(e) => !isLocked && e.key === 'Enter' && onSearch(input)}
                            disabled={isLocked}
                            className={`w-full bg-transparent px-0.5 py-1.5 text-[15px] outline-none border-b-[1.5px] ${isLocked ? 'cursor-not-allowed' : ''}`}
                            style={{ color: isLocked ? 'var(--ink-3)' : 'var(--ink)', borderColor: 'var(--ink)' }}
                            placeholder={placeholder}
                        />
                        {isLocked && (
                            <div className="absolute inset-0 flex items-center justify-end pr-1 pointer-events-none">
                                <Lock size={15} style={{ color: 'var(--ink-3)' }} />
                            </div>
                        )}
                    </div>
                    <button
                        onClick={() => !isLocked && onSearch(input)}
                        disabled={isLocked}
                        className="fs-btn sm"
                    >
                        {isLocked ? '鎖定' : <><RefreshCw size={16} /> 搜尋</>}
                    </button>

                    {isLocked && (
                        <button
                            onClick={() => {
                                const emailParam = symbol ? `&checkout[custom][symbol]=${symbol}` : '';
                                // Note: We need email here. SymbolSearch consumes 'symbol' and 'onSearch'. 
                                // It receives 'isLocked'. It doesn't receive user email.
                                // We need to update SymbolSearch props.
                            }}
                            className="fs-btn sm"
                        >
                            <Sparkles size={16} /> 升級 Pro
                        </button>
                    )}
                </div>
            );
        };

        // --- 時間範圍選擇器 ---
        const TimeRangeSelector = ({ range, onRangeChange }) => {
            const ranges = ['1M', '6M', '1Y', 'ALL'];
            return (
                <div className="flex gap-1 text-[13px]">
                    {ranges.map(r => (
                        <button
                            key={r}
                            onClick={() => onRangeChange(r)}
                            className="px-2 py-0.5 num transition-colors"
                            style={range === r
                                ? { border: '1.5px solid var(--ink)', color: 'var(--ink)', fontWeight: 700 }
                                : { border: '1.5px solid transparent', color: 'var(--ink-2)' }}
                        >
                            {r}
                        </button>
                    ))}
                </div>
            );
        };

        // Lemon Squeezy Configuration
        const LEMON_CHECKOUT_URL = 'https://smartdca.lemonsqueezy.com/buy/931e6d89-3193-4216-8c9b-a2d88c6e4acc';
        const LEMON_PORTAL_URL = 'https://smartdca.lemonsqueezy.com/billing';

        // ─────────────────────────────────────────────
        // useEditableCards — manage a user-editable, draggable card list (persisted to localStorage)
        // Each card is { id, symbol, name }. id is the stable key, symbol is the API ticker, name is what the user sees.
        // ─────────────────────────────────────────────
        const useEditableCards = (storageKey, defaults) => {
            const [cards, setCards] = useState(() => {
                try {
                    const raw = localStorage.getItem(storageKey);
                    if (raw) {
                        const parsed = JSON.parse(raw);
                        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
                    }
                } catch {}
                return defaults;
            });
            useEffect(() => {
                try { localStorage.setItem(storageKey, JSON.stringify(cards)); } catch {}
            }, [storageKey, cards]);

            const addCard = (card) => setCards(prev => prev.some(c => c.id === card.id) ? prev : [...prev, card]);
            const removeCard = (id) => setCards(prev => prev.filter(c => c.id !== id));
            const moveCard = (from, to) => setCards(prev => {
                if (from === to || from < 0 || to < 0 || from >= prev.length || to >= prev.length) return prev;
                const next = [...prev];
                const [it] = next.splice(from, 1);
                next.splice(to, 0, it);
                return next;
            });
            const resetCards = () => setCards(defaults);
            return { cards, setCards, addCard, removeCard, moveCard, resetCards };
        };

        // ─────────────────────────────────────────────
        // EditableCardGrid — wraps a row of price cards with drag-to-reorder, ✕ remove, and + add buttons.
        // children is expected to be a flat list of card-shaped React nodes, one per item in `cards`, in the same order.
        // ─────────────────────────────────────────────
        const EditableCardGrid = ({ cards, editMode, onToggleEdit, onMove, onRemove, onAdd, onReset, addLabel = '+ 新增', title, children, mobileRow = false }) => {
            const dragIdx = useRef(null);
            const [dragOverIdx, setDragOverIdx] = useState(null);
            const childArr = React.Children.toArray(children);

            return (
                <div>
                    <div className="flex items-center justify-between mt-3 mb-1">
                        <p className="label">{title}</p>
                        <div className="flex items-center gap-1.5">
                            {editMode && onReset && (
                                <button
                                    onClick={() => { if (confirm('還原為預設清單？')) onReset(); }}
                                    className="fs-btn sm"
                                >還原預設</button>
                            )}
                            <button
                                onClick={onToggleEdit}
                                className={`fs-btn sm ${editMode ? 'solid' : ''}`}
                            >{editMode ? '完成' : '編輯'}</button>
                        </div>
                    </div>
                    <div className={`fs-tickers flex gap-2 md:gap-4 flex-wrap ${mobileRow ? 'flex-row' : 'flex-col md:flex-row'}`}>
                        {childArr.map((child, i) => (
                            <div
                                key={(cards[i] && cards[i].id) || i}
                                draggable={editMode}
                                onDragStart={() => { dragIdx.current = i; }}
                                onDragEnd={() => { dragIdx.current = null; setDragOverIdx(null); }}
                                onDragOver={(e) => { if (editMode) { e.preventDefault(); setDragOverIdx(i); } }}
                                onDragLeave={() => setDragOverIdx(prev => prev === i ? null : prev)}
                                onDrop={(e) => {
                                    e.preventDefault();
                                    const from = dragIdx.current;
                                    if (editMode && from !== null && from !== i) onMove(from, i);
                                    dragIdx.current = null;
                                    setDragOverIdx(null);
                                }}
                                className={`relative flex-1 ${mobileRow ? 'basis-0 min-w-[96px]' : 'min-w-0'} md:min-w-[180px] transition-transform ${editMode ? 'cursor-move' : ''} ${dragOverIdx === i && dragIdx.current !== null && dragIdx.current !== i ? 'ring-2 ring-offset-2 scale-[1.02]' : ''}`}
                            >
                                {editMode && (
                                    <>
                                        <span
                                            className="absolute top-2 left-2 z-10 text-sm select-none px-1.5 py-0.5 rounded"
                                            style={{ background: 'rgba(0,0,0,0.5)', color: 'var(--text-2)' }}
                                            title="拖曳排序"
                                        >⋮⋮</span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); onRemove(cards[i].id); }}
                                            className="absolute top-2 right-2 z-10 w-6 h-6 rounded-full flex items-center justify-center hover:bg-red-500/30"
                                            style={{ background: 'rgba(0,0,0,0.5)', color: 'var(--down)' }}
                                            title="移除"
                                        >✕</button>
                                    </>
                                )}
                                {child}
                            </div>
                        ))}
                        {editMode && (
                            <button
                                onClick={onAdd}
                                className={`flex-1 ${mobileRow ? 'basis-0 min-w-[96px]' : 'min-w-0'} md:min-w-[180px] rounded-xl p-4 text-sm font-semibold transition-all hover:bg-white/[0.04] flex items-center justify-center gap-2`}
                                style={{ border: '2px dashed var(--line)', color: 'var(--text-3)', minHeight: '110px' }}
                            >{addLabel}</button>
                        )}
                    </div>
                </div>
            );
        };

        // --- 美股儀表板元件 ---
        const StockDashboard = ({ notificationsEnabled, toggleNotifications, userInfo = { isPremium: false, watchlist: [] }, onUpdateWatchlist }) => {
            const [showAdvanced, setShowAdvanced] = useState(false);
            const [selectedSymbol, setSelectedSymbol] = useState('SPY');
            const [timeRange, setTimeRange] = useState('1Y');
            const [stockData, setStockData] = useState(null);
            const [fullStockData, setFullStockData] = useState(null);
            const [fngData, setFngData] = useState(null);
            const [fngHistory, setFngHistory] = useState([]);
            const [stockNews, setStockNews] = useState([]);
            const [indices, setIndices] = useState({});
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            // Editable card list — symbol = Yahoo ticker, id is stable key, name is shown to the user
            const usDefaults = [
                { id: 'spy', symbol: 'SPY', name: 'S&P 500 ETF' },
                { id: 'qqq', symbol: 'QQQ', name: 'NASDAQ 100 ETF' },
                { id: 'gld', symbol: 'GLD', name: 'Gold ETF' },
            ];
            const usCards = useEditableCards('us-dashboard-cards', usDefaults);
            const [usEditMode, setUsEditMode] = useState(false);

            const fetchStockData = async () => {
                setLoading(true);
                setError(null);
                try {
                    // 1. 獲取真實 CNN Fear & Greed Index (透過 CORS Proxy)
                    const proxyUrl = 'https://cors.hellokai07.com/?' + encodeURIComponent('https://production.dataviz.cnn.io/index/fearandgreed/graphdata');
                    let fngMap = new Map();
                    try {
                        const fngResponse = await fetch(proxyUrl);
                        if (!fngResponse.ok) throw new Error('無法獲取 CNN 數據');
                        const fngJson = await fngResponse.json();

                        if (fngJson.fear_and_greed) {
                            let score = Math.round(fngJson.fear_and_greed.score);
                            let rating = fngJson.fear_and_greed.rating;
                            let lastUpdated = fngJson.fear_and_greed.timestamp;
                            const ratingMap = { "Extreme Fear": "極度恐懼", "extreme fear": "極度恐懼", "Fear": "恐懼", "fear": "恐懼", "Neutral": "中立", "neutral": "中立", "Greed": "貪婪", "greed": "貪婪", "Extreme Greed": "極度貪婪", "extreme greed": "極度貪婪" };
                            rating = ratingMap[rating] || rating;
                            setFngData({ value: score, classification: rating, lastUpdated: lastUpdated });
                        }

                        if (fngJson.fear_and_greed_historical && fngJson.fear_and_greed_historical.data) {
                            const history = fngJson.fear_and_greed_historical.data.map(item => {
                                const dateStr = new Date(item.x).toISOString().split('T')[0];
                                const val = Math.round(item.y);
                                fngMap.set(dateStr, val);
                                return { x: item.x, y: val };
                            });
                            setFngHistory(history);
                        }
                    } catch (e) {
                        console.error("FNG Fetch Error:", e);
                    }

                    // 2. 獲取美股歷史數據 (Yahoo Finance)
                    // 強制獲取 1 年 (1y) 數據，因為圖表需要至少一年來顯示交互
                    const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${selectedSymbol}?interval=1d&range=1y`;
                    const historyRes = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yahooUrl));

                    if (!historyRes.ok) throw new Error("Yahoo Finance API Error");
                    const historyJson = await historyRes.json();

                    if (!historyJson.chart || !historyJson.chart.result || historyJson.chart.result.length === 0) {
                        throw new Error("查無此股票代號或無數據");
                    }

                    const result = historyJson.chart.result[0];
                    const timestamps = result.timestamp;
                    const closeQuotes = result.indicators.quote[0].close;

                    if (!timestamps || !closeQuotes) throw new Error("數據格式錯誤");

                    const prices = timestamps.map((t, i) => {
                        const price = closeQuotes[i];
                        if (price === null || price === undefined) return null;
                        const date = new Date(t * 1000).toISOString().split('T')[0];
                        // FNG Mapping Logic: Fallback to previous known value if missing for a specific day to avoid gaps
                        let fngVal = fngMap.get(date);
                        if (fngVal === undefined) fngVal = 50; // Default Neutral

                        return {
                            date: date,
                            price: price,
                            fng: fngVal
                        };
                    }).filter(item => item !== null).sort((a, b) => new Date(a.date) - new Date(b.date));

                    if (prices.length === 0) throw new Error("無有效股價數據");

                    setFullStockData(prices);
                    // Initial set based on current TimeRange (triggered by useEffect dependency or shared state update)
                    // We will let the useEffect([timeRange, fullStockData]) handle the updating of 'stockData'

                    // 3. 獲取大盤概況
                    const fetchYahooPrice = async (symbol) => {
                        try {
                            const yfUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1d`;
                            const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yfUrl));
                            const data = await res.json();
                            const meta = data.chart.result[0].meta;
                            const price = meta.regularMarketPrice;
                            const prevClose = meta.chartPreviousClose || meta.previousClose || price;
                            return { price, prevClose, change: price - prevClose, changePercent: ((price - prevClose) / prevClose) * 100 };
                        } catch (e) { return null; }
                    };

                    // Drive cards from user-editable list
                    Promise.all(usCards.cards.map(c => fetchYahooPrice(c.symbol)))
                        .then(results => {
                            const map = {};
                            usCards.cards.forEach((c, i) => { map[c.id] = results[i]; });
                            setIndices(map);
                        })
                        .catch(e => console.error("Indices Fetch Error:", e));

                    // 4. 獲取新聞
                    const rssUrl = 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664';
                    fetch(`https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssUrl)}`)
                        .then(r => r.json())
                        .then(d => { if (d.status === 'ok') setStockNews(d.items); })
                        .catch(e => console.error("News Error:", e));

                } catch (e) {
                    console.error("Global Stock Error:", e);
                    setError(e.message || "數據載入失敗");
                } finally {
                    setLoading(false);
                }
            };

            useEffect(() => {
                fetchStockData();
            }, [selectedSymbol]);

            // Re-fetch just the card prices when the user edits the card list (avoids re-fetching FNG/history)
            useEffect(() => {
                const fetchOne = async (symbol) => {
                    try {
                        const yfUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1d`;
                        const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yfUrl));
                        const data = await res.json();
                        const meta = data.chart.result[0].meta;
                        const price = meta.regularMarketPrice;
                        const prevClose = meta.chartPreviousClose || meta.previousClose || price;
                        return { price, prevClose, change: price - prevClose, changePercent: ((price - prevClose) / prevClose) * 100 };
                    } catch (e) { return null; }
                };
                Promise.all(usCards.cards.map(c => fetchOne(c.symbol)))
                    .then(results => {
                        const map = {};
                        usCards.cards.forEach((c, i) => { map[c.id] = results[i]; });
                        setIndices(map);
                    });
            }, [usCards.cards]);

            // Filter Data based on Time Range (Local Filtering)
            useEffect(() => {
                if (!fullStockData || fullStockData.length === 0) return;

                const now = new Date();
                let filterDate = new Date();
                let isAll = false;

                if (timeRange === '1M') filterDate.setMonth(now.getMonth() - 1);
                else if (timeRange === '6M') filterDate.setMonth(now.getMonth() - 6);
                else if (timeRange === '1Y') isAll = true;
                else if (timeRange === 'ALL') isAll = true;

                const filtered = fullStockData.filter(d => isAll || new Date(d.date) >= filterDate);
                setStockData(filtered);
            }, [timeRange, fullStockData]);

            // handle Watchlist Toggle
            const toggleWatchlist = () => {
                if (!onUpdateWatchlist) return;
                const currentList = userInfo.watchlist || [];
                // Check if symbol exists (simple check, assume stock symbols are unique enough or use prefix in real app)
                // For this app, let's just store the symbol string "SPY", "BTC" etc.
                const exists = currentList.includes(selectedSymbol);
                let newList;
                if (exists) {
                    newList = currentList.filter(s => s !== selectedSymbol);
                } else {
                    newList = [...currentList, selectedSymbol];
                }
                onUpdateWatchlist(newList);
            };

            const isWatchlisted = (userInfo.watchlist || []).includes(selectedSymbol);

            const getSuggestion = (score) => {
                const val = parseInt(score);
                if (val <= 25) return {
                    title: "極度恐懼 (Extreme Fear)",
                    action: "強力買入",
                    desc: "市場極度恐慌，股價可能被低估，是長期投資的買入良機。",
                    bg: "bg-red-900/30",
                    border: "border-red-500",
                    text: "text-red-400"
                };
                if (val <= 45) return {
                    title: "恐懼 (Fear)",
                    action: "分批買入",
                    desc: "市場情緒低迷，投資者趨於保守，適合執行 DCA 策略。",
                    bg: "bg-orange-900/30",
                    border: "border-orange-500",
                    text: "text-orange-400"
                };
                if (val <= 55) return {
                    title: "中立 (Neutral)",
                    action: "持有/觀望",
                    desc: "市場缺乏明確方向，建議保持現有部位，觀察後續變化。",
                    bg: "bg-gray-800",
                    border: "border-gray-600",
                    text: "text-gray-400"
                };
                if (val <= 75) return {
                    title: "貪婪 (Greed)",
                    action: "停止買入",
                    desc: "市場情緒樂觀，風險逐漸升高，不建議追高。",
                    bg: "bg-green-900/30",
                    border: "border-green-500",
                    text: "text-green-400"
                };
                return {
                    title: "極度貪婪 (Extreme Greed)",
                    action: "考慮止盈",
                    desc: "市場極度樂觀，隨時可能回調，建議分批獲利了結。",
                    bg: "bg-blue-900/30",
                    border: "border-blue-500",
                    text: "text-blue-400"
                };
            };
            const suggestion = fngData ? getSuggestion(fngData.value) : null;
            useThemeVersion();

            // Stock Price vs FNG Chart (Mixed Chart)
            const StockFNGChart = ({ data, symbol }) => {
                if (!data || data.length === 0) return null;

                const chartData = {
                    labels: data.map(d => d.date),
                    datasets: [
                        {
                            label: `${symbol} 價格`,
                            data: data.map(d => d.price),
                            borderColor: CHART.line,
                            backgroundColor: makePriceGradient,
                            yAxisID: 'y',
                            borderWidth: 1.6,
                            fill: true,
                            tension: 0.25,
                            ...fearPointProps(data.map(d => d.fng)),
                        }
                    ]
                };

                const options = {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => {
                                    const idx = context.dataIndex;
                                    const d = data[idx];
                                    return [
                                        `價格: $${d.price.toLocaleString()}`,
                                        `市場 FNG: ${d.fng}`,
                                        d.fng <= 40 ? '🔴 建議 DCA 買入點' : '⚪ 觀望'
                                    ];
                                }
                            }
                        },
                        title: figureTitle(`${symbol} 價格走勢與市場恐懼指數 (紅/橘點代表市場恐懼)`)
                    },
                    scales: {
                        y: { type: 'linear', display: true, position: 'left', grid: { color: CHART.grid }, ticks: { color: CHART.tick } },
                        x: figureX()
                    }
                };
                return <ChartComponent data={chartData} options={options} />;
            };

            return (
                <div>
                    <div className="py-2.5 max-w-[520px]" style={{ borderBottom: '1px solid var(--rule)' }}>
                        <SymbolSearch
                            symbol={selectedSymbol}
                            onSearch={setSelectedSymbol}
                            isLocked={!userInfo.isPremium}
                            userEmail={userInfo.email}
                        />
                    </div>

                    <EditableCardGrid
                        title="美股指數 / ETF"
                        mobileRow
                        cards={usCards.cards}
                        editMode={usEditMode}
                        onToggleEdit={() => setUsEditMode(!usEditMode)}
                        onMove={usCards.moveCard}
                        onRemove={usCards.removeCard}
                        onReset={usCards.resetCards}
                        onAdd={() => {
                            const sym = (prompt('輸入美股代號（例如 AAPL、TSLA、VOO）：') || '').trim().toUpperCase();
                            if (!sym) return;
                            const name = (prompt('顯示名稱（可留空）：') || '').trim() || sym;
                            usCards.addCard({ id: sym.toLowerCase(), symbol: sym, name });
                        }}
                    >
                        {usCards.cards.map(c => (
                            <IndexPriceCard
                                key={c.id}
                                name={c.name}
                                symbol={c.symbol}
                                logo={c.logo}
                                data={indices[c.id]}
                                active={selectedSymbol === c.symbol}
                                editMode={usEditMode}
                                onClick={() => setSelectedSymbol(c.symbol)}
                                currency="$"
                            />
                        ))}
                    </EditableCardGrid>

                    <SentimentToggleCard
                        storageKey="us-sentiment-view"
                        fngTitle="美股恐懼與貪婪指數"
                        macdTitle={`週線 MACD 背離（${selectedSymbol}）`}
                        market="us"
                        symbol={selectedSymbol}
                        label={selectedSymbol}
                    >
                        {loading ? (
                            <div className="py-12 flex justify-center"><RefreshCw className="animate-spin" size={28} style={{ color: 'var(--ink-3)' }} /></div>
                        ) : error ? (
                            <div className="flex items-start gap-3 py-6">
                                <AlertTriangle size={22} className="shrink-0 mt-0.5" style={{ color: 'var(--down)' }} />
                                <div>
                                    <div className="text-[15px]" style={{ color: 'var(--down)' }}>{error}</div>
                                    <button onClick={fetchStockData} className="fs-btn sm mt-3">重試</button>
                                </div>
                            </div>
                        ) : (
                            <>
                                <div className="grid md:grid-cols-[7fr_5fr] gap-5 md:gap-10 items-start">
                                    {fngData && <FearGreedGauge fngValue={fngData.value} classification={fngData.classification} />}
                                    {suggestion && (
                                        <DcaStrategy suggestion={suggestion} label="DCA 策略建議" actions={MARKET_ACTIONS}>
                                            {fngHistory && fngHistory.length > 0 && <SentimentScarcityBar data={fngHistory.map(d => d.y)} title="過去 365 天美股買入機會分佈" />}
                                        </DcaStrategy>
                                    )}
                                </div>
                                {fngData && (
                                    <div className="mt-7">
                                        <AIAdviceBlock
                                            assetName={`美股 (${selectedSymbol})`}
                                            marketData={`資產: ${selectedSymbol}\n恐懼貪婪指數: ${fngData.value} (${fngData.classification})`}
                                            priceStats={stockData && stockData.length > 0 ? {
                                                current: stockData[stockData.length - 1].price,
                                                high: Math.max(...stockData.map(d => d.price)),
                                                low: Math.min(...stockData.map(d => d.price))
                                            } : null}
                                            isLocked={!userInfo.isPremium}
                                        />
                                    </div>
                                )}
                            </>
                        )}
                    </SentimentToggleCard>

                    {!loading && !error && stockData && stockData.length > 0 && (
                        <HistorySection
                            title={`${selectedSymbol} 股價與恐懼指數走勢`}
                            watch={onUpdateWatchlist && <WatchStar on={isWatchlisted} onClick={toggleWatchlist} />}
                            onAdvanced={() => setShowAdvanced(true)}
                            range={timeRange}
                            onRange={setTimeRange}
                        >
                            <StockFNGChart data={stockData} symbol={selectedSymbol} />
                        </HistorySection>
                    )}

                    <NewsBlock title="美股財經頭條 (CNBC)" news={stockNews} loading={loading} error={error} ready={!loading && !error} isLocked={!userInfo.isPremium} />

                    <SourceNote>資料來源: CNN Business, Yahoo Finance, CNBC</SourceNote>
                    {showAdvanced && (
                        <TechnicalChartModal
                            symbol={selectedSymbol}
                            type="US"
                            onClose={() => setShowAdvanced(false)}
                        />
                    )}
                </div>
            );
        };

        // --- Google News RSS ---
        // rss2json 已抓不到 Google News(約 7 秒後回 500),改走自家 CORS proxy + DOMParser,
        // 失敗才退回 rss2json。回傳 [{ title, link, pubDate, thumbnail }]
        const fetchWithTimeout = async (url, ms) => {
            const ctrl = new AbortController();
            const timer = setTimeout(() => ctrl.abort(), ms);
            try {
                return await fetch(url, { signal: ctrl.signal });
            } finally {
                clearTimeout(timer);
            }
        };
        const fetchGoogleNews = async (query, limit = 6) => {
            const rssUrl = 'https://news.google.com/rss/search?q=' + encodeURIComponent(query) + '&hl=zh-TW&gl=TW&ceid=TW:zh-Hant';
            try {
                const res = await fetchWithTimeout('https://cors.hellokai07.com/?' + encodeURIComponent(rssUrl), 8000);
                if (res.ok) {
                    const doc = new DOMParser().parseFromString(await res.text(), 'text/xml');
                    const items = [...doc.querySelectorAll('item')].slice(0, limit).map(item => ({
                        title: item.querySelector('title')?.textContent || '',
                        link: item.querySelector('link')?.textContent || '',
                        pubDate: item.querySelector('pubDate')?.textContent || '',
                        thumbnail: '',
                    }));
                    if (items.length) return items;
                }
            } catch (e) {
                console.warn('Google News via proxy failed:', e);
            }
            const res = await fetchWithTimeout(`https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssUrl)}`, 8000);
            const json = await res.json();
            if (json.status !== 'ok' || !Array.isArray(json.items)) throw new Error(json.message || '新聞來源暫時無法使用');
            return json.items.slice(0, limit).map(item => ({
                title: item.title || '',
                link: item.link || '',
                pubDate: item.pubDate || '',
                thumbnail: item.thumbnail || item.enclosure?.link || '',
            }));
        };

        // --- RSI 計算與分類工具 ---
        const calculateRSI = (prices, period = 14) => {
            if (prices.length < period + 1) return [];
            let gains = 0;
            let losses = 0;
            for (let i = 1; i <= period; i++) {
                const change = prices[i] - prices[i - 1];
                if (change > 0) gains += change;
                else losses -= change;
            }
            let avgGain = gains / period;
            let avgLoss = losses / period;
            const rsiArray = [];
            // Initial RSI
            let rs = avgGain / avgLoss;
            let rsi = 100 - (100 / (1 + rs));
            rsiArray.push({ index: period, rsi, price: prices[period] });

            // Smoothed RSI
            for (let i = period + 1; i < prices.length; i++) {
                const change = prices[i] - prices[i - 1];
                let gain = change > 0 ? change : 0;
                let loss = change < 0 ? -change : 0;
                avgGain = ((avgGain * (period - 1)) + gain) / period;
                avgLoss = ((avgLoss * (period - 1)) + loss) / period;
                rs = avgGain / avgLoss;
                rsi = 100 - (100 / (1 + rs));
                rsiArray.push({ index: i, rsi, price: prices[i] });
            }
            return rsiArray;
        };

        const getRsiClassification = (rsi) => {
            if (rsi <= 25) return "極度恐懼";
            if (rsi <= 40) return "恐懼";
            if (rsi <= 60) return "中立";
            if (rsi <= 75) return "貪婪";
            return "極度貪婪";
        };

        // --- 臺股儀表板元件 ---
        const TaiwanDashboard = ({ userInfo = { isPremium: false, watchlist: [] }, onUpdateWatchlist }) => {
            const [showAdvanced, setShowAdvanced] = useState(false);
            const [selectedSymbol, setSelectedSymbol] = useState('0050');
            const [timeRange, setTimeRange] = useState('1Y');
            const [rsiData, setRsiData] = useState(null);
            const [indices, setIndices] = useState({});
            const [news, setNews] = useState([]);
            const [newsLoading, setNewsLoading] = useState(true);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [dataSource, setDataSource] = useState('Yahoo');
            // Editable card list (Yahoo tickers — TW codes need .TW suffix unless it's an index like ^TWII)
            const twDefaults = [
                { id: 'tsmc', symbol: '2330.TW', name: '台積電 (2330)' },
                { id: 'tw50', symbol: '0050.TW', name: '元大台灣50 (0050)' },
                { id: 'tw56', symbol: '0056.TW', name: '元大高股息 (0056)' },
            ];
            const twCards = useEditableCards('tw-dashboard-cards', twDefaults);
            const [twEditMode, setTwEditMode] = useState(false);
            useThemeVersion();

            const getSuggestion = (rsi) => {
                if (rsi <= 25) return {
                    title: "極度恐懼 (Extreme Fear)",
                    action: "強力買入",
                    desc: "RSI 顯示市場極度超賣，為歷史低點，建議強力買入。",
                    bg: "bg-red-900/30",
                    border: "border-red-500",
                    text: "text-red-400"
                };
                if (rsi <= 40) return {
                    title: "恐懼 (Fear)",
                    action: "分批買入",
                    desc: "RSI 處於低檔，市場情緒保守，適合執行 DCA 策略。",
                    bg: "bg-orange-900/30",
                    border: "border-orange-50",
                    text: "text-orange-400"
                };
                if (rsi <= 60) return {
                    title: "中立 (Neutral)",
                    action: "持有/觀望",
                    desc: "RSI 處於中性區間，建議觀望或持有。",
                    bg: "bg-gray-800",
                    border: "border-gray-600",
                    text: "text-gray-400"
                };
                if (rsi <= 75) return {
                    title: "貪婪 (Greed)",
                    action: "停止買入",
                    desc: "RSI 顯示市場過熱，不宜追高。",
                    bg: "bg-green-900/30",
                    border: "border-green-500",
                    text: "text-green-400"
                };
                return {
                    title: "極度貪婪 (Extreme Greed)",
                    action: "考慮止盈",
                    desc: "RSI 顯示市場極度超買，隨時可能回調。",
                    bg: "bg-blue-900/30",
                    border: "border-blue-500",
                    text: "text-blue-400"
                };
            };

            const fetchData = async () => {
                setLoading(true);
                setError(null);
                try {
                    // Helper: Yahoo Price Fetcher (per-card try/catch so one failure doesn't drop the whole row)
                    const fetchYahooPrice = async (ticker) => {
                        try {
                            const yfUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=1d`;
                            const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yfUrl));
                            if (!res.ok) return null;
                            const data = await res.json();
                            const meta = data?.chart?.result?.[0]?.meta;
                            if (!meta) return null;
                            const price = meta.regularMarketPrice;
                            const prevClose = meta.chartPreviousClose || meta.previousClose || price;
                            if (price == null) return null;
                            return { price, prevClose, change: price - prevClose, changePercent: prevClose ? ((price - prevClose) / prevClose) * 100 : 0 };
                        } catch (e) { return null; }
                    };

                    // 1. Fetch Headline Indices (Parallel) — driven by user-editable card list
                    Promise.all(twCards.cards.map(c => fetchYahooPrice(c.symbol)))
                        .then(results => {
                            const map = {};
                            twCards.cards.forEach((c, i) => { map[c.id] = results[i]; });
                            setIndices(map);
                        })
                        .catch(e => console.error("Indices Error:", e));

                    // 2. Fetch Selected Symbol Data
                    // Strategy: Try Fugle for Real-time Quote (if allowed key), Yahoo for History (RSI).

                    // Prepare tickers
                    const yahooSymbol = selectedSymbol.includes('.') ? selectedSymbol : `${selectedSymbol}.TW`;

                    // A. Fetch History (Yahoo) - Needed for RSI & Chart
                    let range = '1y';
                    if (timeRange === '1M') range = '1mo';
                    else if (timeRange === '6M') range = '6mo';
                    else if (timeRange === '1Y') range = '1y';
                    else if (timeRange === 'ALL') range = 'max';

                    const historyUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${yahooSymbol}?interval=1d&range=${range}`;
                    const historyRes = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(historyUrl));
                    const historyJson = await historyRes.json();

                    if (!historyJson.chart || !historyJson.chart.result) throw new Error("無效的股票代號 (Yahoo)");

                    const timestamps = historyJson.chart.result[0].timestamp;
                    const closes = historyJson.chart.result[0].indicators.quote[0].close;

                    // Filter Nulls
                    const cleanData = timestamps.map((t, i) => ({ t, c: closes[i] })).filter(d => d.c !== null);
                    const cleanCloses = cleanData.map(d => d.c);
                    const cleanTimestamps = cleanData.map(d => d.t);

                    if (cleanCloses.length < 15) throw new Error("歷史數據不足");

                    const rsiValues = calculateRSI(cleanCloses);
                    const currentRSI = rsiValues[rsiValues.length - 1];
                    const currentPriceFromYahoo = cleanCloses[cleanCloses.length - 1];

                    // B. Try Fugle Realtime Quote (skip if key is still placeholder)
                    let fuglePrice = null;
                    let fugleSource = false;
                    const fugleToken = "__FUGLE_KEY__";
                    if (!fugleToken.includes("FUGLE_KEY")) {
                        try {
                            const fugleUrl = `https://api.fugle.tw/realtime/v0.3/intraday/quote?symbolId=${selectedSymbol}&apiToken=${fugleToken}`;
                            const fRes = await fetch(fugleUrl);
                            if (fRes.ok) {
                                const fData = await fRes.json();
                                if (fData.data && fData.data.quote && fData.data.quote.trade) {
                                    fuglePrice = fData.data.quote.trade.price;
                                    fugleSource = true;
                                }
                            }
                        } catch (e) {
                            console.warn("Fugle Fetch Failed:", e);
                        }
                    }

                    setDataSource(fugleSource ? 'Fugle API' : 'Yahoo Finance');

                    // Merge Data
                    // If Fugle has newer price, update the last history point or display separately?
                    // For simplicity, we trust Yahoo for the history chart, but could overlay real-time price.

                    setRsiData({
                        value: Math.round(currentRSI.rsi),
                        classification: getRsiClassification(currentRSI.rsi),
                        price: fuglePrice || currentPriceFromYahoo,
                        history: rsiValues.map((d, i) => ({
                            x: cleanTimestamps[d.index] * 1000,
                            y: d.price,
                            rsi: d.rsi
                        }))
                    });


                } catch (e) {
                    console.error("Taiwan data error:", e);
                    setError(`無法獲取 ${selectedSymbol} 數據: ${e.message}`);
                } finally {
                    setLoading(false);
                }
            };

            // 臺股新聞獨立抓取,不阻塞 RSI / 圖表的載入
            const fetchNews = async () => {
                setNewsLoading(true);
                try {
                    setNews(await fetchGoogleNews(`${selectedSymbol} 股票新聞`));
                } catch (e) {
                    console.warn('TW news fetch error:', e);
                    setNews([]);
                } finally {
                    setNewsLoading(false);
                }
            };

            useEffect(() => {
                fetchData();
            }, [selectedSymbol, timeRange]);

            useEffect(() => {
                fetchNews();
            }, [selectedSymbol]);

            // Refetch just the card prices when the user edits the card list
            useEffect(() => {
                const fetchOne = async (ticker) => {
                    try {
                        const yfUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=1d`;
                        const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yfUrl));
                        const data = await res.json();
                        const meta = data.chart.result[0].meta;
                        const price = meta.regularMarketPrice;
                        const prevClose = meta.chartPreviousClose || meta.previousClose || price;
                        return { price, prevClose, change: price - prevClose, changePercent: ((price - prevClose) / prevClose) * 100 };
                    } catch (e) { return null; }
                };
                Promise.all(twCards.cards.map(c => fetchOne(c.symbol)))
                    .then(results => {
                        const map = {};
                        twCards.cards.forEach((c, i) => { map[c.id] = results[i]; });
                        setIndices(map);
                    });
            }, [twCards.cards]);

            // Watchlist Logic
            const toggleWatchlist = () => {
                if (!onUpdateWatchlist) return;
                const currentList = userInfo.watchlist || [];
                const exists = currentList.includes(selectedSymbol);
                let newList;
                if (exists) {
                    newList = currentList.filter(s => s !== selectedSymbol);
                } else {
                    newList = [...currentList, selectedSymbol];
                }
                onUpdateWatchlist(newList);
            };

            const isWatchlisted = (userInfo.watchlist || []).includes(selectedSymbol);

            if (loading) return <div className="flex justify-center items-center h-64"><RefreshCw className="animate-spin" size={28} style={{ color: 'var(--ink-3)' }} /></div>;
            if (error) return (
                <section className="fs-section mt-6" style={{ borderTopColor: 'var(--down)' }}>
                    <div className="flex items-start gap-3">
                        <AlertTriangle size={22} className="shrink-0 mt-1" style={{ color: 'var(--down)' }} />
                        <div>
                            <h3 className="fs-title-sm">臺股數據載入失敗</h3>
                            <p className="text-sm my-2" style={{ color: 'var(--ink-2)' }}>{error}</p>
                            <button onClick={fetchData} className="fs-btn sm">重試連線</button>
                        </div>
                    </div>
                </section>
            );

            const suggestion = rsiData ? getSuggestion(rsiData.value) : null;

            // Chart Config
            const chartData = {
                labels: rsiData.history.map(d => new Date(d.x).toLocaleDateString()),
                datasets: [{
                    label: `${selectedSymbol} 股價`,
                    data: rsiData.history.map(d => d.y),
                    borderColor: CHART.line,
                    backgroundColor: makePriceGradient,
                    borderWidth: 1.6,
                    ...fearPointProps(rsiData.history.map(d => d.rsi)),
                    fill: true,
                    tension: 0.25
                }]
            };

            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const idx = context.dataIndex;
                                const d = rsiData.history[idx];
                                const rsi = d?.rsi?.toFixed(1) || 'N/A';
                                return [
                                    `價格: ${formatPrice(context.parsed.y)}`,
                                    `市場 RSI: ${rsi}`,
                                    (d && d.rsi <= 44) ? '🔴 建議 DCA 買入點' : '⚪ 觀望'
                                ];
                            }
                        }
                    },
                    title: figureTitle(`${selectedSymbol} 歷史走勢 (紅/橘點為 RSI 買入訊號)`)
                },
                scales: {
                    x: figureX(),
                    y: { grid: { color: CHART.grid }, ticks: { color: CHART.tick } }
                }
            };

            return (
                <div>
                    {/* 大盤指數卡片 */}
                    <div className="py-2.5 max-w-[520px]" style={{ borderBottom: '1px solid var(--rule)' }}>
                        <SymbolSearch
                            symbol={selectedSymbol}
                            onSearch={setSelectedSymbol}
                            placeholder="輸入代號 (e.g. 2330)"
                            isLocked={!userInfo.isPremium}
                            userEmail={userInfo.email}
                        />
                    </div>

                    <EditableCardGrid
                        title="台股指數 / ETF"
                        mobileRow
                        cards={twCards.cards}
                        editMode={twEditMode}
                        onToggleEdit={() => setTwEditMode(!twEditMode)}
                        onMove={twCards.moveCard}
                        onRemove={twCards.removeCard}
                        onReset={twCards.resetCards}
                        onAdd={() => {
                            let sym = (prompt('輸入台股代號（例如 2330、0056；指數請含 ^，例如 ^TWII）：') || '').trim();
                            if (!sym) return;
                            // Auto-append .TW for plain numeric codes
                            if (/^\d+$/.test(sym)) sym = sym + '.TW';
                            const name = (prompt('顯示名稱（可留空）：') || '').trim() || sym;
                            twCards.addCard({ id: sym.toLowerCase().replace(/\W+/g, '_'), symbol: sym, name });
                        }}
                    >
                        {twCards.cards.map(c => (
                            <IndexPriceCard
                                key={c.id}
                                name={c.name}
                                symbol={c.symbol}
                                logo={c.logo}
                                data={indices[c.id]}
                                active={selectedSymbol === c.symbol}
                                editMode={twEditMode}
                                onClick={() => setSelectedSymbol(c.symbol)}
                                currency=""
                            />
                        ))}
                    </EditableCardGrid>

                    {/* 情緒指標 + 策略 */}
                    <SentimentToggleCard
                        storageKey="tw-sentiment-view"
                        fngTitle={`臺股情緒指標 (基於 ${selectedSymbol} RSI)`}
                        macdTitle={`週線 MACD 背離（${selectedSymbol}）`}
                        market="tw"
                        symbol={selectedSymbol}
                        label={selectedSymbol}
                        subtitle={
                            <p className="text-[13px] mt-2 max-w-[60ch] leading-relaxed" style={{ color: 'var(--ink-2)' }}>
                                此指標並非恐懼貪婪指數。數值由『{selectedSymbol}』的 14 日 RSI 強弱指標計算得出。RSI 低於 30 代表市場超賣 (恐懼)，高於 70 代表市場超買 (貪婪)。
                                <span className="block mt-1" style={{ color: 'var(--ink-3)' }}>數據來源: {dataSource}</span>
                            </p>
                        }
                    >
                        <div className="grid md:grid-cols-[7fr_5fr] gap-5 md:gap-10 items-start">
                            {rsiData && <FearGreedGauge fngValue={rsiData.value} classification={rsiData.classification} />}
                            {suggestion && (
                                <DcaStrategy suggestion={suggestion} label="臺股 DCA 策略" actions={MARKET_ACTIONS}>
                                    {rsiData && rsiData.history && <SentimentScarcityBar data={rsiData.history.map(d => d.rsi)} title="過去 1 年臺股買入機會分佈 (RSI)" />}
                                </DcaStrategy>
                            )}
                        </div>
                        {rsiData && (
                            <div className="mt-7">
                                <AIAdviceBlock
                                    assetName={`台股 (${selectedSymbol})`}
                                    marketData={`RSI(14): ${rsiData.value} (${rsiData.classification})`}
                                    priceStats={{
                                        current: rsiData.price,
                                        high: Math.max(...rsiData.history.map(d => d.y)),
                                        low: Math.min(...rsiData.history.map(d => d.y))
                                    }}
                                    isLocked={!userInfo.isPremium}
                                />
                            </div>
                        )}
                    </SentimentToggleCard>

                    {/* 歷史走勢圖 */}
                    {rsiData && (
                        <HistorySection
                            title={`${selectedSymbol} 歷史走勢與買入訊號`}
                            watch={onUpdateWatchlist && <WatchStar on={isWatchlisted} onClick={toggleWatchlist} />}
                            onAdvanced={() => setShowAdvanced(true)}
                            range={timeRange}
                            onRange={setTimeRange}
                            height="h-[350px]"
                        >
                            <ChartComponent data={chartData} options={chartOptions} />
                        </HistorySection>
                    )}

                    {/* 新聞區 */}
                    <NewsBlock title="臺股財經頭條 (Google News)" news={news} loading={newsLoading} error={null} ready={!loading && !error} isLocked={!userInfo.isPremium} />

                    <SourceNote>資料來源: Yahoo Finance, Google News, Smart DCA Bot (0050 歷史數據運算)</SourceNote>
                    {showAdvanced && (
                        <TechnicalChartModal
                            symbol={selectedSymbol.includes('.') ? selectedSymbol : selectedSymbol + '.TW'}
                            type="TW"
                            onClose={() => setShowAdvanced(false)}
                        />
                    )}
                </div>
            );
        };




        // --- 加密貨幣儀表板 (Updated to use CoinMarketCap) ---
        const CryptoDashboard = ({ notificationsEnabled, toggleNotifications, userInfo = { isPremium: false, watchlist: [] }, onUpdateWatchlist }) => {
            const [showAdvanced, setShowAdvanced] = useState(false);
            const [selectedCoin, setSelectedCoin] = useState('bitcoin'); // slug or id
            const [timeRange, setTimeRange] = useState('1Y');
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [historicalData, setHistoricalData] = useState([]);
            const [fullHistoricalData, setFullHistoricalData] = useState([]);
            const [currentFNG, setCurrentFNG] = useState(null);
            const [lastUpdated, setLastUpdated] = useState(null);
            const [prices, setPrices] = useState(null);
            const [newsData, setNewsData] = useState([]);
            const [newsLoading, setNewsLoading] = useState(true);
            const [newsError, setNewsError] = useState(null);
            // Editable card list — slug = CoinGecko id (also used by CMC proxy)
            const cryptoDefaults = [
                { id: 'bitcoin', symbol: 'bitcoin', name: 'BTC', logo: 'https://cryptologos.cc/logos/bitcoin-btc-logo.png' },
                { id: 'ethereum', symbol: 'ethereum', name: 'ETH', logo: 'https://cryptologos.cc/logos/ethereum-eth-logo.png' },
            ];
            const cryptoCards = useEditableCards('crypto-dashboard-cards', cryptoDefaults);
            const themeV = useThemeVersion();
            const [cryptoEditMode, setCryptoEditMode] = useState(false);
            const [cryptoView, setCryptoView] = useState('market'); // 'market' | 'onchain'

            // Fetch current prices from Binance (no key, CORS-friendly, same source as history)
            const fetchCurrentPrices = async (slugs = ['bitcoin', 'ethereum']) => {
                try {
                    const results = await Promise.all(slugs.map(async (slug) => {
                        const binSym = getBinanceSymbol(slug);
                        if (!binSym) return [slug, null];
                        try {
                            const res = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${binSym}`);
                            if (!res.ok) return [slug, null];
                            const d = await res.json();
                            return [slug, {
                                usd: +d.lastPrice,
                                usd_24h_change: +d.priceChangePercent,
                                symbol: binSym.replace(/USDT$/, ''),
                            }];
                        } catch (e) { return [slug, null]; }
                    }));
                    const newPrices = {};
                    for (const [slug, p] of results) if (p) newPrices[slug] = p;
                    setPrices(newPrices);
                } catch (e) {
                    console.warn("Binance Price Fetch Error:", e);
                }
            };

            const fetchData = async () => {
                setLoading(true);
                setError(null);

                // Fetch Prices first — include all user cards plus selected coin
                const slugs = cryptoCards.cards.map(c => c.symbol);
                if (!slugs.includes(selectedCoin)) slugs.push(selectedCoin);
                fetchCurrentPrices(slugs);

                try {
                    // 1. Fetch FNG (CoinMarketCap 優先,失敗退回 alternative.me)
                    const fngList = await fetchCryptoFNG(365);
                    const fngJson = { data: fngList };

                    // 2. Process FNG (must happen before history merge)
                    const fngMap = new Map();
                    fngJson.data.forEach(item => {
                        const date = new Date(item.timestamp * 1000).toISOString().split('T')[0];
                        let rating = item.value_classification;
                        const ratingMap = {
                            "Extreme Fear": "極度恐懼", "extreme fear": "極度恐懼",
                            "Fear": "恐懼", "fear": "恐懼",
                            "Neutral": "中立", "neutral": "中立",
                            "Greed": "貪婪", "greed": "貪婪",
                            "Extreme Greed": "極度貪婪", "extreme greed": "極度貪婪"
                        };
                        rating = ratingMap[rating] || rating;
                        fngMap.set(date, { value: parseInt(item.value), classification: rating });
                    });

                    // Set Current FNG
                    const currentItem = fngJson.data[0];
                    let currentRating = currentItem.value_classification;
                    const ratingMap = { "Extreme Fear": "極度恐懼", "extreme fear": "極度恐懼", "Fear": "恐懼", "fear": "恐懼", "Neutral": "中立", "neutral": "中立", "Greed": "貪婪", "greed": "貪婪", "Extreme Greed": "極度貪婪", "extreme greed": "極度貪婪" };
                    setCurrentFNG({ ...currentItem, value_classification: ratingMap[currentRating] || currentRating });

                    // 3. Fetch Historical Prices — Binance klines (single source, no key, real daily OHLC)
                    const binSym = getBinanceSymbol(selectedCoin);
                    if (!binSym) throw new Error(`不支援的幣種：${selectedCoin}（不在 Binance 上市）`);
                    const days = 365;
                    const limit = Math.min(1000, days + 5);
                    const binResponse = await fetch(`https://api.binance.com/api/v3/klines?symbol=${binSym}&interval=1d&limit=${limit}`);
                    if (!binResponse.ok) throw new Error(`Binance API Error (status ${binResponse.status})`);
                    const klines = await binResponse.json();

                    // Merge price + FNG by date
                    const mergedData = klines.map(k => {
                        const timestamp = k[0];
                        const dateStr = new Date(timestamp).toISOString().split('T')[0];
                        const fng = fngMap.get(dateStr) || { value: 50, classification: 'Neutral' };
                        return {
                            date: dateStr,
                            timestamp,
                            price: +k[4], // close
                            open: +k[1],
                            high: +k[2],
                            low: +k[3],
                            fng: fng.value,
                            classification: fng.classification
                        };
                    });

                    setFullHistoricalData(mergedData);
                    setLastUpdated(new Date());

                } catch (err) {
                    console.error("Fetch Data Failed:", err);
                    setError(err.message + " (可能需等待由API限制)");
                } finally {
                    setLoading(false);
                }
            };

            const fetchNews = async () => {
                setNewsLoading(true);
                try {
                    setNewsData(await fetchGoogleNews('加密貨幣'));
                } catch (e) {
                    setNewsError(e.message);
                } finally {
                    setNewsLoading(false);
                }
            };

            useEffect(() => {
                fetchData();
                fetchNews();
            }, [selectedCoin]);

            // Local Filtering Effect
            useEffect(() => {
                if (!fullHistoricalData || fullHistoricalData.length === 0) return;

                const now = new Date();
                let filterDate = new Date();
                let isAll = false;

                if (timeRange === '1M') filterDate.setMonth(now.getMonth() - 1);
                else if (timeRange === '6M') filterDate.setMonth(now.getMonth() - 6);
                else if (timeRange === '1Y') isAll = true;
                else if (timeRange === 'ALL') isAll = true;

                const filtered = fullHistoricalData.filter(d => isAll || new Date(d.date) >= filterDate);
                setHistoricalData(filtered);

            }, [timeRange, fullHistoricalData]);

            // Refetch card prices when the user edits the crypto card list
            useEffect(() => {
                const slugs = cryptoCards.cards.map(c => c.symbol);
                if (!slugs.includes(selectedCoin)) slugs.push(selectedCoin);
                if (slugs.length > 0) fetchCurrentPrices(slugs);
            }, [cryptoCards.cards]);

            // Watchlist Logic
            const toggleWatchlist = () => {
                if (!onUpdateWatchlist) return;
                const currentList = userInfo.watchlist || [];
                const exists = currentList.includes(selectedCoin);
                let newList;
                if (exists) {
                    newList = currentList.filter(s => s !== selectedCoin);
                } else {
                    newList = [...currentList, selectedCoin];
                }
                onUpdateWatchlist(newList);
            };

            const isWatchlisted = (userInfo.watchlist || []).includes(selectedCoin);


            const chartData = useMemo(() => {
                if (!historicalData.length) return null;
                const labels = historicalData.map(d => d.date);
                const prices = historicalData.map(d => d.price);

                return {
                    labels,
                    datasets: [{
                        label: `${selectedCoin.toUpperCase()} 價格 (USD)`,
                        data: prices,
                        borderColor: CHART.line,
                        backgroundColor: makePriceGradient,
                        borderWidth: 1.6,
                        // Fear buy points carry a shape as well as a colour: extreme fear = filled triangle, fear = hollow ring
                        pointStyle: historicalData.map(d => d.fng <= 25 ? 'triangle' : 'circle'),
                        pointBackgroundColor: historicalData.map(d => d.fng <= 25 ? CHART.fear : 'rgba(0,0,0,0)'),
                        pointBorderColor: historicalData.map(d => d.fng <= 25 ? CHART.fear : (d.fng <= 44 ? CHART.fearLight : 'rgba(0,0,0,0)')),
                        pointBorderWidth: historicalData.map(d => d.fng <= 25 ? 0 : 1.2),
                        pointRadius: historicalData.map(d => d.fng <= 25 ? 3.5 : (d.fng <= 44 ? 2.2 : 0)),
                        pointHoverRadius: 5,
                        fill: true,
                        tension: 0.25
                    }]
                };
            }, [historicalData, selectedCoin, themeV]);

            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: CHART.tooltipBg,
                        callbacks: {
                            label: (context) => {
                                const index = context.dataIndex;
                                const data = historicalData[index];
                                return [`價格: $${formatPrice(data.price)}`, `恐懼貪婪指數: ${data.fng} (${data.classification})`, data.fng <= 40 ? '🔴 建議 DCA 買入點' : '⚪ 觀望'];
                            }
                        }
                    },
                    title: { display: true, text: `${selectedCoin.toUpperCase()} 歷史走勢 (紅點為恐懼買入訊號)`, color: CHART.tick, align: 'start', font: { size: 12, weight: '500' } }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { color: CHART.tick, maxTicksLimit: window.innerWidth < 640 ? 4 : 8, maxRotation: 0, autoSkipPadding: 12 } },
                    y: { grid: { color: CHART.grid }, ticks: { color: CHART.tick, callback: (value) => `$${formatPrice(value)}` } }
                }
            };

            const getSuggestion = () => {
                if (!currentFNG) return { text: "載入中...", color: "text-gray-400" };
                const val = parseInt(currentFNG.value);
                if (val <= 25) return { title: "極度恐懼 (Extreme Fear)", action: "強力買入", desc: "市場極度恐慌，這是歷史上最佳的累積籌碼時機。", bg: "bg-red-900/30", border: "border-red-500", text: "text-red-400" };
                if (val <= 44) return { title: "恐懼 (Fear)", action: "分批買入", desc: "市場情緒低迷，適合執行標準 DCA 策略。", bg: "bg-orange-900/30", border: "border-orange-500", text: "text-orange-400" };
                if (val <= 55) return { title: "中立 (Neutral)", action: "持有觀望", desc: "市場方向不明，建議暫停大額買入，保持觀望。", bg: "bg-gray-800", border: "border-gray-600", text: "text-gray-400" };
                if (val <= 74) return { title: "貪婪 (Greed)", action: "停止買入", desc: "市場情緒過熱，風險增加，不建議此時進行 DCA。", bg: "bg-green-900/30", border: "border-green-500", text: "text-green-400" };
                return { title: `極度貪婪 (Extreme Greed)`, action: "止盈/減倉", desc: "市場極度過熱，強烈建議止盈或停止所有買入操作。", bg: "bg-blue-900/30", border: "border-blue-500", text: "text-blue-400" };
            };
            const suggestion = getSuggestion();

            return (
                <div>
                    <div className="flex items-center justify-between gap-4 flex-wrap py-2.5" style={{ borderBottom: '1px solid var(--rule)' }}>
                        <div className="flex-1 min-w-[260px] max-w-[520px]">
                            <SymbolSearch
                                symbol={selectedCoin}
                                onSearch={setSelectedCoin}
                                placeholder="輸入幣種 (e.g. solana)"
                                transformInput={(s) => s.toLowerCase()}
                                isLocked={!userInfo.isPremium}
                                userEmail={userInfo.email}
                            />
                        </div>
                        {/* 子分頁:行情 / 鏈上估值 / 合約數據(單層,不再巢狀) */}
                        <div className="fs-toggle overflow-x-auto no-scrollbar">
                            {[
                                { key: 'market', label: '行情' },
                                { key: 'valuation', label: '鏈上估值' },
                                { key: 'deriv', label: '合約數據' },
                            ].map(v => (
                                <button key={v.key} onClick={() => setCryptoView(v.key)} className={cryptoView === v.key ? 'on' : ''}>{v.label}</button>
                            ))}
                        </div>
                    </div>

                    <EditableCardGrid
                        title="加密貨幣"
                        mobileRow
                        cards={cryptoCards.cards}
                        editMode={cryptoEditMode}
                        onToggleEdit={() => setCryptoEditMode(!cryptoEditMode)}
                        onMove={cryptoCards.moveCard}
                        onRemove={cryptoCards.removeCard}
                        onReset={cryptoCards.resetCards}
                        onAdd={() => {
                            const slug = (prompt('輸入幣種 slug（CoinGecko id，例如 bitcoin、solana、dogecoin）：') || '').trim().toLowerCase();
                            if (!slug) return;
                            const name = (prompt('顯示名稱（如 BTC、SOL，可留空）：') || '').trim() || slug.toUpperCase();
                            cryptoCards.addCard({ id: slug, symbol: slug, name, logo: '' });
                        }}
                    >
                        {cryptoCards.cards.map(c => {
                            const p = prices && prices[c.symbol];
                            const active = selectedCoin === c.symbol;
                            return (
                                <div
                                    key={c.id}
                                    onClick={() => !cryptoEditMode && setSelectedCoin(c.symbol)}
                                    className={`${cryptoEditMode ? '' : 'cursor-pointer'} fs-ticker ${active ? 'on' : ''}`}
                                >
                                    <span className="fs-ticker-name">
                                        {c.logo && <img src={c.logo} className="w-4 h-4 shrink-0" alt="" onError={(e) => { e.target.style.display = 'none'; }} />}
                                        {c.name}
                                    </span>
                                    <span className="fs-ticker-price num">{p ? `$${formatPrice(p.usd)}` : '...'}</span>
                                    {p && (
                                        <span className="fs-ticker-change num" style={{ color: p.usd_24h_change >= 0 ? 'var(--up)' : 'var(--down)' }}>
                                            {p.usd_24h_change >= 0 ? '▲' : '▼'} {p.usd_24h_change.toFixed(2)}%
                                        </span>
                                    )}
                                </div>
                            );
                        })}
                    </EditableCardGrid>

                    {cryptoView === 'valuation' ? (
                        <div className="mt-8"><OnChainDashboard defaultAsset={selectedCoin} /></div>
                    ) : cryptoView === 'deriv' ? (
                        <div className="mt-8"><DerivativesPanel defaultAsset={selectedCoin} /></div>
                    ) : loading && !historicalData.length ? (
                        <div className="h-64 flex flex-col items-center justify-center gap-4" style={{ color: 'var(--ink-3)' }}><RefreshCw className="animate-spin" size={28} /><p>正在同步 CoinMarketCap 數據...</p></div>
                    ) : error ? (
                        <div className="fs-section mt-8" style={{ borderTopColor: 'var(--down)' }}>
                            <div className="flex items-start gap-3">
                                <AlertTriangle size={22} style={{ color: 'var(--down)' }} className="mt-1 shrink-0" />
                                <div><h3 className="fs-title-sm">載入失敗</h3><p className="text-sm my-2" style={{ color: 'var(--ink-2)' }}>{error}</p><button onClick={fetchData} className="fs-btn sm">重試</button></div>
                            </div>
                        </div>
                    ) : (
                        <>
                            <SentimentToggleCard
                                storageKey="crypto-sentiment-view"
                                fngTitle="加密貨幣恐懼與貪婪指數"
                                macdTitle={`週線 MACD 背離（${selectedCoin.toUpperCase()}）`}
                                market="crypto"
                                symbol={selectedCoin}
                                label={selectedCoin.toUpperCase()}
                            >
                                <div className="grid md:grid-cols-[7fr_5fr] gap-5 md:gap-10 items-start">
                                    <FearGreedGauge fngValue={currentFNG?.value} classification={suggestion.title.split(' ')[0]} />
                                    <DcaStrategy suggestion={suggestion}>
                                        {historicalData.length > 0 && <SentimentScarcityBar data={historicalData.map(d => d.fng)} title="過去 365 天機會分佈" />}
                                    </DcaStrategy>
                                </div>
                                {currentFNG && prices && historicalData.length > 0 && (
                                    <div className="mt-7">
                                        <AIAdviceBlock
                                            assetName={`加密貨幣 (${selectedCoin.toUpperCase()})`}
                                            priceStats={{ current: prices[selectedCoin]?.usd, high: Math.max(...historicalData.map(d => d.price)), low: Math.min(...historicalData.map(d => d.price)) }}
                                            marketData={`資產: ${selectedCoin.toUpperCase()}\n價格: $${formatPrice(prices[selectedCoin]?.usd)}\n恐懼貪婪指數: ${currentFNG.value} (${currentFNG.value_classification})`}
                                            isLocked={!userInfo.isPremium}
                                        />
                                    </div>
                                )}
                            </SentimentToggleCard>

                            <section className="fs-section mt-10">
                                <div className="fs-head">
                                    <h2 className="fs-title flex items-center gap-2">
                                        歷史走勢 ({selectedCoin.toUpperCase()})
                                        {onUpdateWatchlist && (
                                            <button
                                                onClick={toggleWatchlist}
                                                className="p-1 transition-colors"
                                                aria-pressed={isWatchlisted}
                                                aria-label="觀察清單"
                                                style={{ color: isWatchlisted ? 'var(--accent)' : 'var(--ink-3)' }}
                                            >
                                                <svg width="20" height="20" viewBox="0 0 24 24" fill={isWatchlisted ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true">
                                                    <path d="M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z" />
                                                </svg>
                                            </button>
                                        )}
                                    </h2>
                                    <div className="flex items-center gap-4 flex-wrap">
                                        <button onClick={() => setShowAdvanced(true)} className="text-[14px] underline underline-offset-4" style={{ color: 'var(--ink)' }} title="進階技術分析">
                                            進階分析
                                        </button>
                                        <TimeRangeSelector range={timeRange} onRangeChange={setTimeRange} />
                                    </div>
                                </div>
                                <div className="h-[300px] sm:h-[350px] w-full relative">
                                    {chartData && <ChartComponent data={chartData} options={chartOptions} />}
                                </div>
                            </section>

                            <section className="fs-section mt-10">
                                <div className="fs-head"><h2 className="fs-title">今日幣圈頭條</h2></div>
                                {!newsLoading && !newsError && newsData.length > 0 && <><NewsAIAnalysis newsItems={newsData} isLocked={!userInfo.isPremium} /><NewsSummary news={newsData} /></>}
                                <NewsSection news={newsData} loading={newsLoading} error={newsError} />
                            </section>

                            <section className="fs-section mt-10">
                                <div className="fs-head"><h3 className="fs-title-sm">關於數據來源</h3></div>
                                <div className="text-sm space-y-1.5" style={{ color: 'var(--ink-2)' }}><p><strong style={{ color: 'var(--ink)' }}>價格數據:</strong> CoinMarketCap (即時), Binance (歷史)</p><p><strong style={{ color: 'var(--ink)' }}>情緒指標:</strong> CoinMarketCap Crypto Fear &amp; Greed Index (無法取得時退回 Alternative.me)</p></div>
                            </section>
                        </>
                    )}
                    {showAdvanced && (
                        <TechnicalChartModal
                            symbol={selectedCoin.toLowerCase()}
                            type="CRYPTO"
                            onClose={() => setShowAdvanced(false)}
                        />
                    )}
                </div>
            );
        };

        // --- Shared Chart Components & Logic ---

        // 1. Universal FNG Chart (Price + Colored Dots)
        const UniversalFNGChart = ({ data, symbol, title }) => {
            if (!data || data.length === 0) return null;

            const chartData = {
                labels: data.map(d => d.date),
                datasets: [
                    {
                        label: `${symbol} 價格`,
                        data: data.map(d => d.price),
                        borderColor: CHART.line,
                        backgroundColor: makePriceGradient,
                        yAxisID: 'y',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: data.map(d => {
                            // Logic adapted for general FNG (0-100) or RSI (0-100)
                            // For FNG: Low is Fear (Buy) -> Red/Orange
                            // For RSI: Low is Oversold (Buy) -> Red/Orange
                            // So logic is consistent: Low Value = Buy Signal (Red/Orange)
                            if (d.fng <= 25) return CHART.fear; // Red
                            if (d.fng <= 45) return CHART.fearLight; // Orange
                            return 'rgba(0,0,0,0)';
                        }),
                        pointBorderColor: data.map(d => {
                            if (d.fng <= 25) return CHART.fear;
                            if (d.fng <= 45) return CHART.fearLight;
                            return 'rgba(0,0,0,0)';
                        }),
                        pointRadius: data.map(d => {
                            if (d.fng <= 45) return 3;
                            return 0;
                        }),
                        pointHoverRadius: 4,
                    }
                ]
            };

            const options = {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        callbacks: {
                            label: (context) => {
                                const idx = context.dataIndex;
                                const d = data[idx];
                                return [
                                    `價格: $${d.price.toLocaleString()}`,
                                    `指標: ${d.fng} (${d.classification || 'N/A'})`,
                                    d.fng <= 40 ? '🔴 建議 DCA 買入點' : '⚪ 觀望'
                                ];
                            }
                        }
                    },
                    title: { display: true, text: title || `${symbol} 趨勢圖 (紅/橘點為買入訊號)`, color: '#94a3b8' }
                },
                scales: {
                    y: { type: 'linear', display: true, position: 'left', grid: { color: CHART.grid }, ticks: { color: CHART.tick } },
                    x: { ticks: { color: CHART.tick, maxTicksLimit: 8 }, grid: { display: false } }
                }
            };
            return <ChartComponent data={chartData} options={options} />;
        };

        // 2. Chart Modal
        const ChartModal = ({ symbol, type, onClose }) => {
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [chartData, setChartData] = useState([]);
            const [stats, setStats] = useState(null); // { currentFng, classification, etc }

            useEffect(() => {
                const loadData = async () => {
                    setLoading(true);
                    setError(null);
                    try {
                        let mergedData = [];
                        let currentStats = {};

                        if (type === 'US') {
                            // Fetch CNN FNG + Yahoo
                            const proxyUrl = 'https://cors.hellokai07.com/?' + encodeURIComponent('https://production.dataviz.cnn.io/index/fearandgreed/graphdata');
                            const fngRes = await fetch(proxyUrl);
                            const fngJson = await fngRes.json();
                            const fngMap = new Map();
                            if (fngJson.fear_and_greed_historical?.data) {
                                fngJson.fear_and_greed_historical.data.forEach(item => {
                                    fngMap.set(new Date(item.x).toISOString().split('T')[0], Math.round(item.y));
                                });
                            }

                            // Yahoo
                            const yfUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=1y`;
                            const yfRes = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yfUrl));
                            const yfJson = await yfRes.json();
                            const result = yfJson.chart.result[0];
                            const quotes = result.indicators.quote[0].close;
                            const times = result.timestamp;

                            mergedData = times.map((t, i) => {
                                const date = new Date(t * 1000).toISOString().split('T')[0];
                                const price = quotes[i];
                                if (!price) return null;
                                let fng = fngMap.get(date) || 50;
                                return { date, price, fng, classification: fng <= 25 ? 'Extreme Fear' : fng <= 45 ? 'Fear' : 'Neutral' };
                            }).filter(d => d);

                            currentStats = {
                                value: Math.round(fngJson.fear_and_greed.score),
                                label: fngJson.fear_and_greed.rating
                            };
                        } else if (type === 'CRYPTO') {
                            // Binance klines + Alt.me FNG
                            const id = symbol.toLowerCase();
                            const binSym = getBinanceSymbol(id);
                            if (!binSym) throw new Error(`不支援的幣種：${symbol}（不在 Binance 上市）`);

                            // FNG
                            const fngRes = await fetch('https://api.alternative.me/fng/?limit=365');
                            const fngJson = await fngRes.json();
                            const fngMap = new Map();
                            fngJson.data.forEach(item => fngMap.set(new Date(item.timestamp * 1000).toISOString().split('T')[0], parseInt(item.value)));

                            // Binance (real daily OHLC)
                            const binRes = await fetch(`https://api.binance.com/api/v3/klines?symbol=${binSym}&interval=1d&limit=370`);
                            if (!binRes.ok) throw new Error(`Binance returned ${binRes.status}`);
                            const klines = await binRes.json();

                            mergedData = klines.map(k => {
                                const date = new Date(k[0]).toISOString().split('T')[0];
                                const fng = fngMap.get(date) || 50;
                                return { date, price: +k[4], fng, classification: fng <= 25 ? 'Extreme Fear' : fng <= 45 ? 'Fear' : 'Neutral' };
                            });
                            const last = fngJson.data[0];
                            currentStats = { value: last.value, label: last.value_classification };

                        } else if (type === 'TW') {
                            // Yahoo + RSI
                            const yfSymbol = symbol.includes('.') ? symbol : `${symbol}.TW`;
                            const yfUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${yfSymbol}?interval=1d&range=1y`;
                            const yfRes = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yfUrl));
                            const yfJson = await yfRes.json();
                            const result = yfJson.chart.result[0];
                            const quotes = result.indicators.quote[0].close;
                            const times = result.timestamp;

                            const cleanData = times.map((t, i) => ({ price: quotes[i], date: new Date(t * 1000).toISOString().split('T')[0] })).filter(d => d.price);

                            // Calc RSI (Global Helper Logic)
                            const rsiData = [];
                            let period = 14;
                            let gains = 0, losses = 0;
                            // Note: simplified RSI calc for brevity inside modal
                            for (let i = 1; i <= period; i++) {
                                const chg = cleanData[i].price - cleanData[i - 1].price;
                                if (chg > 0) gains += chg; else losses -= chg;
                            }
                            let avgGain = gains / period, avgLoss = losses / period;

                            for (let i = period + 1; i < cleanData.length; i++) {
                                const chg = cleanData[i].price - cleanData[i - 1].price;
                                avgGain = ((avgGain * 13) + (chg > 0 ? chg : 0)) / 14;
                                avgLoss = ((avgLoss * 13) + (chg < 0 ? -chg : 0)) / 14;
                                let rs = avgGain / avgLoss;
                                let rsi = 100 - (100 / (1 + rs));
                                rsiData.push({ ...cleanData[i], fng: Math.round(rsi), classification: rsi <= 30 ? 'Oversold' : rsi >= 70 ? 'Overbought' : 'Neutral' });
                            }
                            mergedData = rsiData;
                            if (mergedData.length > 0) {
                                const last = mergedData[mergedData.length - 1];
                                currentStats = { value: last.fng, label: last.classification };
                            }
                        }

                        setChartData(mergedData);
                        setStats(currentStats);
                    } catch (e) {
                        console.error(e);
                        setError("無法載入圖表數據: " + e.message);
                    } finally {
                        setLoading(false);
                    }
                };
                loadData();
            }, [symbol, type]);

            return (
                <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-[100] p-4" onClick={onClose}>
                    <div className="bg-slate-900 w-full max-w-3xl rounded-2xl border border-slate-700 shadow-2xl relative overflow-hidden flex flex-col max-h-[90vh]" onClick={e => e.stopPropagation()}>
                        <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50 backdrop-blur">
                            <div>
                                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                                    <TrendingUp size={20} className="text-blue-500" />
                                    {symbol} 趨勢分析
                                </h3>
                                <p className="text-xs text-slate-400">{type === 'TW' ? '價格 vs RSI (趨勢強弱)' : '價格 vs 恐懼貪婪指數'}</p>
                            </div>
                            <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white">
                                ✕
                            </button>
                        </div>

                        <div className="p-6 overflow-y-auto">
                            {loading ? (
                                <div className="h-64 flex items-center justify-center text-slate-500 gap-2">
                                    <RefreshCw className="animate-spin" /> 分析數據中...
                                </div>
                            ) : error ? (
                                <div className="h-64 flex items-center justify-center text-red-400 gap-2">
                                    <AlertTriangle /> {error}
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    <div className="h-[350px] w-full bg-slate-950/50 rounded-xl p-2">
                                        <UniversalFNGChart
                                            data={chartData}
                                            symbol={symbol}
                                            title={type === 'TW' ? `${symbol} 歷史走勢 (紅點=RSI超賣)` : undefined}
                                        />
                                    </div>

                                    {stats && (
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                                                <div className="text-slate-400 text-xs mb-1">當前指標 ({type === 'TW' ? 'RSI' : 'FNG'})</div>
                                                <div className="text-2xl font-bold text-white flex items-baseline gap-2">
                                                    {stats.value}
                                                    <span className="text-sm font-normal text-slate-400">/ 100</span>
                                                </div>
                                                <div className={`text-sm font-bold ${stats.value <= 40 ? 'text-red-400' : stats.value >= 75 ? 'text-green-400' : 'text-blue-400'}`}>
                                                    {stats.label}
                                                </div>
                                            </div>
                                            <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
                                                <div className="text-slate-400 text-xs mb-1">AI 策略建議</div>
                                                <div className="text-lg font-bold text-white">
                                                    {stats.value <= 30 ? '🟢 強力買入區' : stats.value <= 50 ? '🟡 分批買入區 (DCA)' : '⚪ 觀望 / 止盈'}
                                                </div>
                                                <p className="text-xs text-slate-500 mt-1">
                                                    {stats.value <= 50 ? '市場情緒低迷，適合執行 DCA 累積籌碼。' : '市場情緒樂觀，建議謹慎操作。'}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            );
        };

        // --- Watchlist Dashboard Feature ---
        // ─────────────────────────────────────────────
        // Shared RSI helper (used by Watchlist + future Insights)
        // ─────────────────────────────────────────────
        const computeRSI = (closes, period = 14) => {
            if (!closes || closes.length < period + 1) return null;
            let gains = 0, losses = 0;
            for (let i = 1; i <= period; i++) {
                const d = closes[i] - closes[i - 1];
                if (d > 0) gains += d; else losses -= d;
            }
            let avgGain = gains / period;
            let avgLoss = losses / period;
            // Smoothed RSI for the rest
            for (let i = period + 1; i < closes.length; i++) {
                const d = closes[i] - closes[i - 1];
                const gain = d > 0 ? d : 0;
                const loss = d < 0 ? -d : 0;
                avgGain = (avgGain * (period - 1) + gain) / period;
                avgLoss = (avgLoss * (period - 1) + loss) / period;
            }
            if (avgLoss === 0) return 100;
            const rs = avgGain / avgLoss;
            return 100 - 100 / (1 + rs);
        };

        // ─────────────────────────────────────────────
        // EMA & MACD helpers (used by TechnicalChartModal)
        // ─────────────────────────────────────────────
        const computeEMA = (values, period) => {
            if (!values || values.length < period) return values ? values.map(() => null) : [];
            const k = 2 / (period + 1);
            const out = new Array(values.length).fill(null);
            // Seed with SMA at index period-1
            let sum = 0;
            for (let i = 0; i < period; i++) sum += values[i];
            out[period - 1] = sum / period;
            for (let i = period; i < values.length; i++) {
                out[i] = values[i] * k + out[i - 1] * (1 - k);
            }
            return out;
        };

        const computeMACD = (closes, fast = 12, slow = 26, signal = 9) => {
            const emaFast = computeEMA(closes, fast);
            const emaSlow = computeEMA(closes, slow);
            const dif = closes.map((_, i) =>
                (emaFast[i] !== null && emaSlow[i] !== null) ? emaFast[i] - emaSlow[i] : null
            );
            // Compute DEA only on the valid range
            const firstValid = dif.findIndex(v => v !== null);
            const validDif = firstValid >= 0 ? dif.slice(firstValid) : [];
            const deaValid = computeEMA(validDif.filter(v => v !== null), signal);
            const dea = new Array(closes.length).fill(null);
            for (let i = 0; i < deaValid.length; i++) {
                dea[firstValid + i] = deaValid[i];
            }
            const histogram = dif.map((v, i) =>
                (v !== null && dea[i] !== null) ? v - dea[i] : null
            );
            return { dif, dea, histogram };
        };

        // Composite signal: RSI + 1Y position
        // RSI thresholds stay standard (30/70). 1Y position acts as amplifier to catch
        // cases like "RSI 60 but price at 95% of 1Y high" → still a sell warning.
        const getCompositeSignal = (rsi, position) => {
            if (rsi === null || rsi === undefined) return { label: '—', emoji: '⚪', color: 'var(--text-3)', advice: '計算中' };
            const pos = (position === null || position === undefined) ? 0.5 : position;

            // 強力加碼: RSI 極度超賣 OR (RSI 超賣 且 接近 1Y 低)
            if (rsi <= 25 || (rsi <= 30 && pos <= 0.10)) {
                return { label: '強力加碼', emoji: '🔥', color: 'var(--up)', advice: '極度超賣' };
            }
            // 加碼: RSI 超賣 OR (RSI 偏低 且 接近 1Y 低)
            if (rsi <= 30 || (rsi <= 40 && pos <= 0.20)) {
                return { label: '加碼', emoji: '🟢', color: 'var(--z3)', advice: '超賣' };
            }
            // 強力賣出: RSI 極度超買 OR (RSI 偏高 且 1Y 高點)
            if (rsi >= 75 || (rsi >= 65 && pos >= 0.95)) {
                return { label: '強力賣出', emoji: '🚨', color: 'var(--down)', advice: '極度超買' };
            }
            // 賣出: RSI 超買 OR (RSI 偏高 且 接近 1Y 高)
            if (rsi >= 70 || (rsi >= 60 && pos >= 0.90)) {
                return { label: '賣出', emoji: '🔴', color: 'var(--down)', advice: '超買' };
            }
            // 警示: RSI 接近超買 OR 純粹接近 1Y 高
            if (rsi >= 65 || pos >= 0.85) {
                return { label: '警示', emoji: '🟡', color: 'var(--amber)', advice: '接近超買' };
            }
            // 觀望: RSI 中性偏強
            if (rsi >= 50) {
                return { label: '觀望', emoji: '⚪', color: 'var(--text-2)', advice: '中性偏強' };
            }
            // 持有: RSI 中性偏弱
            return { label: '持有', emoji: '⚪', color: 'var(--text-2)', advice: '中性偏弱' };
        };
        // Backwards-compat alias
        const getRsiSignal = getCompositeSignal;

        // ─────────────────────────────────────────────
        // useLocalState — persist a piece of state to localStorage so reload remembers it
        // ─────────────────────────────────────────────
        const useLocalState = (key, defaultValue) => {
            const [v, setV] = useState(() => {
                try {
                    const raw = localStorage.getItem(key);
                    return raw !== null ? JSON.parse(raw) : defaultValue;
                } catch { return defaultValue; }
            });
            useEffect(() => {
                try { localStorage.setItem(key, JSON.stringify(v)); } catch {}
            }, [key, v]);
            return [v, setV];
        };

        // CoinGecko slug → Binance USDT pair. Crypto data goes through Binance entirely
        // (no key, CORS-friendly, real daily klines, single source = no merge artefacts).
        const COIN_TO_BINANCE = {
            bitcoin: 'BTCUSDT', ethereum: 'ETHUSDT', solana: 'SOLUSDT', binancecoin: 'BNBUSDT',
            ripple: 'XRPUSDT', cardano: 'ADAUSDT', dogecoin: 'DOGEUSDT', polkadot: 'DOTUSDT',
            'avalanche-2': 'AVAXUSDT', tron: 'TRXUSDT', chainlink: 'LINKUSDT',
            'matic-network': 'MATICUSDT', litecoin: 'LTCUSDT', 'shiba-inu': 'SHIBUSDT',
            cosmos: 'ATOMUSDT', uniswap: 'UNIUSDT', near: 'NEARUSDT', aptos: 'APTUSDT',
            sui: 'SUIUSDT', arbitrum: 'ARBUSDT', optimism: 'OPUSDT', filecoin: 'FILUSDT',
            'internet-computer': 'ICPUSDT', stellar: 'XLMUSDT', 'bitcoin-cash': 'BCHUSDT',
            algorand: 'ALGOUSDT', vechain: 'VETUSDT', 'the-graph': 'GRTUSDT',
            aave: 'AAVEUSDT', maker: 'MKRUSDT', tezos: 'XTZUSDT', monero: 'XMRUSDT',
            pepe: 'PEPEUSDT', floki: 'FLOKIUSDT', 'bonk': 'BONKUSDT',
            ondo: 'ONDOUSDT', injective: 'INJUSDT', sei: 'SEIUSDT', kaspa: 'KASUSDT',
            'render-token': 'RNDRUSDT', 'fetch-ai': 'FETUSDT', worldcoin: 'WLDUSDT',
            'ethereum-classic': 'ETCUSDT', 'hedera-hashgraph': 'HBARUSDT',
            'first-digital-usd': 'FDUSDUSDT', 'true-usd': 'TUSDUSDT',
        };

        // Resolve a CoinGecko slug or short ticker to a Binance symbol. Returns null if not on Binance.
        const getBinanceSymbol = (slug) => {
            const lower = (slug || '').toLowerCase();
            if (COIN_TO_BINANCE[lower]) return COIN_TO_BINANCE[lower];
            // Heuristic: short alphanumeric (likely a ticker like 'pepe' or 'ada') → try LOWERUSDT
            if (/^[a-z0-9]{2,7}$/.test(lower)) return lower.toUpperCase() + 'USDT';
            return null;
        };

        // ─────────────────────────────────────────────
        // 週線 MACD 背離(柱狀體波段法)
        // 主頁情緒卡片的第二個檢視,只做輔助判讀;DCA 進出場訊號仍以恐懼貪婪 / RSI 為主。
        //
        // 做法:把柱狀體依正負切成一段段「波」,拿相鄰兩段同號的波比較 —
        //   底背離:後一段紅柱波的價格低點更低,但該波 DIF 的最低點沒有更低
        //   頂背離:後一段綠柱波的價格高點更高,但該波 DIF 的最高點沒有更高
        // 兩段同號波之間一定隔著一段反向波,確保動能「回來過」,不會把同一段走勢重複配對。
        // 只用已收盤的週 K 判斷,避免當週 K 棒變動讓訊號出現又消失。
        // ─────────────────────────────────────────────
        const MACD_DIV_CFG = {
            minWaveBars: 3,    // 柱狀體同號少於 3 週視為雜訊,併入前一波
            maxGap: 52,        // 兩波的價格極值相隔超過 52 週就不比較
            confirmBars: 2,    // 進行中的波:柱狀體需從極值縮短至少 2 週才算成立
            minDifChange: 0.1, // 兩波 DIF 極值至少相差 10%,差太少等於沒有背離
            activeWithin: 8,   // 右波極值落在最近 8 週內 → 視為仍在進行
            minBars: 60,       // 至少 60 根週 K 才算得出穩定的 MACD
        };

        // 把柱狀體依正負切成波段;太短的波併入前一波,再把合併後相鄰的同號波接起來
        const splitHistogramWaves = (hist, minLen) => {
            const raw = [];
            for (let i = 0; i < hist.length; i++) {
                if (hist[i] == null) continue;
                const sign = hist[i] >= 0 ? 1 : -1;
                const last = raw[raw.length - 1];
                if (last && last.sign === sign && last.end === i - 1) last.end = i;
                else raw.push({ sign, start: i, end: i });
            }
            const out = [];
            for (const w of raw) {
                const prev = out[out.length - 1];
                if (prev && (prev.sign === w.sign || w.end - w.start + 1 < minLen)) { prev.end = w.end; continue; }
                out.push({ ...w });
            }
            return out;
        };

        // bars: [{ t, high, low, close, closed? }] — 由舊到新的週 K;closed === false 代表尚未收盤
        const detectMACDDivergence = (bars) => {
            const { minWaveBars, maxGap, confirmBars, minDifChange, activeWithin, minBars } = MACD_DIV_CFG;
            if (!bars || bars.length < minBars) return null;

            const closes = bars.map(b => b.close);
            const { dif, dea, histogram } = computeMACD(closes);
            const lows  = bars.map(b => (b.low  != null ? b.low  : b.close));
            const highs = bars.map(b => (b.high != null ? b.high : b.close));
            const n = bars.length;
            // 背離只看已收盤的週 K(圖上仍會畫出當週)
            let nc = n;
            while (nc > 0 && bars[nc - 1].closed === false) nc--;
            const waves = splitHistogramWaves(histogram.slice(0, nc), minWaveBars);

            const scanAll = (kind) => {
                const isBull = kind === 'bullish';
                const series = isBull ? lows : highs;
                const beyond = (a, b) => (isBull ? a < b : a > b); // a 比 b 更極端
                const ws = waves.filter(w => w.sign === (isBull ? -1 : 1)).map(w => {
                    let pi = w.start, di = w.start, hi = w.start;
                    for (let i = w.start; i <= w.end; i++) {
                        if (beyond(series[i], series[pi])) pi = i;
                        if (dif[i] != null && (dif[di] == null || beyond(dif[i], dif[di]))) di = i;
                        if (beyond(histogram[i], histogram[hi])) hi = i;
                    }
                    return { ...w, pi, di, hi };
                });

                const found = [];
                for (let k = 1; k < ws.length; k++) {
                    const a = ws[k - 1], b = ws[k];
                    if (dif[a.di] == null || dif[b.di] == null) continue;
                    if (b.pi - a.pi > maxGap) continue;
                    const ongoing = b.end === nc - 1;
                    if (ongoing && nc - 1 - b.hi < confirmBars) continue;
                    const priceDiverges = beyond(series[b.pi], series[a.pi]);
                    const difChange = (isBull ? dif[b.di] - dif[a.di] : dif[a.di] - dif[b.di]) / Math.max(1e-12, Math.abs(dif[a.di]));
                    if (!priceDiverges || difChange < minDifChange) continue;
                    found.push({
                        kind,
                        i1: a.pi, i2: b.pi,       // 價格極值所在的週
                        j1: a.di, j2: b.di,       // DIF 極值所在的週
                        t1: bars[a.pi].t, t2: bars[b.pi].t,
                        price1: series[a.pi], price2: series[b.pi],
                        dif1: dif[a.di], dif2: dif[b.di],
                        weeks: b.pi - a.pi,
                        barsSince: n - 1 - b.pi,
                        // 底背離出現在零軸下方 / 頂背離在零軸上方,參考價值較高
                        strong: isBull ? dif[a.di] < 0 && dif[b.di] < 0 : dif[a.di] > 0 && dif[b.di] > 0,
                    });
                }
                return found;
            };

            const bullAll = scanAll('bullish');
            const bearAll = scanAll('bearish');
            const all = [...bullAll, ...bearAll].sort((a, b) => a.i2 - b.i2);
            const bullish = bullAll.length ? bullAll[bullAll.length - 1] : null;
            const bearish = bearAll.length ? bearAll[bearAll.length - 1] : null;

            let active = null;
            const candidates = [bullish, bearish].filter(d => d && d.barsSince <= activeWithin);
            if (candidates.length) {
                active = candidates.reduce((a, b) => (b.i2 > a.i2 ? b : a));
            }

            // 最近一次 DIF × DEA 交叉
            let cross = null;
            for (let i = n - 1; i >= 1; i--) {
                if (dif[i] == null || dea[i] == null || dif[i - 1] == null || dea[i - 1] == null) break;
                const prev = dif[i - 1] - dea[i - 1];
                const cur  = dif[i] - dea[i];
                if (prev <= 0 && cur > 0) { cross = { type: 'golden', index: i, t: bars[i].t, weeksAgo: n - 1 - i }; break; }
                if (prev >= 0 && cur < 0) { cross = { type: 'death',  index: i, t: bars[i].t, weeksAgo: n - 1 - i }; break; }
            }

            return {
                bars, dif, dea, histogram,
                all, bullish, bearish, active, cross,
                last: {
                    dif: dif[n - 1], dea: dea[n - 1], hist: histogram[n - 1],
                    close: closes[n - 1], t: bars[n - 1].t,
                },
            };
        };

        // 抓週 K:美股/台股走 Yahoo(經自家 CORS proxy),加密走 Binance
        const fetchWeeklyBars = async (market, symbol) => {
            if (market === 'crypto') {
                const binSym = getBinanceSymbol(symbol);
                if (!binSym) throw new Error('此幣種不在 Binance 上,無法取得週線');
                const res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${binSym}&interval=1w&limit=260`);
                if (!res.ok) throw new Error('Binance 週線取得失敗');
                const kl = await res.json();
                if (!Array.isArray(kl) || !kl.length) throw new Error('查無週線資料');
                const now = Date.now();
                // k[6] = 收盤時間;還沒到的就是本週進行中的 K 棒
                return kl.map(k => ({ t: k[0], high: parseFloat(k[2]), low: parseFloat(k[3]), close: parseFloat(k[4]), closed: k[6] < now }));
            }

            const ticker = market === 'tw'
                ? (symbol.includes('.') || symbol.startsWith('^') ? symbol : `${symbol}.TW`)
                : symbol;
            const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1wk&range=5y`;
            const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(yahooUrl));
            if (!res.ok) throw new Error('Yahoo 週線取得失敗');
            const json = await res.json();
            const result = json?.chart?.result?.[0];
            const quote = result?.indicators?.quote?.[0];
            if (!result || !quote || !result.timestamp) throw new Error('查無此代號的週線資料');
            const now = Date.now();
            // Yahoo 週 K 的時間戳是該週起點;起點 + 7 天還沒到 → 本週尚未收盤
            return result.timestamp
                .map((t, i) => ({ t: t * 1000, high: quote.high?.[i], low: quote.low?.[i], close: quote.close?.[i], closed: t * 1000 + 7 * 864e5 <= now }))
                .filter(b => b.close != null);
        };

        // Dev-only: synthetic weekly series with a designed divergence, for reviewing the diagram.
        // import.meta.env.DEV is false in `vite build`, so this never reaches production.
        const makeMacdFixture = (kind) => {
            const n = 80, bull = [];
            let p = 70000;
            for (let i = 0; i < n; i++) {
                const ph = i < 22 ? 0.012 : i < 40 ? -0.028 : i < 50 ? 0.012 : i < 66 ? -0.010 : 0.018;
                p *= 1 + ph + Math.sin(i * 1.7) * 0.012;
                bull.push(p);
            }
            const K = Math.max(...bull) + Math.min(...bull);
            const close = kind === 'bear' ? bull.map(v => K - v) : bull;
            const ema = (a, k) => { const al = 2 / (k + 1); let e = a[0]; return a.map(v => (e = v * al + e * (1 - al))); };
            const e12 = ema(close, 12), e26 = ema(close, 26);
            const dif = close.map((_, i) => e12[i] - e26[i]), dea = ema(dif, 9), histogram = dif.map((v, i) => v - dea[i]);
            const pick = (a, b) => { let k = a; for (let i = a; i <= b; i++) if (bull[i] < bull[k]) k = i; return k; };
            const i1 = pick(34, 46), i2 = pick(58, 70);
            const bars = close.map((c, i) => ({ t: Date.now() - (n - 1 - i) * 7 * 864e5, close: c, high: c, low: c }));
            const div = {
                kind: kind === 'bear' ? 'bearish' : 'bullish', i1, i2, j1: i1, j2: i2, t1: bars[i1].t, t2: bars[i2].t,
                price1: close[i1], price2: close[i2], dif1: dif[i1], dif2: dif[i2], weeks: i2 - i1, barsSince: n - 1 - i2,
                strong: kind === 'bear' ? dif[i2] > 0 : dif[i2] < 0,
            };
            return {
                bars, dif, dea, histogram,
                all: [div], bullish: kind === 'bear' ? null : div, bearish: kind === 'bear' ? div : null, active: div,
                cross: { type: 'golden', index: n - 7, t: bars[n - 7].t, weeksAgo: 6 },
                last: { dif: dif[n - 1], dea: dea[n - 1], hist: histogram[n - 1], close: close[n - 1], t: bars[n - 1].t },
            };
        };

        const useWeeklyMacd = (market, symbol) => {
            const [state, setState] = useState({ loading: true, error: null, data: null });
            useEffect(() => {
                if (!symbol) return;
                if (import.meta.env.DEV) {
                    const fx = new URLSearchParams(window.location.search).get('macdFixture');
                    if (fx === 'bull' || fx === 'bear') { setState({ loading: false, error: null, data: makeMacdFixture(fx) }); return; }
                }
                let cancelled = false;
                setState({ loading: true, error: null, data: null });
                (async () => {
                    try {
                        const bars = await fetchWeeklyBars(market, symbol);
                        const data = detectMACDDivergence(bars);
                        if (cancelled) return;
                        if (!data) throw new Error(`週線資料不足(需至少 ${MACD_DIV_CFG.minBars} 週)`);
                        setState({ loading: false, error: null, data });
                    } catch (e) {
                        if (!cancelled) setState({ loading: false, error: e.message || '載入失敗', data: null });
                    }
                })();
                return () => { cancelled = true; };
            }, [market, symbol]);
            return state;
        };

        const fmtWeek = (t) => {
            const d = new Date(t);
            return `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`;
        };
        const fmtNum = (v, digits = 2) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(digits));

        const MACD_VIEW_BARS = 80; // 圖上顯示最近 80 週

        const WeeklyMacdPanel = ({ market, symbol, label }) => {
            const { loading, error, data } = useWeeklyMacd(market, symbol);
            const [boxRef, width] = useWidth();
            const [hover, setHover] = useState(null);

            if (loading) {
                return <div ref={boxRef} className="py-12 flex justify-center"><RefreshCw className="animate-spin" size={28} style={{ color: 'var(--ink-3)' }} /></div>;
            }
            if (error) {
                return (
                    <div ref={boxRef} className="flex flex-col items-center gap-2 py-8 text-center">
                        <AlertTriangle size={26} style={{ color: 'var(--down)' }} />
                        <div className="text-sm" style={{ color: 'var(--down)' }}>{error}</div>
                    </div>
                );
            }
            if (!data) return <div ref={boxRef} />;

            const act = data.active;
            const status = act
                ? (act.kind === 'bullish'
                    ? { title: '週線 MACD 底背離', desc: '價格創新低、MACD 沒有跟著創新低 — 下跌動能轉弱,分批買入的參考點。', color: 'var(--up)' }
                    : { title: '週線 MACD 頂背離', desc: '價格創新高、MACD 沒有跟著創新高 — 上漲動能轉弱,加碼前留意風險。', color: 'var(--down)' })
                : { title: '目前無明顯背離', desc: '最近幾週價格與 MACD 同向,沒有偵測到有效背離。', color: 'var(--ink-3)' };

            // ── Diagram geometry (real pixels) ──
            const total = data.bars.length;
            const start = Math.max(0, total - MACD_VIEW_BARS);
            const n = total - start;
            const closes = data.bars.slice(start).map(b => b.close);
            const dif = data.dif.slice(start), dea = data.dea.slice(start), hist = data.histogram.slice(start);
            const W = Math.max(300, width || 0);
            const small = W < 560;
            const L = small ? 44 : 64, R = 12, T = 14;
            // 兩個面板:價格 + MACD。柱狀圖和 DIF/DEA 畫在同一格(共用零軸),但高度用自己的刻度換算 —
            // 柱狀圖數值通常比 DIF/DEA 小一個數量級,共用刻度會整排貼在零軸上看不出變化。
            const H1 = small ? 170 : 210, H2 = small ? 176 : 216;
            const B1 = H1 - 22, B2 = H2 - 22;
            const X = (i) => L + (i / Math.max(1, n - 1)) * (W - L - R);
            const pMin = Math.min(...closes), pMax = Math.max(...closes);
            const yP = (v) => B1 - ((v - pMin) / ((pMax - pMin) || 1)) * (B1 - T);
            const lim = Math.max(1e-9, ...[...dif, ...dea].filter(v => v != null).map(Math.abs));
            const yO = (v) => T + (1 - (v + lim) / (2 * lim)) * (B2 - T);
            const hLim = Math.max(1e-9, ...hist.filter(v => v != null).map(Math.abs));
            const hMid = yO(0);                       // 與 DIF/DEA 同一條零軸
            const hHalf = ((B2 - T) / 2) * 0.86;      // 高度用柱狀圖自己的刻度
            const yH = (v) => hMid - (v / hLim) * hHalf;
            const path = (arr, yf) => arr.map((v, i) => (v == null ? null : `${X(i)} ${yf(v)}`)).filter(Boolean).map((p, i) => (i ? 'L' : 'M') + p).join(' ');
            const fmtAxis = (v) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}k` : fmtNum(v, Math.abs(v) >= 10 ? 0 : 2));

            // 視窗內的每一組背離都標點(不只最新一組)
            const divs = (data.all || [])
                .filter(d => d.i1 >= start)
                .map(d => ({
                    ...d, a: d.i1 - start, b: d.i2 - start,
                    // DIF 的極值不一定和價格極值同一週,各自標在自己的轉折點
                    da: (d.j1 ?? d.i1) - start, db: (d.j2 ?? d.i2) - start,
                    isActive: act === d, color: d.kind === 'bullish' ? 'var(--up)' : 'var(--down)',
                }));
            const visBull = divs.filter(d => d.kind === 'bullish').length;
            const visBear = divs.length - visBull;
            // 同一根週 K 常同時是好幾組背離的端點 — 去重後再畫,最新那組優先決定顏色
            const makeDotList = (ka, kb) => {
                const m = new Map();
                divs.forEach(d => [d[ka], d[kb]].forEach(i => {
                    if (i < 0) return;
                    const prev = m.get(i);
                    if (!prev || (d.isActive && !prev.isActive)) m.set(i, { i, color: d.color, isActive: d.isActive });
                }));
                return [...m.values()];
            };
            const priceDots = makeDotList('a', 'b');
            const difDots = makeDotList('da', 'db');
            // 背離只用圓點標出轉折點:不畫連線、不加文字標註
            const pivotDots = (dotList, arr, yf, key) => dotList.map(p => (arr[p.i] == null ? null : (
                <circle key={`${key}${p.i}`} cx={X(p.i)} cy={yf(arr[p.i])} r={p.isActive ? 5 : 4}
                    style={{ fill: 'var(--paper)', stroke: p.color, strokeWidth: 2.5, opacity: p.isActive ? 1 : 0.55 }} />
            )));

            const onMove = (e) => {
                const r = e.currentTarget.getBoundingClientRect();
                const x = (e.clientX - r.left) * (W / r.width);
                const i = Math.round(((x - L) / (W - L - R)) * (n - 1));
                setHover(i >= 0 && i < n ? i : null);
            };
            const hv = hover != null ? hover : n - 1;
            const bw = Math.max(1.5, ((W - L - R) / n) * 0.62);
            const guideX = hover != null ? X(hover) : null;

            return (
                <div ref={boxRef} className="w-full flex flex-col text-left">
                    <div className="grid gap-x-4 gap-y-1.5 pb-5" style={{ gridTemplateColumns: 'auto 1fr', borderBottom: '1px solid var(--rule)' }}>
                        <svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true" style={{ gridRow: 'span 3' }}>
                            <circle cx="22" cy="22" r="20" fill="none" style={{ stroke: status.color, strokeWidth: 3 }} />
                            {act && act.kind === 'bullish' && <path d="M12 16 L22 30 L32 16" fill="none" style={{ stroke: status.color, strokeWidth: 3, strokeLinejoin: 'round' }} />}
                            {act && act.kind === 'bearish' && <path d="M12 28 L22 14 L32 28" fill="none" style={{ stroke: status.color, strokeWidth: 3, strokeLinejoin: 'round' }} />}
                            {!act && <line x1="13" y1="22" x2="31" y2="22" style={{ stroke: status.color, strokeWidth: 3 }} />}
                        </svg>
                        <h3 className="text-xl md:text-2xl font-black flex items-center gap-2.5 flex-wrap" style={{ color: 'var(--ink)' }}>
                            {status.title}
                            {act && act.strong && (
                                <span className="fs-chip" style={{ color: status.color }}>
                                    {act.kind === 'bullish' ? '零軸下方，參考性較高' : '零軸上方，參考性較高'}
                                </span>
                            )}
                        </h3>
                        <p className="text-[15px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>{status.desc}</p>
                        {act && (
                            <div className="text-[13px] leading-relaxed num" style={{ color: 'var(--ink-2)' }}>
                                <div>
                                    比較區間：{fmtWeek(act.t1)} → {fmtWeek(act.t2)}（相隔 {act.weeks} 週，{act.barsSince === 0 ? '本週' : `${act.barsSince} 週前`}成形）
                                </div>
                                <div>
                                    {act.kind === 'bullish' ? '低點' : '高點'} {fmtNum(act.price1, 2)} → {fmtNum(act.price2, 2)}　|
                                    DIF {fmtNum(act.dif1, 3)} → {fmtNum(act.dif2, 3)}
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="fs-kv num">
                        {[
                            { k: 'DIF', v: fmtNum(data.last.dif, 3) },
                            { k: 'DEA', v: fmtNum(data.last.dea, 3) },
                            { k: '柱狀圖', v: fmtNum(data.last.hist, 3) },
                            {
                                k: '最近交叉',
                                v: data.cross
                                    ? `${data.cross.type === 'golden' ? '金叉' : '死叉'} · ${data.cross.weeksAgo} 週前`
                                    : '—',
                            },
                        ].map(item => (
                            <div key={item.k}>
                                <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>{item.k}</div>
                                <div className="text-lg md:text-[22px] font-bold truncate" style={{ color: 'var(--ink)' }}>{item.v}</div>
                            </div>
                        ))}
                    </div>

                    {width > 0 && (
                        <>
                            <div className="flex justify-between gap-3 flex-wrap text-[12px] mt-5 mb-1.5 num" style={{ color: 'var(--ink-3)' }}>
                                <span>{label} 週線收盤（標記＝背離的兩個轉折點）</span>
                                <span style={{ color: 'var(--ink-2)' }}>
                                    {fmtWeek(data.bars[start + hv].t)}　{fmtNum(closes[hv], 2)}
                                </span>
                            </div>
                            <svg width="100%" viewBox={`0 0 ${W} ${H1}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)} style={{ display: 'block', overflow: 'visible' }}>
                                {[pMin, (pMin + pMax) / 2, pMax].map((v, i) => (
                                    <g key={i}>
                                        <line x1={L} x2={W - R} y1={yP(v)} y2={yP(v)} style={{ stroke: 'var(--faint)' }} />
                                        <text x={L - 8} y={yP(v) + 4} textAnchor="end" style={{ fill: 'var(--ink-3)', fontSize: 11 }}>{fmtAxis(v)}</text>
                                    </g>
                                ))}
                                <path d={path(closes, yP)} fill="none" style={{ stroke: 'var(--ink)', strokeWidth: 1.6 }} />
                                {guideX != null && <line x1={guideX} x2={guideX} y1={T} y2={B1} style={{ stroke: 'var(--ink-3)' }} />}
                                {pivotDots(priceDots, closes, yP, 'p')}
                            </svg>

                            <div className="flex justify-between gap-3 flex-wrap text-[12px] mt-4 mb-1.5 num" style={{ color: 'var(--ink-3)' }}>
                                <span>週線 MACD (12, 26, 9)</span>
                                <span className="flex items-center gap-3" style={{ color: 'var(--ink-2)' }}>
                                    <span className="inline-flex items-center gap-1"><i style={{ width: 14, height: 2, background: 'var(--ink)', display: 'inline-block' }} />DIF {fmtNum(dif[hv], 3)}</span>
                                    <span className="inline-flex items-center gap-1"><i style={{ width: 14, height: 2, background: 'var(--ink-3)', display: 'inline-block' }} />DEA {fmtNum(dea[hv], 3)}</span>
                                    <span className="inline-flex items-center gap-1"><i style={{ width: 6, height: 8, background: 'var(--up)', opacity: 0.5, display: 'inline-block' }} /><i style={{ width: 6, height: 8, background: 'var(--down)', opacity: 0.5, display: 'inline-block' }} />柱狀圖 {fmtNum(hist[hv], 3)}<span style={{ color: 'var(--ink-3)' }}>（獨立刻度）</span></span>
                                </span>
                            </div>
                            <svg width="100%" viewBox={`0 0 ${W} ${H2}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)} style={{ display: 'block', overflow: 'visible' }}>
                                {[-lim * 0.8, lim * 0.8].map((v, i) => (
                                    <g key={i}>
                                        <line x1={L} x2={W - R} y1={yO(v)} y2={yO(v)} style={{ stroke: 'var(--faint)' }} />
                                        <text x={L - 8} y={yO(v) + 4} textAnchor="end" style={{ fill: 'var(--ink-3)', fontSize: 11 }}>{fmtAxis(v)}</text>
                                    </g>
                                ))}
                                <text x={L - 8} y={yO(0) + 4} textAnchor="end" style={{ fill: 'var(--ink-3)', fontSize: 11 }}>0</text>
                                {hist.map((v, i) => v == null ? null : (
                                    <rect key={i} x={X(i) - bw / 2} y={Math.min(yH(v), hMid)} width={bw} height={Math.max(1, Math.abs(yH(v) - hMid))}
                                        style={{ fill: v >= 0 ? 'var(--up)' : 'var(--down)', opacity: 0.45 }} />
                                ))}
                                <line x1={L} x2={W - R} y1={yO(0)} y2={yO(0)} style={{ stroke: 'var(--ink)' }} />
                                <path d={path(dea, yO)} fill="none" style={{ stroke: 'var(--ink-3)', strokeWidth: 1.4 }} />
                                <path d={path(dif, yO)} fill="none" style={{ stroke: 'var(--ink)', strokeWidth: 1.8 }} />
                                {guideX != null && <line x1={guideX} x2={guideX} y1={T} y2={B2} style={{ stroke: 'var(--ink-3)' }} />}
                                {pivotDots(difDots, dif, yO, 'o')}
                                {[0, Math.floor((n - 1) / 2), n - 1].map((i, k) => (
                                    <text key={i} x={X(i)} y={H2 - 6} textAnchor={k === 0 ? 'start' : k === 2 ? 'end' : 'middle'} style={{ fill: 'var(--ink-3)', fontSize: 11 }}>{fmtWeek(data.bars[start + i].t)}</text>
                                ))}
                            </svg>

                        </>
                    )}

                    {divs.length > 0 && (
                        <div className="text-[12px] mt-4" style={{ color: 'var(--ink-3)' }}>
                            圖上這 {n} 週內共 {divs.length} 組背離（底背離 {visBull}、頂背離 {visBear}）。
                            圓點是每一組的轉折點，綠＝底背離、紅＝頂背離；最新一組畫得較深，其餘較淡。
                        </div>
                    )}

                    <p className="text-[12px] leading-relaxed mt-2 max-w-[80ch]" style={{ color: 'var(--ink-3)' }}>
                        背離＝價格與動能不同步：這一段紅柱的價格低點比上一段更低，但 DIF 的低點沒有更低，為「底背離」；綠柱段反之為「頂背離」。
                        這裡用週線、12/26/9 參數，只看已收盤的週 K；DIF 至少要相差 10%，進行中的一段需柱狀體縮短 2 週才確認。
                        <span style={{ color: 'var(--ink-2)' }}>買賣訊號仍以原本的恐懼貪婪 / RSI 判讀為主，此指標僅作輔助。</span>
                    </p>
                </div>
            );
        };

        const SentimentToggleCard = ({ storageKey, fngTitle, macdTitle, market, symbol, label, subtitle, children }) => {
            const [view, setView] = useLocalState(storageKey, 'fng');
            return (
                <section className="fs-section mt-6 md:mt-8">
                    <div className="fs-head">
                        <div className="text-left">
                            <h2 className="fs-title">{view === 'fng' ? fngTitle : macdTitle}</h2>
                            {view === 'fng' && subtitle}
                        </div>
                        <div className="fs-toggle shrink-0">
                            {[{ k: 'fng', l: '恐懼貪婪' }, { k: 'macd', l: '週線 MACD' }].map(o => (
                                <button key={o.k} onClick={() => setView(o.k)} className={view === o.k ? 'on' : ''}>{o.l}</button>
                            ))}
                        </div>
                    </div>
                    {view === 'fng'
                        ? <div className="w-full">{children}</div>
                        : <WeeklyMacdPanel market={market} symbol={symbol} label={label} />}
                </section>
            );
        };

        // Days lookup shared by price + FNG fetchers
        const rangeToDays = (r) => (
            r === '1mo' ? 30 : r === '3mo' ? 90 : r === '6mo' ? 180 :
            r === '1y' ? 365 : r === '2y' ? 730 : r === '5y' ? 1825 :
            r === 'max' ? 1825 : 365
        );

        // ─────────────────────────────────────────────
        // aggregateToWeekly — collapse a daily price/OHLC series into weekly buckets (Mon-anchored)
        // Each weekly bar: open = first day's open or first price, high = max high, low = min low,
        //                  price (close) = last day's close, date = Monday of the week
        // ─────────────────────────────────────────────
        const aggregateToWeekly = (daily) => {
            if (!Array.isArray(daily) || daily.length === 0) return daily || [];
            const buckets = new Map();
            for (const p of daily) {
                if (!p || !p.date) continue;
                const d = new Date(p.date + 'T00:00:00Z');
                if (isNaN(d.getTime())) continue;
                // Find Monday of the same ISO week (getUTCDay: 0=Sun,1=Mon,...)
                const dow = d.getUTCDay();
                const diff = dow === 0 ? -6 : 1 - dow; // shift back to Monday
                const mon = new Date(d.getTime() + diff * 86400000);
                const key = mon.toISOString().slice(0, 10);
                let b = buckets.get(key);
                if (!b) {
                    b = { date: key, _firstTs: d.getTime(), _lastTs: d.getTime(),
                          open: p.open ?? p.price, high: p.high ?? p.price, low: p.low ?? p.price, price: p.price };
                    buckets.set(key, b);
                } else {
                    if (d.getTime() < b._firstTs) { b.open = p.open ?? p.price; b._firstTs = d.getTime(); }
                    if (d.getTime() > b._lastTs)  { b.price = p.price; b._lastTs = d.getTime(); }
                    const ph = p.high ?? p.price, pl = p.low ?? p.price;
                    if (ph != null) b.high = b.high == null ? ph : Math.max(b.high, ph);
                    if (pl != null) b.low  = b.low  == null ? pl : Math.min(b.low,  pl);
                }
            }
            // Sort by date
            const out = Array.from(buckets.values()).sort((a, b) => a.date.localeCompare(b.date));
            // Clean internal fields
            for (const b of out) { delete b._firstTs; delete b._lastTs; }
            return out;
        };

        // ─────────────────────────────────────────────
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

        // ─── CoinMetrics 社群 API(免金鑰、CORS 開放)───
        // MVRV 與 Realized Price:Realized Price = 現價 ÷ MVRV
        // (社群版拿不到 CapRealUSD,但 MVRV = 市值/實現市值,故 RP = Price/MVRV)
        // 支援資產:btc / eth / link / doge(SOL、HYPE 無 MVRV 資料)
        const CM_ASSET_MAP = {
            btc: 'btc', bitcoin: 'btc',
            eth: 'eth', ethereum: 'eth',
            link: 'link', chainlink: 'link',
            doge: 'doge', dogecoin: 'doge',
        };
        const cmCache = {}; // asset → [{date, mvrv, rp}](session 內共用,modal 和鏈上頁都會用)
        const fetchCoinMetrics = async (asset) => {
            if (cmCache[asset]) return cmCache[asset];
            const url = `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=${asset}&metrics=CapMVRVCur,PriceUSD&frequency=1d&page_size=10000&start_time=2013-01-01`;
            const res = await fetch(url);
            if (!res.ok) throw new Error(`CoinMetrics ${res.status}`);
            const json = await res.json();
            const arr = (json.data || []).map(d => {
                const mvrv = parseFloat(d.CapMVRVCur);
                const price = parseFloat(d.PriceUSD);
                return {
                    date: d.time.slice(0, 10),
                    mvrv: isFinite(mvrv) ? mvrv : null,
                    price: isFinite(price) ? price : null,
                    rp: (isFinite(mvrv) && mvrv > 0 && isFinite(price)) ? price / mvrv : null,
                };
            }).filter(d => d.mvrv != null && d.rp != null);
            cmCache[asset] = arr;
            return arr;
        };
        const mvrvClassify = (v) => {
            if (v == null || !isFinite(v)) return null;
            if (v < 1) return { tone: 'bull', label: '低估(抄底區)' };
            if (v <= 2.4) return { tone: 'neutral', label: '中性' };
            return { tone: 'bear', label: '過熱' };
        };

        // ─── 加密貨幣恐懼與貪婪指數 ───
        // 主來源改用 CoinMarketCap(方法學與 SoSoValue 相近,較不會長期卡在「極度恐懼」;
        // 且全站價格已用 CMC,來源一致)。CMC 這支端點無 CORS,經 cors 代理取用
        // (代理僅允許正式站 origin,本機預覽會失敗)→ 失敗時退回 alternative.me。
        // 回傳與 alternative.me 相同格式:最新在前的 [{value, value_classification, timestamp(秒)}]。
        const FNG_NAME_NORM = {
            'extreme fear': 'Extreme Fear', 'fear': 'Fear', 'neutral': 'Neutral',
            'greed': 'Greed', 'extreme greed': 'Extreme Greed',
        };
        async function fetchCryptoFNG(days = 365) {
            try {
                const now = Math.floor(Date.now() / 1000);
                const start = now - (days + 5) * 86400;
                const cmcUrl = `https://api.coinmarketcap.com/data-api/v3/fear-greed/chart?start=${start}&end=${now}`;
                const res = await fetch('https://cors.hellokai07.com/?' + encodeURIComponent(cmcUrl));
                if (res.ok) {
                    const json = await res.json();
                    const list = json && json.data && json.data.dataList;
                    if (Array.isArray(list) && list.length) {
                        // dataList 為最舊在前 → 反轉成最新在前(與 alternative.me 一致)
                        return list.map(d => ({
                            value: String(d.score),
                            value_classification: FNG_NAME_NORM[String(d.name).toLowerCase()] || d.name,
                            timestamp: String(d.timestamp),
                        })).reverse();
                    }
                }
            } catch (e) { /* fall through to alternative.me */ }
            const res = await fetch(`https://api.alternative.me/fng/?limit=${days}`);
            const json = await res.json();
            return json.data || [];
        }

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
            const rpSeriesRef = useRef(null);          // Realized Price 線(主圖,crypto 限定)
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
            const lastHoverTimeRef = useRef(null);   // crosshair 防抖:同一根 K 棒不重複 setState

            const [showHelp, setShowHelp] = useState(false);
            const [activeTool, setActiveTool] = useState(null); // { type, name, requiredAnchors, collected }
            const [openCategory, setOpenCategory] = useState(null);
            const [hasSelection, setHasSelection] = useState(false);
            const [hoverInfo, setHoverInfo] = useState(null); // { time, open, high, low, close, prevClose }

            const [showEMA, setShowEMA] = useLocalState('tech-show-ema', true);
            const [showMA200, setShowMA200] = useLocalState('tech-show-ma200', true);
            const [showRP, setShowRP] = useLocalState('tech-show-rp', true);
            const [showRSI, setShowRSI] = useLocalState('tech-show-rsi', true);
            const [showMACD, setShowMACD] = useLocalState('tech-show-macd', true);
            const [showFearSignal, setShowFearSignal] = useLocalState('tech-show-fear', true);
            const [showComposite, setShowComposite] = useLocalState('tech-show-composite', true);
            const [showSignalPanel, setShowSignalPanel] = useLocalState('tech-show-panel', true);
            const [timeframe, setTimeframe] = useLocalState('tech-timeframe', '1d'); // '1d' | '1wk'

            const signalSource = (type === 'CRYPTO' || type === 'crypto' || type === 'US') ? 'fng' : 'rsi';
            const isCryptoType = (type === 'CRYPTO' || type === 'crypto');
            // crypto 用鏈上 Realized Price 取代 MA200(Binance 只給 1000 根 K 棒,MA200 沒參考性)
            const cmAsset = isCryptoType ? (CM_ASSET_MAP[String(symbol || '').toLowerCase()] || null) : null;
            const [cmData, setCmData] = useState([]);
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
                            const data = await fetchCryptoFNG(days);
                            if (cancelled) return;
                            const arr = data.map(d => ({
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

            // ─── Fetch MVRV / Realized Price(crypto 限定,session 內快取)───
            useEffect(() => {
                if (!cmAsset) { setCmData([]); return; }
                let cancelled = false;
                (async () => {
                    try {
                        const arr = await fetchCoinMetrics(cmAsset);
                        if (!cancelled) setCmData(arr);
                    } catch (e) {
                        console.warn('CoinMetrics fetch failed:', e);
                        if (!cancelled) setCmData([]);
                    }
                })();
                return () => { cancelled = true; };
            }, [cmAsset]);

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
                    timeScale: {
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderVisible: true,
                        visible: true,
                        timeVisible: true,
                        secondsVisible: false,
                        rightOffset: 6,
                        barSpacing: 8,
                        minBarSpacing: 0.5,
                        tickMarkMaxCharacterLength: 10,
                    },
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

                // ─── Hover-info overlay: read OHLC out of the crosshair payload ───
                // 只在移到「不同的 K 棒」時才 setState:滑鼠在同一根上垂直移動不重繪,
                // 避免整個 modal 在畫線 / 平移時被 React 重繪拖慢。
                chart.subscribeCrosshairMove((param) => {
                    if (!param || param.time == null || !param.seriesData) {
                        if (lastHoverTimeRef.current !== null) { lastHoverTimeRef.current = null; setHoverInfo(null); }
                        return;
                    }
                    const cur = param.seriesData.get(candleSeries);
                    if (!cur || cur.open == null) {
                        if (lastHoverTimeRef.current !== null) { lastHoverTimeRef.current = null; setHoverInfo(null); }
                        return;
                    }
                    if (lastHoverTimeRef.current === param.time) return;
                    lastHoverTimeRef.current = param.time;
                    const data = candleDataRef.current;
                    let prev = null;
                    if (param.logical != null && data.length > 1) {
                        const idx = Math.round(param.logical);
                        if (idx >= 1 && idx < data.length) prev = data[idx - 1];
                    }
                    setHoverInfo({
                        time: param.time,
                        open: cur.open, high: cur.high, low: cur.low, close: cur.close,
                        prevClose: prev ? prev.close : null,
                    });
                });

                if (window.LightweightChartsDrawing && window.LightweightChartsDrawing.DrawingManager) {
                    const manager = new window.LightweightChartsDrawing.DrawingManager();
                    manager.attach(chart, candleSeries, containerRef.current);
                    drawingManagerRef.current = manager;
                    // 選取標註時鎖住圖表平移/縮放:拖曳錨點才不會同時把整張圖拖走
                    manager.on && manager.on('drawing:selected', (e) => {
                        selectedDrawingIdRef.current = e.drawingId || null;
                        setHasSelection(true);
                        try { chart.applyOptions({ handleScroll: false, handleScale: false }); } catch {}
                    });
                    manager.on && manager.on('drawing:deselected', () => {
                        selectedDrawingIdRef.current = null;
                        setHasSelection(false);
                        if (!activeToolRef.current) {
                            try { chart.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
                        }
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
                        // 繪製完成:恢復圖表平移/縮放
                        try { chart.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
                    }
                };
                chart.subscribeClick(onClick);

                // ─── Touch → mouse polyfill so the drawing manager's mouse-only
                //     anchor-drag handlers also respond to finger drags on mobile.
                //     只在「拖曳」且有作用對象(繪製中/已選取標註)時才合成滑鼠事件:
                //     - 純點按交給瀏覽器的相容性事件(否則合成+相容性事件重複觸發,
                //       點一下標註會「選取後立刻取消選取」)
                //     - 一般單指平移圖表時不合成,避免跟繪圖管理器互搶手勢 ───
                const container = containerRef.current;
                const dispatchMouse = (type, t) => {
                    const ev = new MouseEvent(type, {
                        bubbles: true, cancelable: true, view: window,
                        clientX: t.clientX, clientY: t.clientY,
                        button: 0,
                    });
                    container.dispatchEvent(ev);
                };
                let touchDrag = null; // { startT, dragging }
                const onTouchStart = (e) => {
                    if (e.touches.length !== 1) { touchDrag = null; return; }
                    touchDrag = { startT: e.touches[0], dragging: false };
                };
                const onTouchMove = (e) => {
                    if (!touchDrag || e.touches.length !== 1) return;
                    if (!activeToolRef.current && !selectedDrawingIdRef.current) return;
                    if (!touchDrag.dragging) {
                        dispatchMouse('mousedown', touchDrag.startT);
                        touchDrag.dragging = true;
                    }
                    dispatchMouse('mousemove', e.touches[0]);
                };
                const onTouchEnd = (e) => {
                    if (touchDrag && touchDrag.dragging) {
                        const t = e.changedTouches[0];
                        if (t) dispatchMouse('mouseup', t);
                    }
                    touchDrag = null;
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
                            try { chart.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
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
                            // 程式移除不會觸發 deselected 事件,手動解鎖平移
                            if (!activeToolRef.current) {
                                try { chart.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
                            }
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
                    rpSeriesRef.current = null;
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
                // crypto 一律不畫 MA200(K 棒最多 1000 根,MA200 沒參考性;主圖改用 Realized Price)
                if (!showMA200 || isCryptoType || candleData.length < 50) return;
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

            // ─── Realized Price 線(主圖,crypto 限定)───
            // 鏈上平均成本線:價格跌破 = 歷史級抄底區。資料為日頻,
            // 依 K 棒時間對齊取「當日或之前最近」的值(週線也能正確對齊)。
            useEffect(() => {
                const chart = chartRef.current;
                if (!chart) return;
                if (rpSeriesRef.current) { try { chart.removeSeries(rpSeriesRef.current); } catch {} rpSeriesRef.current = null; }
                if (!showRP || !cmAsset || candleData.length === 0 || cmData.length === 0) return;
                const data = [];
                let di = 0, lastRp = null;
                for (const bar of candleData) {
                    const ds = new Date(bar.time * 1000).toISOString().slice(0, 10);
                    while (di < cmData.length && cmData[di].date <= ds) { lastRp = cmData[di].rp; di++; }
                    if (lastRp != null) data.push({ time: bar.time, value: lastRp });
                }
                if (!data.length) return;
                const s = chart.addSeries(window.LightweightCharts.LineSeries, {
                    color: '#f59e0b',
                    lineWidth: 2,
                    priceLineVisible: false,
                    lastValueVisible: true,
                    title: 'Realized Price',
                }, 0);
                s.setData(data);
                rpSeriesRef.current = s;
            }, [showRP, cmAsset, cmData, candleData]);

            // ─── RSI + MACD 副圖(pane 動態配置)───
            // lightweight-charts v5 在某個 pane 的最後一條 series 被移除後,
            // 後面的 pane 會往前遞補;因此不能寫死「RSI=pane1、MACD=pane2」——
            // 舊寫法在「關 RSI 再開」時會把 RSI 加進 MACD 佔用的 pane,兩個指標重疊。
            // 改為單一 effect:先移除兩者、清掉空 pane,再依開關「依序」分配索引,
            // 並在 series 建立之後才設定 pane 高度比例(舊版在建立前設定,無效)。
            useEffect(() => {
                const chart = chartRef.current;
                const LWC = window.LightweightCharts;
                if (!chart) return;

                // 1) 移除既有 RSI / MACD series
                if (rsiSeriesRef.current) { try { chart.removeSeries(rsiSeriesRef.current); } catch {} rsiSeriesRef.current = null; }
                rsiMarkersRef.current = null;
                ['dif', 'dea', 'hist'].forEach(k => {
                    if (macdRefs.current[k]) { try { chart.removeSeries(macdRefs.current[k]); } catch {} macdRefs.current[k] = null; }
                });
                macdMarkersRef.current = null;

                // 2) 清掉殘留的空副圖 pane,讓索引從乾淨狀態開始分配
                try {
                    const panes = chart.panes();
                    for (let i = panes.length - 1; i >= 1; i--) {
                        const series = panes[i].getSeries ? panes[i].getSeries() : [];
                        if (!series || series.length === 0) chart.removePane(i);
                    }
                } catch {}

                if (candleData.length === 0) return;
                const closes = candleData.map(d => d.close);
                let nextPane = 1;

                // 3) RSI — 70 / 50 / 30 參考線 + 上穿30 △ / 下穿70 ▽ 標記
                if (showRSI && closes.length > 14) {
                    const period = 14;
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
                    }, nextPane);
                    s.setData(out);
                    try { s.createPriceLine({ price: 70, color: 'rgba(255,91,110,0.5)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false }); } catch {}
                    try { s.createPriceLine({ price: 50, color: 'rgba(255,255,255,0.18)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false }); } catch {}
                    try { s.createPriceLine({ price: 30, color: 'rgba(0,214,143,0.5)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false }); } catch {}
                    rsiSeriesRef.current = s;

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
                    nextPane++;
                }

                // 4) MACD — DIF×DEA 金叉 △ / 死叉 ▽ 標記在 DIF 線上
                if (showMACD) {
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
                    macdRefs.current.hist = chart.addSeries(LWC.HistogramSeries, { priceLineVisible: false, lastValueVisible: false }, nextPane);
                    macdRefs.current.hist.setData(histData);
                    macdRefs.current.dif = chart.addSeries(LWC.LineSeries, { color: '#3b82f6', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, title: 'DIF' }, nextPane);
                    macdRefs.current.dif.setData(difData);
                    macdRefs.current.dea = chart.addSeries(LWC.LineSeries, { color: '#fbbf24', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, title: 'DEA' }, nextPane);
                    macdRefs.current.dea.setData(deaData);

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
                    nextPane++;
                }

                // 5) pane 高度比例:主圖 6、每個副圖 2(此時 pane 一定已存在才會生效)
                try {
                    const panes = chart.panes();
                    if (panes[0]) panes[0].setStretchFactor(6);
                    for (let i = 1; i < panes.length; i++) panes[i].setStretchFactor(2);
                } catch {}
            }, [showRSI, showMACD, candleData]);

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

                // MACD: DIF 相對 DEA 決定多空,柱狀體與前一根比較看動能是否在減弱
                // (柱狀體 = DIF − DEA,所以「DIF 在 DEA 之上」和「柱狀體為正」是同一件事,不能拿來當兩個條件)
                if (showMACD) {
                    const m = computeMACD(closes);
                    const len = m.histogram.length;
                    const lHist = m.histogram[len - 1];
                    const pHist = m.histogram[len - 2];
                    if (lHist != null && pHist != null) {
                        const above = lHist > 0;
                        const fading = above ? lHist < pHist : lHist > pHist;
                        const tone = fading ? 'neutral' : (above ? 'bull' : 'bear');
                        const label = above ? (fading ? '多方減弱' : '多') : (fading ? '空方減弱' : '空');
                        out.push({ key: 'macd', name: 'MACD', tone, label });
                    }
                }

                // 200MA(非 crypto 限定)
                if (showMA200 && !isCryptoType && closes.length >= 200) {
                    let sum = 0;
                    for (let i = closes.length - 200; i < closes.length; i++) sum += closes[i];
                    const ma200 = sum / 200;
                    const bull = lastPrice >= ma200;
                    out.push({ key: 'ma200', name: 'MA200', tone: bull ? 'bull' : 'bear', label: bull ? '價格站上' : '價格跌破' });
                }

                // MVRV + Realized Price(crypto 限定,鏈上估值)
                if (cmAsset && cmData.length) {
                    const last = cmData[cmData.length - 1];
                    const cls = mvrvClassify(last.mvrv);
                    if (cls) out.push({ key: 'mvrv', name: 'MVRV', tone: cls.tone, label: `${last.mvrv.toFixed(2)} ${cls.label}` });
                    if (showRP && last.rp != null) {
                        const above = lastPrice >= last.rp;
                        out.push({ key: 'rp', name: 'R.Price', tone: above ? 'bull' : 'bear', label: above ? '價格站上' : '跌破(抄底區)' });
                    }
                }

                return out;
            }, [candleData, fngHistory, signalSource, showEMA, showRSI, showMACD, showMA200, showComposite, cmAsset, cmData, showRP]);

            // ─── Tool picker handlers ───
            const pickTool = (tool) => {
                pendingAnchorsRef.current = [];
                activeToolRef.current = { type: tool.type, requiredAnchors: tool.a };
                setActiveTool({ type: tool.type, name: tool.name, requiredAnchors: tool.a, collected: 0 });
                setOpenCategory(null);
                if (containerRef.current) containerRef.current.style.cursor = 'crosshair';
                // 繪製期間鎖住平移/縮放:點錨點時圖表不會跟著滑動,下錨更準
                try { chartRef.current && chartRef.current.applyOptions({ handleScroll: false, handleScale: false }); } catch {}
            };
            const cancelTool = () => {
                activeToolRef.current = null;
                pendingAnchorsRef.current = [];
                setActiveTool(null);
                if (containerRef.current) containerRef.current.style.cursor = '';
                try { chartRef.current && chartRef.current.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
            };
            // 程式移除標註不會觸發 deselected 事件 → 手動解鎖圖表平移/縮放
            const unlockChart = () => {
                if (activeToolRef.current) return;
                try { chartRef.current && chartRef.current.applyOptions({ handleScroll: true, handleScale: true }); } catch {}
            };
            const clearAllDrawings = () => {
                const m = drawingManagerRef.current;
                if (!m) return;
                if (!confirm('清除所有畫線標註？')) return;
                m.clearAll();
                drawingsRef.current = [];
                selectedDrawingIdRef.current = null;
                setHasSelection(false);
                unlockChart();
            };
            const deleteSelected = () => {
                const m = drawingManagerRef.current;
                const id = selectedDrawingIdRef.current;
                if (!m || !id) return;
                m.removeDrawing(id);
                drawingsRef.current = drawingsRef.current.filter(x => x !== id);
                selectedDrawingIdRef.current = null;
                setHasSelection(false);
                unlockChart();
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
                <div className="fixed inset-0 z-[110] flex flex-col" style={{ background: 'var(--bg)', height: '100dvh', paddingTop: 'env(safe-area-inset-top, 0px)', paddingBottom: 'env(safe-area-inset-bottom, 0px)' }}>
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
                            <div className="flex items-center gap-0.5 p-0.5 rounded-lg shrink-0" style={{ background: 'var(--wash)', border: '1px solid var(--line)' }}>
                                {rangeBtn('1mo', '1M')}
                                {rangeBtn('3mo', '3M')}
                                {rangeBtn('6mo', '6M')}
                                {rangeBtn('1y', '1Y')}
                                {rangeBtn('max', 'ALL')}
                            </div>

                            <div className="flex items-center gap-0.5 p-0.5 rounded-lg shrink-0" style={{ background: 'var(--wash)', border: '1px solid var(--line)' }} title="K 線週期">
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
                            {cmAsset ? (
                                <EyeToggle label="Realized Price" checked={showRP} onChange={setShowRP} color="#f59e0b" />
                            ) : !isCryptoType ? (
                                <EyeToggle label="MA200" checked={showMA200} onChange={setShowMA200} color="#f97316" />
                            ) : null}
                            <EyeToggle label="RSI" checked={showRSI} onChange={setShowRSI} color="#a78bfa" />
                            <EyeToggle label="MACD" checked={showMACD} onChange={setShowMACD} color="#fbbf24" />
                            <EyeToggle label="綜合信號" checked={showComposite} onChange={setShowComposite} color={UP_COLOR} />
                        </div>
                    </div>

                    {/* Body: drawing toolbar + chart */}
                    <div className="flex-1 min-h-0 flex">
                        {/* Vertical toolbar (left). Categories pop a submenu of tools. */}
                        <div className="shrink-0 flex flex-col items-stretch gap-1 py-2 px-1.5" style={{ borderRight: '1px solid var(--line)', width: 48, background: 'var(--wash)' }}>
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

                        {/* Chart container — absolute inset so the chart canvas can NEVER
                            overflow its parent and clip the bottom time axis. The min-height
                            lives on the parent so a tiny viewport still gets a usable chart. */}
                        <div className="flex-1 min-w-0 min-h-0 relative" style={{ minHeight: '280px' }}>
                            <div ref={containerRef} className="absolute inset-1 md:inset-2"></div>

                            {/* OHLC hover overlay — top-left, compact. Driven by crosshair. */}
                            {hoverInfo && (() => {
                                const fmtPrice = (v) => v == null ? '—' : (Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : v.toFixed(Math.abs(v) >= 1 ? 2 : 4));
                                const change = (hoverInfo.prevClose != null) ? hoverInfo.close - hoverInfo.prevClose : null;
                                const changePct = (change != null && hoverInfo.prevClose) ? (change / hoverInfo.prevClose) * 100 : null;
                                const changeColor = change == null ? 'var(--text-3)' : (change >= 0 ? UP_COLOR : DOWN_COLOR);
                                const d = new Date(hoverInfo.time * 1000);
                                const dateStr = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
                                const isUp = hoverInfo.close >= hoverInfo.open;
                                return (
                                    <div className="absolute top-2 left-2 z-20 px-2 py-1.5 rounded-md backdrop-blur pointer-events-none mono"
                                        style={{ background: 'rgba(18,20,28,0.85)', border: '1px solid var(--line)', fontSize: 10, lineHeight: 1.35, color: 'var(--text-2)' }}>
                                        <div className="font-bold" style={{ color: 'var(--text)' }}>{dateStr}</div>
                                        <div className="flex flex-wrap gap-x-2 gap-y-0.5 mt-0.5">
                                            <span>開 <span style={{ color: 'var(--text)' }}>{fmtPrice(hoverInfo.open)}</span></span>
                                            <span>高 <span style={{ color: UP_COLOR }}>{fmtPrice(hoverInfo.high)}</span></span>
                                            <span>低 <span style={{ color: DOWN_COLOR }}>{fmtPrice(hoverInfo.low)}</span></span>
                                            <span>收 <span style={{ color: isUp ? UP_COLOR : DOWN_COLOR }}>{fmtPrice(hoverInfo.close)}</span></span>
                                        </div>
                                        {change != null && (
                                            <div className="mt-0.5" style={{ color: changeColor }}>
                                                {change >= 0 ? '+' : ''}{fmtPrice(change)} ({changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%)
                                            </div>
                                        )}
                                    </div>
                                );
                            })()}

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
                                    <p><strong className="text-white">RSI / MACD</strong>：RSI 紫線恆色，加 70 / 50 / 30 參考線，上穿 30 標 △、下穿 70 標 ▽；MACD DIF×DEA 金叉 △、死叉 ▽。</p>
                                    <p><strong className="text-white">Realized Price / MVRV</strong>（加密貨幣）：橘線為鏈上平均持倉成本（Realized Price），價格跌破 = 歷史級抄底區；MVRV = 市值 ÷ 實現市值，&lt;1 低估、&gt;2.4 過熱。資料源 CoinMetrics（BTC / ETH / LINK / DOGE）。美股 / 台股則顯示 MA200。</p>
                                    <p><strong className="text-white">信號總覽</strong>：右上角面板獨立列出各指標的多 / 空狀態，不做綜合判斷；可隨時收起。</p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            );
        };

        // ═══════════════════════════════════════════════
        // OnChainDashboard — 鏈上數據頁
        //   1. MVRV & Realized Price(CoinMetrics 社群 API,免金鑰)
        //   2. 美國現貨 ETF 淨流入(SoSoValue 公開端點,免金鑰)
        //   3. 穩定幣市值與淨流入(DefiLlama,免金鑰)
        // ═══════════════════════════════════════════════
        const ONCHAIN_MVRV_ASSETS = [
            { id: 'btc', label: 'BTC' },
            { id: 'eth', label: 'ETH' },
            { id: 'link', label: 'LINK' },
            { id: 'doge', label: 'DOGE' },
            { id: 'sol', label: 'SOL', unsupported: true },
            { id: 'hype', label: 'HYPE', unsupported: true },
        ];
        // 只列出實際有現貨 ETF 資料的幣種(SoSoValue 目前僅 BTC/ETH/SOL 有);
        // 其餘(XRP/DOGE/BNB/LINK/HYPE)回空陣列,不列出以免版面溢出。
        const ETF_TYPES = [
            { id: 'us-btc-spot', label: 'BTC' },
            { id: 'us-eth-spot', label: 'ETH' },
            { id: 'us-sol-spot', label: 'SOL' },
        ];
        const fmtUsdCompact = (v) => {
            if (v == null || !isFinite(v)) return '—';
            const abs = Math.abs(v);
            const sign = v < 0 ? '-' : '';
            if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
            if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
            if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
            if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
            return `${sign}$${abs.toFixed(2)}`;
        };

        // 低階 Chart.js 畫布:data 變了就重建(沿用全站深色主題預設)。
        // withZoom = true 時掛上滾輪/雙指縮放 + 拖曳平移(僅放大檢視用)。
        const ChartCanvas = ({ build, deps, withZoom, onChart }) => {
            const canvasRef = useRef(null);
            const chartRef = useRef(null);
            useEffect(() => {
                if (!canvasRef.current || typeof Chart === 'undefined') return;
                if (chartRef.current) { try { chartRef.current.destroy(); } catch {} chartRef.current = null; }
                const cfg = build();
                if (!cfg) return;
                if (withZoom) {
                    cfg.options = cfg.options || {};
                    cfg.options.plugins = cfg.options.plugins || {};
                    cfg.options.plugins.zoom = {
                        pan: { enabled: true, mode: 'x' },
                        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                    };
                }
                const c = new Chart(canvasRef.current.getContext('2d'), cfg);
                chartRef.current = c;
                onChart && onChart(c);
                return () => { if (chartRef.current) { try { chartRef.current.destroy(); } catch {} chartRef.current = null; } };
            }, deps); // eslint-disable-line
            return <canvas ref={canvasRef}></canvas>;
        };

        // 通用鏈上圖:內嵌小圖 + 右上角「放大」→ 全螢幕可縮放檢視
        const OnChainChart = ({ build, deps, height = 260, title }) => {
            const [expanded, setExpanded] = useState(false);
            const modalChartRef = useRef(null);
            return (
                <>
                    <div className="relative">
                        <div style={{ height }} className="relative"><ChartCanvas build={build} deps={deps} /></div>
                        <button
                            onClick={() => setExpanded(true)}
                            className="absolute top-1 right-1 w-7 h-7 rounded-md flex items-center justify-center backdrop-blur hover:bg-white/[0.12]"
                            style={{ background: 'rgba(18,20,28,0.8)', border: '1px solid var(--line)', color: 'var(--text-2)' }}
                            title="放大檢視"
                        >
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/>
                            </svg>
                        </button>
                    </div>
                    {expanded && (
                        <div
                            className="fixed inset-0 z-[130] flex flex-col p-3"
                            style={{ background: 'var(--bg)', paddingTop: 'calc(env(safe-area-inset-top, 0px) + 8px)', paddingBottom: 'calc(env(safe-area-inset-bottom, 0px) + 8px)' }}
                        >
                            <div className="flex items-center justify-between mb-2 px-1 gap-2">
                                <span className="text-sm font-bold text-white truncate">{title || '圖表'}</span>
                                <div className="flex items-center gap-2 shrink-0">
                                    <span className="hidden sm:inline text-[10px]" style={{ color: 'var(--text-3)' }}>滾輪 / 雙指縮放 · 拖曳平移</span>
                                    <button
                                        onClick={() => { try { modalChartRef.current && modalChartRef.current.resetZoom(); } catch {} }}
                                        className="h-7 px-2 rounded-md text-[11px] hover:bg-white/[0.08]"
                                        style={{ color: 'var(--text-2)', border: '1px solid var(--line)' }}
                                    >⤾ 重置</button>
                                    <button
                                        onClick={() => setExpanded(false)}
                                        className="w-7 h-7 rounded-md flex items-center justify-center hover:bg-white/10"
                                        style={{ color: 'var(--text-2)', border: '1px solid var(--line)' }}
                                    >✕</button>
                                </div>
                            </div>
                            <div className="flex-1 min-h-0 relative">
                                <div className="absolute inset-0">
                                    <ChartCanvas build={build} deps={deps} withZoom onChart={(c) => { modalChartRef.current = c; }} />
                                </div>
                            </div>
                        </div>
                    )}
                </>
            );
        };

        // 緊湊下拉選單:取代整排 pill(尤其時間維度),放在卡片標題右側,點開才展開選項。
        // 按鈕顯示目前選中值 + 前綴 icon;點外面關閉。
        const Dropdown = ({ value, options, onChange, icon }) => {
            const [open, setOpen] = useState(false);
            const ref = useRef(null);
            useEffect(() => {
                if (!open) return;
                const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
                document.addEventListener('mousedown', onDoc);
                return () => document.removeEventListener('mousedown', onDoc);
            }, [open]);
            const current = options.find(o => o.key === value);
            return (
                <div className="relative shrink-0" ref={ref}>
                    <button
                        onClick={() => setOpen(o => !o)}
                        className="flex items-center gap-1.5 pl-2.5 pr-2 h-8 rounded-lg text-[11px] font-bold hover:bg-white/[0.08] transition-colors"
                        style={{ background: 'var(--wash)', border: '1px solid ' + (open ? 'var(--brand-1)' : 'var(--line)'), color: open ? 'var(--brand-1)' : 'var(--text-2)' }}
                    >
                        {icon && (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>
                            </svg>
                        )}
                        <span>{current ? current.label : value}</span>
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }}>
                            <polyline points="6 9 12 15 18 9"/>
                        </svg>
                    </button>
                    {open && (
                        <div className="absolute right-0 mt-1 z-40 rounded-lg py-1 shadow-2xl" style={{ background: 'var(--surface-2)', border: '1px solid var(--line-2)', minWidth: 108 }}>
                            {options.map(o => (
                                <button
                                    key={o.key}
                                    onClick={() => { if (!o.disabled) { onChange(o.key); setOpen(false); } }}
                                    disabled={o.disabled}
                                    className={`block w-full text-left px-3 py-1.5 text-[11px] font-semibold ${o.disabled ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white/[0.07]'}`}
                                    style={{ color: o.key === value ? 'var(--brand-1)' : 'var(--text-2)' }}
                                >
                                    {o.label}
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            );
        };

        // ─── 合約數據(Coinglass 風格):Binance 合約公開 API,免金鑰、CORS 開放 ───
        // 資金費率 / 持倉量 / 全體帳戶多空比 / 大戶持倉多空比 / 主動買賣比。
        // futures/data 系列僅保留最近 30 天;爆倉數據 Binance 只提供 WebSocket,略過。
        const DERIV_COINS = [
            { sym: 'BTCUSDT', label: 'BTC' },
            { sym: 'ETHUSDT', label: 'ETH' },
            { sym: 'SOLUSDT', label: 'SOL' },
            { sym: 'LINKUSDT', label: 'LINK' },
            { sym: 'HYPEUSDT', label: 'HYPE' },
        ];
        const derivCache = {}; // `${sym}|${period}` → data(session 內快取)
        async function fetchDerivatives(sym, period) {
            const key = `${sym}|${period}`;
            if (derivCache[key]) return derivCache[key];
            const B = 'https://fapi.binance.com';
            const j = (url) => fetch(url).then(r => { if (!r.ok) throw new Error(`Binance ${r.status}`); return r.json(); });
            const [premium, oiNow, funding, oiHist, globalLS, topLS, taker] = await Promise.all([
                j(`${B}/fapi/v1/premiumIndex?symbol=${sym}`),
                j(`${B}/fapi/v1/openInterest?symbol=${sym}`),
                j(`${B}/fapi/v1/fundingRate?symbol=${sym}&limit=90`),                              // 8h 一筆 ≈ 30 天
                j(`${B}/futures/data/openInterestHist?symbol=${sym}&period=${period}&limit=500`),
                j(`${B}/futures/data/globalLongShortAccountRatio?symbol=${sym}&period=${period}&limit=500`),
                j(`${B}/futures/data/topLongShortPositionRatio?symbol=${sym}&period=${period}&limit=500`),
                j(`${B}/futures/data/takerlongshortRatio?symbol=${sym}&period=${period}&limit=500`),
            ]);
            const price = parseFloat(premium.markPrice);
            const data = {
                price,
                fundingNow: parseFloat(premium.lastFundingRate) * 100,          // %
                oiNowUsd: parseFloat(oiNow.openInterest) * price,
                funding: (funding || []).map(d => ({ t: d.fundingTime, v: parseFloat(d.fundingRate) * 100 })),
                oi: (oiHist || []).map(d => ({ t: d.timestamp, v: parseFloat(d.sumOpenInterestValue) })),
                globalLS: (globalLS || []).map(d => ({ t: d.timestamp, v: parseFloat(d.longShortRatio), longPct: parseFloat(d.longAccount) * 100 })),
                topLS: (topLS || []).map(d => ({ t: d.timestamp, v: parseFloat(d.longShortRatio) })),
                taker: (taker || []).map(d => ({ t: d.timestamp, v: parseFloat(d.buySellRatio) })),
            };
            derivCache[key] = data;
            return data;
        }
        const derivTimeLabel = (t) => {
            const d = new Date(t);
            return `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:00`;
        };

        const DerivativesPanel = ({ defaultAsset }) => {
            const initialSym = (() => {
                try {
                    const bin = getBinanceSymbol(String(defaultAsset || '').toLowerCase());
                    if (bin && DERIV_COINS.some(c => c.sym === bin)) return bin;
                } catch {}
                return 'BTCUSDT';
            })();
            const [sym, setSym] = useState(initialSym);
            const [period, setPeriod] = useState('4h'); // 1h | 4h | 12h | 1d
            const [data, setData] = useState(null);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);

            useEffect(() => {
                let cancelled = false;
                (async () => {
                    setLoading(true);
                    setError(null);
                    try {
                        const d = await fetchDerivatives(sym, period);
                        if (!cancelled) setData(d);
                    } catch (e) {
                        if (!cancelled) { setError(String(e.message || e)); setData(null); }
                    } finally {
                        if (!cancelled) setLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, [sym, period]);

            const pills = (items, active, onPick) => (
                <div className="flex gap-0.5 p-0.5 rounded-lg overflow-x-auto no-scrollbar" style={{ background: 'var(--wash)', border: '1px solid var(--line)', width: 'fit-content', maxWidth: '100%' }}>
                    {items.map(it => (
                        <button key={it.key}
                            onClick={() => onPick(it.key)}
                            className={`px-2.5 h-7 rounded-md text-[11px] font-bold transition-all shrink-0 ${active === it.key ? 'pill-grad' : 'hover:bg-white/[0.08]'}`}
                            style={active === it.key ? { color: 'var(--brand-ink)' } : { color: 'var(--text-2)' }}
                        >{it.label}</button>
                    ))}
                </div>
            );
            const statCard = (label, value, color) => (
                <div className="rounded-xl p-3" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                    <p className="label">{label}</p>
                    <p className="text-base md:text-lg font-extrabold num mt-0.5" style={{ color: color || 'var(--text)' }}>{value}</p>
                </div>
            );
            const gridColor = 'rgba(255,255,255,0.05)';
            const coinLabel = (DERIV_COINS.find(c => c.sym === sym) || {}).label || sym;
            const lastGlobal = data && data.globalLS.length ? data.globalLS[data.globalLS.length - 1] : null;

            // 比值線圖(多空比/買賣比):以 1.0 為分界,上綠下紅
            const ratioChart = (rows, label) => ({
                type: 'line',
                data: {
                    labels: rows.map(d => derivTimeLabel(d.t)),
                    datasets: [{
                        label,
                        data: rows.map(d => d.v),
                        borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, tension: 0.2,
                        fill: { target: { value: 1 }, above: 'rgba(0,214,143,0.12)', below: 'rgba(255,91,110,0.12)' },
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false, animation: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: { legend: { display: false }, title: { display: true, text: `${label}（>1 偏多 · <1 偏空）`, font: { size: 11 } } },
                    scales: {
                        x: { ticks: { maxTicksLimit: 6, maxRotation: 0 }, grid: { color: gridColor } },
                        y: { grid: { color: gridColor } },
                    },
                },
            });

            return (
                <div className="space-y-4">
                    <div className="rounded-2xl ring-soft p-4 md:p-5 space-y-3" style={{ background: 'var(--surface)' }}>
                        <div className="flex items-center justify-between gap-2">
                            <h3 className="font-bold text-white">合約數據（Binance 永續）</h3>
                            <Dropdown value={period} onChange={setPeriod} options={[{ key: '1h', label: '1 小時' }, { key: '4h', label: '4 小時' }, { key: '12h', label: '12 小時' }, { key: '1d', label: '1 天' }]} />
                        </div>
                        {pills(DERIV_COINS.map(c => ({ key: c.sym, label: c.label })), sym, setSym)}
                        <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                            資金費率為正 = 多方付費給空方（市場偏多）；持倉量驟增伴隨價格急拉 / 急殺常見於軋空、殺多。歷史區間為最近 30 天。
                        </p>
                        {loading ? (
                            <div className="text-center py-12">
                                <div className="inline-block w-7 h-7 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: 'var(--brand-1)' }}></div>
                            </div>
                        ) : error ? (
                            <p className="text-sm py-6 text-center" style={{ color: 'var(--down)' }}>讀取失敗：{error}</p>
                        ) : !data ? null : (
                            <>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    {statCard('標記價格', fmtUsdCompact(data.price))}
                                    {statCard('當前資金費率', `${data.fundingNow >= 0 ? '+' : ''}${data.fundingNow.toFixed(4)}%`, data.fundingNow >= 0 ? 'var(--up)' : 'var(--down)')}
                                    {statCard('持倉量', fmtUsdCompact(data.oiNowUsd))}
                                    {statCard('多空比(帳戶)', lastGlobal ? `${lastGlobal.v.toFixed(2)}（多 ${lastGlobal.longPct.toFixed(0)}%）` : '—', lastGlobal ? (lastGlobal.v >= 1 ? 'var(--up)' : 'var(--down)') : undefined)}
                                </div>

                                {/* 資金費率 */}
                                <OnChainChart
                                    height={200}
                                    title={`${coinLabel} · 資金費率（8 小時）`}
                                    deps={[data]}
                                    build={() => ({
                                        type: 'bar',
                                        data: {
                                            labels: data.funding.map(d => derivTimeLabel(d.t)),
                                            datasets: [{
                                                label: '資金費率 %',
                                                data: data.funding.map(d => d.v),
                                                backgroundColor: data.funding.map(d => d.v >= 0 ? 'rgba(0,214,143,0.7)' : 'rgba(255,91,110,0.7)'),
                                            }],
                                        },
                                        options: {
                                            responsive: true, maintainAspectRatio: false, animation: false,
                                            interaction: { mode: 'index', intersect: false },
                                            plugins: { legend: { display: false }, title: { display: true, text: '資金費率 %（正 = 多方付費）', font: { size: 11 } } },
                                            scales: {
                                                x: { ticks: { maxTicksLimit: 6, maxRotation: 0 }, grid: { color: gridColor } },
                                                y: { grid: { color: gridColor }, ticks: { callback: (v) => Number(v).toFixed(3) + '%' } },
                                            },
                                        },
                                    })}
                                />

                                {/* 持倉量 */}
                                <OnChainChart
                                    height={220}
                                    title={`${coinLabel} · 持倉量（USD）`}
                                    deps={[data]}
                                    build={() => ({
                                        type: 'line',
                                        data: {
                                            labels: data.oi.map(d => derivTimeLabel(d.t)),
                                            datasets: [{
                                                label: '持倉量',
                                                data: data.oi.map(d => d.v),
                                                borderColor: '#4cc2ff', borderWidth: 1.5, pointRadius: 0, tension: 0.2,
                                                fill: true, backgroundColor: 'rgba(76,194,255,0.08)',
                                            }],
                                        },
                                        options: {
                                            responsive: true, maintainAspectRatio: false, animation: false,
                                            interaction: { mode: 'index', intersect: false },
                                            plugins: { legend: { display: false }, title: { display: true, text: '未平倉合約價值', font: { size: 11 } } },
                                            scales: {
                                                x: { ticks: { maxTicksLimit: 6, maxRotation: 0 }, grid: { color: gridColor } },
                                                y: { grid: { color: gridColor }, ticks: { callback: (v) => fmtUsdCompact(v) } },
                                            },
                                        },
                                    })}
                                />

                                {/* 多空比三張 */}
                                <OnChainChart height={180} title={`${coinLabel} · 全體帳戶多空比`} deps={[data]} build={() => ratioChart(data.globalLS, '全體帳戶多空比')} />
                                <OnChainChart height={180} title={`${coinLabel} · 大戶持倉多空比`} deps={[data]} build={() => ratioChart(data.topLS, '大戶持倉多空比')} />
                                <OnChainChart height={180} title={`${coinLabel} · 主動買賣比`} deps={[data]} build={() => ratioChart(data.taker, '主動買賣比（Taker）')} />
                            </>
                        )}
                    </div>

                    <p className="text-[11px] px-1 pt-1" style={{ color: 'var(--text-3)' }}>合約數據資料源：Binance Futures 公開 API（免金鑰；futures/data 系列歷史僅保留 30 天）</p>
                </div>
            );
        };

        const OnChainDashboard = ({ defaultAsset }) => {
            // ── 1. MVRV / Realized Price ──
            // 預設跟隨加密分頁選中的幣;不支援 MVRV 的幣(SOL/HYPE)則退回 BTC
            const [mvrvAsset, setMvrvAsset] = useState(() => CM_ASSET_MAP[String(defaultAsset || '').toLowerCase()] || 'btc');
            const [mvrvRange, setMvrvRange] = useState('4y'); // 1y | 2y | 4y | all
            const [mvrvData, setMvrvData] = useState([]);
            const [mvrvLoading, setMvrvLoading] = useState(true);
            const [mvrvError, setMvrvError] = useState(null);

            useEffect(() => {
                let cancelled = false;
                (async () => {
                    setMvrvLoading(true);
                    setMvrvError(null);
                    try {
                        const arr = await fetchCoinMetrics(mvrvAsset);
                        if (!cancelled) setMvrvData(arr);
                    } catch (e) {
                        if (!cancelled) { setMvrvError(String(e.message || e)); setMvrvData([]); }
                    } finally {
                        if (!cancelled) setMvrvLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, [mvrvAsset]);

            const mvrvView = useMemo(() => {
                if (!mvrvData.length) return [];
                const days = mvrvRange === '1y' ? 365 : mvrvRange === '2y' ? 730 : mvrvRange === '4y' ? 1461 : Infinity;
                const sliced = days === Infinity ? mvrvData : mvrvData.slice(-days);
                // ALL 範圍點太多會拖慢 Chart.js → 週抽樣
                if (sliced.length > 1500) return sliced.filter((_, i) => i % 7 === 0 || i === sliced.length - 1);
                return sliced;
            }, [mvrvData, mvrvRange]);

            const mvrvLast = mvrvData.length ? mvrvData[mvrvData.length - 1] : null;
            const mvrvCls = mvrvLast ? mvrvClassify(mvrvLast.mvrv) : null;

            // ── 2. ETF 淨流入 ──
            const [etfType, setEtfType] = useState('us-btc-spot');
            const [etfRange, setEtfRange] = useState('3mo'); // 1mo | 3mo | 1y | all
            const [etfData, setEtfData] = useState([]);       // asc by date
            const [etfLoading, setEtfLoading] = useState(true);
            const [etfError, setEtfError] = useState(null);
            const etfCacheRef = useRef({});

            useEffect(() => {
                let cancelled = false;
                (async () => {
                    setEtfLoading(true);
                    setEtfError(null);
                    try {
                        let rows = etfCacheRef.current[etfType];
                        if (!rows) {
                            const res = await fetch('https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ type: etfType }),
                            });
                            const json = await res.json();
                            rows = (json.data || [])
                                .map(d => ({
                                    date: d.date,
                                    flow: parseFloat(d.totalNetInflow),
                                    cum: parseFloat(d.cumNetInflow),
                                    assets: parseFloat(d.totalNetAssets),
                                }))
                                .filter(d => d.date && isFinite(d.flow))
                                .sort((a, b) => a.date < b.date ? -1 : 1);
                            etfCacheRef.current[etfType] = rows;
                        }
                        if (!cancelled) setEtfData(rows);
                    } catch (e) {
                        if (!cancelled) { setEtfError(String(e.message || e)); setEtfData([]); }
                    } finally {
                        if (!cancelled) setEtfLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, [etfType]);

            const etfView = useMemo(() => {
                if (!etfData.length) return [];
                const days = etfRange === '1mo' ? 30 : etfRange === '3mo' ? 90 : etfRange === '1y' ? 365 : Infinity;
                return days === Infinity ? etfData : etfData.slice(-days);
            }, [etfData, etfRange]);

            const etfLast = etfData.length ? etfData[etfData.length - 1] : null;

            // ── 3. 穩定幣(DefiLlama)──
            const [stableView, setStableView] = useState('all'); // all | usdt
            const [stableRange, setStableRange] = useState('1y'); // 3mo | 1y | all
            const [stableData, setStableData] = useState({ all: null, usdt: null });
            const [stableLoading, setStableLoading] = useState(true);

            useEffect(() => {
                let cancelled = false;
                (async () => {
                    setStableLoading(true);
                    try {
                        // 全部穩定幣:/stablecoincharts/all(每筆 totalCirculatingUSD)
                        const parseAll = (json) => (json || [])
                            .map(d => ({ date: new Date(Number(d.date) * 1000).toISOString().slice(0, 10), mcap: d.totalCirculatingUSD?.peggedUSD }))
                            .filter(d => d.mcap != null && isFinite(d.mcap));
                        // USDT:改用 /stablecoin/1(tokens[].circulating)。
                        // /stablecoincharts/all?stablecoin=1 的 CDN 會回傳「重複的
                        // Access-Control-Allow-Origin」標頭,瀏覽器判定無效而擋掉。
                        const parseCoin = (json) => ((json && json.tokens) || [])
                            .map(d => ({ date: new Date(Number(d.date) * 1000).toISOString().slice(0, 10), mcap: d.circulating?.peggedUSD }))
                            .filter(d => d.mcap != null && isFinite(d.mcap));
                        const [allRes, usdtRes] = await Promise.all([
                            fetch('https://stablecoins.llama.fi/stablecoincharts/all').then(r => r.json()),
                            fetch('https://stablecoins.llama.fi/stablecoin/1').then(r => r.json()),
                        ]);
                        if (cancelled) return;
                        setStableData({ all: parseAll(allRes), usdt: parseCoin(usdtRes) });
                    } catch (e) {
                        console.warn('DefiLlama fetch failed:', e);
                        if (!cancelled) setStableData({ all: [], usdt: [] });
                    } finally {
                        if (!cancelled) setStableLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, []);

            const stableSeries = stableData[stableView] || [];
            const stableViewData = useMemo(() => {
                if (!stableSeries.length) return [];
                const days = stableRange === '3mo' ? 90 : stableRange === '1y' ? 365 : Infinity;
                let arr = days === Infinity ? stableSeries : stableSeries.slice(-days);
                if (arr.length > 1500) arr = arr.filter((_, i) => i % 7 === 0 || i === arr.length - 1);
                // 淨流入 = 市值日變化
                return arr.map((d, i) => ({ ...d, flow: i > 0 ? d.mcap - arr[i - 1].mcap : 0 }));
            }, [stableSeries, stableRange]);

            const stableStats = useMemo(() => {
                const s = stableSeries;
                if (!s || s.length < 31) return null;
                const last = s[s.length - 1];
                return {
                    mcap: last.mcap,
                    d7: last.mcap - s[s.length - 8].mcap,
                    d30: last.mcap - s[s.length - 31].mcap,
                };
            }, [stableSeries]);

            // ── UI atoms ──
            const pills = (items, active, onPick) => (
                <div className="flex gap-0.5 p-0.5 rounded-lg overflow-x-auto no-scrollbar" style={{ background: 'var(--wash)', border: '1px solid var(--line)', width: 'fit-content', maxWidth: '100%' }}>
                    {items.map(it => (
                        <button key={it.key}
                            onClick={() => !it.disabled && onPick(it.key)}
                            disabled={it.disabled}
                            title={it.disabled ? '暫無資料' : undefined}
                            className={`px-2.5 h-7 rounded-md text-[11px] font-bold transition-all shrink-0 ${active === it.key ? 'pill-grad' : it.disabled ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white/[0.08]'}`}
                            style={active === it.key ? { color: 'var(--brand-ink)' } : { color: 'var(--text-2)' }}
                        >{it.label}</button>
                    ))}
                </div>
            );
            const statCard = (label, value, color) => (
                <div className="rounded-xl p-3" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                    <p className="label">{label}</p>
                    <p className="text-base md:text-lg font-extrabold num mt-0.5" style={{ color: color || 'var(--text)' }}>{value}</p>
                </div>
            );
            const spinner = (
                <div className="text-center py-12">
                    <div className="inline-block w-7 h-7 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: 'var(--brand-1)' }}></div>
                </div>
            );
            const gridColor = 'rgba(255,255,255,0.05)';

            return (
                <div className="space-y-4">
                    {/* ── 1. MVRV / Realized Price ── */}
                    <div className="rounded-2xl ring-soft p-4 md:p-5 space-y-3" style={{ background: 'var(--surface)' }}>
                        <div className="flex items-center justify-between gap-2">
                            <h3 className="font-bold text-white">MVRV & Realized Price</h3>
                            <Dropdown icon value={mvrvRange} onChange={setMvrvRange} options={[{ key: '1y', label: '近 1 年' }, { key: '2y', label: '近 2 年' }, { key: '4y', label: '近 4 年' }, { key: 'all', label: '全部' }]} />
                        </div>
                        {pills(ONCHAIN_MVRV_ASSETS.map(a => ({ key: a.id, label: a.label, disabled: a.unsupported })), mvrvAsset, setMvrvAsset)}
                        <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                            Realized Price = 全網平均持倉成本；價格跌破橘線 = 鏈上持有者整體虧損，歷史上是週期底部區。MVRV &lt; 1 低估（綠區）、&gt; 2.4 過熱。SOL / HYPE 無公開 MVRV 資料。
                        </p>
                        {mvrvLoading ? spinner : mvrvError ? (
                            <p className="text-sm py-6 text-center" style={{ color: 'var(--down)' }}>讀取失敗：{mvrvError}</p>
                        ) : mvrvView.length === 0 ? (
                            <p className="text-sm py-6 text-center" style={{ color: 'var(--text-3)' }}>此資產暫無 MVRV 資料</p>
                        ) : (
                            <>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    {statCard('現價', mvrvLast?.price != null ? fmtUsdCompact(mvrvLast.price) : '—')}
                                    {statCard('Realized Price', mvrvLast?.rp != null ? fmtUsdCompact(mvrvLast.rp) : '—', '#f59e0b')}
                                    {statCard('MVRV', mvrvLast ? mvrvLast.mvrv.toFixed(2) : '—')}
                                    {statCard('估值狀態', mvrvCls ? mvrvCls.label : '—', mvrvCls ? (mvrvCls.tone === 'bull' ? 'var(--up)' : mvrvCls.tone === 'bear' ? 'var(--down)' : 'var(--text-2)') : undefined)}
                                </div>
                                <OnChainChart
                                    height={280}
                                    title={`${mvrvAsset.toUpperCase()} · 價格 vs Realized Price`}
                                    deps={[mvrvView]}
                                    build={() => ({
                                        type: 'line',
                                        data: {
                                            labels: mvrvView.map(d => d.date),
                                            datasets: [
                                                { label: '價格', data: mvrvView.map(d => d.price), borderColor: '#e2e8f0', borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
                                                { label: 'Realized Price', data: mvrvView.map(d => d.rp), borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0, tension: 0.2 },
                                            ],
                                        },
                                        options: {
                                            responsive: true, maintainAspectRatio: false, animation: false,
                                            interaction: { mode: 'index', intersect: false },
                                            plugins: { legend: { display: true, labels: { boxWidth: 10, boxHeight: 2 } } },
                                            scales: {
                                                x: { ticks: { maxTicksLimit: 7, maxRotation: 0 }, grid: { color: gridColor } },
                                                y: { type: 'logarithmic', grid: { color: gridColor }, ticks: { callback: (v) => fmtUsdCompact(v) } },
                                            },
                                        },
                                    })}
                                />
                                <OnChainChart
                                    height={160}
                                    title={`${mvrvAsset.toUpperCase()} · MVRV`}
                                    deps={[mvrvView]}
                                    build={() => ({
                                        type: 'line',
                                        data: {
                                            labels: mvrvView.map(d => d.date),
                                            datasets: [{
                                                label: 'MVRV',
                                                data: mvrvView.map(d => d.mvrv),
                                                borderColor: '#a78bfa', borderWidth: 1.5, pointRadius: 0, tension: 0.2,
                                                fill: { target: { value: 1 }, above: 'rgba(0,0,0,0)', below: 'rgba(0,214,143,0.15)' },
                                            }],
                                        },
                                        options: {
                                            responsive: true, maintainAspectRatio: false, animation: false,
                                            interaction: { mode: 'index', intersect: false },
                                            plugins: { legend: { display: false }, title: { display: true, text: 'MVRV（<1 低估 · >2.4 過熱）', font: { size: 11 } } },
                                            scales: {
                                                x: { ticks: { maxTicksLimit: 7, maxRotation: 0 }, grid: { color: gridColor } },
                                                y: { grid: { color: gridColor } },
                                            },
                                        },
                                    })}
                                />
                            </>
                        )}
                    </div>

                    {/* ── 2. ETF 淨流入 ── */}
                    <div className="rounded-2xl ring-soft p-4 md:p-5 space-y-3" style={{ background: 'var(--surface)' }}>
                        <div className="flex items-center justify-between gap-2">
                            <h3 className="font-bold text-white">美國現貨 ETF 淨流入</h3>
                            <Dropdown icon value={etfRange} onChange={setEtfRange} options={[{ key: '1mo', label: '近 1 個月' }, { key: '3mo', label: '近 3 個月' }, { key: '1y', label: '近 1 年' }, { key: 'all', label: '全部' }]} />
                        </div>
                        {pills(ETF_TYPES.map(t => ({ key: t.id, label: t.label })), etfType, setEtfType)}
                        {etfLoading ? spinner : etfError ? (
                            <p className="text-sm py-6 text-center" style={{ color: 'var(--down)' }}>讀取失敗：{etfError}</p>
                        ) : etfView.length === 0 ? (
                            <p className="text-sm py-6 text-center" style={{ color: 'var(--text-3)' }}>此幣種的美國現貨 ETF 尚無資料（SoSoValue 開放後會自動顯示）</p>
                        ) : (
                            <>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    {statCard('最新單日淨流入', fmtUsdCompact(etfLast?.flow), etfLast && etfLast.flow >= 0 ? 'var(--up)' : 'var(--down)')}
                                    {statCard('累計淨流入', fmtUsdCompact(etfLast?.cum))}
                                    {statCard('ETF 總淨資產', fmtUsdCompact(etfLast?.assets))}
                                    {statCard('資料日期', etfLast?.date || '—')}
                                </div>
                                <OnChainChart
                                    height={280}
                                    title={`${(ETF_TYPES.find(t => t.id === etfType) || {}).label || ''} 現貨 ETF 淨流入`}
                                    deps={[etfView]}
                                    build={() => ({
                                        data: {
                                            labels: etfView.map(d => d.date),
                                            datasets: [
                                                {
                                                    type: 'bar', label: '單日淨流入', data: etfView.map(d => d.flow),
                                                    backgroundColor: etfView.map(d => d.flow >= 0 ? 'rgba(0,214,143,0.7)' : 'rgba(255,91,110,0.7)'),
                                                    yAxisID: 'y',
                                                },
                                                {
                                                    type: 'line', label: '累計淨流入', data: etfView.map(d => d.cum),
                                                    borderColor: '#e2e8f0', borderWidth: 1.5, pointRadius: 0, tension: 0.2, yAxisID: 'y1',
                                                },
                                            ],
                                        },
                                        options: {
                                            responsive: true, maintainAspectRatio: false, animation: false,
                                            interaction: { mode: 'index', intersect: false },
                                            plugins: { legend: { display: true, labels: { boxWidth: 10, boxHeight: 2 } } },
                                            scales: {
                                                x: { ticks: { maxTicksLimit: 7, maxRotation: 0 }, grid: { color: gridColor } },
                                                y: { position: 'left', grid: { color: gridColor }, ticks: { callback: (v) => fmtUsdCompact(v) } },
                                                y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { callback: (v) => fmtUsdCompact(v) } },
                                            },
                                        },
                                    })}
                                />
                            </>
                        )}
                    </div>

                    {/* ── 3. 穩定幣 ── */}
                    <div className="rounded-2xl ring-soft p-4 md:p-5 space-y-3" style={{ background: 'var(--surface)' }}>
                        <div className="flex items-center justify-between flex-wrap gap-2">
                            <h3 className="font-bold text-white">穩定幣供給（購買力指標）</h3>
                            <Dropdown icon value={stableRange} onChange={setStableRange} options={[{ key: '3mo', label: '近 3 個月' }, { key: '1y', label: '近 1 年' }, { key: 'all', label: '全部' }]} />
                        </div>
                        {pills([{ key: 'all', label: '全部穩定幣' }, { key: 'usdt', label: 'USDT' }], stableView, setStableView)}
                        <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                            穩定幣市值持續增加 = 場外資金進場待命（利多）；持續縮水 = 資金撤出。綠柱 / 紅柱為每日淨增發 / 淨贖回。
                        </p>
                        {stableLoading ? spinner : stableViewData.length === 0 ? (
                            <p className="text-sm py-6 text-center" style={{ color: 'var(--text-3)' }}>讀取失敗，稍後再試</p>
                        ) : (
                            <>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    {statCard(stableView === 'usdt' ? 'USDT 市值' : '穩定幣總市值', fmtUsdCompact(stableStats?.mcap))}
                                    {statCard('7 天淨流入', fmtUsdCompact(stableStats?.d7), stableStats && stableStats.d7 >= 0 ? 'var(--up)' : 'var(--down)')}
                                    {statCard('30 天淨流入', fmtUsdCompact(stableStats?.d30), stableStats && stableStats.d30 >= 0 ? 'var(--up)' : 'var(--down)')}
                                    {statCard('資料日期', stableSeries.length ? stableSeries[stableSeries.length - 1].date : '—')}
                                </div>
                                <OnChainChart
                                    height={280}
                                    title={stableView === 'usdt' ? 'USDT 市值與淨流入' : '穩定幣總市值與淨流入'}
                                    deps={[stableViewData]}
                                    build={() => ({
                                        data: {
                                            labels: stableViewData.map(d => d.date),
                                            datasets: [
                                                {
                                                    type: 'bar', label: '淨流入', data: stableViewData.map(d => d.flow),
                                                    backgroundColor: stableViewData.map(d => d.flow >= 0 ? 'rgba(0,214,143,0.7)' : 'rgba(255,91,110,0.7)'),
                                                    yAxisID: 'y1',
                                                },
                                                {
                                                    type: 'line', label: '市值', data: stableViewData.map(d => d.mcap),
                                                    borderColor: '#e2e8f0', borderWidth: 1.5, pointRadius: 0, tension: 0.2, yAxisID: 'y',
                                                },
                                            ],
                                        },
                                        options: {
                                            responsive: true, maintainAspectRatio: false, animation: false,
                                            interaction: { mode: 'index', intersect: false },
                                            plugins: { legend: { display: true, labels: { boxWidth: 10, boxHeight: 2 } } },
                                            scales: {
                                                x: { ticks: { maxTicksLimit: 7, maxRotation: 0 }, grid: { color: gridColor } },
                                                y: { position: 'left', grid: { color: gridColor }, ticks: { callback: (v) => fmtUsdCompact(v) } },
                                                y1: { position: 'right', grid: { drawOnChartArea: false }, ticks: { callback: (v) => fmtUsdCompact(v) } },
                                            },
                                        },
                                    })}
                                />
                            </>
                        )}
                    </div>

                    <p className="text-[11px] px-1 pt-1" style={{ color: 'var(--text-3)' }}>MVRV 估值 · 美國現貨 ETF 資金流 · 穩定幣供給（資料源：CoinMetrics / SoSoValue / DefiLlama）</p>
                </div>
            );
        };

        const WatchlistDashboard = ({ watchlist, onUpdateWatchlist, isPremium, user }) => {
            const [assetsData, setAssetsData] = useState([]);
            const [loading, setLoading] = useState(true);
            const [selectedChartItem, setSelectedChartItem] = useState(null);
            const [sortKey, setSortKey] = useState('signal'); // signal | symbol | change | rsi
            const [sortDir, setSortDir] = useState('asc');
            const [showSignalHelp, setShowSignalHelp] = useState(false);

            if (!isPremium) {
                return (
                    <div className="py-10 max-w-[60ch]">
                        <div className="flex items-center gap-2.5">
                            <Lock size={20} style={{ color: 'var(--ink)' }} />
                            <span className="fs-chip" style={{ color: 'var(--ink)' }}>PRO</span>
                        </div>
                        <div className="mt-4">
                            <h2 className="fs-title mb-2">解鎖我的觀察清單</h2>
                            <p className="text-[15px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>
                                升級至 Premium 會員，即可建立跨市場觀察清單，並獲得 RSI 即時信號與 DCA 策略建議。
                            </p>
                        </div>
                        <button
                            onClick={() => {
                                const checkoutUrl = `${LEMON_CHECKOUT_URL}?checkout[email]=${encodeURIComponent(user.email)}`;
                                window.open(checkoutUrl, '_blank');
                            }}
                            className="fs-btn solid mt-6"
                        >
                            <Sparkles size={16} />
                            立即升級 Pro 版本
                        </button>
                    </div>
                );
            }

            // ─── Asset classification ───
            const classifyAsset = (symbol) => {
                if (symbol.toUpperCase().includes('.TW') || /^\d+$/.test(symbol)) return 'TW';
                const commonCryptos = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'TRX', 'LINK', 'MATIC', 'LTC', 'BITCOIN', 'ETHEREUM', 'SOLANA'];
                if (commonCryptos.includes(symbol.toUpperCase())) return 'CRYPTO';
                if (/^[A-Za-z]+$/.test(symbol) && symbol.length <= 5) return 'US';
                return 'UNKNOWN';
            };

            // Crypto slug normalization (BTC → bitcoin)
            const normalizeCrypto = (symbol) => {
                const map = { BTC: 'bitcoin', ETH: 'ethereum', SOL: 'solana', BNB: 'binancecoin', XRP: 'ripple', ADA: 'cardano', DOGE: 'dogecoin', AVAX: 'avalanche-2', DOT: 'polkadot', TRX: 'tron', LINK: 'chainlink', MATIC: 'matic-network', LTC: 'litecoin' };
                return map[symbol.toUpperCase()] || symbol.toLowerCase();
            };

            // ─── Fetch via price-proxy with 1y history (so we can compute RSI + range position) ───
            const fetchData = async () => {
                if (!watchlist || watchlist.length === 0) {
                    setAssetsData([]);
                    setLoading(false);
                    return;
                }
                setLoading(true);

                const proxy = `${SUPABASE_URL}/functions/v1/price-proxy`;
                const headers = { 'apikey': SUPABASE_ANON_KEY, 'Authorization': 'Bearer ' + SUPABASE_ANON_KEY };

                // Group by type for batch fetch
                const grouped = { US: [], TW: [], CRYPTO: [] };
                watchlist.forEach(sym => {
                    const t = classifyAsset(sym);
                    if (grouped[t]) grouped[t].push(sym);
                });

                const enrichStockSymbols = (syms, isTW) =>
                    syms.map(s => isTW && !s.includes('.') ? s + '.TW' : s);

                // Crypto: hit Binance directly (single source, matches the rest of the app and avoids CoinGecko rate limits)
                const fetchCryptoOne = async (slug) => {
                    const binSym = getBinanceSymbol(slug);
                    if (!binSym) return { symbol: slug, error: `不在 Binance: ${slug}` };
                    try {
                        const res = await fetch(`https://api.binance.com/api/v3/klines?symbol=${binSym}&interval=1d&limit=400`);
                        if (!res.ok) return { symbol: slug, error: `Binance ${res.status}` };
                        const klines = await res.json();
                        const history = klines.map(k => ({
                            date: new Date(k[0]).toISOString().slice(0, 10),
                            price: +k[4], open: +k[1], high: +k[2], low: +k[3],
                        }));
                        const last = klines[klines.length - 1];
                        const prev = klines[klines.length - 2];
                        return {
                            symbol: slug,
                            history,
                            price: last ? +last[4] : null,
                            previousClose: prev ? +prev[4] : null,
                        };
                    } catch (e) { return { symbol: slug, error: String(e) }; }
                };

                try {
                    const [usRes, twRes, cryptoData] = await Promise.all([
                        grouped.US.length ? fetch(`${proxy}?symbols=${enrichStockSymbols(grouped.US, false).join(',')}&type=stock&history=1y`, { headers }).then(r => r.json()) : { data: [] },
                        grouped.TW.length ? fetch(`${proxy}?symbols=${enrichStockSymbols(grouped.TW, true).join(',')}&type=stock&history=1y`, { headers }).then(r => r.json()) : { data: [] },
                        grouped.CRYPTO.length ? Promise.all(grouped.CRYPTO.map(s => fetchCryptoOne(normalizeCrypto(s)))) : [],
                    ]);
                    const cryptoRes = { data: cryptoData };

                    const buildItem = (rawSymbol, displaySymbol, type, apiResult) => {
                        if (!apiResult || apiResult.error || !apiResult.history || apiResult.history.length < 14) {
                            return { symbol: rawSymbol, type, error: apiResult?.error || 'no data' };
                        }
                        const closes = apiResult.history.map(p => p.price);
                        const high = Math.max(...closes);
                        const low = Math.min(...closes);
                        const price = apiResult.price ?? closes[closes.length - 1];
                        const prev = apiResult.previousClose ?? closes[closes.length - 2];
                        const changePercent = (price !== null && prev) ? ((price - prev) / prev) * 100 : null;
                        const rsi = computeRSI(closes, 14);
                        return {
                            symbol: rawSymbol,
                            displaySymbol,
                            type,
                            price,
                            changePercent,
                            high,
                            low,
                            rsi,
                            position: (high - low) > 0 ? (price - low) / (high - low) : 0.5,
                        };
                    };

                    const items = [];
                    grouped.US.forEach((s, i) => items.push(buildItem(s, s, 'US', (usRes.data || [])[i])));
                    grouped.TW.forEach((s, i) => {
                        const enriched = enrichStockSymbols([s], true)[0];
                        const apiItem = (twRes.data || []).find(d => d.symbol === enriched);
                        items.push(buildItem(s, enriched, 'TW', apiItem));
                    });
                    grouped.CRYPTO.forEach((s, i) => {
                        const slug = normalizeCrypto(s);
                        const apiItem = (cryptoRes.data || []).find(d => d.symbol === slug);
                        items.push(buildItem(s, slug, 'CRYPTO', apiItem));
                    });

                    setAssetsData(items.filter(it => !it.error));
                } catch (e) {
                    console.error('Watchlist fetch failed:', e);
                } finally {
                    setLoading(false);
                }
            };

            useEffect(() => { fetchData(); }, [JSON.stringify(watchlist)]);

            const handleDelete = (symbol) => {
                if (!confirm(`移除「${symbol}」？`)) return;
                onUpdateWatchlist(watchlist.filter(s => s !== symbol));
            };

            // Sort logic
            const sortedAssets = useMemo(() => {
                if (!assetsData.length) return [];
                const arr = [...assetsData];
                arr.sort((a, b) => {
                    let av, bv;
                    if (sortKey === 'symbol') { av = a.symbol; bv = b.symbol; }
                    else if (sortKey === 'change') { av = a.changePercent ?? 0; bv = b.changePercent ?? 0; }
                    else if (sortKey === 'rsi') { av = a.rsi ?? 50; bv = b.rsi ?? 50; }
                    else if (sortKey === 'signal') {
                        // Lower RSI = stronger buy signal → sort ascending (best buy first)
                        av = a.rsi ?? 50; bv = b.rsi ?? 50;
                    } else { av = a.symbol; bv = b.symbol; }
                    if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
                    return sortDir === 'asc' ? av - bv : bv - av;
                });
                return arr;
            }, [assetsData, sortKey, sortDir]);

            const headerBtn = (key, label) => {
                const active = sortKey === key;
                return (
                    <button
                        onClick={() => {
                            if (active) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
                            else { setSortKey(key); setSortDir(key === 'signal' || key === 'rsi' ? 'asc' : 'desc'); }
                        }}
                        className="inline-flex items-center gap-1 hover:text-white transition-colors"
                        style={{ color: active ? 'var(--brand-1)' : 'var(--text-3)' }}
                    >
                        {label}
                        {active && (sortDir === 'asc' ? <ArrowUp size={10} /> : <ArrowDown size={10} />)}
                    </button>
                );
            };

            const formatPrice = (p) => p === null || p === undefined ? '—' : p.toLocaleString(undefined, { maximumFractionDigits: 2 });

            // ─── Render ───
            if (loading) return <div className="flex justify-center py-20"><RefreshCw className="animate-spin text-slate-500" size={32} /></div>;

            if (!watchlist || watchlist.length === 0) return (
                <div className="rounded-2xl ring-soft p-10 text-center relative overflow-hidden" style={{ background: 'var(--surface)' }}>
                    <div className="absolute inset-0 dotgrid opacity-40 pointer-events-none"></div>
                    <div className="relative">
                        <Sparkles size={48} className="mx-auto mb-4 opacity-50" style={{ color: 'var(--text-3)' }} />
                        <h3 className="text-xl font-bold text-white">觀察清單是空的</h3>
                        <p className="mt-2 text-sm" style={{ color: 'var(--text-2)' }}>到加密 / 美股 / 台股 dashboard 點標題旁的星星圖示加入</p>
                    </div>
                </div>
            );

            return (
                <>
                    {/* Stats header */}
                    <div className="rounded-2xl ring-soft p-4 flex items-center justify-between flex-wrap gap-3 mb-4" style={{ background: 'var(--surface)' }}>
                        <div className="flex items-center gap-6 text-xs flex-wrap">
                            <div>
                                <p className="label">追蹤中</p>
                                <p className="text-xl font-extrabold text-white num mt-0.5">{assetsData.length}</p>
                            </div>
                            <div>
                                <p className="label">買進信號</p>
                                <p className="text-xl font-extrabold num mt-0.5" style={{ color: 'var(--up)' }}>
                                    {assetsData.filter(a => a.rsi !== null && a.rsi <= 30).length}
                                </p>
                            </div>
                            <div>
                                <p className="label">賣出警示</p>
                                <p className="text-xl font-extrabold num mt-0.5" style={{ color: 'var(--down)' }}>
                                    {assetsData.filter(a => a.rsi !== null && a.rsi >= 65).length}
                                </p>
                            </div>
                        </div>
                        <button onClick={fetchData} className="fs-btn icon" title="重新整理" aria-label="重新整理">
                            <RefreshCw size={14} />
                        </button>
                    </div>

                    {/* Table */}
                    <div className="rounded-2xl ring-soft overflow-hidden" style={{ background: 'var(--surface)' }}>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr style={{ borderBottom: '1px solid var(--line)' }}>
                                        <th className="text-left p-2 md:p-3 label">{headerBtn('symbol', '標的')}</th>
                                        <th className="text-left p-2 md:p-3 label">
                                            <span className="inline-flex items-center gap-1.5">
                                                {headerBtn('signal', '信號')}
                                                <button
                                                    onClick={() => setShowSignalHelp(true)}
                                                    className="w-4 h-4 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors"
                                                    style={{ color: 'var(--brand-1)', border: '1px solid var(--brand-1)' }}
                                                    title="信號邏輯說明"
                                                >
                                                    <span className="text-[9px] font-bold leading-none">?</span>
                                                </button>
                                            </span>
                                        </th>
                                        <th className="text-right p-2 md:p-3 label">現價</th>
                                        <th className="text-right p-2 md:p-3 label">{headerBtn('change', '漲跌%')}</th>
                                        <th className="text-right p-2 md:p-3 label hidden lg:table-cell">1Y 區間</th>
                                        <th className="text-right p-2 md:p-3 label hidden sm:table-cell">{headerBtn('rsi', 'RSI')}</th>
                                        <th className="text-right p-2 md:p-3 label"></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {sortedAssets.map((item) => {
                                        const signal = getCompositeSignal(item.rsi, item.position);
                                        const changeUp = (item.changePercent ?? 0) >= 0;
                                        const positionPct = Math.max(0, Math.min(100, (item.position ?? 0.5) * 100));
                                        return (
                                            <tr key={item.symbol} className="hover:bg-white/[0.02] transition-colors" style={{ borderBottom: '1px solid var(--line)' }}>
                                                <td className="p-2 md:p-3">
                                                    <button
                                                        onClick={() => setSelectedChartItem({ symbol: item.displaySymbol || item.symbol, type: item.type })}
                                                        className="font-bold text-white hover:text-grad text-left"
                                                    >
                                                        {item.symbol}
                                                    </button>
                                                </td>
                                                <td className="p-2 md:p-3">
                                                    <div className="fs-chip whitespace-nowrap" style={{ color: signal.color }}>
                                                        <span>{signal.emoji}</span>
                                                        <span>{signal.label}</span>
                                                    </div>
                                                    <div className="text-[10px] mono mt-0.5 hidden md:block" style={{ color: 'var(--text-3)' }}>{signal.advice}</div>
                                                </td>
                                                <td className="p-2 md:p-3 text-right mono">${formatPrice(item.price)}</td>
                                                <td className="p-2 md:p-3 text-right mono font-semibold" style={{ color: changeUp ? 'var(--up)' : 'var(--down)' }}>
                                                    {changeUp ? '+' : ''}{item.changePercent?.toFixed(2) ?? '—'}%
                                                </td>
                                                <td className="p-2 md:p-3 text-right hidden lg:table-cell">
                                                    <div className="flex items-center justify-end gap-2">
                                                        <span className="text-[10px] mono" style={{ color: 'var(--text-3)' }}>{positionPct.toFixed(0)}%</span>
                                                        <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--wash-2)' }}>
                                                            <div className="h-full rounded-full" style={{
                                                                width: positionPct + '%',
                                                                background: positionPct < 30 ? 'var(--up)' : positionPct > 70 ? 'var(--down)' : 'var(--amber)'
                                                            }}></div>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td className="p-2 md:p-3 text-right mono font-bold hidden sm:table-cell" style={{ color: signal.color }}>
                                                    {item.rsi !== null ? item.rsi.toFixed(1) : '—'}
                                                </td>
                                                <td className="p-2 md:p-3 text-right">
                                                    <button
                                                        onClick={() => handleDelete(item.symbol)}
                                                        className="p-1 rounded hover:bg-red-500/20 transition-colors"
                                                        style={{ color: 'var(--text-3)' }}
                                                        title="移除"
                                                    >
                                                        ✕
                                                    </button>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {selectedChartItem && (
                        <ChartModal
                            symbol={selectedChartItem.symbol}
                            type={selectedChartItem.type}
                            onClose={() => setSelectedChartItem(null)}
                        />
                    )}

                    {showSignalHelp && (
                        <div className="fixed inset-0 flex items-center justify-center z-[100] p-4" style={{ background: 'rgba(7,8,12,0.78)', backdropFilter: 'blur(8px)' }}>
                            <div className="glass-strong p-6 rounded-3xl shadow-2xl max-w-lg w-full relative max-h-[90vh] overflow-y-auto custom-scrollbar">
                                <button
                                    onClick={() => setShowSignalHelp(false)}
                                    className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors"
                                    style={{ color: 'var(--text-3)' }}
                                >
                                    ✕
                                </button>

                                <div className="mb-5">
                                    <p className="label">SIGNAL LOGIC</p>
                                    <h3 className="text-xl font-extrabold text-white mt-1 tracking-tight">信號邏輯說明</h3>
                                    <p className="text-xs mt-1.5" style={{ color: 'var(--text-2)' }}>
                                        信號由 <strong className="text-white">RSI(14)</strong> + <strong className="text-white">1Y 區間位置</strong> 複合判斷
                                    </p>
                                </div>

                                <div className="space-y-3 text-sm">
                                    <div className="p-3 rounded-xl ring-soft" style={{ background: 'rgba(60,224,168,0.08)', border: '1px solid rgba(60,224,168,0.25)' }}>
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-base">🔥</span>
                                            <span className="font-bold" style={{ color: 'var(--up)' }}>強力加碼</span>
                                        </div>
                                        <p className="text-xs mono" style={{ color: 'var(--text-2)' }}>RSI ≤ 25 <strong>或</strong> (RSI ≤ 30 <strong>且</strong> 接近 1Y 低 ≤ 10%)</p>
                                    </div>

                                    <div className="p-3 rounded-xl ring-soft" style={{ background: 'rgba(132,215,106,0.05)', border: '1px solid rgba(132,215,106,0.2)' }}>
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-base">🟢</span>
                                            <span className="font-bold" style={{ color: 'var(--z3)' }}>加碼</span>
                                        </div>
                                        <p className="text-xs mono" style={{ color: 'var(--text-2)' }}>RSI ≤ 30 <strong>或</strong> (RSI ≤ 40 <strong>且</strong> 接近 1Y 低 ≤ 20%)</p>
                                    </div>

                                    <div className="p-3 rounded-xl ring-soft" style={{ background: 'var(--wash)' }}>
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-base">⚪</span>
                                            <span className="font-bold" style={{ color: 'var(--text)' }}>持有 / 觀望</span>
                                        </div>
                                        <p className="text-xs mono" style={{ color: 'var(--text-2)' }}>RSI 介於 30 ~ 65 之間，無明顯訊號</p>
                                    </div>

                                    <div className="p-3 rounded-xl ring-soft" style={{ background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.25)' }}>
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-base">🟡</span>
                                            <span className="font-bold" style={{ color: 'var(--amber)' }}>警示</span>
                                        </div>
                                        <p className="text-xs mono" style={{ color: 'var(--text-2)' }}>RSI ≥ 65 <strong>或</strong> 接近 1Y 高 ≥ 85%</p>
                                    </div>

                                    <div className="p-3 rounded-xl ring-soft" style={{ background: 'rgba(255,125,140,0.08)', border: '1px solid rgba(255,125,140,0.25)' }}>
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-base">🔴</span>
                                            <span className="font-bold" style={{ color: 'var(--down)' }}>賣出</span>
                                        </div>
                                        <p className="text-xs mono" style={{ color: 'var(--text-2)' }}>RSI ≥ 70 <strong>或</strong> (RSI ≥ 60 <strong>且</strong> 接近 1Y 高 ≥ 90%)</p>
                                    </div>

                                    <div className="p-3 rounded-xl ring-soft" style={{ background: 'rgba(255,91,110,0.10)', border: '1px solid rgba(255,91,110,0.3)' }}>
                                        <div className="flex items-center gap-2 mb-1.5">
                                            <span className="text-base">🚨</span>
                                            <span className="font-bold" style={{ color: 'var(--down)' }}>強力賣出</span>
                                        </div>
                                        <p className="text-xs mono" style={{ color: 'var(--text-2)' }}>RSI ≥ 75 <strong>或</strong> (RSI ≥ 65 <strong>且</strong> 接近 1Y 高 ≥ 95%)</p>
                                    </div>
                                </div>

                                <div className="mt-5 p-3 rounded-xl text-xs leading-relaxed" style={{ background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.2)', color: 'var(--text-2)' }}>
                                    <p className="mb-1.5"><strong className="text-white">RSI 是什麼？</strong></p>
                                    <p className="mb-2">相對強弱指標 (Relative Strength Index)。看過去 14 天「上漲幅度 vs 下跌幅度」的比例，0~100 之間。標準上 &lt;30 是超賣（買進）、&gt;70 是超買（賣出）。</p>
                                    <p className="mb-1.5"><strong className="text-white">1Y 區間位置</strong>：</p>
                                    <p>目前股價在過去 1 年高低點之間的位置。0% = 在低點、100% = 在高點。即使 RSI 還沒衝過 70，如果價格已經接近 1 年高點，也算過熱訊號。</p>
                                </div>

                                <button
                                    onClick={() => setShowSignalHelp(false)}
                                    className="w-full mt-5 py-2.5 pill-grad rounded-xl text-sm font-bold transition-all glow-brand"
                                >
                                    知道了
                                </button>
                            </div>
                        </div>
                    )}
                </>
            );
        };

        // ═════════════════════════════════════════════
        // PortfolioDashboard — track real holdings & P/L
        // ═════════════════════════════════════════════
        const ASSET_TYPES = {
            tw_stock: { label: '台股', currency: 'TWD', symbol: 'NT$' },
            us_stock: { label: '美股', currency: 'USD', symbol: '$' },
            crypto:   { label: '加密貨幣', currency: 'USD', symbol: '$' },
            cash:     { label: '現金', currency: 'dynamic', symbol: 'dynamic' },  // currency from holding.symbol
        };

        // Helper: get currency code for a holding (cash uses its own symbol as currency)
        const getHoldingCurrency = (h) => h.asset_type === 'cash' ? h.symbol : ASSET_TYPES[h.asset_type].currency;
        const getHoldingSymbol = (h) => {
            const ccy = getHoldingCurrency(h);
            return ccy === 'TWD' ? 'NT$' : '$';
        };
        const currencyToSymbol = (ccy) => ccy === 'TWD' ? 'NT$' : '$';

        // Dev-only sample portfolio for reviewing the layout without an account.
        // import.meta.env.DEV is false in `vite build`, so none of this reaches production.
        const PORTFOLIO_FIXTURE = import.meta.env.DEV && new URLSearchParams(window.location.search).get('portfolioFixture') === '1';
        const FIXTURE_USER = { id: 'fixture-user', email: 'fixture@localhost' };
        const FIXTURE_TRANSACTIONS = [
            { id: 'f1', symbol: 'BITCOIN', asset_type: 'crypto', type: 'buy', shares: 0.01, price: 62000, fee: 0, date: '2025-11-03', note: '定期定額' },
            { id: 'f2', symbol: 'BITCOIN', asset_type: 'crypto', type: 'buy', shares: 0.012, price: 58000, fee: 0, date: '2026-02-12', note: '恐懼加碼' },
            { id: 'f3', symbol: 'ETHEREUM', asset_type: 'crypto', type: 'buy', shares: 0.4, price: 2350, fee: 0, date: '2026-03-01', note: '' },
            { id: 'f4', symbol: 'ETHEREUM', asset_type: 'crypto', type: 'sell', shares: 0.1, price: 2600, fee: 0, date: '2026-08-20', note: '' },
            { id: 'f5', symbol: 'VOO', asset_type: 'us_stock', type: 'buy', shares: 3, price: 560, fee: 1, date: '2025-12-15', note: '' },
            { id: 'f6', symbol: 'VOO', asset_type: 'us_stock', type: 'buy', shares: 2, price: 590, fee: 1, date: '2026-04-02', note: '' },
            { id: 'f7', symbol: 'VOO', asset_type: 'us_stock', type: 'dividend', shares: 0, price: 18.4, fee: 0, date: '2026-06-28', note: '' },
            { id: 'f8', symbol: 'USD', asset_type: 'cash', type: 'buy', shares: 1200, price: 1, fee: 0, date: '2026-05-10', note: '備用金' },
        ];

        // Compute current holdings from a flat list of transactions
        const computeHoldings = (transactions) => {
            const map = new Map();
            // Sort by date ascending so weighted-average cost is computed correctly
            const sorted = [...transactions].sort((a, b) => new Date(a.date) - new Date(b.date));
            for (const tx of sorted) {
                const key = `${tx.symbol}::${tx.asset_type}`;
                if (!map.has(key)) {
                    map.set(key, {
                        symbol: tx.symbol,
                        asset_type: tx.asset_type,
                        totalShares: 0,
                        totalCost: 0,
                        sharesBought: 0,
                        sharesSold: 0,
                        dividends: 0,
                        totalFees: 0,
                        firstBuyDate: null,
                        lastTxDate: tx.date,
                    });
                }
                const h = map.get(key);
                h.lastTxDate = tx.date;
                const shares = Number(tx.shares) || 0;
                const price = Number(tx.price) || 0;
                const fee = Number(tx.fee) || 0;
                if (tx.type === 'buy') {
                    h.totalShares += shares;
                    h.totalCost += shares * price + fee;
                    h.sharesBought += shares;
                    h.totalFees += fee;
                    if (!h.firstBuyDate || tx.date < h.firstBuyDate) h.firstBuyDate = tx.date;
                } else if (tx.type === 'sell') {
                    const avgCost = h.sharesBought > 0 ? h.totalCost / h.totalShares : 0;
                    h.totalShares -= shares;
                    h.totalCost -= avgCost * shares;
                    h.sharesSold += shares;
                    h.totalFees += fee;
                } else if (tx.type === 'dividend') {
                    // For dividends, "price" field stores the total dividend amount in local currency
                    h.dividends += price;
                }
            }
            return Array.from(map.values()).filter(h => h.totalShares > 0.0001);
        };

        // ─────────────────────────────────────────────
        // AddTransactionModal
        // ─────────────────────────────────────────────
        const AddTransactionModal = ({ supabase, user, existing, onClose, onSaved }) => {
            const [symbol, setSymbol] = useState(existing?.symbol || '');
            const [assetType, setAssetType] = useState(existing?.asset_type || 'tw_stock');
            const [txType, setTxType] = useState(existing?.type || 'buy');
            const [date, setDate] = useState(existing?.date || new Date().toISOString().slice(0, 10));
            const [shares, setShares] = useState(existing?.shares || '');
            const [price, setPrice] = useState(existing?.price || '');
            const [fee, setFee] = useState(existing?.fee || '');
            const [note, setNote] = useState(existing?.note || '');
            const [saving, setSaving] = useState(false);
            const [error, setError] = useState(null);

            const inputStyle = { background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text)' };

            const handleSubmit = async (e) => {
                e.preventDefault();
                if (assetType !== 'cash' && !symbol.trim()) return setError('請輸入代號');
                if (txType !== 'dividend' && assetType !== 'cash' && !shares) return setError('請輸入股數');
                if (!price) return setError(
                    assetType === 'cash' ? '請輸入金額' :
                    txType === 'dividend' ? '請輸入配息金額' :
                    '請輸入價格'
                );
                setSaving(true);
                setError(null);
                // For cash: store amount in shares, price=1 (so cost == amount, MV == amount)
                const isCash = assetType === 'cash';
                const payload = {
                    user_id: user.id,
                    symbol: isCash ? symbol : symbol.trim().toUpperCase(),
                    asset_type: assetType,
                    type: txType,
                    date,
                    shares: isCash ? Number(price) : (txType === 'dividend' ? 0 : Number(shares)),
                    price: isCash ? 1 : Number(price),
                    fee: isCash ? 0 : (Number(fee) || 0),
                    note: note.trim() || null,
                };
                let result;
                if (existing?.id) {
                    result = await supabase.from('transactions').update(payload).eq('id', existing.id).select().single();
                } else {
                    result = await supabase.from('transactions').insert(payload).select().single();
                }
                setSaving(false);
                if (result.error) {
                    setError(result.error.message);
                    return;
                }
                onSaved(result.data);
                onClose();
            };

            return (
                <div className="fixed inset-0 flex items-center justify-center z-[100] p-4" style={{ background: 'rgba(7,8,12,0.78)', backdropFilter: 'blur(8px)' }}>
                    <div className="glass-strong p-6 rounded-3xl shadow-2xl max-w-md w-full relative max-h-[90vh] overflow-y-auto custom-scrollbar">
                        <button
                            onClick={onClose}
                            className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors"
                            style={{ color: 'var(--text-3)' }}
                        >
                            ✕
                        </button>

                        <div className="mb-5">
                            <p className="label">{existing ? 'EDIT TRANSACTION' : 'NEW TRANSACTION'}</p>
                            <h3 className="text-xl font-extrabold text-white mt-1">
                                {existing ? '編輯交易' : '新增交易'}
                            </h3>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-3">
                            {/* Asset type + Transaction type */}
                            <div className="grid grid-cols-2 gap-2">
                                <div>
                                    <label className="label block mb-1.5">市場</label>
                                    <select value={assetType} onChange={e => {
                                        const newType = e.target.value;
                                        setAssetType(newType);
                                        // Reset symbol when switching to/from cash
                                        if (newType === 'cash' && !['TWD','USD'].includes(symbol.toUpperCase())) setSymbol('TWD');
                                        // Cash doesn't support dividend
                                        if (newType === 'cash' && txType === 'dividend') setTxType('buy');
                                    }} className="w-full rounded-xl px-3 py-2.5 text-sm outline-none" style={inputStyle}>
                                        <option value="tw_stock">台股</option>
                                        <option value="us_stock">美股 / ETF</option>
                                        <option value="crypto">加密貨幣</option>
                                        <option value="cash">現金</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="label block mb-1.5">類型</label>
                                    <select value={txType} onChange={e => setTxType(e.target.value)} className="w-full rounded-xl px-3 py-2.5 text-sm outline-none" style={inputStyle}>
                                        <option value="buy">{assetType === 'cash' ? '存入' : '買入'}</option>
                                        <option value="sell">{assetType === 'cash' ? '提領' : '賣出'}</option>
                                        {assetType !== 'cash' && <option value="dividend">配息</option>}
                                    </select>
                                </div>
                            </div>

                            {/* Symbol */}
                            <div>
                                <label className="label block mb-1.5">{assetType === 'cash' ? '幣別' : '代號'}</label>
                                {assetType === 'cash' ? (
                                    <select
                                        value={symbol}
                                        onChange={e => setSymbol(e.target.value)}
                                        className="w-full rounded-xl px-3 py-2.5 text-sm outline-none"
                                        style={inputStyle}
                                    >
                                        <option value="TWD">新台幣 (TWD)</option>
                                        <option value="USD">美金 (USD)</option>
                                    </select>
                                ) : (
                                    <input
                                        type="text"
                                        required
                                        value={symbol}
                                        onChange={e => setSymbol(e.target.value)}
                                        placeholder={assetType === 'tw_stock' ? '例如 0050.TW' : assetType === 'us_stock' ? '例如 AAPL' : '例如 bitcoin'}
                                        className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
                                        style={inputStyle}
                                    />
                                )}
                                <p className="text-[10px] mono mt-1" style={{ color: 'var(--text-3)' }}>
                                    {assetType === 'tw_stock' && '台股要加 .TW 後綴'}
                                    {assetType === 'us_stock' && '輸入 ticker 大寫'}
                                    {assetType === 'crypto' && '使用 CoinMarketCap slug (例如 bitcoin、ethereum)'}
                                    {assetType === 'cash' && '記錄存入/提領金額'}
                                </p>
                            </div>

                            {/* Date */}
                            <div>
                                <label className="label block mb-1.5">日期</label>
                                <input type="date" required value={date} onChange={e => setDate(e.target.value)}
                                    className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none" style={inputStyle} />
                            </div>

                            {/* Shares (hide for dividend AND cash; cash uses 'amount' field below) */}
                            {txType !== 'dividend' && assetType !== 'cash' && (
                                <div>
                                    <label className="label block mb-1.5">股數 / 數量</label>
                                    <input type="number" step="any" required value={shares} onChange={e => setShares(e.target.value)} placeholder="0"
                                        className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none" style={inputStyle} />
                                </div>
                            )}

                            <div>
                                <label className="label block mb-1.5">
                                    {assetType === 'cash'
                                        ? `${txType === 'sell' ? '提領金額' : '存入金額'} (${symbol === 'TWD' ? 'NT$' : '$'})`
                                        : txType === 'dividend' ? `配息總額 (${ASSET_TYPES[assetType].symbol})` : `成交價 (${ASSET_TYPES[assetType].symbol})`}
                                </label>
                                <input type="number" step="any" required value={price} onChange={e => setPrice(e.target.value)} placeholder="0"
                                    className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none" style={inputStyle} />
                            </div>

                            {/* Fee */}
                            {txType !== 'dividend' && assetType !== 'cash' && (
                                <div>
                                    <label className="label block mb-1.5">手續費 (選填)</label>
                                    <input type="number" step="any" value={fee} onChange={e => setFee(e.target.value)} placeholder="0"
                                        className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none" style={inputStyle} />
                                </div>
                            )}

                            {/* Note */}
                            <div>
                                <label className="label block mb-1.5">備註 (選填)</label>
                                <input type="text" value={note} onChange={e => setNote(e.target.value)}
                                    className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none" style={inputStyle} />
                            </div>

                            {error && (
                                <div className="text-xs text-center px-3 py-2 rounded-lg" style={{ background: 'rgba(255,91,110,0.08)', color: 'var(--down)', border: '1px solid rgba(255,91,110,0.2)' }}>
                                    {error}
                                </div>
                            )}

                            <button
                                type="submit"
                                disabled={saving}
                                className="w-full py-3 pill-grad text-white rounded-xl text-sm font-bold transition-all glow-brand disabled:opacity-50"
                            >
                                {saving ? '儲存中...' : (existing ? '更新交易' : '新增交易')}
                            </button>
                        </form>
                    </div>
                </div>
            );
        };

        // ─────────────────────────────────────────────
        // PortfolioValueChart — TRUE historical market value
        // X axis = daily timeline; Y axis = sum(holdings[d] × price[symbol][d])
        // Also shows cost basis line as reference
        // ─────────────────────────────────────────────
        const PortfolioValueChart = ({ transactions, holdings, prices, currency = 'USD' }) => {
            const canvasRef = useRef(null);
            const chartRef = useRef(null);
            const [timeRange, setTimeRange] = useState('1y');  // 1mo|3mo|6mo|1y|max
            const [historicalPrices, setHistoricalPrices] = useState({}); // {symbol::asset_type: [{date, price}]}
            const [historyLoading, setHistoryLoading] = useState(false);
            const themeV = useThemeVersion();

            const ccyMatch = currency === 'TWD' ? ['tw_stock'] : ['us_stock', 'crypto'];

            // Get unique symbols for the relevant currency
            const symbolKeys = useMemo(() => {
                const set = new Set();
                transactions.forEach(t => {
                    if (ccyMatch.includes(t.asset_type)) {
                        set.add(`${t.symbol}::${t.asset_type}`);
                    }
                });
                return Array.from(set);
            }, [transactions, currency]);

            // Fetch historical prices for all symbols when timeRange or symbols change
            useEffect(() => {
                if (symbolKeys.length === 0) return;
                let cancelled = false;
                (async () => {
                    setHistoryLoading(true);
                    try {
                        const proxy = `${SUPABASE_URL}/functions/v1/price-proxy`;
                        const headers = { 'apikey': SUPABASE_ANON_KEY, 'Authorization': 'Bearer ' + SUPABASE_ANON_KEY };
                        const stockSyms = symbolKeys.filter(k => k.endsWith('::tw_stock') || k.endsWith('::us_stock')).map(k => k.split('::')[0]);
                        const cryptoSyms = symbolKeys.filter(k => k.endsWith('::crypto')).map(k => k.split('::')[0]);

                        const results = await Promise.all([
                            stockSyms.length ? fetch(`${proxy}?symbols=${stockSyms.join(',')}&type=stock&history=${timeRange}`, { headers }).then(r => r.json()) : { data: [] },
                            cryptoSyms.length ? fetch(`${proxy}?symbols=${cryptoSyms.join(',')}&type=crypto&history=${timeRange}`, { headers }).then(r => r.json()) : { data: [] },
                        ]);
                        if (cancelled) return;

                        const map = {};
                        (results[0].data || []).forEach(r => {
                            if (r.history && r.history.length) {
                                const key = symbolKeys.find(k => k.split('::')[0] === r.symbol);
                                if (key) map[key] = r.history;
                            }
                        });
                        (results[1].data || []).forEach(r => {
                            if (r.history && r.history.length) {
                                const key = symbolKeys.find(k => k.split('::')[0].toLowerCase() === r.symbol.toLowerCase());
                                if (key) map[key] = r.history;
                            }
                        });
                        setHistoricalPrices(map);
                    } catch (e) {
                        console.error('Historical price fetch failed', e);
                    } finally {
                        setHistoryLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, [symbolKeys.join(','), timeRange]);

            // Build the daily timeline
            const { dates, valueSeries, costSeries, currentMarketValue, totalCost } = useMemo(() => {
                const filteredTxs = transactions.filter(t => ccyMatch.includes(t.asset_type));
                if (filteredTxs.length === 0 || Object.keys(historicalPrices).length === 0) {
                    return { dates: [], valueSeries: [], costSeries: [], currentMarketValue: 0, totalCost: 0 };
                }

                // Build a Set of all dates from historical prices
                const dateSet = new Set();
                Object.values(historicalPrices).forEach(arr => arr.forEach(p => dateSet.add(p.date)));
                const allDates = Array.from(dateSet).sort();

                // Index: symbol_key -> { date: price }
                const priceByDate = {};
                for (const [key, arr] of Object.entries(historicalPrices)) {
                    priceByDate[key] = {};
                    for (const p of arr) priceByDate[key][p.date] = p.price;
                }

                // Sort transactions ascending for cumulative computation
                const txsAsc = [...filteredTxs].sort((a, b) => new Date(a.date) - new Date(b.date));

                const valueSeries = [];
                const costSeries = [];
                let netCost = 0;
                let txIndex = 0;
                const holdingsByKey = {}; // current shares per key based on txs up to date

                for (const date of allDates) {
                    // Apply all transactions up to (and including) this date
                    while (txIndex < txsAsc.length && txsAsc[txIndex].date <= date) {
                        const tx = txsAsc[txIndex];
                        const key = `${tx.symbol}::${tx.asset_type}`;
                        const shares = Number(tx.shares) || 0;
                        const price = Number(tx.price) || 0;
                        const fee = Number(tx.fee) || 0;
                        if (!holdingsByKey[key]) holdingsByKey[key] = 0;

                        if (tx.type === 'buy') {
                            holdingsByKey[key] += shares;
                            netCost += shares * price + fee;
                        } else if (tx.type === 'sell') {
                            holdingsByKey[key] -= shares;
                            netCost -= shares * price;
                        } else if (tx.type === 'dividend') {
                            netCost -= price;
                        }
                        txIndex++;
                    }

                    // Compute portfolio value on this date
                    let value = 0;
                    for (const [key, shares] of Object.entries(holdingsByKey)) {
                        if (shares <= 0.000001) continue;
                        const px = priceByDate[key] && priceByDate[key][date];
                        // If no price for this exact date, use last known price
                        if (px !== undefined) {
                            value += shares * px;
                        } else {
                            // Find nearest earlier date
                            const symbolDates = Object.keys(priceByDate[key] || {}).sort();
                            const candidates = symbolDates.filter(d => d <= date);
                            if (candidates.length > 0) {
                                value += shares * priceByDate[key][candidates[candidates.length - 1]];
                            }
                        }
                    }
                    valueSeries.push(value);
                    costSeries.push(netCost);
                }

                // Current market value from live prices
                let currentMarketValue = 0;
                holdings.filter(h => ccyMatch.includes(h.asset_type)).forEach(h => {
                    const p = prices[`${h.symbol}::${h.asset_type}`];
                    if (p && p.price) currentMarketValue += h.totalShares * p.price;
                });

                return { dates: allDates, valueSeries, costSeries, currentMarketValue, totalCost: netCost };
            }, [transactions, historicalPrices, holdings, prices, currency]);

            useEffect(() => {
                if (!canvasRef.current || dates.length === 0) return;
                if (chartRef.current) { chartRef.current.destroy(); }

                const ctx = canvasRef.current.getContext('2d');
                chartRef.current = new window.Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [
                            {
                                label: '總市值',
                                data: valueSeries,
                                borderColor: CHART.line,
                                backgroundColor: makePriceGradient,
                                fill: true,
                                pointRadius: 0,
                                pointHoverRadius: 4,
                                borderWidth: 1.8,
                                tension: 0.2,
                            },
                            {
                                label: '累計投入成本',
                                data: costSeries,
                                borderColor: CHART.tick,
                                borderDash: [5, 5],
                                fill: false,
                                pointRadius: 0,
                                pointHoverRadius: 3,
                                borderWidth: 1.5,
                                tension: 0.3,
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    title: (items) => items[0].label,
                                    label: (item) => {
                                        const sym = currency === 'TWD' ? 'NT$' : '$';
                                        return `${item.dataset.label}: ${sym}${item.parsed.y.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: figureX(),
                            y: {
                                grid: { color: CHART.grid },
                                ticks: {
                                    color: CHART.tick,
                                    callback: (val) => {
                                        const sym = currency === 'TWD' ? 'NT$' : '$';
                                        return `${sym}${val.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
                                    },
                                }
                            }
                        }
                    },
                    plugins: [dotGridPlugin, lineGlowPlugin, crosshairPlugin]
                });

                return () => {
                    if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
                };
            }, [dates, valueSeries, costSeries, currency, themeV]);

            if (symbolKeys.length === 0) return null;

            const pl = currentMarketValue - totalCost;
            const plPct = totalCost > 0 ? (pl / totalCost) * 100 : 0;
            const isUp = pl >= 0;
            const sym = currency === 'TWD' ? 'NT$' : '$';

            const rangeBtn = (key, label) => (
                <button
                    key={key}
                    onClick={() => setTimeRange(key)}
                    className="px-2 py-0.5 num transition-colors"
                    style={timeRange === key
                        ? { border: '1.5px solid var(--ink)', color: 'var(--ink)', fontWeight: 700 }
                        : { border: '1.5px solid transparent', color: 'var(--ink-2)' }}
                >
                    {label}
                </button>
            );

            return (
                <section className="fs-section mt-6">
                    <div className="flex items-end justify-between gap-4 flex-wrap mb-4">
                        <div>
                            <p className="fs-lbl">TOTAL · {currency}</p>
                            <p className="text-[34px] md:text-[40px] font-black leading-none mt-1 num tracking-tight" style={{ color: 'var(--ink)' }}>
                                {sym}{currentMarketValue.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </p>
                            <p className="text-[13px] num mt-2 font-semibold" style={{ color: isUp ? 'var(--up)' : 'var(--down)' }}>
                                {isUp ? '▲' : '▼'} {sym}{Math.abs(pl).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                                {' '}({isUp ? '+' : ''}{plPct.toFixed(2)}%)
                                <span className="ml-2 font-normal" style={{ color: 'var(--ink-3)' }}>vs 成本 {sym}{totalCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                            </p>
                        </div>
                        <div className="flex items-center gap-3">
                            {historyLoading && (
                                <span className="flex items-center gap-1.5 text-[12px]" style={{ color: 'var(--ink-3)' }}>
                                    <RefreshCw className="animate-spin" size={11} />
                                    載入歷史價...
                                </span>
                            )}
                            <div className="flex gap-1 text-[13px]">
                                {rangeBtn('1mo', '1M')}
                                {rangeBtn('3mo', '3M')}
                                {rangeBtn('6mo', '6M')}
                                {rangeBtn('1y', '1Y')}
                                {rangeBtn('max', 'ALL')}
                            </div>
                        </div>
                    </div>

                    <div className="relative" style={{ height: '260px' }}>
                        {dates.length === 0 && !historyLoading && (
                            <div className="absolute inset-0 flex items-center justify-center text-[13px]" style={{ color: 'var(--ink-3)' }}>
                                等待歷史價...
                            </div>
                        )}
                        <canvas ref={canvasRef}></canvas>
                    </div>

                    <div className="flex items-center gap-5 mt-3 text-[12px]" style={{ color: 'var(--ink-3)' }}>
                        <span className="flex items-center gap-1.5">
                            <span className="inline-block w-4" style={{ height: 2, background: 'var(--ink)' }}></span>
                            總市值
                        </span>
                        <span className="flex items-center gap-1.5">
                            <span className="inline-block w-4" style={{ borderTop: '1.5px dashed var(--ink-3)' }}></span>
                            累計投入成本
                        </span>
                    </div>
                </section>
            );
        };

        // ─────────────────────────────────────────────
        // PortfolioDashboard — main component
        // ─────────────────────────────────────────────
        // ─── ForumGate：論壇函式庫(marked/KaTeX/EasyMDE 等)改為點開論壇才載入,首頁不再背這些成本 ───
        // Dev-only forum sample (?forumFixture=1 as admin, ?forumFixture=reader as a member): an in-memory
        // stand-in for forum_posts + 判讀 tables so the list / post / editor layouts can be reviewed without
        // an account. Stripped from `vite build`.
        const FORUM_FIXTURE = import.meta.env.DEV && new URLSearchParams(window.location.search).get('forumFixture');
        const makeForumFixtureDb = () => {
            const posts = [
                { id: 'p1', title: '週線 MACD 底背離實戰：2026 年 BTC 的兩次訊號', tags: ['BTC', 'MACD', '背離'], published: true, created_at: '2026-09-20T09:30:00Z',
                  content: `## 為什麼看週線

日線雜訊太多，**週線**的背離參考性較高。

| 日期 | 收盤 | DIF |
|---|---:|---:|
| 2025/12/20 | 54,970 | -6,256 |
| 2026/06/20 | 52,593 | -2,966 |

價格破底、DIF 沒破底 → 底背離。

![週線 MACD 底背離](./icon-512.png)

\`\`\`js
const isBull = price2 < price1 && dif2 > dif1;
\`\`\`` },
                { id: 'p2', title: '恐懼貪婪指數低於 25 時分批加碼的回測', tags: ['DCA', '恐懼貪婪'], published: true, created_at: '2026-09-12T14:05:00Z',
                  content: `過去一年有 112 天處於極度恐懼。

- 每週定額
- 極度恐懼時加倍

> 紀律比預測重要。` },
                { id: 'p4', title: 'MSFT 週線：POC 與 0.618 之間的決戰', tags: ['MSFT', '美股', '技術分析', 'POC', '斐波那契'], published: true, no_judgment: true, created_at: '2026-09-15T16:00:00Z',
                  content: `本週 K 棒未收，不預判方向。`, },
                { id: 'p3', title: '（草稿）Wyckoff LPS 進場清單', tags: ['Wyckoff'], published: false, created_at: '2026-09-25T02:00:00Z',
                  content: `待整理。` },
            ];
            const judgments = [
                { id: 'j1', post_id: 'p1', market: 'crypto', symbol: 'BTC', timeframe: '1w', directions: ['up'], target_price: 72000,
                  range_low: null, range_high: null, methods: ['dow', 'volume'], sort: 0, chart_url: './icon-512.png',
                  levels: [{ type: 'POC', price: 58400 }, { type: '阻力', price: 68000 }, { type: '冰線', price: 52600 }],
                  reason: '週線價格破底但 DIF 沒破，量縮回測 POC 不破，看回到前高。',
                  locked_at: '2026-09-20T09:30:00Z', withdrawn_at: null, created_at: '2026-09-20T09:00:00Z' },
                { id: 'j2', post_id: 'p1', market: 'us', symbol: 'SPY', timeframe: '1d', directions: ['range', 'down'], target_price: 540,
                  range_low: 548, range_high: 575, methods: ['wyckoff'], sort: 1, chart_url: null, levels: [],
                  reason: '區間上緣兩次 UTAD，傾向跌回下緣。',
                  locked_at: '2026-09-20T09:30:00Z', withdrawn_at: '2026-09-24T02:10:00Z', created_at: '2026-09-20T09:00:00Z' },
                { id: 'j3', post_id: 'p3', market: 'crypto', symbol: 'ETH', timeframe: '4h', directions: ['up'], target_price: 2900,
                  range_low: null, range_high: null, methods: ['wyckoff'], sort: 0, chart_url: null, levels: [{ type: 'VAL', price: 2380 }],
                  reason: '', locked_at: null, withdrawn_at: null, created_at: '2026-09-25T02:00:00Z' },
                { id: 'j4', post_id: 'p2', market: 'crypto', symbol: 'BTC', timeframe: '1w', directions: ['up'], target_price: 82300, range_low: null, range_high: null,
                  methods: ['wyckoff', 'volume'], sort: 0, chart_url: null, levels: [], reason: '極度恐懼區分批，吸籌完成後看回前高。', backfilled: true,
                  locked_at: '2026-06-08T08:56:00Z', withdrawn_at: null, created_at: '2026-06-08T08:56:00Z' },
                { id: 'j5', post_id: 'p2', market: 'tw', symbol: '2371', timeframe: '1d', directions: ['down'], target_price: 22, range_low: null, range_high: null,
                  methods: ['pattern'], sort: 1, chart_url: null, levels: [], reason: '', backfilled: true,
                  locked_at: '2026-05-18T03:20:00Z', withdrawn_at: null, created_at: '2026-05-18T03:20:00Z' },
            ];
            const results = [
                { judgment_id: 'j4', status: 'right', settled_at: '2026-09-07T22:30:00Z', entry_price: 63237, exit_price: 76838, bars_total: 13, bars_elapsed: 13, change_pct: 0.215, max_progress: 0.72, regime: 'down', neutral_pct: 0.053 },
                { judgment_id: 'j5', status: 'wrong', settled_at: '2026-06-15T22:30:00Z', entry_price: 27.1, exit_price: 29.3, bars_total: 20, bars_elapsed: 20, change_pct: -0.081, max_progress: 0.1, regime: 'range', neutral_pct: 0.03 },
                { judgment_id: 'j1', status: 'pending', entry_price: 61250, bars_total: 13, bars_elapsed: 5, change_pct: 0.042, max_progress: 0.38,
                  regime: 'range', vpvr: { poc: 58420, vah: 66100, val: 54300 }, neutral_pct: 0.021 },
                { judgment_id: 'j2', status: 'withdrawn', settled_at: '2026-09-24T22:30:00Z', exit_price: 562.9, entry_price: 566.2, bars_total: 20, bars_elapsed: 3, change_pct: -0.006, max_progress: 0.12 },
            ];
            const tables = { forum_posts: posts, forum_judgments: judgments, forum_judgment_results: results };
            const now = () => new Date().toISOString();
            const builder = (table) => {
                const rows = tables[table] || [];
                const q = { op: 'select', filters: [], single: false, payload: null };
                const api = {
                    select() { return api; }, order() { return api; },
                    insert(p) { q.op = 'insert'; q.payload = p; return api; },
                    update(p) { q.op = 'update'; q.payload = p; return api; },
                    delete() { q.op = 'delete'; return api; },
                    eq(k, v) { q.filters.push(r => String(r[k]) === String(v)); return api; },
                    in(k, vs) { q.filters.push(r => vs.map(String).includes(String(r[k]))); return api; },
                    single() { q.single = true; return api; },
                    then(resolve, reject) {
                        const match = rows.filter(r => q.filters.every(f => f(r)));
                        let out = match;
                        if (q.op === 'insert') {
                            out = [].concat(q.payload).map(p => {
                                const r = { id: `${table}-${Math.random().toString(36).slice(2, 8)}`, created_at: now(), ...p };
                                // Mirrors the DB trigger: a call added to a published post locks at once
                                if (table === 'forum_judgments') r.locked_at = posts.some(x => x.id === r.post_id && x.published) ? now() : null;
                                rows.push(r);
                                return r;
                            });
                        } else if (q.op === 'update') {
                            match.forEach(r => Object.assign(r, q.payload, { updated_at: now() }));
                            // Mirrors the DB trigger: publishing locks the post's calls
                            if (table === 'forum_posts' && q.payload.published) {
                                match.forEach(post => judgments.filter(x => x.post_id === post.id && !x.locked_at).forEach(x => { x.locked_at = now(); }));
                            }
                        } else if (q.op === 'delete') {
                            match.forEach(r => rows.splice(rows.indexOf(r), 1));
                        }
                        return Promise.resolve({ data: q.single ? (out[0] || null) : out, error: null }).then(resolve, reject);
                    },
                };
                return api;
            };
            return {
                from: (table) => builder(table),
                storage: { from: () => ({ upload: async () => ({ error: null }), getPublicUrl: () => ({ data: { publicUrl: '' } }) }) },
            };
        };

        const ForumGate = (rawProps) => {
            const fixtureDb = useMemo(() => (FORUM_FIXTURE ? makeForumFixtureDb() : null), []);
            const props = FORUM_FIXTURE
                ? { ...rawProps, supabase: fixtureDb, user: { id: 'fixture-user', email: 'fixture@localhost' }, isAdmin: FORUM_FIXTURE !== 'reader', isPremium: true }
                : rawProps;
            const [ready, setReady] = useState(false);
            useEffect(() => {
                let cancelled = false;
                // 載入失敗也照樣渲染:forum.js 對缺少的函式庫有防呆,只是少了對應功能
                window.__loadForumLibs().catch(() => {}).then(() => { if (!cancelled) setReady(true); });
                return () => { cancelled = true; };
            }, []);
            if (!ready || !window.ForumApp) {
                return (
                    <div className="text-center py-16">
                        <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: 'var(--brand-1)' }}></div>
                    </div>
                );
            }
            return <window.ForumApp {...props} />;
        };

        // ─── 交易日誌(合約開單備忘錄):進場邏輯、止盈止損、事後反省 ───
        const calcJournalRR = (e) => {
            const entry = parseFloat(e.entry_price), sl = parseFloat(e.stop_loss), tp = parseFloat(e.take_profit);
            if (!entry || !sl || !tp) return null;
            const risk = e.direction === 'long' ? entry - sl : sl - entry;
            const reward = e.direction === 'long' ? tp - entry : entry - tp;
            if (risk <= 0 || reward <= 0) return null;
            return reward / risk;
        };
        const calcJournalPnlPct = (e) => {
            const entry = parseFloat(e.entry_price), exit = parseFloat(e.exit_price);
            if (!entry || !exit) return null;
            const raw = (exit - entry) / entry * (e.direction === 'long' ? 1 : -1) * 100;
            return raw * (parseFloat(e.leverage) || 1);
        };

        // ─── 開倉邏輯(Wyckoff 結構 + 全倉風控)────────────────────────────
        // 結構清單:只在「最後一點」進場,因為它自帶客觀的失效價,止損才不是用猜的。
        const WYCKOFF_SETUPS = [
            { v: 'lps', label: 'LPS｜最後支撐點', dir: 'long', desc: '吸籌區 Spring / SOS 之後最後一次回踩不破,買方最後接手處。失效價 = 該回踩低點。' },
            { v: 'lpsy', label: 'LPSY｜最後供給點', dir: 'short', desc: '派發區 UTAD / SOW 之後的無力反彈高點,賣方最後出貨處。失效價 = 該反彈高點。' },
            { v: 'spring', label: 'Spring｜破底翻甩測', dir: 'long', desc: '跌破區間下沿後快速收回,掃掉散戶止損拿走流動性。通常等它之後的 LPS 更穩。' },
            { v: 'utad', label: 'UTAD｜假突破出貨', dir: 'short', desc: '突破區間上沿後迅速跌回,誘多後派發。通常等它之後的 LPSY 更穩。' },
            { v: 'sos_bu', label: 'SOS 後 BU 回踩', dir: 'long', desc: '強勢上攻(SOS)後回踩前緣不破(Back Up),屬於 LPS 的一種型態。' },
            { v: 'sow_retest', label: 'SOW 後反抽回測', dir: 'short', desc: '破位下跌(SOW)後反抽回測破位處,屬於 LPSY 的一種型態。' },
            { v: 'other', label: '其他 / 非結構單', dir: null, desc: '不是 LPS / LPSY 的進場。請在下面理由欄寫清楚依據,事後才檢討得出東西。' },
        ];
        const setupMeta = (v) => WYCKOFF_SETUPS.find(s => s.v === v) || null;

        // 一筆單的完整風險體檢:名目、有效槓桿、單筆風險佔淨值、強平距離 + 警示
        const calcJournalRisk = (f) => {
            const n = (v) => { const x = parseFloat(v); return isNaN(x) ? null : x; };
            const entry = n(f.entry_price), sl = n(f.stop_loss), tp = n(f.take_profit);
            const lev = n(f.leverage) || 1;
            const margin = n(f.position_size);
            const equity = n(f.account_equity);
            const liq = n(f.liq_price);
            const isLong = f.direction === 'long';

            const notional = (margin != null && margin > 0) ? margin * lev : null;
            const effLev = (notional != null && equity > 0) ? notional / equity : null;
            const slDistPct = (entry > 0 && sl != null) ? Math.abs(entry - sl) / entry * 100 : null;
            const riskAmt = (notional != null && slDistPct != null) ? notional * slDistPct / 100 : null;
            const riskPctEquity = (riskAmt != null && equity > 0) ? riskAmt / equity * 100 : null;
            const liqDistPct = (entry > 0 && liq != null && liq > 0) ? Math.abs(entry - liq) / entry * 100 : null;
            const rr = calcJournalRR(f);
            // 反推:照 1% / 2% 風險原則,這個止損距離應該下多少名目
            const sizing = (equity > 0 && slDistPct > 0)
                ? { r1: equity * 0.01 / (slDistPct / 100), r2: equity * 0.02 / (slDistPct / 100) }
                : null;

            const warn = [];
            const st = setupMeta(f.setup);
            if (sl == null) warn.push({ lv: 'bad', t: '沒有止損 → 等於把強平價當止損,全倉時這是帳戶歸零的走法。' });
            if (st && st.dir && st.dir !== f.direction) warn.push({ lv: 'warn', t: `${st.label} 是${st.dir === 'long' ? '做多' : '做空'}結構,和目前方向不一致。` });
            if (sl != null && entry > 0) {
                const slWrongSide = isLong ? sl >= entry : sl <= entry;
                if (slWrongSide) warn.push({ lv: 'bad', t: '止損放在錯誤的一側,這筆單一進場就是虧的。' });
            }
            if (riskPctEquity != null) {
                if (riskPctEquity > 2) warn.push({ lv: 'bad', t: `單筆風險 ${riskPctEquity.toFixed(2)}% 淨值,超過 2% 上限 → 把名目降到 ${sizing ? sizing.r2.toFixed(0) : '—'} U 以下。` });
                else if (riskPctEquity > 1) warn.push({ lv: 'warn', t: `單筆風險 ${riskPctEquity.toFixed(2)}% 淨值,已超過 1% 的舒適區。` });
            }
            if (effLev != null) {
                if (effLev > 3) warn.push({ lv: 'bad', t: `有效槓桿 ${effLev.toFixed(2)}x 超過 3x 上限。槓桿檔位可以高,實際曝險不行。` });
                else if (effLev > 2) warn.push({ lv: 'warn', t: `有效槓桿 ${effLev.toFixed(2)}x,接近 3x 上限。` });
            }
            if (liqDistPct != null) {
                if (liqDistPct < 25) warn.push({ lv: 'bad', t: `強平價只距離 ${liqDistPct.toFixed(1)}%,一根插針就到。全倉的意義就是把它推到 30% 以外。` });
                else if (liqDistPct < 35) warn.push({ lv: 'warn', t: `強平距離 ${liqDistPct.toFixed(1)}%,極端行情仍有風險。` });
            }
            if (rr != null && rr < 2) warn.push({ lv: 'warn', t: `R:R 只有 1:${rr.toFixed(2)},低於 1:2 的話勝率要很高才划算。` });
            if (f.margin_mode === 'cross' && equity == null) warn.push({ lv: 'warn', t: '全倉沒填帳戶淨值 → 算不出有效槓桿,等於沒有風控。' });
            if (tp == null) warn.push({ lv: 'warn', t: '沒設止盈目標,出場容易變成憑感覺。' });

            return { notional, effLev, slDistPct, riskAmt, riskPctEquity, liqDistPct, rr, sizing, warn, equity, margin, lev };
        };

        // 側邊說明:老師整套開倉邏輯(結構 → 全倉 → 部位 → 防線 → 常見死法)
        const ENTRY_LOGIC_GUIDE = [
            {
                k: 'structure', icon: '①', title: '只打結構的「最後一點」',
                body: [
                    'LPS(Last Point of Support)= 吸籌區裡買方最後一次接手的回踩;LPSY(Last Point of Supply)= 派發區裡賣方最後一次出貨的反彈。',
                    '為什麼只打這兩個點:它們自帶一個客觀的失效價。LPS 的低點被跌破,做多的邏輯就死了 —— 止損不是憑感覺畫的,是結構告訴你的。',
                    '順序不能跳:① 先框出交易區間(TR)與階段 → ② 等 Spring / UTAD 拿走流動性 → ③ 等 SOS / SOW 確認方向 → ④ 回踩 LPS、反彈 LPSY 才進場。沒有 ①②③ 就沒有 ④。',
                ],
            },
            {
                k: 'cross', icon: '②', title: '槓桿倍數 ≠ 風險:全倉真正的用法',
                body: [
                    '平台上的 50x 只決定這筆「佔用多少起始保證金」(IM = 名目 ÷ 槓桿檔位),它完全不改變你賺賠多少。',
                    '真正的風險是 有效槓桿 = 名目部位 ÷ 帳戶淨值。實單案例:名目 30 萬鎂、帳戶淨值約 17 萬鎂 → 有效槓桿只有 1.7x,強平價落在 -33% 之外,而不是 50x 教科書上的 -2%。',
                    '全倉(Cross)的作用,是讓整個錢包替這個倉位墊底,把強平價推到極端行情也掃不到的地方。',
                    '代價:全倉是共命的,一個倉位爆 = 全部一起爆。所以全倉的前提永遠是「倉位夠小 + 一定掛主動止損」。',
                    '強平價不是止損,它只是最後一道牆。碰到牆等於整個帳戶結束,不是這筆單結束。',
                ],
            },
            {
                k: 'sizing', icon: '③', title: '部位是算出來的,不是想出來的',
                body: [
                    '1. 單筆可虧金額 R = 帳戶淨值 × 1%~2%',
                    '2. 止損距離 d = |進場價 − 止損價| ÷ 進場價',
                    '3. 名目部位 = R ÷ d',
                    '4. 實際下的保證金 = 名目部位 ÷ 槓桿檔位',
                    '例:淨值 10,000U、單筆風險 1%(100U)、LPS 止損距離 2.5% → 名目 = 100 ÷ 0.025 = 4,000U。用 20x 檔位下單只佔 200U 保證金,有效槓桿 0.4x,但打到止損就是賠 100U,不多不少。',
                ],
            },
            {
                k: 'guards', icon: '④', title: '四道防線,缺一不可',
                body: [
                    '1. 結構失效止損:LPS 低點下方 / LPSY 高點上方,進場的同時就掛。',
                    '2. 單筆風險 ≤ 淨值 2%(理想 1%)。',
                    '3. 有效槓桿 ≤ 3x,不看平台顯示的倍數。',
                    '4. 強平價距離 ≥ 25~30%,這是全倉唯一的意義。',
                ],
            },
            {
                k: 'traps', icon: '⑤', title: '最常見的四種死法',
                body: [
                    'ROE 爽度加倉:+247% 是對起始保證金算的虛榮數字,對帳戶淨值其實只有 +10%。看淨值,不看 ROE。',
                    '浮盈加倉:賺錢後淨值變大、有效槓桿自然下降;順手加倉等於把它推回原點,前面的風控全部作廢。',
                    '不掛止損想靠全倉硬扛:扛到最後不是賠這筆,是賠整個帳戶。',
                    '幣本位雙重曝險:抵押品是 BTC、部位又做多 BTC,跌的時候兩邊一起縮,強平來得比線性計算更快。要玩幣本位,有效槓桿再砍半(≤1.5x)。',
                ],
            },
        ];

        const EntryLogicGuide = () => (
            <div className="space-y-3">
                <div className="rounded-xl p-3" style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)' }}>
                    <p className="text-xs font-extrabold mb-1" style={{ color: 'var(--brand-1)' }}>這套邏輯一句話</p>
                    <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>
                        用 Wyckoff 的 LPS / LPSY 決定「在哪裡進場、止損放哪」,再用全倉把強平價推遠、用有效槓桿決定「下多大」。結構給你失效點,風控給你活下來的次數。
                    </p>
                </div>
                {ENTRY_LOGIC_GUIDE.map(sec => (
                    <details key={sec.k} open={sec.k === 'structure' || sec.k === 'cross'} className="rounded-xl overflow-hidden" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                        <summary className="px-3 py-2 cursor-pointer text-xs font-extrabold text-white select-none">
                            <span style={{ color: 'var(--brand-2)' }}>{sec.icon}</span> {sec.title}
                        </summary>
                        <div className="px-3 pb-3 space-y-1.5">
                            {sec.body.map((p, i) => (
                                <p key={i} className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>{p}</p>
                            ))}
                        </div>
                    </details>
                ))}
                <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                    ※ 這是課程筆記整理出來的交易紀律框架,不是投資建議。市場沒有任何方法能保證獲利,參數請照自己的資金與承受度調整。
                </p>
            </div>
        );

        // ISO 時間 → datetime-local 欄位值(本地時區,非 UTC)
        const toLocalDatetimeValue = (iso) => {
            const d = iso ? new Date(iso) : new Date();
            const p = (n) => String(n).padStart(2, '0');
            return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
        };

        const TradeJournalFormModal = ({ supabase, user, editing, onClose, onSaved }) => {
            const isEdit = !!editing;
            const [form, setForm] = useState(() => ({
                symbol: editing?.symbol || '',
                direction: editing?.direction || 'long',
                leverage: editing?.leverage ?? '',
                entry_price: editing?.entry_price ?? '',
                position_size: editing?.position_size ?? '',
                stop_loss: editing?.stop_loss ?? '',
                take_profit: editing?.take_profit ?? '',
                entry_reason: editing?.entry_reason || '',
                status: editing?.status || 'open',
                exit_price: editing?.exit_price ?? '',
                review: editing?.review || '',
                entry_at: toLocalDatetimeValue(editing?.entry_at),
                setup: editing?.setup || '',
                wyckoff_phase: editing?.wyckoff_phase || '',
                margin_mode: editing?.margin_mode || 'cross',
                account_equity: editing?.account_equity ?? '',
                liq_price: editing?.liq_price ?? '',
            }));
            const [saving, setSaving] = useState(false);
            const [showGuide, setShowGuide] = useState(true);
            const set = (k) => (ev) => setForm(f => ({ ...f, [k]: ev.target.value }));

            const rr = calcJournalRR(form);
            const risk = calcJournalRisk(form);
            const pnlPct = form.status === 'closed' ? calcJournalPnlPct(form) : null;
            const fmt = (v, d = 2) => (v == null || isNaN(v)) ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: d });

            const handleSave = async () => {
                if (!form.symbol.trim() || !form.entry_price) { alert('標的與進場價為必填。'); return; }
                setSaving(true);
                const num = (v) => (v === '' || v === null || isNaN(parseFloat(v))) ? null : parseFloat(v);
                const row = {
                    user_id: user.id,
                    symbol: form.symbol.trim().toUpperCase(),
                    direction: form.direction,
                    leverage: num(form.leverage),
                    entry_price: num(form.entry_price),
                    position_size: num(form.position_size),
                    stop_loss: num(form.stop_loss),
                    take_profit: num(form.take_profit),
                    entry_reason: form.entry_reason.trim() || null,
                    status: form.status,
                    exit_price: form.status === 'closed' ? num(form.exit_price) : null,
                    pnl: form.status === 'closed' ? calcJournalPnlPct(form) : null,
                    review: form.review.trim() || null,
                    entry_at: new Date(form.entry_at).toISOString(),
                    closed_at: form.status === 'closed' ? (editing?.closed_at || new Date().toISOString()) : null,
                    setup: form.setup || null,
                    wyckoff_phase: form.wyckoff_phase || null,
                    margin_mode: form.margin_mode || null,
                    account_equity: num(form.account_equity),
                    liq_price: num(form.liq_price),
                };
                const q = isEdit
                    ? supabase.from('trade_journal').update(row).eq('id', editing.id)
                    : supabase.from('trade_journal').insert(row);
                const { error: err } = await q;
                setSaving(false);
                if (err) {
                    // 還沒跑過 supabase/trade_journal_risk.sql 時,新欄位會找不到
                    if (/column|schema cache/i.test(err.message || '')) {
                        alert('儲存失敗：資料表還沒有風控欄位。\n請到 Supabase → SQL Editor 執行 supabase/trade_journal_risk.sql 後再試一次。\n\n原始訊息：' + err.message);
                    } else alert('儲存失敗：' + err.message);
                    return;
                }
                onSaved();
            };

            const inputCls = "w-full px-3 py-2 rounded-xl text-sm text-white outline-none focus:ring-1";
            const inputStyle = { background: 'var(--bg-soft)', border: '1px solid var(--line)' };
            const labelCls = "label mb-1.5 block";

            return (
                <div className="fixed inset-0 z-[100] flex items-start md:items-center justify-center p-4 overflow-y-auto" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(4px)' }} onClick={onClose}>
                    <div className="p-5 w-full max-w-4xl my-8" style={{ background: 'var(--paper)', borderTop: '3px solid var(--ink)', boxShadow: '0 24px 48px -16px rgba(0,0,0,0.45)' }} onClick={ev => ev.stopPropagation()}>
                        <div className="flex items-center justify-between mb-4 gap-2">
                            <h3 className="text-lg font-extrabold text-white">{isEdit ? '編輯開單紀錄' : '新增開單紀錄'}</h3>
                            <div className="flex items-center gap-2">
                                <button onClick={() => setShowGuide(v => !v)} className="px-2.5 py-1 rounded-lg text-xs font-bold" style={{ background: showGuide ? 'rgba(139,92,246,0.18)' : 'rgba(255,255,255,0.06)', color: showGuide ? 'var(--brand-2)' : 'var(--text-3)', border: '1px solid var(--line)' }} title="開倉邏輯說明">
                                    📖 開倉邏輯
                                </button>
                                <button onClick={onClose} className="text-slate-400 hover:text-white text-xl leading-none">×</button>
                            </div>
                        </div>
                        <div className={"grid gap-4 " + (showGuide ? "lg:grid-cols-[minmax(0,1fr)_320px]" : "grid-cols-1")}>
                        <div className="space-y-3">
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <label className={labelCls}>標的 *</label>
                                    <input className={inputCls + " mono uppercase"} style={inputStyle} placeholder="BTCUSDT" value={form.symbol} onChange={set('symbol')} />
                                </div>
                                <div>
                                    <label className={labelCls}>方向</label>
                                    <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                                        <button onClick={() => setForm(f => ({ ...f, direction: 'long' }))} className="flex-1 py-1 rounded-lg text-xs font-bold transition-all" style={form.direction === 'long' ? { background: 'rgba(0,214,143,0.18)', color: 'var(--up)' } : { color: 'var(--text-3)' }}>做多 ↑</button>
                                        <button onClick={() => setForm(f => ({ ...f, direction: 'short' }))} className="flex-1 py-1 rounded-lg text-xs font-bold transition-all" style={form.direction === 'short' ? { background: 'rgba(255,91,110,0.18)', color: 'var(--down)' } : { color: 'var(--text-3)' }}>做空 ↓</button>
                                    </div>
                                </div>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <div className="col-span-2">
                                    <label className={labelCls}>進場結構(Wyckoff)</label>
                                    <select className={inputCls} style={inputStyle} value={form.setup} onChange={set('setup')}>
                                        <option value="">— 選擇進場結構 —</option>
                                        {WYCKOFF_SETUPS.map(s => <option key={s.v} value={s.v}>{s.label}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label className={labelCls}>階段</label>
                                    <select className={inputCls} style={inputStyle} value={form.wyckoff_phase} onChange={set('wyckoff_phase')}>
                                        <option value="">—</option>
                                        {['A', 'B', 'C', 'D', 'E'].map(p => <option key={p} value={p}>Phase {p}</option>)}
                                    </select>
                                </div>
                            </div>
                            {setupMeta(form.setup) && (
                                <p className="text-xs leading-relaxed -mt-1 px-1" style={{ color: 'var(--text-3)' }}>
                                    {setupMeta(form.setup).desc}
                                </p>
                            )}
                            <div className="grid grid-cols-3 gap-3">
                                <div>
                                    <label className={labelCls}>進場價 *</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={inputStyle} value={form.entry_price} onChange={set('entry_price')} />
                                </div>
                                <div>
                                    <label className={labelCls}>槓桿 (x)</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={inputStyle} placeholder="10" value={form.leverage} onChange={set('leverage')} />
                                </div>
                                <div>
                                    <label className={labelCls}>保證金 (USDT)</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={inputStyle} placeholder="這筆佔用" value={form.position_size} onChange={set('position_size')} />
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <label className={labelCls}>止損 SL</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={{ ...inputStyle, borderColor: 'rgba(255,91,110,0.3)' }} value={form.stop_loss} onChange={set('stop_loss')} />
                                </div>
                                <div>
                                    <label className={labelCls}>止盈 TP</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={{ ...inputStyle, borderColor: 'rgba(0,214,143,0.3)' }} value={form.take_profit} onChange={set('take_profit')} />
                                </div>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                                <div>
                                    <label className={labelCls}>保證金模式</label>
                                    <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                                        <button onClick={() => setForm(f => ({ ...f, margin_mode: 'cross' }))} className="flex-1 py-1 rounded-lg text-xs font-bold transition-all" style={form.margin_mode === 'cross' ? { background: 'rgba(59,130,246,0.18)', color: 'var(--brand-1)' } : { color: 'var(--text-3)' }}>全倉</button>
                                        <button onClick={() => setForm(f => ({ ...f, margin_mode: 'isolated' }))} className="flex-1 py-1 rounded-lg text-xs font-bold transition-all" style={form.margin_mode === 'isolated' ? { background: 'var(--wash-2)', color: 'var(--text)' } : { color: 'var(--text-3)' }}>逐倉</button>
                                    </div>
                                </div>
                                <div>
                                    <label className={labelCls}>帳戶淨值 (USDT)</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={inputStyle} placeholder="算有效槓桿用" value={form.account_equity} onChange={set('account_equity')} />
                                </div>
                                <div>
                                    <label className={labelCls}>強平價</label>
                                    <input type="number" step="any" className={inputCls + " num"} style={inputStyle} placeholder="平台顯示" value={form.liq_price} onChange={set('liq_price')} />
                                </div>
                            </div>

                            {/* ── 風險體檢:有效槓桿才是真槓桿 ── */}
                            {!!form.entry_price && (
                            <div className="rounded-xl p-3 space-y-2.5" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                                <div className="flex items-center justify-between">
                                    <p className="label">風險體檢</p>
                                    {risk.effLev != null && (
                                        <span className="px-2 py-0.5 rounded-full text-[11px] font-extrabold mono" style={{ background: risk.effLev > 3 ? 'rgba(255,91,110,0.15)' : risk.effLev > 2 ? 'rgba(245,158,11,0.15)' : 'rgba(0,214,143,0.15)', color: risk.effLev > 3 ? 'var(--down)' : risk.effLev > 2 ? 'var(--warn)' : 'var(--up)' }}>
                                            有效槓桿 {risk.effLev.toFixed(2)}x
                                        </span>
                                    )}
                                </div>
                                <div className="grid grid-cols-3 gap-2">
                                    <div><p className="label">名目部位</p><p className="text-sm font-bold num text-white mt-0.5">{risk.notional != null ? fmt(risk.notional, 0) : '—'}</p></div>
                                    <div><p className="label">止損距離</p><p className="text-sm font-bold num mt-0.5" style={{ color: 'var(--text-2)' }}>{risk.slDistPct != null ? risk.slDistPct.toFixed(2) + '%' : '—'}</p></div>
                                    <div>
                                        <p className="label">單筆風險 / 淨值</p>
                                        <p className="text-sm font-bold num mt-0.5" style={{ color: risk.riskPctEquity == null ? 'var(--text-3)' : risk.riskPctEquity > 2 ? 'var(--down)' : risk.riskPctEquity > 1 ? 'var(--warn)' : 'var(--up)' }}>
                                            {risk.riskAmt != null ? fmt(risk.riskAmt, 0) : '—'}{risk.riskPctEquity != null ? ` · ${risk.riskPctEquity.toFixed(2)}%` : ''}
                                        </p>
                                    </div>
                                    <div><p className="label">R:R</p><p className="text-sm font-bold num text-white mt-0.5">{rr ? `1 : ${rr.toFixed(2)}` : '—'}</p></div>
                                    <div><p className="label">距強平</p><p className="text-sm font-bold num mt-0.5" style={{ color: risk.liqDistPct == null ? 'var(--text-3)' : risk.liqDistPct < 25 ? 'var(--down)' : risk.liqDistPct < 35 ? 'var(--warn)' : 'var(--up)' }}>{risk.liqDistPct != null ? risk.liqDistPct.toFixed(1) + '%' : '—'}</p></div>
                                    <div><p className="label">槓桿檔位佔用</p><p className="text-sm font-bold num text-white mt-0.5">{risk.margin != null ? fmt(risk.margin, 0) : '—'}</p></div>
                                </div>
                                {risk.sizing && (
                                    <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-3)' }}>
                                        照這個止損距離,1% 風險的名目應為 <span className="num" style={{ color: 'var(--text-2)' }}>{fmt(risk.sizing.r1, 0)}</span> U、
                                        2% 上限為 <span className="num" style={{ color: 'var(--text-2)' }}>{fmt(risk.sizing.r2, 0)}</span> U
                                        {risk.lev ? `(÷ ${risk.lev}x = 保證金 ${fmt(risk.sizing.r1 / risk.lev, 0)} ~ ${fmt(risk.sizing.r2 / risk.lev, 0)} U)` : ''}。
                                    </p>
                                )}
                                {risk.warn.length > 0 && (
                                    <div className="space-y-1 pt-1" style={{ borderTop: '1px solid var(--line)' }}>
                                        {risk.warn.map((w, i) => (
                                            <p key={i} className="text-xs leading-relaxed flex gap-1.5" style={{ color: w.lv === 'bad' ? 'var(--down)' : 'var(--warn)' }}>
                                                <span>{w.lv === 'bad' ? '⛔' : '⚠️'}</span><span style={{ color: 'var(--text-2)' }}>{w.t}</span>
                                            </p>
                                        ))}
                                    </div>
                                )}
                            </div>
                            )}
                            <div>
                                <label className={labelCls}>為什麼進場？（技術依據）</label>
                                <textarea rows="3" className={inputCls} style={inputStyle} placeholder="例：4H 跌破後 SFP 收回 + 日線 EMA 支撐共振、量價背離…" value={form.entry_reason} onChange={set('entry_reason')} />
                            </div>
                            <div>
                                <label className={labelCls}>進場時間</label>
                                <input type="datetime-local" className={inputCls + " mono"} style={inputStyle} value={form.entry_at} onChange={set('entry_at')} />
                            </div>
                            <div className="pt-2" style={{ borderTop: '1px solid var(--line)' }}>
                                <div className="flex items-center justify-between mb-2">
                                    <label className="label">狀態</label>
                                    <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'var(--bg-soft)', border: '1px solid var(--line)' }}>
                                        <button onClick={() => setForm(f => ({ ...f, status: 'open' }))} className="px-3 py-1 rounded-lg text-xs font-bold transition-all" style={form.status === 'open' ? { background: 'rgba(245,158,11,0.18)', color: 'var(--warn)' } : { color: 'var(--text-3)' }}>持倉中</button>
                                        <button onClick={() => setForm(f => ({ ...f, status: 'closed' }))} className="px-3 py-1 rounded-lg text-xs font-bold transition-all" style={form.status === 'closed' ? { background: 'var(--wash-2)', color: 'var(--text)' } : { color: 'var(--text-3)' }}>已平倉</button>
                                    </div>
                                </div>
                                {form.status === 'closed' && (
                                    <div className="space-y-3">
                                        <div className="grid grid-cols-2 gap-3 items-end">
                                            <div>
                                                <label className={labelCls}>出場價</label>
                                                <input type="number" step="any" className={inputCls + " num"} style={inputStyle} value={form.exit_price} onChange={set('exit_price')} />
                                            </div>
                                            {pnlPct != null && (
                                                <p className="text-lg font-extrabold num pb-1" style={{ color: pnlPct >= 0 ? 'var(--up)' : 'var(--down)' }}>
                                                    {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
                                                </p>
                                            )}
                                        </div>
                                        <div>
                                            <label className={labelCls}>事後反省</label>
                                            <textarea rows="3" className={inputCls} style={inputStyle} placeholder="哪裡做對？哪裡做錯？下次同樣情境怎麼處理？" value={form.review} onChange={set('review')} />
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div className="flex justify-end gap-2 pt-2">
                                <button onClick={onClose} className="px-4 py-2 rounded-xl text-sm font-semibold" style={{ background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text-2)' }}>取消</button>
                                <button onClick={handleSave} disabled={saving} className="px-5 py-2 rounded-xl text-sm font-bold pill-grad glow-brand">{saving ? '儲存中…' : '儲存'}</button>
                            </div>
                        </div>
                        {showGuide && (
                            <div className="rounded-xl p-3 lg:max-h-[70vh] lg:overflow-y-auto" style={{ background: 'var(--wash)', border: '1px solid var(--line)' }}>
                                <p className="text-sm font-extrabold text-white mb-2">開倉邏輯 · Wyckoff + 全倉風控</p>
                                <EntryLogicGuide />
                            </div>
                        )}
                        </div>
                    </div>
                </div>
            );
        };

        const TradeJournalDashboard = ({ supabase, user }) => {
            const [entries, setEntries] = useState([]);
            const [loading, setLoading] = useState(true);
            const [tableMissing, setTableMissing] = useState(false);
            const [error, setError] = useState(null);
            const [showForm, setShowForm] = useState(false);
            const [editing, setEditing] = useState(null);
            const [filter, setFilter] = useState('all'); // 'all' | 'open' | 'closed'

            const reload = async () => {
                if (!user || !supabase) return;
                setLoading(true);
                const { data, error: err } = await supabase
                    .from('trade_journal')
                    .select('*')
                    .order('entry_at', { ascending: false });
                if (err) {
                    if (err.code === '42P01' || err.code === 'PGRST205' || /trade_journal/.test(err.message || '')) setTableMissing(true);
                    else setError(err.message);
                } else {
                    setEntries(data || []);
                }
                setLoading(false);
            };
            useEffect(() => { reload(); }, [user]);

            const stats = useMemo(() => {
                const closed = entries.filter(e => e.status === 'closed' && e.pnl != null);
                const wins = closed.filter(e => e.pnl > 0).length;
                return {
                    open: entries.filter(e => e.status === 'open').length,
                    closed: closed.length,
                    winRate: closed.length ? (wins / closed.length * 100) : null,
                    avgPnl: closed.length ? closed.reduce((s, e) => s + e.pnl, 0) / closed.length : null,
                };
            }, [entries]);

            const shown = entries.filter(e => filter === 'all' || e.status === filter);

            const handleDelete = async (id) => {
                if (!confirm('確定刪除這筆開單紀錄？')) return;
                const { error: err } = await supabase.from('trade_journal').delete().eq('id', id);
                if (err) { alert('刪除失敗：' + err.message); return; }
                reload();
            };

            if (loading) {
                return (
                    <div className="text-center py-16">
                        <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: 'var(--brand-1)' }}></div>
                    </div>
                );
            }

            if (tableMissing) {
                return (
                    <div className="rounded-2xl ring-soft p-8" style={{ background: 'var(--surface)' }}>
                        <h3 className="text-lg font-bold text-white mb-2">📓 交易日誌需要先建立資料表</h3>
                        <p className="text-sm mb-4" style={{ color: 'var(--text-2)' }}>
                            到 Supabase Dashboard → SQL Editor 執行 repo 裡的 <code className="mono px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-soft)' }}>supabase/trade_journal.sql</code>，重新整理即可使用。
                        </p>
                    </div>
                );
            }

            return (
                <div className="space-y-4">
                    {/* Stats + actions */}
                    <div className="rounded-2xl ring-soft p-4 flex items-center justify-between flex-wrap gap-3" style={{ background: 'var(--surface)' }}>
                        <div className="flex gap-6 flex-wrap">
                            <div><p className="label">持倉中</p><p className="text-xl font-extrabold num mt-0.5" style={{ color: 'var(--warn)' }}>{stats.open}</p></div>
                            <div><p className="label">已平倉</p><p className="text-xl font-extrabold num mt-0.5 text-white">{stats.closed}</p></div>
                            <div><p className="label">勝率</p><p className="text-xl font-extrabold num mt-0.5 text-white">{stats.winRate == null ? '—' : stats.winRate.toFixed(0) + '%'}</p></div>
                            <div><p className="label">平均報酬</p><p className="text-xl font-extrabold num mt-0.5" style={{ color: stats.avgPnl == null ? 'var(--text)' : stats.avgPnl >= 0 ? 'var(--up)' : 'var(--down)' }}>{stats.avgPnl == null ? '—' : (stats.avgPnl >= 0 ? '+' : '') + stats.avgPnl.toFixed(1) + '%'}</p></div>
                        </div>
                        <button onClick={() => { setEditing(null); setShowForm(true); }} className="fs-btn solid">
                            <span>＋</span> 新增開單
                        </button>
                    </div>

                    {/* Filter pills */}
                    <div className="fs-toggle">
                        {[['all', '全部'], ['open', '持倉中'], ['closed', '已平倉']].map(([k, label]) => (
                            <button key={k} onClick={() => setFilter(k)} className={filter === k ? 'on' : ''}>
                                {label}
                            </button>
                        ))}
                    </div>

                    {error && <p className="text-sm" style={{ color: 'var(--down)' }}>讀取失敗：{error}</p>}

                    {shown.length === 0 && !error && (
                        <div className="rounded-2xl ring-soft p-10 text-center" style={{ background: 'var(--surface)' }}>
                            <h3 className="fs-title-sm mb-1">還沒有開單紀錄</h3>
                            <p className="text-sm" style={{ color: 'var(--text-2)' }}>每一單都寫下「為什麼進場」與事後反省，是最快變強的方式。</p>
                        </div>
                    )}

                    {shown.map(e => {
                        const rr = calcJournalRR(e);
                        const isLong = e.direction === 'long';
                        const pnl = e.pnl;
                        const er = calcJournalRisk(e);
                        const st = setupMeta(e.setup);
                        return (
                            <div key={e.id} className="rounded-2xl ring-soft p-4" style={{ borderTop: `2px solid ${e.status === 'open' ? 'var(--warn)' : pnl == null ? 'var(--ink)' : pnl >= 0 ? 'var(--up)' : 'var(--down)'}` }}>
                                <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <span className="font-extrabold text-white mono">{e.symbol}</span>
                                        <span className="px-2 py-0.5 rounded-full text-xs font-bold" style={isLong ? { background: 'rgba(0,214,143,0.15)', color: 'var(--up)' } : { background: 'rgba(255,91,110,0.15)', color: 'var(--down)' }}>
                                            {isLong ? '多' : '空'}{e.leverage ? ` ${e.leverage}x` : ''}
                                        </span>
                                        {st && (
                                            <span className="px-2 py-0.5 rounded-full text-xs font-bold" style={{ background: 'rgba(139,92,246,0.15)', color: 'var(--brand-2)' }} title={st.desc}>
                                                {st.label.split('｜')[0]}{e.wyckoff_phase ? ` · ${e.wyckoff_phase}` : ''}
                                            </span>
                                        )}
                                        {e.margin_mode && (
                                            <span className="px-2 py-0.5 rounded-full text-xs font-bold" style={{ background: 'var(--wash)', color: 'var(--text-3)' }}>
                                                {e.margin_mode === 'cross' ? '全倉' : '逐倉'}
                                            </span>
                                        )}
                                        <span className="px-2 py-0.5 rounded-full text-xs font-bold" style={e.status === 'open' ? { background: 'rgba(245,158,11,0.15)', color: 'var(--warn)' } : { background: 'var(--wash-2)', color: 'var(--text-2)' }}>
                                            {e.status === 'open' ? '持倉中' : '已平倉'}
                                        </span>
                                        {pnl != null && (
                                            <span className="font-extrabold num text-sm" style={{ color: pnl >= 0 ? 'var(--up)' : 'var(--down)' }}>
                                                {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs mono" style={{ color: 'var(--text-3)' }}>{new Date(e.entry_at).toLocaleString('zh-TW', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</span>
                                        {e.status === 'open' && (
                                            <button onClick={() => { setEditing({ ...e, status: 'closed' }); setShowForm(true); }} className="px-2.5 py-1 rounded-lg text-xs font-bold" style={{ background: 'rgba(59,130,246,0.15)', color: 'var(--brand-1)' }}>平倉</button>
                                        )}
                                        <button onClick={() => { setEditing(e); setShowForm(true); }} className="px-2 py-1 rounded-lg text-xs" style={{ color: 'var(--text-2)' }} title="編輯">✎</button>
                                        <button onClick={() => handleDelete(e.id)} className="px-2 py-1 rounded-lg text-xs" style={{ color: 'var(--text-3)' }} title="刪除">✕</button>
                                    </div>
                                </div>
                                <div className="grid grid-cols-3 md:grid-cols-7 gap-3 mb-3">
                                    <div><p className="label">進場</p><p className="text-sm font-bold num text-white mt-0.5">{e.entry_price}</p></div>
                                    <div><p className="label">止損</p><p className="text-sm font-bold num mt-0.5" style={{ color: 'var(--down)' }}>{e.stop_loss ?? '—'}</p></div>
                                    <div><p className="label">止盈</p><p className="text-sm font-bold num mt-0.5" style={{ color: 'var(--up)' }}>{e.take_profit ?? '—'}</p></div>
                                    <div><p className="label">R:R</p><p className="text-sm font-bold num text-white mt-0.5">{rr ? `1:${rr.toFixed(1)}` : '—'}</p></div>
                                    <div>
                                        <p className="label">有效槓桿</p>
                                        <p className="text-sm font-bold num mt-0.5" style={{ color: er.effLev == null ? 'var(--text-3)' : er.effLev > 3 ? 'var(--down)' : er.effLev > 2 ? 'var(--warn)' : 'var(--up)' }}>
                                            {er.effLev != null ? `${er.effLev.toFixed(2)}x` : '—'}
                                        </p>
                                    </div>
                                    <div>
                                        <p className="label">風險/淨值</p>
                                        <p className="text-sm font-bold num mt-0.5" style={{ color: er.riskPctEquity == null ? 'var(--text-3)' : er.riskPctEquity > 2 ? 'var(--down)' : er.riskPctEquity > 1 ? 'var(--warn)' : 'var(--up)' }}>
                                            {er.riskPctEquity != null ? `${er.riskPctEquity.toFixed(2)}%` : '—'}
                                        </p>
                                    </div>
                                    <div><p className="label">出場</p><p className="text-sm font-bold num text-white mt-0.5">{e.exit_price ?? '—'}</p></div>
                                </div>
                                {e.entry_reason && (
                                    <div className="rounded-xl p-3 mb-2" style={{ background: 'var(--bg-soft)' }}>
                                        <p className="label mb-1">進場理由</p>
                                        <p className="text-sm whitespace-pre-wrap" style={{ color: 'var(--text-2)' }}>{e.entry_reason}</p>
                                    </div>
                                )}
                                {e.review && (
                                    <div className="rounded-xl p-3" style={{ background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.2)' }}>
                                        <p className="label mb-1" style={{ color: 'var(--brand-2)' }}>事後反省</p>
                                        <p className="text-sm whitespace-pre-wrap" style={{ color: 'var(--text-2)' }}>{e.review}</p>
                                    </div>
                                )}
                            </div>
                        );
                    })}

                    {showForm && (
                        <TradeJournalFormModal
                            supabase={supabase}
                            user={user}
                            editing={editing}
                            onClose={() => setShowForm(false)}
                            onSaved={() => { setShowForm(false); reload(); }}
                        />
                    )}
                </div>
            );
        };

        const PortfolioDashboard = ({ supabase, user, watchlist, onUpdateWatchlist, isPremium }) => {
            const [subTab, setSubTab] = useState('holdings'); // 'holdings' | 'watchlist' | 'journal'
            const [ccyView, setCcyView] = useState('USD'); // 投組顯示幣別:預設 USD,有多幣別時可切換(例如 TWD)
            const [transactions, setTransactions] = useState([]);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [prices, setPrices] = useState({}); // {symbol::asset_type: {price, changePercent, currency}}
            const [pricesLoading, setPricesLoading] = useState(false);
            const [showModal, setShowModal] = useState(false);
            const [editingTx, setEditingTx] = useState(null);

            // 1. Fetch transactions
            const reload = async () => {
                if (PORTFOLIO_FIXTURE) { setTransactions(FIXTURE_TRANSACTIONS); setLoading(false); return; }
                if (!user || !supabase) return;
                setLoading(true);
                const { data, error: err } = await supabase
                    .from('transactions')
                    .select('*')
                    .order('date', { ascending: false });
                if (err) setError(err.message);
                else setTransactions(data || []);
                setLoading(false);
            };

            useEffect(() => { reload(); }, [user, supabase]);
            useThemeVersion();

            // 2. Compute holdings
            const holdings = useMemo(() => computeHoldings(transactions), [transactions]);

            // 3. Fetch current prices for all holdings
            useEffect(() => {
                if (holdings.length === 0) return;
                let cancelled = false;
                (async () => {
                    setPricesLoading(true);
                    // Skip cash entries (always price 1) and filter by asset type
                    const stockSyms = holdings.filter(h => h.asset_type === 'tw_stock' || h.asset_type === 'us_stock').map(h => h.symbol);
                    const cryptoSyms = holdings.filter(h => h.asset_type === 'crypto').map(h => h.symbol);
                    const proxy = `${SUPABASE_URL}/functions/v1/price-proxy`;
                    const headers = { 'apikey': SUPABASE_ANON_KEY, 'Authorization': 'Bearer ' + SUPABASE_ANON_KEY };
                    try {
                        const results = await Promise.all([
                            stockSyms.length ? fetch(`${proxy}?symbols=${stockSyms.join(',')}&type=stock`, { headers }).then(r => r.json()) : { data: [] },
                            cryptoSyms.length ? fetch(`${proxy}?symbols=${cryptoSyms.join(',')}&type=crypto`, { headers }).then(r => r.json()) : { data: [] },
                        ]);
                        if (cancelled) return;
                        const map = {};
                        const sd = results[0].data || [];
                        const cd = results[1].data || [];
                        holdings.forEach(h => {
                            if (h.asset_type === 'crypto') {
                                const found = cd.find(p => p.symbol === h.symbol.toLowerCase());
                                if (found) map[`${h.symbol}::${h.asset_type}`] = found;
                            } else {
                                const found = sd.find(p => p.symbol === h.symbol);
                                if (found) map[`${h.symbol}::${h.asset_type}`] = found;
                            }
                        });
                        // Inject synthetic price=1 for cash holdings
                        holdings.filter(h => h.asset_type === 'cash').forEach(h => {
                            map[`${h.symbol}::${h.asset_type}`] = { symbol: h.symbol, price: 1, previousClose: 1, change: 0, changePercent: 0, currency: h.symbol };
                        });
                        setPrices(map);
                    } catch (e) {
                        console.error('Price fetch failed', e);
                    } finally {
                        setPricesLoading(false);
                    }
                })();
                return () => { cancelled = true; };
            }, [holdings.length, transactions.length]);

            // 4. Summary totals
            const summary = useMemo(() => {
                const byCurrency = {};
                holdings.forEach(h => {
                    const ccy = getHoldingCurrency(h);
                    if (!byCurrency[ccy]) byCurrency[ccy] = { cost: 0, marketValue: 0, dividends: 0, count: 0, cash: 0 };
                    if (h.asset_type === 'cash') {
                        // Cash: cost == market value == net amount; price always 1
                        byCurrency[ccy].cost += h.totalCost;
                        byCurrency[ccy].marketValue += h.totalShares; // each "share" = 1 unit of currency
                        byCurrency[ccy].cash += h.totalShares;
                        byCurrency[ccy].count += 1;
                    } else {
                        byCurrency[ccy].cost += h.totalCost;
                        byCurrency[ccy].dividends += h.dividends;
                        byCurrency[ccy].count += 1;
                        const p = prices[`${h.symbol}::${h.asset_type}`];
                        if (p?.price) byCurrency[ccy].marketValue += h.totalShares * p.price;
                    }
                });
                return byCurrency;
            }, [holdings, prices]);

            // 5. Delete transaction
            const handleDelete = async (id) => {
                if (!confirm('確定要刪除這筆交易？此操作無法復原。')) return;
                const { error: err } = await supabase.from('transactions').delete().eq('id', id);
                if (err) { alert('刪除失敗：' + err.message); return; }
                reload();
            };

            if (!user) {
                return (
                    <section className="fs-section mt-6">
                        <div className="flex items-start gap-3 py-4">
                            <Lock size={20} className="mt-1 shrink-0" style={{ color: 'var(--ink-3)' }} />
                            <div>
                                <h3 className="fs-title-sm">請先登入</h3>
                                <p className="text-sm mt-1" style={{ color: 'var(--ink-2)' }}>登入後才能管理你的投資組合</p>
                            </div>
                        </div>
                    </section>
                );
            }

            if (loading) {
                return (
                    <div className="flex justify-center py-16">
                        <RefreshCw className="animate-spin" size={26} style={{ color: 'var(--ink-3)' }} />
                    </div>
                );
            }

            const SUB_TABS = [
                { k: 'holdings', l: '持股' },
                { k: 'watchlist', l: '觀察清單' },
                { k: 'journal', l: '交易日誌' },
            ];

            return (
                <div>
                    {/* Header */}
                    <section className="fs-section mt-6">
                        <div className="flex items-end justify-between flex-wrap gap-3">
                            <div>
                                <div className="flex items-baseline gap-3 flex-wrap">
                                    <h2 className="fs-title">
                                        {subTab === 'holdings' ? '我的投資組合' : subTab === 'watchlist' ? '觀察清單' : '交易日誌'}
                                    </h2>
                                    <span className="fs-lbl">MY INVESTMENTS</span>
                                </div>
                                <p className="text-[13px] mt-1 num" style={{ color: 'var(--ink-2)' }}>
                                    {subTab === 'holdings'
                                        ? `${holdings.length} 檔持股 · ${transactions.length} 筆交易`
                                        : subTab === 'watchlist'
                                            ? `${(watchlist || []).length} 檔追蹤中`
                                            : '合約開單備忘 · 進場邏輯與事後檢討'
                                    }
                                </p>
                            </div>
                            {subTab === 'holdings' && (
                                <div className="flex gap-1.5">
                                    <button onClick={reload} className="fs-btn icon" title="重新整理" aria-label="重新整理">
                                        <RefreshCw size={15} className={pricesLoading ? "animate-spin" : ""} />
                                    </button>
                                    <button onClick={() => { setEditingTx(null); setShowModal(true); }} className="fs-btn solid">
                                        ＋ 新增交易
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Sub-tab switcher */}
                        <div className="flex gap-6 mt-4 overflow-x-auto scrollbar-none" style={{ borderBottom: '1px solid var(--rule)' }}>
                            {SUB_TABS.map(t => (
                                <button
                                    key={t.k}
                                    onClick={() => setSubTab(t.k)}
                                    className="py-2 text-[15px] whitespace-nowrap"
                                    style={subTab === t.k
                                        ? { color: 'var(--ink)', fontWeight: 700, boxShadow: 'inset 0 -3px 0 var(--ink)' }
                                        : { color: 'var(--ink-2)' }}
                                >{t.l}</button>
                            ))}
                        </div>
                    </section>

                    {/* Watchlist sub-view */}
                    {subTab === 'watchlist' && (
                        <WatchlistDashboard
                            watchlist={watchlist}
                            onUpdateWatchlist={onUpdateWatchlist}
                            isPremium={isPremium}
                            user={user}
                        />
                    )}

                    {/* Trade journal sub-view */}
                    {subTab === 'journal' && (
                        <TradeJournalDashboard supabase={supabase} user={user} />
                    )}

                    {/* Holdings sub-view */}
                    {subTab !== 'holdings' ? null : <>

                    {/* Portfolio value chart + summary(單一幣別檢視:預設 USD,持有多種計價幣別時可切換) */}
                    {(() => {
                        const ccyList = Object.keys(summary);
                        if (ccyList.length === 0) return null;
                        const activeCcy = ccyList.includes(ccyView) ? ccyView : (ccyList.includes('USD') ? 'USD' : ccyList[0]);
                        const s = summary[activeCcy];
                        const pl = s.marketValue - s.cost;
                        const plPct = s.cost > 0 ? (pl / s.cost) * 100 : 0;
                        const isUp = pl >= 0;
                        const sym = activeCcy === 'TWD' ? 'NT$' : '$';
                        return (
                            <>
                                {/* 幣別切換:僅在同時持有多種計價幣別時顯示(例如 USD / TWD) */}
                                {ccyList.length > 1 && (
                                    <div className="fs-toggle mt-5">
                                        {ccyList.map(ccy => (
                                            <button key={ccy} onClick={() => setCcyView(ccy)} className={activeCcy === ccy ? 'on' : ''}>{ccy}</button>
                                        ))}
                                    </div>
                                )}

                                {/* Portfolio value chart(選中幣別) */}
                                <PortfolioValueChart
                                    transactions={transactions}
                                    holdings={holdings}
                                    prices={prices}
                                    currency={activeCcy}
                                />

                                {/* Summary figures(選中幣別) */}
                                <div className="fs-kv num mt-6" style={{ borderTop: '1px solid var(--ink)' }}>
                                    <div>
                                        <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>{activeCcy} · 總市值</div>
                                        <div className="text-lg md:text-[22px] font-bold" style={{ color: 'var(--ink)' }}>
                                            {sym}{s.marketValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>總成本</div>
                                        <div className="text-lg md:text-[22px] font-bold" style={{ color: 'var(--ink)' }}>
                                            {sym}{s.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>未實現損益</div>
                                        <div className="text-lg md:text-[22px] font-bold" style={{ color: isUp ? 'var(--up)' : 'var(--down)' }}>
                                            {isUp ? '+' : ''}{sym}{pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                            <span className="text-[13px] font-semibold ml-1.5">{isUp ? '▲' : '▼'} {isUp ? '+' : ''}{plPct.toFixed(2)}%</span>
                                        </div>
                                    </div>
                                    <div>
                                        <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>累計配息</div>
                                        <div className="text-lg md:text-[22px] font-bold" style={{ color: 'var(--ink)' }}>
                                            {sym}{s.dividends.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                        </div>
                                    </div>
                                </div>
                            </>
                        );
                    })()}

                    {/* Empty state */}
                    {holdings.length === 0 && transactions.length === 0 && (
                        <section className="fs-section mt-8">
                            <h3 className="fs-title-sm">還沒有任何交易</h3>
                            <p className="text-sm mt-1 mb-4" style={{ color: 'var(--ink-2)' }}>新增第一筆交易來開始追蹤你的投資組合</p>
                            <button onClick={() => { setEditingTx(null); setShowModal(true); }} className="fs-btn solid">
                                ＋ 新增第一筆交易
                            </button>
                        </section>
                    )}

                    {/* Holdings table */}
                    {holdings.length > 0 && (
                        <section className="fs-section mt-10">
                            <div className="fs-head"><h3 className="fs-title">目前持股</h3></div>
                            <div className="overflow-x-auto">
                                <table className="w-full text-[14px] num" style={{ minWidth: 620 }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--ink)' }}>
                                            {['代號', '持股', '均價', '現價', '市值', '損益'].map((h, i) => (
                                                <th key={h} className={`py-2 font-medium text-[12px] ${i === 0 ? 'text-left sticky left-0 z-[1]' : 'text-right'} ${i ? 'pl-3' : ''}`} style={{ color: 'var(--ink-3)', background: i === 0 ? 'var(--paper)' : undefined }}>{h}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {holdings.map((h) => {
                                            const isCash = h.asset_type === 'cash';
                                            const p = prices[`${h.symbol}::${h.asset_type}`];
                                            const avgCost = h.totalShares > 0 ? h.totalCost / h.totalShares : 0;
                                            const currentPrice = p?.price ?? null;
                                            const marketValue = currentPrice ? h.totalShares * currentPrice : null;
                                            const pl = isCash ? 0 : (marketValue !== null ? marketValue - h.totalCost : null);
                                            const plPct = pl !== null && h.totalCost > 0 && !isCash ? (pl / h.totalCost) * 100 : null;
                                            const sym = getHoldingSymbol(h);
                                            const isUp = pl !== null && pl >= 0;
                                            const plColor = isUp ? 'var(--up)' : 'var(--down)';
                                            return (
                                                <tr key={`${h.symbol}::${h.asset_type}`} style={{ borderBottom: '1px solid var(--rule)' }}>
                                                    <td className="py-3 pr-3 sticky left-0 z-[1]" style={{ background: 'var(--paper)' }}>
                                                        <div className="font-bold" style={{ color: 'var(--ink)' }}>{isCash ? `${h.symbol} 現金` : h.symbol}</div>
                                                        <div className="text-[12px]" style={{ color: 'var(--ink-3)' }}>
                                                            {ASSET_TYPES[h.asset_type].label}
                                                            {h.dividends > 0 && ` · 配息 ${sym}${h.dividends.toFixed(0)}`}
                                                        </div>
                                                    </td>
                                                    <td className="py-3 pl-3 text-right" style={{ color: 'var(--ink)' }}>
                                                        {isCash ? `${sym}${h.totalShares.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : h.totalShares.toLocaleString(undefined, { maximumFractionDigits: 6 })}
                                                    </td>
                                                    <td className="py-3 pl-3 text-right" style={{ color: 'var(--ink-2)' }}>
                                                        {isCash ? '—' : `${sym}${avgCost.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                                                    </td>
                                                    <td className="py-3 pl-3 text-right" style={{ color: 'var(--ink)' }}>
                                                        {isCash ? '—' : (currentPrice !== null ? `${sym}${currentPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : '—')}
                                                        {!isCash && p?.changePercent !== null && p?.changePercent !== undefined && (
                                                            <div className="text-[12px]" style={{ color: p.changePercent >= 0 ? 'var(--up)' : 'var(--down)' }}>
                                                                {p.changePercent >= 0 ? '▲ +' : '▼ '}{p.changePercent.toFixed(2)}%
                                                            </div>
                                                        )}
                                                    </td>
                                                    <td className="py-3 pl-3 text-right font-semibold" style={{ color: 'var(--ink)' }}>
                                                        {marketValue !== null ? `${sym}${marketValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—'}
                                                    </td>
                                                    <td className="py-3 pl-3 text-right">
                                                        {isCash ? <span style={{ color: 'var(--ink-3)' }}>—</span> : (pl !== null ? (
                                                            <div>
                                                                <div className="font-bold" style={{ color: plColor }}>
                                                                    {isUp ? '+' : ''}{sym}{pl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                                </div>
                                                                <div className="text-[12px]" style={{ color: plColor }}>
                                                                    {isUp ? '▲ +' : '▼ '}{plPct.toFixed(2)}%
                                                                </div>
                                                            </div>
                                                        ) : '—')}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </section>
                    )}

                    {/* Recent transactions */}
                    {transactions.length > 0 && (
                        <section className="fs-section mt-10">
                            <div className="fs-head">
                                <h3 className="fs-title">交易紀錄</h3>
                                <span className="text-[12px] num" style={{ color: 'var(--ink-3)' }}>共 {transactions.length} 筆</span>
                            </div>
                            <div>
                                {transactions.slice(0, 20).map((tx) => {
                                    const isCashTx = tx.asset_type === 'cash';
                                    const sym = isCashTx ? (tx.symbol === 'TWD' ? 'NT$' : '$') : (ASSET_TYPES[tx.asset_type]?.symbol || '$');
                                    const typeColor = tx.type === 'buy' ? 'var(--up)' : tx.type === 'sell' ? 'var(--down)' : 'var(--amber)';
                                    const typeLabel = isCashTx
                                        ? (tx.type === 'buy' ? '存入' : '提領')
                                        : (tx.type === 'buy' ? '買入' : tx.type === 'sell' ? '賣出' : '配息');
                                    return (
                                        <div key={tx.id} className="py-3 flex items-center gap-3" style={{ borderBottom: '1px solid var(--rule)' }}>
                                            <span className="fs-chip whitespace-nowrap" style={{ color: typeColor }}>{typeLabel}</span>
                                            <div className="flex-1 min-w-0">
                                                <div className="font-semibold text-[15px]" style={{ color: 'var(--ink)' }}>
                                                    {isCashTx ? `${tx.symbol} 現金` : tx.symbol}
                                                </div>
                                                <div className="text-[12px] num mt-0.5 truncate" style={{ color: 'var(--ink-3)' }}>
                                                    {tx.date}
                                                    {isCashTx && ` · ${sym}${Number(tx.shares).toLocaleString()}`}
                                                    {!isCashTx && tx.type !== 'dividend' && ` · ${Number(tx.shares).toLocaleString()} 股 @ ${sym}${Number(tx.price).toLocaleString()}`}
                                                    {!isCashTx && tx.type === 'dividend' && ` · 配息 ${sym}${Number(tx.price).toLocaleString()}`}
                                                    {tx.note && ` · ${tx.note}`}
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => { setEditingTx(tx); setShowModal(true); }}
                                                className="px-2 py-1 text-[13px] underline-offset-4 hover:underline"
                                                style={{ color: 'var(--ink-2)' }}
                                            >
                                                編輯
                                            </button>
                                            <button
                                                onClick={() => handleDelete(tx.id)}
                                                className="px-2 py-1 text-[13px] underline-offset-4 hover:underline"
                                                style={{ color: 'var(--down)' }}
                                            >
                                                刪除
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                            {transactions.length > 20 && (
                                <div className="pt-3 text-[12px] num" style={{ color: 'var(--ink-3)' }}>
                                    僅顯示最近 20 筆 · 總共 {transactions.length} 筆
                                </div>
                            )}
                        </section>
                    )}

                    </>}

                    {/* Modal (shared by both sub-tabs) */}
                    {showModal && (
                        <AddTransactionModal
                            supabase={supabase}
                            user={user}
                            existing={editingTx}
                            onClose={() => { setShowModal(false); setEditingTx(null); }}
                            onSaved={reload}
                        />
                    )}
                </div>
            );
        };

        // ─────────────────────────────────────────────
        // InstallHelpModal — tabbed instructions for all platforms
        // ─────────────────────────────────────────────
        const InstallHelpModal = ({ detectedPlatform, onClose, canPrompt, onPrompt }) => {
            // Decide initial tab from detected platform
            const initialTab = (detectedPlatform === 'ios') ? 'ios'
                : (detectedPlatform === 'android') ? 'android'
                : 'desktop';
            const [tab, setTab] = useState(initialTab);

            const isFile = typeof window !== 'undefined' && window.location.protocol === 'file:';
            // LINE / Facebook / Instagram in-app browsers cannot install web apps at all
            const inApp = typeof navigator !== 'undefined' && /\bLine\/|FBAN|FBAV|Instagram/i.test(navigator.userAgent);

            const Step = ({ n, children }) => (
                <li className="flex gap-3 py-2.5" style={{ borderTop: n > 1 ? '1px solid var(--rule)' : 'none' }}>
                    <span className="num font-black text-[15px] w-5 shrink-0" style={{ color: 'var(--ink)' }}>{n}</span>
                    <div className="text-[14px] leading-relaxed" style={{ color: 'var(--ink-2)' }}>{children}</div>
                </li>
            );
            const B = ({ children }) => <strong style={{ color: 'var(--ink)' }}>{children}</strong>;
            const Note = ({ children }) => (
                <p className="text-[13px] leading-relaxed mt-3 pt-3" style={{ color: 'var(--ink-2)', borderTop: '1px solid var(--rule)' }}>{children}</p>
            );

            return (
                <div className="fixed inset-0 flex items-center justify-center z-[100] p-4" style={{ background: 'rgba(0,0,0,0.55)' }} onClick={onClose}>
                    <div className="max-w-md w-full relative max-h-[90vh] overflow-y-auto custom-scrollbar p-6"
                        style={{ background: 'var(--paper)', borderTop: '3px solid var(--ink)', boxShadow: '0 24px 48px -16px rgba(0,0,0,0.45)' }}
                        onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="install-title">
                        <button onClick={onClose} className="absolute top-4 right-4 fs-btn icon" aria-label="關閉">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
                        </button>

                        <div className="flex items-center gap-3 mb-5 pr-10">
                            <LogoMark size={36} />
                            <div>
                                <p className="fs-lbl">INSTALL AS APP</p>
                                <h3 id="install-title" className="fs-title-sm">將 SmartDCA 加到主畫面</h3>
                            </div>
                        </div>

                        {inApp && (
                            <p className="text-[13px] leading-relaxed mb-4 p-3" style={{ color: 'var(--ink)', border: '1.5px solid var(--amber)' }}>
                                你正在 LINE／Facebook 的內建瀏覽器中，這裡無法安裝 App。請點右上角選單，選 <B>「以瀏覽器開啟」</B>（iPhone 用 Safari、Android 用 Chrome），再按一次「安裝 App」。
                            </p>
                        )}

                        {canPrompt && (
                            <button onClick={onPrompt} className="fs-btn solid w-full mb-4">立即安裝</button>
                        )}

                        {/* Platform tabs */}
                        <div className="flex gap-6 mb-2" style={{ borderBottom: '1px solid var(--rule)' }}>
                            {[['desktop', '電腦'], ['ios', 'iPhone / iPad'], ['android', 'Android']].map(([k, l]) => (
                                <button key={k} onClick={() => setTab(k)} className="py-2 text-[14px] whitespace-nowrap"
                                    style={tab === k ? { color: 'var(--ink)', fontWeight: 700, boxShadow: 'inset 0 -3px 0 var(--ink)' } : { color: 'var(--ink-2)' }}>
                                    {l}
                                </button>
                            ))}
                        </div>

                        {tab === 'desktop' && (
                            <div>
                                <p className="fs-title-sm mt-3">Chrome / Edge</p>
                                <ol>
                                    <Step n={1}>網址列右側出現 <B>安裝圖示</B>（螢幕加向下箭頭）時，直接點它，再按 <B>「安裝」</B>。</Step>
                                    <Step n={2}>沒看到圖示：Chrome 點右上角 <B>⋮</B> → <B>「投放、儲存與分享」</B> → <B>「安裝網頁…」</B>；Edge 點 <B>⋯</B> → <B>「應用程式」</B> → <B>「將此網站安裝為應用程式」</B>。</Step>
                                    <Step n={3}>安裝後 SmartDCA 會出現在桌面與開始選單，像一般 App 一樣開啟。</Step>
                                </ol>
                                <Note>如果之前關掉過安裝提示、或安裝後又移除，Chrome 會暫時不主動詢問，請用第 2 步的選單安裝。</Note>
                                <p className="fs-title-sm mt-5">Safari（macOS）</p>
                                <ol>
                                    <Step n={1}>選單列點 <B>檔案</B> → <B>「加入 Dock」</B>（macOS Sonoma 以上）。</Step>
                                </ol>
                                <p className="fs-title-sm mt-5">Firefox</p>
                                <p className="text-[14px] mt-1" style={{ color: 'var(--ink-2)' }}>桌面版 Firefox 目前不支援 PWA 安裝，請改用 Chrome 或 Edge。</p>
                                {isFile && (
                                    <Note>你目前用 <code>file://</code> 開啟，PWA 必須在 HTTPS 或 localhost 才能安裝。</Note>
                                )}
                            </div>
                        )}

                        {tab === 'ios' && (
                            <div>
                                <ol className="mt-2">
                                    <Step n={1}>用 <B>Safari</B> 開啟本網站（iOS 16.4 以上的 Chrome、Edge 也可以），點 <B>分享</B> 按鈕（方框加向上箭頭）。Safari 在下方工具列，Chrome 在網址列右側。</Step>
                                    <Step n={2}>往下滑，選 <B>「加入主畫面」</B>。</Step>
                                    <Step n={3}>右上角點 <B>「新增」</B> 完成。</Step>
                                </ol>
                                <Note>主畫面上的舊圖示不會自動更新：長按舊圖示 → 移除 App，再依上面步驟重新加入，就會換成新 Logo。</Note>
                            </div>
                        )}

                        {tab === 'android' && (
                            <div>
                                <ol className="mt-2">
                                    <Step n={1}>用 <B>Chrome</B> 開啟本網站，點右上角 <B>⋮</B> 選單。</Step>
                                    <Step n={2}>選 <B>「安裝應用程式」</B>（舊版 Chrome 顯示為「加到主畫面」）。</Step>
                                    <Step n={3}>確認 <B>「安裝」</B>，App 會出現在主畫面與應用程式清單。</Step>
                                </ol>
                                <Note>已安裝的 App 圖示，Chrome 通常會在一天內自動更新成新 Logo；若沒有，長按圖示解除安裝後重新安裝。Samsung Internet 也支援安裝，選單位置略有不同。</Note>
                            </div>
                        )}

                        <button onClick={onClose} className="fs-btn w-full mt-6">知道了</button>
                    </div>
                </div>
            );
        };

        // ─────────────────────────────────────────────
        // SettingsModal — profile + start page + personal Gemini key
        // ─────────────────────────────────────────────
        const SettingsModal = ({ supabase, user, profile, onClose, onSaved }) => {
            const [displayName, setDisplayName] = useState(profile?.display_name || '');
            const [startPage, setStartPage] = useState(profile?.start_page || 'crypto');
            const [apiKey, setApiKey] = useState(profile?.personal_gemini_key || '');
            const [showKey, setShowKey] = useState(false);
            const [saving, setSaving] = useState(false);
            const [error, setError] = useState(null);
            const [success, setSuccess] = useState(false);
            const [section, setSection] = useState('profile'); // 'profile' | 'preferences' | 'api'

            const inputStyle = { background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text)' };

            const handleSave = async () => {
                setSaving(true);
                setError(null);
                setSuccess(false);
                const payload = {
                    display_name: displayName.trim() || null,
                    start_page: startPage,
                    personal_gemini_key: apiKey.trim() || null,
                };
                const { data, error: err } = await supabase
                    .from('user_profiles')
                    .update(payload)
                    .eq('id', user.id)
                    .select()
                    .single();
                setSaving(false);
                if (err) { setError(err.message); return; }
                setSuccess(true);
                window.__USER_GEMINI_KEY = data.personal_gemini_key || null;
                onSaved(data);
                setTimeout(() => setSuccess(false), 1800);
            };

            const sectionBtn = (key, label, icon) => (
                <button
                    onClick={() => setSection(key)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-xs font-semibold transition-colors flex items-center gap-2 ${section === key ? 'pill-grad' : 'hover:bg-white/[0.05]'}`}
                    style={section === key ? { color: 'var(--brand-ink)' } : { color: 'var(--text-2)' }}
                >
                    {icon}
                    <span>{label}</span>
                </button>
            );

            const iconUser = <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;
            const iconSliders = <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>;
            const iconKey = <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>;

            return (
                <div className="fixed inset-0 flex items-center justify-center z-[110] p-4" style={{ background: 'rgba(7,8,12,0.78)', backdropFilter: 'blur(8px)' }}>
                    <div className="glass-strong rounded-3xl shadow-2xl max-w-2xl w-full relative max-h-[90vh] overflow-hidden flex flex-col md:flex-row">
                        <button
                            onClick={onClose}
                            className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors z-10"
                            style={{ color: 'var(--text-3)' }}
                        >
                            ✕
                        </button>

                        {/* Sidebar */}
                        <div className="md:w-48 p-4 md:border-r" style={{ borderColor: 'var(--line)' }}>
                            <div className="mb-4">
                                <p className="label">SETTINGS</p>
                                <h3 className="text-base font-extrabold text-white mt-1">設定</h3>
                            </div>
                            <div className="space-y-1 flex md:flex-col gap-1 overflow-x-auto md:overflow-visible">
                                {sectionBtn('profile', '個人資料', iconUser)}
                                {sectionBtn('preferences', '偏好設定', iconSliders)}
                                {sectionBtn('api', 'API Keys', iconKey)}
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 p-6 overflow-y-auto custom-scrollbar">
                            {section === 'profile' && (
                                <div className="space-y-4">
                                    <div>
                                        <p className="label">PROFILE</p>
                                        <h4 className="text-lg font-bold text-white mt-1">個人資料</h4>
                                    </div>

                                    <div>
                                        <label className="label block mb-1.5">Email</label>
                                        <div className="px-3.5 py-2.5 rounded-xl text-sm mono" style={{ background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text-3)' }}>
                                            {user?.email}
                                        </div>
                                        <p className="text-[10px] mt-1" style={{ color: 'var(--text-3)' }}>無法修改</p>
                                    </div>

                                    <div>
                                        <label className="label block mb-1.5">顯示名稱</label>
                                        <input
                                            type="text"
                                            value={displayName}
                                            onChange={e => setDisplayName(e.target.value)}
                                            placeholder="自訂顯示名稱（預設用 email 前綴）"
                                            className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
                                            style={inputStyle}
                                        />
                                    </div>
                                </div>
                            )}

                            {section === 'preferences' && (
                                <div className="space-y-4">
                                    <div>
                                        <p className="label">PREFERENCES</p>
                                        <h4 className="text-lg font-bold text-white mt-1">偏好設定</h4>
                                    </div>

                                    <div>
                                        <label className="label block mb-1.5">起始頁面</label>
                                        <select
                                            value={startPage}
                                            onChange={e => setStartPage(e.target.value)}
                                            className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none"
                                            style={inputStyle}
                                        >
                                            <option value="crypto">加密貨幣</option>
                                            <option value="stock">美股</option>
                                            <option value="taiwan">台股</option>
                                            <option value="portfolio">我的投資</option>
                                            <option value="forum">論壇</option>
                                        </select>
                                        <p className="text-[10px] mt-1" style={{ color: 'var(--text-3)' }}>登入時自動開啟這個頁面</p>
                                    </div>
                                </div>
                            )}

                            {section === 'api' && (
                                <div className="space-y-4">
                                    <div>
                                        <p className="label">API KEYS</p>
                                        <h4 className="text-lg font-bold text-white mt-1">個人 API 金鑰</h4>
                                    </div>

                                    <div className="p-3 rounded-xl text-xs" style={{ background: 'rgba(196,244,50,0.06)', border: '1px solid rgba(196,244,50,0.2)', color: 'var(--text-2)' }}>
                                        💡 使用自己的 Gemini key 可享有：
                                        <ul className="list-disc list-inside mt-2 space-y-1">
                                            <li>不受預設配額限制</li>
                                            <li>使用紀錄歸到你自己的 Google 帳號</li>
                                            <li>不會耗用站方額度</li>
                                        </ul>
                                    </div>

                                    <div>
                                        <label className="label block mb-1.5">Gemini API Key</label>
                                        <div className="relative">
                                            <input
                                                type={showKey ? 'text' : 'password'}
                                                value={apiKey}
                                                onChange={e => setApiKey(e.target.value)}
                                                placeholder="AIzaSy..."
                                                className="w-full rounded-xl px-3.5 py-2.5 pr-20 text-sm text-white outline-none mono"
                                                style={inputStyle}
                                                autoComplete="off"
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowKey(!showKey)}
                                                className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 text-[10px] rounded hover:bg-white/10"
                                                style={{ color: 'var(--text-3)' }}
                                            >
                                                {showKey ? '隱藏' : '顯示'}
                                            </button>
                                        </div>
                                        <p className="text-[10px] mt-1" style={{ color: 'var(--text-3)' }}>
                                            申請：<a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener" style={{ color: 'var(--brand-1)' }} className="underline">Google AI Studio</a>
                                            ・留空則使用站方預設 key
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Footer */}
                            <div className="mt-6 pt-4 flex items-center gap-2 justify-end" style={{ borderTop: '1px solid var(--line)' }}>
                                {error && <p className="text-xs flex-1" style={{ color: 'var(--down)' }}>{error}</p>}
                                {success && <p className="text-xs flex-1" style={{ color: 'var(--up)' }}>✓ 已儲存</p>}
                                <button
                                    onClick={onClose}
                                    className="px-4 py-2 rounded-xl text-xs font-semibold transition-colors"
                                    style={{ background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text-2)' }}
                                >
                                    關閉
                                </button>
                                <button
                                    onClick={handleSave}
                                    disabled={saving}
                                    className="px-5 py-2 rounded-xl text-xs font-bold pill-grad glow-brand disabled:opacity-50"
                                >
                                    {saving ? '儲存中...' : '儲存設定'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            );
        };

        // Market tab: engraved-free text tab, underline marks the current market
        const NavTab = ({ active, onClick, full, short }) => (
            <button
                onClick={onClick}
                aria-current={active ? 'page' : undefined}
                className="py-2.5 text-[15px] whitespace-nowrap shrink-0 transition-colors"
                style={active
                    ? { color: 'var(--ink)', fontWeight: 700, boxShadow: 'inset 0 -3px 0 var(--ink)' }
                    : { color: 'var(--ink-2)', fontWeight: 500 }}
            >
                <span className="hidden md:inline">{full}</span>
                <span className="md:hidden">{short}</span>
            </button>
        );

        const App = () => {
            const [theme, setTheme] = useTheme();
            const [activeTab, setActiveTab] = useState(() =>
                window.location.hash.startsWith('#/forum') ? 'forum' : (PORTFOLIO_FIXTURE ? 'portfolio' : 'crypto')
            );
            const [notificationsEnabled, setNotificationsEnabled] = useState(false);

            // PWA install prompt
            // index.html captures beforeinstallprompt before this bundle mounts (window.__installPrompt)
            const [installPrompt, setInstallPrompt] = useState(() => window.__installPrompt || null);
            const [isPwaInstalled, setIsPwaInstalled] = useState(() => !!window.__appInstalled);
            const [showInstallHelp, setShowInstallHelp] = useState(false);

            useEffect(() => {
                // Detect if already running as installed PWA
                const standalone = window.matchMedia('(display-mode: standalone)').matches
                    || window.navigator.standalone === true;
                if (standalone) {
                    setIsPwaInstalled(true);
                    return;
                }

                const onInstallable = () => setInstallPrompt(window.__installPrompt || null);
                const onInstalled = () => {
                    setInstallPrompt(null);
                    setIsPwaInstalled(true);
                    setShowInstallHelp(false);
                };
                onInstallable();
                window.addEventListener('smartdca-installable', onInstallable);
                window.addEventListener('smartdca-installed', onInstalled);
                return () => {
                    window.removeEventListener('smartdca-installable', onInstallable);
                    window.removeEventListener('smartdca-installed', onInstalled);
                };
            }, []);

            const handleInstallApp = async () => {
                // Native prompt available (Chrome / Edge / Samsung Internet on HTTPS / localhost)
                const promptEvent = installPrompt || window.__installPrompt;
                if (promptEvent) {
                    try {
                        promptEvent.prompt();
                        const { outcome } = await promptEvent.userChoice;
                        if (outcome === 'accepted') {
                            setIsPwaInstalled(true);
                            setShowInstallHelp(false);
                        }
                    } catch (e) {
                        // A prompt event can only be used once; fall back to the instructions
                        setShowInstallHelp(true);
                    }
                    window.__installPrompt = null;
                    setInstallPrompt(null);
                    return;
                }
                // Fallback: show manual instructions
                setShowInstallHelp(true);
            };

            // Detect platform for help modal
            const getPlatform = () => {
                const ua = navigator.userAgent;
                if (/iPad|iPhone|iPod/.test(ua)) return 'ios';
                if (/Android/.test(ua)) return 'android';
                if (/Edg/.test(ua)) return 'edge';
                if (/Chrome/.test(ua)) return 'chrome';
                if (/Safari/.test(ua)) return 'safari';
                if (/Firefox/.test(ua)) return 'firefox';
                return 'other';
            };

            // Sync hash → activeTab (so direct links like /#/forum/abc work)
            useEffect(() => {
                const sync = () => {
                    if (window.location.hash.startsWith('#/forum')) {
                        setActiveTab('forum');
                    }
                };
                window.addEventListener('hashchange', sync);
                return () => window.removeEventListener('hashchange', sync);
            }, []);

            // User State
            const [user, setUser] = useState(null);
            const [isPremium, setIsPremium] = useState(false);
            const [isAdmin, setIsAdmin] = useState(false);
            const [watchlist, setWatchlist] = useState([]);
            const [appLoading, setAppLoading] = useState(true);
            const [userProfile, setUserProfile] = useState(null);
            const [showSettings, setShowSettings] = useState(false);
            const startPageAppliedRef = useRef(false);

            // 1. Check Auth Session & Subscribe to Changes
            useEffect(() => {
                if (!supabase) {
                    setAppLoading(false);
                    return;
                }

                // Initial Session Check
                supabase.auth.getSession().then(({ data: { session } }) => {
                    if (session) {
                        setUser(session.user);
                    }
                    setAppLoading(false);
                });

                // Listen for Auth Changes (Login, Logout, OAuth Redirects)
                const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
                    setUser(session?.user ?? null);
                    setAppLoading(false);

                    // Clean up the URL hash if it contains auth tokens
                    if (session && window.location.hash && window.location.hash.includes('access_token')) {
                        window.history.replaceState(null, '', window.location.pathname);
                    }
                });

                return () => subscription.unsubscribe();
            }, []);

            // 2. Fetch User Profile
            useEffect(() => {
                const fetchProfile = async () => {
                    if (!user || !supabase) {
                        setIsPremium(false);
                        setIsAdmin(false);
                        setWatchlist([]);
                        setUserProfile(null);
                        window.__USER_GEMINI_KEY = null;
                        startPageAppliedRef.current = false;
                        return;
                    }
                    try {
                        const { data, error } = await supabase
                            .from('user_profiles')
                            .select('is_premium, is_admin, watchlist, display_name, start_page, personal_gemini_key')
                            .eq('id', user.id)
                            .single();

                        if (data) {
                            setIsPremium(data.is_premium || false);
                            setIsAdmin(data.is_admin || false);
                            setWatchlist(data.watchlist || []);
                            setUserProfile(data);
                            window.__USER_GEMINI_KEY = data.personal_gemini_key || null;
                            // Apply start_page once on initial login (don't override user's manual tab clicks)
                            if (data.start_page && !startPageAppliedRef.current && !window.location.hash.startsWith('#/forum')) {
                                setActiveTab(data.start_page);
                                startPageAppliedRef.current = true;
                            }
                        } else if (error && error.code === 'PGRST116') {
                            // Profile doesn't exist, create default
                            const { error: insertErr } = await supabase
                                .from('user_profiles')
                                .insert({ id: user.id, is_premium: false, is_admin: false });
                            if (insertErr) console.error('Profile create failed:', insertErr.message);
                        }
                    } catch (error) {
                        console.error("Profile Error:", error);
                    }
                };
                fetchProfile();
            }, [user]);

            // 3. Update Watchlist (Sync to DB)
            const handleUpdateWatchlist = async (newWatchlist) => {
                setWatchlist(newWatchlist); // Optimistic update
                if (user && supabase) {
                    await supabase
                        .from('user_profiles')
                        .update({ watchlist: newWatchlist })
                        .eq('id', user.id);
                }
            };

            // 初始檢查通知權限 (No changes)
            useEffect(() => {
                if ("Notification" in window && Notification.permission === 'granted') {
                    setNotificationsEnabled(true);
                }
            }, []);

            // 處理通知開關切換 (No changes to logic, just copied for context)
            // VAPID Public Key
            const VAPID_PUBLIC_KEY = 'BP0TmK3lqnO-j0x1H9OGzYJgOV5x2FRN4roE_AQGecl9nEbuCqHZjk0yNP0omI0xgqDubzxdSY50XauvlyhjzNc';

            const urlBase64ToUint8Array = (base64String) => {
                const padding = '='.repeat((4 - base64String.length % 4) % 4);
                const base64 = (base64String + padding).replace(/\-/g, '+').replace(/_/g, '/');
                const rawData = window.atob(base64);
                const outputArray = new Uint8Array(rawData.length);
                for (let i = 0; i < rawData.length; ++i) {
                    outputArray[i] = rawData.charCodeAt(i);
                }
                return outputArray;
            };

            useEffect(() => {
                if (!user || !isPremium || !('serviceWorker' in navigator) || !('PushManager' in window)) return;
                let cancelled = false;
                navigator.serviceWorker.getRegistration()
                    .then(reg => reg && reg.pushManager.getSubscription())
                    .then(sub => { if (!cancelled) setNotificationsEnabled(!!sub && Notification.permission === 'granted'); })
                    .catch(() => {});
                return () => { cancelled = true; };
            }, [user, isPremium]);

            const toggleNotifications = async () => {
                // 1. Strict Check for Premium / Login
                if (!user || !isPremium) {
                    alert("🚫 需升級 Pro 版才能啟用推播通知\n\nPlease upgrade to Premium to enable customized push notifications.");

                    // Open Checkout if user is logged in but not premium; otherwise ask them to log in
                    if (user && !isPremium) {
                        const checkoutUrl = `${LEMON_CHECKOUT_URL}?checkout[email]=${encodeURIComponent(user.email)}`;
                        window.open(checkoutUrl, '_blank');
                    } else {
                        alert("請先登入");
                    }
                    return;
                }

                if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
                    alert("您的瀏覽器不支援推播通知");
                    return;
                }

                if (notificationsEnabled) {
                    // Really turn it off: drop the browser subscription and the stored one, so the daily push stops
                    try {
                        const reg = await navigator.serviceWorker.getRegistration();
                        const sub = reg && await reg.pushManager.getSubscription();
                        if (sub) await sub.unsubscribe();
                        if (supabase) await supabase.from('user_profiles').update({ push_subscription: null }).eq('id', user.id);
                    } catch (e) {
                        console.error("Unsubscribe Error:", e);
                    }
                    setNotificationsEnabled(false);
                    return;
                }

                try {
                    const permission = await Notification.requestPermission();
                    if (permission !== 'granted') {
                        alert("請允許通知權限以接收快訊");
                        return;
                    }

                    // Register SW & Subscribe
                    const registration = await navigator.serviceWorker.register('./sw.js');
                    const subscription = await registration.pushManager.subscribe({
                        userVisibleOnly: true,
                        applicationServerKey: urlBase64ToUint8Array(VAPID_PUBLIC_KEY)
                    });

                    // Save to Supabase if logged in
                    if (user && supabase) {
                        // Check if subscription changed to avoid redundant writes? Or just overwrite.
                        const { error } = await supabase
                            .from('user_profiles')
                            .update({ push_subscription: subscription })
                            .eq('id', user.id);

                        if (error) {
                            console.error("Supabase Error:", error);
                            // We don't block UI if DB write fails, but warn?
                            // alert("無法儲存訂閱設定 (DB Error)");
                        }
                    } else if (!user) {
                        // Still enable local state, but warn
                        // alert("請登入以同步推播設定");
                    }

                    setNotificationsEnabled(true);

                    // Confirmation through the service worker: `new Notification()` throws on Android Chrome
                    try {
                        await registration.showNotification("推播已啟用 ✅", {
                            body: "將為您監控市場！\n我們會在每天早上 8:00 發送您的自選股分析通知。",
                            icon: "./app-icon.png",
                        });
                    } catch (e) { /* the alert below still confirms */ }
                    alert("✅ 推播通知已成功啟用！\n\n系統將在每天早上 8:00 為您發送自選股的市場分析報告。");

                } catch (e) {
                    console.error("Push Error:", e);
                    alert("啟用失敗: " + e.message);
                }
            };

            // Notification Center State
            const [showNotificationDropdown, setShowNotificationDropdown] = React.useState(false);
            const [showCalendar, setShowCalendar] = React.useState(false);
            // Dev-only sample notifications alongside the portfolio fixture (stripped from production builds)
            const [notificationHistory, setNotificationHistory] = React.useState(() => PORTFOLIO_FIXTURE ? [
                { id: 'n1', title: '每日市場報告', body: 'BTC 恐懼貪婪指數 73（貪婪）' + String.fromCharCode(10) + '觀察清單 2 檔 RSI 高於 65', created_at: '2026-09-27T00:00:00Z', is_read: false },
                { id: 'n2', title: '論壇新文章', body: '週線 MACD 底背離實戰：2026 年 BTC 的兩次訊號', created_at: '2026-09-26T09:30:00Z', is_read: true },
            ] : []);
            const [unreadCount, setUnreadCount] = React.useState(0);

            // Fetch Notifications from user_profiles JSONB
            React.useEffect(() => {
                if (user && supabase) {
                    const fetchNotifications = async () => {
                        const { data, error } = await supabase
                            .from('user_profiles')
                            .select('notifications')
                            .eq('id', user.id)
                            .single();

                        if (data && data.notifications) {
                            // Ensure it's an array
                            const notifs = Array.isArray(data.notifications) ? data.notifications : [];
                            setNotificationHistory(notifs);
                            setUnreadCount(notifs.filter(n => !n.is_read).length);
                        }
                    };
                    fetchNotifications();
                }
            }, [user]);

            const markAsRead = async () => {
                if (unreadCount > 0 && user && notificationHistory.length > 0) {
                    // Update local state first
                    const updatedHistory = notificationHistory.map(n => ({ ...n, is_read: true }));
                    setNotificationHistory(updatedHistory);
                    setUnreadCount(0);

                    // Sync to DB (Optimistic Update)
                    await supabase
                        .from('user_profiles')
                        .update({ notifications: updatedHistory })
                        .eq('id', user.id);
                }
            };

            useEffect(() => {
                if (!showNotificationDropdown) return;
                const onKey = (e) => { if (e.key === 'Escape') setShowNotificationDropdown(false); };
                window.addEventListener('keydown', onKey);
                return () => window.removeEventListener('keydown', onKey);
            }, [showNotificationDropdown]);
            useEffect(() => { setShowNotificationDropdown(false); }, [activeTab]);

            const handleBellClick = () => {
                if (!user && !PORTFOLIO_FIXTURE) {
                    alert("請先登入以查看通知");
                    return;
                }
                setShowNotificationDropdown(!showNotificationDropdown);
                if (!showNotificationDropdown) {
                    markAsRead();
                }
            };

            return (
                <div className="min-h-screen pb-20" style={{ color: 'var(--ink)' }}>
                    <div className="max-w-5xl mx-auto px-4 sm:px-6">
                        {/* Header */}
                        <header className="flex justify-between items-center gap-3 pt-4 pb-3 relative" style={{ borderBottom: '3px solid var(--ink)' }}>
                            {/* Logo */}
                            <div className="flex items-center gap-2 sm:gap-2.5 min-w-0 shrink">
                                <LogoMark size={24} className="sm:w-7 sm:h-7" />
                                <h1 className="text-[18px] sm:text-[22px] font-black tracking-tight leading-tight whitespace-nowrap" style={{ color: 'var(--ink)' }}>Smart DCA</h1>
                                <p className="label hidden sm:block whitespace-nowrap">Intelligent investing</p>
                            </div>

                            {/* Right controls */}
                            <div className="flex items-center gap-1 sm:gap-1.5 shrink-0">
                                {/* PWA Install button — always visible until installed */}
                                {!isPwaInstalled && (
                                    <button
                                        onClick={handleInstallApp}
                                        className="hidden sm:inline-flex fs-btn sm h-9"
                                        title="安裝為桌面 App"
                                    >
                                        <Smartphone size={14} />
                                        <span>安裝 App</span>
                                    </button>
                                )}
                                {!isPwaInstalled && (
                                    <button
                                        onClick={handleInstallApp}
                                        className="sm:hidden fs-btn icon"
                                        title="安裝為桌面 App"
                                    >
                                        <Smartphone size={16} />
                                    </button>
                                )}

                                {/* Economic Calendar */}
                                <button
                                    onClick={() => setShowCalendar(true)}
                                    className="relative fs-btn icon"
                                    title="經濟日曆"
                                >
                                    <CalendarDays size={17} />
                                    {hasHighImpactSoon() && (
                                        <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--amber)', boxShadow: '0 0 0 2px var(--paper)' }}></span>
                                    )}
                                </button>

                                {/* Notification Center: above page content and the sticky nav, below every modal */}
                                <div className="relative">
                                    <button
                                        onClick={handleBellClick}
                                        disabled={!user && !PORTFOLIO_FIXTURE}
                                        className="relative fs-btn icon"
                                        aria-label="通知中心"
                                        aria-expanded={showNotificationDropdown}
                                        title="通知中心"
                                    >
                                        <BellRing size={17} style={{ color: !user ? 'var(--ink-3)' : (notificationsEnabled ? 'var(--accent)' : 'var(--ink)') }} />
                                        {unreadCount > 0 && user && (
                                            <span className="absolute top-1 right-1 w-2 h-2 rounded-full" style={{ background: 'var(--down)', boxShadow: '0 0 0 2px var(--paper)' }}></span>
                                        )}
                                    </button>

                                    {showNotificationDropdown && (
                                        <>
                                            <div className="fixed inset-0 z-[69] fs-scrim" onClick={() => setShowNotificationDropdown(false)} aria-hidden="true" />
                                            <div
                                                role="dialog"
                                                aria-label="通知中心"
                                                className="fixed top-20 left-1/2 -translate-x-1/2 w-[92vw] max-w-sm overflow-hidden animate-in fade-in zoom-in-95 duration-200 z-[70] md:absolute md:top-full md:left-auto md:right-0 md:translate-x-0 md:w-96 md:mt-2"
                                                style={{ background: 'var(--paper)', borderTop: '3px solid var(--ink)', boxShadow: '0 18px 40px -14px rgba(0,0,0,0.45)', border: '1px solid var(--rule)' }}
                                            >
                                                {/* Header: title, daily-push switch, close */}
                                                <div className="px-4 py-3 flex justify-between items-center gap-3" style={{ borderBottom: '1px solid var(--rule)' }}>
                                                    <h3 className="fs-title-sm">通知中心</h3>
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-[11px] num" style={{ color: 'var(--ink-3)' }}>{notificationsEnabled ? "DAILY · ON" : "DAILY · OFF"}</span>
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); toggleNotifications(); }}
                                                            role="switch"
                                                            aria-checked={notificationsEnabled}
                                                            aria-label="每日推播通知"
                                                            className="w-10 h-6 p-[3px] transition-colors shrink-0"
                                                            style={{ background: notificationsEnabled ? 'var(--ink)' : 'transparent', border: '1.5px solid var(--ink)' }}
                                                        >
                                                            <div className="w-4 h-4 transform transition-transform"
                                                                style={{ background: notificationsEnabled ? 'var(--paper)' : 'var(--ink)', transform: notificationsEnabled ? 'translateX(14px)' : 'translateX(0)' }}></div>
                                                        </button>
                                                        <button onClick={() => setShowNotificationDropdown(false)} className="fs-btn icon ml-1" aria-label="關閉通知中心" title="關閉">
                                                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
                                                        </button>
                                                    </div>
                                                </div>

                                                {/* Notification List */}
                                                <div className="max-h-[60vh] md:max-h-80 overflow-y-auto custom-scrollbar">
                                                    {notificationHistory.length > 0 ? (
                                                        notificationHistory.map((item) => (
                                                            <div key={item.id} className="px-4 py-3" style={{ borderBottom: '1px solid var(--rule)' }}>
                                                                <div className="flex justify-between items-start gap-3 mb-1">
                                                                    <h4 className="text-[14px] font-bold" style={{ color: 'var(--ink)' }}>{item.title || "系統通知"}</h4>
                                                                    <span className="text-[11px] num shrink-0" style={{ color: 'var(--ink-3)' }}>{new Date(item.created_at).toLocaleDateString()}</span>
                                                                </div>
                                                                <p className="text-[13px] whitespace-pre-line leading-relaxed" style={{ color: 'var(--ink-2)' }}>{item.body}</p>
                                                            </div>
                                                        ))
                                                    ) : (
                                                        <div className="px-4 py-8 text-center" style={{ color: 'var(--ink-3)' }}>
                                                            <BellOff size={26} className="mx-auto mb-2" />
                                                            <p className="text-[14px]">尚無歷史通知</p>
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Footer */}
                                                <div className="px-4 py-2.5">
                                                    <p className="text-[11px] num" style={{ color: 'var(--ink-3)' }}>DAILY REPORT · 08:00 AM</p>
                                                </div>
                                            </div>
                                        </>
                                    )}
                                </div>

                                {/* Theme switch: follows the system until the visitor picks one */}
                                <button
                                    onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                                    className="fs-btn icon"
                                    aria-label={theme === 'dark' ? '切換為淺色' : '切換為深色'}
                                    title={theme === 'dark' ? '切換為淺色' : '切換為深色'}
                                >
                                    {theme === 'dark'
                                        ? <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5" /><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8" /></svg>
                                        : <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden="true"><path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z" /></svg>}
                                </button>

                                <AuthComponent user={user} setUser={setUser} isPremium={isPremium} onOpenSettings={() => setShowSettings(true)} userProfile={userProfile} />
                            </div>
                        </header>

                        {/* Navigation Tabs */}
                        <nav className="flex gap-6 sticky top-0 z-50 overflow-x-auto scrollbar-none" style={{ background: 'var(--paper)', borderBottom: '1px solid var(--rule)' }}>
                            <NavTab active={activeTab === 'crypto'} onClick={() => setActiveTab('crypto')} full="加密貨幣" short="加密" />
                            <NavTab active={activeTab === 'stock'} onClick={() => setActiveTab('stock')} full="美股市場" short="美股" />
                            <NavTab active={activeTab === 'taiwan'} onClick={() => setActiveTab('taiwan')} full="台股市場" short="台股" />
                            {(user || PORTFOLIO_FIXTURE) && <NavTab active={activeTab === 'portfolio'} onClick={() => setActiveTab('portfolio')} full="我的投資" short="投資" />}
                            {FORUM_ENABLED && <NavTab active={activeTab === 'forum'} onClick={() => { setActiveTab('forum'); window.location.hash = '#/forum'; }} full="論壇" short="論壇" />}
                        </nav>

                        {/* Content Area */}
                        <main className="pt-2">
                            {appLoading ? (
                                <div className="flex justify-center py-20"><RefreshCw className="animate-spin text-slate-500" /></div>
                            ) : (
                                <>
                                    {activeTab === 'crypto' && (
                                        <CryptoDashboard
                                            notificationsEnabled={notificationsEnabled}
                                            toggleNotifications={toggleNotifications}
                                            userInfo={{ isPremium, watchlist, email: user?.email }}
                                            onUpdateWatchlist={handleUpdateWatchlist}
                                        />
                                    )}
                                    {activeTab === 'stock' && (
                                        <StockDashboard
                                            notificationsEnabled={notificationsEnabled}
                                            toggleNotifications={toggleNotifications}
                                            userInfo={{ isPremium, watchlist, email: user?.email }}
                                            onUpdateWatchlist={handleUpdateWatchlist}
                                        />
                                    )}
                                    {activeTab === 'taiwan' && (
                                        <TaiwanDashboard
                                            userInfo={{ isPremium, watchlist, email: user?.email }}
                                            onUpdateWatchlist={handleUpdateWatchlist}
                                        />
                                    )}
                                    {activeTab === 'portfolio' && (
                                        <PortfolioDashboard
                                            supabase={supabase}
                                            user={user || (PORTFOLIO_FIXTURE ? FIXTURE_USER : null)}
                                            watchlist={watchlist}
                                            onUpdateWatchlist={handleUpdateWatchlist}
                                            isPremium={isPremium}
                                        />
                                    )}
                                    {activeTab === 'forum' && FORUM_ENABLED && (
                                        <ForumGate
                                            supabase={supabase}
                                            user={user}
                                            isAdmin={isAdmin}
                                            isPremium={isPremium}
                                        />
                                    )}
                                </>
                            )}
                        </main>

                        {/* PWA Install Help Modal — all platforms */}
                        {showInstallHelp && (
                            <InstallHelpModal
                                detectedPlatform={getPlatform()}
                                canPrompt={!!installPrompt}
                                onPrompt={handleInstallApp}
                                onClose={() => setShowInstallHelp(false)}
                            />
                        )}

                        {showCalendar && <EconomicCalendarModal onClose={() => setShowCalendar(false)} />}

                        {/* Settings Modal */}
                        {showSettings && user && (
                            <SettingsModal
                                supabase={supabase}
                                user={user}
                                profile={userProfile}
                                onClose={() => setShowSettings(false)}
                                onSaved={(updated) => setUserProfile(updated)}
                            />
                        )}
                    </div>
                </div>
            );
        };

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    