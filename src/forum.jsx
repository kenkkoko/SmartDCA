import React from 'react';
// =========================================================
// SmartDCA Forum — 技術分析論壇 (Visual Refresh)
// 透過 <script type="text/babel" src="forum.js"> 載入
// 元件透過 window.* 暴露給 index.html 的主 script 使用
// =========================================================

const useHashRoute = () => {
  const [hash, setHash] = React.useState(
    typeof window !== 'undefined' ? window.location.hash : ''
  );
  React.useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);
  return hash;
};

const parseForumRoute = (hash) => {
  if (hash === '#/forum/new') return { view: 'editor', mode: 'new' };
  const editMatch = hash.match(/^#\/forum\/([^/]+)\/edit$/);
  if (editMatch)   return { view: 'editor', mode: 'edit', postId: editMatch[1] };
  const detailMatch = hash.match(/^#\/forum\/([^/]+)$/);
  if (detailMatch) return { view: 'detail', postId: detailMatch[1] };
  return { view: 'list' };
};

const navigate = (path) => {
  if (window.location.hash === path) return;
  window.location.hash = path;
};

const getExtFromFile = (file) => {
  const fromName = file.name && file.name.match(/\.([^.]+)$/);
  if (fromName) return fromName[1].toLowerCase();
  if (file.type === 'image/png')  return 'png';
  if (file.type === 'image/jpeg') return 'jpg';
  if (file.type === 'image/gif')  return 'gif';
  if (file.type === 'image/webp') return 'webp';
  return 'png';
};

const uploadImageToStorage = async (supabase, file) => {
  const ext = getExtFromFile(file);
  const id  = (window.crypto && window.crypto.randomUUID)
    ? window.crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const path = `posts/${id}.${ext}`;

  const { error } = await supabase.storage
    .from('forum-images')
    .upload(path, file, { cacheControl: '3600', upsert: false });

  if (error) throw error;

  const { data } = supabase.storage.from('forum-images').getPublicUrl(path);
  return data.publicUrl;
};

const renderMarkdownToHtml = (md) => {
  if (!md) return '';
  if (!window.marked || !window.DOMPurify) {
    return window.DOMPurify ? window.DOMPurify.sanitize(`<pre>${md}</pre>`) : '';
  }
  const rawHtml = window.marked.parse(md, { breaks: true, gfm: true });
  return window.DOMPurify.sanitize(rawHtml);
};

const PostCard = ({ post, onOpen }) => {
  const preview = (post.content || '').replace(/[#*`_>\-!\[\]()]/g, '').replace(/\s+/g, ' ').trim().slice(0, 100);
  const dateStr = new Date(post.created_at).toLocaleDateString('zh-TW', { year: 'numeric', month: '2-digit', day: '2-digit' });
  const timeStr = new Date(post.created_at).toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });
  return (
    <button
      onClick={() => onOpen(post.id)}
      className="group w-full text-left rounded-2xl p-5 transition-all ring-soft hover:scale-[1.005]"
      style={{ background: 'var(--surface)' }}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <h3 className="text-lg font-bold text-white flex-1 leading-snug group-hover:text-grad transition-colors">{post.title}</h3>
        {!post.published && (
          <span className="chip chip-warn whitespace-nowrap shrink-0">DRAFT · 草稿</span>
        )}
      </div>
      {preview && (
        <p className="text-sm leading-relaxed line-clamp-2 mb-3" style={{ color: 'var(--text-2)' }}>
          {preview}{preview.length >= 100 ? '...' : ''}
        </p>
      )}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex flex-wrap gap-1.5">
          {(post.tags || []).slice(0, 5).map((t) => (
            <span key={t} className="text-[11px] px-2 py-0.5 rounded-md mono" style={{ background: 'var(--wash)', color: 'var(--text-2)' }}>
              #{t}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-1.5 text-[11px] mono" style={{ color: 'var(--text-3)' }}>
          <span>{dateStr}</span>
          <span style={{ color: 'var(--rule)' }}>·</span>
          <span>{timeStr}</span>
        </div>
      </div>
    </button>
  );
};

const PostList = ({ supabase, isAdmin, onOpen }) => {
  const [posts, setPosts] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  const [searchQuery, setSearchQuery] = React.useState('');
  const [selectedTags, setSelectedTags] = React.useState([]);
  const [sortDesc, setSortDesc] = React.useState(true);
  const [draftsOnly, setDraftsOnly] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      const { data, error: err } = await supabase
        .from('forum_posts')
        .select('id, title, tags, published, content, created_at')
        .order('created_at', { ascending: false });
      if (cancelled) return;
      if (err) setError(err.message);
      else setPosts(data || []);
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [supabase]);

  const tagCounts = React.useMemo(() => {
    const m = new Map();
    for (const p of posts) {
      for (const t of (p.tags || [])) m.set(t, (m.get(t) || 0) + 1);
    }
    return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
  }, [posts]);

  const visiblePosts = React.useMemo(() => {
    let r = posts;
    if (draftsOnly)            r = r.filter((p) => !p.published);
    if (selectedTags.length)   r = r.filter((p) => selectedTags.every((t) => (p.tags || []).includes(t)));
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      r = r.filter((p) =>
        (p.title || '').toLowerCase().includes(q) ||
        (p.content || '').toLowerCase().includes(q) ||
        (p.tags || []).some((t) => t.toLowerCase().includes(q))
      );
    }
    r = [...r].sort((a, b) => {
      const da = new Date(a.created_at).getTime();
      const db = new Date(b.created_at).getTime();
      return sortDesc ? db - da : da - db;
    });
    return r;
  }, [posts, draftsOnly, selectedTags, searchQuery, sortDesc]);

  const toggleTag = (tag) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const clearFilters = () => {
    setSearchQuery('');
    setSelectedTags([]);
    setDraftsOnly(false);
  };
  const filtersActive = searchQuery || selectedTags.length > 0 || draftsOnly;

  const inputStyle = { background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text)' };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 justify-between">
        <div className="flex flex-wrap items-center gap-2 flex-1">
          <div className="relative flex-1 min-w-[200px]">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-3)' }}>
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="搜尋標題 / 內容 / 標籤..."
              className="w-full rounded-xl pl-9 pr-8 py-2.5 text-sm outline-none transition-colors"
              style={inputStyle}
              onFocus={(e) => e.target.style.borderColor = 'var(--ink)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--line)'}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full flex items-center justify-center hover:bg-white/10"
                style={{ color: 'var(--text-3)' }}
                aria-label="清除搜尋"
              >
                ✕
              </button>
            )}
          </div>
          <select
            value={sortDesc ? 'desc' : 'asc'}
            onChange={(e) => setSortDesc(e.target.value === 'desc')}
            className="rounded-xl px-3 py-2.5 text-sm outline-none cursor-pointer"
            style={inputStyle}
          >
            <option value="desc">最新優先</option>
            <option value="asc">最舊優先</option>
          </select>
          {isAdmin && (
            <label className="flex items-center gap-2 text-xs px-3 py-2.5 rounded-xl cursor-pointer hover:text-white transition-colors"
              style={{ ...inputStyle, color: 'var(--text-2)' }}>
              <input
                type="checkbox"
                checked={draftsOnly}
                onChange={(e) => setDraftsOnly(e.target.checked)}
                className="accent-purple-500"
              />
              只看草稿
            </label>
          )}
        </div>

        {isAdmin && (
          <button
            onClick={() => navigate('#/forum/new')}
            className="fs-btn solid whitespace-nowrap"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            新增文章
          </button>
        )}
      </div>

      {tagCounts.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="label">TAGS</span>
          {tagCounts.map(([tag, count]) => {
            const active = selectedTags.includes(tag);
            return (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                className="px-2.5 py-1 rounded-md transition-colors mono"
                style={active
                  ? { background: 'var(--ink)', color: 'var(--paper)' }
                  : { background: 'var(--wash)', color: 'var(--text-2)', border: '1px solid var(--line)' }
                }
              >
                #{tag} <span className="opacity-60">×{count}</span>
              </button>
            );
          })}
          {filtersActive && (
            <button
              onClick={clearFilters}
              className="ml-1 underline transition-colors"
              style={{ color: 'var(--text-3)' }}
            >
              清除全部
            </button>
          )}
        </div>
      )}

      {!loading && !error && posts.length > 0 && (
        <p className="text-xs mono" style={{ color: 'var(--text-3)' }}>
          顯示 <span className="text-white font-bold">{visiblePosts.length}</span> / {posts.length} 篇
          {filtersActive && <span className="ml-2" style={{ color: 'var(--accent)' }}>· 已套用篩選</span>}
        </p>
      )}

      {loading && (
        <div className="text-center py-16">
          <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'var(--rule)', borderTopColor: 'var(--ink)' }}></div>
          <p className="mt-3 text-sm mono" style={{ color: 'var(--text-3)' }}>LOADING POSTS...</p>
        </div>
      )}
      {error && (
        <div className="text-sm rounded-xl p-4" style={{ background: 'rgba(255,91,110,0.08)', border: '1px solid rgba(255,91,110,0.2)', color: 'var(--down)' }}>
          讀取失敗：{error}
        </div>
      )}
      {!loading && !error && posts.length === 0 && (
        <div className="text-center py-20 rounded-2xl ring-soft" style={{ background: 'var(--surface)' }}>
          <svg className="mx-auto mb-4 opacity-30" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--text-3)' }}>
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          <p style={{ color: 'var(--text-2)' }}>目前沒有任何文章</p>
          {isAdmin && <p className="text-xs mt-2 mono" style={{ color: 'var(--text-3)' }}>點右上「新增文章」開始</p>}
        </div>
      )}
      {!loading && posts.length > 0 && visiblePosts.length === 0 && (
        <div className="text-center py-20 rounded-2xl ring-soft" style={{ background: 'var(--surface)' }}>
          <svg className="mx-auto mb-4 opacity-30" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--text-3)' }}>
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <p style={{ color: 'var(--text-2)' }}>找不到符合條件的文章</p>
          <button
            onClick={clearFilters}
            className="text-xs mt-2 underline"
            style={{ color: 'var(--accent)' }}
          >
            清除篩選
          </button>
        </div>
      )}
      <div className="space-y-3">
        {!loading && visiblePosts.map((p) => <PostCard key={p.id} post={p} onOpen={onOpen} />)}
      </div>
    </div>
  );
};

// ───────────────────────────────────
// ImageLightbox — click any forum image to view full-screen.
//   • ESC / click backdrop / click ✕ → close
//   • ← / → keys or buttons → prev / next (multi-image posts)
//   • Body scroll is locked while open
//   • Adjacent images are preloaded for instant switching
// ───────────────────────────────────
const ImageLightbox = ({ images, index, onClose, onPrev, onNext }) => {
  const total = images.length;
  const hasMultiple = total > 1;
  // Controls (✕, ←/→, counter) hidden by default — tap anywhere to toggle.
  // ESC always closes regardless of controls state.
  const [controlsVisible, setControlsVisible] = React.useState(false);

  // Keyboard shortcuts + body scroll lock
  React.useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft' && hasMultiple) {
        onPrev();
        setControlsVisible(true);  // Surface controls when navigating via keyboard
      }
      else if (e.key === 'ArrowRight' && hasMultiple) {
        onNext();
        setControlsVisible(true);
      }
    };
    document.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [hasMultiple, onClose, onPrev, onNext]);

  // Preload neighbors so left/right feels instant
  React.useEffect(() => {
    if (!hasMultiple) return;
    [(index + 1) % total, (index - 1 + total) % total].forEach((i) => {
      const img = new Image();
      img.src = images[i];
    });
  }, [index, images, hasMultiple, total]);

  const toggleControls = () => setControlsVisible((v) => !v);

  const btnStyle = {
    background: 'rgba(255,255,255,0.08)',
    border: '1px solid rgba(255,255,255,0.18)',
    backdropFilter: 'blur(8px)',
    WebkitBackdropFilter: 'blur(8px)',
  };
  const fadeStyle = {
    opacity: controlsVisible ? 1 : 0,
    pointerEvents: controlsVisible ? 'auto' : 'none',
    transition: 'opacity 0.2s ease',
  };

  return (
    <div
      onClick={toggleControls}
      className="fixed inset-0 z-[1000] flex items-center justify-center p-4 md:p-8"
      style={{
        background: 'rgba(0,0,0,0.92)',
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        animation: 'forum-lightbox-fade 0.18s ease-out',
        cursor: 'pointer',
      }}
      role="dialog"
      aria-modal="true"
    >
      {/* Image itself — clicking it also toggles controls */}
      <img
        key={images[index]}
        src={images[index]}
        alt=""
        onClick={(e) => { e.stopPropagation(); toggleControls(); }}
        className="max-h-full max-w-full object-contain rounded-lg select-none"
        style={{ boxShadow: '0 24px 64px rgba(0,0,0,0.55)', cursor: 'pointer' }}
        draggable={false}
      />

      {/* Close button — toggle visibility */}
      <button
        onClick={(e) => { e.stopPropagation(); onClose(); }}
        className="absolute top-4 right-4 w-11 h-11 rounded-full flex items-center justify-center text-white text-lg hover:scale-110 z-10"
        style={{ ...btnStyle, ...fadeStyle, transition: 'opacity 0.2s ease, transform 0.15s ease' }}
        aria-label="關閉"
      >
        ✕
      </button>

      {/* Prev / Next + counter — toggle visibility */}
      {hasMultiple && (
        <>
          <button
            onClick={(e) => { e.stopPropagation(); onPrev(); }}
            className="absolute left-4 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full flex items-center justify-center text-white text-2xl hover:scale-110 z-10"
            style={{ ...btnStyle, ...fadeStyle, transition: 'opacity 0.2s ease, transform 0.15s ease' }}
            aria-label="上一張"
          >
            ←
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onNext(); }}
            className="absolute right-4 top-1/2 -translate-y-1/2 w-12 h-12 rounded-full flex items-center justify-center text-white text-2xl hover:scale-110 z-10"
            style={{ ...btnStyle, ...fadeStyle, transition: 'opacity 0.2s ease, transform 0.15s ease' }}
            aria-label="下一張"
          >
            →
          </button>
          <div
            onClick={(e) => e.stopPropagation()}
            className="absolute bottom-6 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full text-white text-sm mono z-10"
            style={{ ...btnStyle, ...fadeStyle }}
          >
            {index + 1} / {total}
          </div>
        </>
      )}
    </div>
  );
};

// ─── 判讀：每篇分析裡可以事後驗證的觀點 ───
// 讀者看得到判讀內容；驗證結果（forum_judgment_results）只有管理員讀得到，由資料庫 RLS 把關。
// 文章發布後判讀即鎖定，只能撤回（見 supabase/forum_judgments.sql）。
const JG_TIMEFRAMES = [
  { v: '4h', label: '4H', bars: 30, span: '約 5 天' },
  { v: '1d', label: '日線', bars: 20, span: '約 1 個月' },
  { v: '1w', label: '週線', bars: 13, span: '約 1 季' },
  { v: '1M', label: '月線', bars: 6, span: '約半年' },
];
const JG_MARKETS = [
  { v: 'crypto', label: '加密貨幣', ph: 'BTC' },
  { v: 'us', label: '美股', ph: 'SPY' },
  { v: 'tw', label: '台股', ph: '0050' },
];
const JG_DIRS = {
  up:    { label: '看多', color: 'var(--up)' },
  down:  { label: '看空', color: 'var(--down)' },
  range: { label: '盤整', color: 'var(--ink-2)' },
};
const JG_METHODS = [
  { v: 'dow', label: '道氏' },
  { v: 'wyckoff', label: '威科夫' },
  { v: 'pattern', label: '型態' },
  { v: 'volume', label: '量價' },
  { v: 'other', label: '其他' },
];
const JG_LEVELS = ['阻力', '支撐', '冰線', '溪流', 'POC', 'VAH', 'VAL'];
const JG_STATUS = {
  pending:   { label: '驗證中', color: 'var(--ink-2)' },
  hit:       { label: '達標', color: 'var(--up)' },
  right:     { label: '方向對', color: 'var(--up)' },
  flat:      { label: '持平', color: 'var(--amber)' },
  wrong:     { label: '錯', color: 'var(--down)' },
  withdrawn: { label: '已撤回', color: 'var(--ink-3)' },
};
const JG_REGIME = { up: '上升', down: '下降', range: '盤整' };
const JG_CRYPTO = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'TON', 'SUI', 'TRX'];
const JG_NOT_TICKERS = ['MACD', 'RSI', 'DCA', 'KD', 'MA', 'EMA', 'SMA', 'ATR', 'POC', 'VAH', 'VAL', 'VPVR', 'OBV', 'LPS', 'LPSY', 'SOS', 'SOW', 'UTAD', 'ETF'];

const jgTimeframe = (v) => JG_TIMEFRAMES.find((t) => t.v === v) || JG_TIMEFRAMES[1];
const jgNum = (v) => {
  if (v == null || v === '') return null;
  const x = parseFloat(String(v).replace(/,/g, ''));
  return Number.isFinite(x) ? x : null;
};
const jgPrice = (v) => (v == null || v === '' ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: 8 }));
const jgPct = (v) => (v == null ? '—' : `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}%`);
const jgDate = (v) => (v ? new Date(v).toLocaleDateString('zh-TW') : '—');

// 從文章標籤猜標的：BTC → 加密貨幣、0050 → 台股、其他全大寫代號 → 美股
const jgGuessSymbol = (tags) => {
  const up = tags.map((t) => t.trim().toUpperCase()).filter(Boolean);
  const crypto = up.find((t) => JG_CRYPTO.includes(t));
  if (crypto) return { market: 'crypto', symbol: crypto };
  const tw = up.find((t) => /^\d{4,6}[A-Z]?$/.test(t));
  if (tw) return { market: 'tw', symbol: tw };
  const us = tags.map((t) => t.trim()).find((t) => /^[A-Z]{1,5}$/.test(t) && !JG_NOT_TICKERS.includes(t));
  if (us) return { market: 'us', symbol: us };
  return { market: 'crypto', symbol: '' };
};

// 文章 Markdown 裡已貼上的圖片，給判讀挑一張當圖表
const jgExtractImages = (md) =>
  Array.from(new Set(Array.from((md || '').matchAll(/!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)/g), (m) => m[1])));

const jgEmpty = (tags) => ({
  key: Math.random().toString(36).slice(2),
  ...jgGuessSymbol(tags),
  timeframe: '1d',
  directions: [],
  target_price: '',
  range_low: '',
  range_high: '',
  methods: [],
  levels: [],
  chart_url: '',
  reason: '',
});

// 資料庫列 → 編輯器狀態（數字轉字串，方便 input 編輯）
const jgFromRow = (r) => ({
  key: r.id,
  id: r.id,
  market: r.market,
  symbol: r.symbol || '',
  timeframe: r.timeframe,
  directions: r.directions || [],
  target_price: r.target_price ?? '',
  range_low: r.range_low ?? '',
  range_high: r.range_high ?? '',
  methods: r.methods || [],
  levels: (r.levels || []).map((l) => ({ type: l.type, price: l.price ?? '' })),
  chart_url: r.chart_url || '',
  reason: r.reason || '',
  locked_at: r.locked_at,
  withdrawn_at: r.withdrawn_at,
  created_at: r.created_at,
});

const jgToRow = (j, sort) => {
  const isRange = j.directions.includes('range');
  return {
    market: j.market,
    symbol: j.symbol.trim().toUpperCase(),
    timeframe: j.timeframe,
    directions: j.directions,
    target_price: jgNum(j.target_price),
    range_low: isRange ? jgNum(j.range_low) : null,
    range_high: isRange ? jgNum(j.range_high) : null,
    methods: j.methods,
    levels: j.levels.filter((l) => jgNum(l.price) != null).map((l) => ({ type: l.type, price: jgNum(l.price) })),
    chart_url: j.chart_url || null,
    reason: j.reason.trim() || null,
    sort,
  };
};

// 發佈前檢查；回傳錯誤字串或 null
const jgValidate = (j, n) => {
  if (!j.symbol.trim()) return `判讀 ${n}：請填標的`;
  if (!j.directions.length) return `判讀 ${n}：請選方向（看多／看空／盤整）`;
  for (const [k, lbl] of [['target_price', '目標價'], ['range_low', '區間下緣'], ['range_high', '區間上緣']]) {
    if (j[k] !== '' && jgNum(j[k]) == null) return `判讀 ${n}：${lbl}不是數字`;
  }
  if (j.directions.includes('range')) {
    const lo = jgNum(j.range_low), hi = jgNum(j.range_high);
    if (lo == null || hi == null || lo >= hi) return `判讀 ${n}：盤整需要填區間，且下緣要小於上緣`;
  }
  return null;
};

const withdrawJudgment = async (supabase, j) => {
  if (!window.confirm('撤回後會以撤回當下的價格結算，並標示為「已撤回」，無法恢復。確定撤回？')) return null;
  const { data, error } = await supabase
    .from('forum_judgments')
    .update({ withdrawn_at: new Date().toISOString() })
    .eq('id', j.id)
    .select()
    .single();
  if (error) { window.alert(`撤回失敗：${error.message}`); return null; }
  return data;
};

const JgDirChips = ({ directions }) => (
  <span className="inline-flex gap-1.5">
    {['up', 'down', 'range'].filter((d) => directions.includes(d)).map((d) => (
      <span key={d} className="fs-chip" style={{ color: JG_DIRS[d].color }}>{JG_DIRS[d].label}</span>
    ))}
  </span>
);

// 一則判讀的呈現：讀者看內容；管理員另外看到驗證結果與撤回
// showResult=false：編輯器裡只給撤回，不顯示驗證數字（編輯器不讀結果表）
const JudgmentCard = ({ j, result, isAdmin, onWithdraw, onOpenChart, showResult = true }) => {
  const tf = jgTimeframe(j.timeframe);
  const isRange = (j.directions || []).includes('range');
  const status = j.withdrawn_at ? 'withdrawn' : (result?.status || 'pending');
  const levels = j.levels || [];
  const methods = (j.methods || []).map((m) => (JG_METHODS.find((x) => x.v === m) || {}).label).filter(Boolean);

  return (
    <div className="py-4" style={{ borderTop: '1px solid var(--rule)', opacity: j.withdrawn_at && !isAdmin ? 0.72 : 1 }}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <JgDirChips directions={j.directions || []} />
          <span className="text-[16px] font-extrabold num" style={{ color: 'var(--ink)' }}>{j.symbol}</span>
          <span className="text-[13px]" style={{ color: 'var(--ink-2)' }}>{tf.label} · 驗證 {tf.bars} 根（{tf.span}）</span>
        </div>
        {(isAdmin || j.withdrawn_at) && (
          <span className="fs-chip" style={{ color: JG_STATUS[status].color }}>
            {JG_STATUS[status].label}{j.withdrawn_at ? ` · ${jgDate(j.withdrawn_at)}` : ''}
          </span>
        )}
      </div>

      <div className="fs-kv mt-3" style={{ borderTop: '1px solid var(--rule)' }}>
        <div>
          <div className="fs-lbl">目標價</div>
          <div className="text-[15px] font-bold num" style={{ color: 'var(--ink)' }}>{jgPrice(j.target_price)}</div>
        </div>
        <div>
          <div className="fs-lbl">盤整區間</div>
          <div className="text-[15px] font-bold num" style={{ color: 'var(--ink)' }}>
            {isRange ? `${jgPrice(j.range_low)} – ${jgPrice(j.range_high)}` : '—'}
          </div>
        </div>
        <div>
          <div className="fs-lbl">依據</div>
          <div className="text-[14px] font-bold" style={{ color: 'var(--ink)' }}>{methods.length ? methods.join('、') : '—'}</div>
        </div>
        <div>
          <div className="fs-lbl">判讀時間</div>
          <div className="text-[14px] font-bold num" style={{ color: 'var(--ink)' }}>{jgDate(j.locked_at || j.created_at)}</div>
        </div>
      </div>

      {levels.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
          <span className="fs-lbl">關鍵價位</span>
          {levels.map((l, i) => (
            <span key={i} style={{ color: 'var(--ink-2)' }}>
              {l.type} <b className="num" style={{ color: 'var(--ink)' }}>{jgPrice(l.price)}</b>
            </span>
          ))}
        </div>
      )}

      {(j.reason || j.chart_url) && (
        <div className="mt-3 flex gap-4 items-start">
          {j.reason && <p className="flex-1 text-[14px] leading-relaxed" style={{ color: 'var(--ink)' }}>{j.reason}</p>}
          {j.chart_url && (
            <button type="button" onClick={() => onOpenChart && onOpenChart(j.chart_url)} className="shrink-0" aria-label="放大判讀圖表" title="放大圖表">
              <img src={j.chart_url} alt="判讀圖表" className="w-32 h-20 md:w-40 md:h-24 object-cover" style={{ border: '1px solid var(--rule)' }} />
            </button>
          )}
        </div>
      )}

      {isAdmin && (showResult || !j.withdrawn_at) && (
        <div className="mt-3 pt-3 flex flex-wrap items-baseline justify-between gap-3" style={{ borderTop: '1px dashed var(--rule)' }}>
          {showResult ? (
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[13px]" style={{ color: 'var(--ink-2)' }}>
            <span className="fs-lbl">只有你看得到</span>
            <span>進場 <b className="num" style={{ color: 'var(--ink)' }}>{jgPrice(result?.entry_price)}</b></span>
            <span>進度 <b className="num" style={{ color: 'var(--ink)' }}>{result?.bars_elapsed ?? 0}/{result?.bars_total ?? tf.bars}</b> 根</span>
            <span>漲跌 <b className="num" style={{ color: result?.change_pct > 0 ? 'var(--up)' : result?.change_pct < 0 ? 'var(--down)' : 'var(--ink)' }}>{jgPct(result?.change_pct)}</b></span>
            {j.target_price != null && <span>最遠走到目標 <b className="num" style={{ color: 'var(--ink)' }}>{result?.max_progress == null ? '—' : `${Math.round(result.max_progress * 100)}%`}</b></span>}
            {result?.settled_at && <span>結算 <b className="num" style={{ color: 'var(--ink)' }}>{jgPrice(result.exit_price)}</b> · {jgDate(result.settled_at)}</span>}
            {result?.regime && <span>當時市場 <b style={{ color: 'var(--ink)' }}>{JG_REGIME[result.regime] || result.regime}</b></span>}
            {result?.vpvr && <span>VPVR <b className="num" style={{ color: 'var(--ink)' }}>POC {jgPrice(result.vpvr.poc)} · VAH {jgPrice(result.vpvr.vah)} · VAL {jgPrice(result.vpvr.val)}</b></span>}
            {result?.neutral_pct != null && <span>持平範圍 <b className="num" style={{ color: 'var(--ink)' }}>±{(result.neutral_pct * 100).toFixed(1)}%</b></span>}
            {j.backfilled && <span className="fs-chip" style={{ color: 'var(--ink-3)' }}>補登</span>}
            {(!result || !result.entry_price) && j.locked_at && !result?.note && <span>等待每日驗證</span>}
            {result?.note && <span style={{ color: 'var(--amber)' }}>{result.note}</span>}
          </div>
          ) : (
            <span className="fs-lbl">已發佈，判讀已鎖定</span>
          )}
          {onWithdraw && j.locked_at && !j.withdrawn_at && status === 'pending' && (
            <button type="button" onClick={() => onWithdraw(j)} className="fs-btn sm" style={{ color: 'var(--down)', borderColor: 'var(--down)' }}>
              撤回判讀
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// 文章頁頂端的判讀區塊
const JudgmentsSection = ({ supabase, postId, isAdmin, onOpenChart }) => {
  const [items, setItems] = React.useState([]);
  const [results, setResults] = React.useState({});

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data, error } = await supabase
        .from('forum_judgments')
        .select('*')
        .eq('post_id', postId)
        .order('sort', { ascending: true });
      if (cancelled || error || !data) return;   // 資料表尚未建立時靜默略過
      setItems(data);
      if (isAdmin && data.length) {
        const res = await supabase
          .from('forum_judgment_results')
          .select('*')
          .in('judgment_id', data.map((j) => j.id));
        if (cancelled || res.error || !res.data) return;
        setResults(Object.fromEntries(res.data.map((r) => [r.judgment_id, r])));
      }
    })();
    return () => { cancelled = true; };
  }, [supabase, postId, isAdmin]);

  if (!items.length) return null;

  const onWithdraw = async (j) => {
    const updated = await withdrawJudgment(supabase, j);
    if (updated) setItems((xs) => xs.map((x) => (x.id === updated.id ? updated : x)));
  };

  return (
    <section className="mb-6 fs-section">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="fs-title-sm">判讀</h2>
        <span className="fs-lbl">{items.length} 則 · 級別越大，驗證期間越長</span>
      </div>
      <div className="mt-2">
        {items.map((j) => (
          <JudgmentCard key={j.id} j={j} result={results[j.id]} isAdmin={isAdmin} onWithdraw={isAdmin ? onWithdraw : null} onOpenChart={onOpenChart} />
        ))}
      </div>
    </section>
  );
};

// 編輯器裡的判讀表單
const JgToggle = ({ on, onClick, children, color, disabled }) => (
  <button
    type="button"
    onClick={onClick}
    disabled={disabled}
    aria-pressed={on}
    className={`fs-btn sm${on ? ' solid' : ''}`}
    style={on && color ? { background: color, borderColor: color, color: 'var(--paper)' } : undefined}
  >
    {children}
  </button>
);

const JudgmentForm = ({ j, n, images, onChange, onRemove, disabled, inputStyle }) => {
  const set = (patch) => onChange({ ...j, ...patch });
  const toggleDir = (d) => {
    let dirs = j.directions.includes(d) ? j.directions.filter((x) => x !== d) : [...j.directions, d];
    if (d === 'up' && dirs.includes('up')) dirs = dirs.filter((x) => x !== 'down');
    if (d === 'down' && dirs.includes('down')) dirs = dirs.filter((x) => x !== 'up');
    const patch = { directions: dirs };
    // 勾盤整且區間空白時，先帶入已填的 VAL–VAH
    if (d === 'range' && dirs.includes('range') && j.range_low === '' && j.range_high === '') {
      const val = j.levels.find((l) => l.type === 'VAL' && jgNum(l.price) != null);
      const vah = j.levels.find((l) => l.type === 'VAH' && jgNum(l.price) != null);
      if (val) patch.range_low = val.price;
      if (vah) patch.range_high = vah.price;
    }
    set(patch);
  };
  const toggleMethod = (m) => set({ methods: j.methods.includes(m) ? j.methods.filter((x) => x !== m) : [...j.methods, m] });
  const setLevel = (i, patch) => set({ levels: j.levels.map((l, k) => (k === i ? { ...l, ...patch } : l)) });
  const tf = jgTimeframe(j.timeframe);
  const market = JG_MARKETS.find((m) => m.v === j.market) || JG_MARKETS[0];
  const field = 'rounded-xl px-3 py-2 text-sm outline-none num';

  return (
    <div className="py-4 space-y-4" style={{ borderTop: '1px solid var(--rule)' }}>
      <div className="flex items-baseline justify-between">
        <span className="text-[14px] font-extrabold" style={{ color: 'var(--ink)' }}>判讀 {n}</span>
        <button type="button" onClick={onRemove} disabled={disabled} className="text-[13px]" style={{ color: 'var(--down)' }}>刪除</button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[auto_1fr] gap-x-5 gap-y-3 items-center">
        <span className="fs-lbl">標的</span>
        <div className="flex flex-wrap gap-2">
          <select value={j.market} onChange={(e) => set({ market: e.target.value })} disabled={disabled} className="rounded-xl px-3 py-2 text-sm outline-none" style={inputStyle} aria-label="市場">
            {JG_MARKETS.map((m) => <option key={m.v} value={m.v}>{m.label}</option>)}
          </select>
          <input value={j.symbol} onChange={(e) => set({ symbol: e.target.value })} disabled={disabled} placeholder={market.ph} className={`${field} w-32 uppercase`} style={inputStyle} aria-label="標的代號" />
        </div>

        <span className="fs-lbl">級別</span>
        <div className="flex flex-wrap items-center gap-2">
          {JG_TIMEFRAMES.map((t) => (
            <JgToggle key={t.v} on={j.timeframe === t.v} onClick={() => set({ timeframe: t.v })} disabled={disabled}>{t.label}</JgToggle>
          ))}
          <span className="fs-lbl ml-1">驗證 {tf.bars} 根 K 棒（{tf.span}）</span>
        </div>

        <span className="fs-lbl">方向</span>
        <div className="flex flex-wrap items-center gap-2">
          {['up', 'down', 'range'].map((d) => (
            <JgToggle key={d} on={j.directions.includes(d)} color={d === 'range' ? null : JG_DIRS[d].color} onClick={() => toggleDir(d)} disabled={disabled}>{JG_DIRS[d].label}</JgToggle>
          ))}
          <span className="fs-lbl ml-1">盤整可以和看多或看空一起選</span>
        </div>

        <span className="fs-lbl">目標價</span>
        <div className="flex flex-wrap items-center gap-2">
          <input value={j.target_price} onChange={(e) => set({ target_price: e.target.value })} disabled={disabled} inputMode="decimal" placeholder="建議填寫" className={`${field} w-40`} style={inputStyle} aria-label="目標價" />
          <span className="fs-lbl">驗證期間內碰到就算達標</span>
        </div>

        {j.directions.includes('range') && (
          <>
            <span className="fs-lbl">盤整區間 *</span>
            <div className="flex flex-wrap items-center gap-2">
              <input value={j.range_low} onChange={(e) => set({ range_low: e.target.value })} disabled={disabled} inputMode="decimal" placeholder="下緣（VAL）" className={`${field} w-36`} style={inputStyle} aria-label="區間下緣" />
              <span style={{ color: 'var(--ink-3)' }}>–</span>
              <input value={j.range_high} onChange={(e) => set({ range_high: e.target.value })} disabled={disabled} inputMode="decimal" placeholder="上緣（VAH）" className={`${field} w-36`} style={inputStyle} aria-label="區間上緣" />
            </div>
          </>
        )}

        <span className="fs-lbl">依據</span>
        <div className="flex flex-wrap gap-2">
          {JG_METHODS.map((m) => (
            <JgToggle key={m.v} on={j.methods.includes(m.v)} onClick={() => toggleMethod(m.v)} disabled={disabled}>{m.label}</JgToggle>
          ))}
        </div>

        <span className="fs-lbl self-start pt-2">關鍵價位</span>
        <div className="space-y-2">
          {j.levels.map((l, i) => (
            <div key={i} className="flex flex-wrap items-center gap-2">
              <select value={l.type} onChange={(e) => setLevel(i, { type: e.target.value })} disabled={disabled} className="rounded-xl px-3 py-2 text-sm outline-none" style={inputStyle} aria-label="價位類型">
                {JG_LEVELS.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
              <input value={l.price} onChange={(e) => setLevel(i, { price: e.target.value })} disabled={disabled} inputMode="decimal" placeholder="價格" className={`${field} w-36`} style={inputStyle} aria-label="價位" />
              <button type="button" onClick={() => set({ levels: j.levels.filter((_, k) => k !== i) })} disabled={disabled} className="fs-btn icon" aria-label="移除價位" title="移除">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
              </button>
            </div>
          ))}
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => set({ levels: [...j.levels, { type: '阻力', price: '' }] })} disabled={disabled} className="fs-btn sm">＋ 價位</button>
            <span className="fs-lbl">選填，只記錄不計分</span>
          </div>
        </div>

        <span className="fs-lbl self-start pt-2">圖表</span>
        <div>
          {images.length ? (
            <div className="flex flex-wrap gap-2">
              {images.map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => set({ chart_url: j.chart_url === src ? '' : src })}
                  disabled={disabled}
                  aria-pressed={j.chart_url === src}
                  title={j.chart_url === src ? '取消選擇' : '用這張當判讀圖表'}
                  style={{ outline: j.chart_url === src ? '3px solid var(--accent)' : '1px solid var(--rule)', outlineOffset: j.chart_url === src ? 1 : 0 }}
                >
                  <img src={src} alt="" className="w-24 h-16 object-cover block" />
                </button>
              ))}
            </div>
          ) : (
            <p className="fs-lbl pt-2">先在文章裡貼上圖表截圖，就能在這裡挑一張</p>
          )}
        </div>

        <span className="fs-lbl">一句話理由</span>
        <input value={j.reason} onChange={(e) => set({ reason: e.target.value })} disabled={disabled} maxLength={140} placeholder="例如：週線 Spring 後回測不破，量縮" className="rounded-xl px-3 py-2 text-sm outline-none w-full" style={inputStyle} aria-label="一句話理由" />
      </div>
    </div>
  );
};

const JudgmentsEditor = ({ items, setItems, noJudgment, setNoJudgment, images, tags, disabled, inputStyle, onWithdraw }) => {
  const drafts = items.filter((j) => !j.locked_at);
  const locked = items.filter((j) => j.locked_at);
  return (
    <div className="fs-section">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="fs-title-sm">判讀</h3>
        <span className="fs-lbl">發佈前至少一則；發佈後鎖定，只能撤回</span>
      </div>

      {locked.map((j) => (
        <JudgmentCard key={j.key} j={j} result={null} isAdmin showResult={false} onWithdraw={onWithdraw} />
      ))}

      {drafts.map((j) => (
        <JudgmentForm
          key={j.key}
          j={j}
          n={items.indexOf(j) + 1}
          images={images}
          disabled={disabled}
          inputStyle={inputStyle}
          onChange={(next) => setItems((xs) => xs.map((x) => (x.key === j.key ? next : x)))}
          onRemove={() => setItems((xs) => xs.filter((x) => x.key !== j.key))}
        />
      ))}

      <div className="pt-4 flex flex-wrap items-center gap-4" style={{ borderTop: '1px solid var(--rule)' }}>
        {!noJudgment && (
          <button type="button" onClick={() => setItems((xs) => [...xs, jgEmpty(tags)])} disabled={disabled} className="fs-btn sm">
            ＋ 新增判讀
          </button>
        )}
        {items.length === 0 && (
          <label className="inline-flex items-center gap-2 text-[14px] cursor-pointer" style={{ color: 'var(--ink-2)' }}>
            <input type="checkbox" checked={noJudgment} onChange={(e) => setNoJudgment(e.target.checked)} disabled={disabled} />
            本篇不含判讀（公告、心得等）
          </label>
        )}
      </div>
    </div>
  );
};

const PostDetail = ({ supabase, postId, isAdmin, onBack }) => {
  const [post, setPost] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [lightbox, setLightbox] = React.useState(null); // { images: [src,...], index }
  const [showScrollTop, setShowScrollTop] = React.useState(false);
  const contentRef = React.useRef(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setLightbox(null);  // Clear lightbox when navigating to a different post
      window.scrollTo(0, 0);  // Always start a new post at the top
      const { data, error: err } = await supabase
        .from('forum_posts')
        .select('*')
        .eq('id', postId)
        .single();
      if (cancelled) return;
      if (err) setError(err.message);
      else setPost(data);
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [supabase, postId]);

  // Scroll-to-top button visibility: appears after scrolling ~one screen
  React.useEffect(() => {
    const onScroll = () => setShowScrollTop(window.scrollY > 400);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();  // Sync initial state
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const scrollToTop = () => window.scrollTo({ top: 0, behavior: 'smooth' });

  React.useEffect(() => {
    if (!post || !contentRef.current) return;
    if (window.hljs) {
      contentRef.current.querySelectorAll('pre code').forEach((el) => {
        try { window.hljs.highlightElement(el); } catch (_) {}
      });
    }
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(contentRef.current, {
          delimiters: [
            { left: '$$',  right: '$$',  display: true },
            { left: '\\[', right: '\\]', display: true },
            { left: '$',   right: '$',   display: false },
            { left: '\\(', right: '\\)', display: false },
          ],
          throwOnError: false,
          errorColor: '#f87171',
        });
      } catch (_) {}
    }

    // Wire up image clicks → open lightbox
    // (Runs after hljs/KaTeX so DOM is stable.)
    const imgs = Array.from(contentRef.current.querySelectorAll('img'));
    const srcs = imgs.map((img) => img.src);
    imgs.forEach((img, i) => {
      img.style.cursor = 'zoom-in';
      img.style.transition = 'opacity 0.15s ease';
      img.addEventListener('mouseenter', () => { img.style.opacity = '0.92'; });
      img.addEventListener('mouseleave', () => { img.style.opacity = '1'; });
      img.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setLightbox({ images: srcs, index: i });
      };
    });
  }, [post]);

  const handleDelete = async () => {
    if (!window.confirm(`確定要刪除「${post.title}」嗎？無法復原。`)) return;
    const { error: err } = await supabase
      .from('forum_posts')
      .delete()
      .eq('id', postId);
    if (err) { window.alert(`刪除失敗：${err.message}`); return; }
    navigate('#/forum');
  };

  if (loading) return (
    <div className="text-center py-16">
      <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'var(--rule)', borderTopColor: 'var(--ink)' }}></div>
    </div>
  );
  if (error || !post) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="text-sm flex items-center gap-1.5 transition-colors" style={{ color: 'var(--accent)' }}>
          <span>←</span> 返回列表
        </button>
        <div className="text-sm rounded-xl p-4" style={{ background: 'rgba(255,91,110,0.08)', border: '1px solid rgba(255,91,110,0.2)', color: 'var(--down)' }}>
          {error || '文章不存在'}
        </div>
      </div>
    );
  }

  const html = renderMarkdownToHtml(post.content);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <button onClick={onBack} className="text-sm flex items-center gap-1.5 transition-colors hover:text-white" style={{ color: 'var(--text-2)' }}>
          <span>←</span> 返回列表
        </button>
        {isAdmin && (
          <div className="flex gap-2">
            <button
              onClick={() => navigate(`#/forum/${postId}/edit`)}
              className="px-3 py-1.5 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
              style={{ background: 'transparent', border: '1.5px solid var(--ink)', color: 'var(--ink)' }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" /></svg>
              編輯
            </button>
            <button
              onClick={handleDelete}
              className="px-3 py-1.5 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
              style={{ background: 'transparent', border: '1.5px solid var(--down)', color: 'var(--down)' }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-2 14a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
              刪除
            </button>
          </div>
        )}
      </div>

      <article className="rounded-2xl ring-soft p-6 md:p-8" style={{ background: 'var(--surface)' }}>
        <header className="mb-6 pb-5 border-b" style={{ borderColor: 'var(--line)' }}>
          <div className="flex items-start justify-between gap-3 mb-3">
            <h1 className="text-3xl font-extrabold text-white leading-tight tracking-tight">{post.title}</h1>
            {!post.published && (
              <span className="chip chip-warn whitespace-nowrap shrink-0">DRAFT</span>
            )}
          </div>
          {post.tags && post.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {post.tags.map((t) => (
                <span key={t} className="text-[11px] px-2 py-0.5 rounded-md mono" style={{ background: 'var(--wash)', color: 'var(--text-2)' }}>
                  #{t}
                </span>
              ))}
            </div>
          )}
          <p className="text-xs mono" style={{ color: 'var(--text-3)' }}>
            {new Date(post.created_at).toLocaleString('zh-TW')}
            {post.updated_at && post.updated_at !== post.created_at && (
              <span className="ml-2">· 編輯於 {new Date(post.updated_at).toLocaleString('zh-TW')}</span>
            )}
          </p>
        </header>

        <JudgmentsSection
          supabase={supabase}
          postId={postId}
          isAdmin={isAdmin}
          onOpenChart={(src) => setLightbox({ images: [src], index: 0 })}
        />

        <div
          ref={contentRef}
          className="forum-md"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </article>

      {lightbox && (
        <ImageLightbox
          images={lightbox.images}
          index={lightbox.index}
          onClose={() => setLightbox(null)}
          onPrev={() => setLightbox((lb) => ({
            ...lb,
            index: (lb.index - 1 + lb.images.length) % lb.images.length
          }))}
          onNext={() => setLightbox((lb) => ({
            ...lb,
            index: (lb.index + 1) % lb.images.length
          }))}
        />
      )}

      {/* Scroll-to-top — fades in after scrolling ~one screen */}
      <button
        onClick={scrollToTop}
        aria-label="回到頂部"
        className="fixed bottom-6 right-6 w-12 h-12 rounded-full flex items-center justify-center text-white text-xl z-50 hover:scale-110"
        style={{
          background: 'rgba(15,23,42,0.88)',
          border: '1px solid var(--rule)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
          opacity: showScrollTop ? 1 : 0,
          pointerEvents: showScrollTop ? 'auto' : 'none',
          transform: showScrollTop ? 'translateY(0)' : 'translateY(20px)',
          transition: 'opacity 0.25s ease, transform 0.25s ease',
        }}
      >
        ↑
      </button>
    </div>
  );
};

const PostEditor = ({ supabase, user, mode, postId, onCancel }) => {
  const [title, setTitle] = React.useState('');
  const [tagsInput, setTagsInput] = React.useState('');
  const [origPublished, setOrigPublished] = React.useState(false);
  const [loading, setLoading] = React.useState(mode === 'edit');
  const [saving, setSaving] = React.useState(false);
  const [uploadingImage, setUploadingImage] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [judgments, setJudgments] = React.useState([]);
  const [noJudgment, setNoJudgment] = React.useState(false);
  const [contentImages, setContentImages] = React.useState([]);
  const [loadedJudgmentIds, setLoadedJudgmentIds] = React.useState([]);

  const textareaRef = React.useRef(null);
  const easymdeRef  = React.useRef(null);
  // 新文章第一次存檔後記住 id，重試時改成更新，不會重複建立
  const savedIdRef  = React.useRef(mode === 'edit' ? postId : null);

  React.useEffect(() => {
    if (mode !== 'edit' || !postId) return;
    let cancelled = false;
    (async () => {
      const { data, error: err } = await supabase
        .from('forum_posts')
        .select('*')
        .eq('id', postId)
        .single();
      if (cancelled) return;
      if (err) { setError(err.message); setLoading(false); return; }
      setTitle(data.title || '');
      setTagsInput((data.tags || []).join(', '));
      setOrigPublished(!!data.published);
      setNoJudgment(!!data.no_judgment);
      const jr = await supabase
        .from('forum_judgments')
        .select('*')
        .eq('post_id', postId)
        .order('sort', { ascending: true });
      if (cancelled) return;
      if (!jr.error && jr.data) {
        setJudgments(jr.data.map(jgFromRow));
        setLoadedJudgmentIds(jr.data.filter((r) => !r.locked_at).map((r) => r.id));
      }
      if (easymdeRef.current) easymdeRef.current.value(data.content || '');
      else window.__pendingEditorContent = data.content || '';
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [mode, postId, supabase]);

  React.useEffect(() => {
    if (!textareaRef.current || easymdeRef.current || !window.EasyMDE) return;

    const insertAtCursor = (markdown) => {
      const cm = easymdeRef.current.codemirror;
      const doc = cm.getDoc();
      doc.replaceSelection(markdown);
      cm.focus();
    };

    const handleImageFile = async (file) => {
      setUploadingImage(true);
      setError(null);
      try {
        const url = await uploadImageToStorage(supabase, file);
        insertAtCursor(`\n![](${url})\n`);
      } catch (e) {
        setError(`圖片上傳失敗：${e.message}`);
      } finally {
        setUploadingImage(false);
      }
    };

    easymdeRef.current = new window.EasyMDE({
      element: textareaRef.current,
      autoDownloadFontAwesome: true,
      spellChecker: false,
      status: ['lines', 'words'],
      placeholder: '在這裡寫文章... 可以 Ctrl+V 直接貼截圖，或把圖片拖進來。\n\n支援：\n  # 標題、**粗體**、*斜體*、- 清單\n  ```python ... ``` 程式碼區塊\n  $E=mc^2$ LaTeX 公式\n',
      toolbar: [
        'bold', 'italic', 'heading', '|',
        'quote', 'unordered-list', 'ordered-list', '|',
        'link', 'image', 'table', 'code', '|',
        'preview', 'side-by-side', 'fullscreen', '|',
        'guide',
      ],
      previewRender: (plainText) => renderMarkdownToHtml(plainText),
      uploadImage: true,
      imageUploadFunction: async (file, onSuccess, onError) => {
        try {
          setUploadingImage(true);
          const url = await uploadImageToStorage(supabase, file);
          setUploadingImage(false);
          onSuccess(url);
        } catch (e) {
          setUploadingImage(false);
          onError(e.message);
        }
      },
    });

    const syncImages = () => {
      if (!easymdeRef.current) return;
      const next = jgExtractImages(easymdeRef.current.value());
      setContentImages((prev) => (prev.join('|') === next.join('|') ? prev : next));
    };
    easymdeRef.current.codemirror.on('change', syncImages);

    if (window.__pendingEditorContent != null) {
      easymdeRef.current.value(window.__pendingEditorContent);
      delete window.__pendingEditorContent;
    }
    syncImages();

    easymdeRef.current.codemirror.on('paste', (cm, e) => {
      const items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      for (const item of items) {
        if (item.kind === 'file' && item.type.startsWith('image/')) {
          e.preventDefault();
          const file = item.getAsFile();
          if (file) handleImageFile(file);
          return;
        }
      }
    });

    return () => {
      if (easymdeRef.current) {
        try { easymdeRef.current.toTextArea(); } catch (_) {}
        easymdeRef.current = null;
      }
    };
  }, [supabase, loading]);

  const handleSave = async (publishedFlag) => {
    if (!title.trim()) {
      setError('標題不能空白');
      return;
    }
    const content = easymdeRef.current ? easymdeRef.current.value() : '';
    const tagsArray = tagsInput
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

    const drafts = judgments.filter((jg) => !jg.locked_at);
    for (const jg of drafts) {
      const msg = jgValidate(jg, judgments.indexOf(jg) + 1);
      if (msg) { setError(msg); return; }
    }
    if (publishedFlag && judgments.length === 0 && !noJudgment) {
      setError('發佈前請至少新增一則判讀，或勾選「本篇不含判讀」');
      return;
    }

    setSaving(true);
    setError(null);

    const wasPublished = mode === 'edit' && origPublished;
    const payload = {
      title: title.trim(),
      content,
      tags: tagsArray,
      no_judgment: judgments.length === 0 && noJudgment,
      // 尚未發佈的文章先存成草稿，判讀寫完再發佈（發佈時資料庫會鎖定判讀）
      published: wasPublished && publishedFlag,
    };

    const fail = (msg) => { setSaving(false); setError(msg); };

    // 1. 文章本體
    let result;
    if (savedIdRef.current) {
      result = await supabase
        .from('forum_posts')
        .update(payload)
        .eq('id', savedIdRef.current)
        .select()
        .single();
    } else {
      result = await supabase
        .from('forum_posts')
        .insert({ ...payload, author_id: user.id })
        .select()
        .single();
    }
    if (result.error) return fail(result.error.message);
    const id = result.data.id;
    savedIdRef.current = id;

    // 2. 判讀：刪掉被移除的草稿判讀，更新／新增其餘（已鎖定的不動）
    const keptIds = new Set(drafts.map((jg) => jg.id).filter(Boolean));
    const removed = loadedJudgmentIds.filter((jid) => !keptIds.has(jid));
    if (removed.length) {
      const del = await supabase.from('forum_judgments').delete().in('id', removed);
      if (del.error) return fail(`判讀刪除失敗：${del.error.message}`);
      setLoadedJudgmentIds((ids) => ids.filter((jid) => !removed.includes(jid)));
    }
    for (const jg of drafts) {
      const row = jgToRow(jg, judgments.indexOf(jg));
      const res = jg.id
        ? await supabase.from('forum_judgments').update(row).eq('id', jg.id).select().single()
        : await supabase.from('forum_judgments').insert({ ...row, post_id: id }).select().single();
      if (res.error) return fail(`判讀儲存失敗：${res.error.message}`);
      if (!jg.id) {
        setJudgments((xs) => xs.map((x) => (x.key === jg.key ? { ...x, id: res.data.id } : x)));
        setLoadedJudgmentIds((ids) => [...ids, res.data.id]);
        jg.id = res.data.id;
      }
    }

    // 3. 發佈
    // 排程 notify_new_posts.py 靠 notification_sent 決定要推播哪幾篇。
    // 草稿第一次轉為發佈時把旗標歸零,否則那篇文章永遠不會進推播佇列。
    if (publishedFlag && !wasPublished) {
      const pub = await supabase
        .from('forum_posts')
        .update({ published: true, notification_sent: false })
        .eq('id', id)
        .select()
        .single();
      if (pub.error) return fail(`文章已存成草稿，但發佈失敗：${pub.error.message}`);
    }

    setSaving(false);
    navigate(`#/forum/${id}`);
  };

  const onWithdrawInEditor = async (jg) => {
    const updated = await withdrawJudgment(supabase, jg);
    if (updated) setJudgments((xs) => xs.map((x) => (x.id === updated.id ? { ...x, withdrawn_at: updated.withdrawn_at } : x)));
  };

  const inputStyle = { background: 'var(--wash)', border: '1px solid var(--line)', color: 'var(--text)' };

  if (loading) return (
    <div className="text-center py-16">
      <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'var(--rule)', borderTopColor: 'var(--ink)' }}></div>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="label">{mode === 'edit' ? 'EDIT POST' : 'NEW POST'}</div>
          <h2 className="text-2xl font-extrabold text-white mt-1">
            {mode === 'edit' ? '編輯文章' : '撰寫新文章'}
          </h2>
        </div>
        <button
          onClick={onCancel}
          className="text-sm transition-colors hover:text-white"
          style={{ color: 'var(--text-2)' }}
          disabled={saving}
        >
          取消
        </button>
      </div>

      {error && (
        <div className="text-sm rounded-xl p-3" style={{ background: 'rgba(255,91,110,0.08)', border: '1px solid rgba(255,91,110,0.2)', color: 'var(--down)' }}>
          {error}
        </div>
      )}
      {uploadingImage && (
        <div className="text-sm rounded-xl p-3 flex items-center gap-2" style={{ background: 'var(--wash)', border: '1px solid var(--rule)', color: 'var(--accent)' }}>
          <div className="w-3 h-3 rounded-full border-2 animate-spin" style={{ borderColor: 'var(--rule)', borderTopColor: 'var(--accent)' }}></div>
          上傳圖片中...
        </div>
      )}

      <div className="rounded-2xl ring-soft p-5 space-y-4" style={{ background: 'var(--surface)' }}>
        <div>
          <label className="label block mb-1.5">標題 *</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
            style={inputStyle}
            placeholder="例如：BTC 短期觀察"
            disabled={saving}
            onFocus={(e) => e.target.style.borderColor = 'var(--ink)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--line)'}
          />
        </div>
        <div>
          <label className="label block mb-1.5">標籤 (用逗號分隔)</label>
          <input
            type="text"
            value={tagsInput}
            onChange={(e) => setTagsInput(e.target.value)}
            className="w-full rounded-xl px-3.5 py-2.5 text-sm text-white outline-none transition-colors"
            style={inputStyle}
            placeholder="BTC, 技術分析, RSI"
            disabled={saving}
            onFocus={(e) => e.target.style.borderColor = 'var(--ink)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--line)'}
          />
        </div>
      </div>

      <div>
        <textarea ref={textareaRef} />
      </div>

      <JudgmentsEditor
        items={judgments}
        setItems={setJudgments}
        noJudgment={noJudgment}
        setNoJudgment={setNoJudgment}
        images={contentImages}
        tags={tagsInput.split(',').map((t) => t.trim()).filter(Boolean)}
        disabled={saving}
        inputStyle={inputStyle}
        onWithdraw={onWithdrawInEditor}
      />

      <div className="flex flex-wrap gap-2 justify-end">
        <button
          onClick={onCancel}
          disabled={saving}
          className="px-4 py-2.5 rounded-xl text-white text-sm font-semibold transition-colors disabled:opacity-50"
          style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid var(--line)' }}
        >
          取消
        </button>
        <button
          onClick={() => handleSave(false)}
          disabled={saving}
          className="px-4 py-2.5 rounded-xl text-sm font-bold transition-all disabled:opacity-50"
          style={{ background: 'transparent', border: '1.5px solid var(--amber)', color: 'var(--amber)' }}
        >
          {saving ? '儲存中...' : '儲存為草稿'}
        </button>
        <button
          onClick={() => handleSave(true)}
          disabled={saving}
          className="fs-btn solid disabled:opacity-50"
        >
          {saving ? '發佈中...' : (origPublished && mode === 'edit' ? '更新並發佈' : '發佈文章')}
        </button>
      </div>
    </div>
  );
};

const LoginRequiredView = () => (
  <div className="rounded-2xl ring-soft p-10 text-center relative overflow-hidden" style={{ background: 'var(--surface)' }}>
    <div className="absolute inset-0 dotgrid opacity-40 pointer-events-none"></div>
    <div className="relative">
      <div className="w-14 h-14 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'var(--wash)', border: '1px solid var(--line)' }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--text-2)' }}>
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
          <path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
      </div>
      <h3 className="text-xl font-bold text-white mb-2">請先登入</h3>
      <p className="text-sm mb-4" style={{ color: 'var(--text-2)' }}>
        技術分析論壇需要登入會員才能查看
      </p>
      <p className="text-xs mono" style={{ color: 'var(--text-3)' }}>
        登入後請聯絡作者開通會員權限
      </p>
    </div>
  </div>
);

const PremiumRequiredView = ({ user }) => (
  <div className="rounded-2xl p-10 text-center relative overflow-hidden" style={{ background: 'transparent', borderTop: '3px solid var(--ink)', borderRadius: 0 }}>
    <div className="absolute inset-0 dotgrid opacity-30 pointer-events-none"></div>
    <div className="relative">
      <div className="w-14 h-14 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'var(--ink)', color: 'var(--paper)' }}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
          <path d="M12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z" />
        </svg>
      </div>
      <h3 className="text-xl font-extrabold text-white mb-2">會員專屬內容</h3>
      <p className="text-sm mb-1" style={{ color: 'var(--text)' }}>
        技術分析論壇是付費會員專屬功能
      </p>
      <p className="text-sm mb-5" style={{ color: 'var(--text-2)' }}>
        升級後解鎖所有技術分析文章與圖表觀察
      </p>
      <div className="inline-block rounded-xl px-4 py-3 ring-soft" style={{ background: 'rgba(7,8,12,0.5)' }}>
        <p className="label mb-1">當前帳號</p>
        <p className="text-sm font-semibold text-white mono">{user?.email}</p>
        <p className="text-[10px] mt-2 mono" style={{ color: 'var(--text-3)' }}>請聯絡作者開通會員</p>
      </div>
    </div>
  </div>
);

const ForumApp = ({ supabase, user, isAdmin, isPremium }) => {
  const hash = useHashRoute();
  const route = parseForumRoute(hash);

  if (!supabase) return <p style={{ color: 'var(--down)' }}>Supabase 未初始化</p>;

  const isEditorRoute = route.view === 'editor';
  if (isEditorRoute && !isAdmin) {
    navigate('#/forum');
    return null;
  }

  const canRead = isAdmin || isPremium;

  const headerBadge = (() => {
    if (isAdmin)   return { txt: 'ADMIN',   style: { background: 'transparent', color: 'var(--up)', border: '1.5px solid var(--up)', borderRadius: 0 } };
    if (isPremium) return { txt: 'MEMBER',  style: { background: 'transparent', color: 'var(--ink)', border: '1.5px solid var(--ink)', borderRadius: 0 } };
    if (user)      return { txt: 'LOCKED',  style: { background: 'var(--wash)', color: 'var(--text-3)', border: '1px solid var(--line)' } };
    return            { txt: 'GUEST',    style: { background: 'var(--wash)', color: 'var(--text-3)', border: '1px solid var(--line)' } };
  })();

  return (
    <div className="space-y-4">
      <div className="rounded-2xl ring-soft p-5 flex items-center justify-between relative overflow-hidden" style={{ background: 'transparent' }}>
        <div className="relative flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl flex items-center justify-center" style={{ background: 'var(--ink)', color: 'var(--paper)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white tracking-tight">技術分析論壇</h2>
            <p className="text-xs mt-0.5 mono" style={{ color: 'var(--text-3)' }}>
              {user ? user.email : '未登入'}
            </p>
          </div>
        </div>
        <span className="text-[10px] font-bold tracking-widest px-3 py-1.5 rounded-full mono" style={headerBadge.style}>
          {headerBadge.txt}
        </span>
      </div>

      {!user && <LoginRequiredView />}
      {user && !canRead && <PremiumRequiredView user={user} />}

      {canRead && route.view === 'list' && (
        <PostList
          supabase={supabase}
          isAdmin={isAdmin}
          onOpen={(id) => navigate(`#/forum/${id}`)}
        />
      )}
      {canRead && route.view === 'detail' && (
        <PostDetail
          supabase={supabase}
          postId={route.postId}
          isAdmin={isAdmin}
          onBack={() => navigate('#/forum')}
        />
      )}
      {canRead && route.view === 'editor' && (
        <PostEditor
          key={route.mode === 'edit' ? route.postId : 'new'}
          supabase={supabase}
          user={user}
          mode={route.mode}
          postId={route.postId}
          onCancel={() => navigate(route.mode === 'edit' ? `#/forum/${route.postId}` : '#/forum')}
        />
      )}
    </div>
  );
};

window.ForumApp = ForumApp;
