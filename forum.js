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
            <span key={t} className="text-[11px] px-2 py-0.5 rounded-md mono" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-2)' }}>
              #{t}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-1.5 text-[11px] mono" style={{ color: 'var(--text-3)' }}>
          <span>{dateStr}</span>
          <span style={{ color: 'rgba(255,255,255,0.15)' }}>·</span>
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

  const inputStyle = { background: 'rgba(255,255,255,0.04)', border: '1px solid var(--line)', color: 'var(--text)' };

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
              onFocus={(e) => e.target.style.borderColor = 'rgba(139,92,246,0.5)'}
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
            className="px-4 py-2.5 rounded-xl text-white text-sm font-bold transition-all flex items-center gap-2 whitespace-nowrap glow-brand"
            style={{ background: 'linear-gradient(135deg,#8b5cf6,#ec4899)' }}
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
                  ? { background: 'linear-gradient(135deg,#8b5cf6,#ec4899)', color: '#fff' }
                  : { background: 'rgba(255,255,255,0.04)', color: 'var(--text-2)', border: '1px solid var(--line)' }
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
          {filtersActive && <span className="ml-2" style={{ color: '#c4b5fd' }}>· 已套用篩選</span>}
        </p>
      )}

      {loading && (
        <div className="text-center py-16">
          <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: '#8b5cf6' }}></div>
          <p className="mt-3 text-sm mono" style={{ color: 'var(--text-3)' }}>LOADING POSTS...</p>
        </div>
      )}
      {error && (
        <div className="text-sm rounded-xl p-4" style={{ background: 'rgba(255,91,110,0.08)', border: '1px solid rgba(255,91,110,0.2)', color: '#ff7d8c' }}>
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
            style={{ color: '#c4b5fd' }}
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

const PostDetail = ({ supabase, postId, isAdmin, onBack }) => {
  const [post, setPost] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [lightbox, setLightbox] = React.useState(null); // { images: [src,...], index }
  const contentRef = React.useRef(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setLightbox(null);  // Clear lightbox when navigating to a different post
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
      <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: '#8b5cf6' }}></div>
    </div>
  );
  if (error || !post) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="text-sm flex items-center gap-1.5 transition-colors" style={{ color: '#c4b5fd' }}>
          <span>←</span> 返回列表
        </button>
        <div className="text-sm rounded-xl p-4" style={{ background: 'rgba(255,91,110,0.08)', border: '1px solid rgba(255,91,110,0.2)', color: '#ff7d8c' }}>
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
              style={{ background: 'rgba(59,130,246,0.18)', border: '1px solid rgba(59,130,246,0.4)', color: '#7eb6ff' }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" /></svg>
              編輯
            </button>
            <button
              onClick={handleDelete}
              className="px-3 py-1.5 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
              style={{ background: 'rgba(255,91,110,0.12)', border: '1px solid rgba(255,91,110,0.3)', color: '#ff7d8c' }}
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
                <span key={t} className="text-[11px] px-2 py-0.5 rounded-md mono" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-2)' }}>
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

  const textareaRef = React.useRef(null);
  const easymdeRef  = React.useRef(null);

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

    if (window.__pendingEditorContent != null) {
      easymdeRef.current.value(window.__pendingEditorContent);
      delete window.__pendingEditorContent;
    }

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

    setSaving(true);
    setError(null);

    const payload = {
      title: title.trim(),
      content,
      tags: tagsArray,
      published: publishedFlag,
    };

    let result;
    if (mode === 'edit') {
      result = await supabase
        .from('forum_posts')
        .update(payload)
        .eq('id', postId)
        .select()
        .single();
    } else {
      result = await supabase
        .from('forum_posts')
        .insert({ ...payload, author_id: user.id })
        .select()
        .single();
    }

    setSaving(false);
    if (result.error) {
      setError(result.error.message);
      return;
    }
    navigate(`#/forum/${result.data.id}`);
  };

  const inputStyle = { background: 'rgba(255,255,255,0.04)', border: '1px solid var(--line)', color: 'var(--text)' };

  if (loading) return (
    <div className="text-center py-16">
      <div className="inline-block w-8 h-8 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(255,255,255,0.1)', borderTopColor: '#8b5cf6' }}></div>
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
        <div className="text-sm rounded-xl p-3" style={{ background: 'rgba(255,91,110,0.08)', border: '1px solid rgba(255,91,110,0.2)', color: '#ff7d8c' }}>
          {error}
        </div>
      )}
      {uploadingImage && (
        <div className="text-sm rounded-xl p-3 flex items-center gap-2" style={{ background: 'rgba(76,194,255,0.08)', border: '1px solid rgba(76,194,255,0.2)', color: '#7eb6ff' }}>
          <div className="w-3 h-3 rounded-full border-2 animate-spin" style={{ borderColor: 'rgba(76,194,255,0.3)', borderTopColor: '#7eb6ff' }}></div>
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
            onFocus={(e) => e.target.style.borderColor = 'rgba(139,92,246,0.5)'}
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
            onFocus={(e) => e.target.style.borderColor = 'rgba(139,92,246,0.5)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--line)'}
          />
        </div>
      </div>

      <div>
        <textarea ref={textareaRef} />
      </div>

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
          style={{ background: 'rgba(245,158,11,0.18)', border: '1px solid rgba(245,158,11,0.4)', color: '#fbbf24' }}
        >
          {saving ? '儲存中...' : '儲存為草稿'}
        </button>
        <button
          onClick={() => handleSave(true)}
          disabled={saving}
          className="px-5 py-2.5 rounded-xl text-white text-sm font-bold transition-all disabled:opacity-50 glow-brand"
          style={{ background: 'linear-gradient(135deg,#8b5cf6,#ec4899)' }}
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
      <div className="w-14 h-14 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--line)' }}>
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
  <div className="rounded-2xl p-10 text-center relative overflow-hidden" style={{ background: 'linear-gradient(135deg, #14141c 0%, #1a1730 100%)', border: '1px solid rgba(139,92,246,0.3)' }}>
    <div className="absolute inset-0 dotgrid opacity-30 pointer-events-none"></div>
    <div className="absolute -top-20 -right-20 w-64 h-64 rounded-full blur-3xl opacity-30" style={{ background: 'radial-gradient(circle, #8b5cf6, transparent)' }}></div>
    <div className="relative">
      <div className="w-14 h-14 mx-auto mb-4 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#f59e0b,#f43f5e)' }}>
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

  if (!supabase) return <p style={{ color: '#ff7d8c' }}>Supabase 未初始化</p>;

  const isEditorRoute = route.view === 'editor';
  if (isEditorRoute && !isAdmin) {
    navigate('#/forum');
    return null;
  }

  const canRead = isAdmin || isPremium;

  const headerBadge = (() => {
    if (isAdmin)   return { txt: 'ADMIN',   style: { background: 'rgba(0,214,143,0.15)',  color: '#3ce0a8', border: '1px solid rgba(0,214,143,0.3)' } };
    if (isPremium) return { txt: 'MEMBER',  style: { background: 'linear-gradient(135deg,#f59e0b,#f43f5e)', color: '#fff', border: 'none' } };
    if (user)      return { txt: 'LOCKED',  style: { background: 'rgba(255,255,255,0.04)', color: 'var(--text-3)', border: '1px solid var(--line)' } };
    return            { txt: 'GUEST',    style: { background: 'rgba(255,255,255,0.04)', color: 'var(--text-3)', border: '1px solid var(--line)' } };
  })();

  return (
    <div className="space-y-4">
      <div className="rounded-2xl ring-soft p-5 flex items-center justify-between relative overflow-hidden" style={{ background: 'linear-gradient(135deg, var(--surface) 0%, #181b25 100%)' }}>
        <div className="absolute top-0 right-0 w-48 h-48 rounded-full blur-3xl opacity-20 pointer-events-none" style={{ background: 'radial-gradient(circle, #8b5cf6, transparent)' }}></div>
        <div className="relative flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg,#8b5cf6,#ec4899)' }}>
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
