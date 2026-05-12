// =========================================================
// SmartDCA Forum — 技術分析論壇
// 透過 <script type="text/babel" src="forum.js"> 載入
// 元件透過 window.* 暴露給 index.html 的主 script 使用
// =========================================================

// ───────────────────────────────────
// Hash route
//   #/forum               → list
//   #/forum/new           → editor (new)
//   #/forum/{id}          → detail
//   #/forum/{id}/edit     → editor (edit)
// ───────────────────────────────────
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

// ───────────────────────────────────
// Image upload helper (Supabase Storage)
// ───────────────────────────────────
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

// ───────────────────────────────────
// Markdown → safe HTML
// ───────────────────────────────────
const renderMarkdownToHtml = (md) => {
  if (!md) return '';
  if (!window.marked || !window.DOMPurify) {
    return window.DOMPurify ? window.DOMPurify.sanitize(`<pre>${md}</pre>`) : '';
  }
  const rawHtml = window.marked.parse(md, { breaks: true, gfm: true });
  return window.DOMPurify.sanitize(rawHtml);
};

// ───────────────────────────────────
// PostCard
// ───────────────────────────────────
const PostCard = ({ post, onOpen }) => (
  <button
    onClick={() => onOpen(post.id)}
    className="w-full text-left bg-slate-900/50 hover:bg-slate-800/60 border border-slate-700/50 hover:border-purple-500/50 rounded-xl p-4 transition-all"
  >
    <div className="flex items-start justify-between gap-3 mb-2">
      <h3 className="text-lg font-bold text-white flex-1">{post.title}</h3>
      {!post.published && (
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30 whitespace-nowrap">
          草稿
        </span>
      )}
    </div>
    {post.tags && post.tags.length > 0 && (
      <div className="flex flex-wrap gap-1 mb-2">
        {post.tags.map((t) => (
          <span key={t} className="text-xs px-2 py-0.5 rounded bg-slate-700/50 text-slate-300">
            #{t}
          </span>
        ))}
      </div>
    )}
    <p className="text-xs text-slate-500">
      {new Date(post.created_at).toLocaleString('zh-TW', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit'
      })}
    </p>
  </button>
);

// ───────────────────────────────────
// PostList (with filtering / search / sort)
// ───────────────────────────────────
const PostList = ({ supabase, isAdmin, onOpen }) => {
  const [posts, setPosts] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  // Filter state
  const [searchQuery, setSearchQuery] = React.useState('');
  const [selectedTags, setSelectedTags] = React.useState([]);
  const [sortDesc, setSortDesc] = React.useState(true);
  const [draftsOnly, setDraftsOnly] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      // 拉 content 進來給搜尋用 — 你的文章量不大,網路成本可忽略
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

  // Derive all tags + counts (from full result, before filtering)
  const tagCounts = React.useMemo(() => {
    const m = new Map();
    for (const p of posts) {
      for (const t of (p.tags || [])) m.set(t, (m.get(t) || 0) + 1);
    }
    return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
  }, [posts]);

  // Apply filters + sort
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

  return (
    <div className="space-y-3">
      {/* Action bar */}
      <div className="flex flex-wrap items-center gap-2 justify-between">
        <div className="flex flex-wrap items-center gap-2 flex-1">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="🔍 搜尋標題 / 內容 / 標籤..."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white"
                aria-label="清除搜尋"
              >
                ×
              </button>
            )}
          </div>
          {/* Sort */}
          <select
            value={sortDesc ? 'desc' : 'asc'}
            onChange={(e) => setSortDesc(e.target.value === 'desc')}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-purple-500 focus:outline-none"
          >
            <option value="desc">📅 最新優先</option>
            <option value="asc">📅 最舊優先</option>
          </select>
          {/* Drafts only (admin) */}
          {isAdmin && (
            <label className="flex items-center gap-1.5 text-xs text-slate-400 px-2 py-2 rounded-lg bg-slate-900 border border-slate-700 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={draftsOnly}
                onChange={(e) => setDraftsOnly(e.target.checked)}
                className="accent-orange-500"
              />
              只看草稿
            </label>
          )}
        </div>

        {isAdmin && (
          <button
            onClick={() => navigate('#/forum/new')}
            className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-sm font-bold transition-colors flex items-center gap-1.5 whitespace-nowrap"
          >
            <span>✏️</span> 新增文章
          </button>
        )}
      </div>

      {/* Tag chips */}
      {tagCounts.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-slate-500 mr-1">標籤:</span>
          {tagCounts.map(([tag, count]) => {
            const active = selectedTags.includes(tag);
            return (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                className={`px-2 py-0.5 rounded transition-colors ${
                  active
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700'
                }`}
              >
                #{tag} <span className="opacity-60">×{count}</span>
              </button>
            );
          })}
          {filtersActive && (
            <button
              onClick={clearFilters}
              className="ml-1 text-slate-400 hover:text-white underline"
            >
              清除全部
            </button>
          )}
        </div>
      )}

      {/* Result count */}
      {!loading && !error && posts.length > 0 && (
        <p className="text-xs text-slate-500">
          顯示 {visiblePosts.length} / {posts.length} 篇
          {filtersActive && <span className="text-purple-400 ml-1">(已套用篩選)</span>}
        </p>
      )}

      {/* Results */}
      {loading && <p className="text-slate-500 text-center py-12">讀取中...</p>}
      {error && (
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          ❌ 讀取失敗:{error}
        </div>
      )}
      {!loading && !error && posts.length === 0 && (
        <div className="text-center py-20 text-slate-500">
          <p className="text-4xl mb-3">📭</p>
          <p>目前沒有任何文章</p>
          {isAdmin && <p className="text-xs mt-2 text-slate-600">點右上「✏️ 新增文章」開始</p>}
        </div>
      )}
      {!loading && posts.length > 0 && visiblePosts.length === 0 && (
        <div className="text-center py-20 text-slate-500">
          <p className="text-4xl mb-3">🔍</p>
          <p>找不到符合條件的文章</p>
          <button
            onClick={clearFilters}
            className="text-xs mt-2 text-purple-400 hover:text-purple-300 underline"
          >
            清除篩選
          </button>
        </div>
      )}
      {!loading && visiblePosts.map((p) => <PostCard key={p.id} post={p} onOpen={onOpen} />)}
    </div>
  );
};

// ───────────────────────────────────
// PostDetail
// ───────────────────────────────────
const PostDetail = ({ supabase, postId, isAdmin, onBack }) => {
  const [post, setPost] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const contentRef = React.useRef(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
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

  // Post-render: syntax highlight + LaTeX
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
  }, [post]);

  const handleDelete = async () => {
    if (!window.confirm(`確定要刪除「${post.title}」嗎?無法復原。`)) return;
    const { error: err } = await supabase
      .from('forum_posts')
      .delete()
      .eq('id', postId);
    if (err) { window.alert(`刪除失敗:${err.message}`); return; }
    navigate('#/forum');
  };

  if (loading) return <p className="text-slate-500 text-center py-12">讀取中...</p>;
  if (error || !post) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="text-purple-400 hover:text-purple-300 text-sm">
          ← 返回列表
        </button>
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          ❌ {error || '文章不存在'}
        </div>
      </div>
    );
  }

  const html = renderMarkdownToHtml(post.content);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <button onClick={onBack} className="text-purple-400 hover:text-purple-300 text-sm">
          ← 返回列表
        </button>
        {isAdmin && (
          <div className="flex gap-2">
            <button
              onClick={() => navigate(`#/forum/${postId}/edit`)}
              className="px-3 py-1.5 rounded-lg bg-blue-600/80 hover:bg-blue-500 text-white text-xs font-bold"
            >
              ✏️ 編輯
            </button>
            <button
              onClick={handleDelete}
              className="px-3 py-1.5 rounded-lg bg-red-600/80 hover:bg-red-500 text-white text-xs font-bold"
            >
              🗑 刪除
            </button>
          </div>
        )}
      </div>

      <article className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-6">
        <header className="mb-4 pb-4 border-b border-slate-700/50">
          <div className="flex items-start justify-between gap-3 mb-2">
            <h1 className="text-2xl font-bold text-white">{post.title}</h1>
            {!post.published && (
              <span className="text-xs px-2 py-1 rounded-full bg-orange-500/20 text-orange-400 border border-orange-500/30">
                草稿
              </span>
            )}
          </div>
          {post.tags && post.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {post.tags.map((t) => (
                <span key={t} className="text-xs px-2 py-0.5 rounded bg-slate-700/50 text-slate-300">
                  #{t}
                </span>
              ))}
            </div>
          )}
          <p className="text-xs text-slate-500">
            {new Date(post.created_at).toLocaleString('zh-TW')}
            {post.updated_at && post.updated_at !== post.created_at && (
              <span className="ml-2 text-slate-600">
                (編輯於 {new Date(post.updated_at).toLocaleString('zh-TW')})
              </span>
            )}
          </p>
        </header>

        <div
          ref={contentRef}
          className="forum-md"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </article>
    </div>
  );
};

// ───────────────────────────────────
// PostEditor — admin only
// ───────────────────────────────────
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

  // Load existing post when editing
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
      // Set editor content once it's mounted
      if (easymdeRef.current) easymdeRef.current.value(data.content || '');
      else window.__pendingEditorContent = data.content || '';
      setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [mode, postId, supabase]);

  // Mount EasyMDE once
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
        setError(`圖片上傳失敗:${e.message}`);
      } finally {
        setUploadingImage(false);
      }
    };

    easymdeRef.current = new window.EasyMDE({
      element: textareaRef.current,
      autoDownloadFontAwesome: true,
      spellChecker: false,
      status: ['lines', 'words'],
      placeholder: '在這裡寫文章... 可以 Ctrl+V 直接貼截圖,或把圖片拖進來。\n\n支援:\n  # 標題、**粗體**、*斜體*、- 清單\n  ```python ... ``` 程式碼區塊\n  $E=mc^2$ LaTeX 公式\n',
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

    // Apply any pending content (when loading data finished before EasyMDE mounted)
    if (window.__pendingEditorContent != null) {
      easymdeRef.current.value(window.__pendingEditorContent);
      delete window.__pendingEditorContent;
    }

    // Paste handler — clipboard image (e.g. Ctrl+V after screenshot)
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
    // Re-run when `loading` flips false — edit mode early-returns a
    // loading screen first, so the textarea isn't in the DOM on the
    // initial effect pass. This ensures EasyMDE mounts the second time.
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

  if (loading) return <p className="text-slate-500 text-center py-12">讀取中...</p>;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-white">
          {mode === 'edit' ? '✏️ 編輯文章' : '✏️ 新增文章'}
        </h2>
        <button
          onClick={onCancel}
          className="text-slate-400 hover:text-white text-sm"
          disabled={saving}
        >
          取消
        </button>
      </div>

      {error && (
        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/30 rounded-lg p-3">
          ❌ {error}
        </div>
      )}
      {uploadingImage && (
        <div className="text-blue-400 text-sm bg-blue-500/10 border border-blue-500/30 rounded-lg p-3">
          📤 上傳圖片中...
        </div>
      )}

      <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4 space-y-3">
        <div>
          <label className="block text-xs text-slate-400 mb-1">標題 *</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
            placeholder="例如:BTC 短期觀察"
            disabled={saving}
          />
        </div>
        <div>
          <label className="block text-xs text-slate-400 mb-1">
            標籤(用逗號分隔)
          </label>
          <input
            type="text"
            value={tagsInput}
            onChange={(e) => setTagsInput(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:outline-none"
            placeholder="BTC, 技術分析, RSI"
            disabled={saving}
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
          className="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white text-sm disabled:opacity-50"
        >
          取消
        </button>
        <button
          onClick={() => handleSave(false)}
          disabled={saving}
          className="px-4 py-2 rounded-lg bg-orange-600/80 hover:bg-orange-500 text-white text-sm font-bold disabled:opacity-50"
        >
          {saving ? '儲存中...' : '💾 儲存為草稿'}
        </button>
        <button
          onClick={() => handleSave(true)}
          disabled={saving}
          className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 text-white text-sm font-bold disabled:opacity-50"
        >
          {saving ? '發佈中...' : (origPublished && mode === 'edit' ? '✅ 更新並保持發佈' : '🚀 發佈')}
        </button>
      </div>
    </div>
  );
};

// ───────────────────────────────────
// LoginRequiredView — 訪客看到
// ───────────────────────────────────
const LoginRequiredView = () => (
  <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-8 text-center">
    <p className="text-5xl mb-4">🔒</p>
    <h3 className="text-xl font-bold text-white mb-2">請先登入</h3>
    <p className="text-sm text-slate-400 mb-4">
      技術分析論壇需要登入會員才能查看。
    </p>
    <p className="text-xs text-slate-500">
      請點右上角的登入按鈕,登入後可向作者申請會員開通。
    </p>
  </div>
);

// ───────────────────────────────────
// PremiumRequiredView — 登入但非會員看到
// ───────────────────────────────────
const PremiumRequiredView = ({ user }) => (
  <div className="bg-gradient-to-br from-slate-900/60 to-purple-900/20 rounded-xl border border-purple-500/30 p-8 text-center">
    <p className="text-5xl mb-4">⭐</p>
    <h3 className="text-xl font-bold text-white mb-2">會員專屬內容</h3>
    <p className="text-sm text-slate-300 mb-1">
      技術分析論壇是付費會員專屬功能。
    </p>
    <p className="text-sm text-slate-400 mb-4">
      升級會員後可解鎖所有技術分析文章與圖表觀察。
    </p>
    <div className="bg-slate-950/50 rounded-lg border border-slate-700/50 p-3 inline-block">
      <p className="text-xs text-slate-500 mb-1">目前帳號</p>
      <p className="text-sm text-slate-300">{user?.email}</p>
      <p className="text-[10px] text-slate-600 mt-1">請聯絡作者開通會員</p>
    </div>
  </div>
);

// ───────────────────────────────────
// ForumApp
// ───────────────────────────────────
const ForumApp = ({ supabase, user, isAdmin, isPremium }) => {
  const hash = useHashRoute();
  const route = parseForumRoute(hash);

  if (!supabase) return <p className="text-red-400">Supabase 未初始化</p>;

  // Editor routes always require admin (premium not enough to write)
  const isEditorRoute = route.view === 'editor';
  if (isEditorRoute && !isAdmin) {
    navigate('#/forum');
    return null;
  }

  // Access tier check
  const canRead = isAdmin || isPremium;

  const headerBadge = (() => {
    if (isAdmin)   return { txt: '✅ Admin',  cls: 'bg-green-500/20 text-green-400 border border-green-500/30' };
    if (isPremium) return { txt: '⭐ 會員',   cls: 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' };
    if (user)      return { txt: '🔒 未開通', cls: 'bg-slate-700/50 text-slate-400 border border-slate-600/30' };
    return            { txt: '🔒 訪客',     cls: 'bg-slate-700/50 text-slate-400 border border-slate-600/30' };
  })();

  return (
    <div className="space-y-4">
      <div className="bg-slate-900/50 rounded-xl border border-slate-700/50 p-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">📊 技術分析論壇</h2>
          <p className="text-xs text-slate-500 mt-1">
            {user ? user.email : '未登入'}
          </p>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${headerBadge.cls}`}>
          {headerBadge.txt}
        </span>
      </div>

      {/* Access gate */}
      {!user && <LoginRequiredView />}
      {user && !canRead && <PremiumRequiredView user={user} />}

      {/* Full forum (only for admin or premium) */}
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
