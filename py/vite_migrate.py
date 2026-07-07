# -*- coding: utf-8 -*-
"""One-shot migration: split the monolithic index.html into a Vite project.

- Extracts the inline <script type="text/babel"> body into src/app.jsx
- Converts forum.js into src/forum.jsx (adds React import)
- Rewrites index.html: drops React/Babel/Tailwind/Supabase CDN tags,
  keeps chart CDN globals + inline styles, adds the Vite module entry.

Preserves file bytes as-is (the HTML contains NBSP characters).
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(p):
    with io.open(p, 'r', encoding='utf-8', newline='') as f:
        return f.read()

def write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write(s)

html = read(os.path.join(ROOT, 'index.html'))

# ---- 1. Extract the inline babel script ----
start_tag = '<script type="text/babel">'
# rindex: an HTML comment near the top also contains this literal string
start = html.rindex(start_tag)
body_start = start + len(start_tag)
end = html.index('</script>', body_start)
app_src = html[body_start:end]
assert 'createRoot' in app_src, 'app source missing createRoot'

# ---- 2. Adapt app source for ESM ----
replacements = [
    # supabase: CDN global -> npm import
    ("const supabase = (typeof window.supabase !== 'undefined' && !SUPABASE_URL.includes('PLACEHOLDER'))",
     "const supabase = (!SUPABASE_URL.includes('PLACEHOLDER'))"),
    ("? window.supabase.createClient(", "? createClient("),
    # config: allow Vite env vars (build-time) in addition to local-config.js
    ("const SUPABASE_URL = LOCAL_CFG?.SUPABASE_URL || '__SUPABASE_URL_PLACEHOLDER__';",
     "const SUPABASE_URL = LOCAL_CFG?.SUPABASE_URL || import.meta.env.VITE_SUPABASE_URL || '__SUPABASE_URL_PLACEHOLDER__';"),
    ("const SUPABASE_ANON_KEY = LOCAL_CFG?.SUPABASE_ANON_KEY || '__SUPABASE_ANON_KEY_PLACEHOLDER__';",
     "const SUPABASE_ANON_KEY = LOCAL_CFG?.SUPABASE_ANON_KEY || import.meta.env.VITE_SUPABASE_ANON_KEY || '__SUPABASE_ANON_KEY_PLACEHOLDER__';"),
]
for old, new in replacements:
    assert old in app_src, 'missing: ' + old[:60]
    app_src = app_src.replace(old, new)

header = (
    "import React from 'react';\n"
    "import ReactDOM from 'react-dom/client';\n"
    "import { createClient } from '@supabase/supabase-js';\n"
)
write(os.path.join(ROOT, 'src', 'app.jsx'), header + app_src)

# ---- 3. forum.js -> src/forum.jsx ----
forum = read(os.path.join(ROOT, 'forum.js'))
write(os.path.join(ROOT, 'src', 'forum.jsx'), "import React from 'react';\n" + forum)

# ---- 4. Rewrite index.html ----
# Replace the inline babel block with the Vite module entry
html = html[:start] + '<script type="module" src="/src/main.jsx"></script>' + html[end + len('</script>'):]

# Drop tags that the bundle now provides (line-based, match by substring)
drop_substrings = [
    'unpkg.com/react@',
    'unpkg.com/react-dom@',
    'unpkg.com/@babel/standalone',
    'cdn.tailwindcss.com',
    'cdn.jsdelivr.net/npm/@supabase/supabase-js',
    'type="text/babel" data-presets="react" src="forum.js"',
    'rel="preconnect" href="https://unpkg.com"',
]
kept = []
for line in html.split('\n'):
    if any(s in line for s in drop_substrings):
        continue
    kept.append(line)
html = '\n'.join(kept)

# Remove now-stale comments for removed scripts
for stale in [
    '        <!-- 引入 React 和 ReactDOM -->',
    '        <!-- 引入 Babel 用於解析 JSX -->',
    '        <!-- 引入 Tailwind CSS -->',
    '    <!-- Supabase JS -->',
    '    <!-- Forum module (loaded before main script so window.ForumApp is ready) -->',
]:
    html = html.replace(stale + '\n', '')
# Babel pin comment block (multi-line)
html = re.sub(r'    <!-- Pin to Babel 7:.*?-->\n', '', html, flags=re.S)

write(os.path.join(ROOT, 'index.html'), html)
print('OK  src/app.jsx lines:', app_src.count('\n'))
print('OK  index.html bytes:', len(html))
