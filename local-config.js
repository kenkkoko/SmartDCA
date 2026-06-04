// =========================================================
// 本地開發專用設定檔 — 不會被推到 GitHub(已在 .gitignore)
// 跑本地 server 時,index.html 會優先讀這裡的設定
// =========================================================
window.__LOCAL_CONFIG__ = {
  // dev Supabase 專案
  SUPABASE_URL: 'https://cimoqkrzuubulinlxako.supabase.co',
  SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNpbW9xa3J6dXVidWxpbmx4YWtvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU2OTc1MjYsImV4cCI6MjA4MTI3MzUyNn0.cqhl31h2CLUR3iUfbUbP_QJwH-VQGc5wQ_m62UfIIQE',

  // 是否啟用論壇(本地測試時設 true)
  FORUM_ENABLED: true,
};
