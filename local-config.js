// =========================================================
// 本地開發專用設定檔 — 不會被推到 GitHub(已在 .gitignore)
// 跑本地 server 時,index.html 會優先讀這裡的設定
// =========================================================
window.__LOCAL_CONFIG__ = {
  // dev Supabase 專案
  SUPABASE_URL: 'https://sycndjfizoihpckjdmto.supabase.co',
  SUPABASE_ANON_KEY: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5Y25kamZpem9paHBja2pkbXRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg1NjEwNjMsImV4cCI6MjA5NDEzNzA2M30.o-N64nUd4KWKGWpZcNIwPOtMURTPFyD3CCIS8PLJxgI',

  // 是否啟用論壇(本地測試時設 true)
  FORUM_ENABLED: true,
};
