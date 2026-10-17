-- 圖表資料跨裝置同步(畫線、圖表設定、觀察清單、指標開關)
-- 在 Supabase Dashboard → SQL Editor 貼上執行一次即可。
-- 每個帳號、每個項目一列;key 例如 smartdca.chart.drawings.tech.crypto.BTC

create table if not exists public.chart_sync (
  user_id    uuid not null default auth.uid() references auth.users(id) on delete cascade,
  key        text not null,
  data       jsonb,                      -- null = 這台裝置刪除了這個項目
  updated_at timestamptz not null default now(),
  primary key (user_id, key)
);

create index if not exists chart_sync_user_updated_idx
  on public.chart_sync (user_id, updated_at);

-- 時間一律以伺服器為準(各裝置時鐘不同,用來判斷哪一邊比較新)
create or replace function public.chart_sync_touch()
returns trigger language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end $$;

drop trigger if exists chart_sync_touch on public.chart_sync;
create trigger chart_sync_touch
  before insert or update on public.chart_sync
  for each row execute function public.chart_sync_touch();

alter table public.chart_sync enable row level security;

-- 只能讀寫自己的資料
drop policy if exists "own rows" on public.chart_sync;
create policy "own rows" on public.chart_sync
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
