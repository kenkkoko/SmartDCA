-- 交易日誌(合約開單備忘錄)資料表
-- 在 Supabase Dashboard → SQL Editor 貼上執行一次即可。

create table if not exists public.trade_journal (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid not null default auth.uid() references auth.users(id) on delete cascade,
  symbol        text not null,
  direction     text not null check (direction in ('long', 'short')),
  leverage      numeric,
  entry_price   numeric not null,
  position_size numeric,        -- 倉位/保證金(USDT)
  stop_loss     numeric,
  take_profit   numeric,
  entry_reason  text,           -- 為什麼進場(技術依據)
  status        text not null default 'open' check (status in ('open', 'closed')),
  exit_price    numeric,
  pnl           numeric,        -- 已含槓桿的報酬率(%)
  review        text,           -- 事後反省
  entry_at      timestamptz not null default now(),
  closed_at     timestamptz,
  created_at    timestamptz not null default now()
);

create index if not exists trade_journal_user_entry_idx
  on public.trade_journal (user_id, entry_at desc);

alter table public.trade_journal enable row level security;

-- 只能看見、修改自己的紀錄
drop policy if exists "own rows" on public.trade_journal;
create policy "own rows" on public.trade_journal
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
