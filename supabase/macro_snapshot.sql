-- 每日總經快照(main.py 每天寫入一筆,gemini-proxy 讀最新一筆塞進 AI prompt)
-- 在 Supabase Dashboard → SQL Editor 貼上執行一次即可(可重複執行)。

create table if not exists public.macro_snapshot (
  date                date primary key,         -- 台灣日期
  us10y               numeric,                  -- 10 年期公債殖利率 %
  us2y                numeric,                  -- 2 年期公債殖利率 %
  spread              numeric,                  -- 10Y - 2Y
  yields_as_of        date,                     -- 殖利率資料日期(美國交易日)
  us10y_1m_ago        numeric,                  -- 約 21 個交易日前的 10Y
  fed_funds           numeric,                  -- 有效聯邦基金利率 %
  cpi_yoy             numeric,                  -- CPI 年增率 %
  cpi_yoy_prev        numeric,
  cpi_yoy_prev2       numeric,
  cpi_month           date,                     -- CPI 最新資料月份
  unemployment        numeric,                  -- 失業率 %
  unemployment_prev   numeric,
  unemployment_month  date,
  macro_context       text,                     -- 給 AI 的總經純文字
  upcoming_events     jsonb,                    -- 未來 10 天事件 [{at(台灣時間 ISO), type, title, impact}]
  updated_at          timestamptz not null default now()
);

-- 只給 service role(main.py / Edge Function)讀寫:
-- 開 RLS 但不建任何 policy,並收回 anon / authenticated 的表權限。
alter table public.macro_snapshot enable row level security;
revoke all on table public.macro_snapshot from anon, authenticated;
