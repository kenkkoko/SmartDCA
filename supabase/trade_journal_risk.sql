-- 交易日誌 v2:Wyckoff 開倉結構 + 全倉風控欄位
-- 在 Supabase Dashboard → SQL Editor 貼上執行一次即可(可重複執行)。
-- 對應前端:新增/編輯開單紀錄的「進場結構」「保證金模式」「帳戶淨值」「強平價」。

alter table public.trade_journal
  add column if not exists setup          text,      -- Wyckoff 結構:lps / lpsy / spring / utad / sos_bu / sow_retest / other
  add column if not exists wyckoff_phase  text,      -- Wyckoff 階段:A / B / C / D / E
  add column if not exists margin_mode    text,      -- cross(全倉)/ isolated(逐倉)
  add column if not exists account_equity numeric,   -- 下單當下的帳戶淨值(USDT)→ 用來算有效槓桿與單筆風險佔比
  add column if not exists liq_price      numeric;   -- 平台顯示的強平價 → 用來算距強平幅度

-- 欄位說明(方便日後在 Supabase 介面上看)
comment on column public.trade_journal.setup          is 'Wyckoff 進場結構,老師幾乎只打 lps(做多)與 lpsy(做空)';
comment on column public.trade_journal.wyckoff_phase  is 'Wyckoff 階段 A~E';
comment on column public.trade_journal.margin_mode    is 'cross = 全倉(整個錢包墊底,強平價推遠);isolated = 逐倉';
comment on column public.trade_journal.account_equity is '下單當下帳戶淨值(USDT)。有效槓桿 = 名目部位 / 帳戶淨值';
comment on column public.trade_journal.liq_price      is '平台顯示的強平價。強平價不是止損,只是最後一道牆';

-- 值域檢查(用 do 區塊避免重複建立時報錯)
do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'trade_journal_margin_mode_chk') then
    alter table public.trade_journal
      add constraint trade_journal_margin_mode_chk
      check (margin_mode is null or margin_mode in ('cross', 'isolated'));
  end if;
  if not exists (select 1 from pg_constraint where conname = 'trade_journal_phase_chk') then
    alter table public.trade_journal
      add constraint trade_journal_phase_chk
      check (wyckoff_phase is null or wyckoff_phase in ('A', 'B', 'C', 'D', 'E'));
  end if;
end $$;
