-- 論壇「判讀」— 在 Supabase Dashboard → SQL Editor 貼上執行一次即可（可重複執行）。
--
-- 設計：
--   forum_judgments          判讀內容（方向、級別、目標價、依據、關鍵價位…）
--                            看得到文章的人都看得到；只有管理員能新增／修改。
--   forum_judgment_results   驗證結果（狀態、進場價、結算…）
--                            只有管理員讀得到；前端完全不能寫，由每日排程用 service role 寫入。
--
--   文章一發布，判讀就鎖定（locked_at）。鎖定後資料庫拒絕任何修改，只允許「撤回」。
--   已發布文章之後才補上的判讀，一建立就立即鎖定。
--   進場價不由前端提供，由排程依 locked_at 從歷史 K 線查出。

-- 0. 文章可標記「本篇不含判讀」（公告、心得類文章）
alter table public.forum_posts
    add column if not exists no_judgment boolean not null default false;

-- 管理員判斷（沿用 user_profiles.is_admin）
create or replace function public.is_forum_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select coalesce((select is_admin from public.user_profiles where id = auth.uid()), false);
$$;

-- 1. 判讀內容表 — post_id 的型別跟著 forum_posts.id 自動決定（uuid 或 bigint 皆可）
do $$
declare
    post_id_type text;
begin
    select format_type(a.atttypid, a.atttypmod) into post_id_type
      from pg_attribute a
     where a.attrelid = 'public.forum_posts'::regclass and a.attname = 'id';

    execute format($f$
        create table if not exists public.forum_judgments (
            id            uuid primary key default gen_random_uuid(),
            post_id       %s not null references public.forum_posts(id) on delete cascade,
            market        text not null check (market in ('crypto', 'us', 'tw')),
            symbol        text not null,                 -- crypto: BTC；us: SPY；tw: 0050（排程負責換成行情代號）
            timeframe     text not null check (timeframe in ('4h', '1d', '1w', '1M')),
            directions    text[] not null,               -- 'up' / 'down' / 'range'，盤整可與方向複選
            target_price  numeric,
            range_low     numeric,                       -- 有選盤整時的區間
            range_high    numeric,
            methods       text[] not null default '{}',  -- dow / wyckoff / pattern / volume / other
            levels        jsonb  not null default '[]',  -- [{type, price}]  阻力／支撐／冰線／溪流／POC／VAH／VAL，只記錄不計分
            chart_url     text,                          -- 從文章內已貼的圖片挑一張
            reason        text,
            sort          int    not null default 0,
            locked_at     timestamptz,                   -- 文章發布時自動填入，之後不可修改
            withdrawn_at  timestamptz,                   -- 撤回時間（鎖定後唯一可改的欄位）
            created_at    timestamptz not null default now(),
            updated_at    timestamptz not null default now(),
            constraint forum_judgments_directions_ok check (
                cardinality(directions) between 1 and 2
                and directions <@ array['up', 'down', 'range']::text[]
                and not (directions @> array['up', 'down']::text[])
            ),
            constraint forum_judgments_range_ok check (
                not (directions @> array['range']::text[])
                or (range_low is not null and range_high is not null and range_low < range_high)
            )
        )
    $f$, post_id_type);
end $$;

create index if not exists forum_judgments_post_idx on public.forum_judgments (post_id, sort);

-- 2. 驗證結果表（只有管理員看得到）
create table if not exists public.forum_judgment_results (
    judgment_id    uuid primary key references public.forum_judgments(id) on delete cascade,
    status         text not null default 'pending'
                   check (status in ('pending', 'hit', 'right', 'flat', 'wrong', 'withdrawn')),
    entry_price    numeric,
    entry_at       timestamptz,
    bars_total     int,
    bars_elapsed   int,
    last_price     numeric,
    change_pct     numeric,          -- 相對進場價，依判讀方向取正負
    max_progress   numeric,          -- 最遠走到目標的比例（0–1+）
    exit_price     numeric,
    settled_at     timestamptz,      -- 結算後固定，排程不再重跑
    regime         text,             -- 系統判定的當下市場狀態：up / down / range
    vpvr           jsonb,            -- {poc, vah, val}
    level_events   jsonb,            -- 關鍵價位的觸及／突破紀錄
    updated_at     timestamptz not null default now()
);

create index if not exists forum_judgment_results_pending_idx
    on public.forum_judgment_results (status) where status = 'pending';

-- 3. 權限
alter table public.forum_judgments        enable row level security;
alter table public.forum_judgment_results enable row level security;

-- 判讀：看得到該篇文章的人就看得到（沿用 forum_posts 自身的 RLS）
drop policy if exists "judgments readable with post" on public.forum_judgments;
create policy "judgments readable with post" on public.forum_judgments
    for select
    using (exists (select 1 from public.forum_posts p where p.id = post_id));

drop policy if exists "judgments admin insert" on public.forum_judgments;
create policy "judgments admin insert" on public.forum_judgments
    for insert with check (public.is_forum_admin());

drop policy if exists "judgments admin update" on public.forum_judgments;
create policy "judgments admin update" on public.forum_judgments
    for update using (public.is_forum_admin()) with check (public.is_forum_admin());

drop policy if exists "judgments admin delete" on public.forum_judgments;
create policy "judgments admin delete" on public.forum_judgments
    for delete using (public.is_forum_admin() and locked_at is null);

-- 結果：只有管理員能讀；沒有任何 insert / update / delete policy → 前端一律不能寫
drop policy if exists "results admin read" on public.forum_judgment_results;
create policy "results admin read" on public.forum_judgment_results
    for select using (public.is_forum_admin());

-- 4. 鎖定規則
-- 4a. 新增判讀時：若文章已發布，立即鎖定
create or replace function public.forum_judgments_before_insert()
returns trigger language plpgsql as $$
begin
    if exists (select 1 from public.forum_posts p where p.id = new.post_id and p.published) then
        new.locked_at := now();
    else
        new.locked_at := null;
    end if;
    new.withdrawn_at := null;
    return new;
end $$;

drop trigger if exists forum_judgments_before_insert on public.forum_judgments;
create trigger forum_judgments_before_insert
    before insert on public.forum_judgments
    for each row execute function public.forum_judgments_before_insert();

-- 4b. 修改判讀時：鎖定後只允許把 withdrawn_at 從空值填入（撤回），其他一律拒絕
create or replace function public.forum_judgments_before_update()
returns trigger language plpgsql as $$
begin
    if old.locked_at is not null then
        if old.withdrawn_at is null and new.withdrawn_at is not null
           and (to_jsonb(new) - 'withdrawn_at' - 'updated_at' - 'locked_at')
             = (to_jsonb(old) - 'withdrawn_at' - 'updated_at' - 'locked_at') then
            new.locked_at := old.locked_at;
            new.withdrawn_at := now();
            new.updated_at := now();
            return new;
        end if;
        -- 文章發布觸發的鎖定更新（locked_at 本身不變）直接通過
        if (to_jsonb(new) - 'updated_at') = (to_jsonb(old) - 'updated_at') then
            return new;
        end if;
        raise exception '判讀已鎖定（文章已發布），只能撤回，不能修改';
    end if;
    new.withdrawn_at := null;   -- 未鎖定（草稿）不需要撤回，刪除即可
    new.updated_at := now();
    return new;
end $$;

drop trigger if exists forum_judgments_before_update on public.forum_judgments;
create trigger forum_judgments_before_update
    before update on public.forum_judgments
    for each row execute function public.forum_judgments_before_update();

-- 4c. 文章轉為發布時：鎖定該篇所有尚未鎖定的判讀
--     （security definer：由文章更新觸發，不受 judgments 的 RLS 影響）
create or replace function public.forum_posts_lock_judgments()
returns trigger language plpgsql security definer set search_path = public as $$
begin
    if new.published then
        update public.forum_judgments
           set locked_at = now()
         where post_id = new.id and locked_at is null;
    end if;
    return new;
end $$;

drop trigger if exists forum_posts_lock_judgments on public.forum_posts;
create trigger forum_posts_lock_judgments
    after insert or update of published on public.forum_posts
    for each row execute function public.forum_posts_lock_judgments();

-- 4d. 判讀鎖定時，建立一筆「驗證中」的結果，等待每日排程結算
create or replace function public.forum_judgments_create_result()
returns trigger language plpgsql security definer set search_path = public as $$
begin
    if new.locked_at is not null and (tg_op = 'INSERT' or old.locked_at is null) then
        insert into public.forum_judgment_results (judgment_id, status)
        values (new.id, 'pending')
        on conflict (judgment_id) do nothing;
    end if;
    if tg_op = 'UPDATE' and old.withdrawn_at is null and new.withdrawn_at is not null then
        -- 撤回：交給排程依撤回當下價格結算（狀態保持 pending，排程看到 withdrawn_at 會處理）
        update public.forum_judgment_results set updated_at = now() where judgment_id = new.id;
    end if;
    return new;
end $$;

drop trigger if exists forum_judgments_create_result on public.forum_judgments;
create trigger forum_judgments_create_result
    after insert or update on public.forum_judgments
    for each row execute function public.forum_judgments_create_result();

-- 5. 檢查：應該看到兩張表與它們的 policy
select tablename, policyname, cmd
  from pg_policies
 where tablename in ('forum_judgments', 'forum_judgment_results')
 order by tablename, policyname;
