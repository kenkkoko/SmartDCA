-- 判讀 v2 — 每日驗證需要的欄位與保護。接在 forum_judgments.sql 之後，在 SQL Editor 執行一次（可重複執行）。
--
--   1. 結算後的結果列不可再改：排程只處理 pending，資料庫也拒絕改動已結算列。
--   2. 驗證用欄位：中性門檻（依 ATR）、錯誤訊息。
--   3. 補登舊文章的判讀：backfilled = true，判讀時間用文章發布時間，
--      只能從 SQL Editor／service role 寫入（前端登入者一律照舊鎖在「現在」）。

alter table public.forum_judgments
    add column if not exists backfilled boolean not null default false;

alter table public.forum_judgment_results
    add column if not exists neutral_pct numeric,   -- 方向對／持平的分界：0.5 × ATR(14) ÷ 進場價
    add column if not exists note text;             -- 抓不到行情等問題，留給管理員看

-- 1. 已結算不可改
create or replace function public.forum_judgment_results_guard()
returns trigger language plpgsql as $$
begin
    if old.settled_at is not null then
        raise exception '判讀已結算，結果不可再修改';
    end if;
    if new.settled_at is not null and new.status = 'pending' then
        raise exception '結算時必須給定結果狀態';
    end if;
    new.updated_at := now();
    return new;
end $$;

drop trigger if exists forum_judgment_results_guard on public.forum_judgment_results;
create trigger forum_judgment_results_guard
    before update on public.forum_judgment_results
    for each row execute function public.forum_judgment_results_guard();

-- 3. 新增判讀：補登（非登入者寫入）保留指定的判讀時間；其他照舊
create or replace function public.forum_judgments_before_insert()
returns trigger language plpgsql as $$
begin
    if new.backfilled and auth.uid() is null then
        if new.locked_at is null then
            raise exception '補登判讀需要指定 locked_at（文章發布時間）';
        end if;
        new.withdrawn_at := null;
        return new;
    end if;
    new.backfilled := false;
    if exists (select 1 from public.forum_posts p where p.id = new.post_id and p.published) then
        new.locked_at := now();
    else
        new.locked_at := null;
    end if;
    new.withdrawn_at := null;
    return new;
end $$;

-- 撤回時只碰還沒結算的結果（已結算的撤回不影響成績）
create or replace function public.forum_judgments_create_result()
returns trigger language plpgsql security definer set search_path = public as $$
begin
    if new.locked_at is not null and (tg_op = 'INSERT' or old.locked_at is null) then
        insert into public.forum_judgment_results (judgment_id, status)
        values (new.id, 'pending')
        on conflict (judgment_id) do nothing;
    end if;
    if tg_op = 'UPDATE' and old.withdrawn_at is null and new.withdrawn_at is not null then
        update public.forum_judgment_results set updated_at = now()
         where judgment_id = new.id and settled_at is null;
    end if;
    return new;
end $$;

-- 檢查：應該看到 backfilled、neutral_pct、note 三個欄位
select table_name, column_name
  from information_schema.columns
 where table_schema = 'public'
   and column_name in ('backfilled', 'neutral_pct', 'note');
