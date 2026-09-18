-- Forum post notifications — run once in the Supabase SQL editor.
--
-- Why: notify_new_posts.py looks for posts where notification_sent is false.
-- If the column was added without a DEFAULT, every row is NULL, and `eq.false`
-- never matches NULL — so the scheduled job found "nothing to send" and passed
-- green while no LINE / web push ever went out.

-- 1. Make sure the column exists and always has a real boolean value.
alter table public.forum_posts
    add column if not exists notification_sent boolean;

alter table public.forum_posts
    alter column notification_sent set default false;

-- 2. Backfill existing rows.
--    Posts published more than 7 days ago are marked as already notified so
--    turning this on does not blast subscribers with your whole archive.
update public.forum_posts
   set notification_sent = true
 where notification_sent is null
   and published = true
   and created_at < now() - interval '7 days';

update public.forum_posts
   set notification_sent = false
 where notification_sent is null;

alter table public.forum_posts
    alter column notification_sent set not null;

-- 3. Keeps the every-30-min job cheap.
create index if not exists forum_posts_pending_notify_idx
    on public.forum_posts (created_at)
    where published = true and notification_sent = false;

-- 4. Sanity check — should list only what you actually want pushed out next run.
select id, title, published, notification_sent, created_at
  from public.forum_posts
 where published = true
   and notification_sent = false
 order by created_at;
