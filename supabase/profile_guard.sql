-- 防止使用者自己把帳號改成 Premium / 管理員
-- 前端用自己的登入身分建立與更新 user_profiles（存觀察清單、暱稱等），
-- 如果 RLS 允許「改自己那一列」，也就能把 is_premium / is_admin 改成 true，
-- 免費使用伺服器的 Gemini 金鑰，或取得論壇管理權限。
-- 這個觸發器讓一般登入者（authenticated / anon）不能設定這兩個欄位；
-- 在 SQL Editor、Dashboard 或用 service_role 執行的程式不受影響。
-- 在 Supabase Dashboard → SQL Editor 貼上執行一次即可。

create or replace function public.protect_profile_flags()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  if coalesce(auth.role(), '') in ('authenticated', 'anon') then
    if tg_op = 'INSERT' then
      new.is_premium := false;
      new.is_admin := false;
    elsif new.is_premium is distinct from old.is_premium
       or new.is_admin is distinct from old.is_admin then
      raise exception '不能自行修改會員或管理員身分';
    end if;
  end if;
  return new;
end $$;

drop trigger if exists protect_profile_flags on public.user_profiles;
create trigger protect_profile_flags
  before insert or update on public.user_profiles
  for each row execute function public.protect_profile_flags();

-- ---------------------------------------------------------------------------
-- 檢查：每張表有沒有開 RLS、有哪些規則（結果貼回來即可）
-- ---------------------------------------------------------------------------
-- select tablename, rowsecurity from pg_tables where schemaname = 'public' order by 1;
-- select tablename, policyname, cmd, roles, qual, with_check from pg_policies where schemaname = 'public' order by 1, 2;
