"""
Notify subscribers about newly published forum posts.

Runs on a schedule (every 30 min via .github/workflows/notify_new_posts.yml)
or via workflow_dispatch.

Flow:
  1. Find posts where published=true AND notification_sent is false/NULL
  2. For each post:
     - Send LINE broadcast (all bot friends)
     - Send Web Push to every user that has a push_subscription
  3. Mark notification_sent=true — ONLY if at least one channel delivered.
     If nothing was delivered the post stays queued and the job exits non-zero,
     so the Actions run goes red instead of silently passing.

Env:
  SUPABASE_URL / SUPABASE_KEY      required (service_role key)
  LINE_CHANNEL_ACCESS_TOKEN        optional — LINE step skipped if unset
  LINE_USER_ID                     optional — fallback push target if broadcast fails
  VAPID_PRIVATE_KEY                optional — defaults to the historic key
  SITE_URL                         optional
  DRY_RUN=1                        diagnose only: no sending, no DB writes
"""

import os
import json
import sys
from supabase import create_client, Client
from pywebpush import webpush, WebPushException

# --- Configuration ---
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')  # service_role key
LINE_TOKEN   = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_USER_ID = os.environ.get('LINE_USER_ID')  # fallback target if broadcast fails
DRY_RUN      = os.environ.get('DRY_RUN', '').lower() in ('1', 'true', 'yes')

# VAPID — matches main.py
VAPID_PRIVATE_KEY = os.environ.get(
    'VAPID_PRIVATE_KEY',
    '0yUZd0zYl6aJz7NscDAnkKhvVQUaQEacKZHo3vLRI9o'
)
VAPID_CLAIMS = {"sub": "mailto:admin@smartdca.com"}

SITE_URL = os.environ.get('SITE_URL', 'https://dca.hellokai07.com')


# ─────────────────────────────────────────────────────────────
# Diagnostics — prints exactly why nothing is queued, if that's the case
# ─────────────────────────────────────────────────────────────
def print_diagnostics(supabase: Client) -> None:
    try:
        res = supabase.table('forum_posts') \
            .select('id, title, published, notification_sent') \
            .order('created_at', desc=True) \
            .limit(20) \
            .execute()
        rows = res.data or []
    except Exception as e:
        print(f"[Diag] Could not read forum_posts: {e}")
        raise

    total = len(rows)
    pub = [r for r in rows if r.get('published')]
    null_flag = [r for r in pub if r.get('notification_sent') is None]
    sent = [r for r in pub if r.get('notification_sent') is True]
    unsent = [r for r in pub if r.get('notification_sent') is False]

    print("──── Diagnostics (latest 20 posts) ────")
    print(f"  total={total}  published={len(pub)}  "
          f"notification_sent: true={len(sent)} false={len(unsent)} NULL={len(null_flag)}")
    if null_flag:
        print("  ⚠ Some published posts have notification_sent = NULL.")
        print("    Run supabase/forum_notifications.sql once so the column defaults to false.")
    print("───────────────────────────────────────")


def send_line_broadcast(message_text: str) -> bool:
    """Send a LINE broadcast. Returns True on success."""
    if not LINE_TOKEN:
        print("  [LINE] Skipped — LINE_CHANNEL_ACCESS_TOKEN is not set in repo secrets.")
        return False
    if DRY_RUN:
        print("  [LINE] DRY_RUN — would broadcast:")
        print("    " + message_text.replace("\n", "\n    "))
        return False
    try:
        from linebot.v3.messaging import (
            Configuration, ApiClient, MessagingApi,
            BroadcastRequest, TextMessage, PushMessageRequest
        )
        config = Configuration(access_token=LINE_TOKEN)
        api_client = ApiClient(config)
        api = MessagingApi(api_client)
        try:
            api.broadcast(BroadcastRequest(messages=[TextMessage(text=message_text)]))
            print("  [LINE] Broadcast sent.")
            return True
        except Exception as broadcast_err:
            # Typical causes: monthly free quota exhausted, bot has no friends,
            # token belongs to a different channel, or the channel is in "chat mode".
            print(f"  [LINE] Broadcast failed: {type(broadcast_err).__name__}: {broadcast_err}")
            body = getattr(broadcast_err, 'body', None)
            if body:
                print(f"  [LINE] API body: {body}")
            if LINE_USER_ID:
                api.push_message(PushMessageRequest(
                    to=LINE_USER_ID, messages=[TextMessage(text=message_text)]
                ))
                print("  [LINE] Fallback push to LINE_USER_ID sent.")
                return True
            return False
    except Exception as e:
        print(f"  [LINE] Failed: {type(e).__name__}: {e}")
        return False


def send_web_push(supabase: Client, post: dict) -> tuple:
    """Send Web Push to every user with a subscription. Returns (success, fail)."""
    # Subscriptions are opt-in already (the user had to enable notifications in the app),
    # so don't filter by is_premium here — that silently excluded most subscribers.
    res = supabase.table('user_profiles') \
        .select('id, push_subscription') \
        .not_.is_('push_subscription', 'null') \
        .execute()

    users = [u for u in (res.data or []) if u.get('push_subscription')]
    if not users:
        print("  [Push] No users with a saved push_subscription.")
        return (0, 0)

    payload = json.dumps({
        "title": "📊 新技術分析文章",
        "body": post['title'],
        "url": f"{SITE_URL}/#/forum/{post['id']}"
    })

    if DRY_RUN:
        print(f"  [Push] DRY_RUN — would push to {len(users)} subscriber(s).")
        return (0, 0)

    success = 0
    fail = 0
    dead = []
    for user in users:
        sub = user['push_subscription']
        if isinstance(sub, str):
            try:
                sub = json.loads(sub)
            except Exception:
                print(f"  [Push] User {user['id']}: subscription is not valid JSON, skipped.")
                fail += 1
                continue
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
            success += 1
        except WebPushException as e:
            fail += 1
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            print(f"  [Push] Failed for user {user['id']} (HTTP {status}): {e}")
            # 404/410 = subscription is gone for good → clear it so it stops failing
            if status in (404, 410):
                dead.append(user['id'])

    for uid in dead:
        try:
            supabase.table('user_profiles') \
                .update({'push_subscription': None}) \
                .eq('id', uid) \
                .execute()
            print(f"  [Push] Cleared expired subscription for user {uid}.")
        except Exception as e:
            print(f"  [Push] Could not clear subscription for {uid}: {e}")

    print(f"  [Push] {success} sent, {fail} failed (of {len(users)} subscribers).")
    return (success, fail)


def build_line_message(post: dict) -> str:
    title = post['title']
    tags = post.get('tags') or []
    tag_str = ' '.join(f'#{t}' for t in tags[:3])
    url = f"{SITE_URL}/#/forum/{post['id']}"

    lines = ["📊 新技術分析文章發佈", "", f"【{title}】"]
    if tag_str:
        lines += ["", tag_str]
    lines += ["", f"👉 {url}"]
    return "\n".join(lines)


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY missing.")
        sys.exit(1)

    print(f"LINE token configured: {'yes' if LINE_TOKEN else 'NO'}")
    print(f"Dry run: {DRY_RUN}")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print_diagnostics(supabase)

    # notification_sent may be NULL on rows created before the column had a default —
    # `.eq(False)` never matches NULL, which is why this job used to pass silently.
    res = supabase.table('forum_posts') \
        .select('id, title, tags') \
        .eq('published', True) \
        .or_('notification_sent.is.null,notification_sent.eq.false') \
        .order('created_at', desc=False) \
        .execute()

    posts = res.data or []
    if not posts:
        print("No unsent published posts. Nothing to do.")
        return

    print(f"Found {len(posts)} unsent post(s).")

    undelivered = []

    for post in posts:
        print(f"\n→ Processing: {post['title']} ({post['id']})")

        # 1. LINE
        line_ok = send_line_broadcast(build_line_message(post))

        # 2. Web Push
        push_ok, _ = send_web_push(supabase, post)

        if DRY_RUN:
            print("  [DB] DRY_RUN — not marking as sent.")
            continue

        # 3. Mark done only if something actually went out. A post that reached
        #    nobody stays queued so the next run (or a fixed secret) can deliver it.
        if line_ok or push_ok > 0:
            try:
                supabase.table('forum_posts') \
                    .update({'notification_sent': True}) \
                    .eq('id', post['id']) \
                    .execute()
                print("  [DB] Marked notification_sent=true.")
            except Exception as e:
                print(f"  [DB] Failed to mark as sent: {e}")
        else:
            undelivered.append(post['title'])
            print("  [DB] Nothing delivered — leaving notification_sent unchanged.")

    if undelivered:
        print("\nERROR: no channel delivered these posts:")
        for t in undelivered:
            print(f"  - {t}")
        print("Check: LINE_CHANNEL_ACCESS_TOKEN secret / LINE monthly quota / "
              "whether anyone has enabled web push.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
