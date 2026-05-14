"""
Notify subscribers about newly published forum posts.

Runs on a schedule (every 30 min via .github/workflows/notify_new_posts.yml)
or via workflow_dispatch.

Flow:
  1. Find posts where published=true AND notification_sent=false
  2. For each post:
     - Send LINE broadcast (all bot friends)
     - Send Web Push to premium users with push_subscription
  3. Mark notification_sent=true

If LINE_CHANNEL_ACCESS_TOKEN is unset, LINE step is skipped (web push still runs).
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

# VAPID — matches main.py
VAPID_PRIVATE_KEY = os.environ.get(
    'VAPID_PRIVATE_KEY',
    '0yUZd0zYl6aJz7NscDAnkKhvVQUaQEacKZHo3vLRI9o'
)
VAPID_CLAIMS = {"sub": "mailto:admin@smartdca.com"}

SITE_URL = os.environ.get('SITE_URL', 'https://dca.hellokai07.com')


def send_line_broadcast(message_text: str) -> bool:
    """Send a LINE broadcast. Returns True on success."""
    if not LINE_TOKEN:
        print("  [LINE] Skipped — no token configured.")
        return False
    try:
        from linebot.v3.messaging import (
            Configuration, ApiClient, MessagingApi,
            BroadcastRequest, TextMessage
        )
        config = Configuration(access_token=LINE_TOKEN)
        api_client = ApiClient(config)
        api = MessagingApi(api_client)
        api.broadcast(BroadcastRequest(messages=[TextMessage(text=message_text)]))
        print("  [LINE] Broadcast sent.")
        return True
    except Exception as e:
        print(f"  [LINE] Broadcast failed: {e}")
        return False


def send_web_push_to_premium(supabase: Client, post: dict) -> tuple[int, int]:
    """Send Web Push to all premium users with a subscription. Returns (success, fail)."""
    res = supabase.table('user_profiles') \
        .select('id, push_subscription') \
        .eq('is_premium', True) \
        .not_.is_('push_subscription', 'null') \
        .execute()

    users = [u for u in (res.data or []) if u.get('push_subscription')]
    if not users:
        print("  [Push] No premium subscribers.")
        return (0, 0)

    payload = json.dumps({
        "title": "📊 新技術分析文章",
        "body": post['title'],
        "url": f"{SITE_URL}/#/forum/{post['id']}"
    })

    success = 0
    fail = 0
    for user in users:
        try:
            webpush(
                subscription_info=user['push_subscription'],
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
            success += 1
        except WebPushException as e:
            fail += 1
            print(f"  [Push] Failed for user {user['id']}: {e}")

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

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    res = supabase.table('forum_posts') \
        .select('id, title, tags') \
        .eq('published', True) \
        .eq('notification_sent', False) \
        .order('created_at', desc=False) \
        .execute()

    posts = res.data or []
    if not posts:
        print("No unsent published posts. Nothing to do.")
        return

    print(f"Found {len(posts)} unsent post(s).")

    for post in posts:
        print(f"\n→ Processing: {post['title']} ({post['id']})")

        # 1. LINE
        line_text = build_line_message(post)
        send_line_broadcast(line_text)

        # 2. Web Push
        send_web_push_to_premium(supabase, post)

        # 3. Mark done — always, even if LINE/push partially failed (don't spam retries)
        try:
            supabase.table('forum_posts') \
                .update({'notification_sent': True}) \
                .eq('id', post['id']) \
                .execute()
            print("  [DB] Marked notification_sent=true.")
        except Exception as e:
            print(f"  [DB] Failed to mark as sent: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
