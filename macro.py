"""
總經模組:經濟日曆比對 + Alpha Vantage 總經數據 + Supabase macro_snapshot

main.py 用法:
    snapshot = fetch_macro_snapshot(ALPHA_VANTAGE_KEY)       # 5 次 API 呼叫
    calendar_text = format_calendar_section(snapshot)        # 推播用(含 emoji)
    ai_context = get_macro_context(snapshot)                 # AI prompt 用(純文字)
    save_macro_snapshot(supabase, snapshot)                  # 給 gemini-proxy 讀
"""
import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
TW = ZoneInfo("Asia/Taipei")
CALENDAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "economic_calendar.json")
WEEKDAYS = "一二三四五六日"
IMPACT_ICON = {"high": "🔴", "medium": "🟠", "low": "⚪"}


# ---------------------------------------------------------------------------
# 經濟日曆
# ---------------------------------------------------------------------------
def load_calendar(path=CALENDAR_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["events"]
    except Exception as e:
        print(f"Error loading economic calendar: {e}")
        return []


def event_time_tw(event):
    """美東時間 -> 台灣時間(自動處理美國日光節約時間)"""
    d = dt.date.fromisoformat(event["date"])
    hh, mm = map(int, event["time_et"].split(":"))
    return dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(TW)


def get_upcoming_events(now=None, days=7, events=None):
    """回傳 (24 小時內事件, 之後到 days 天內的事件),每筆附上 tw 時間"""
    now = now or dt.datetime.now(TW)
    events = load_calendar() if events is None else events
    soon, later = [], []
    for ev in events:
        t = event_time_tw(ev)
        if now <= t < now + dt.timedelta(hours=24):
            soon.append({**ev, "tw": t})
        elif now + dt.timedelta(hours=24) <= t < now + dt.timedelta(days=days):
            later.append({**ev, "tw": t})
    key = lambda e: e["tw"]
    return sorted(soon, key=key), sorted(later, key=key)


def _previous_value_hint(ev, snapshot):
    """事件對應的前值(讓推播從「有 CPI」變成「CPI 前值 X%」)"""
    if not snapshot:
        return ""
    if ev["type"] == "CPI" and snapshot.get("cpi_yoy") is not None:
        return f"前值年增 {snapshot['cpi_yoy']:.1f}%"
    if ev["type"] == "NFP" and snapshot.get("unemployment") is not None:
        return f"前值失業率 {snapshot['unemployment']:.1f}%"
    if ev["type"] == "FOMC" and snapshot.get("fed_funds") is not None:
        return f"目前有效利率 {snapshot['fed_funds']:.2f}%"
    return ""


def _title(ev):
    return ev["title"] + ("(暫定)" if not ev.get("confirmed", True) else "")


def format_calendar_section(snapshot=None, now=None, events=None):
    """推播用的日曆區塊;沒有事件時回傳空字串"""
    now = now or dt.datetime.now(TW)
    soon, later = get_upcoming_events(now, events=events)
    if not soon and not later:
        return ""

    lines = ["📅 經濟日曆 (台灣時間):"]
    if soon:
        lines.append("   【24 小時內】")
        for ev in soon:
            when = ev["tw"].strftime("%H:%M")
            if ev["tw"].date() != now.date():
                when = f"明日{'凌晨' if ev['tw'].hour < 6 else ''} {when}"
            hint = _previous_value_hint(ev, snapshot)
            lines.append(f"   {IMPACT_ICON.get(ev['impact'], '⚪')} {when} {_title(ev)}" + (f"|{hint}" if hint else ""))
        if any(ev["impact"] == "high" for ev in soon):
            lines.append("   ⚠️ 數據公布前後波動較大，大額單筆加碼建議分批")
    if later:
        lines.append("   【本週後續】")
        for ev in later:
            t = ev["tw"]
            lines.append(f"   {IMPACT_ICON.get(ev['impact'], '⚪')} {t.month}/{t.day}({WEEKDAYS[t.weekday()]}) {t:%H:%M} {_title(ev)}")
    return "\n".join(lines)


def get_calendar_context(now=None, events=None):
    """AI prompt 用的日曆純文字(無 emoji)"""
    now = now or dt.datetime.now(TW)
    soon, later = get_upcoming_events(now, events=events)
    if not soon and not later:
        return "未來 7 天無重大美國經濟數據公布。"
    rows = []
    for label, group in (("24 小時內", soon), ("7 天內", later)):
        for ev in group:
            t = ev["tw"]
            impact = {"high": "高影響", "medium": "中影響"}.get(ev["impact"], "低影響")
            rows.append(f"- [{label}] {t.month}/{t.day} {t:%H:%M} 台灣時間:{_title(ev)}({impact})")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Alpha Vantage 總經數據(免費方案 25 次/天,這裡每天用 5 次)
# ---------------------------------------------------------------------------
AV_URL = "https://www.alphavantage.co/query"


def _av_series(api_key, params):
    """回傳 [(date, float), ...] 新到舊;失敗回傳 []"""
    try:
        r = requests.get(AV_URL, params={**params, "apikey": api_key}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if "data" not in data:
            # 額度用完或參數錯誤時 AV 仍回 200,訊息放在 Information / Note / Error Message
            msg = data.get("Information") or data.get("Note") or data.get("Error Message") or data
            print(f"Alpha Vantage {params.get('function')} 無資料: {msg}")
            return []
        out = []
        for row in data["data"]:
            try:
                out.append((dt.date.fromisoformat(row["date"]), float(row["value"])))
            except (ValueError, KeyError):
                continue  # 假日的值是 "."
        return out
    except Exception as e:
        print(f"Error fetching Alpha Vantage {params.get('function')}: {e}")
        return []


def _yoy(series, i):
    """series[i] 相對 12 個月前的年增率(%)"""
    if len(series) <= i:
        return None
    d, v = series[i]
    base = dict(series).get(d.replace(year=d.year - 1))
    return round((v / base - 1) * 100, 2) if base else None


def fetch_macro_snapshot(api_key, pause=1.5):
    """抓 10Y/2Y 殖利率、有效聯邦基金利率、CPI、失業率。全部失敗時回傳 None"""
    if not api_key:
        print("Skipping macro data: ALPHA_VANTAGE_KEY not set.")
        return None

    calls = [
        ("us10y", {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year"}),
        ("us2y", {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "2year"}),
        ("fed_funds", {"function": "FEDERAL_FUNDS_RATE", "interval": "daily"}),
        ("cpi", {"function": "CPI", "interval": "monthly"}),
        ("unemployment", {"function": "UNEMPLOYMENT"}),
    ]
    raw = {}
    for i, (name, params) in enumerate(calls):
        if i:
            time.sleep(pause)
        raw[name] = _av_series(api_key, params)

    if not any(raw.values()):
        return None

    def latest(name):
        return raw[name][0][1] if raw[name] else None

    def prev(name):
        return raw[name][1][1] if len(raw[name]) > 1 else None

    def as_of(name):
        return raw[name][0][0].isoformat() if raw[name] else None

    us10y, us2y = latest("us10y"), latest("us2y")
    snap = {
        "date": dt.datetime.now(TW).date().isoformat(),
        "us10y": us10y,
        "us2y": us2y,
        "spread": round(us10y - us2y, 2) if us10y is not None and us2y is not None else None,
        "yields_as_of": as_of("us10y"),
        "fed_funds": latest("fed_funds"),
        "cpi_yoy": _yoy(raw["cpi"], 0),
        "cpi_yoy_prev": _yoy(raw["cpi"], 1),
        "cpi_yoy_prev2": _yoy(raw["cpi"], 2),
        "cpi_month": as_of("cpi"),
        "unemployment": latest("unemployment"),
        "unemployment_prev": prev("unemployment"),
        "unemployment_month": as_of("unemployment"),
    }
    # 10Y 一個月前(約 21 個交易日)的值,用來判斷殖利率方向
    snap["us10y_1m_ago"] = raw["us10y"][21][1] if len(raw["us10y"]) > 21 else None
    return snap


# ---------------------------------------------------------------------------
# 文字輸出
# ---------------------------------------------------------------------------
def _arrow(cur, prev, eps=0.05):
    if cur is None or prev is None:
        return ""
    if cur > prev + eps:
        return "↑"
    if cur < prev - eps:
        return "↓"
    return "→"


def _cpi_trend_text(s):
    a, b, c = s.get("cpi_yoy"), s.get("cpi_yoy_prev"), s.get("cpi_yoy_prev2")
    if None in (a, b, c):
        return ""
    if a < b < c:
        return "連兩月回落"
    if a > b > c:
        return "連兩月升溫"
    return ""


def format_macro_section(snapshot):
    """推播用的總經區塊"""
    if not snapshot:
        return ""
    s = snapshot
    lines = ["🏛️ 美國總經:"]
    if s.get("us10y") is not None:
        y = f"   美債 10Y {s['us10y']:.2f}%"
        if s.get("us2y") is not None:
            tag = "倒掛" if s["spread"] < 0 else "正斜率"
            y += f" / 2Y {s['us2y']:.2f}%(利差 {s['spread']:+.2f},{tag})"
        lines.append(y)
    if s.get("fed_funds") is not None:
        lines.append(f"   聯邦基金有效利率 {s['fed_funds']:.2f}%")
    if s.get("cpi_yoy") is not None:
        trend = _cpi_trend_text(s)
        prev = f"前值 {s['cpi_yoy_prev']:.1f}%" if s.get("cpi_yoy_prev") is not None else ""
        extra = ",".join(x for x in (prev, trend) if x)
        lines.append(f"   CPI 年增 {s['cpi_yoy']:.1f}% {_arrow(s['cpi_yoy'], s.get('cpi_yoy_prev'))}" + (f"({extra})" if extra else ""))
    if s.get("unemployment") is not None:
        prev = f"(前值 {s['unemployment_prev']:.1f}%)" if s.get("unemployment_prev") is not None else ""
        lines.append(f"   失業率 {s['unemployment']:.1f}% {_arrow(s['unemployment'], s.get('unemployment_prev'))}{prev}")
    return "\n".join(lines) if len(lines) > 1 else ""


def get_macro_context(snapshot, now=None, events=None):
    """AI prompt 用的總經 + 日曆純文字(main.py 每日推播的 AI 建議使用)"""
    macro = get_macro_only_context(snapshot)
    calendar = "未來 7 天經濟事件:\n" + get_calendar_context(now, events)
    return f"{macro}\n\n{calendar}" if macro else calendar


def get_macro_only_context(snapshot):
    """總經數值純文字;存進 macro_snapshot.macro_context 給 gemini-proxy 使用"""
    parts = []
    s = snapshot or {}
    if s:
        m = []
        if s.get("us10y") is not None:
            m.append(f"- 美國 10 年期公債殖利率 {s['us10y']:.2f}%"
                     + (f"(一個月前 {s['us10y_1m_ago']:.2f}%)" if s.get("us10y_1m_ago") is not None else ""))
        if s.get("spread") is not None:
            m.append(f"- 10Y-2Y 利差 {s['spread']:+.2f} 個百分點" + ("(殖利率曲線倒掛)" if s["spread"] < 0 else ""))
        if s.get("fed_funds") is not None:
            m.append(f"- 聯邦基金有效利率 {s['fed_funds']:.2f}%")
        if s.get("cpi_yoy") is not None:
            hist = [s.get(k) for k in ("cpi_yoy_prev2", "cpi_yoy_prev", "cpi_yoy")]
            m.append(f"- CPI 年增率 {s['cpi_yoy']:.1f}%(近三個月:"
                     + " → ".join(f"{v:.1f}%" for v in hist if v is not None) + ")")
        if s.get("unemployment") is not None:
            m.append(f"- 失業率 {s['unemployment']:.1f}%"
                     + (f"(前值 {s['unemployment_prev']:.1f}%)" if s.get("unemployment_prev") is not None else ""))
        if m:
            parts.append(f"美國總經數據(截至 {s.get('date')}):\n" + "\n".join(m))
    return "\n\n".join(parts)


def get_upcoming_events_payload(now=None, days=10, events=None):
    """未來 days 天事件(台灣時間 ISO 格式),存進 macro_snapshot.upcoming_events。
    gemini-proxy 依請求當下時間自行過濾,已公布的事件不會被當成「即將公布」。"""
    now = now or dt.datetime.now(TW)
    soon, later = get_upcoming_events(now, days=days, events=events)
    return [
        {"at": ev["tw"].isoformat(), "type": ev["type"], "title": _title(ev), "impact": ev["impact"]}
        for ev in soon + later
    ]


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SNAPSHOT_COLUMNS = (
    "date", "us10y", "us2y", "spread", "yields_as_of", "us10y_1m_ago", "fed_funds",
    "cpi_yoy", "cpi_yoy_prev", "cpi_yoy_prev2", "cpi_month",
    "unemployment", "unemployment_prev", "unemployment_month",
)


def save_macro_snapshot(supabase, snapshot, now=None):
    """upsert 今天這筆(同一天重跑會覆蓋)。需要 service role key。"""
    if supabase is None or not snapshot:
        return
    try:
        row = {k: snapshot.get(k) for k in SNAPSHOT_COLUMNS}
        row["macro_context"] = get_macro_only_context(snapshot)
        row["upcoming_events"] = get_upcoming_events_payload(now)
        row["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        supabase.table("macro_snapshot").upsert(row, on_conflict="date").execute()
        print("macro_snapshot saved.")
    except Exception as e:
        print(f"Error saving macro_snapshot: {e}")
