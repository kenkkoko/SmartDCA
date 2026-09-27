"""
Daily verification of forum 判讀 (calls).

Runs every morning via .github/workflows/verify_judgments.yml (or workflow_dispatch).

Only rows still `pending` are touched. A settled row is never recomputed: this script
filters on status = 'pending' AND settled_at IS NULL, and the database trigger
(supabase/forum_judgments_v2.sql) rejects any update to a settled row.

Per call:
  1. Once, on first sight: entry price at the lock time (close of the last finished
     1h bar, falling back to the daily close), the market regime and VPVR (POC/VAH/VAL)
     from the bars before the call, and the neutral band = 0.5 x ATR(14) / entry.
     These are stored and reused; they are not recalculated.
  2. Every run: walk the verification window (the first N bars of the call's timeframe
     after the lock time: 4H 30 / 1D 20 / 1W 13 / 1M 6) in order.
       - Invalidation first (conservative when both happen in the same bar):
           range + up   : a close below the range -> wrong
           range + down : a close above the range -> wrong
           range only   : a close outside the range -> wrong
       - Target touched (high >= target when bullish, low <= target when bearish) -> hit
     Settles at once on hit/wrong. When the window has fully closed:
           beyond the neutral band in the called direction  -> right
           range + direction and still inside the range      -> flat
           within the neutral band                          -> flat
           otherwise                                        -> wrong
       - Range-only calls that never closed outside the range -> right
  3. Withdrawn calls: anything that happened before the withdrawal still counts;
     otherwise settle as `withdrawn` at the price when it was withdrawn.

Env:
  SUPABASE_URL / SUPABASE_KEY   required (service_role key: the results table is admin-only)
  DRY_RUN=1                     compute and print, write nothing
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

TF = {
    '4h': {'bars': 30, 'interval': '1h', 'resample': '4h', 'dur': pd.Timedelta(hours=4), 'lookback': timedelta(days=60)},
    '1d': {'bars': 20, 'interval': '1d', 'resample': None, 'dur': pd.Timedelta(days=1), 'lookback': timedelta(days=200)},
    '1w': {'bars': 13, 'interval': '1wk', 'resample': None, 'dur': pd.Timedelta(weeks=1), 'lookback': timedelta(days=900)},
    '1M': {'bars': 6, 'interval': '1mo', 'resample': None, 'dur': None, 'lookback': timedelta(days=3700)},
}
INTRADAY_LIMIT = timedelta(days=700)   # yfinance serves 1h bars for about the last 730 days


# ── Market data ────────────────────────────────────────────────────────────

def yf_tickers(market, symbol):
    s = symbol.strip().upper()
    if market == 'crypto':
        return [s if '-' in s else f'{s}-USD']
    if market == 'tw':
        return [s] if '.' in s else [f'{s}.TW', f'{s}.TWO']
    return [s]


def _download(ticker, interval, start, end):
    df = yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=False, actions=False)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']].dropna(subset=['close'])
    idx = df.index
    df.index = idx.tz_convert('UTC') if idx.tz is not None else idx.tz_localize('UTC')
    return df


def fetch(market, symbol, interval, start, end):
    """Raw (unadjusted) OHLCV with a UTC index; tries each ticker spelling in turn."""
    for t in yf_tickers(market, symbol):
        try:
            df = _download(t, interval, start, end)
        except Exception as e:  # noqa: BLE001 — yfinance raises many types
            print(f'    {t} {interval}: {e}')
            df = None
        if df is not None and len(df):
            return df
    return None


def bars_for(market, symbol, tf, start, end):
    cfg = TF[tf]
    df = fetch(market, symbol, cfg['interval'], start, end)
    if df is None:
        return None
    if cfg['resample']:
        df = df.resample(cfg['resample'], origin='epoch').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna(subset=['close'])
    return df


def bar_ends(index, tf):
    if tf == '1M':
        return index + pd.DateOffset(months=1)
    return index + TF[tf]['dur']


def price_at(market, symbol, ts, now):
    """Close of the last bar that had finished by `ts` (1h when available, else daily)."""
    if now - ts < INTRADAY_LIMIT:
        df = fetch(market, symbol, '1h', ts - timedelta(days=10), ts + timedelta(days=1))
        if df is not None:
            done = df[df.index + pd.Timedelta(hours=1) <= ts]
            if len(done):
                return float(done['close'].iloc[-1]), done.index[-1] + pd.Timedelta(hours=1)
    df = fetch(market, symbol, '1d', ts - timedelta(days=20), ts + timedelta(days=1))
    if df is not None:
        done = df[df.index + pd.Timedelta(days=1) <= ts]
        if len(done):
            return float(done['close'].iloc[-1]), done.index[-1] + pd.Timedelta(days=1)
    return None, None


# ── Indicators from the bars before the call ───────────────────────────────

def atr(df, n=14):
    if len(df) < n + 1:
        return None
    prev = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - prev).abs(), (df['low'] - prev).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def regime(df):
    """up / down / range from EMA20 vs EMA50 and the EMA50 slope."""
    if len(df) < 55:
        return None
    c = df['close']
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    last, slope = c.iloc[-1], e50.iloc[-1] - e50.iloc[-6]
    if last > e50.iloc[-1] and e20.iloc[-1] > e50.iloc[-1] and slope > 0:
        return 'up'
    if last < e50.iloc[-1] and e20.iloc[-1] < e50.iloc[-1] and slope < 0:
        return 'down'
    return 'range'


def vpvr(df, bins=50, value_area=0.7):
    """Volume profile over the given bars: each bar's volume spread evenly over its high-low range."""
    df = df[df['volume'] > 0]
    if len(df) < 10:
        return None
    lo, hi = float(df['low'].min()), float(df['high'].max())
    if not hi > lo:
        return None
    step = (hi - lo) / bins
    vol = [0.0] * bins
    for h, l, v in zip(df['high'], df['low'], df['volume']):
        a = min(bins - 1, int((l - lo) / step))
        b = min(bins - 1, int((h - lo) / step))
        share = float(v) / (b - a + 1)
        for k in range(a, b + 1):
            vol[k] += share
    poc = max(range(bins), key=lambda k: vol[k])
    total, acc, a, b = sum(vol), vol[poc], poc, poc
    while acc < total * value_area and (a > 0 or b < bins - 1):
        down = vol[a - 1] if a > 0 else -1
        up = vol[b + 1] if b < bins - 1 else -1
        if up >= down:
            b += 1; acc += vol[b]
        else:
            a -= 1; acc += vol[a]
    mid = lambda k: lo + (k + 0.5) * step  # noqa: E731
    return {'poc': round(mid(poc), 8), 'vah': round(lo + (b + 1) * step, 8), 'val': round(lo + a * step, 8), 'bars': int(len(df))}


# ── Scoring ────────────────────────────────────────────────────────────────

def evaluate(j, win, ends, entry, neutral, now, stop=None):
    """Score one call over its window. `win` = the window bars (already cut to N),
    `ends` = their end times, `stop` = withdrawal time (bars starting after it are ignored)."""
    dirs = set(j['directions'])
    up, down, rng = 'up' in dirs, 'down' in dirs, 'range' in dirs
    total = TF[j['timeframe']]['bars']
    lo, hi = j.get('range_low'), j.get('range_high')
    target = j.get('target_price')
    if target is not None and not ((up and target > entry) or (down and target < entry)):
        target = None  # a target on the wrong side of entry cannot be "hit"
    sign = -1 if down else 1

    if stop is not None:
        keep = win.index < stop
        win, ends = win[keep], ends[keep]

    out = {'status': 'pending', 'exit_price': None, 'settled_bar': None}
    for (ts, row), end in zip(win.iterrows(), ends):
        c = row['close']
        broke = (rng and up and c < lo) or (rng and down and c > hi) or (rng and not up and not down and (c < lo or c > hi))
        if broke:
            out.update(status='wrong', exit_price=float(c), settled_bar=ts)
            break
        if target is not None and ((up and row['high'] >= target) or (down and row['low'] <= target)):
            out.update(status='hit', exit_price=float(target), settled_bar=ts)
            break

    seen = win if out['settled_bar'] is None else win[win.index <= out['settled_bar']]
    done = int((ends <= now).sum())
    last = float(seen['close'].iloc[-1]) if len(seen) else None

    if out['status'] == 'pending' and stop is None and done >= total and len(win) >= total:
        close = float(win['close'].iloc[total - 1])
        chg = (close - entry) / entry * sign
        if rng and not up and not down:
            status = 'right'
        elif chg > neutral:
            status = 'right'
        elif rng and lo <= close <= hi:
            status = 'flat'
        elif abs(chg) <= neutral:
            status = 'flat'
        else:
            status = 'wrong'
        out.update(status=status, exit_price=close, settled_bar=win.index[total - 1])
        last = close

    progress = None
    if target is not None and len(seen):
        best = seen['high'].max() if up else seen['low'].min()
        progress = max(0.0, (best - entry) / (target - entry))

    out.update(
        bars_elapsed=min(done, total),
        last_price=last,
        change_pct=None if last is None else (last - entry) / entry * sign,
        max_progress=progress,
    )
    return out


def level_events(levels, win):
    events = []
    for lv in levels or []:
        p = lv.get('price')
        if p is None:
            continue
        touched = win[(win['low'] <= p) & (win['high'] >= p)]
        events.append({'type': lv.get('type'), 'price': p,
                       'touched_at': touched.index[0].isoformat() if len(touched) else None})
    return events


def clean(v):
    """JSON-safe scalars (numpy → float, NaN → None, rounded)."""
    if v is None:
        return None
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if math.isnan(f) or math.isinf(f) else round(f, 8)


def to_ts(s):
    return pd.Timestamp(s).tz_convert('UTC') if pd.Timestamp(s).tzinfo else pd.Timestamp(s).tz_localize('UTC')


# ── One call ───────────────────────────────────────────────────────────────

def verify(j, res, now):
    tf = j['timeframe']
    cfg = TF[tf]
    locked = to_ts(j['locked_at'])
    withdrawn = to_ts(j['withdrawn_at']) if j.get('withdrawn_at') else None
    if tf == '4h' and now - locked > INTRADAY_LIMIT:
        return {'note': '4H 判讀超過兩年，免費行情已無小時資料，無法驗證'}

    bars = bars_for(j['market'], j['symbol'], tf, locked - cfg['lookback'], now + timedelta(days=1))
    if bars is None:
        return {'note': f"抓不到 {j['symbol']} 的行情，請確認市場與代號"}
    ends = bar_ends(bars.index, tf)
    prior = bars[ends <= locked]
    after = bars.index >= locked
    win, win_ends = bars[after][:cfg['bars']], ends[after][:cfg['bars']]

    # Fixed once, on first sight
    row = {}
    entry = res.get('entry_price')
    if entry is None:
        entry, entry_at = price_at(j['market'], j['symbol'], locked, now)
        if entry is None:
            return {'note': '找不到判讀當時的價格'}
        a = atr(prior)
        row.update(
            entry_price=entry,
            entry_at=entry_at.isoformat(),
            bars_total=cfg['bars'],
            regime=regime(prior),
            vpvr=vpvr(prior.iloc[-100:]),
            neutral_pct=max(0.002, 0.5 * a / entry) if a else 0.01,
        )
    neutral = row.get('neutral_pct', res.get('neutral_pct')) or 0.01

    r = evaluate(j, win, win_ends, entry, neutral, now, stop=withdrawn)
    if withdrawn is not None and r['status'] == 'pending':
        exit_price, _ = price_at(j['market'], j['symbol'], withdrawn, now)
        if exit_price is None:
            return {**row, 'note': '找不到撤回當時的價格'}
        sign = -1 if 'down' in j['directions'] else 1
        r.update(status='withdrawn', exit_price=exit_price, settled_bar=withdrawn,
                 last_price=exit_price, change_pct=(exit_price - entry) / entry * sign)

    row.update(
        bars_elapsed=r['bars_elapsed'],
        last_price=r['last_price'],
        change_pct=r['change_pct'],
        max_progress=r['max_progress'],
        level_events=level_events(j.get('levels'), win),
        note=None,
    )
    if r['status'] != 'pending':
        row.update(status=r['status'], exit_price=r['exit_price'], settled_at=now.isoformat())
    return {k: (v if isinstance(v, (dict, list, str)) or v is None else clean(v)) for k, v in row.items()}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    url, key = os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_KEY')
    dry = os.environ.get('DRY_RUN', '').lower() in ('1', 'true', 'yes')
    if not url or not key:
        print('SUPABASE_URL / SUPABASE_KEY 未設定')
        sys.exit(1)
    from supabase import create_client
    sb = create_client(url, key)

    rows = (sb.table('forum_judgment_results')
              .select('*, forum_judgments(*)')
              .eq('status', 'pending')
              .is_('settled_at', 'null')
              .execute().data)
    print(f'待驗證 {len(rows)} 則{"（DRY RUN）" if dry else ""}')
    now = pd.Timestamp(datetime.now(timezone.utc))

    settled = errors = 0
    for res in rows:
        j = res.get('forum_judgments')
        if not j or not j.get('locked_at'):
            continue
        label = f"{j['symbol']} {j['timeframe']} {'+'.join(j['directions'])}"
        try:
            row = verify(j, res, now)
        except Exception as e:  # noqa: BLE001 — one bad call must not stop the rest
            row = {'note': f'驗證失敗：{e}'[:300]}
        if row.get('note'):
            errors += 1
        if row.get('settled_at'):
            settled += 1
        print(f"  {label}: {row.get('status', 'pending')} "
              f"進度 {row.get('bars_elapsed', '-')}/{TF[j['timeframe']]['bars']} "
              f"漲跌 {row.get('change_pct')} {row.get('note') or ''}")
        if not dry:
            (sb.table('forum_judgment_results')
               .update(row)
               .eq('judgment_id', res['judgment_id'])
               .is_('settled_at', 'null')
               .execute())

    print(f'完成：結算 {settled} 則，問題 {errors} 則')


if __name__ == '__main__':
    main()
