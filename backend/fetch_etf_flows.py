"""Fetch US spot BTC ETF daily net flows from SoSoValue into Supabase `etf_flows`.

Runs daily from GitHub Actions (.github/workflows/etf_flows.yml).
Env: SOSO_API_KEY, SUPABASE_URL, SUPABASE_KEY (service role).
SoSoValue returns the latest records (newest first, at most 300 per call);
rows are upserted, so re-running only refreshes recent days.
"""
import os
import sys

import requests
from supabase import create_client

API = "https://openapi.sosovalue.com/openapi/v1/etfs/summary-history"
SYMBOLS = ["BTC"]


def fetch(symbol: str, limit: int) -> list[dict]:
    res = requests.get(
        API,
        params={"symbol": symbol, "country_code": "US", "limit": limit},
        headers={"x-soso-api-key": os.environ["SOSO_API_KEY"]},
        timeout=30,
    )
    res.raise_for_status()
    body = res.json()
    if body.get("code") not in (0, "0", None):
        raise RuntimeError(f"SoSoValue error {body.get('code')}: {body.get('message')}")
    return body.get("data") or []


def num(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


TFTC = "https://www.tftc.io/bitcoin-etf-flows/data.json"


def backfill(db) -> None:
    """One-off: daily totals since the Jan 2024 launch from The Commoner Bitcoin ETF
    Tracker by TFTC (CC BY 4.0, tftc.io/bitcoin-etf-flows). Existing rows (SoSoValue)
    are kept; the cumulative total is a running sum of the daily flows."""
    days = requests.get(TFTC, timeout=60).json()["days"]
    days.sort(key=lambda d: d["date"])
    cum = 0.0
    rows = []
    for d in days:
        flow = num(d.get("netFlowUsd"))
        if flow is None:
            continue
        cum += flow
        rows.append({"symbol": "BTC", "date": d["date"], "net_inflow": flow,
                     "net_assets": num(d.get("totalNetAssetsUsd")), "cum_net_inflow": cum})
    for i in range(0, len(rows), 500):
        db.table("etf_flows").upsert(rows[i:i + 500], on_conflict="symbol,date", ignore_duplicates=True).execute()
    print(f"backfill: {len(rows)} days ({rows[0]['date']} ~ {rows[-1]['date']}), cumulative {cum / 1e9:.2f}B USD")


# US spot BTC ETFs. Daily dollar volume = shares traded x close, summed over the
# funds (Yahoo Finance daily bars), so the whole history has the same source.
ETF_TICKERS = ["IBIT", "FBTC", "GBTC", "BTC", "ARKB", "BITB", "HODL", "BRRR", "EZBC", "BTCO", "BTCW", "MSBT"]


def dollar_volume(rng: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for t in ETF_TICKERS:
        try:
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{t}",
                params={"range": rng, "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30,
            )
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            n = 0
            for ts, vol, close in zip(res.get("timestamp") or [], q.get("volume") or [], q.get("close") or []):
                if vol is None or close is None:
                    continue
                # Yahoo's daily bar timestamp is the US session open; the date part is the trading day
                day = __import__("datetime").datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                totals[day] = totals.get(day, 0.0) + vol * close
                n += 1
            print(f"  {t}: {n} days")
        except Exception as e:  # a fund missing (delisted / new) must not stop the rest
            print(f"  {t}: skipped ({e})")
    return totals


def update_value_traded(db, rng: str) -> None:
    totals = dollar_volume(rng)
    days = sorted(d for d in totals if d >= "2024-01-11")
    rows = [{"symbol": "BTC", "date": d, "value_traded": totals[d]} for d in days]
    for i in range(0, len(rows), 500):
        db.table("etf_flows").upsert(rows[i:i + 500], on_conflict="symbol,date").execute()
    if rows:
        print(f"value traded: {len(rows)} days ({days[0]} ~ {days[-1]}), latest {rows[-1]['value_traded'] / 1e9:.2f}B USD")


def main() -> int:
    limit = int(os.environ.get("ETF_LIMIT", "300"))
    db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    if os.environ.get("ETF_BACKFILL") == "true":
        backfill(db)
    for symbol in SYMBOLS:
        rows = [
            {
                "symbol": symbol,
                "date": r["date"],
                "net_inflow": num(r.get("total_net_inflow")),
                "value_traded": num(r.get("total_value_traded")),
                "net_assets": num(r.get("total_net_assets")),
                "cum_net_inflow": num(r.get("cum_net_inflow")),
            }
            for r in fetch(symbol, limit)
            if r.get("date")
        ]
        if not rows:
            print(f"{symbol}: no data returned")
            continue
        db.table("etf_flows").upsert(rows, on_conflict="symbol,date").execute()
        rows.sort(key=lambda r: r["date"])
        last = rows[-1]
        print(f"{symbol}: upserted {len(rows)} days ({rows[0]['date']} ~ {last['date']}); "
              f"latest net inflow {last['net_inflow'] / 1e6:+.1f}M USD")
    # after SoSoValue, so its own value_traded is replaced by the one-source Yahoo series
    update_value_traded(db, "5y" if os.environ.get("ETF_BACKFILL") == "true" else "1mo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
