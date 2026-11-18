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


def main() -> int:
    limit = int(os.environ.get("ETF_LIMIT", "300"))
    db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
