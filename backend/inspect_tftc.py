"""Print the structure of TFTC's ETF flow history JSON (one-off, to write the backfill)."""
import json
import requests

d = requests.get("https://www.tftc.io/bitcoin-etf-flows/data.json", timeout=60).json()
for k in ("units", "totalDefinition", "updatedThrough"):
    print(k, "=", json.dumps(d.get(k))[:500])
days = d["days"]
print("days type", type(days).__name__, "len", len(days))
items = list(days.items())[:2] if isinstance(days, dict) else days[:2]
print("first", json.dumps(items)[:800])
last = list(days.items())[-1] if isinstance(days, dict) else days[-1]
print("last", json.dumps(last)[:800])
