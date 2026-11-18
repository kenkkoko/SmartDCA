"""Print the structure of TFTC's ETF flow history JSON (one-off, to write the backfill)."""
import json
import requests

r = requests.get("https://www.tftc.io/bitcoin-etf-flows/data.json", timeout=60)
print("status", r.status_code, "bytes", len(r.content))
d = r.json()

def show(x, depth=0, key="root"):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{key}: dict keys={list(x.keys())[:30]}")
        if depth < 3:
            for k in list(x.keys())[:8]:
                show(x[k], depth + 1, k)
    elif isinstance(x, list):
        print(f"{pad}{key}: list len={len(x)}")
        if x and depth < 3:
            show(x[0], depth + 1, f"{key}[0]")
            print(f"{pad}  {key}[-1] = {json.dumps(x[-1])[:400]}")
    else:
        print(f"{pad}{key}: {type(x).__name__} = {str(x)[:120]}")

show(d)
