"""
產生 economic_calendar.json(repo 根目錄)

用法:  python py/gen_calendar.py

資料來源(2026 年全部為官方時程,confirmed=true):
  - FOMC:     https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  - CPI:      https://www.bls.gov/schedule/news_release/cpi.htm
  - 非農:     https://www.bls.gov/schedule/news_release/empsit.htm
  - GDP/PCE:  https://www.bea.gov/news/schedule
  - 零售銷售: https://www.census.gov/retail/release_schedule.html

2027 年:FOMC 為聯準會公布的暫定日期(confirmed=true,聯準會自註 tentative);
其餘機構 2027 時程通常在前一年第四季才發布,這裡用規則推估(confirmed=false),
程式會顯示「(暫定)」。官方公布後把日期補進 OFFICIAL_* 再重跑即可。

時間一律存「美東時間」,台灣時間由讀取端依日光節約時間即時換算。
"""
import datetime as dt
import json
import os

D = dt.date
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "economic_calendar.json")
START, END = D(2026, 9, 1), D(2027, 12, 31)

# ---------------------------------------------------------------------------
# 官方時程
# ---------------------------------------------------------------------------
# FOMC:(會議第二天 = 決議公布日, 是否有 SEP 經濟預測)
OFFICIAL_FOMC = [
    (D(2026, 1, 28), False), (D(2026, 3, 18), True), (D(2026, 4, 29), False),
    (D(2026, 6, 17), True), (D(2026, 7, 29), False), (D(2026, 9, 16), True),
    (D(2026, 10, 28), False), (D(2026, 12, 9), True),
    (D(2027, 1, 27), False), (D(2027, 3, 17), True), (D(2027, 4, 28), False),
    (D(2027, 6, 9), True), (D(2027, 7, 28), False), (D(2027, 9, 15), True),
    (D(2027, 10, 27), False), (D(2027, 12, 8), True),
]
# CPI:參考月 -> 公布日
OFFICIAL_CPI = {
    (2025, 12): D(2026, 1, 13), (2026, 1): D(2026, 2, 13), (2026, 2): D(2026, 3, 11),
    (2026, 3): D(2026, 4, 10), (2026, 4): D(2026, 5, 12), (2026, 5): D(2026, 6, 10),
    (2026, 6): D(2026, 7, 14), (2026, 7): D(2026, 8, 12), (2026, 8): D(2026, 9, 11),
    (2026, 9): D(2026, 10, 14), (2026, 10): D(2026, 11, 10), (2026, 11): D(2026, 12, 10),
}
# 非農:參考月 -> 公布日
OFFICIAL_NFP = {
    (2025, 12): D(2026, 1, 9), (2026, 1): D(2026, 2, 11), (2026, 2): D(2026, 3, 6),
    (2026, 3): D(2026, 4, 3), (2026, 4): D(2026, 5, 8), (2026, 5): D(2026, 6, 5),
    (2026, 6): D(2026, 7, 2), (2026, 7): D(2026, 8, 7), (2026, 8): D(2026, 9, 4),
    (2026, 9): D(2026, 10, 2), (2026, 10): D(2026, 11, 6), (2026, 11): D(2026, 12, 4),
}
# 零售銷售:參考月 -> 公布日
OFFICIAL_RETAIL = {
    (2026, 4): D(2026, 5, 14), (2026, 5): D(2026, 6, 17), (2026, 6): D(2026, 7, 16),
    (2026, 7): D(2026, 8, 14), (2026, 8): D(2026, 9, 16), (2026, 9): D(2026, 10, 15),
    (2026, 10): D(2026, 11, 17), (2026, 11): D(2026, 12, 16),
}
# PCE(個人所得與支出):參考月 -> 公布日
OFFICIAL_PCE = {
    (2026, 8): D(2026, 9, 30), (2026, 9): D(2026, 10, 29),
    (2026, 10): D(2026, 11, 25), (2026, 11): D(2026, 12, 23),
}
# GDP:(公布日, 標題)
OFFICIAL_GDP = [
    (D(2026, 9, 30), "美國 Q2 GDP 終值"),
    (D(2026, 10, 29), "美國 Q3 GDP 初值"),
    (D(2026, 11, 25), "美國 Q3 GDP 修正值"),
    (D(2026, 12, 23), "美國 Q3 GDP 終值"),
]

# ---------------------------------------------------------------------------
# 聯邦假日(推估時避開)
# ---------------------------------------------------------------------------
def nth_weekday(y, m, weekday, n):
    """當月第 n 個星期 weekday(Mon=0);n=-1 表示最後一個"""
    if n > 0:
        d = D(y, m, 1)
        d += dt.timedelta((weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    nxt = D(y + (m == 12), m % 12 + 1, 1)
    d = nxt - dt.timedelta(1)
    return d - dt.timedelta((d.weekday() - weekday) % 7)


def observed(d):
    if d.weekday() == 5:
        return d - dt.timedelta(1)
    if d.weekday() == 6:
        return d + dt.timedelta(1)
    return d


def federal_holidays(y):
    return {
        observed(D(y, 1, 1)), nth_weekday(y, 1, 0, 3), nth_weekday(y, 2, 0, 3),
        nth_weekday(y, 5, 0, -1), observed(D(y, 6, 19)), observed(D(y, 7, 4)),
        nth_weekday(y, 9, 0, 1), nth_weekday(y, 10, 0, 2), observed(D(y, 11, 11)),
        nth_weekday(y, 11, 3, 4), observed(D(y, 12, 25)),
    }


HOLIDAYS = federal_holidays(2025) | federal_holidays(2026) | federal_holidays(2027) | federal_holidays(2028)


def is_business_day(d):
    return d.weekday() < 5 and d not in HOLIDAYS


def next_business_day(d):
    while not is_business_day(d):
        d += dt.timedelta(1)
    return d


def prev_business_day(d):
    while not is_business_day(d):
        d -= dt.timedelta(1)
    return d


def next_month(y, m):
    return (y + (m == 12), m % 12 + 1)


# ---------------------------------------------------------------------------
# 推估規則(僅用於官方尚未公布的月份)
# ---------------------------------------------------------------------------
def estimate_nfp(y, m):
    """BLS 規則:含當月 12 日那週(週日~週六)結束後的第三個週五。
    遇假日提前一個營業日;落在 1 月前兩天(年假)則延後一週。"""
    d12 = D(y, m, 12)
    week_end = d12 + dt.timedelta((5 - d12.weekday()) % 7)  # 週六
    release = week_end + dt.timedelta(6 + 14)  # 第三個週五
    if release.month == 1 and release.day <= 2:
        return release + dt.timedelta(7)
    if not is_business_day(release):
        return prev_business_day(release)
    return release


def estimate_cpi(y, m):
    """BLS 沒有硬規則。經驗法則:非農公布後下一週的週三,
    但不早於當月 10 日;遇假日順延。2026 年回測誤差多在 ±2 天內。"""
    ny, nm = next_month(y, m)
    nfp = OFFICIAL_NFP.get((y, m)) or estimate_nfp(y, m)
    wed = nfp + dt.timedelta((2 - nfp.weekday()) % 7 or 7)
    if wed.day < 10 and wed.month == nm:
        wed = D(ny, nm, 10)
    return next_business_day(wed)


def estimate_retail(y, m):
    """Census:次月 15 日起第一個營業日"""
    ny, nm = next_month(y, m)
    return next_business_day(D(ny, nm, 15))


def estimate_pce(y, m):
    """BEA 2026 起與 GDP 同日公布,大致在次月最後一週;取次月最後一個週四(遇假日提前)。
    12 月避開聖誕週,取 12/23 前最近營業日(2026 官方為 12/23)。"""
    ny, nm = next_month(y, m)
    if nm == 12:
        return prev_business_day(D(ny, 12, 23))
    return prev_business_day(nth_weekday(ny, nm, 3, -1))


QUARTER_NAME = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}


def estimate_gdp():
    """GDP 初值:季結束後次月最後一個週四;修正值、終值各再晚約一個月(與 PCE 同日)"""
    out = []
    for y in (2026, 2027):
        for q in (1, 2, 3, 4):
            qy, qm = (y + 1, 1) if q == 4 else (y, q * 3 + 1)
            for k, label in enumerate(("初值", "修正值", "終值")):
                my, mm = qy, qm + k
                if mm > 12:
                    my, mm = my + 1, mm - 12
                pm_y, pm_m = (my - 1, 12) if mm == 1 else (my, mm - 1)
                out.append((estimate_pce(pm_y, pm_m), f"美國 {QUARTER_NAME[q]} GDP {label}"))
    return out


# ---------------------------------------------------------------------------
def month_label(m):
    return f"{m} 月"


def monthly_title(m, name):
    return f"美國 {month_label(m)}{name}"


def build():
    events = []

    def add(date, etype, title, time_et, impact, confirmed, note=None):
        if START <= date <= END:
            ev = {
                "date": date.isoformat(),
                "time_et": time_et,
                "type": etype,
                "title": title,
                "impact": impact,
                "confirmed": confirmed,
            }
            if note:
                ev["note"] = note
            events.append(ev)

    for d, sep in OFFICIAL_FOMC:
        add(d, "FOMC", "FOMC 利率決議" + ("(含經濟預測 SEP)" if sep else ""),
            "14:00", "high", True, "決議公布 30 分鐘後主席記者會")

    months = []
    y, m = 2026, 7
    while (y, m) <= (2027, 11):
        months.append((y, m))
        y, m = next_month(y, m)

    for y, m in months:
        cpi = OFFICIAL_CPI.get((y, m))
        add(cpi or estimate_cpi(y, m), "CPI", monthly_title(m, " CPI"), "08:30", "high", cpi is not None)

        nfp = OFFICIAL_NFP.get((y, m))
        add(nfp or estimate_nfp(y, m), "NFP", monthly_title(m, "非農就業"), "08:30", "high", nfp is not None)

        rs = OFFICIAL_RETAIL.get((y, m))
        add(rs or estimate_retail(y, m), "RETAIL", monthly_title(m, "零售銷售"), "08:30", "medium", rs is not None)

        pce = OFFICIAL_PCE.get((y, m))
        add(pce or estimate_pce(y, m), "PCE", monthly_title(m, " PCE 物價"), "08:30", "medium", pce is not None)

    for d, title in OFFICIAL_GDP:
        add(d, "GDP", title, "08:30", "medium", True)
    # 官方 GDP 時程涵蓋到 2026 年底,推估只補之後的月份
    last_official_month = max(d for d, _ in OFFICIAL_GDP).replace(day=1)
    for d, title in estimate_gdp():
        if d.replace(day=1) > last_official_month:
            add(d, "GDP", title, "08:30", "medium", False)

    events.sort(key=lambda e: (e["date"], e["time_et"], e["type"]))
    return {
        "meta": {
            "generated_at": dt.date.today().isoformat(),
            "timezone": "America/New_York",
            "range": [START.isoformat(), END.isoformat()],
            "next_review": "2026-12-15",
            "note": "time_et 為美東時間;confirmed=false 為依規則推估,官方時程公布後請更新 py/gen_calendar.py 並重跑",
        },
        "events": events,
    }


def backtest():
    """用 2026 官方時程驗證推估規則"""
    for name, official, fn in (("NFP", OFFICIAL_NFP, estimate_nfp),
                               ("CPI", OFFICIAL_CPI, estimate_cpi),
                               ("RETAIL", OFFICIAL_RETAIL, estimate_retail)):
        diffs = [(k, (fn(*k) - v).days) for k, v in official.items()]
        exact = sum(1 for _, x in diffs if x == 0)
        print(f"{name}: {exact}/{len(diffs)} 完全命中, 誤差 {[x for _, x in diffs]}")


if __name__ == "__main__":
    backtest()
    data = build()
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    n = len(data["events"])
    c = sum(e["confirmed"] for e in data["events"])
    print(f"寫入 {OUT}: {n} 筆事件({c} 筆官方確認)")
