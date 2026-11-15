"""v3.5 美股第二輪研究重現腳本（docs/v3.5_protocol.md）：
下載標普 500 時點成分股資料集 → 4 母池 × 9 模型 × 3 期間 → 對 SPY 的驗收規則；並跑 5 個 ETF 策略。

    python scripts/us_research.py            # 下載約 240 MB、計算約 10 分鐘
資料：GitHub Johnbrick123/sp500-data（含被剔除、下市的成分股；1996 年後逐日核對）。只供研究，不影響 App。
（第一輪 v3.4 用的是 eshaanmattoo/us_stocks，只有目前仍掛牌的股票，結論見 docs/v3.4_us_protocol.md。）
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kosto.us import backtest as BT  # noqa: E402
from kosto.us import model as M  # noqa: E402

REL = "https://github.com/Johnbrick123/sp500-data/releases/download/data/"
PER = {"dev": ("2004-01-01", "2021-12-31"), "val": ("2022-01-01", "2024-12-31"), "blind": ("2025-01-01", None),
       "dev1": ("2004-01-01", "2013-12-31"), "dev2": ("2014-01-01", "2021-12-31")}
UNI = {"SP500": None, "SP300": 300, "SP200": 200, "SP100": 100}


def get(name: str) -> bytes:
    r = requests.get(REL + name, timeout=600)
    r.raise_for_status()
    return r.content


def panel():
    iv = pd.read_parquet(io.BytesIO(get("membership_intervals.parquet")))
    iv["end"] = iv["end"].fillna(pd.Timestamp("2100-01-01"))
    iv = iv[iv["end"] >= "2002-01-01"]
    p = pd.read_parquet(io.BytesIO(get("prices.parquet")), columns=["date", "ticker", "open", "close", "adj_close", "volume"])
    p = p[(p.date >= "2001-01-01") & p.ticker.isin(set(iv.ticker) | {"SPY"})]
    idx = pd.DatetimeIndex(sorted(p.loc[p.ticker == "SPY", "date"].unique()))

    def wide(c):
        return p.pivot_table(index="date", columns="ticker", values=c, aggfunc="last").reindex(idx)
    C, RAW, OP, V = wide("adj_close"), wide("close"), wide("open"), wide("volume")
    mem = pd.DataFrame(False, idx, C.columns)
    for _, r in iv.iterrows():
        if r.ticker in mem.columns:
            mem.loc[(idx >= r.start) & (idx < r.end), r.ticker] = True
    nt = pd.read_csv(io.BytesIO(get("no_trade_days.csv")), parse_dates=["date"])
    NT = pd.DataFrame(False, idx, C.columns)
    for _, r in nt[nt.date >= "2001-01-01"].iterrows():
        if r.ticker in NT.columns and r.date in NT.index:
            NT.at[r.date, r.ticker] = True
    return C, OP * (C / RAW), RAW * V, mem, NT


def main():
    C, O, DV, mem, NT = panel()
    spy = C.pop("SPY").pct_change()
    O, DV, mem, NT = (x.drop(columns="SPY", errors="ignore") for x in (O, DV, mem, NT))
    F = M.factors(C, None)
    start = int(np.searchsorted(C.index, pd.Timestamp("2003-01-02")))
    rows = []
    for un, top in UNI.items():
        U = M.sp_universe(C, DV, mem, top=top, no_trade=NT)
        for h, hc in M.HORIZONS.items():
            for m, w in M.MODELS.items():
                S = M.score(F, U, w)
                r = BT.run(C, O, S, U, hc["days"], k=M.TOP_K, hyst=hc["hyst"], cost=0.001, start=start)
                r2 = BT.run(C, O, S, U, hc["days"], k=M.TOP_K, hyst=hc["hyst"], cost=0.0025, start=start)
                for p, (a, z) in PER.items():
                    rows.append({"uni": un, "h": h, "model": m, "period": p, **BT.stats(r, spy, a, z),
                                 "excess_stress": BT.stats(r2, spy, a, z).get("excess")})
                print(un, h, m, flush=True)
    df = pd.DataFrame(rows)

    def passes(x):
        return x.excess >= 0.03 and x.ir >= 0.4 and x.mdd - x.bench_mdd >= -0.10 and x.fold_win >= 0.5 and x.excess_stress > 0
    df["pass"] = df.apply(passes, axis=1)
    pd.set_option("display.width", 220)
    print(df[df.period.isin(["dev", "val", "blind"])].pivot_table(index=["h", "uni", "model"], columns="period",
                                                                     values=["excess", "ir"]).round(3).to_string())
    print("開發期與驗證期都通過的組合：", df[df.period.isin(["dev", "val"])].groupby(["h", "uni", "model"])["pass"].all().sum())


if __name__ == "__main__":
    main()
