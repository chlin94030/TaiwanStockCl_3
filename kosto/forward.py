"""v1.9 前瞻紀錄：每天實際公布的名單，照「隔日開盤買進、持有 H 個交易日、開盤賣出」計算真實績效（不經回測挑選）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def entry_returns(B, log: list[dict]) -> pd.DataFrame:
    """每一筆紀錄：等權報酬（未到期者算到最新收盤）、同期 0050、是否到期。"""
    if not log or "R" not in B.a:
        return pd.DataFrame()
    days = B.days
    R, OC = B.a["R"].astype(float), B.a["OC"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        GAP = np.where(np.isfinite(OC), (1 + R) / (1 + OC) - 1, 0.0)
    bench = B.a["BENCH"].astype(float)
    tix = {t: j for j, t in enumerate(B.tickers)}
    rows = []
    for x in log:
        d = pd.Timestamp(x["date"])
        t = int(days.searchsorted(d))
        if t >= len(days) or days[t] != d:
            continue
        e = t + 1
        if e >= len(days):
            rows.append({**x, "ret": np.nan, "bench": np.nan, "matured": False, "held_days": 0})
            continue
        H = int(x.get("days", 20))
        last = min(t + H, len(days) - 1)
        js = [tix[s] for s in x["picks"] if s in tix]
        if not js:
            continue
        vals = []
        for j in js:
            v = 1 + np.nan_to_num(OC[e, j])
            for k in range(e + 1, last + 1):
                v *= 1 + np.nan_to_num(R[k, j])
            matured = t + H + 1 < len(days)
            if matured:
                v *= 1 + np.nan_to_num(GAP[t + H + 1, j])
            vals.append(v - 1)
        b = float(np.prod(1 + bench[e: last + 1]) - 1)
        rows.append({**x, "ret": float(np.mean(vals)), "bench": b, "matured": t + H + 1 < len(days), "held_days": last - t})
    return pd.DataFrame(rows)


def summary(B, log: list[dict]) -> pd.DataFrame:
    t = entry_returns(B, log)
    if t.empty:
        return pd.DataFrame()
    t["excess"] = t["ret"] - t["bench"]
    g = t.dropna(subset=["ret"]).groupby("h")
    out = pd.DataFrame({"筆數": g.size(), "已到期": g["matured"].sum(), "平均報酬": g["ret"].mean(), "0050同期": g["bench"].mean(),
                        "平均超額": g["excess"].mean(), "勝過0050比例": g["excess"].apply(lambda s: float((s > 0).mean()))})
    return out.reindex([h for h in ("short", "mid", "long") if h in out.index])
