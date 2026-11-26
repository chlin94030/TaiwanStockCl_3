"""v3.24 籌碼擁擠研究（docs/v3.24_protocol.md）：投信（主動式 ETF／基金）重倉的股票，在大跌日、大漲日與之後的表現。

「擁擠」＝前一個換股日「投信 60 日累計買超（占股本）」在成交金額前 300 大裡最高 10%（只用當時以前的資料）。
主動式 ETF 申購時，投信要拿現金去買持股；大量買回時要賣持股——這些股票是不是「漲多、跌也多」？跌完會不會繼續被賣？
只做研究與顯示，不改選股模型；擁擠風險的候選見 watch.py 的 W11、W12（影子模型）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR = "c_trust60"
TOP_N = 300
Q = 0.9
ERA = "2026-01-01"                       # 主動式 ETF 規模暴增的一年（2025 底 1,302 億 → 2026-10 約 1 兆）


def _R(B) -> np.ndarray:
    R = np.asarray(B.a["R"], float)
    return np.where(np.isfinite(R) & (np.abs(R) < 0.5), R, np.nan)


def crowd_sets(B, top_n: int = TOP_N, q: float = Q) -> dict | None:
    """每個換股日 i：(成交金額前 top_n 大的欄位, 其中擁擠前 10% 的布林)。"""
    F = list(B.factors)
    if FACTOR not in F:
        return None
    j = F.index(FACTOR)
    U = B.a["U"]
    cap = np.asarray(B.a["CAP"], float)
    out = {}
    for i in range(len(B.sig_pos)):
        c = np.nan_to_num(cap[i])
        cols = np.argsort(-c)[:top_n]
        cols = cols[c[cols] > 0]
        u = np.asarray(U[i, cols, j], float)
        ok = u > 0
        if ok.sum() < 30:
            continue
        hi = ok & (u >= np.quantile(u[ok], q))
        out[i] = (cols[ok], hi[ok])
    return out


def event_study(B, thr: float = -0.03, sets: dict | None = None) -> dict:
    """0050 單日報酬 ≤ thr（thr<0）或 ≥ thr（thr>0）的日子：擁擠組 − 其他組（等權），當天與之後 5／20 日；分兩個時期。"""
    sets = sets if sets is not None else crowd_sets(B)
    if not sets:
        return {"error": "研究資料包沒有投信 60 日累計買超"}
    R = _R(B)
    D = R.shape[0]
    bench = np.nan_to_num(np.asarray(B.a["BENCH"], float))
    days = pd.DatetimeIndex(B.days)
    sp = np.asarray(B.sig_pos, int)
    keys = sorted(sets)
    rows = []
    hit = np.where(bench <= thr)[0] if thr < 0 else np.where(bench >= thr)[0]
    for d in hit:
        k = int(np.searchsorted(sp, d, side="left")) - 1               # 前一個換股日（d 之前，不偷看）
        while k >= 0 and k not in sets:
            k -= 1
        if k < 0 or k < keys[0]:
            continue
        cols, hi = sets[k]

        def spread(a, b):
            if b >= D:
                return np.nan
            g = np.nanprod(1 + R[a: b + 1][:, cols], axis=0) - 1
            return float(np.nanmean(g[hi]) - np.nanmean(g[~hi]))
        rows.append({"date": str(days[d].date()), "bench": float(bench[d]), "d0": spread(d, d),
                     "d5": spread(d + 1, d + 5), "d20": spread(d + 1, d + 20)})
    df = pd.DataFrame(rows)
    out = {"thr": thr, "n": int(len(df)), "events": df.round(4).to_dict("records") if len(df) else []}
    summ = []
    for lab, m in (("全部", np.ones(len(df), bool)), ("2019–2025", (df["date"] < ERA).to_numpy() if len(df) else np.zeros(0, bool)),
                   ("2026（主動式 ETF 時代）", (df["date"] >= ERA).to_numpy() if len(df) else np.zeros(0, bool))):
        x = df[m] if len(df) else df
        if not len(x):
            summ.append({"period": lab, "n": 0})
            continue
        summ.append({"period": lab, "n": int(len(x)), **{k: float(x[k].mean()) for k in ("d0", "d5", "d20")},
                     "d0_neg": float((x["d0"] < 0).mean())})
    out["summary"] = summ
    return out


def baseline(B, sets: dict | None = None) -> dict:
    """所有交易日的平均差（擁擠組 − 其他組）：大跌日的差要和平常比才有意義。"""
    sets = sets if sets is not None else crowd_sets(B)
    if not sets:
        return {}
    R = _R(B)
    sp = np.asarray(B.sig_pos, int)
    diffs = []
    for k, (cols, hi) in sets.items():
        a = int(sp[k]) + 1
        b = int(sp[k + 1]) if k + 1 < len(sp) else R.shape[0] - 1
        for d in range(a, min(b, R.shape[0] - 1) + 1):
            r = R[d, cols]
            diffs.append(float(np.nanmean(r[hi]) - np.nanmean(r[~hi])))
    x = np.asarray(diffs)
    return {"days": int(len(x)), "mean": float(np.nanmean(x)) if len(x) else np.nan, "ann": float(np.nanmean(x) * 245) if len(x) else np.nan}


def latest_crowded(B, n: int = 15) -> list:
    """最新換股日的擁擠前 n 名（成交金額前 300 大）：代號、名稱、投信 60 日累計買超的百分位。"""
    sets = crowd_sets(B)
    if not sets:
        return []
    i = max(sets)
    cols, hi = sets[i]
    j = list(B.factors).index(FACTOR)
    u = np.asarray(B.a["U"][i, cols, j], float)
    order = np.argsort(-u)[:n]
    names = B.meta.get("names") or [""] * len(B.tickers)
    groups = B.meta.get("groups") or [""] * len(B.tickers)
    return [{"ticker": B.tickers[cols[k]], "name": names[cols[k]], "group": groups[cols[k]], "pct": float(u[k] / 65535)} for k in order]


def summary(R, res: dict | None = None) -> dict:
    """每日研究呼叫；各部分獨立。"""
    B = R.B
    out = {"asof": str(pd.Timestamp(B.days[-1]).date()), "errors": []}
    try:
        sets = crowd_sets(B)
        out["down"] = event_study(B, -0.03, sets)
        out["up"] = event_study(B, 0.03, sets)
        out["base"] = baseline(B, sets)
        out["latest"] = latest_crowded(B)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"擁擠研究：{e}")
    w = (res or {}).get("watch") or {}
    out["watch"] = [r for r in w.get("rows", []) if r.get("key") in ("W11", "W12")]
    return out
