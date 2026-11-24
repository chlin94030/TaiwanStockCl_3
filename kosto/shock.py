"""v3.21 重大消息衝擊（docs/v3.21_protocol.md）：台股對「輝達特有消息」的連動、事件區間的跌幅／反彈排名、消息對短中長線的影響。

「輝達特有」＝輝達前一晚報酬扣掉那斯達克能解釋的部分（OLS 殘差）：黃仁勳去留這類只跟輝達有關的消息，主要表現在這一部分；
整體科技股漲跌（利率、景氣）則在那斯達克那一部分。美股資料用研究資料包的 WORLD（台股 T 日之前最後一個美股收盤，不偷看）。
全部只做資訊與研究，不改選股模型（模型權重的候選見 watch.py 的 W8～W10，先當影子模型追蹤）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHOCK = 0.05                       # 輝達特有報酬 |e| ≥ 5% 視為重大消息日
HUANG = {"name": "黃仁勳去留（2026-11）", "start": "2026-11-09", "trough": "2026-11-20", "back": "2026-11-23"}
HORIZONS = (("開盤", "gap"), ("當天", "d0"), ("之後 5 日", "d5"), ("之後 20 日", "d20"), ("之後 60 日", "d60"))


def _world(B, key: str) -> pd.Series | None:
    keys = list(B.meta.get("world_keys") or [])
    if "WORLD" not in B.a or key not in keys:
        return None
    return pd.Series(np.asarray(B.a["WORLD"], float)[:, keys.index(key)], index=pd.DatetimeIndex(B.days)).ffill()


def nvda_news(B) -> pd.DataFrame | None:
    """每個台股交易日：輝達前一晚報酬 x、那指 m、輝達特有 e＝x − b·m（b 用全期 OLS）。"""
    nv, ix = _world(B, "NVDA"), _world(B, "IXIC")
    if nv is None or ix is None:
        return None
    x, m = nv / nv.shift(1) - 1, ix / ix.shift(1) - 1
    ok = x.notna() & m.notna() & (x != 0)
    b = float(np.cov(x[ok], m[ok])[0, 1] / np.var(m[ok], ddof=1)) if ok.sum() > 60 else 1.0
    return pd.DataFrame({"x": x, "m": m, "e": (x - b * m).where(ok), "b": b})


def _daily(B):
    R = np.asarray(B.a["R"], float)
    OC = np.asarray(B.a["OC"], float)
    gap = np.where(np.isfinite(OC), (1 + R) / (1 + OC) - 1, np.nan)
    R = np.where(np.abs(R) < 0.5, R, np.nan)
    gap = np.where(np.abs(gap) < 0.105, gap, np.nan)
    return R, gap


def _liquid(B, top: int) -> np.ndarray:
    cap = np.nan_to_num(np.asarray(B.a["CAP"][-1], float))
    return np.argsort(-cap)[:top]


def linkage(B, top: int = 400, days: int = 750, min_obs: int = 250) -> dict:
    """個股與族群對輝達的連動：β_total（輝達前一晚報酬 → 台股當天報酬）、β_idio（輝達特有）、相關。"""
    nd = nvda_news(B)
    if nd is None:
        return {"error": "研究資料包沒有輝達（NVDA）或那斯達克的收盤"}
    R, _ = _daily(B)
    sl = slice(max(0, len(B.days) - days), len(B.days))
    x, e = nd["x"].to_numpy()[sl], nd["e"].to_numpy()[sl]
    cols = _liquid(B, top)
    tick, groups = list(B.tickers), list(B.meta.get("groups") or [""] * len(B.tickers))
    rows = []
    for j in cols:
        y = R[sl, j]
        ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(e)
        if ok.sum() < min_obs:
            continue
        yv, xv, ev = y[ok], x[ok], e[ok]
        bt = float(np.cov(yv, xv)[0, 1] / np.var(xv, ddof=1))
        bi = float(np.cov(yv, ev)[0, 1] / np.var(ev, ddof=1))
        rows.append({"ticker": tick[j], "group": groups[j], "beta": bt, "beta_idio": bi, "corr": float(np.corrcoef(yv, xv)[0, 1]),
                     "n": int(ok.sum())})
    if not rows:
        return {"error": "資料不足"}
    df = pd.DataFrame(rows)
    g = (df.groupby("group").agg(n=("ticker", "size"), beta=("beta", "median"), beta_idio=("beta_idio", "median"), corr=("corr", "median"))
         .query("n >= 3").sort_values("beta_idio", ascending=False))
    return {"asof": str(pd.Timestamp(B.days[-1]).date()), "days": int(min(days, len(B.days))), "b_nasdaq": float(nd["b"].iloc[0]),
            "stocks": df.sort_values("beta_idio", ascending=False).round(4).to_dict("records"),
            "groups": g.reset_index().round(4).to_dict("records")}


def episode(B, start: str, trough: str, end: str | None = None, top: int = 400) -> dict:
    """事件區間：start 收盤 → trough 收盤（跌幅）、trough 收盤 → end（或最新）收盤（反彈）；個股、族群、0050。資料沒涵蓋就回傳 error。"""
    days = pd.DatetimeIndex(B.days)
    a, t = days.searchsorted(pd.Timestamp(start), "right") - 1, days.searchsorted(pd.Timestamp(trough), "right") - 1
    z = (days.searchsorted(pd.Timestamp(end), "right") - 1) if end else len(days) - 1
    if a < 0 or t <= a or days[t] < pd.Timestamp(trough) - pd.Timedelta(days=5):
        return {"error": f"研究資料只到 {days[-1].date()}，還沒涵蓋 {trough}"}
    R, _ = _daily(B)
    bn = np.nan_to_num(np.asarray(B.a["BENCH"], float))

    def cum(r, i0, i1):
        return float(np.prod(1 + np.nan_to_num(r[i0 + 1: i1 + 1])) - 1) if i1 > i0 else np.nan
    b_down, b_up = cum(bn, a, t), cum(bn, t, z)
    cols = _liquid(B, top)
    tick, groups = list(B.tickers), list(B.meta.get("groups") or [""] * len(B.tickers))
    rows = []
    for j in cols:
        if not np.isfinite(R[a + 1: z + 1, j]).any():
            continue
        dn, up = cum(R[:, j], a, t), cum(R[:, j], t, z)
        rows.append({"ticker": tick[j], "group": groups[j], "down": dn, "up": up, "down_rel": dn - b_down, "up_rel": up - b_up,
                     "recover": (1 + dn) * (1 + up) - 1})
    df = pd.DataFrame(rows)
    g = df.groupby("group").agg(n=("ticker", "size"), down=("down", "median"), up=("up", "median"), recover=("recover", "median")).query("n >= 3")
    return {"start": str(days[a].date()), "trough": str(days[t].date()), "end": str(days[z].date()), "bench_down": b_down, "bench_up": b_up,
            "rebound_days": int(z - t), "stocks": df.round(4).to_dict("records"), "groups": g.reset_index().round(4).to_dict("records"),
            "corr_down_up": float(df[["down", "up"]].corr().iloc[0, 1]) if len(df) > 10 and z > t else None}


def shock_study(B, thr: float = SHOCK, top: int = 300, lookback: int = 250) -> dict:
    """輝達特有消息日（|e| ≥ thr）之後，台股「高連動」組（β_idio 前三分之一）相對「低連動」組的表現，依消息方向調整正負號。
    β 只用事件前 250 日估（不偷看）。正值＝跟著消息方向走、負值＝反轉。"""
    nd = nvda_news(B)
    if nd is None:
        return {"error": "研究資料包沒有輝達（NVDA）或那斯達克的收盤"}
    R, gap = _daily(B)
    e = nd["e"].to_numpy()
    cols = _liquid(B, top)
    D = len(B.days)
    ev = [t for t in np.where(np.abs(np.nan_to_num(e)) >= thr)[0] if lookback <= t < D - 1]
    out = []
    for t in ev:
        w = slice(t - lookback, t)
        ew = e[w]
        okx = np.isfinite(ew)
        Y = R[w][:, cols]
        beta = np.array([np.cov(Y[okx, k][np.isfinite(Y[okx, k])], ew[okx][np.isfinite(Y[okx, k])])[0, 1] / np.var(ew[okx][np.isfinite(Y[okx, k])], ddof=1)
                         if np.isfinite(Y[okx, k]).sum() > 120 else np.nan for k in range(len(cols))])
        ok = np.isfinite(beta)
        if ok.sum() < 30:
            continue
        q1, q3 = np.nanquantile(beta, [1 / 3, 2 / 3])
        hi, lo = cols[ok & (beta >= q3)], cols[ok & (beta <= q1)]
        s = np.sign(e[t])

        def spread(arr_fn):
            return s * (np.nanmean(arr_fn(hi)) - np.nanmean(arr_fn(lo)))

        def fwd(k):
            return lambda c: np.prod(1 + np.nan_to_num(R[t + 1: t + 1 + k][:, c]), axis=0) - 1 if t + k < D else np.full(len(c), np.nan)
        row = {"date": str(pd.Timestamp(B.days[t]).date()), "e": float(e[t]), "x": float(nd["x"].iloc[t]),
               "gap": spread(lambda c: gap[t, c]), "d0": spread(lambda c: R[t, c])}
        for k, nm in ((5, "d5"), (20, "d20"), (60, "d60")):
            row[nm] = spread(fwd(k)) if t + k < D else np.nan
        out.append(row)
    if not out:
        return {"events": [], "summary": []}
    df = pd.DataFrame(out)
    summ = []
    for side, m in (("全部", df.index == df.index), ("利多（輝達特有上漲）", df["e"] > 0), ("利空（輝達特有下跌）", df["e"] < 0)):
        x = df[m]
        if not len(x):
            continue
        summ.append({"side": side, "n": int(len(x)), **{k: float(x[k].mean()) for _, k in HORIZONS},
                     **{f"{k}_pos": float((x[k] > 0).mean()) for _, k in HORIZONS if x[k].notna().any()}})
    return {"thr": thr, "events": df.round(4).to_dict("records"), "summary": summ}


def summary(B) -> dict:
    """每日研究呼叫：連動排名＋黃仁勳事件區間（資料涵蓋時）＋消息日研究。失敗不影響研究。"""
    out = {}
    for k, fn in (("linkage", lambda: linkage(B)), ("study", lambda: shock_study(B)),
                  ("huang", lambda: episode(B, HUANG["start"], HUANG["trough"]))):
        try:
            out[k] = fn()
        except Exception as e:  # noqa: BLE001
            out[k] = {"error": str(e)}
    out["huang_meta"] = HUANG
    return out
