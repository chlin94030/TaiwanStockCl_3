"""v1.7 台股與美股／日韓的連動（描述統計；不參與選股）。

  overnight_table  前一夜費半漲跌分級 → 台股大型股（市值前 300）隔日：開盤跳空、開盤→收盤、全日、再隔一日
  yearly_linkage   逐年：台股大型股日報酬 vs 前一夜費半/那斯達克、同日 KOSPI/日經 的相關係數與 β
  world_state      最新一個交易日（研究時間）已知的國際市場狀態：費半 vs 百日線、VIX、韓股/台幣 20 日變化
  STUDY            v1.7 十面向預先宣告研究的結果（kosto/data/v17_study.csv；見 docs/v1.7_protocol.md）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = [-1.0, -0.025, -0.01, 0.01, 0.025, 1.0]
BUCKET_ZH = ["費半 ≤ −2.5%", "−2.5% ～ −1%", "±1% 以內", "+1% ～ +2.5%", "費半 ≥ +2.5%"]
STUDY_PATH = Path(__file__).with_name("data") / "v17_study.csv"


def _world(B) -> pd.DataFrame:
    keys = B.meta.get("world_keys") or []
    if "WORLD" not in B.a or not keys:
        return pd.DataFrame(index=B.days)
    return pd.DataFrame(B.a["WORLD"].astype(float), index=B.days, columns=keys)


def _large_cap_daily(B, size: int = 300) -> pd.DataFrame:
    """每日：前一個訊號日市值前 size 名的等權 開盤跳空 / 開盤→收盤 / 全日 / 隔日 報酬（電子、非電子分開）。"""
    R, OC = B.a["R"].astype(float), B.a["OC"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        GAP = np.where(np.isfinite(OC), (1 + R) / (1 + OC) - 1, np.nan)
    OCn = np.where(np.isfinite(OC), OC, np.nan)
    rules = B.meta.get("rules") or []
    rk = B.a["RANKS"][rules.index("MCAP")] if "MCAP" in rules else B.a["RANKS"][0]
    groups = B.meta.get("groups") or ["其他"] * len(B.tickers)
    elec = np.array([str(g).startswith("電子") for g in groups])
    sp = B.sig_pos
    D = len(B.days)
    sig_of_day = np.searchsorted(sp, np.arange(D), side="right") - 1
    rows = []
    for d in range(1, D):
        s = sig_of_day[d - 1]
        if s < 0:
            continue
        mem = (rk[s] >= 1) & (rk[s] <= size)
        ok = mem & np.isfinite(OCn[d]) & np.isfinite(GAP[d])
        if ok.sum() < 30:
            continue
        nxt = np.nanmean(R[d + 1, ok]) if d + 1 < D else np.nan
        rows.append({"date": B.days[d], "gap": np.nanmean(GAP[d, ok]), "oc": np.nanmean(OCn[d, ok]), "day": np.nanmean(R[d, ok]),
                     "next": nxt, "gap_e": np.nanmean(GAP[d, ok & elec]) if (ok & elec).any() else np.nan,
                     "oc_e": np.nanmean(OCn[d, ok & elec]) if (ok & elec).any() else np.nan,
                     "gap_n": np.nanmean(GAP[d, ok & ~elec]) if (ok & ~elec).any() else np.nan,
                     "oc_n": np.nanmean(OCn[d, ok & ~elec]) if (ok & ~elec).any() else np.nan})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _rets(W: pd.DataFrame) -> pd.DataFrame:
    return (W / W.shift(1) - 1).where(lambda x: x.abs() < 0.25)


def overnight_table(B, T: pd.DataFrame | None = None) -> pd.DataFrame:
    W = _world(B)
    if "SOX" not in W or not np.isfinite(W["SOX"]).any():
        return pd.DataFrame()
    T = _large_cap_daily(B) if T is None else T
    if T.empty:
        return pd.DataFrame()
    x = T.join(_rets(W)[["SOX"]].rename(columns={"SOX": "sox"}), how="inner").dropna(subset=["sox", "gap"])
    x["級距"] = pd.cut(x["sox"], BUCKETS, labels=BUCKET_ZH)
    t = x.groupby("級距", observed=True).agg(天數=("sox", "size"), 費半平均=("sox", "mean"), 台股開盤跳空=("gap", "mean"),
                                           開盤到收盤=("oc", "mean"), 全日=("day", "mean"), 隔一日=("next", "mean"),
                                           電子股跳空=("gap_e", "mean"), 電子股盤中=("oc_e", "mean"),
                                           非電子跳空=("gap_n", "mean"), 非電子盤中=("oc_n", "mean")).reset_index()
    t.attrs["period"] = f"{x.index[0].date()} ~ {x.index[-1].date()}"
    b = {}
    for k in ("gap", "oc", "day"):
        z = x[["sox", k]].dropna()
        b[k] = float(np.polyfit(z["sox"], z[k], 1)[0]) if len(z) > 30 else np.nan
    t.attrs["beta"] = b
    return t


def yearly_linkage(B, T: pd.DataFrame | None = None) -> pd.DataFrame:
    W = _world(B)
    if W.shape[1] == 0:
        return pd.DataFrame()
    T = _large_cap_daily(B) if T is None else T
    if T.empty:
        return pd.DataFrame()
    x = T[["day"]].join(_rets(W), how="inner")
    out = []
    for y, g in x.groupby(x.index.year):
        row = {"年": int(y), "天數": int(len(g))}
        for k, zh in (("SOX", "費半(前一夜)"), ("IXIC", "那斯達克(前一夜)"), ("KS11", "韓股(同日)"), ("N225", "日經(同日)")):
            if k not in g:
                continue
            z = g[["day", k]].dropna()
            row[f"相關_{zh}"] = float(z["day"].corr(z[k])) if len(z) > 30 else np.nan
            row[f"β_{zh}"] = float(np.polyfit(z[k], z["day"], 1)[0]) if len(z) > 30 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def world_state(B) -> dict:
    """研究時間（台股收盤後）已知的最新國際狀態（美股＝前一個美股交易日收盤）。"""
    W = _world(B)
    if W.shape[1] == 0 or W.dropna(how="all").empty:
        return {}
    last = W.dropna(how="all").index[-1]
    st = {"date": str(last.date())}

    def lv(k, n=0):
        if k not in W:
            return np.nan
        s = W[k].dropna()
        return float(s.iloc[-1 - n]) if len(s) > n else np.nan
    for k in W.columns:
        st[k] = lv(k)
        st[f"{k}_d1"] = lv(k) / lv(k, 1) - 1 if np.isfinite(lv(k, 1)) else np.nan
        st[f"{k}_d20"] = lv(k) / lv(k, 20) - 1 if np.isfinite(lv(k, 20)) else np.nan
    if "SOX" in W:
        ma = W["SOX"].dropna().rolling(100, min_periods=80).mean()
        st["SOX_ma100"] = float(ma.iloc[-1]) if len(ma) and np.isfinite(ma.iloc[-1]) else np.nan
        st["SOX_vs_ma100"] = st["SOX"] / st["SOX_ma100"] - 1 if np.isfinite(st.get("SOX_ma100", np.nan)) else np.nan
    risk = []
    if np.isfinite(st.get("SOX_vs_ma100", np.nan)) and st["SOX_vs_ma100"] < 0:
        risk.append("費半跌破百日線")
    if np.isfinite(st.get("VIX", np.nan)) and st["VIX"] > 30:
        risk.append("VIX > 30")
    st["risk"] = risk
    st["light"] = "red" if len(risk) >= 2 else ("amber" if risk else "green")
    return st


def study() -> pd.DataFrame:
    return pd.read_csv(STUDY_PATH) if STUDY_PATH.is_file() else pd.DataFrame()


def summary(B) -> dict:
    try:
        T = _large_cap_daily(B)
        return {"overnight": overnight_table(B, T), "yearly": yearly_linkage(B, T), "state": world_state(B), "study": study()}
    except Exception as e:  # noqa: BLE001 - descriptive only, never block the research run
        return {"error": str(e), "overnight": pd.DataFrame(), "yearly": pd.DataFrame(), "state": {}, "study": study()}
