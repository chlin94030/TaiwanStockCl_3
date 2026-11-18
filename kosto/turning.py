"""v3.11 重要轉折點驗證（docs/v3.11_protocol.md）：0050 回落 ≥10% 的每一段，模型下跌段與反彈段 vs 0050。
資料來源＝每日研究的回測淨值（RES["horizons"][h]["nav"]，欄位「策略」「0050」），所以每天自動包含最新的轉折。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def episodes(s: pd.Series, th: float = 0.10) -> list[tuple]:
    """(高點日, 最低點日, 回到高點日或 None)；回落 ≥ th 才算一段。"""
    s = s.dropna().astype(float)
    v = s.to_numpy()
    out, peak, trough, in_dd = [], 0, 0, False
    for i in range(len(v)):
        if not in_dd and v[i] >= v[peak]:
            peak = i
        if not in_dd and v[i] / v[peak] - 1 <= -th:
            in_dd, trough = True, i
        if in_dd:
            if v[i] < v[trough]:
                trough = i
            if v[i] >= v[peak]:
                out.append((s.index[peak], s.index[trough], s.index[i]))
                in_dd, peak = False, i
    if in_dd:
        out.append((s.index[peak], s.index[trough], None))
    return out


def decompose(navs: dict[str, pd.DataFrame], th: float = 0.10) -> pd.DataFrame:
    """navs＝{期間: 淨值表(含「策略」「0050」)}。每段：下跌段（高點→低點）、低點後 20／60 日、目前仍在進行中的段落。"""
    if not navs:
        return pd.DataFrame()
    base = next(iter(navs.values()))["0050"].dropna()
    rows = []
    for p, t, r in episodes(base, th):
        ti = base.index.get_loc(t)
        row = {"peak": p, "trough": t, "recover": r, "ongoing": r is None, "b_dn": base[t] / base[p] - 1,
               "b_now": base.iloc[-1] / base[p] - 1}
        for H in (20, 60):
            row[f"b_up{H}"] = base.iloc[ti + H] / base[t] - 1 if ti + H < len(base) else np.nan
        for h, n in navs.items():
            s = n["策略"].reindex(base.index).ffill()
            row[f"{h}_dn"] = s[t] / s[p] - 1
            row[f"{h}_now"] = s.iloc[-1] / s[p] - 1
            for H in (20, 60):
                row[f"{h}_up{H}"] = s.iloc[ti + H] / s[t] - 1 if ti + H < len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
