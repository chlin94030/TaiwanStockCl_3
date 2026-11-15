"""v3.4 美股選股模型：價量因子、母池、綜合分數（研究與每日正式名單共用同一份程式）。

所有輸入都是「日期 × 股票」的寬表（pandas DataFrame）：C 還原收盤、O 開盤、H 最高、L 最低、V 成交量（股）。
因子在每天收盤計算；分數＝母池內排名百分位的加權平均（見 docs/v3.4_us_protocol.md）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 預先宣告的候選模型（固定權重）
MODELS: dict[str, dict[str, float]] = {
    "LEADERS": {"RS120": 0.60, "MOM120x20": 0.25, "HIGH252": 0.15},
    "BALANCED": {"RS120": 0.40, "MOM120x20": 0.25, "HIGH252": 0.20, "LOWIDIO": 0.15},
    "MOM": {"MOM12_1": 1.0},
    "HIGH": {"HIGH252": 1.0},
    "TREND_Q": {"MOM12_1": 0.35, "TREND": 0.25, "LOWVOL": 0.20, "MAX21": 0.20},
    "BAL_REV": {"RS120": 0.32, "MOM120x20": 0.20, "HIGH252": 0.16, "LOWIDIO": 0.12, "REV5": 0.20},
    # v3.5 第二輪新增（docs/v3.5_protocol.md）：參考美股 ETF 選股方法
    "SPMO": {"MOMVOL": 1.0},
    "MINVOL": {"LOWVOL": 1.0},
    "QTREND": {"HIGH252": 0.35, "LOWIDIO": 0.35, "MAX21": 0.30},
}
MODEL_ZH = {"LEADERS": "強勢領頭羊（台股長線假說）", "BALANCED": "平衡動能（台股短中線假說）",
            "MOM": "12 個月經典動能", "HIGH": "接近 52 週新高", "TREND_Q": "趨勢＋低波動品質", "BAL_REV": "平衡動能＋短期反轉",
            "SPMO": "風險調整動能（標普動能指數法）", "MINVOL": "最小波動（低波動 ETF 法）", "QTREND": "品質趨勢（新高＋穩＋非彩券）"}
FACTOR_ZH = {
    "RS120": "近半年漲幅（120 日）", "MOM120x20": "中期動能（半年扣最近一個月）", "HIGH252": "接近一年新高",
    "LOWIDIO": "走勢穩（特異波動低）", "MOM12_1": "一年動能（扣最近一個月）", "TREND": "站上 200 日均線幅度",
    "REV5": "近 5 日拉回", "LOWVOL": "低波動", "MAX21": "不是暴漲暴跌的彩券股",
    "MOMVOL": "風險調整動能（一年動能 ÷ 波動）",
}
UNIVERSE = {"top": 700, "min_price": 10.0, "min_hist": 260, "adv_days": 60}
HORIZONS = {"short": {"days": 5, "hyst": 2.5, "label": "短線（約 1 週）"},
            "mid": {"days": 20, "hyst": 2.0, "label": "中線（約 1 個月）"},
            "long": {"days": 60, "hyst": 2.5, "label": "長線（約 3 個月）"}}
TOP_K = 10


def _ret(C: pd.DataFrame) -> pd.DataFrame:
    return C / C.shift(1) - 1


def factors(C: pd.DataFrame, V: pd.DataFrame | None = None, need=None) -> dict[str, pd.DataFrame]:
    """全部價量因子（原始值，越大越好）。need＝只算需要的因子。"""
    need = set(need or FACTOR_ZH)
    r = _ret(C)
    out = {}
    if "RS120" in need:
        out["RS120"] = C / C.shift(120) - 1
    if "MOM120x20" in need:
        out["MOM120x20"] = C.shift(20) / C.shift(120) - 1
    if "HIGH252" in need:
        out["HIGH252"] = C / C.rolling(252, min_periods=120).max()
    if "MOM12_1" in need:
        out["MOM12_1"] = C.shift(21) / C.shift(252) - 1
    if "TREND" in need:
        out["TREND"] = C / C.rolling(200, min_periods=150).mean() - 1
    if "REV5" in need:
        out["REV5"] = -(C / C.shift(5) - 1)
    if "LOWVOL" in need:
        out["LOWVOL"] = -r.rolling(63, min_periods=40).std()
    if "MAX21" in need:
        out["MAX21"] = -r.rolling(21, min_periods=15).max()
    if "MOMVOL" in need:
        out["MOMVOL"] = (C.shift(21) / C.shift(252) - 1) / r.rolling(252, min_periods=200).std()
    if "LOWIDIO" in need:
        mkt = r.mean(axis=1)
        mm, mv = mkt.rolling(60, min_periods=40).mean(), mkt.rolling(60, min_periods=40).var()
        beta = (r.mul(mkt, axis=0).rolling(60, min_periods=40).mean()
                - r.rolling(60, min_periods=40).mean().mul(mm, axis=0)).div(mv, axis=0)
        out["LOWIDIO"] = -r.sub(beta.mul(mkt, axis=0)).rolling(60, min_periods=40).std()
    return out


def universe(C: pd.DataFrame, V: pd.DataFrame, top: int | None = None, min_price: float | None = None,
             min_hist: int | None = None, adv_days: int | None = None) -> pd.DataFrame:
    """每天的母池（布林寬表）：股價 ≥ 10、有資料滿 260 日、60 日平均成交金額前 700 名（全部用當天以前的資料）。"""
    top = top or UNIVERSE["top"]
    min_price = UNIVERSE["min_price"] if min_price is None else min_price
    min_hist = min_hist or UNIVERSE["min_hist"]
    adv_days = adv_days or UNIVERSE["adv_days"]
    adv = (C * V).rolling(adv_days, min_periods=int(adv_days * 0.7)).mean()
    hist = C.notna().cumsum()
    ok = (C >= min_price) & (hist >= min_hist) & adv.notna()
    rk = adv.where(ok).rank(axis=1, ascending=False, method="first")
    return ok & (rk <= top)


def sp_universe(C: pd.DataFrame, DV: pd.DataFrame, member: pd.DataFrame, top: int | None = None, min_price: float = 5.0,
                min_hist: int = 260, no_trade: pd.DataFrame | None = None) -> pd.DataFrame:
    """v3.5 標普 500 母池：當天成分股 ∩ 股價 ≥ 5 ∩ 資料滿 260 日 ∩ 近 5 日沒停牌；top＝成分股中 60 日成交金額前幾名。"""
    adv = DV.rolling(60, min_periods=40).mean()
    ok = member.reindex_like(C).fillna(False).astype(bool) & (C >= min_price) & (C.notna().cumsum() >= min_hist) & adv.notna()
    if no_trade is not None:
        ok &= ~no_trade.reindex_like(C).fillna(False).astype(bool).rolling(5, min_periods=1).max().astype(bool)
    if top:
        ok &= adv.where(ok).rank(axis=1, ascending=False, method="first") <= top
    return ok


def pct_rank(F: pd.DataFrame, U: pd.DataFrame) -> pd.DataFrame:
    """母池內排名百分位（0～1，越大越好）；缺值給 0.5（中性）。"""
    x = F.where(U)
    return x.rank(axis=1, pct=True).where(U).fillna(0.5).where(U)


def score(F: dict[str, pd.DataFrame], U: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    tot = sum(weights.values())
    s = None
    for k, w in weights.items():
        p = pct_rank(F[k], U) * (w / tot)
        s = p if s is None else s + p
    return s.where(U)


def factor_ranks(F: dict[str, pd.DataFrame], U: pd.DataFrame, day, names=None) -> pd.DataFrame:
    """某一天每檔股票各因子的母池百分位（App 個股頁「為什麼入選」用）。"""
    rows = {}
    for k in names or F:
        rows[k] = pct_rank(F[k].loc[[day]], U.loc[[day]]).iloc[0]
    return pd.DataFrame(rows)


def rank_weights(n: int, weighting: str = "rank") -> np.ndarray:
    """與台股 plan.stock_weights 相同：名次加權 1.5 → 0.5。"""
    if n <= 0:
        return np.zeros(0)
    w = np.linspace(1.5, 0.5, n) if (weighting == "rank" and n > 1) else np.ones(n)
    return w / w.sum()


def pick(scores: pd.Series, k: int = TOP_K, held=None, hyst: float = 2.5) -> list[str]:
    """排序取前 k 名；held（目前持有）中名次仍在前 hyst×k 的優先續抱。"""
    s = scores.dropna().sort_values(ascending=False)
    if s.empty:
        return []
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)
    keep = [t for t in (held or []) if t in rank.index and rank[t] <= hyst * k]
    out = list(dict.fromkeys(keep))[:k]
    for t in s.index:
        if len(out) >= k:
            break
        if t not in out:
            out.append(t)
    return sorted(out, key=lambda t: rank[t])
