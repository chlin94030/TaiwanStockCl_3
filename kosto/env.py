"""v3.13 環境判定卡、站上各均線比例、估值溫度、關鍵價位（底線）、個股標籤（docs/v3.13_protocol.md）。

參考 Yoda Research「綜合報告」的版面重建；所有東西只做顯示，不影響選股。
歷史驗證（只報告）：環境判定卡與止跌條件都不能預測之後的大盤報酬；Yoda 式選股標籤加進模型會變差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORT = (5, 10, 20)
LONG = (20, 60, 120)
STATE_ZH = {3: "順風", 2: "偏強", 1: "偏弱", 0: "逆風"}
STATE_TONE = {3: "up", 2: "up", 1: "down", 0: "down"}
BREADTH_WINS = (5, 10, 20, 60, 120, 240)
LEVELS = [(20, "月線"), (60, "季線"), (120, "半年線"), (240, "年線")]
TAGS = {"Y1": "外資連買＋創月高", "Y2": "法人週月都買", "Y3": "剛放量起漲", "LEAD": "族群帶頭",
        "R_UP": "營收利多順漲", "R_DN": "營收利多反跌", "R_RB": "營收利空反漲"}


def card_counts(s: pd.Series, wins=SHORT) -> pd.Series:
    """每天收盤高於幾條均線（0～len(wins)）；均線資料不足的日子為 NaN。"""
    s = pd.Series(s, dtype=float).dropna()
    out = pd.Series(0.0, index=s.index)
    ok = pd.Series(True, index=s.index)
    for w in wins:
        ma = s.rolling(w, min_periods=int(w * 0.8)).mean()
        ok &= ma.notna()
        out += (s > ma).astype(float)
    return out.where(ok)


def state_of(n) -> tuple[int, str]:
    if n is None or not np.isfinite(n):
        return -1, "資料不足"
    k = int(round(float(n)))
    return k, STATE_ZH.get(k, "資料不足")


def env_card(s: pd.Series, n_hist: int = 60) -> dict:
    """一個指數的短線（5／10／20 日線）與長線（20／60／120 日線）判定＋最近 n_hist 天的條數。"""
    s = pd.Series(s, dtype=float).dropna()
    if len(s) < 30:
        return {}
    cs, cl = card_counts(s, SHORT), card_counts(s, LONG)
    ks, zs = state_of(cs.iloc[-1])
    kl, zl = state_of(cl.iloc[-1])
    return {"date": str(s.index[-1].date()), "close": float(s.iloc[-1]), "short": ks, "short_zh": zs, "long": kl, "long_zh": zl,
            "short_hist": cs.tail(n_hist).dropna().tolist(), "long_hist": cl.tail(n_hist).dropna().tolist(),
            "short_days": _days_in_state(cs), "long_days": _days_in_state(cl)}


def _days_in_state(c: pd.Series) -> int:
    c = c.dropna()
    if c.empty:
        return 0
    last, n = c.iloc[-1], 0
    for v in c.iloc[::-1]:
        if v != last:
            break
        n += 1
    return n


def ma_breadth(px: pd.DataFrame, valid: pd.DataFrame, wins=BREADTH_WINS) -> pd.DataFrame:
    """站上各條均線的股票比例（0～1）；px＝收盤（已往前補值），valid＝可交易且有價格。"""
    out = {}
    for w in wins:
        ma = px.rolling(w, min_periods=int(w * 0.75)).mean()
        ok = valid & ma.notna()
        n = ok.sum(axis=1).replace(0, np.nan)
        out[f"ab{w}"] = ((px > ma) & ok).sum(axis=1) / n
    return pd.DataFrame(out)


def key_levels(tx: pd.Series) -> list[dict]:
    """加權指數的關鍵價位與收盤距離（底線）：月／季／半年／年線、近 60 日低點、近 20 日高點。"""
    tx = pd.Series(tx, dtype=float).dropna()
    if len(tx) < 25:
        return []
    px = float(tx.iloc[-1])
    rows = []
    for w, nm in LEVELS:
        if len(tx) >= int(w * 0.8):
            lv = float(tx.tail(w).mean())
            rows.append({"name": f"{nm}（{w} 日均）", "level": lv, "dist": px / lv - 1, "above": px >= lv})
    lo = float(tx.tail(60).min())
    hi = float(tx.tail(20).max())
    rows.append({"name": "近 60 日最低收盤", "level": lo, "dist": px / lo - 1, "above": px >= lo})
    rows.append({"name": "近 20 日最高收盤", "level": hi, "dist": px / hi - 1, "above": px >= hi})
    return rows


def valuation(pe: pd.Series, min_days: int = 750) -> dict:
    """市值加權本益比的歷史百分位（到當天為止的全部歷史；至少 min_days 天）＋月底序列。"""
    pe = pd.Series(pe, dtype=float).dropna()
    pe = pe[pe > 0]
    if len(pe) < 20:
        return {}
    last = float(pe.iloc[-1])
    pct = float((pe <= last).mean()) if len(pe) >= min_days else np.nan
    me = pe.groupby(pe.index.to_period("M")).last()
    hist = [[str(p), float(v)] for p, v in me.items()]
    hi12 = float(pe.tail(250).max())
    return {"asof": str(pe.index[-1].date()), "pe": last, "pct": pct, "since": str(pe.index[0].date()), "n_days": int(len(pe)),
            "min": float(pe.min()), "max": float(pe.max()), "median": float(pe.median()), "high_12m": hi12,
            "off_high_12m": last / hi12 - 1, "hist": hist}


def stock_tags(close: pd.Series, volume: pd.Series, foreign: pd.Series, total: pd.Series) -> list[str]:
    """單一個股最近一天的 Yoda 式標籤（Y1～Y3）；序列為日資料（張、元），缺資料就不標。"""
    out = []
    c = pd.Series(close, dtype=float).ffill()
    if len(c.dropna()) < 61:
        return out
    hi20 = bool(c.iloc[-1] >= c.tail(20).max() - 1e-9)
    f = pd.Series(foreign, dtype=float)
    if len(f.dropna()) >= 3 and bool((f.tail(3) > 0).all()) and hi20:
        out.append("Y1")
    t = pd.Series(total, dtype=float)
    if len(t.dropna()) >= 20 and t.tail(5).sum() > 0 and t.tail(20).sum() > 0:
        out.append("Y2")
    v = pd.Series(volume, dtype=float)
    if len(v.dropna()) >= 60 and v.tail(5).mean() >= 1.5 * v.tail(60).mean() and hi20 and c.iloc[-1] / c.iloc[-61] - 1 <= 0.30:
        out.append("Y3")
    return out


def revenue_reaction(yoy_rank: float, react_rank: float) -> str | None:
    """營收公布反應分類（T5）：yoy_rank／react_rank＝當月全市場百分位（0～1）。"""
    if not (np.isfinite(yoy_rank) and np.isfinite(react_rank)):
        return None
    if yoy_rank >= 0.7 and react_rank >= 0.7:
        return "R_UP"
    if yoy_rank >= 0.7 and react_rank <= 0.3:
        return "R_DN"
    if yoy_rank <= 0.3 and react_rank >= 0.7:
        return "R_RB"
    return None
