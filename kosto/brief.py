"""個股基本面短評（v1.8）：由月營收、本益比推算的近四季 EPS、估值位置、殖利率自動整理「亮點／隱憂」與一句總評。

資料只用公開數字（月營收、交易所每日本益比／殖利率），不含法說會展望與新聞；這兩項在 App 以連結補充。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def revenue_frame(B, j: int) -> pd.DataFrame:
    """月營收（億元）、年增、月增；index＝月份（Period）。"""
    if "REV_M" not in B.a or not B.meta.get("rev_ym"):
        return pd.DataFrame()
    v = B.a["REV_M"][:, j].astype(float)
    idx = pd.PeriodIndex([pd.Period(f"{ym // 100}-{ym % 100:02d}", "M") for ym in B.meta["rev_ym"]])
    s = pd.Series(v, index=idx)
    f = pd.DataFrame({"rev": s / 1e8})
    f["yoy"] = s / s.shift(12) - 1
    f["mom"] = s / s.shift(1) - 1
    f = f[np.isfinite(f["rev"]) & (f["rev"] > 0)]
    return f


def valuation_frame(B, j: int) -> pd.DataFrame:
    qd = B.meta.get("q_days")
    if not qd or "Q_PE" not in B.a:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(qd)
    f = pd.DataFrame({"close": B.a["Q_C"][:, j], "pe": B.a["Q_PE"][:, j], "pb": B.a["Q_PB"][:, j], "dy": B.a["Q_DY"][:, j]},
                     index=idx).astype(float)
    f["pe"] = f["pe"].where(f["pe"] > 0)
    f["eps"] = f["close"] / f["pe"]                       # 近四季 EPS（交易所本益比 = 股價 / 近四季 EPS）
    return f


def brief(B, j: int) -> dict:
    """回傳 {'tone': 正向/中性/保守, 'title': 一句總評, 'good': [...], 'bad': [...], 'rev': DataFrame}。"""
    good, bad, score, val = [], [], 0, ""
    rv = revenue_frame(B, j)
    if len(rv):
        last = rv.iloc[-1]
        m = rv.index[-1]
        yoy, mom = _f(last["yoy"]), _f(last["mom"])
        txt = f"{m.month} 月營收 {last['rev']:,.1f} 億"
        if np.isfinite(yoy):
            txt += f"，年增 {yoy:+.0%}"
        if np.isfinite(mom):
            txt += f"、月增 {mom:+.0%}"
        (good if (np.isfinite(yoy) and yoy >= 0.1) else bad if (np.isfinite(yoy) and yoy < 0) else good).append(txt)
        score += 1 if np.isfinite(yoy) and yoy >= 0.15 else (-1 if np.isfinite(yoy) and yoy < 0 else 0)
        if len(rv) >= 15:
            r3 = rv["rev"].iloc[-3:].sum() / rv["rev"].iloc[-15:-12].sum() - 1 if rv["rev"].iloc[-15:-12].sum() > 0 else np.nan
            p3 = rv["rev"].iloc[-6:-3].sum() / rv["rev"].iloc[-18:-15].sum() - 1 if len(rv) >= 18 and rv["rev"].iloc[-18:-15].sum() > 0 else np.nan
            if np.isfinite(r3):
                if np.isfinite(p3) and r3 > p3 + 0.05 and r3 > 0:
                    good.append(f"近 3 個月營收年增 {r3:+.0%}，成長加速（前 3 個月 {p3:+.0%}）")
                    score += 1
                elif np.isfinite(p3) and r3 < p3 - 0.1:
                    bad.append(f"近 3 個月營收年增 {r3:+.0%}，成長放緩（前 3 個月 {p3:+.0%}）")
                    score -= 1
        yo = rv["yoy"].dropna()
        streak = 0
        for x in yo[::-1]:
            if x > 0:
                streak += 1
            else:
                break
        if streak >= 6:
            good.append(f"營收連續 {streak} 個月年增")
        if len(rv) >= 13 and last["rev"] >= rv["rev"].iloc[-24:].max() * 0.999:
            good.append("單月營收創近兩年新高")
            score += 1
    vf = valuation_frame(B, j)
    if len(vf) and vf["pe"].notna().any():
        pe = _f(vf["pe"].dropna().iloc[-1])
        eps_now = _f(vf["eps"].dropna().iloc[-1])
        e1 = vf["eps"].dropna()
        e1 = e1[e1.index <= e1.index[-1] - pd.Timedelta(days=360)]
        eps_1y = _f(e1.iloc[-1]) if len(e1) else np.nan
        if np.isfinite(eps_now):
            t = f"近四季 EPS 約 {eps_now:.2f} 元"
            if np.isfinite(eps_1y) and eps_1y > 0:
                g = eps_now / eps_1y - 1
                t += f"（一年前 {eps_1y:.2f} 元，{g:+.0%}）"
                if g >= 0.15:
                    good.append(t)
                    score += 1
                elif g < -0.1:
                    bad.append(t)
                    score -= 1
                else:
                    good.append(t)
            else:
                good.append(t)
        pes = vf["pe"].dropna()
        if np.isfinite(pe) and len(pes) > 120:
            pct = float((pes < pe).mean())
            lvl = "偏低" if pct < 0.3 else ("偏高" if pct > 0.8 else "中間")
            val = lvl
            t = f"本益比 {pe:.1f} 倍，位於近兩年 {pct:.0%} 分位（{lvl}）"
            (bad if pct > 0.8 else good).append(t)
            score += -1 if pct > 0.9 else (1 if pct < 0.2 else 0)
    elif len(vf):
        bad.append("近四季虧損（無本益比）")
        score -= 1
    if len(vf) and vf["dy"].notna().any():
        dy = _f(vf["dy"].dropna().iloc[-1])
        if np.isfinite(dy) and dy >= 4:
            good.append(f"現金殖利率 {dy:.1f}%")
    tone = "正向" if score >= 2 else ("保守" if score <= -1 else "中性")
    grow = {"正向": "營收／獲利成長強", "中性": "營運平穩", "保守": "營運動能轉弱"}[tone]
    vtxt = {"偏低": "估值偏低", "中間": "估值合理", "偏高": "但估值已偏高、追價要小心"}.get(val, "")
    if tone == "保守" and val == "偏高":
        vtxt = "且估值偏高"
    title = f"{grow}，{vtxt}" if vtxt else grow
    if not good and not bad:
        title, tone = "資料不足（新上市或尚無月營收／本益比）", "中性"
    return {"tone": tone, "title": title, "good": good[:5], "bad": bad[:3], "rev": rv}


def links(ticker: str, board: str = "") -> list[tuple[str, str]]:
    sfx = "TWO" if board == "TPEX" else "TW"
    return [("新聞", f"https://tw.stock.yahoo.com/quote/{ticker}.{sfx}/news"),
            ("營收／財報", f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={ticker}"),
            ("公開資訊觀測站（法說會／重大訊息）", "https://mops.twse.com.tw/mops/web/index")]
