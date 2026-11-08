"""v2.0 市場解讀：大盤資料序列、止穩檢查表、修正事件統計、修正期反彈候選、資金配置建議。

研究與預先宣告規則見 docs/v2.0_rebound_protocol.md；這裡的東西**不改動主選股模型**，只在「市場解讀」與「ETF」頁顯示。
所有指標只用當天收盤後已知的資料（國際市場用 align_world 對齊後的值）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 止穩檢查表（預先宣告的 8 項）
CHECKS = [
    ("c1", "不再破底", "近 3 日沒有創 20 日新低"),
    ("c2", "站回 5 日線", "加權指數收盤 > 5 日均線"),
    ("c3", "廣度轉強", "近 5 日上漲家數合計 > 下跌家數合計"),
    ("c4", "價漲量增", "近 3 日內有「上漲且成交金額 > 20 日均量」"),
    ("c5", "外資賣壓減輕", "近 5 日外資買賣超合計 > 前 5 日"),
    ("c6", "融資去槓桿", "融資張數 10 日減少 ≥ 3%"),
    ("c7", "費半止穩", "費半收盤 > 5 日均線"),
    ("c8", "恐慌消退", "VIX < 近 10 日最高 × 0.85"),
]
CORR_DD = -0.10            # 修正期：距 120 日最高收盤回落 ≥ 10%
DEV_END = "2024-01-01"     # 開發期／驗證期分界

# ------------------------------------------------------------------ 反彈分數（預先宣告規則選出的 4 個特性；見協議與 technical.md v2.0）
REBOUND = [("ret20", -1, "近 20 日跌幅大"), ("rev_rel", 1, "營收年增優於同業"), ("trust20", -1, "投信沒有追高（非擁擠）"),
           ("dd60", -1, "距 60 日高點跌深")]


def market_series(P) -> pd.DataFrame:
    """全期間的大盤日資料（個股加總估算，單位：億元／家數）＋國際市場。"""
    d = P.dates
    out = pd.DataFrame(index=d)
    close = P.m["close"].astype(float)
    px = close.ffill()
    trad = P.m["tradable"]
    out["close"] = P.taiex.reindex(d) if P.taiex is not None else np.nan
    ret = P.m["ret"].astype(float).where(trad)
    out["up"] = (ret > 0).sum(axis=1)
    out["down"] = (ret < 0).sum(axis=1)
    out["amount"] = P.m["amount"].astype(float).sum(axis=1, min_count=1) / 1e8
    for c in ("foreign", "trust", "dealer"):
        if c in P.m:
            out[c] = (P.m[c].astype(float) * px).sum(axis=1, min_count=1) / 1e8
    if "mbal" in P.m:
        out["mlots"] = P.m["mbal"].astype(float).sum(axis=1, min_count=1)
    valid = trad & px.notna()
    n = valid.sum(axis=1).replace(0, np.nan)
    ma20 = px.rolling(20, min_periods=15).mean()
    ma60 = px.rolling(60, min_periods=40).mean()
    out["ab20"] = ((px > ma20) & valid).sum(axis=1) / n
    out["ab60"] = ((px > ma60) & valid).sum(axis=1) / n
    out["hi20"] = ((px >= px.rolling(20, min_periods=15).max()) & valid).sum(axis=1)
    out["lo20"] = ((px <= px.rolling(20, min_periods=15).min()) & valid).sum(axis=1)
    if "mcap" in P.m and "pe" in P.m:
        mc = P.m["mcap"].astype(float)
        pe = P.m["pe"].astype(float)
        ok = (pe > 0) & mc.notna()
        out["pe_mkt"] = mc.where(ok).sum(axis=1) / (mc / pe).where(ok).sum(axis=1)
        if "dy" in P.m:
            dy = P.m["dy"].astype(float)
            okd = dy.notna() & mc.notna()
            out["dy_mkt"] = (mc * dy).where(okd).sum(axis=1) / mc.where(okd).sum(axis=1)
    W = P.world if P.world is not None else pd.DataFrame(index=d)
    for k, c in (("SOX", "sox"), ("VIX", "vix"), ("IXIC", "nasdaq"), ("TSM", "tsm"), ("TWD", "twd")):
        out[c] = W[k].reindex(d).astype(float) if k in W else np.nan
    return out


def checklist(m: pd.DataFrame) -> pd.DataFrame:
    """8 項止穩檢查（布林）＋分數；缺資料的項目＝False。"""
    tx = m["close"].astype(float)
    c = pd.DataFrame(index=m.index)
    newlow = tx <= tx.rolling(20, min_periods=15).min()
    c["c1"] = ~newlow.rolling(3, min_periods=1).max().fillna(1).astype(bool)
    c["c2"] = tx > tx.rolling(5).mean()
    c["c3"] = m["up"].rolling(5).sum() > m["down"].rolling(5).sum()
    upday = (tx.pct_change() > 0) & (m["amount"] > m["amount"].rolling(20).mean())
    c["c4"] = upday.rolling(3, min_periods=1).max().fillna(0).astype(bool)
    f5 = m["foreign"].rolling(5).sum() if "foreign" in m else pd.Series(np.nan, index=m.index)
    c["c5"] = f5 > f5.shift(5)
    ml = m["mlots"] if "mlots" in m else pd.Series(np.nan, index=m.index)
    c["c6"] = ml / ml.shift(10) - 1 <= -0.03
    sox = m["sox"].ffill() if "sox" in m else pd.Series(np.nan, index=m.index)
    c["c7"] = sox > sox.rolling(5).mean()
    vix = m["vix"].ffill() if "vix" in m else pd.Series(np.nan, index=m.index)
    c["c8"] = vix < vix.rolling(10).max() * 0.85
    c = c.fillna(False).astype(bool)
    c["score"] = c[[k for k, _, _ in CHECKS]].sum(axis=1)
    return c


def drawdown(tx: pd.Series, n: int = 120) -> pd.Series:
    return tx / tx.rolling(n, min_periods=20).max() - 1


def episodes(tx: pd.Series, thr: float = CORR_DD, gap: int = 20) -> list[tuple]:
    dd = drawdown(tx)
    corr = (dd <= thr).fillna(False)
    out = []
    run = None
    idx = list(tx.index)
    for i, (t, v) in enumerate(corr.items()):
        if v and run is None:
            run = [i, i]
        elif v:
            run[1] = i
        elif run is not None:
            out.append(run)
            run = None
    if run is not None:
        out.append(run)
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    res = []
    for a, b in merged:
        seg = tx.iloc[a:b + 1]
        lo = seg.idxmin()
        li = idx.index(lo)
        f60 = tx.iloc[li + 60] / tx.iloc[li] - 1 if li + 60 < len(tx) else np.nan
        res.append({"start": str(idx[a].date()), "end": str(idx[b].date()), "days": b - a + 1, "max_dd": float(dd.iloc[a:b + 1].min()),
                    "low_date": str(lo.date()), "after_low_60d": float(f60) if np.isfinite(f60) else None,
                    "ongoing": b == len(tx) - 1})
    return res


def history_stats(m: pd.DataFrame, c: pd.DataFrame | None = None) -> dict:
    """修正期（回落 ≥10%）日子依檢查表分數分組：之後 20／60 日加權指數報酬、20 日內再跌幅度；開發期／驗證期分列。"""
    c = checklist(m) if c is None else c
    tx = m["close"].astype(float)
    dd = drawdown(tx)
    corr = (dd <= CORR_DD).fillna(False)
    f20 = tx.shift(-20) / tx - 1
    f60 = tx.shift(-60) / tx - 1
    low20 = tx[::-1].rolling(20, min_periods=1).min()[::-1].shift(-1) / tx - 1
    t = pd.DataFrame({"score": c["score"], "f20": f20, "f60": f60, "low20": low20, "corr": corr})
    out = {"episodes": episodes(tx), "buckets": {}, "items": {}, "n_days": int(corr.sum())}
    for nm, msk in (("dev", t.index < DEV_END), ("val", t.index >= DEV_END), ("all", np.ones(len(t), bool))):
        r = t[msk & t["corr"] & t["f20"].notna()]
        r = r.assign(b=pd.cut(r["score"], [-1, 2, 4, 8], labels=["0–2", "3–4", "5–8"]))
        rows = []
        for lab, g in list(r.groupby("b", observed=False)) + [("全部", r)]:
            rows.append({"bucket": str(lab), "n": int(len(g)), "f20": _f(g["f20"].mean()), "win20": _f((g["f20"] > 0).mean()),
                         "f60": _f(g["f60"].mean()), "low20": _f(g["low20"].mean())})
        out["buckets"][nm] = rows
        items = {}
        for k, _, _ in CHECKS:
            on = c.loc[r.index, k]
            items[k] = {"on": _f(r.loc[on, "f20"].mean()), "off": _f(r.loc[~on, "f20"].mean()), "n_on": int(on.sum())}
        out["items"][nm] = items
    return out


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def state(m: pd.DataFrame, c: pd.DataFrame, stage: str = "") -> dict:
    """最新一天的市場狀態摘要（給市場解讀頁的結論與資金配置建議）。"""
    tx = m["close"].astype(float)
    last = m.index[-1]
    dd = float(drawdown(tx).iloc[-1])
    hi = float(tx.rolling(120, min_periods=20).max().iloc[-1])
    ma = {n: float(tx.rolling(n, min_periods=int(n * 0.7)).mean().iloc[-1]) for n in (5, 20, 60, 120, 240)}
    sc = int(c["score"].iloc[-1])
    px = float(tx.iloc[-1])
    trend_up = px > ma[60] and ma[20] > ma[60]
    if dd <= -0.20:
        regime = "空頭／深度修正"
    elif dd <= CORR_DD:
        regime = "修正"
    elif dd <= -0.05:
        regime = "拉回"
    else:
        regime = "多頭" if trend_up else "盤整"
    if dd <= CORR_DD:
        if sc <= 2:
            view, tone = "修正中、尚未止穩：資金觀望為主，想進場請分批、小量", "red"
        elif sc <= 4:
            view, tone = "修正中、出現部分止穩訊號：可開始分批布局", "amber"
        else:
            view, tone = "修正後止穩訊號多數出現：可逐步加碼", "green"
    elif dd <= -0.05:
        view, tone = ("拉回整理：照計畫持有，新資金分批", "amber") if sc <= 3 else ("拉回後轉強：照計畫進場", "green")
    else:
        view, tone = ("多頭格局：照模型操作", "green") if trend_up else ("盤整：照模型操作、控制持股檔數", "amber")
    return {"date": str(last.date()), "close": px, "dd": dd, "high120": hi, "ma": ma, "score": sc, "regime": regime,
            "view": view, "tone": tone, "stage": stage, "corr": dd <= CORR_DD}


# ------------------------------------------------------------------ ETF／個股資金配置（核心＋衛星）
PROFILES = {"保守型": 0.50, "穩健型": 0.20, "積極型": 0.0}     # ETF（0050 等大盤型）占比；與「資金配置」頁 plan.PROFILES 一致
TILT = 0.20


def allocation(profile: str, st: dict) -> dict:
    """依風險屬性的 ETF 核心占比；修正期（回落 ≥10%）且止穩分數 ≤2 → ETF 再加 20 個百分點
    （回測 2021-2026：50/50 配置加上這條規則，三個期間年化 +1.5～2pp、回撤略小；樣本約 5 次修正，見 technical.md v2.0）。"""
    base = PROFILES.get(profile, 0.20)
    tilt = TILT if (st.get("corr") and st.get("score", 9) <= 2) else 0.0
    etf = min(0.9, base + tilt)
    why = ("修正期且止穩訊號 ≤2 項：個股（β 較高）在這段時間通常跌得比 0050 多，暫時把 ETF 提高 20 個百分點；"
           "止穩訊號 ≥3 項或回落小於 10% 時恢復" if tilt else "目前不需調整，照風險屬性的基本比例")
    return {"etf": etf, "stock": 1 - etf, "base": base, "tilt": etf - base, "why": why}


def tilt_weights(w: dict, tilt: float) -> dict:
    """把 0050 提高 tilt（其餘依比例縮小），給「資金配置」頁用。"""
    if tilt <= 0:
        return dict(w)
    old = float(w.get("0050", 0.0))
    new = min(0.9, old + tilt)
    rest = 1 - old
    out = {k: (v * (1 - new) / rest if rest > 0 else 0.0) for k, v in w.items() if k != "0050"}
    out["0050"] = new
    return out


# ------------------------------------------------------------------ 修正期反彈候選（與主模型分開）
def rebound_candidates(P, rev_rel: np.ndarray | None, n_univ: int = 500, top: int = 30) -> list[dict]:
    """最新一天：成交金額前 500 名（上市滿 250 日）中，依 4 個預先選出的特性等權排名 → 反彈分數。"""
    adj = P.m["adj"].astype(float)
    i = len(P.dates) - 1
    adv = P.m["amount"].astype(float).rolling(60, min_periods=40).mean().iloc[i].to_numpy()
    listed = P.m["listed_days"].astype(float).iloc[i].to_numpy()
    a = np.where(np.isfinite(adv) & (listed >= 250), adv, -1)
    uni = np.argsort(-a)[:n_univ]
    uni = uni[a[uni] > 0]
    if not len(uni):
        return []
    ret20 = (adj.iloc[i] / adj.iloc[i - 20] - 1).to_numpy() if i >= 20 else np.full(adj.shape[1], np.nan)
    hi60 = adj.iloc[max(0, i - 59):i + 1].max().to_numpy()
    dd60 = adj.iloc[i].to_numpy() / hi60 - 1
    sh = P.m["shares"].astype(float).iloc[i].to_numpy()
    tr20 = P.m["trust"].astype(float).iloc[max(0, i - 19):i + 1].sum().to_numpy() / sh if "trust" in P.m else np.full(len(sh), np.nan)
    comp = {"ret20": ret20, "rev_rel": rev_rel if rev_rel is not None else np.full(len(sh), np.nan), "trust20": tr20, "dd60": dd60}
    sc = np.zeros(len(uni))
    cnt = np.zeros(len(uni))
    for k, sgn, _ in REBOUND:
        x = comp[k][uni] * sgn
        ok = np.isfinite(x)
        r = pd.Series(x).rank(pct=True).to_numpy()
        sc[ok] += r[ok]
        cnt[ok] += 1
    score = np.where(cnt >= 3, sc / np.maximum(cnt, 1), np.nan)
    order = np.argsort(-np.nan_to_num(score, nan=-1))[:top]
    tick = list(P.tickers)
    close = P.m["close"].astype(float).iloc[i].to_numpy()
    out = []
    for o in order:
        j = uni[o]
        if not np.isfinite(score[o]):
            continue
        out.append({"ticker": tick[j], "name": str(P.info["name"].iloc[j] or ""), "sector": str(P.info["sector"].iloc[j] or ""),
                    "close": _f(close[j]), "score": _f(score[o]), "ret20": _f(ret20[j]), "dd60": _f(dd60[j]),
                    "rev_rel": _f(comp["rev_rel"][j]), "trust20": _f(tr20[j])})
    return out


# ------------------------------------------------------------------ v2.2 急跌後的歷史基準（docs/v2.2_crash_protocol.md 的 D）
CRASH_TYPES = {"E1": "加權指數距一年高點回落達 15%", "E2": "前一晚費半跌 ≥7%（台股休市期間累計）", "E3": "加權指數連 5 跌以上、累計 ≥8%"}

# 研究時（資料到 2026-10-22）的結果；每日研究跑過後會用最新資料重算（B.meta["crash_stats"]）
CRASH_FALLBACK = {
    "asof": "2026-10-22",
    "summary": {
        "E1": {"n": 6, "r20": 0.056, "win20": 0.83, "r60": 0.082, "win60": 0.80, "min20": -0.043, "min60": -0.078, "worst60": -0.160,
               "low_hits": 1, "low_n": 5, "lump60": 0.082, "staged60": 0.085},
        "E2": {"n": 14, "r20": 0.040, "win20": 0.79, "r60": 0.114, "win60": 1.0, "min20": -0.062, "min60": -0.072, "worst60": -0.211,
               "low_hits": 2, "low_n": 14, "lump60": 0.114, "staged60": 0.113, "day0": -0.031},
        "E3": {"n": 3, "r20": 0.101, "win20": 1.0, "r60": 0.092, "win60": 0.5, "min20": -0.006, "min60": -0.056, "worst60": -0.080,
               "low_hits": 0, "low_n": 2, "lump60": 0.092, "staged60": 0.082},
    },
    "events": [],
}


def crash_stats(tx: pd.Series, sox: pd.Series | None = None) -> dict:
    """急跌事件（E1 回落 15%、E2 費半單日 ≤−7%、E3 連 5 跌 ≥8%）之後的加權指數走勢。
    tx：加權指數收盤（台股交易日）；sox：已對齊到台股日期的費半收盤（T 日＝前一晚美股）。"""
    tx = tx.astype(float).dropna()
    n = len(tx)
    if n < 300:
        return {}

    def fwd(i, k):
        return float(tx.iloc[i + k] / tx.iloc[i] - 1) if i + k < n else None

    def mind(i, k):
        seg = tx.iloc[i + 1: i + k + 1]
        return float(seg.min() / tx.iloc[i] - 1) if len(seg) == k else None

    def staged(i):
        if i + 60 >= n:
            return None, None
        cost = float(np.mean([tx.iloc[i], tx.iloc[i + 5], tx.iloc[i + 10]]))
        return float(tx.iloc[i + 60] / tx.iloc[i] - 1), float(tx.iloc[i + 60] / cost - 1)

    def row(i, typ, **kw):
        seg = tx.iloc[i + 1: i + 61]
        is_low = bool(seg.min() >= tx.iloc[i]) if len(seg) == 60 else None
        lump, stg = staged(i)
        hi = float(tx.iloc[max(0, i - 250): i + 1].max())
        return {"type": typ, "date": str(tx.index[i].date()), "close": float(tx.iloc[i]), "dd250": float(tx.iloc[i] / hi - 1),
                "r5": fwd(i, 5), "r20": fwd(i, 20), "r60": fwd(i, 60), "min20": mind(i, 20), "min60": mind(i, 60),
                "is_low60": is_low, "lump60": lump, "staged60": stg, **kw}
    ev = []
    dd = tx / tx.rolling(250, min_periods=60).max() - 1
    armed = True
    for i in range(n):
        if armed and dd.iloc[i] <= -0.15:
            ev.append(row(i, "E1"))
            armed = False
        if not armed and dd.iloc[i] > -0.05:
            armed = True
    if sox is not None and len(sox.dropna()) > 300:
        s = sox.reindex(tx.index).ffill()
        sr = s.pct_change()
        for i in np.where((sr <= -0.07).to_numpy())[0]:
            ev.append(row(int(i), "E2", sox=float(sr.iloc[i]), day0=float(tx.iloc[i] / tx.iloc[i - 1] - 1)))
    rt = tx.pct_change().to_numpy()
    streak, start, fired = 0, None, False
    for i in range(1, n):
        if rt[i] < 0:
            if streak == 0:
                start = i - 1
            streak += 1
            if streak >= 5 and not fired and tx.iloc[i] / tx.iloc[start] - 1 <= -0.08:
                ev.append(row(i, "E3", streak=streak))
                fired = True
        else:
            streak, fired = 0, False
    t = pd.DataFrame(ev)
    summ = {}
    for typ in CRASH_TYPES:
        g = t[t["type"] == typ] if len(t) else pd.DataFrame()
        if not len(g):
            continue
        low = g["is_low60"].dropna()
        both = g.dropna(subset=["lump60", "staged60"])
        summ[typ] = {"n": int(len(g)), "r20": _f(g["r20"].mean()), "win20": _f((g["r20"].dropna() > 0).mean()),
                     "r60": _f(g["r60"].mean()), "win60": _f((g["r60"].dropna() > 0).mean()), "min20": _f(g["min20"].mean()),
                     "min60": _f(g["min60"].mean()), "worst60": _f(g["min60"].min()), "low_hits": int(low.sum()), "low_n": int(len(low)),
                     "lump60": _f(both["lump60"].mean()), "staged60": _f(both["staged60"].mean())}
        if typ == "E2" and "day0" in g:
            summ[typ]["day0"] = _f(g["day0"].mean())
    # 目前狀態
    i = n - 1
    k = 0
    while k < i and rt[i - k] < 0:
        k += 1
    cur = {"date": str(tx.index[-1].date()), "close": float(tx.iloc[-1]), "dd250": float(dd.iloc[-1]),
           "high250": float(tx.tail(250).max()), "streak": int(k), "streak_ret": float(tx.iloc[-1] / tx.iloc[-1 - k] - 1) if k else 0.0}
    if sox is not None and len(sox.dropna()) > 2:
        s = sox.reindex(tx.index).ffill()
        cur["sox_day"] = _f(s.iloc[-1] / s.iloc[-2] - 1)
        cur["sox_dd"] = _f(s.iloc[-1] / s.tail(250).max() - 1)
    events = [{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()} for r in ev]
    return {"asof": str(tx.index[-1].date()), "summary": summ, "events": sorted(events, key=lambda r: r["date"])[-40:], "current": cur}


def crash_flags(cur: dict) -> list[tuple[str, str, bool]]:
    """目前觸發了哪些急跌條件：[(代號, 說明, 是否觸發)]。"""
    if not cur:
        return []
    return [("E1", f"距一年高點 {cur.get('dd250', 0):+.1%}（門檻 −15%）", (cur.get("dd250") or 0) <= -0.15),
            ("E2", f"前一晚費半 {cur['sox_day']:+.1%}（門檻 −7%）" if cur.get("sox_day") is not None else "費半資料不足",
             (cur.get("sox_day") or 0) <= -0.07),
            ("E3", f"連 {cur.get('streak', 0)} 跌、累計 {cur.get('streak_ret', 0):+.1%}（門檻 連5跌 −8%）",
             cur.get("streak", 0) >= 5 and (cur.get("streak_ret") or 0) <= -0.08)]


def intraday_checks(tx: dict | None, nq: dict | None = None, tsmc: dict | None = None) -> list[tuple[str, str, bool | None]]:
    """盤中止跌觀察（只用當天分時）：加權從低點反彈 ≥1%、加權高於開盤、那指期翻紅、台積電從低點反彈 ≥1.5%。
    tx／nq／tsmc：kosto.intraday 的資料（last、low、open、prev）。"""
    out = []
    if tx:
        rb = tx["last"] / tx["low"] - 1 if tx.get("low") else np.nan
        out.append(("加權從今日低點反彈", f"{rb:+.2%}（≥ +1% 成立）" if np.isfinite(rb) else "–", bool(rb >= 0.01) if np.isfinite(rb) else None))
        oc = tx["last"] / tx["open"] - 1 if tx.get("open") else np.nan
        out.append(("加權高於今日開盤", f"{oc:+.2%}" if np.isfinite(oc) else "–", bool(oc > 0) if np.isfinite(oc) else None))
    if nq:
        r = nq["last"] / nq["prev"] - 1
        out.append(("那斯達克期貨翻紅", f"{r:+.2%}", bool(r > 0)))
    if tsmc:
        rb = tsmc["last"] / tsmc["low"] - 1 if tsmc.get("low") else np.nan
        out.append(("台積電從低點反彈", f"{rb:+.2%}（≥ +1.5% 成立）" if np.isfinite(rb) else "–", bool(rb >= 0.015) if np.isfinite(rb) else None))
    return out
