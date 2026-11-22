"""v3.17 崩跌因應（docs/v3.17_protocol.md）：市況定位、去槓桿進度、反轉觀察清單、個股急跌風險分級。

全部只做「顯示與提醒」，不改選股模型：v3.17 測過的急跌選股規則（避開高融資、避開連續跌停、減碼轉 0050、改用反彈候選）
都沒有通過預先寫好的門檻。歷史統計來自 2017～2026 的 6 次 0050 跌 15% 以上的崩跌（review.V317_*）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PANIC = {"r20": -0.08, "dd120": 0.12}         # 急跌狀態（v2.2 定義）
LIMIT = -0.094                                # 收盤跌停（含四捨五入誤差）


def _nav(B) -> pd.Series:
    return pd.Series(np.cumprod(1 + np.nan_to_num(np.asarray(B.a["BENCH"], float))), index=pd.DatetimeIndex(B.days))


def market_state(B) -> dict:
    """最新研究資料日的市況：0050 距高點、急跌狀態、融資餘額自 120 日高點的變化、跌停家數、外資、VIX、費半。"""
    out = {}
    try:
        nav = _nav(B)
        d = nav.index[-1]
        out["date"] = str(d.date())
        out["dd250"] = float(nav.iloc[-1] / nav.tail(250).max() - 1)
        out["dd120"] = float(nav.iloc[-1] / nav.tail(120).max() - 1)
        out["r20"] = float(nav.iloc[-1] / nav.iloc[-21] - 1) if len(nav) > 21 else np.nan
        out["r5"] = float(nav.iloc[-1] / nav.iloc[-6] - 1) if len(nav) > 6 else np.nan
        out["panic"] = bool(out["r20"] <= PANIC["r20"] or out["dd120"] <= -PANIC["dd120"])
        R = np.asarray(B.a["R"][-1], float)
        out["limit_down"] = int(np.nansum(R <= LIMIT))
        out["limit_up"] = int(np.nansum(R >= -LIMIT))
        Rk = np.asarray(B.a["R"][-5:], float)
        out["limit_down5"] = [int(np.nansum(x <= LIMIT)) for x in Rk]
        if "Q_MKT" in B.a:
            cols = list(B.meta.get("q_mkt_cols") or [])
            M = pd.DataFrame(B.a["Q_MKT"].astype(float), index=pd.DatetimeIndex(B.meta["q_days"]), columns=cols)
            if "mbal" in M:
                mb = M["mbal"].dropna()
                if len(mb) > 20:
                    out["margin"] = float(mb.iloc[-1])
                    out["margin_chg120"] = float(mb.iloc[-1] / mb.tail(120).max() - 1)
                    out["margin_chg5"] = float(mb.iloc[-1] / mb.iloc[-6] - 1)
            if "foreign" in M:
                f = M["foreign"].dropna()
                out["foreign1"], out["foreign5"] = float(f.iloc[-1]), float(f.tail(5).sum())
                out["foreign_streak"] = int(_streak(f))
            for k in ("vix", "sox"):
                if k in M and M[k].notna().any():
                    s = M[k].dropna()
                    out[k] = float(s.iloc[-1])
                    if k == "sox" and len(s) > 20:
                        out["sox_vs_ma20"] = float(s.iloc[-1] / s.tail(20).mean() - 1)
                    if k == "vix":
                        out["vix_max20"] = float(s.tail(20).max())
            if "close" in M:
                c = M["close"].dropna()
                out["taiex"] = float(c.iloc[-1])
                out["taiex_dd"] = float(c.iloc[-1] / c.tail(250).max() - 1)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def _streak(s: pd.Series) -> int:
    """連續買超（正）或賣超（負）天數。"""
    n, sign = 0, 0
    for x in s.iloc[::-1]:
        sg = 1 if x > 0 else -1 if x < 0 else 0
        if sg == 0 or (sign and sg != sign):
            break
        sign = sg
        n += 1
    return n * sign


def regime(st: dict) -> tuple[str, str]:
    """(代碼, 中文)：normal 正常／correction 修正（距高點 −10%）／crash 急跌（急跌狀態或距高點 −20%）。"""
    dd = st.get("dd250", 0) or 0
    if st.get("panic") or dd <= -0.20:
        return "crash", "急跌"
    if dd <= -0.10:
        return "correction", "修正"
    return "normal", "正常"


def reversal_checklist(st: dict, live_vix: float | None = None) -> list[dict]:
    """反轉觀察清單（v3.17 歷史統計：急跌開始後第一次出現 → 之後 60 日 0050 平均報酬）。"""
    from .review import V317_SIGNALS
    hist = {x[0]: x for x in V317_SIGNALS}
    vix = live_vix if live_vix is not None and np.isfinite(live_vix) else st.get("vix")
    ld = st.get("limit_down5") or []
    rows = [
        ("VIX 衝上 35 後回到 30 以下", (st.get("vix_max20") or 0) >= 35 and vix is not None and vix < 30,
         f"VIX {vix:.1f}（近 20 日最高 {st.get('vix_max20', float('nan')):.1f}）" if vix is not None else "VIX 資料缺"),
        ("跌停家數由 ≥200 降到 ≤30", bool(ld) and max(ld) >= 200 and ld[-1] <= 30,
         "近 5 日跌停家數 " + "、".join(str(x) for x in ld) if ld else "資料缺"),
        ("融資餘額自高點減 25%", (st.get("margin_chg120") or 0) <= -0.25,
         f"融資餘額較 120 日高點 {st['margin_chg120'] * 100:+.0f}%" if st.get("margin_chg120") is not None else "資料缺"),
        ("外資連 3 日買超", (st.get("foreign_streak") or 0) >= 3,
         f"外資連 {abs(st.get('foreign_streak', 0))} 日{'買' if (st.get('foreign_streak') or 0) > 0 else '賣'}超" if st.get("foreign_streak") else "資料缺"),
        ("費半站回 20 日線", (st.get("sox_vs_ma20") or -1) > 0,
         f"費半距 20 日線 {st['sox_vs_ma20'] * 100:+.1f}%" if st.get("sox_vs_ma20") is not None else "資料缺"),
    ]
    out = []
    for name, ok, now in rows:
        h = hist.get(name)
        out.append({"name": name, "ok": bool(ok), "now": now, "n": h[1] if h else None, "days_vs_trough": h[2] if h else None,
                    "r20": h[3] if h else None, "r60": h[4] if h else None, "further": h[5] if h else None})
    return out


def analogs(dd: float) -> list[dict]:
    """歷史上 0050 也跌到這個深度的崩跌：之後最多再跌多少、何時見底、60／120 日反彈。"""
    from .review import V317_EPISODES
    out = []
    for pk, tr, d, ndown, nrec, mchg, mld, vix, mmod, mreb, b60, b120 in V317_EPISODES:
        if d / 100 <= dd + 0.02:                                        # 最後跌得比現在還深（或差不多）的事件
            out.append({"peak": pk, "trough": tr, "dd": d, "further": (1 + d / 100) / (1 + dd) - 1 if dd > -1 else np.nan,
                        "margin": mchg, "r60": b60, "r120": b120, "recover_days": nrec})
    return out


def crash_risk(B) -> dict:
    """個股急跌風險：融資使用率（融資餘額÷融資限額）與 60 日波動在成交金額前 800 名中的百分位平均 → 高（前 20%）／中／低（後 40%）。
    v3.17 歷史：高風險組在 6 次崩跌平均跌 31%、57% 出現 2 次以上收盤跌停；低風險組跌 16%、12%。"""
    try:
        if "Q_MB" not in B.a or "Q_ML" not in B.a or "VOL60" not in B.a:
            return {}
        mb = np.asarray(B.a["Q_MB"][-1], float)
        ml = np.asarray(B.a["Q_ML"][-1], float)
        mu = np.where(ml > 0, mb / np.where(ml > 0, ml, 1), np.nan)
        vol = np.asarray(B.a["VOL60"][-1], float)
        cap = np.asarray(B.a["CAP"][-1], float) if "CAP" in B.a else np.ones(len(mu))
        liquid = pd.Series(np.nan_to_num(cap)).rank(ascending=False).to_numpy() <= 800
        f = pd.DataFrame({"mu": mu, "vol": vol}, index=list(B.tickers))
        sub = f[liquid]
        pm, pv = sub["mu"].rank(pct=True), sub["vol"].rank(pct=True)
        sc = ((pm.fillna(0.5) + pv.fillna(0.5)) / 2).rank(pct=True)
        tier = pd.cut(sc, [0, 0.4, 0.8, 1.0], labels=["低", "中", "高"], include_lowest=True)
        return {t: {"tier": str(tier[t]), "score": float(sc[t]), "mu": None if not np.isfinite(sub.at[t, "mu"]) else float(sub.at[t, "mu"]),
                    "mu_pct": float(pm[t]) if np.isfinite(pm[t]) else None, "vol_pct": float(pv[t]) if np.isfinite(pv[t]) else None}
                for t in sub.index}
    except Exception:  # noqa: BLE001
        return {}


def model_weather(B, live: dict | None = None) -> dict:
    """v3.18 模型天氣：現在處於哪些狀態、歷史上中線模型在這些狀態的年化超額（review.V318_REGIME）。
    live 可帶較新的 {"vix": 即時 VIX, "sox": 費半最新收盤, "dd250": 加權今天距一年高點}（只用來更新狀態，不改歷史統計）。"""
    from .review import V318_REGIME
    live = live or {}
    stat = {(d, s): (m, n, t, sh, lg, weak) for d, s, m, n, t, sh, lg, weak in V318_REGIME}
    out = {"rows": [], "weak": 0}
    try:
        nav = _nav(B)
        st = market_state(B)
        keys = list(B.meta.get("world_keys") or [])
        W = pd.DataFrame(np.asarray(B.a["WORLD"], float), index=pd.DatetimeIndex(B.days), columns=keys) if "WORLD" in B.a and keys else pd.DataFrame()
        out["date"] = str(nav.index[-1].date())
        ma200 = nav.rolling(200, min_periods=150).mean().iloc[-1]
        above = bool(nav.iloc[-1] > ma200) if np.isfinite(ma200) else None
        if live.get("dd250") is not None and np.isfinite(live["dd250"]) and live["dd250"] <= -0.15:
            above = False                                               # 今天加權距一年高點 −15% 以上 → 視同跌破長期均線
        rows = []
        if above is not None:
            rows.append(("台股趨勢", "0050 在 200 日線之上" if above else "0050 在 200 日線之下"))
        code, _ = regime({**st, "dd250": min(st.get("dd250", 0) or 0, live.get("dd250") or 0)})
        r = rebound_window(B)
        rows.append(("台股市況", "急跌" if code == "crash" else "急跌結束後 12 週（反彈期）" if r else "修正（距一年高 −10%）" if code == "correction" else "正常"))
        v60 = pd.Series(np.asarray(B.a["BENCH"], float)).rolling(60, min_periods=40).std()
        rows.append(("台股波動", "60 日波動偏高" if v60.iloc[-1] > v60.tail(250).median() else "60 日波動偏低"))
        if "VIX" in W:
            vix = live.get("vix") if live.get("vix") is not None else float(W["VIX"].dropna().iloc[-1])
            rows.append(("美股恐慌", "VIX ≥ 30" if vix >= 30 else "VIX 20～30" if vix >= 20 else "VIX < 20"))
            out["vix"] = vix
        if "SOX" in W:
            sx = W["SOX"].dropna()
            last = live.get("sox") if live.get("sox") is not None else float(sx.iloc[-1])
            ma50 = float(pd.concat([sx, pd.Series([last])]).tail(50).mean()) if live.get("sox") is not None else float(sx.tail(50).mean())
            rows.append(("美股半導體", "費半在 50 日線之上" if last > ma50 else "費半在 50 日線之下"))
            out["sox_vs_ma50"] = last / ma50 - 1
            out["sox_ma50"] = float(sx.tail(50).mean())
            if "VIX" in W:
                out["us_off"] = bool(last < ma50 and out["vix"] >= 25)        # W7 觸發條件
        if "TWD" in W:
            tw = W["TWD"].dropna()
            if len(tw) > 61:
                rows.append(("匯率", "新台幣近 60 日貶值" if tw.iloc[-1] / tw.iloc[-61] - 1 > 0 else "新台幣近 60 日升值"))
        if st.get("margin_chg5") is not None and "Q_MKT" in B.a:
            cols = list(B.meta.get("q_mkt_cols") or [])
            if "mbal" in cols:
                mb = pd.Series(np.asarray(B.a["Q_MKT"], float)[:, cols.index("mbal")]).dropna()
                if len(mb) > 21:
                    rows.append(("槓桿", "融資餘額近 20 日增加" if mb.iloc[-1] / mb.iloc[-21] - 1 > 0 else "融資餘額近 20 日減少"))
        for d, s in rows:
            m, n, t, sh, lg, weak = stat.get((d, s), (None, None, None, None, None, False))
            out["rows"].append({"dim": d, "state": s, "mid": m, "days": n, "t": t, "short": sh, "long": lg, "weak": bool(weak)})
        out["weak"] = sum(r["weak"] for r in out["rows"])
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    return out


def rebound_window(B, weeks: int = 12) -> bool:
    """急跌狀態結束後 12 週內（v3.18 狀態定義；只用到今天以前的資料）。"""
    nav = _nav(B)
    r20 = nav / nav.shift(20) - 1
    dd = nav / nav.rolling(120, min_periods=60).max() - 1
    p = ((r20 <= PANIC["r20"]) | (dd <= -PANIC["dd120"])).fillna(False)
    return bool((not p.iloc[-1]) and p.tail(weeks * 5).any())


def recent_limit_downs(B, t: str, n: int = 5) -> int:
    """近 n 日收盤跌停次數。"""
    if t not in B.tickers:
        return 0
    j = list(B.tickers).index(t)
    return int(np.nansum(np.asarray(B.a["R"][-n:, j], float) <= LIMIT))
