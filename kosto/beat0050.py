"""v3.23 報酬與回撤都勝過 0050 的組合、費氏回撤、知名理論、短中長線重疊（docs/v3.23_protocol.md；規則先寫好再看結果）。

* D1／D2 診斷：短中長線名單重疊與動能占比；為什麼難贏 0050（市值集中、台積電）。
* B1a～B7 候選：權值股母池、0050 增強、台積電錨定、200 日線趨勢切換、核心衛星＋趨勢、0050 正 2＋200 日線；同一套門檻＋BH-FDR。
* E1 真實 ETF：國內股票型與槓桿 ETF 和 0050 同期間比較（不是回測）；真實 00631L＋200 日線。
* F1～F3 費氏回撤：反彈停在哪裡、貼文規則（0.382／0.5／0.618）、當選股因子。
* T1～T7 知名理論：神奇公式、葛拉漢、低波動、52 週新高、雙動能、萬聖節效應、200 日線。
全部只做研究與顯示；主模型不變。每日研究呼叫 summary(R, res)；換股日平移的資料包呼叫 phase_row(Rk, k)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run
from .metrics import TDAYS, perf
from .research import HKEY

FWD_START = "2026-11-25"                 # 前瞻（真正樣本外）起點
CASH_Y = 0.01                            # 現金年利率
SWITCH_COST = 0.005                      # 趨勢切換時被換掉的部位另計成本
TILT = 2.0                               # B2：市值 × e^(TILT × 名次)
TOP = 50
SMA = 200
LEV, LEV_COST = 2.0, 0.025               # B7：0050 正 2（模擬）＝2 × 0050 日報酬 − 年 2.5%（管理費約 1%＋期貨轉倉／資金成本約 1.5%）
MOM_FACTORS = {"t_rs120", "t_mom120x20", "t_high252", "t_mom20", "t_mom60", "t_ma_align", "t_above_ma60", "t_high20", "t_high60",
               "t_rmom240", "t_mom_secrel"}
CANDS = [("B1a", "權值股母池：市值前 50 大選 8 檔"), ("B1b", "權值股母池：市值前 100 大選 8 檔"),
         ("B2", "0050 增強：前 50 大市值權重 × 模型分數傾斜"), ("B3", "台積電錨定＋中線模型"),
         ("B4", "趨勢切換：200 日線之上模型、之下 0050"), ("B5", "核心衛星＋趨勢：0050 五成＋模型五成（之下模型改現金）"),
         ("B6", "0050 增強＋趨勢：200 日線之下留五成現金"), ("B7", "0050 正 2＋200 日線（之下全現金；Gayed & Bilello）")]
REFS = [("R0", "0050"), ("R1", "0050＋200 日線（之下留五成現金）"), ("R2", "中線模型"), ("R3", "前 50 大市值加權（0050 近似）"),
        ("R4", "0050 正 2 一直持有（模擬）")]
THEORIES = [("T1", "神奇公式（盈餘殖利率＋ROE）", {"f_ep": 0.5, "f_roe": 0.5}),
            ("T2", "葛拉漢價值（盈餘殖利率、淨值股價比、殖利率）", {"f_ep": 1 / 3, "f_bp": 1 / 3, "f_dy": 1 / 3}),
            ("T3", "低波動異常", {"t_lowvol": 1.0}),
            ("T4", "52 週新高（George & Hwang）", {"t_high252": 1.0}),
            ("T5", "雙動能（0050 近 12 個月 > 現金才持有）", None),
            ("T6", "萬聖節效應（11～4 月持有、5～10 月現金）", None),
            ("T7", "200 日均線（Faber；之下留五成現金）", None)]
FIB = (0.236, 0.382, 0.5, 0.618, 0.786)
GRID = (0.236, 0.30, 0.382, 0.44, 0.5, 0.55, 0.618, 0.70, 0.786)
HALF_W = 0.025


# ---------------------------------------------------------------------------------------------- 共用
def _days(B) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(B.days)


def _R(B) -> np.ndarray:
    R = np.asarray(B.a["R"], float)
    return np.where(np.isfinite(R) & (np.abs(R) < 0.5), R, 0.0)


def _bench(B) -> pd.Series:
    return pd.Series(np.nan_to_num(np.asarray(B.a["BENCH"], float)), index=_days(B))


def _costs(cfg) -> tuple[float, float]:
    c = cfg["costs"]
    return c["fee"] * c["fee_discount"] + c["slippage"], c["fee"] * c["fee_discount"] + c["tax"] + c["slippage"]


def _top(B, i: int, n: int = TOP) -> np.ndarray:
    ri = B.meta["rules"].index("MCAP")
    rk = np.asarray(B.a["RANKS"][ri][i])
    tr = np.asarray(B.a["TRADE"][i], bool)
    return np.where((rk >= 1) & (rk <= n) & tr)[0]


def fixed_w(B, d: dict) -> np.ndarray:
    names, dirs = list(B.factors), list(B.meta["directions"])
    w = np.zeros(len(names), np.float32)
    for f, x in d.items():
        if f in names:
            w[names.index(f)] = x * (dirs[names.index(f)] or 1)
    return w


def drift_returns(R: np.ndarray, starts: list, weights: list, D: int, cost_fn=None) -> np.ndarray:
    """權重在 starts[k] 當天收盤後生效（從 starts[k]+1 起算報酬），持有到下一次；權重會隨漲跌漂移。
    weights[k]＝{j: w}（j=-1 為外部報酬序列，見呼叫端）；回傳每日報酬（生效前為 0）。cost_fn(old, new) → 成本比例。"""
    out = np.zeros(D)
    old = {}
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else D - 1
        if s + 1 > e:
            continue
        w = weights[k]
        if cost_fn is not None:
            out[s + 1] -= cost_fn(old, w)
        js = np.array([j for j in w], int)
        if not len(js):
            old = {}
            continue
        v0 = np.array([w[j] for j in js])
        g = np.cumprod(1 + R[s + 1: e + 1][:, js], axis=0)          # 每檔累積
        val = g * v0
        tot = val.sum(1)
        prev = np.r_[v0.sum(), tot[:-1]]
        out[s + 1: e + 1] += np.where(prev > 0, tot / prev - 1, 0.0)
        last = val[-1] / tot[-1] if tot[-1] > 0 else v0
        old = dict(zip(js.tolist(), last.tolist()))
    return out


def _trade_cost(buy_c, sell_c):
    def fn(old, new):
        keys = set(old) | set(new)
        return sum(max(0.0, new.get(j, 0) - old.get(j, 0)) * buy_c + max(0.0, old.get(j, 0) - new.get(j, 0)) * sell_c for j in keys)
    return fn


def trend_state(B, sma: int = SMA) -> pd.Series:
    """每個交易日：0050（含息淨值）收盤是否在 sma 日均線之上（只用當天以前的資料）。"""
    b = (1 + _bench(B)).cumprod()
    m = b.rolling(sma, min_periods=sma // 2).mean()
    return (b > m).where(m.notna(), True)


def signal_states(B, sma: int = SMA) -> pd.Series:
    """換股日判斷、隔一個交易日生效（換股日收盤看到 → 下一個交易日收盤換倉 → 再下一天起算報酬），其餘日子沿用。"""
    st = trend_state(B, sma)
    days = _days(B)
    out = pd.Series(np.nan, index=days)
    for p in np.asarray(B.sig_pos, int):
        if p + 2 < len(days):
            out.iloc[p + 2] = float(st.iloc[p])
    return out.ffill().fillna(1.0).astype(bool)


def mix_returns(parts: list, state: pd.Series | None = None, switch_cost: float = SWITCH_COST) -> pd.Series:
    """parts＝[(weight_up, weight_down, returns Series)]；state True＝之上。換狀態那天，被換掉的比例扣 switch_cost。"""
    idx = parts[0][2].index
    s = (state.reindex(idx).fillna(True) if state is not None else pd.Series(True, index=idx)).astype(bool)
    r = sum(np.where(s, wu, wd) * x.reindex(idx).fillna(0).to_numpy() for wu, wd, x in parts)
    r = pd.Series(r, index=idx)
    if state is not None:
        chg = s.ne(s.shift(1)) & s.shift(1).notna()
        moved = sum(abs(wu - wd) for wu, wd, _ in parts) / 2
        r[chg] -= moved * switch_cost
    return r


def nw_t(x: np.ndarray, lag: int = 20) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def boot_p(x: np.ndarray, block: int = 20, reps: int = 1000, seed: int = 7) -> float:
    """區塊自助法：日超額平均 > 0 的機率。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block * 3:
        return np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    return float((x[idx].mean(1) > 0).mean())


def bh_q(p: list) -> list:
    p = np.asarray([np.nan if v is None else v for v in p], float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if ok.sum():
        pv = p[ok]
        m = len(pv)
        order = np.argsort(pv)
        qq = pv[order] * m / np.arange(1, m + 1)
        qq = np.minimum.accumulate(qq[::-1])[::-1]
        tmp = np.empty(m)
        tmp[order] = np.minimum(qq, 1.0)
        q[ok] = tmp
    return q.tolist()


def _norm_sf(t: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 - erf(t / sqrt(2))) if np.isfinite(t) else np.nan


def _seg(nav: pd.Series, bnav: pd.Series, a=None, b=None) -> dict:
    x = nav if a is None else nav[nav.index >= a]
    x = x if b is None else x[x.index < b]
    if len(x) < 21:
        return {}
    bb = bnav.reindex(x.index)
    p = perf(x / x.iloc[0], bb / bb.iloc[0])
    pb = perf(bb / bb.iloc[0])
    return {"cagr": p.get("cagr"), "mdd": p.get("mdd"), "sharpe": p.get("sharpe"), "excess": p.get("excess_cagr"),
            "b_cagr": pb.get("cagr"), "b_mdd": pb.get("mdd"), "days": len(x)}


def evaluate_series(r: pd.Series, rb: pd.Series, holdout, fwd_start=FWD_START) -> dict:
    """日報酬 r（候選）與 rb（0050）→ 各段績效、逐年、自助法、NW t。"""
    nav, bnav = (1 + r).cumprod(), (1 + rb.reindex(r.index).fillna(0)).cumprod()
    out = {"full": _seg(nav, bnav), "dev": _seg(nav, bnav, b=holdout), "recent": _seg(nav, bnav, a=holdout),
           "fwd": _seg(nav, bnav, a=pd.Timestamp(fwd_start))}
    yr = []
    for y, g in r.groupby(r.index.year):
        gb = rb.reindex(g.index).fillna(0)
        yr.append((int(y), float((1 + g).prod() - 1), float((1 + gb).prod() - 1)))
    out["years"] = yr
    ex = (r - rb.reindex(r.index).fillna(0)).to_numpy()
    out["t_nw"] = nw_t(ex)
    out["p_boot"] = boot_p(ex)
    out["p_one"] = _norm_sf(out["t_nw"])
    return out


def gates(ev: dict, phases: list | None = None) -> dict:
    f, d, c = ev.get("full") or {}, ev.get("dev") or {}, ev.get("recent") or {}
    g = {}
    g["g1"] = bool(f and f["cagr"] - f["b_cagr"] >= 0.03)
    g["g2"] = bool(f and f["mdd"] - f["b_mdd"] >= 0.01)
    g["g3"] = bool(d and c and d["cagr"] - d["b_cagr"] >= -0.03 and c["cagr"] - c["b_cagr"] >= -0.03
                   and d["mdd"] - d["b_mdd"] >= -0.02 and c["mdd"] - c["b_mdd"] >= -0.02)
    yrs = ev.get("years") or []
    g["g4"] = bool(yrs and sum(a > b for _, a, b in yrs) >= 2 / 3 * len(yrs))
    g["g5"] = bool(np.isfinite(ev.get("p_boot", np.nan)) and ev["p_boot"] >= 0.90)
    if phases:
        ok = sum(bool(x.get("g1") and x.get("g2")) for x in phases)
        g["g6"] = ok >= min(4, len(phases))
        g["g6_n"] = (ok, len(phases))
    else:
        g["g6"] = None
    return g


# ---------------------------------------------------------------------------------------------- D1／D2 診斷
def overlap(res: dict) -> dict:
    H = res.get("horizons") or {}
    pk = {h: {int(r["i"]): set(r["picks"] or []) for _, r in hb["tranches"].iterrows()} for h, hb in H.items()
          if hb.get("tranches") is not None and len(hb["tranches"])}
    out = {"pairs": []}
    hold = pd.Timestamp(res.get("protocol", {}).get("holdout_start") or "2100-01-01")
    for a, b in (("short", "mid"), ("mid", "long"), ("short", "long")):
        if a not in pk or b not in pk:
            continue
        common = sorted(set(pk[a]) & set(pk[b]))
        ov = [len(pk[a][i] & pk[b][i]) / max(len(pk[a][i]), len(pk[b][i]), 1) for i in common if pk[a][i] and pk[b][i]]
        dates = {int(r["i"]): r["date"] for _, r in H[a]["tranches"].iterrows()}
        rec = [len(pk[a][i] & pk[b][i]) / max(len(pk[a][i]), len(pk[b][i]), 1) for i in common
               if pk[a][i] and pk[b][i] and pd.Timestamp(dates[i]) >= hold]
        now = [len(pk[a][i] & pk[b][i]) / max(len(pk[a][i]), len(pk[b][i]), 1) for i in common[-1:] if pk[a][i] and pk[b][i]]
        out["pairs"].append({"a": a, "b": b, "avg": float(np.mean(ov)) if ov else np.nan, "recent": float(np.mean(rec)) if rec else np.nan,
                             "now": now[0] if now else np.nan, "n": len(ov)})
    mom = {}
    for h, hb in H.items():
        fh = hb.get("fit_history") or []
        w = (fh[-1] or {}).get("weights") if fh else None
        if w:
            tot = sum(abs(x) for x in w.values())
            mom[h] = {"momentum": sum(abs(x) for f, x in w.items() if f in MOM_FACTORS) / tot if tot else np.nan,
                      "top": sorted(((f, x / tot) for f, x in w.items()), key=lambda z: -abs(z[1]))[:5]}
    out["momentum"] = mom
    return out


def replica(B, start_i: int, n: int = TOP, step: int = 1) -> tuple[pd.Series, list]:
    """前 n 大市值加權（每週依市值重設）；回傳日報酬與每次的台積電權重。"""
    R, D = _R(B), len(B.days)
    sp = np.asarray(B.sig_pos, int)
    tk = list(B.tickers)
    jt = tk.index("2330") if "2330" in tk else -1
    starts, ws, tw = [], [], []
    for i in range(start_i, len(sp), step):
        mem = _top(B, i, n)
        m = np.nan_to_num(np.asarray(B.a["MCAP"][i, mem], float))
        if not len(mem) or m.sum() <= 0:
            continue
        w = m / m.sum()
        starts.append(int(sp[i]) + 1)            # 換股日收盤看到 → 隔天收盤換 → 再隔天起算（與 signal_states 一致）
        ws.append(dict(zip(mem.tolist(), w.tolist())))
        tw.append((str(B.sig_dates[i].date()), float(w[list(mem).index(jt)]) if jt in mem else 0.0))
    r = drift_returns(R, starts, ws, D)
    return pd.Series(r, index=_days(B)), tw


def concentration(B, res: dict, start_i: int) -> dict:
    days = _days(B)
    rb = _bench(B)
    rep, tw = replica(B, start_i)
    R = _R(B)
    tk = list(B.tickers)
    sp = np.asarray(B.sig_pos, int)
    s0 = int(sp[start_i]) + 2
    ew = drift_returns(R, [int(sp[i]) + 1 for i in range(start_i, len(sp))],
                       [{j: 1.0 / max(len(_top(B, i)), 1) for j in _top(B, i)} for i in range(start_i, len(sp))], len(days))
    ew = pd.Series(ew, index=days)
    tsmc = pd.Series(R[:, tk.index("2330")], index=days) if "2330" in tk else pd.Series(0.0, index=days)
    nav = (res.get("horizons", {}).get("mid") or {}).get("nav")
    cols = {}
    if nav is not None and len(nav):
        for c in ("策略", "母池等權"):
            if c in nav:
                cols[c] = nav[c].pct_change().reindex(days).fillna(0)
    rows = []
    for y in sorted(set(days[s0:].year)):
        m = (days.year == y) & (np.arange(len(days)) >= s0)
        f = lambda x: float((1 + x[m]).prod() - 1)     # noqa: E731
        tw_y = [w for d, w in tw if d.startswith(str(y))]
        rows.append({"year": int(y), "0050": f(rb), "tsmc": f(tsmc), "rep50": f(rep), "ew50": f(ew),
                     "pool_ew": f(cols["母池等權"]) if "母池等權" in cols else np.nan,
                     "model": f(cols["策略"]) if "策略" in cols else np.nan,
                     "tsmc_w": float(np.mean(tw_y)) if tw_y else np.nan})
    for r_ in rows:
        r_["tsmc_share"] = r_["tsmc_w"] * r_["tsmc"] / r_["0050"] if r_["0050"] and abs(r_["0050"]) > 0.02 else np.nan
    a, b = rep[s0:], rb[s0:]
    return {"years": rows, "rep_corr": float(np.corrcoef(a, b)[0, 1]) if len(a) > 30 else np.nan,
            "rep_te": float((a - b).std() * np.sqrt(TDAYS)), "tsmc_w_now": tw[-1][1] if tw else np.nan}


# ---------------------------------------------------------------------------------------------- 候選
def _mid_model(R, res):
    hb = (res.get("horizons") or {}).get("mid") or {}
    cp = hb.get("chosen_pool") or {}
    rule, size = cp.get("rule", "BLEND"), int(cp.get("size", 500))
    v = R.model_variant("mid", exposure=(R.cfg["horizons"]["mid"].get("model") or {}).get("exposure", "none"))
    return rule, size, v


def tilt_weights(base: np.ndarray, score: np.ndarray, anchor: np.ndarray, tilt: float = TILT) -> np.ndarray:
    """市值權重 base 依分數名次（置中 −0.5～+0.5）乘 e^(tilt × 名次)；anchor（台積電）維持原權重，其餘重新歸一到 1 − anchor。"""
    base = np.asarray(base, float)
    rk = (pd.Series(np.asarray(score, float)).rank().to_numpy() - 0.5) / len(base) - 0.5
    t = base * np.exp(tilt * rk)
    other = ~np.asarray(anchor, bool)
    keep = base[~other].sum()
    if not other.any() or t[other].sum() <= 0:
        return base
    return np.where(other, t / t[other].sum() * (1 - keep), base)


def enhanced(B, R, W: list, start_i: int, every: int = 4, tilt: float = TILT) -> tuple[pd.Series, dict]:
    """B2：前 50 大市值權重，台積電不動；其餘 × e^(tilt × 模型分數名次)。W＝中線模型每個換股日的因子權重。"""
    Rr, D = _R(B), len(B.days)
    sp = np.asarray(B.sig_pos, int)
    view = R.view("MCAP", TOP)
    tk = list(B.tickers)
    jt = tk.index("2330") if "2330" in tk else -1
    buy_c, sell_c = _costs(R.cfg)
    starts, ws, last = [], [], {}
    for i in range(start_i, len(sp), every):
        mem = view.members[i]
        if len(mem) < 5:
            continue
        m = np.nan_to_num(np.asarray(B.a["MCAP"][i, mem], float))
        if m.sum() <= 0:
            continue
        base = m / m.sum()
        w = tilt_weights(base, view.Z[i] @ W[i], np.array([j == jt for j in mem]), tilt) if W[i] is not None else base
        starts.append(int(sp[i]) + 1)
        ws.append(dict(zip(mem.tolist(), w.tolist())))
        last = {"date": str(B.sig_dates[i].date()), "w": sorted(((tk[j], float(x), float(b)) for j, x, b in zip(mem, w, base)),
                                                                key=lambda z: -z[1])}
    r = drift_returns(Rr, starts, ws, D, _trade_cost(buy_c, sell_c))
    return pd.Series(r, index=_days(B)), last


def candidates(B, R, res: dict) -> dict:
    """B1a～B7 與對照的日報酬（從中線模型的第一天起）。"""
    days = _days(B)
    rb = _bench(B)
    rule, size, v = _mid_model(R, res)
    out, cur = {}, {}
    ev = R.evaluate(rule, size, "mid", v, keep=True)
    mid_nav = ev["_res"].nav
    sp = np.asarray(B.sig_pos, int)
    s0 = int(sp[R.first_oos]) + 1
    idx = days[s0:]
    r_mid = mid_nav.pct_change().reindex(idx).fillna(0)
    out["R2"] = r_mid
    for key, n in (("B1a", 50), ("B1b", 100)):
        e2 = R.evaluate("MCAP", n, "mid", v, keep=True)
        out[key] = e2["_res"].nav.pct_change().reindex(idx).fillna(0)
        tr = e2["_res"].tranches
        cur[key] = list(tr["picks"].iloc[-1]) if tr is not None and len(tr) else []
    W, _ = R.schedule(rule, size, "mid", v)
    r_b2, cur["B2"] = enhanced(B, R, W, R.first_oos)
    out["B2"] = r_b2.reindex(idx).fillna(0)
    rep, tw = replica(B, R.first_oos)
    out["R3"] = rep.reindex(idx).fillna(0)
    tk = list(B.tickers)
    Rr = _R(B)
    r_ts = pd.Series(Rr[:, tk.index("2330")], index=days).reindex(idx).fillna(0) if "2330" in tk else pd.Series(0.0, index=idx)
    wts = pd.Series(np.nan, index=days)
    for d, w in tw:
        p = days.searchsorted(pd.Timestamp(d))
        if p + 2 < len(days):
            wts.iloc[p + 2] = w
    wts = wts.ffill().reindex(idx).fillna(0)
    buy_c, sell_c = _costs(R.cfg)
    r_b3 = wts * r_ts + (1 - wts) * r_mid - wts.diff().abs().fillna(0) * (buy_c + sell_c) / 2
    out["B3"] = r_b3
    tr = ev["_res"].tranches
    cur["B3"] = {"tsmc_w": float(wts.iloc[-1]) if len(wts) else np.nan,
                 "model": list(tr["picks"].iloc[-1]) if tr is not None and len(tr) else []}
    st = signal_states(B).reindex(idx)
    cash = pd.Series(CASH_Y / TDAYS, index=idx)
    r0 = rb.reindex(idx).fillna(0)
    out["R0"] = r0
    out["B4"] = mix_returns([(1.0, 0.0, r_mid), (0.0, 1.0, r0)], st)
    out["B5"] = mix_returns([(0.5, 0.5, r0), (0.5, 0.0, r_mid), (0.0, 0.5, cash)], st)
    out["B6"] = mix_returns([(1.0, 0.5, out["B2"]), (0.0, 0.5, cash)], st)
    out["R1"] = mix_returns([(1.0, 0.5, r0), (0.0, 0.5, cash)], st)
    r_lev = LEV * r0 - LEV_COST / TDAYS
    out["R4"] = r_lev
    out["B7"] = mix_returns([(1.0, 0.0, r_lev), (0.0, 1.0, cash)], st)
    ts = trend_state(B)
    b = (1 + rb).cumprod()
    cur["trend"] = {"up": bool(ts.iloc[-1]), "nav": float(b.iloc[-1]), "sma": float(b.rolling(SMA).mean().iloc[-1]),
                    "gap": float(b.iloc[-1] / b.rolling(SMA).mean().iloc[-1] - 1), "date": str(days[-1].date())}
    return {"returns": out, "current": cur, "holdout": B.sig_dates[min(R.holdout, R.S - 1)], "start": str(idx[0].date())}


def theories(B, R, res: dict) -> dict:
    days = _days(B)
    rb = _bench(B)
    sp = np.asarray(B.sig_pos, int)
    s0 = int(sp[R.first_oos]) + 1
    idx = days[s0:]
    r0 = rb.reindex(idx).fillna(0)
    cash = pd.Series(CASH_Y / TDAYS, index=idx)
    out = {}
    v = R.model_variant("mid")
    spec = R.spec("mid", v)
    view = R.view("MCAP", 300)
    for key, _, d in THEORIES:
        if d:
            w = fixed_w(B, d)
            if not np.abs(w).sum():
                continue
            W = [w if i >= R.first_oos else None for i in range(R.S)]
            nav = run(B, view, W, spec, start_i=R.first_oos).nav
            out[key] = nav.pct_change().reindex(idx).fillna(0)
    b = (1 + rb).cumprod()
    m12 = (b / b.shift(250) - 1 > CASH_Y)
    st = pd.Series(np.nan, index=days)
    for p in sp:
        if p + 2 < len(days):
            st.iloc[p + 2] = float(m12.iloc[p])
    st = st.ffill().fillna(1.0).astype(bool).reindex(idx)
    out["T5"] = mix_returns([(1.0, 0.0, r0), (0.0, 1.0, cash)], st)
    hal = pd.Series(np.isin(idx.month, [11, 12, 1, 2, 3, 4]), index=idx)
    out["T6"] = mix_returns([(1.0, 0.0, r0), (0.0, 1.0, cash)], hal)
    out["T7"] = mix_returns([(1.0, 0.5, r0), (0.0, 0.5, cash)], signal_states(B).reindex(idx))
    return out


# ---------------------------------------------------------------------------------------------- E1 真實 ETF（不是回測）
def etf_compare(B, min_days: int = 480, sma: int = SMA) -> dict:
    """研究資料包裡的國內股票型與槓桿 ETF（含息淨值，近約 5 年）：和 0050 在「同一段期間」比年化與最大回撤；
    另算真實 00631L（0050 正 2）＋200 日線（看 0050 的 200 日線，換股日判斷、隔兩天生效，之下放現金）。"""
    if "ETF_TR" not in B.a or not B.meta.get("etf_tickers"):
        return {"error": "研究資料包沒有 ETF 淨值"}
    tr = pd.DataFrame(np.asarray(B.a["ETF_TR"], float), index=pd.DatetimeIndex(B.meta["etf_days"]), columns=B.meta["etf_tickers"])
    if "0050" not in tr:
        return {"error": "ETF 資料沒有 0050"}
    info = {r["ticker"]: r for r in (B.meta.get("etf_table") or [])}
    base = tr["0050"]
    rows = []
    for t in tr.columns:
        tp = (info.get(t) or {}).get("type", "")
        if t == "0050" or tp not in ("國內股票", "槓桿反向") or (tp == "槓桿反向" and not t.endswith("L")):
            continue
        x = tr[t].dropna()
        if len(x) < min_days:
            continue
        b = base.reindex(x.index).ffill().dropna()
        x = x.reindex(b.index)
        p, pb = perf(x / x.iloc[0]), perf(b / b.iloc[0])
        if not p or not pb:
            continue
        yr = [(int(y), float(g.iloc[-1] / g.iloc[0] - 1), float(b[g.index].iloc[-1] / b[g.index].iloc[0] - 1))
              for y, g in x.groupby(x.index.year) if len(g) > 20]
        rows.append({"ticker": t, "name": (info.get(t) or {}).get("name", ""), "type": tp, "from": str(x.index[0].date()),
                     "cagr": p["cagr"], "mdd": p["mdd"], "b_cagr": pb["cagr"], "b_mdd": pb["mdd"],
                     "both": bool(p["cagr"] > pb["cagr"] and p["mdd"] > pb["mdd"]), "years": yr})
    rows.sort(key=lambda r: -(r["cagr"] - r["b_cagr"]))
    out = {"rows": rows, "asof": str(tr.index[-1].date()), "n": len(rows)}
    if "00631L" in tr:
        lv = tr["00631L"].dropna()
        b = base.reindex(lv.index).ffill()
        m = b.rolling(sma, min_periods=sma // 2).mean()
        up = (b > m).where(m.notna(), True).astype(float)
        st = up.shift(2).fillna(1.0).astype(bool)                          # 隔兩天生效（與換股日規則同樣保守）
        rl, rb = lv.pct_change().fillna(0), b.pct_change().fillna(0)
        cash = pd.Series(CASH_Y / TDAYS, index=lv.index)
        r = mix_returns([(1.0, 0.0, rl), (0.0, 1.0, cash)], st)
        nav = (1 + r).cumprod()
        p, pb, pl = perf(nav), perf((1 + rb).cumprod()), perf(lv / lv.iloc[0])
        sim = LEV * rb - LEV_COST / TDAYS
        out["lev"] = {"from": str(lv.index[0].date()), "cagr": p.get("cagr"), "mdd": p.get("mdd"), "b_cagr": pb.get("cagr"), "b_mdd": pb.get("mdd"),
                      "hold_cagr": pl.get("cagr"), "hold_mdd": pl.get("mdd"), "switches": int(st.ne(st.shift(1)).sum() - 1),
                      "sim_corr": float(np.corrcoef(rl.iloc[1:], sim.iloc[1:])[0, 1]) if len(rl) > 30 else np.nan,
                      "sim_gap": float((1 + rl).prod() ** (TDAYS / len(rl)) - (1 + sim).prod() ** (TDAYS / len(sim)))}
    return out


# ---------------------------------------------------------------------------------------------- 費氏回撤
def fib_scan(p: np.ndarray, theta: float, horizon: int = 250, grid=GRID) -> tuple[list, list]:
    """跌幅 ≥ theta 的波段：反彈第一次碰到各位置的事件（結果：先回前高 up／先破底 down／horizon 日內都沒有 range），
    以及每次「反彈後又破底」的反彈最高位置（費氏位置有沒有比較常當壓力）。只用當時以前的資料定義 H、L。"""
    n = len(p)
    ev, peaks = [], []
    H = L = p[0]
    active, touched, xmax, cur = False, set(), 0.0, []

    def close(t, how):
        nonlocal cur
        for e in cur:
            e["out"] = how if t - e["t"] <= horizon else "range"
        cur = []
    for t in range(1, n):
        x_ = p[t]
        if not np.isfinite(x_) or x_ <= 0:
            continue
        if x_ >= H:
            if active:
                close(t, "up")
            H = L = x_
            active, touched, xmax = False, set(), 0.0
            continue
        if x_ < L:
            if active and xmax >= 0.1:
                peaks.append(xmax)
            if active:
                close(t, "down")
            L = x_
            touched, xmax = set(), 0.0
            active = (H - L) / H >= theta
            continue
        if active:
            r = (x_ - L) / (H - L)
            xmax = max(xmax, r)
            for g in grid:
                if g not in touched and r >= g:
                    touched.add(g)
                    e = {"t": t, "level": g, "out": None,
                         "f20": p[t + 20] / x_ - 1 if t + 20 < n else np.nan, "f60": p[t + 60] / x_ - 1 if t + 60 < n else np.nan}
                    ev.append(e)
                    cur.append(e)
    for e in cur:
        e["out"] = "range" if n - 1 - e["t"] > horizon else None
    return [e for e in ev if e["out"]], peaks


def fib_table(events: list, grid=GRID) -> list:
    df = pd.DataFrame(events)
    rows = []
    for g in grid:
        x = df[df["level"] == g] if len(df) else df
        if not len(x):
            rows.append({"level": g, "n": 0})
            continue
        rows.append({"level": g, "n": int(len(x)), "rw": g,                     # 隨機漫步（沒有任何規律）下「先回前高」的機率≈位置本身
                     "up": float((x["out"] == "up").mean()), "down": float((x["out"] == "down").mean()),
                     "range": float((x["out"] == "range").mean()), "f20": float(x["f20"].mean()), "f60": float(x["f60"].mean())})
    return rows


def fib_cluster(peaks: list, gap: float = 0.06) -> dict:
    """反彈最高位置在費氏位置附近（±0.025）的次數，和左右兩個等寬對照窗（±0.06 外）平均比較：
    密度平滑（沒有費氏效果）時，費氏窗 ≈ 左右平均。z＝(費氏 − 期望)／√(費氏 + 左右和／4)。"""
    x = np.asarray(peaks, float)
    x = x[(x > 0.1) & (x < 0.95)]
    cnt = lambda c: int(((x >= c - HALF_W) & (x < c + HALF_W)).sum())      # noqa: E731
    rows, F, E, V = [], 0, 0.0, 0.0
    for f in FIB:
        a, b, n = cnt(f - gap), cnt(f + gap), cnt(f)
        rows.append({"level": f, "n": n, "left": a, "right": b, "ratio": n / ((a + b) / 2) if a + b else np.nan})
        F += n
        E += (a + b) / 2
        V += n + (a + b) / 4
    z = (F - E) / np.sqrt(V) if V > 0 else np.nan
    hist = np.histogram(x, bins=np.arange(0.1, 0.951, 0.05))[0].tolist() if len(x) else []
    return {"n": int(len(x)), "fib": int(F), "expected": float(E), "ratio": F / E if E else np.nan,
            "z": float(z) if np.isfinite(z) else np.nan, "p": _norm_sf(z) if np.isfinite(z) else np.nan, "levels": rows, "hist": hist}


def fib_factor(B, R, res: dict, lookback: int = 250, min_drop: float = 0.2) -> dict:
    """F3：目前位於近一年最大跌幅波段的幾成（反彈比例）→ 中線 IC、扣掉一年新高與相對強度後的部分 IC。"""
    Rr = _R(B)
    P = np.cumprod(1 + Rr, axis=0)
    sp = np.asarray(B.sig_pos, int)
    fwd = np.asarray(B.a["FWD"][HKEY["mid"]], float)
    names = list(B.factors)
    fh, fr = (names.index("t_high252") if "t_high252" in names else None), (names.index("t_rs120") if "t_rs120" in names else None)
    rule, size, _ = _mid_model(R, res)
    view = R.view(rule, size)
    ics, pics, cors = [], [], []
    D = len(B.days)
    for i in range(R.first_oos, len(sp)):
        t = int(sp[i])
        if t < lookback or t + 21 >= D:
            continue
        mem = view.members[i]
        if len(mem) < 30:
            continue
        win = P[t - lookback: t + 1][:, mem]
        a = win.argmax(0)
        H = win[a, np.arange(len(mem))]
        after = np.where(np.arange(win.shape[0])[:, None] >= a[None, :], win, np.inf)
        L = after.min(0)
        cur = win[-1]
        x = np.where((H - L) / H >= min_drop, (cur - L) / np.where(H - L > 0, H - L, np.nan), np.nan)
        y = fwd[i, mem]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        rx = pd.Series(x[ok]).rank().to_numpy()
        ry = pd.Series(y[ok]).rank().to_numpy()
        ics.append(np.corrcoef(rx, ry)[0, 1])
        if fh is not None and fr is not None:
            Z = np.asarray(B.a["U"][i][mem][ok][:, [fh, fr]], float)
            Zr = np.column_stack([pd.Series(Z[:, 0]).rank(), pd.Series(Z[:, 1]).rank(), np.ones(ok.sum())])
            beta, *_ = np.linalg.lstsq(Zr, rx, rcond=None)
            e = rx - Zr @ beta
            if e.std() > 0:
                pics.append(np.corrcoef(e, ry)[0, 1])
            cors.append(np.corrcoef(rx, Zr[:, 0])[0, 1])

    def tstat(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return float(v.mean() / v.std(ddof=1) * np.sqrt(len(v) / 4)) if len(v) > 8 and v.std() > 0 else np.nan
    return {"n": len(ics), "ic": float(np.nanmean(ics)) if ics else np.nan, "t": tstat(ics),
            "pic": float(np.nanmean(pics)) if pics else np.nan, "pt": tstat(pics), "corr_high252": float(np.nanmean(cors)) if cors else np.nan}


def fib_study(B, R, res: dict) -> dict:
    out = {}
    b = (1 + _bench(B)).cumprod().to_numpy()
    ev, pk = fib_scan(b, 0.10)
    out["bench"] = {"table": fib_table(ev), "cluster": fib_cluster(pk), "legs": len(pk)}
    Rr = _R(B)
    P = np.cumprod(1 + Rr, axis=0)
    cap = np.nan_to_num(np.asarray(B.a["CAP"][-1], float))
    cols = np.argsort(-cap)[:300]
    evs, pks = [], []
    for j in cols:
        e, k = fib_scan(P[:, j], 0.20)
        evs += e
        pks += k
    out["stocks"] = {"table": fib_table(evs), "cluster": fib_cluster(pks), "n_stocks": int(len(cols)), "events": len(evs)}
    base = []
    for j in cols:
        x = P[:, j]
        base.append(np.nanmean(x[20:] / x[:-20] - 1))
    out["stocks"]["base_f20"] = float(np.nanmean(base))
    try:
        out["factor"] = fib_factor(B, R, res)
    except Exception as e:  # noqa: BLE001
        out["factor"] = {"error": str(e)}
    out["now"] = fib_now(b, pd.DatetimeIndex(B.days))
    return out


def fib_now(p: np.ndarray, days: pd.DatetimeIndex, theta: float = 0.10) -> dict:
    """目前 0050 最近一次跌 ≥ theta 的波段：高點、低點、現在的反彈比例與各費氏價位（以含息淨值計）。"""
    H = L = p[0]
    hd = ld = 0
    for t in range(1, len(p)):
        if p[t] >= H:
            H = L = p[t]
            hd = ld = t
        elif p[t] < L:
            L, ld = p[t], t
    if (H - L) / H < theta:
        return {"active": False, "high": float(H), "date_high": str(days[hd].date())}
    return {"active": True, "high": float(H), "low": float(L), "date_high": str(days[hd].date()), "date_low": str(days[ld].date()),
            "ratio": float((p[-1] - L) / (H - L)), "drop": float(L / H - 1)}


# ---------------------------------------------------------------------------------------------- 彙整
def _pack(rets: dict, rb: pd.Series, holdout, keys) -> list:
    rows = []
    for k, nm in keys:
        if k not in rets:
            continue
        ev = evaluate_series(rets[k], rb, holdout)
        rows.append({"key": k, "name": nm, **ev})
    return rows


def summary(R, res: dict) -> dict:
    """每日研究呼叫。各部分獨立，失敗只記錯誤。"""
    B = R.B
    out = {"asof": str(pd.Timestamp(B.days[-1]).date()), "fwd_start": FWD_START, "errors": []}
    try:
        out["overlap"] = overlap(res)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"重疊：{e}")
    try:
        out["concentration"] = concentration(B, res, R.first_oos)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"集中度：{e}")
    try:
        c = candidates(B, R, res)
        rb = c["returns"]["R0"]
        hold = c["holdout"]
        out["start"], out["holdout"] = c["start"], str(pd.Timestamp(hold).date())
        out["cands"] = _pack(c["returns"], rb, hold, CANDS)
        out["refs"] = _pack(c["returns"], rb, hold, REFS)
        qs = bh_q([x.get("p_one") for x in out["cands"]])
        for x, q in zip(out["cands"], qs):
            x["q"] = q
            x["gates"] = gates(x)
            x["gates"]["g7"] = bool(np.isfinite(q) and q <= 0.10)
        out["current"] = c["current"]
        navs = pd.DataFrame({k: (1 + v).cumprod() for k, v in c["returns"].items()})
        out["nav"] = navs.iloc[::5].astype(np.float32)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"候選：{e}")
    try:
        t = theories(B, R, res)
        rb = _bench(B).reindex(t["T7"].index).fillna(0)
        hold = B.sig_dates[min(R.holdout, R.S - 1)]
        out["theories"] = _pack(t, rb, hold, [(k, n) for k, n, _ in THEORIES])
        for x in out["theories"]:
            x["gates"] = gates(x)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"理論：{e}")
    try:
        out["etf"] = etf_compare(B)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"真實 ETF：{e}")
    try:
        out["fib"] = fib_study(B, R, res)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"費氏：{e}")
    return finalize(out)


def phase_row(Rk, res: dict, k: int) -> dict:
    """換股日平移 k 天的資料包：各候選整段年化與回撤（門檻 6 用）。"""
    c = candidates(Rk.B, Rk, res)
    rb = c["returns"]["R0"]
    row = {"phase": k}
    for key, _ in CANDS:
        if key in c["returns"]:
            s = evaluate_series(c["returns"][key], rb, c["holdout"])["full"]
            row[key] = {"cagr": s.get("cagr"), "mdd": s.get("mdd"), "b_cagr": s.get("b_cagr"), "b_mdd": s.get("b_mdd"),
                        "g1": bool(s and s["cagr"] - s["b_cagr"] >= 0.03), "g2": bool(s and s["mdd"] - s["b_mdd"] >= 0.01)}
    return row


def attach_phases(out: dict, rows: list) -> dict:
    """把平移結果併入門檻 6（第 0 種＝目前資料包）。"""
    if not out.get("cands"):
        return out
    out["phases"] = rows
    for x in out["cands"]:
        ph = [{"g1": x["gates"]["g1"], "g2": x["gates"]["g2"]}] + [r[x["key"]] for r in rows if x["key"] in r]
        g = gates(x, ph)
        g["g7"] = x["gates"].get("g7")
        x["gates"] = g
    return finalize(out)


def finalize(out: dict) -> dict:
    for x in out.get("cands") or []:
        g = x["gates"]
        core = [g.get(k) for k in ("g1", "g2", "g3", "g4", "g5", "g7")] + ([g["g6"]] if g.get("g6") is not None else [])
        x["pass"] = all(bool(v) for v in core)
        x["observe"] = (not x["pass"]) and all(bool(g.get(k)) for k in ("g1", "g2", "g3", "g4", "g5"))
    return out
