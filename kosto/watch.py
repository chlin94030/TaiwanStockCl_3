"""v3.12 觀察名單前瞻追蹤（docs/v3.12_protocol.md 第五節）。

歷次研究「差一點通過、列入觀察」的候選，以前只寫在文件裡，之後常常沒有再檢查（v3.12 回頭檢視時發現：v1.9 的「營收驚喜」
寫了「下一版第一優先」，到 v3.11 都沒有再測）。同一份歷史資料再測只是多一次運氣，真正的新證據是「之後的資料」。

做法：每日研究多算幾個影子模型——與正式模型相同的資料、母池、走勢前進權重，只多一個預先寫好的改動——
與現行模型比較 SINCE（v3.12 研究沒有看過的第一天）之後的表現。不影響選股、不會自動換模型；
前瞻期滿 NEED_DAYS 個交易日後，依預先寫好的規則標示「前瞻支持／前瞻不支持」，由下一版決定是否採用。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .backtest import run

SINCE = "2026-10-23"          # v3.12 研究資料到 2026-10-22；之後的資料才是新證據
NEED_DAYS = 120               # 約 6 個月（2027-05）才判定
PANIC = {"enabled": True, "factors": [], "r20": -0.08, "dd120": 0.12, "hold": 0}    # v2.2 急跌狀態定義

ITEMS = [
    {"key": "W1", "hz": ("long",), "kind": "return", "name": "長線加「營收驚喜」15%",
     "why": "v1.9 第一優先候選；v3.12 長線五種換股日全部進步（超額 +3.9、回撤淺 3.3），但中線拖累"},
    {"key": "W2", "hz": ("mid",), "kind": "return", "name": "中線加籌碼組合 15%",
     "why": "v2.3 D2；v3.12 中線 +1.9、回撤淺 6.6，但五種換股日只有 3 種為正"},
    {"key": "W3", "hz": ("short", "mid", "long"), "kind": "balanced", "name": "同產業最多 3 檔",
     "why": "v2.2 C3；v3.12 回撤淺 1～4，長線整段樣本外 −1.02（門檻 −1.0）"},
    {"key": "W4", "hz": ("mid",), "kind": "balanced", "name": "急跌時中線加券資比 15%",
     "why": "v2.3 S2：回撤淺 4.7、超額 −0.3；需要新的急跌事件"},
    {"key": "W5", "hz": ("short", "mid", "long"), "kind": "return", "name": "產業調整後動能（動能權重一半）",
     "why": "v3.15 N3（第二份評審策略 B）：歷史門檻全過（中線 +2.1、5/5 換股日、三期間回撤都較淺），多重檢定 q=0.31 未過"},
    {"key": "W6", "hz": ("short", "mid", "long"), "kind": "return", "name": "急跌時改用反彈候選選股",
     "why": "v3.17 C4：急跌換股日改選跌深、營收優於同業、投信沒追高的股票；全期中線 +2.0、短線 +6.1，但急跌期間只多 0.9pp（門檻 1pp）、回撤深 1.8"},
    {"key": "W7", "hz": ("short", "mid", "long"), "kind": "return", "name": "美股風險關閉時動能減半",
     "why": "v3.18 A3：前一晚費半在 50 日線之下且 VIX≥25 → 動能群權重減半；門檻 1～4 全過（中線 +1.7、短線 +3.8、5/5 換股日、弱勢狀態不變差），"
            "但效果集中在 2022 年，q 0.71 未過"},
    {"key": "W8", "since": "2026-11-25", "hz": ("short", "mid"), "kind": "return", "name": "輝達重大消息後押反轉 15%",
     "why": "v3.21 E1：輝達特有消息日（|輝達扣那指| ≥ 5%）後 10 個交易日內，加 15% 押「跟輝達連動高的股票會回吐消息反應」（利空後買、利多後賣）"},
    {"key": "W9", "since": "2026-11-25", "hz": ("short", "mid"), "kind": "return", "name": "輝達重大消息後押延續 15%",
     "why": "v3.21 E2：與 W8 相反方向（消息反應會延續），兩個一起測，由資料決定方向"},
    {"key": "W10", "since": "2026-11-25", "hz": ("short", "mid", "long"), "kind": "balanced", "name": "不新買輝達特有連動最高的 10%",
     "why": "v3.21 E3：降低單一公司消息（如黃仁勳去留）的集中風險；風險型"},
    {"key": "W11", "since": "2026-11-27", "hz": ("short", "mid", "long"), "kind": "balanced", "name": "不新買投信擁擠最高的 10%",
     "why": "v3.24：投信 60 日累計買超（占股本）最高 10%＝主動式 ETF／基金擁擠；大量買回時基金會賣這些股票"},
    {"key": "W12", "since": "2026-11-27", "hz": ("short", "mid", "long"), "kind": "balanced", "name": "急跌時才不新買投信擁擠最高的 10%",
     "why": "v3.24：同 W11，但只在急跌狀態（0050 近 20 日 ≤ −8% 或距 120 日高 ≥ 12%）避開"},
]
CROWD = "c_trust60"                                                    # v3.24 擁擠度：投信 60 日累計買超（占股本）
MOM_GROUP = ("t_rs120", "t_mom120x20", "t_high252", "t_mom60", "t_high20", "t_rmom240")    # v3.18 動能群
REBOUND = (("t_mom20", -1), ("t_high60", -1), ("f_rev_rel", 1), ("c_trust20", -1))
SPLIT = {"enabled": True, "to": "t_mom_secrel", "from": ["t_rs120", "t_mom120x20"], "share": 0.5}
KIND_ZH = {"return": "報酬型", "balanced": "平衡型"}
RULE = {"return": "累積報酬高於現行，且最大回撤不深於 2 個百分點",
        "balanced": "最大回撤淺 ≥ 2 個百分點，且累積報酬少不超過 2 個百分點"}


def _norm(w):
    return w / max(float(np.abs(w).sum()), 1e-12)


def _unit(n, j, d=1.0):
    e = np.zeros(n, np.float32)
    e[j] = d
    return e


def _dev_ic(R, rule, size, hz):
    dev = np.where((np.arange(R.S) >= R.first_oos) & (np.arange(R.S) < R.holdout))[0]
    return np.nanmean(R.ic(rule, size, hz)[dev], axis=0)


def variant(R, key: str, h: str, rule: str, size: int, v, W: list, spec):
    """回傳 (權重序列, 回測設定)；資料包缺少需要的因子時回傳 None。"""
    F = list(R.B.factors)
    n = len(F)
    if key == "W1":
        if "f_rev_surprise" not in F:
            return None
        ov = {f: float(s) for f, s in (R.cfg["research"].get("overlay") or {}).items() if f in F and s}
        tot = sum(ov.values())
        add = sum((_unit(n, F.index(f), s) for f, s in ov.items()), np.zeros(n, np.float32))
        js = F.index("f_rev_surprise")

        def f1(w):
            if w is None:
                return None
            raw = (_norm(w) - add) / max(1 - tot, 1e-9)           # 拿掉現行疊加，回到原模型
            return ((1 - tot - 0.15) * raw + add + 0.15 * _unit(n, js)).astype(np.float32)
        return [f1(w) for w in W], spec
    if key == "W2":
        fam = np.array(R.B.meta["families"])
        hz_of = R.B.meta["horizons"]
        inact = set(R.cfg["research"].get("inactive_factors") or [])
        mic = _dev_ic(R, rule, size, v.label_h or h)
        cs = [j for j, f in enumerate(F) if fam[j] == "C" and f not in inact and h in hz_of[j] and np.isfinite(mic[j])]
        top3 = sorted(cs, key=lambda j: -abs(mic[j]))[:3]
        if not top3:
            return None
        chips = sum((_unit(n, j, np.sign(mic[j]) / len(top3)) for j in top3), np.zeros(n, np.float32))
        return [None if w is None else (0.85 * _norm(w) + 0.15 * chips).astype(np.float32) for w in W], spec
    if key == "W3":
        return W, dataclasses.replace(spec, max_per_sector=3)
    if key == "W6":
        dirs = np.array(R.B.meta["directions"], float)
        dirs[dirs == 0] = 1
        if not all(f in F for f, _ in REBOUND):
            return None
        reb = np.zeros(n, np.float32)
        for f, sg in REBOUND:
            reb[F.index(f)] = sg * dirs[F.index(f)]
        reb /= np.abs(reb).sum()
        panic = R.panic_signals(PANIC)
        return [reb if (panic[i] and w is not None) else w for i, w in enumerate(W)], spec
    if key == "W7":
        Mi = [F.index(f) for f in MOM_GROUP if f in F]
        off = us_off_signals(R.B)
        if not Mi or off is None:
            return None

        def half(w):
            w = _norm(np.asarray(w, float)); w2 = w.copy(); w2[Mi] *= 0.5
            m = np.ones(n, bool); m[Mi] = False
            o = np.abs(w[m]).sum()
            if o > 0:
                w2[m] *= (1 - np.abs(w2[Mi]).sum()) / o
            return w2.astype(np.float32)
        return [half(w) if (w is not None and off[i]) else w for i, w in enumerate(W)], spec
    if key in ("W11", "W12"):
        nn = crowd_mask(R, rule, size)
        if nn is None:
            return None
        if key == "W12":
            panic = R.panic_signals(PANIC)
            nn = nn & np.asarray(panic, bool)[:, None]
        return W, dataclasses.replace(spec, no_new=nn)
    if key in ("W8", "W9", "W10"):
        arr = news_arrays(R.B)
        if arr is None:
            return None
        beta, sgn = arr
        view = R.view(rule, size)
        if key == "W10":
            nn = np.zeros(beta.shape, bool)
            for i, m in enumerate(view.members):
                b = beta[i, m]
                if np.isfinite(b).sum() >= 20:
                    nn[i, m[b >= np.nanquantile(b, 0.9)]] = True
            return W, dataclasses.replace(spec, no_new=nn)
        from .optimize import PoolView, pool_view
        d = -1.0 if key == "W8" else 1.0
        col = np.nan_to_num(d * sgn[:, None] * beta, nan=0.0).astype(np.float32)
        ri = R.B.meta["rules"].index(rule)
        rk = R.B.a["RANKS"][ri]
        vz = pool_view(col[:, :, None], (rk >= 1) & (rk <= size), R.B.a["TRADE"])
        v2 = PoolView(view.members, [np.c_[z, zz] for z, zz in zip(view.Z, vz.Z)])
        W2 = [None if w is None else (np.r_[0.85 * _norm(np.asarray(w, float)), 0.15] if sgn[i] != 0 else np.r_[np.asarray(w, float), 0.0]).astype(np.float32)
              for i, w in enumerate(W)]
        return W2, spec, v2
    if key == "W5":
        if "t_mom_secrel" not in F:
            return None
        return [R.split_momentum(w, SPLIT) for w in W], spec
    if key == "W4":
        if "c_short_ratio" not in F:
            return None
        js = F.index("c_short_ratio")
        mic = _dev_ic(R, rule, size, v.label_h or h)
        e = _unit(n, js, float(np.sign(mic[js]) or 1.0))
        panic = R.panic_signals(PANIC)
        return [None if w is None else ((0.85 * _norm(w) + 0.15 * e).astype(np.float32) if panic[i] else w)
                for i, w in enumerate(W)], spec
    raise KeyError(key)


def crowd_mask(R, rule: str, size: int, q: float = 0.9, factor: str = CROWD):
    """v3.24：每個換股日，母池內「投信 60 日累計買超（占股本）」最高 10% 的股票（只用當天以前的資料；缺值不算）。"""
    F = list(R.B.factors)
    if factor not in F:
        return None
    U = R.B.a["U"]
    j = F.index(factor)
    view = R.view(rule, size)
    nn = np.zeros((len(view.members), U.shape[1]), bool)
    for i, m in enumerate(view.members):
        u = np.asarray(U[i, m, j], float)
        ok = u > 0
        if ok.sum() >= 20:
            nn[i, m[ok & (u >= np.quantile(u[ok], q))]] = True
    return nn


_NEWS: dict = {}


def news_arrays(B, lookback: int = 250, window: int = 10, thr: float = 0.05):
    """v3.21：每個換股日每檔股票對「輝達特有報酬」的 β（前 250 日、只用當天以前）＋最近 10 個交易日最大的輝達特有消息方向（|e| ≥ 5%，否則 0）。"""
    key = (id(B), len(B.days), str(B.days[-1]))
    if key in _NEWS:
        return _NEWS[key]
    from .shock import nvda_news
    nd = nvda_news(B)
    if nd is None:
        return None
    e = nd["e"]
    Rr = pd.DataFrame(np.where(np.abs(np.asarray(B.a["R"], float)) < 0.5, np.asarray(B.a["R"], float), np.nan))
    ee = pd.Series(e.to_numpy())
    Rm = Rr.where(ee.notna(), axis=0)
    mp = int(lookback * 0.5)
    cov = Rm.mul(ee, axis=0).rolling(lookback, min_periods=mp).mean() - Rm.rolling(lookback, min_periods=mp).mean().mul(
        ee.rolling(lookback, min_periods=mp).mean(), axis=0)
    var = ee.rolling(lookback, min_periods=mp).var(ddof=0)
    sp = np.asarray(B.sig_pos, int)
    beta = cov.div(var, axis=0).to_numpy(np.float32)[sp]
    ev = ee.to_numpy()
    sgn = np.zeros(len(sp))
    for i, p in enumerate(sp):
        w = ev[max(0, p - window + 1): p + 1]
        if np.isfinite(w).any() and np.nanmax(np.abs(w)) >= thr:
            sgn[i] = np.sign(w[np.nanargmax(np.abs(w))])
    _NEWS.clear()
    _NEWS[key] = (beta, sgn)
    return _NEWS[key]


def us_off_signals(B, vix_min: float = 25.0, ma: int = 50):
    """v3.18 美股風險關閉（每個換股日）：前一晚費半收盤低於 50 日均線，且 VIX ≥ 25（WORLD＝美國日期 ≤ T−1）。"""
    keys = list(B.meta.get("world_keys") or [])
    if "WORLD" not in B.a or "SOX" not in keys or "VIX" not in keys:
        return None
    Wd = np.asarray(B.a["WORLD"], float)
    sox = pd.Series(Wd[:, keys.index("SOX")]).ffill()
    vix = pd.Series(Wd[:, keys.index("VIX")]).ffill()
    off = ((sox < sox.rolling(ma, min_periods=40).mean()) & (vix >= vix_min)).to_numpy()
    return off[np.asarray(B.sig_pos, int)]


def forward(nav: pd.Series, since: str = SINCE) -> dict:
    """since 起（含）的累積報酬與最大回撤；起點＝since 前一個交易日的淨值。"""
    nav = pd.Series(nav).dropna()
    if nav.empty:
        return {"days": 0}
    k = int(nav.index.searchsorted(pd.Timestamp(since)))
    if k >= len(nav):
        return {"days": 0}
    start = float(nav.iloc[k - 1]) if k > 0 else float(nav.iloc[0])
    f = np.r_[1.0, nav.iloc[k:].to_numpy(float) / start]
    return {"days": len(nav) - k, "ret": float(f[-1] - 1), "mdd": float((f / np.maximum.accumulate(f) - 1).min()),
            "first": str(nav.index[k].date()), "last": str(nav.index[-1].date())}


def verdict(kind: str, days: int, d_ret, d_mdd, need: int = NEED_DAYS) -> tuple[str, str]:
    """(狀態代碼, 中文)；未滿 need 天只顯示目前領先或落後。"""
    if not days or d_ret is None or not np.isfinite(d_ret):
        return "wait", "等待新資料"
    if kind == "return":
        good = d_ret > 0 and d_mdd >= -0.02
    else:
        good = d_mdd >= 0.02 and d_ret >= -0.02
    if days < need:
        return "accum", f"累積中 {days}／{need} 日（目前{'領先' if good else '落後'}）"
    return ("pass", "前瞻支持（下一版可採用）") if good else ("fail", "前瞻不支持（移出觀察）")


def evaluate(R, res: dict, since: str = SINCE, need: int = NEED_DAYS) -> dict:
    """每日研究呼叫：每個影子模型 × 期間，比較現行模型（同母池、同權重流程）在 since 之後的表現。"""
    from . import tournament as T
    out = {"since": since, "need": need, "asof": str((res.get("protocol") or {}).get("last_date", "")), "rows": [], "errors": []}
    base_cache = {}
    for it in ITEMS:
        for h in it["hz"]:
            hb = (res.get("horizons") or {}).get(h)
            if not hb:
                continue
            try:
                rule, size = hb["chosen_pool"]["rule"], int(hb["chosen_pool"]["size"])
                v = T.variant_of(R, h, T.base_params(R.cfg, h))
                W, _ = R.schedule(rule, size, h, v)
                spec = R.spec(h, v)
                view = R.view(rule, size)
                if h not in base_cache:
                    rb = run(R.B, view, W, spec, start_i=R.first_oos)
                    base_cache[h] = (rb, R.segments(rb).get("dev") or {})
                rb, db = base_cache[h]
                vv = variant(R, it["key"], h, rule, size, v, W, spec)
                if vv is None:
                    out["errors"].append(f"{it['key']} {h}：資料包缺少需要的因子")
                    continue
                rv = run(R.B, vv[2] if len(vv) > 2 else view, vv[0], vv[1], start_i=R.first_oos)
                dv = R.segments(rv).get("dev") or {}
                si = it.get("since", since)                                # v3.21：新加入的候選從加入之後才算前瞻（黃仁勳事件不算新證據）
                fb, fv = forward(rb.nav, si), forward(rv.nav, si)
                fz = forward(rb.bench, si)
                d_ret = (fv["ret"] - fb["ret"]) if fv.get("days") and fb.get("days") else None
                d_mdd = (fv["mdd"] - fb["mdd"]) if fv.get("days") and fb.get("days") else None
                code, zh = verdict(it["kind"], fv.get("days", 0), d_ret, d_mdd, need)
                out["rows"].append({"key": it["key"], "name": it["name"], "h": h, "kind": it["kind"], "why": it["why"],
                                    "pool": f"{rule}{size}", "days": fv.get("days", 0), "first": fv.get("first"), "last": fv.get("last"),
                                    "ret_var": fv.get("ret"), "ret_base": fb.get("ret"), "ret_0050": fz.get("ret"),
                                    "mdd_var": fv.get("mdd"), "mdd_base": fb.get("mdd"), "d_ret": d_ret, "d_mdd": d_mdd,
                                    "dev_d_excess": (dv.get("excess_cagr") or np.nan) - (db.get("excess_cagr") or np.nan),
                                    "dev_d_mdd": (dv.get("mdd") or np.nan) - (db.get("mdd") or np.nan),
                                    "status": code, "status_zh": zh})
            except Exception as e:  # noqa: BLE001  影子模型失敗不影響每日研究
                out["errors"].append(f"{it['key']} {h}：{e}")
    return out
