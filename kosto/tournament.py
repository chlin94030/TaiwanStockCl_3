"""v1.9 冠軍／挑戰者自動晉級：每天用五種換股日比較「目前模型（冠軍）」與預先宣告的鄰近設定（挑戰者），
挑戰者連續領先夠久才自動換上，之後再以新冠軍為中心繼續找下一步 → 模型會隨資料慢慢自動優化，又不會追著雜訊跑。

規則（寫在設定檔 research.auto_promote，可關閉）：
  * 候選＝冠軍在 GRID 每個維度往上／往下一格（精簡因子數 k、價格假說比例、成長假說比例、續抱倍數、持股檔數）。
  * 分數只用開發期（盲測期不參與）：五種換股日平均的年化超額報酬與資訊比率；再看 2019-20 未用期間不可明顯退步。
  * 每日合格：超額 ≥ 冠軍 + min_gain、資訊比率不低於冠軍、2019-20 超額不低於冠軍 − 5pp。
  * 同一個挑戰者連續 streak 個交易日都是第一名才換上；兩次換上至少間隔 cooldown 個交易日。
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

GRID = {"k": [3, 5, 8], "hyp_blend": [0.3, 0.5, 0.7], "growth_blend": [0.0, 0.3, 0.5],
        "hysteresis": [2.0, 2.5, 3.0], "top_k": [6, 8, 10]}
LABEL = {"k": "精簡因子數", "hyp_blend": "價格領先假說比例", "growth_blend": "成長動能假說比例", "hysteresis": "續抱倍數",
         "top_k": "持股檔數"}


def base_params(cfg: dict, h: str) -> dict:
    hc = cfg["horizons"][h]
    m = hc.get("model") or {}
    return {"k": int(m.get("k", 5)), "hyp_blend": float(m.get("hyp_blend", 0.0)), "growth_blend": float(m.get("growth_blend", 0.0)),
            "hysteresis": float(hc.get("hysteresis", 2.0)), "top_k": int(hc.get("top_k", cfg["research"]["top_k"]))}


def key(p: dict) -> str:
    return "|".join(f"{k}={p[k]:g}" for k in GRID)


def describe(p: dict, ref: dict | None = None) -> str:
    if ref is None:
        return "、".join(f"{LABEL[k]} {p[k]:g}" for k in GRID)
    diff = [f"{LABEL[k]} {ref[k]:g}→{p[k]:g}" for k in GRID if abs(p[k] - ref[k]) > 1e-9]
    return "、".join(diff) or "與冠軍相同"


def apply(cfg: dict, champions: dict | None) -> dict:
    """把冠軍設定套到設定檔副本（每日研究與 App 都用這個「有效設定」）。"""
    out = copy.deepcopy(cfg)
    for h, p in (champions or {}).items():
        if h not in out["horizons"]:
            continue
        hc = out["horizons"][h]
        m = dict(hc.get("model") or {})
        m.update({"k": int(p["k"]), "hyp_blend": float(p["hyp_blend"]), "growth_blend": float(p["growth_blend"])})
        hc["model"] = m
        hc["hysteresis"] = float(p["hysteresis"])
        hc["top_k"] = int(p["top_k"])
    return out


def challengers(p: dict) -> list[dict]:
    out = []
    for d, vals in GRID.items():
        cur = p[d]
        k = int(np.argmin([abs(v - cur) for v in vals]))
        for j in (k - 1, k + 1):
            if 0 <= j < len(vals) and abs(vals[j] - cur) > 1e-9:
                out.append(dict(p, **{d: vals[j]}))
    return out


def variant_of(R, h: str, p: dict):
    exp = (R.cfg["horizons"][h].get("model") or {}).get("exposure", "none")
    v = R.model_variant(h, k=p["k"], hyp_blend=p["hyp_blend"], growth_blend=p["growth_blend"], exposure=exp)
    v.hysteresis = float(p["hysteresis"])
    v.top_k = int(p["top_k"])
    return v


def evaluate(R, res: dict, cfg_eff: dict, phase: int, presample: bool = False) -> list[dict]:
    """在一個研究資料包（一種換股日）上評估每個期間的冠軍與挑戰者（母池用當天每日研究選定的母池）。"""
    rows = []
    for h, hb in res["horizons"].items():
        rule, size = hb["chosen_pool"]["rule"], int(hb["chosen_pool"]["size"])
        champ = base_params(cfg_eff, h)
        for p in [champ] + challengers(champ):
            v = variant_of(R, h, p)
            e = R.evaluate(rule, size, h, v)
            row = {"phase": phase, "h": h, "key": key(p), "params": p, "champion": key(p) == key(champ),
                   "dev_excess": e.get("dev_excess_cagr"), "dev_ir": e.get("dev_info_ratio"), "dev_cagr": e.get("dev_cagr"),
                   "hold_cagr": e.get("hold_cagr")}
            if presample:
                try:
                    ps = R.presample(h, rule, size, v)
                    row["pre_excess"] = ps.get("excess_cagr")
                except Exception:  # noqa: BLE001
                    row["pre_excess"] = np.nan
            rows.append(row)
    return rows


def update(prev: dict | None, rows: list[dict], res: dict, cfg: dict, cfg_eff: dict) -> dict:
    """依今天的比較結果更新冠軍狀態；回傳新的 auto 狀態（存進 results.pkl，隔天讀回）。"""
    st = copy.deepcopy(prev or {})
    st.setdefault("champions", {})
    st.setdefault("leader", {})
    st.setdefault("last_switch", {})
    st.setdefault("log", [])
    rcfg = cfg["research"]
    on = bool(rcfg.get("auto_promote", True))
    min_gain = float(rcfg.get("auto_min_gain", 0.02))
    need = int(rcfg.get("auto_streak", 10))
    cool = int(rcfg.get("auto_cooldown", 20))
    today = res["protocol"]["last_date"]
    t = pd.DataFrame(rows)
    table = {}
    for h in res["horizons"]:
        champ = base_params(cfg_eff, h)
        st["champions"].setdefault(h, champ)
        if t.empty or h not in set(t["h"]):
            continue
        g = t[t["h"] == h].groupby("key").agg(dev_excess=("dev_excess", "mean"), dev_ir=("dev_ir", "mean"),
                                              hold_cagr=("hold_cagr", "mean"), n=("phase", "nunique"))
        if "pre_excess" in t:
            g = g.join(t[(t["h"] == h) & t["pre_excess"].notna()].groupby("key")["pre_excess"].mean())
        g["params"] = [next(r["params"] for r in rows if r["h"] == h and r["key"] == k_) for k_ in g.index]
        ck = key(champ)
        if ck not in g.index:
            continue
        c = g.loc[ck]
        g["d_excess"] = g["dev_excess"] - c["dev_excess"]
        g["d_ir"] = g["dev_ir"] - c["dev_ir"]
        if "pre_excess" in g and np.isfinite(c.get("pre_excess", np.nan)):
            pre_ok = g["pre_excess"].isna() | (g["pre_excess"] >= c["pre_excess"] - 0.05)
        else:
            pre_ok = pd.Series(True, index=g.index)
        g["qualified"] = (g.index != ck) & (g["d_excess"] >= min_gain) & (g["d_ir"] >= 0) & pre_ok & (g["n"] >= 3)
        q = g[g["qualified"]].sort_values("d_excess", ascending=False)
        ld = st["leader"].get(h) or {}
        if len(q):
            lk = q.index[0]
            streak = int(ld.get("streak", 0)) + 1 if ld.get("key") == lk else 1
            st["leader"][h] = {"key": lk, "params": q.iloc[0]["params"], "streak": streak, "d_excess": float(q.iloc[0]["d_excess"]),
                               "since": ld.get("since") if ld.get("key") == lk else today}
        else:
            st["leader"][h] = {}
        last = st["last_switch"].get(h)
        days_since = int(np.busday_count(pd.Timestamp(last).date(), pd.Timestamp(today).date())) if last else 10 ** 6
        ld = st["leader"].get(h) or {}
        if on and ld and ld["streak"] >= need and days_since >= cool:
            st["log"].append({"date": today, "h": h, "from": champ, "to": ld["params"], "d_excess": ld["d_excess"],
                              "text": describe(ld["params"], champ)})
            st["champions"][h] = ld["params"]
            st["last_switch"][h] = today
            st["leader"][h] = {}
        table[h] = g.drop(columns=["params"]).assign(desc=[describe(p_, champ) for p_ in g["params"]]).reset_index()
    st["table"] = table
    st["enabled"] = on
    st["rule"] = {"min_gain": min_gain, "streak": need, "cooldown": cool}
    st["updated"] = today
    return st


def log_picks(st: dict, res: dict, keep: int = 400) -> dict:
    """前瞻紀錄：保存每天實際公布的名單（日期、期間、名單）。"""
    log = list(st.get("picks_log", []))
    d = res["protocol"]["last_date"]
    seen = {(x["date"], x["h"]) for x in log}
    for h, hb in res["horizons"].items():
        if (d, h) in seen:
            continue
        log.append({"date": d, "h": h, "days": int(hb["days"]), "picks": [p["ticker"] for p in hb["latest"]["picks"]]})
    st["picks_log"] = log[-keep * 3:]
    st.setdefault("live_since", log[0]["date"] if log else d)
    return st
