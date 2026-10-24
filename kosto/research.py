"""Research protocol (pre-registered, chronological):

  1. 母池測試  every rule × size, same default model (λ=0.40, gates on, full exposure), walk-forward
              out-of-sample on the DEVELOPMENT period only → pick the pool per horizon.
  2. 模型測試  on the chosen pool: Kostolany λ × market-exposure profile, plus K-only; baselines
              DATA (no Kostolany), PRIOR (no fitting), POOL_EW, 0050. Pick on development only.
  3. 盲測      the last `holdout_months` are reported for the chosen model but never used to choose.
  4. 消融      switch each Kostolany component off to measure its contribution.
  5. 今日推薦  refit on all matured data and rank the latest signal date.
Weights are re-estimated every `retrain_months` using only labels that had matured by then
(purged: a training label must end before the retrain date).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import choose, market_stage, run, spec_from_cfg
from .bundle import Bundle
from .config import HORIZON_KEYS, STAGE_NAMES
from .metrics import (block_bootstrap_excess, deflated_sharpe, perf, probabilistic_sharpe, selection_score,
                      tranche_stats, yearly_table)
from .optimize import PoolView, fit_weights, ic_matrix, pool_view


HKEY = {"short": 0, "mid": 1, "long": 2}


@dataclass
class Variant:
    name: str
    mode: str = "opt"          # opt | data | kpure | prior
    lam: float = 0.40
    exposure: str = "none"
    gates: bool = True
    hysteresis: float | None = None   # None → horizon default
    stop: str = "default"             # default | off
    exclude: tuple = ()               # factors forced to weight 0 (ablation)
    stage_gate: bool = True           # 個股雞蛋階段閘門
    dtop_gate: bool = True            # M頭閘門

    def label(self) -> str:
        return self.name


class Research:
    def __init__(self, B: Bundle, cfg: dict, log=print):
        self.B, self.cfg, self.log = B, cfg, log
        r = cfg["research"]
        sd = B.sig_dates
        self.S = len(sd)
        self.first_oos = int(sd.searchsorted(sd[0] + pd.DateOffset(months=r["train_min_months"])))
        self.holdout = int(sd.searchsorted(sd[-1] - pd.DateOffset(months=r["holdout_months"])))
        self.retrain = []
        d = sd[min(self.first_oos, self.S - 1)]
        while True:
            i = int(sd.searchsorted(d))
            if i >= self.S:
                break
            self.retrain.append(i)
            d = d + pd.DateOffset(months=r["retrain_months"])
        self._views, self._ics, self._sched = {}, {}, {}

    # ------------------------------------------------------------------ cached pieces
    def view(self, rule: str, size: int) -> PoolView:
        key = (rule, size)
        if key not in self._views:
            if len(self._views) > 6:
                self._views.pop(next(iter(self._views)))
                self._ics = {k: v for k, v in self._ics.items() if (k[0], k[1]) in self._views}
            ri = self.B.meta["rules"].index(rule)
            rk = self.B.a["RANKS"][ri]
            mask = (rk >= 1) & (rk <= size)
            self._views[key] = pool_view(self.B.a["U"], mask, self.B.a["TRADE"])
        return self._views[key]

    def ic(self, rule, size, h) -> np.ndarray:
        key = (rule, size, h)
        if key not in self._ics:
            self._ics[key] = ic_matrix(self.view(rule, size), self.B.a["FWD"][HKEY[h]])
        return self._ics[key]

    def H(self, h) -> int:
        return self.cfg["horizons"][h]["days"]

    def train_rows(self, upto_i: int, h: str) -> np.ndarray:
        sp = self.B.sig_pos
        cut = sp[upto_i] if upto_i < self.S else sp[-1] + 1
        return np.where(sp + self.H(h) + 1 <= cut)[0]

    # ------------------------------------------------------------------ weights
    def schedule(self, rule, size, h, v: Variant):
        r0 = self.cfg["research"]
        key = (rule, size, h, v.mode, round(v.lam, 4), tuple(sorted(self.cfg["horizons"][h]["family_prior"].items())),
               r0["n0_shrink"], r0["max_factor_weight"], tuple(v.exclude))
        if key in self._sched:
            return self._sched[key]
        if len(self._sched) > 64:
            self._sched.clear()
        view, ic = self.view(rule, size), self.ic(rule, size, h)
        r = self.cfg["research"]
        W = [None] * self.S
        hist = []
        pts = self.retrain + [self.S]
        for a, b in zip(pts[:-1], pts[1:]):
            rows = self.train_rows(a, h)
            fit = fit_weights(ic, rows, self.B.meta, h, self.cfg["horizons"][h]["family_prior"], v.lam, v.mode,
                              r["n0_shrink"], r["max_factor_weight"], r["signal_step"], self.H(h), view,
                              self.B.a["FWD"][HKEY[h]], exclude=v.exclude)
            for i in range(a, b):
                W[i] = fit.weights
            hist.append({"from": str(self.B.sig_dates[a].date()), "n_train": fit.n_rows, "alpha": round(fit.alpha, 3),
                         **{f"fam_{k}": round(val, 4) for k, val in fit.family.items()},
                         "weights": {f: float(w) for f, w in zip(self.B.factors, fit.weights) if abs(w) > 1e-6}})
        self._sched[key] = (W, hist)
        return W, hist

    def final_fit(self, rule, size, h, v: Variant):
        r = self.cfg["research"]
        rows = self.train_rows(self.S, h)
        return fit_weights(self.ic(rule, size, h), rows, self.B.meta, h, self.cfg["horizons"][h]["family_prior"],
                           v.lam, v.mode, r["n0_shrink"], r["max_factor_weight"], r["signal_step"], self.H(h),
                           self.view(rule, size), self.B.a["FWD"][HKEY[h]], exclude=v.exclude)

    # ------------------------------------------------------------------ evaluation
    def spec(self, h, v: Variant):
        kw = {"use_gates": v.gates}
        if not v.stage_gate:
            kw.update(block_stages=(), b3_confirm=False)
        if not v.dtop_gate:
            kw["block_dtop"] = False
        if v.hysteresis is not None:
            kw["hysteresis"] = v.hysteresis
        if v.stop == "off":
            kw["stop_loss"] = None
        return spec_from_cfg(self.cfg, h, v.exposure, **kw)

    def segments(self, res):
        sd = self.B.sig_dates
        hd = sd[min(self.holdout, self.S - 1)]
        nav, bench = res.nav, res.bench
        dev = nav[nav.index < hd]
        hol = nav[nav.index >= nav.index[nav.index.searchsorted(hd) - 1]] if hd > nav.index[0] else nav
        return {"dev": perf(dev / dev.iloc[0], bench.reindex(dev.index)) if len(dev) > 20 else {},
                "holdout": perf(hol / hol.iloc[0], bench.reindex(hol.index)) if len(hol) > 20 else {},
                "oos": perf(nav, bench)}

    def evaluate(self, rule, size, h, v: Variant, keep=False) -> dict:
        W, hist = self.schedule(rule, size, h, v)
        res = run(self.B, self.view(rule, size), W, self.spec(h, v), start_i=self.first_oos)
        seg = self.segments(res)
        out = {"rule": rule, "size": size, "horizon": h, "variant": v.name, "mode": v.mode, "lam": v.lam,
               "exposure": v.exposure, "gates": v.gates, **{f"dev_{k}": val for k, val in seg["dev"].items()},
               **{f"hold_{k}": val for k, val in seg["holdout"].items()},
               **{f"oos_{k}": val for k, val in seg["oos"].items()},
               "turnover": res.turnover, **{f"tr_{k}": val for k, val in tranche_stats(
                   res.tranches[res.tranches["date"] < self.B.sig_dates[min(self.holdout, self.S - 1)]]).items()}}
        if keep:
            out["_res"], out["_hist"], out["_W"] = res, hist, W
        return out

    def pool_ew(self, rule, size, h):
        spec = spec_from_cfg(self.cfg, h, "none", equal_weight_all=True, use_gates=False)
        res = run(self.B, self.view(rule, size), [None] * self.S, spec, start_i=self.first_oos)
        return res

    # ------------------------------------------------------------------ studies
    def pool_study(self, h, rules=None, sizes=None, progress=None) -> pd.DataFrame:
        rules = rules or self.cfg["pools"]["rules"]
        sizes = sizes or self.cfg["pools"]["sizes"]
        v = Variant("POOL_TEST", "opt", 0.40, "none", True)
        rows = []
        for rule in rules:
            for size in sizes:
                t0 = time.time()
                row = self.evaluate(rule, size, h, v)
                ew = self.pool_ew(rule, size, h)
                seg = self.segments(ew)["dev"]
                row.update({"pool_ew_dev_cagr": seg.get("cagr"), "pool_ew_dev_sharpe": seg.get("sharpe"),
                            "avg_members": float(np.mean([len(m) for m in self.view(rule, size).members[self.first_oos:]])),
                            "ic_composite_dev": self._composite_dev_ic(rule, size, h)})
                rows.append(row)
                if progress:
                    progress(f"{h} 母池 {rule}{size}: 開發期CAGR {row.get('dev_cagr', np.nan):.1%} "
                             f"Sharpe {row.get('dev_sharpe', np.nan):.2f} ({time.time()-t0:.1f}s)")
        t = pd.DataFrame(rows)
        t["score"] = selection_score(t.rename(columns=lambda c: c.replace("dev_", "")), self.cfg["research"]["selection_weights"])
        return t.sort_values("score", ascending=False).reset_index(drop=True)

    def _composite_dev_ic(self, rule, size, h):
        """Descriptive only (not used to choose): mean of the 5 largest |IC| in the development period."""
        ic = self.ic(rule, size, h)
        rows = np.arange(self.first_oos, self.holdout)
        rows = rows[np.isin(rows, self.train_rows(self.holdout, h))] if len(rows) else rows
        if not len(rows):
            return np.nan
        ok = np.array([h in hz for hz in self.B.meta["horizons"]])
        m = np.abs(np.nanmean(ic[rows][:, ok], 0))
        return float(np.nanmean(np.sort(m)[-5:]))

    def variants(self) -> list[Variant]:
        r = self.cfg["research"]
        vs = [Variant(f"KOSTO λ={lam:.2f} 水位={ex}", "opt", lam, ex, True) for lam in r["lambdas"]
              for ex in r["exposure_profiles"]]
        vs.append(Variant("KOSTO_PURE 只用科斯托蘭尼因子", "kpure", 1.0, "soft", True))
        return vs

    def baselines(self) -> list[Variant]:
        return [Variant("DATA 純數據(無科斯托蘭尼)", "data", 0.0, "none", False),
                Variant("PRIOR 先驗權重(不最佳化)", "prior", 0.40, "soft", True)]

    def variant_study(self, h, rule, size, progress=None) -> pd.DataFrame:
        rows = []
        for v in self.variants() + self.baselines():
            row = self.evaluate(rule, size, h, v)
            row["is_baseline"] = v.mode in ("data", "prior")
            rows.append(row)
            if progress:
                progress(f"{h} 模型 {v.name}: 開發期CAGR {row.get('dev_cagr', np.nan):.1%}")
        t = pd.DataFrame(rows)
        t["score"] = selection_score(t.rename(columns=lambda c: c.replace("dev_", "")), self.cfg["research"]["selection_weights"])
        return t

    def ablation(self, h, rule, size, v: Variant) -> pd.DataFrame:
        from dataclasses import replace
        kf = [f for f, fam, hz in zip(self.B.factors, self.B.meta["families"], self.B.meta["horizons"])
              if fam == "K" and h in hz]
        zh = dict(zip(self.B.factors, self.B.meta["factor_zh"]))
        cases = [("完整科斯托蘭尼模型", v),
                 ("移除個股雞蛋階段閘門", replace(v, stage_gate=False)),
                 ("移除M頭閘門", replace(v, dtop_gate=False)),
                 ("移除個股雞蛋階段/M頭閘門", replace(v, gates=False)),
                 ("移除市場階段持股水位", replace(v, exposure="none")),
                 ("移除科斯托蘭尼因子(K族群)", replace(v, mode="data", lam=0.0)),
                 ("移除耐心續抱(遲滯)", replace(v, hysteresis=1.0)),
                 ("移除停損(戒3)", replace(v, stop="off"))]
        if v.mode in ("opt", "kpure"):
            cases += [(f"移除 {f}（{zh[f]}）", replace(v, exclude=(f,))) for f in kf]
        rows = []
        for label, vv in cases:
            e = self.evaluate(rule, size, h, vv)
            rows.append({"case": label, "dev_cagr": e.get("dev_cagr"), "dev_sharpe": e.get("dev_sharpe"),
                         "dev_mdd": e.get("dev_mdd"), "hold_cagr": e.get("hold_cagr"), "hold_mdd": e.get("hold_mdd"),
                         "turnover": e.get("turnover")})
        t = pd.DataFrame(rows)
        base = t.iloc[0]
        t["貢獻(開發期CAGR)"] = base["dev_cagr"] - t["dev_cagr"]
        return t

    # ------------------------------------------------------------------ latest recommendations
    def latest(self, h, rule, size, v: Variant, res=None) -> dict:
        B = self.B
        fit = self.final_fit(rule, size, h, v)
        spec = self.spec(h, v)
        i = self.S - 1
        nt = max(1, spec.H // spec.step)
        prev = []
        if res is not None and len(res.tranches) > nt:
            prev_row = res.tranches[res.tranches["i"] == i - nt]
            if len(prev_row):
                prev = [B.tickers.index(t) for t in prev_row.iloc[0]["picks"] if t in B.tickers]
        view = self.view(rule, size)
        picks, order, smap = choose(B, view, i, fit.weights, prev, spec)
        stage = market_stage(B, i)
        idx = view.members[i]
        pos_of = {j: k for k, j in enumerate(idx.tolist())}
        Z = view.Z[i]
        fz = B.meta["factor_zh"]

        def describe(j):
            k = pos_of[j]
            contrib = Z[k] * fit.weights
            top = np.argsort(-contrib)[:3]
            neg = np.argsort(contrib)[:1]
            st = int(B.a["STAGE"][i, j])
            close = float(B.a["CLOSE"][i, j])
            stop = self.cfg["horizons"][h]["stop_loss"]
            return {"ticker": B.tickers[j], "name": B.meta["names"][j], "sector": B.meta["sectors"][j],
                    "close": close, "score": float(smap.get(j, np.nan)), "stage": STAGE_NAMES[st],
                    "kept": j in prev, "w_support": bool(B.a["WSUP"][i, j]),
                    "reasons": [f"{fz[f]}(+{contrib[f]:.3f})" for f in top if contrib[f] > 0],
                    "risk": [f"{fz[f]}({contrib[f]:.3f})" for f in neg if contrib[f] < 0],
                    "stop_price": round(close * (1 - stop), 2) if stop else None}

        blocked = []
        top_all = idx[np.argsort(-(Z @ fit.weights))][:30] if len(idx) else []
        chosen = set(picks) | set(order[: 2 * spec.top_k])
        for j in top_all:
            if j in chosen:
                continue
            st = STAGE_NAMES[int(B.a["STAGE"][i, j])]
            why = []
            if st in spec.block_stages:
                why.append(f"個股階段{st}")
            if spec.block_dtop and B.a["DTOP"][i, j]:
                why.append("M頭(多次攻不過)")
            if st == "B3" and spec.b3_confirm and not B.a["WSUP"][i, j]:
                why.append("B3未見止跌確認")
            if np.nan_to_num(B.a["CAP"][i, j]) * spec.max_participation < spec.capital / spec.top_k:
                why.append("流動性不足")
            if why:
                blocked.append({"ticker": B.tickers[j], "name": B.meta["names"][j], "why": "、".join(why)})
        return {"date": str(B.sig_dates[i].date()), "market_stage": stage,
                "exposure": float(spec.exposure.get(stage, 1.0)), "pool": f"{rule}{size}", "variant": v.name,
                "picks": sorted([describe(j) for j in picks], key=lambda d: -d["score"]),
                "alternates": [describe(j) for j in order if j not in picks][:10],
                "blocked": blocked[:10],
                "weights": {f: float(w) for f, w in zip(B.factors, fit.weights) if abs(w) > 1e-6},
                "family": fit.family}

    # ------------------------------------------------------------------ factor table
    def factor_table(self, h, rule, size, weights) -> pd.DataFrame:
        ic = self.ic(rule, size, h)
        B = self.B
        dev = np.arange(self.first_oos if self.first_oos < self.S else 0, self.holdout)
        allr = self.train_rows(self.S, h)
        hol = np.arange(self.holdout, self.S)
        hol = hol[np.isin(hol, allr)]
        step, H = self.cfg["research"]["signal_step"], self.H(h)

        def st(rows):
            X = ic[rows]
            m = np.nanmean(X, 0) if len(rows) else np.full(ic.shape[1], np.nan)
            sd = np.nanstd(X, 0, ddof=1) if len(rows) > 2 else np.full(ic.shape[1], np.nan)
            neff = max(len(rows) * step / max(H, step), 1)
            return m, m / np.where(sd > 0, sd, np.nan), m / np.where(sd > 0, sd, np.nan) * np.sqrt(neff)
        all_rows = self.train_rows(self.S, h)
        m_all, ir_all, t_all = st(all_rows)
        m_dev, ir_dev, t_dev = st(dev[np.isin(dev, all_rows)])
        m_h, _, _ = st(hol)
        rows = []
        for k, f in enumerate(B.factors):
            rows.append({"factor": f, "中文": B.meta["factor_zh"][k], "family": B.meta["families"][k],
                         "prior_dir": B.meta["directions"][k], "horizon_ok": h in B.meta["horizons"][k],
                         "coverage": B.meta["coverage"][f], "ic_all": m_all[k], "icir_all": ir_all[k], "t_all": t_all[k],
                         "ic_dev": m_dev[k], "t_dev": t_dev[k], "ic_holdout": m_h[k], "weight": weights.get(f, 0.0)})
        return pd.DataFrame(rows).sort_values(["family", "weight"], ascending=[True, False])

    # ------------------------------------------------------------------ stage performance
    def stage_table(self, res) -> pd.DataFrame:
        tr = res.tranches.dropna(subset=["ret"]).copy()
        if tr.empty:
            return pd.DataFrame()
        tr["excess"] = tr["ret"] - tr["bench"]
        g = tr.groupby("stage").agg(n=("ret", "size"), avg_ret=("ret", "mean"), avg_bench=("bench", "mean"),
                                    avg_excess=("excess", "mean"), avg_exposure=("exposure", "mean"))
        return g.reset_index()

    # ------------------------------------------------------------------ full pipeline
    def run_all(self, horizons=HORIZON_KEYS, quick=False, progress=print) -> dict:
        t0 = time.time()
        out = {"meta": {k: self.B.meta[k] for k in ("version", "built", "synthetic", "data_quality")},
               "protocol": {"first_oos": str(self.B.sig_dates[min(self.first_oos, self.S - 1)].date()),
                            "holdout_start": str(self.B.sig_dates[min(self.holdout, self.S - 1)].date()),
                            "last_date": str(self.B.sig_dates[-1].date()), "signals": self.S,
                            "retrain_points": [str(self.B.sig_dates[i].date()) for i in self.retrain],
                            "quick": quick},
               "horizons": {}}
        rules = self.cfg["pools"]["rules"]
        sizes = self.cfg["pools"]["sizes"]
        if quick:
            rules = [r for r in rules if r in ("ADV", "KOSTO", "INST2Y")] or rules
            sizes = [s for s in sizes if s in (150, 300)] or sizes[:2]
        for h in horizons:
            progress(f"=== {self.cfg['horizons'][h]['label']} ===")
            pt = self.pool_study(h, rules, sizes, progress)
            best = pt.iloc[0]
            rule, size = best["rule"], int(best["size"])
            progress(f"{h} 選定母池 {rule}{size}")
            vt = self.variant_study(h, rule, size, progress)
            kv = vt[~vt["is_baseline"]].sort_values("score", ascending=False)
            v = next(x for x in self.variants() if x.name == kv.iloc[0]["variant"])
            out["horizons"][h] = self.horizon_block(h, rule, size, v, pt, vt)
            ev = out["horizons"][h]["eval"]
            progress(f"{h} 完成：開發期CAGR {ev.get('dev_cagr', np.nan):.1%}，盲測CAGR {ev.get('hold_cagr', np.nan):.1%} "
                     f"({time.time()-t0:.0f}s)")
        out["market"] = self.market_frame()
        out["elapsed"] = time.time() - t0
        return out

    def market_frame(self) -> pd.DataFrame:
        md = self.B.a["MKT_DAILY"]
        cols = ["A1", "A2", "A3", "B1", "B2", "B3", "confidence", "index", "drawdown", "turnover_ratio", "breadth"]
        mk = pd.DataFrame(md, index=self.B.days, columns=cols)
        mk["stage"] = [STAGE_NAMES[int(c)] for c in self.B.a["MKT_DAILY_CODE"]]
        return mk

    def horizon_block(self, h, rule, size, v: Variant, pt=None, vt=None, ablation=True) -> dict:
        """Everything the report needs for one horizon / pool / model."""
        if pt is None:
            row = self.evaluate(rule, size, h, v)
            row["score"] = np.nan
            pt = pd.DataFrame([row])
        if vt is None:
            rows = [dict(self.evaluate(rule, size, h, v), is_baseline=False)]
            rows += [dict(self.evaluate(rule, size, h, bv), is_baseline=True) for bv in self.baselines()]
            vt = pd.DataFrame(rows)
            vt["score"] = selection_score(vt.rename(columns=lambda c: c.replace("dev_", "")),
                                          self.cfg["research"]["selection_weights"])
        ev = self.evaluate(rule, size, h, v, keep=True)
        res = ev.pop("_res")
        hist = ev.pop("_hist")
        ev.pop("_W")
        base = {}
        for bv in self.baselines():
            be = self.evaluate(rule, size, h, bv, keep=True)
            base[bv.name] = be.pop("_res")
        ew = self.pool_ew(rule, size, h)
        daily = res.nav.pct_change().dropna()
        trials = pd.concat([pt, vt], ignore_index=True)
        sr_std = float(np.nanstd(trials["oos_sharpe"] / np.sqrt(245))) if "oos_sharpe" in trials else np.nan
        robust = {"psr": probabilistic_sharpe(daily), "dsr": deflated_sharpe(daily, len(trials), sr_std),
                  "n_trials": int(len(trials)), **block_bootstrap_excess(res.nav, res.bench)}
        own = vt[vt["variant"] == v.name]["score"]
        data_wins = bool(len(own) and vt[vt["mode"] == "data"]["score"].max() > own.max())
        ab = self.ablation(h, rule, size, v) if ablation else pd.DataFrame(
            columns=["case", "dev_cagr", "dev_sharpe", "dev_mdd", "hold_cagr", "hold_mdd", "turnover", "貢獻(開發期CAGR)"])
        lt = self.latest(h, rule, size, v, res)
        fit = self.final_fit(rule, size, h, v)
        ft = self.factor_table(h, rule, size, lt["weights"])
        return {
            "label": self.cfg["horizons"][h]["label"], "days": self.H(h),
            "pool_table": pt, "chosen_pool": {"rule": rule, "size": size}, "variant_table": vt,
            "chosen": {"variant": v.name, "mode": v.mode, "lam": v.lam, "exposure": v.exposure, "gates": v.gates},
            "data_beats_kostolany": data_wins, "eval": ev, "fit_history": hist, "factor_table": ft,
            "family_final": fit.family, "robust": robust, "ablation": ab, "stage_table": self.stage_table(res),
            "yearly": yearly_table(res.nav, res.bench), "latest": lt,
            "nav": pd.DataFrame({"策略": res.nav, "0050": res.bench, "母池等權": ew.nav.reindex(res.nav.index),
                                 **{k.split()[0]: r.nav.reindex(res.nav.index) for k, r in base.items()}}),
            "tranches": res.tranches, "turnover": res.turnover, "cost_drag": res.cost_drag,
        }

    def single(self, h, rule, size, v: Variant, ablation=True) -> dict:
        """Report-ready result for one user-chosen configuration (Streamlit quick test)."""
        t0 = time.time()
        out = {"meta": {k: self.B.meta[k] for k in ("version", "built", "synthetic", "data_quality")},
               "protocol": {"first_oos": str(self.B.sig_dates[min(self.first_oos, self.S - 1)].date()),
                            "holdout_start": str(self.B.sig_dates[min(self.holdout, self.S - 1)].date()),
                            "last_date": str(self.B.sig_dates[-1].date()), "signals": self.S,
                            "retrain_points": [str(self.B.sig_dates[i].date()) for i in self.retrain], "quick": False},
               "horizons": {h: self.horizon_block(h, rule, size, v, ablation=ablation)}, "market": self.market_frame()}
        out["elapsed"] = time.time() - t0
        return out
