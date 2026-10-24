"""Overlapping-tranche backtester.

Timing (no look-ahead):
  * decide at the close of signal day t using data ≤ t
  * trade at the OPEN of t+1 (limit-up open or no trade → that slot stays in cash)
  * hold H trading days, exit at the open of t+H+1 (open→open = exactly H days of exposure)
Portfolio = H/step capital slots (e.g. 20-day horizon, weekly signals → 4 slots, each re-formed every
4 weeks). Slots are not rebalanced against each other. Costs are charged on actual turnover between a
slot's old and new holdings (kept names cost nothing → rewards patience, 律3/戒6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import STAGE_CODES, STAGES
from .optimize import PoolView


@dataclass
class Spec:
    H: int
    step: int
    top_k: int = 10
    hysteresis: float = 1.5
    stop_loss: float | None = None
    use_gates: bool = True
    block_stages: tuple = ("A3", "B1", "B2")
    b3_confirm: bool = True
    block_dtop: bool = True
    exposure: dict = field(default_factory=lambda: {s: 1.0 for s in list(STAGES) + ["UNKNOWN"]})
    buy_cost: float = 0.001425 * 0.6 + 0.001
    sell_cost: float = 0.001425 * 0.6 + 0.003 + 0.001
    capital: float = 1_000_000
    max_participation: float = 0.02
    equal_weight_all: bool = False   # benchmark mode: hold every pool member, no costs/gates


def spec_from_cfg(cfg: dict, horizon: str, exposure_profile: str = "none", **kw) -> Spec:
    h = cfg["horizons"][horizon]
    c = cfg["costs"]
    k = cfg["kostolany"]
    s = Spec(H=h["days"], step=cfg["research"]["signal_step"], top_k=cfg["research"]["top_k"],
             hysteresis=h["hysteresis"], stop_loss=h["stop_loss"], block_stages=tuple(k["block_stages"]),
             b3_confirm=k["b3_requires_confirmation"], block_dtop=k["block_double_top"],
             exposure=dict(k["exposure"][exposure_profile]),
             buy_cost=c["fee"] * c["fee_discount"] + c["slippage"],
             sell_cost=c["fee"] * c["fee_discount"] + c["tax"] + c["slippage"],
             capital=cfg["research"]["capital"], max_participation=cfg["research"]["max_participation"])
    for key, v in kw.items():
        setattr(s, key, v)
    return s


@dataclass
class Result:
    nav: pd.Series            # portfolio value at each close (starts 1.0)
    bench: pd.Series          # 0050 NAV on the same days
    tranches: pd.DataFrame    # one row per signal
    turnover: float           # average one-way turnover per rebalance
    cost_drag: float          # total cost as fraction of capital per year


def choose(B, view: PoolView, i: int, w, prev: list, spec: Spec):
    """Return (picks, ranked_candidates, scores_by_ticker_index)."""
    idx = view.members[i]
    if w is None or len(idx) == 0:
        return [], [], {}
    score = view.Z[i] @ w
    valid = np.ones(len(idx), bool)
    cap = B.a["CAP"][i, idx]
    valid &= np.nan_to_num(cap) * spec.max_participation >= spec.capital / spec.top_k
    if spec.use_gates:
        st = B.a["STAGE"][i, idx]
        block = [STAGE_CODES[s] for s in spec.block_stages]
        valid &= ~np.isin(st, block)
        if spec.b3_confirm:
            valid &= ~((st == STAGE_CODES["B3"]) & ~B.a["WSUP"][i, idx])
        if spec.block_dtop:
            valid &= ~B.a["DTOP"][i, idx]
    cand = idx[valid]
    sc = score[valid]
    order = cand[np.argsort(-sc, kind="stable")]
    smap = dict(zip(idx.tolist(), score.tolist()))
    K = spec.top_k
    buf = set(order[: max(K, int(round(K * spec.hysteresis)))].tolist())
    keep = sorted([j for j in prev if j in buf], key=lambda j: -smap[j])[:K]
    rest = [j for j in order.tolist() if j not in keep]
    picks = keep + rest[: K - len(keep)]
    return picks, order.tolist(), smap


def market_stage(B, i: int) -> str:
    code = int(B.a["MKT"][i, 6])
    return "UNKNOWN" if code == 0 else STAGES[code - 1]


def run(B, view: PoolView, weights: list, spec: Spec, start_i: int = 0) -> Result:
    """weights[i] = factor weight vector for signal i (None → stay in cash)."""
    R, OC, OK = B.a["R"], B.a["OC"], B.a["OK"]
    D = R.shape[0]
    sig = B.sig_pos
    S = len(sig)
    nt = max(1, spec.H // spec.step)
    with np.errstate(invalid="ignore", divide="ignore"):
        GAP = np.where(np.isfinite(OC), (1 + R) / (1 + OC) - 1, 0.0)
    slot_val = np.full((nt, D), np.nan)
    first_day = sig[start_i] + 1 if start_i < S else D
    for s in range(nt):
        slot_val[s, :first_day] = 1.0 / nt
    slot_state = [{"value": 1.0 / nt, "hold": {}, "last": first_day - 1, "prev": []} for _ in range(nt)]
    rows = []
    turnovers, costs = [], 0.0
    for i in range(start_i, S):
        t = sig[i]
        e, x = t + 1, t + spec.H + 1
        if e >= D:
            break
        st = slot_state[(i - start_i) % nt]
        sl = (i - start_i) % nt
        # fill the slot value for any idle days before this entry (cash or drifting holdings ended at open e)
        V = st["value"]
        old = st["hold"]                                    # {j: value at open e}
        stage = market_stage(B, i)
        expo = float(spec.exposure.get(stage, 1.0))
        if spec.equal_weight_all:
            picks = view.members[i].tolist()
            expo = 1.0
        else:
            picks, _, _ = choose(B, view, i, weights[i], st["prev"], spec)
        filled = [j for j in picks if OK[e, j]]
        n_slots = len(picks) if spec.equal_weight_all else spec.top_k
        new_w = {j: expo / max(n_slots, 1) for j in filled}
        old_w = {j: v / V for j, v in old.items()} if V > 0 else {}
        sell = sum(max(0.0, old_w.get(j, 0) - new_w.get(j, 0)) for j in set(old_w) | set(new_w))
        buy = sum(max(0.0, new_w.get(j, 0) - old_w.get(j, 0)) for j in set(old_w) | set(new_w))
        cost = 0.0 if spec.equal_weight_all else sell * spec.sell_cost + buy * spec.buy_cost
        V_start = V
        V = V * (1 - cost)
        costs += V_start * cost
        turnovers.append((sell + buy) / 2)
        js = np.array(list(new_w), int)
        pv = V * np.array([new_w[j] for j in js]) if len(js) else np.zeros(0)
        base = pv.copy()
        cash = V - pv.sum()
        alive = np.ones(len(js), bool)
        pending = np.zeros(len(js), bool)
        last = min(x - 1, D - 1)
        for d in range(e, last + 1):
            if len(js):
                if d == e:
                    pv = pv * (1 + np.nan_to_num(OC[e, js]))
                else:
                    sell_now = pending & OK[d, js]
                    if sell_now.any():  # stop-loss exit at this open (after the overnight gap)
                        pv[sell_now] *= 1 + GAP[d, js[sell_now]]
                        cash += (pv[sell_now] * (1 - spec.sell_cost)).sum()
                        costs += (pv[sell_now] * spec.sell_cost).sum()
                        pv[sell_now] = 0
                        alive &= ~sell_now
                        pending &= ~sell_now
                    pv = np.where(alive, pv * (1 + R[d, js]), pv)
                if spec.stop_loss:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        breach = alive & ~pending & (pv / np.where(base > 0, base, np.nan) <= 1 - spec.stop_loss)
                    pending |= breach
            slot_val[sl, d] = cash + pv.sum()
        hold = {}
        if x < D:
            if len(js):
                pv = np.where(alive, pv * (1 + GAP[x, js]), pv)
            V_end = cash + pv.sum()
            hold = {int(j): float(v) for j, v in zip(js, pv) if v > 0}
            st.update(value=V_end, hold=hold, last=x - 1)
            ret = V_end / V_start - 1
        else:
            V_end = cash + pv.sum()
            st.update(value=V_end, hold={int(j): float(v) for j, v in zip(js, pv) if v > 0}, last=last)
            ret = np.nan  # still open
        st["prev"] = picks
        b = B.a["BENCH"][e: min(x, D)]
        rows.append({"i": i, "date": B.sig_dates[i], "slot": sl, "stage": stage, "exposure": expo,
                     "n_picks": len(picks), "n_filled": len(filled), "turnover": (sell + buy) / 2, "cost": cost,
                     "ret": ret, "bench": float(np.prod(1 + b) - 1) if x < D else np.nan,
                     "picks": [B.tickers[j] for j in picks]})
    # forward-fill idle slot days (between tranches values are written for every day already)
    sv = pd.DataFrame(slot_val.T).ffill().to_numpy()
    nav = pd.Series(sv.sum(axis=1), index=B.days)
    nav = nav.iloc[max(first_day - 1, 0):]
    nav = nav / nav.iloc[0]
    bench = pd.Series(np.cumprod(1 + B.a["BENCH"].astype(float)), index=B.days).reindex(nav.index)
    bench = bench / bench.iloc[0]
    tr = pd.DataFrame(rows)
    years = max(len(nav) / 245, 1e-9)
    return Result(nav=nav, bench=bench, tranches=tr, turnover=float(np.mean(turnovers)) if turnovers else 0.0,
                  cost_drag=float(costs / years))
