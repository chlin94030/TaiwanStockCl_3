"""v3.4 美股回測：每 5 日一批名單、重疊持有（中線 4 批、長線 12 批），隔日開盤換股，名次加權，含成本。

每一「批次槽」各自買進持有 H 日後在下一次輪到它時換股；整體報酬＝所有批次槽的資產合計。
換股日（訊號日 s 的下一個交易日 d）：舊持股先承受隔夜跳空（開盤價 ÷ 前一日收盤），開盤時換成新名單並扣成本，
新持股當天只算「收盤 ÷ 開盤」。entry="close" 則在訊號日收盤直接換股（盤中即時名單收盤前下單的情境）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import pick, rank_weights


def _nz(x):
    return np.where(np.isfinite(x), x, 1.0)


def _turnover(idx, val, V, nidx, w) -> float:
    """買賣金額合計 ÷ 資產（買 1 次＋賣 1 次＝2）。"""
    old = {}
    if len(idx) and V > 0:
        for i, v in zip(idx.tolist(), (val / V).tolist()):
            old[i] = old.get(i, 0.0) + v
    new = {}
    for i, x in zip(nidx.tolist(), w.tolist()):
        new[i] = new.get(i, 0.0) + x
    keys = old.keys() | new.keys()
    return float(sum(abs(new.get(i, 0.0) - old.get(i, 0.0)) for i in keys))


def run(C: pd.DataFrame, O: pd.DataFrame, S: pd.DataFrame | None, U: pd.DataFrame, H: int, k: int = 10,
        hyst: float = 2.5, weighting: str = "rank", cost: float = 0.001, entry: str = "open",
        step: int = 5, start: int = 0, bench: bool = False, track: bool = False, chooser=None):
    """回傳每日報酬（pd.Series）；track=True 另回傳每次換股紀錄（訊號日、名單、換手率）。
    bench=True：每次買進整個母池等權（不續抱、不扣成本）＝基準。
    chooser(s, held) → (名單, 權重)：v3.6 自訂選股（例如大盤濾網時改抱 SPY；C 需含該欄位）。"""
    dates = C.index
    c, o = C.to_numpy(np.float64), O.to_numpy(np.float64)
    T = len(dates)
    sig = list(range(start, T - 1, step))
    nslot = max(1, H // step)
    cols = np.arange(C.shape[1])
    # 每個槽：目前持股欄位 idx、每檔資產（已含漲跌）
    slots = [{"idx": np.zeros(0, int), "val": np.zeros(0), "names": []} for _ in range(nslot)]
    rebal = {}                                   # 換股生效日 d → (槽, 新名單 idx, 權重, 名單)
    Uv = U.to_numpy(bool)
    Sv = S.to_numpy(np.float64) if S is not None else None
    log = []
    held_by_slot: dict[int, list] = {}
    for j, s in enumerate(sig):
        slot = j % nslot
        u = Uv[s] & np.isfinite(c[s])
        if not u.any():
            continue
        if bench:
            idx = cols[u]
            w = np.full(len(idx), 1.0 / len(idx))
            names = None
        elif chooser is not None:
            names, w = chooser(s, held_by_slot.get(slot, []))
            held_by_slot[slot] = names
            idx = C.columns.get_indexer(names)
            w = np.asarray(w, dtype=float)
            if (idx < 0).any() or not len(idx):
                continue
        else:
            sc = pd.Series(np.where(u, Sv[s], np.nan), index=C.columns)
            names = pick(sc, k, held_by_slot.get(slot, []), hyst)
            held_by_slot[slot] = names
            idx = C.columns.get_indexer(names)
            w = rank_weights(len(idx), weighting)
        d = s + 1 if entry == "open" else s
        if d < T:
            rebal.setdefault(d, []).append((slot, idx, w, names))
    total_prev = None
    rets = np.full(T, np.nan)
    started = np.zeros(nslot, bool)
    for d in range(1, T):
        events = rebal.get(d, [])
        ev_by_slot = {e[0]: e for e in events}
        for si, st in enumerate(slots):
            idx, val = st["idx"], st["val"]
            e = ev_by_slot.get(si)
            if e is None:
                if len(idx):
                    g = _nz(c[d, idx] / c[d - 1, idx])
                    st["val"] = val * g
                continue
            _, nidx, w, names = e
            if entry == "open":
                # 1) 舊持股隔夜跳空
                if len(idx):
                    g = _nz(o[d, idx] / c[d - 1, idx])
                    val = val * g
                V = val.sum() if len(idx) else st.get("cash", 1.0)
                started[si] = True
                turn = _turnover(idx, val, V, nidx, w)
                Vn = V * (1 - (0 if bench else cost) * turn)
                eo = o[d, nidx]
                eo = np.where(np.isfinite(eo), eo, c[d - 1, nidx])
                g = _nz(c[d, nidx] / eo)
                st["idx"], st["val"] = nidx, Vn * w * g
            else:   # 收盤換股：當天先以舊持股收盤計算，再換
                if len(idx):
                    val = val * _nz(c[d, idx] / c[d - 1, idx])
                V = val.sum() if len(idx) else st.get("cash", 1.0)
                started[si] = True
                turn = _turnover(idx, val, V, nidx, w)
                Vn = V * (1 - (0 if bench else cost) * turn)
                st["idx"], st["val"] = nidx, Vn * w
            if track and not bench:
                log.append({"date": dates[d], "slot": si, "names": names, "turnover": turn / 2})
        if started.all():
            tot = sum(st["val"].sum() for st in slots)
            if total_prev is not None and total_prev > 0:
                rets[d] = tot / total_prev - 1
            total_prev = tot
    out = pd.Series(rets, index=dates).dropna()
    return (out, pd.DataFrame(log)) if track else out


def stats(r: pd.Series, b: pd.Series, a: str | None = None, z: str | None = None) -> dict:
    """期間績效：年化報酬、對基準年化超額、資訊比率、最大回撤、半年段落勝率。"""
    r, b = r.align(b, join="inner")
    if a:
        r, b = r[r.index >= a], b[b.index >= a]
    if z:
        r, b = r[r.index <= z], b[b.index <= z]
    if len(r) < 20:
        return {}
    yrs = len(r) / 252

    def cagr(x):
        return float(np.prod(1 + x) ** (1 / yrs) - 1)

    def mdd(x):
        v = np.cumprod(1 + x)
        return float((v / np.maximum.accumulate(v) - 1).min())
    ex = r - b
    halves = pd.Series(r.index.year * 2 + (r.index.month > 6), index=r.index)
    wins = [float(np.prod(1 + r[halves == h]) > np.prod(1 + b[halves == h])) for h in halves.unique()]
    return {"cagr": cagr(r), "bench": cagr(b), "excess": cagr(r) - cagr(b),
            "ir": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else np.nan,
            "mdd": mdd(r), "bench_mdd": mdd(b), "fold_win": float(np.mean(wins)), "n": len(r)}
