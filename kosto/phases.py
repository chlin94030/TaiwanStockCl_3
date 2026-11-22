"""換股日敏感度（v1.8）：每 5 個交易日換股時，「從哪一天開始算」會讓回測差很多（實測同一模型樣本外年化 43%～70%）。

做法：把最後 0～4 個交易日拿掉重建研究資料包 → 訊號日整體平移 0～4 天（涵蓋一週五種換股日），
用每日研究選定的母池與模型重跑回測，報告平均與最差／最好，作為較穩健的預期報酬。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .bundle import build_bundle
from .panel import build_panel
from .research import Research
from .store import Store


def current_book(tranches: pd.DataFrame, n_slots: int) -> dict:
    """v3.20：某一種換股日「現在持有」的組合＝最近 n_slots 批名單，每批 1／n_slots、批內等權（不含已沒有名單的批次）。"""
    if tranches is None or not len(tranches):
        return {}
    tr = tranches.sort_values("i").tail(n_slots)
    book = {}
    for picks in tr["picks"]:
        picks = list(picks or [])
        for t in picks:
            book[t] = book.get(t, 0.0) + 1.0 / n_slots / len(picks)
    tot = sum(book.values())
    return {t: w / tot for t, w in book.items()} if tot > 0 else {}


def consensus(books: dict) -> dict:
    """v3.20 五種換股日共識組合：每種換股日各占 1／n 資金（＝把五種換股日的回測平均，報酬期望值相同、少了「哪一天開始」的運氣）。
    books＝{h: {phase: {ticker: weight}}} → {h: {"n_phases", "rows": [{ticker, weight, phases}]}}。"""
    out = {}
    for h, bk in books.items():
        bk = {k: b for k, b in bk.items() if b}
        if not bk:
            continue
        agg, cnt = {}, {}
        for b in bk.values():
            for t, w in b.items():
                agg[t] = agg.get(t, 0.0) + w / len(bk)
                cnt[t] = cnt.get(t, 0) + 1
        rows = sorted(({"ticker": t, "weight": round(w, 4), "phases": cnt[t]} for t, w in agg.items()), key=lambda r: -r["weight"])
        out[h] = {"n_phases": len(bk), "rows": rows}
    return out


def phase_check(cfg: dict, res: dict, n: int = 4, log=print, on_bundle=None, books: dict | None = None) -> pd.DataFrame:
    """on_bundle(R, k)：每個平移起點建好研究物件後呼叫（v1.9 冠軍／挑戰者比較共用這些資料包，不重建）。
    books（v3.20）：傳入 dict 時，另外記下每種換股日目前持有的組合（給五種換股日共識組合用）。"""
    t0 = time.time()
    rows = []
    if books is not None:
        for h, hb in res["horizons"].items():
            nt = max(1, int(cfg["horizons"][h]["days"]) // int(cfg["research"]["signal_step"]))
            books.setdefault(h, {})[0] = current_book(hb.get("tranches"), nt)
    for h, hb in res["horizons"].items():
        ev = hb["eval"]
        rows.append({"phase": 0, "end": res["protocol"]["last_date"], "h": h, **{k: ev.get(k) for k in
                     ("dev_cagr", "hold_cagr", "oos_cagr", "oos_mdd", "oos_bench_cagr", "hold_bench_cagr")}})
    st = Store(cfg["data"]["dir"])
    px = st.read("price", pd.Timestamp(res["protocol"]["last_date"]) - pd.Timedelta(days=40), None, columns=["date"])
    days = pd.DatetimeIndex(sorted(px["date"].unique())) if not px.empty else pd.DatetimeIndex([])
    days = days[days <= pd.Timestamp(res["protocol"]["last_date"])]
    for k in range(1, n + 1):
        if len(days) <= k:
            break
        end = days[-1 - k]
        try:
            P = build_panel(st, start=cfg["data"]["start"], end=end)
            B = build_bundle(P, cfg, log=lambda *a: None)
            del P
            R = Research(B, cfg, log=lambda *a: None)
            for h, hb in res["horizons"].items():
                rule, size = hb["chosen_pool"]["rule"], int(hb["chosen_pool"]["size"])
                v = R.model_variant(h, exposure=(cfg["horizons"][h].get("model") or {}).get("exposure", "none"))
                ev = R.evaluate(rule, size, h, v, keep=books is not None)
                if books is not None:
                    rk = ev.pop("_res", None)
                    ev.pop("_hist", None)
                    ev.pop("_W", None)
                    nt = max(1, int(cfg["horizons"][h]["days"]) // int(cfg["research"]["signal_step"]))
                    books.setdefault(h, {})[k] = current_book(rk.tranches if rk is not None else None, nt)
                rows.append({"phase": k, "end": str(end.date()), "h": h, **{c: ev.get(c) for c in
                             ("dev_cagr", "hold_cagr", "oos_cagr", "oos_mdd", "oos_bench_cagr", "hold_bench_cagr")}})
            if on_bundle is not None:
                try:
                    on_bundle(R, k)
                except Exception as e:  # noqa: BLE001
                    log(f"冠軍／挑戰者比較（平移 {k} 天）失敗：{e}")
            del B, R
            log(f"換股日敏感度：平移 {k} 天完成（{time.time() - t0:.0f}s）")
        except Exception as e:  # noqa: BLE001 - optional analysis
            log(f"換股日敏感度：平移 {k} 天失敗：{e}")
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> pd.DataFrame:
    if t is None or not len(t):
        return pd.DataFrame()
    g = t.groupby("h")
    out = pd.DataFrame({"樣本外年化_平均": g["oos_cagr"].mean(), "樣本外年化_最低": g["oos_cagr"].min(),
                        "樣本外年化_最高": g["oos_cagr"].max(), "近一年_平均": g["hold_cagr"].mean(),
                        "近一年_最低": g["hold_cagr"].min(), "近一年_最高": g["hold_cagr"].max(),
                        "最大回撤_平均": g["oos_mdd"].mean(), "0050樣本外": g["oos_bench_cagr"].mean(),
                        "0050近一年": g["hold_bench_cagr"].mean(), "起點數": g.size()})
    return out.reindex([h for h in ("short", "mid", "long") if h in out.index])
