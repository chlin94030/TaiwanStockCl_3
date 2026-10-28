"""v1.7：國際市場資料（Yahoo 解析、時間對齊不偷看未來）、新研究選項（減碼、收盤進場、波動倒數）、連動統計。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto.backtest import Spec, book_vol, run
from kosto.panel import align_world
from kosto.sources import world as W
from tests.test_engine import _toy_bundle


def _payload(dates, closes, off=-14400):
    ts = [int((pd.Timestamp(d) + pd.Timedelta(hours=16)).timestamp()) - off for d in dates]   # 16:00 local
    return {"chart": {"result": [{"meta": {"gmtoffset": off, "regularMarketPrice": closes[-1], "regularMarketTime": ts[-1]},
                                  "timestamp": ts, "indicators": {"quote": [{"close": closes}]}}]}}


def test_parse_chart_uses_exchange_local_date_and_drops_nulls():
    f = W.parse_chart(_payload(["2026-10-21", "2026-10-22", "2026-10-23"], [100.0, None, 102.0]), "SOX")
    assert f["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-21", "2026-10-23"]
    assert f["ticker"].unique().tolist() == ["SOX"] and f["close"].tolist() == [100.0, 102.0]


def test_align_world_us_lagged_one_day_asia_same_day():
    tw = pd.DatetimeIndex(["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26"])
    w = pd.DataFrame({"date": pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"] * 2),
                      "ticker": ["SOX"] * 5 + ["KS11"] * 5, "close": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50.0]})
    A = align_world(w, tw)
    # 台股 10/21 收盤後：美股只知道 10/20（10/21 晚上的美股還沒收盤）；韓股知道 10/21
    assert A.loc["2026-10-21", "SOX"] == 2 and A.loc["2026-10-21", "KS11"] == 30
    # 週一 10/26：美股最後是週五 10/23
    assert A.loc["2026-10-26", "SOX"] == 5 and A.loc["2026-10-26", "KS11"] == 50
    stale = align_world(w, pd.DatetimeIndex(["2026-11-30"]))
    assert stale.isna().all().all()                                # 超過 10 天沒有新資料 → 不用


def test_quote_parses_and_skips_failures(monkeypatch):
    class Resp:
        def __init__(self, p):
            self.p = p

        def raise_for_status(self):
            return None

        def json(self):
            return self.p

    def fake_get(url, params=None, headers=None, timeout=None):
        if "SOX" in url:
            return Resp(_payload(["2026-10-21", "2026-10-22"], [100.0, 103.0]))
        raise W.requests.ConnectionError("blocked")
    monkeypatch.setattr(W.requests, "get", fake_get)
    q = W.quote(["SOX", "VIX"])
    assert q["key"].tolist() == ["SOX"] and q["chg"].iloc[0] == pytest.approx(0.03) and len(q["spark"].iloc[0]) == 2


def test_fetch_gives_up_fast_when_host_unreachable(monkeypatch):
    calls = []

    class S:
        headers = {}

        def get(self, *a, **k):
            calls.append(1)
            raise W.requests.ConnectionError("proxy refused")
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    f = W.fetch(session=S(), log=lambda *a: None)
    assert f.empty and len(calls) == 2


def test_exposure_scale_moves_stock_weight_to_cash():
    B, view, Wt = _toy_bundle()
    full = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0))
    half = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, expo_scale=np.full(len(B.sig_pos), 0.5)))
    assert half.tranches["exposure"].iloc[0] == pytest.approx(0.5)
    r_full = full.tranches["ret"].dropna().to_numpy()
    r_half = half.tranches["ret"].dropna().to_numpy()
    assert np.allclose(r_half[0], r_full[0] / 2, rtol=1e-6)       # first tranche: half invested, rest cash (0%)


def test_entry_at_close_skips_open_to_close_of_new_buys():
    B, view, Wt = _toy_bundle()
    S = len(B.sig_pos)
    a = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0))
    b = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, entry_close=np.ones(S, bool)))
    R, OC = B.a["R"].astype(float), B.a["OC"].astype(float)
    # first tranche is a new buy: open→close of day 1 is skipped; later tranches keep the same stock (no effect)
    assert b.nav.iloc[-1] == pytest.approx(a.nav.iloc[-1] / (1 + OC[1, 0]), rel=1e-6)


def test_invvol_weighting_prefers_calm_stock_and_book_vol():
    B, view, Wt = _toy_bundle()
    S = len(B.sig_pos)
    B.a["VOL60"] = np.tile(np.array([[0.01, 0.04]], np.float32), (S, 1))
    res = run(B, view, Wt, Spec(H=5, step=5, top_k=2, buy_cost=0, sell_cost=0, weighting="invvol"))
    assert res.tranches["n_filled"].iloc[0] == 2
    sv = np.vstack([np.linspace(1, 1.2, 40), np.full(40, np.nan)])
    assert np.isfinite(book_vol(sv, 39)) and np.isnan(book_vol(sv, 5))


def test_linkage_tables_from_bundle():
    from kosto import linkage
    rng = np.random.default_rng(3)
    D, N = 400, 60
    days = pd.bdate_range("2022-01-03", periods=D)
    sox = 1000 * np.cumprod(1 + rng.normal(0, 0.02, D))
    us = np.r_[np.nan, sox[:-1]]                                   # aligned: value known at T = US close of T-1
    us_ret = np.r_[np.nan, us[1:] / us[:-1] - 1]
    R = (0.3 * np.nan_to_num(us_ret)[:, None] + rng.normal(0, 0.01, (D, N))).astype(np.float32)
    OC = (R * 0.2).astype(np.float32)
    sig = np.arange(0, D, 5)
    B = SimpleNamespace(a={"R": R, "OC": OC, "WORLD": np.column_stack([us, us * 2]).astype(np.float32),
                           "RANKS": np.tile(np.arange(1, N + 1, dtype=np.int16), (1, len(sig), 1))},
                        meta={"world_keys": ["SOX", "KS11"], "rules": ["MCAP"], "groups": ["電子上游-半導體"] * N},
                        days=days, sig_pos=sig, tickers=[str(i) for i in range(N)])
    out = linkage.summary(B)
    ov = out["overnight"]
    assert len(ov) == 5 and ov["天數"].sum() > 300
    assert ov.attrs["beta"]["gap"] > 0.15                          # most of the US move shows up in the opening gap
    assert out["yearly"]["相關_費半(前一夜)"].min() > 0.3
    assert out["state"]["light"] in ("green", "amber", "red")
