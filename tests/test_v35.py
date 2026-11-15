"""v3.5：美股第二輪（標普 500 時點母池、ETF 策略、瀏覽排行、動能觀察名單）＋台股風險調整動能。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kosto.config import load_config
from kosto.factors import REGISTRY
from kosto.us import engine as UE
from kosto.us import model as M
from kosto.us import names as NM
from kosto.us import view as UV

ROOT = Path(__file__).resolve().parents[1]


def test_sp_universe_and_new_models():
    idx = pd.bdate_range("2024-01-02", periods=300)
    tk = list("ABCDEF")
    C = pd.DataFrame(20.0, idx, tk)
    C["F"] = 3.0                                                             # 股價 < 5
    DV = pd.DataFrame([[6, 5, 4, 3, 2, 1]] * 300, index=idx, columns=tk, dtype=float)
    mem = pd.DataFrame(True, idx, tk)
    mem["E"] = False                                                         # 不是成分股
    NT = pd.DataFrame(False, idx, tk)
    NT.iloc[-2, NT.columns.get_loc("D")] = True                              # 近 5 日停牌
    U = M.sp_universe(C, DV, mem, no_trade=NT)
    assert U.iloc[-1].to_dict() == {"A": True, "B": True, "C": True, "D": False, "E": False, "F": False}
    assert not U.iloc[:259].any().any()
    U2 = M.sp_universe(C, DV, mem, top=2)
    assert list(U2.iloc[-1][U2.iloc[-1]].index) == ["A", "B"]
    assert {"SPMO", "MINVOL", "QTREND"} <= set(M.MODELS) and "MOMVOL" in M.FACTOR_ZH
    rng = np.random.default_rng(1)
    C3 = pd.DataFrame(np.cumprod(1 + rng.normal(0.001, 0.02, (300, 3)), axis=0), idx, list("xyz"))
    mv = M.factors(C3, None, ["MOMVOL", "MOM12_1"])
    r = C3.pct_change()
    assert mv["MOMVOL"].iloc[-1, 0] == pytest.approx(mv["MOM12_1"].iloc[-1, 0] / r.iloc[-252:, 0].std(), rel=1e-6)


def test_research_round2_and_views():
    R = UE.research()
    assert R["round"] == 2 and R["benchmark"].startswith("SPY") and not any(R["passed"].values())
    assert {h: tuple(v) for h, v in R["reference"].items()} == UE.REFERENCE and tuple(R["watch"]) == UE.WATCH
    g = pd.DataFrame(R["grid"])
    assert len(g) == 4 * len(M.MODELS) * 3 * 5 and set(g.uni) == {"SP100", "SP200", "SP300", "SP500"}
    dev = g[g.period == "dev"]
    assert (dev.excess < 0.03).all()                                          # 照實記錄：沒有一組達到 +3pp
    for h, (m, top) in UE.REFERENCE.items():                                  # 參考名單＝開發期第一名
        x = dev[dev.h == h].copy()
        x["rk"] = (x.excess.rank(ascending=False) + x.ir.rank(ascending=False)) / 2
        best = x.sort_values(["rk", "excess"], ascending=[True, False]).iloc[0]
        assert (best.model, best.uni) == (m, f"SP{top}")
    e = pd.DataFrame(R["etf"])
    assert {"R1", "R2", "R3", "R4", "R5"} <= set(e.strat)
    for h in ("short", "mid", "long"):
        html = UV.validation_html(R, h)
        assert html.count("⭐") == 1
    assert "200 日線" in UV.etf_html(R) and UV.gap_html(R) and UV.round1_html(R)


def _bundle():
    info = UE.universe_table(fetch=False, log=None)
    tk = list(info.index[:60]) + ["ASML", "TSM", "SPY"]
    rng = np.random.default_rng(9)
    n = 320
    idx = pd.bdate_range("2025-08-01", periods=n)
    C = pd.DataFrame(80 * np.cumprod(1 + rng.normal(0.0005, 0.015, (n, len(tk))), axis=0), idx, tk)
    P = {"C": C, "O": C, "H": C, "L": C, "V": pd.DataFrame(1e7, idx, tk), "RAW": C, "fail": []}
    return UE.build(P, info=info, log=None, now=pd.Timestamp("2026-11-14 12:00", tz="UTC"))


def test_masks_sets_watch_and_live():
    B = _bundle()
    assert B["version"] == 2 and set(B["umask"]) == {"all", "sp500", "100", "200", "300"}
    assert not B["umask"]["sp500"]["ASML"] and B["umask"]["all"]["ASML"]           # 非標普股只在瀏覽母池
    assert all(not B["umask"]["sp500"][t] for t in ("ASML", "TSM"))
    for h in ("short", "long", "watch"):
        assert set(B["picks"][h]["names"]) <= set(B["umask"]["sp500"][B["umask"]["sp500"]].index)
    sox = UE.set_rank(B, "sox", "SPMO", 15)
    assert set(sox.index) <= set(NM.SOX_NOW) and "TSM" in sox.index and list(sox["rank"]) == list(range(1, len(sox) + 1))
    assert UE.ref(B, "short") == ("MINVOL", 100) and UE.ref({"reference": {"short": "MOM"}}, "short") == ("MOM", 0)
    lp = UE.live_picks(B, {})
    assert set(lp) >= {"short", "mid", "long", "watch", "scores"} and lp["watch"]["new"] == []
    assert "那斯達克" in UV.SET_ZH["ndx"] and "標普 500 前 100 大" in UV.list_title(B, "short")
    html = UV.set_table(B, "ndx", "BAL_REV")
    assert "kq-schip" in html
    assert UV.g1_flag("short", -0.05).count("先看消息") == 1 and "2～3pp" not in UV.g1_flag("short", -0.05)


def test_tw_risk_adjusted_momentum_not_adopted():
    assert "t_rmom240" in REGISTRY and REGISTRY["t_rmom240"].direction == 1
    cfg = load_config()
    assert cfg["research"]["overlay"] == {"f_rev_rel": 0.15}                 # v3.5 測試後不採用：台股模型與 v3.4 相同
    assert "t_rmom240" in cfg["research"]["inactive_factors"]                 # 照常計算、不讓模型自己挑
    from kosto import factors as FX
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(3)
    ret = pd.DataFrame(rng.normal(0.001, 0.02, (300, 2)), idx, ["1101", "2330"])

    class Ctx:
        P = {"ret": ret, "tradable": pd.DataFrame(True, idx, ret.columns)}

        def get(self, k, fn):
            return fn()
    x = FX._t_rmom240(Ctx())
    lc = np.log1p(ret).cumsum()
    want = (np.exp(lc.iloc[-21, 0] - lc.iloc[-241, 0]) - 1) / ret.iloc[-240:, 0].std()
    assert x.iloc[-1, 0] == pytest.approx(want, rel=1e-6)
    from kosto import review as rv
    t1 = [x for x in rv.V35_TESTS if x[0].startswith("T1")][0]
    assert t1[-1].startswith("未採用") and t1[6] == "−27.5"


def test_docs_and_app_v35():
    doc = (ROOT / "docs" / "v3.5_protocol.md").read_text(encoding="utf-8")
    assert "預先宣告" in doc and "## 三、結果" in doc and "T1" in doc and "SPY" in doc
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for s in ("def us_watch_section", "def us_browse_section", "UV.etf_html", "rv.V35_TESTS"):
        assert s in app
    from kosto import review as rv
    assert len(rv.V35_TESTS) == 4 and not any(x[-1].startswith("✅") for x in rv.V35_TESTS)
    assert "二之二" in doc and "缺值" in doc


def test_pool_view_treats_nan_as_missing():
    from kosto.optimize import pool_view
    U = np.array([[[1.0], [np.nan], [3.0], [2.0], [0.0]]], dtype=np.float32)
    z = pool_view(U, np.ones((1, 5), bool), np.ones((1, 5), bool)).Z[0].ravel()
    assert z[1] == 0 and z[4] == 0 and z[2] > z[3] > z[0]                    # NaN 與 0 都是中性，不會排在最前面
