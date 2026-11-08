"""v2.2 急跌應變：急跌事件統計、目前觸發條件、盤中止跌觀察、動能保護（研究用，預設關閉）、儀表板燈號。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import market as M
from kosto import ui


def _crash_series():
    d = pd.bdate_range("2017-01-02", periods=900)
    x = np.ones(900) * 100.0
    x[:400] = 100 * np.exp(np.linspace(0, 0.4, 400))           # 上漲
    top = x[399]
    x[400:410] = top * np.linspace(1.0, 0.82, 10)               # 10 天跌 18%（連跌、回落 15%）
    x[410:900] = top * 0.82 * np.exp(np.linspace(0, 0.3, 490))  # 之後回升
    sox = pd.Series(1000.0, index=d)
    sox.iloc[600] = 1000.0
    sox.iloc[601:] = 900.0                                      # 費半單日 −10%
    return pd.Series(x, index=d), sox


def test_crash_stats_detects_events_and_summaries():
    tx, sox = _crash_series()
    cs = M.crash_stats(tx, sox)
    types = {e["type"] for e in cs["events"]}
    assert {"E1", "E2", "E3"} <= types
    e1 = next(e for e in cs["events"] if e["type"] == "E1")
    assert e1["dd250"] <= -0.15 and e1["r60"] > 0                # 之後回升
    for k in ("E1", "E2", "E3"):
        sm = cs["summary"][k]
        assert sm["n"] >= 1 and {"r20", "r60", "min20", "min60", "low_hits", "lump60", "staged60"} <= set(sm)
    assert cs["current"]["dd250"] > -0.05 and cs["current"]["streak"] == 0
    assert M.crash_stats(tx.iloc[:100]) == {}


def test_crash_flags_and_fallback():
    on = M.crash_flags({"dd250": -0.166, "sox_day": -0.123, "streak": 6, "streak_ret": -0.116})
    assert [f[2] for f in on] == [True, True, True]
    off = M.crash_flags({"dd250": -0.05, "sox_day": 0.01, "streak": 2, "streak_ret": -0.03})
    assert not any(f[2] for f in off)
    assert M.crash_flags({}) == []
    fb = M.CRASH_FALLBACK["summary"]
    assert fb["E2"]["low_hits"] <= fb["E2"]["low_n"] and fb["E1"]["n"] == 6


def test_intraday_checks():
    tx = {"last": 42000.0, "low": 41000.0, "open": 41500.0, "prev": 43000.0}
    nq = {"last": 101.0, "prev": 100.0}
    ts = {"last": 1000.0, "low": 990.0}
    c = M.intraday_checks(tx, nq, ts)
    assert [x[2] for x in c] == [True, True, True, False]
    assert M.intraday_checks(None) == []


def test_momentum_guard_scales_weights_only_in_panic():
    from kosto.research import Research
    r = np.zeros(300)
    r[200:220] = -0.01                                           # 20 天跌約 18% → 急跌狀態
    r[225:290] = 0.0035                                          # 之後漲回高點附近 → 解除
    B = SimpleNamespace(a={"BENCH": r.astype(np.float32)}, sig_pos=np.array([100, 215, 290]),
                        factors=["t_rs120", "t_mom120x20", "t_high252", "f_ep"])
    g = {"enabled": True, "factors": ["t_rs120", "t_mom120x20", "t_high252"], "scale": 0.5, "r20": -0.08, "dd120": 0.12, "hold": 0}
    fake = SimpleNamespace(B=B, cfg={"research": {"momentum_guard": g}}, _sched={})
    fake.panic_days = lambda gg=None: Research.panic_days(fake, gg)
    fake.panic_signals = lambda gg=None: Research.panic_signals(fake, gg)
    ps = fake.panic_signals(g)
    assert list(ps) == [False, True, False]
    w = np.array([0.3, 0.2, 0.1, 0.4], np.float32)
    assert np.allclose(Research._guard(fake, w, 0), w)          # 非急跌：不變
    w2 = Research._guard(fake, w, 1)
    assert abs(np.abs(w2).sum() - 1.0) < 1e-6                    # 權重總和不變
    assert w2[3] / w2[0] == pytest.approx(0.4 / 0.15, rel=1e-5)  # 動能因子減半
    fake.cfg["research"]["momentum_guard"] = dict(g, enabled=False)
    assert np.allclose(Research._guard(fake, w, 1), w)           # 預設關閉


def test_cards_raw_label():
    h = ui.cards([("<i class='kq-dot red'></i>距高點", "-16%", "x")], raw=True)
    assert "<i class='kq-dot red'>" in h
    assert "&lt;i" in ui.cards([("<i>x</i>", "1", "")])


def test_tick_rules_and_price_display():
    """v2.2.1 升降單位：股票 6 級、ETF 2 級；顯示一律對齊到有效價位（不出現 31.83、103.8）。"""
    from kosto.ta import round_tick, tick
    assert [tick(x) for x in (9.9, 10, 49.95, 50, 99.9, 100, 499.5, 500, 999, 1000)] == \
        [0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0, 5.0]
    assert tick(49.0, etf=True) == 0.01 and tick(63.0, etf=True) == 0.05
    assert ui.price(31.83) in ("31.80", "31.85") and ui.price(103.8) == "104.0" and ui.price(103.6) == "103.5"
    assert ui.price(1236) == "1,235" and ui.price(825.5) in ("825", "826") and ui.price(9.876) == "9.88"
    assert ui.price(63.47, etf=True) == "63.45" and ui.price(20.013, etf=True) == "20.01"
    assert round_tick(105.09, "down") == 105.0 and round_tick(31.97, "down") == 31.95 and round_tick(554.28, "down") == 554.0


def test_merge_live_ignores_quotes_that_are_not_trades():
    from kosto import intraday as I
    s0, s1 = I.tw_session("2026-11-06")
    t = pd.date_range(s0, periods=10, freq="2min")
    d = {"t": t, "v": np.linspace(110, 104, 10), "vol": np.ones(10), "prev": 113.0, "last": 104.0, "high": 110.0, "low": 104.0,
         "open": 110.0, "start": s0, "end": s1, "date": s0.date(), "time": t[-1]}
    q = {"price": 103.5, "src": "bid", "prev": 113.0, "date": "20261106", "time": "09:42:00", "high": 110.0, "low": 102.0}
    d2 = I.merge_live(d, q)
    assert len(d2["v"]) == 10 and d2["last"] == 104.0 and d2["low"] == 102.0     # 委買價不接在走勢、不當最新價
    d3 = I.merge_live(d, dict(q, src="trade", price=103.5))
    assert len(d3["v"]) == 11 and d3["last"] == 103.5
    d4 = I.snap_ticks(dict(d, prev=112.99999, last=103.77))
    assert d4["prev"] == 113.0 and d4["last"] == 104.0


def test_responsive_css_for_devices():
    """v2.2.2 電腦／平板／手機：快速切換列、平板欄位換行、觸控加大、寬螢幕。"""
    css = ui.css(True)
    for key in (".kq-qnav", "(min-width: 761px) and (max-width: 1100px)", "flex-wrap:wrap", "(pointer: coarse)",
                "(max-height: 500px) and (orientation: landscape)", "(min-width: 1600px)", "(max-width: 900px)"):
        assert key in css, key
    assert ui.css(False).count("{") == css.count("{")                 # 淺色主題同一套版面
