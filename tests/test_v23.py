"""v2.3 限空令／空單／權重平衡：研究常數的一致性與即時小工具。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kosto import review as rv


def _bundle():
    """3 檔、70 日：A 融券大且最近 10 日大增；B 融券小；C 融券 0。"""
    D, n = 70, 3
    V = np.full((D, n), 1000.0)                       # 張
    C = np.tile(np.linspace(100, 90, D)[:, None], (1, n)).astype(float)
    C[:, 2] = np.linspace(50, 60, D)
    SB = np.zeros((D, n))
    SB[:, 0] = 2000.0
    SB[-10:, 0] = np.linspace(2100, 3000, 10)         # 10 日內 +1000 張（= 均量 100%）
    SB[:, 1] = 600.0
    MB = np.full((D, n), 4000.0)
    MB[:, 1] = 6000.0
    a = {"Q_SB": SB.astype(np.float32), "Q_MB": MB.astype(np.float32), "Q_V": V.astype(np.float32), "Q_C": C.astype(np.float32)}
    return SimpleNamespace(a=a)


def test_short_snapshot_ratio_and_increase():
    B = _bundle()
    s = rv.short_snapshot(B, [0, 1, 2])
    assert s[0]["ratio"] == pytest.approx(3000 / 4000)
    assert s[0]["inc10"] == pytest.approx((3000 - 2000) / 1000, rel=1e-6)
    assert s[1]["ratio"] == pytest.approx(0.1) and s[1]["inc10"] == pytest.approx(0.0)
    assert s[2]["ratio"] == 0.0
    assert rv.short_snapshot(SimpleNamespace(a={}), [0]) == {}


def test_short_crowded_filters_and_sorts():
    df = rv.short_crowded(_bundle(), n=5, top=3, min_sbal=500)
    assert list(df["j"]) == [0, 1]                    # C 沒有融券 → 排除；A 券資比最高排第一
    assert df["r20"].iloc[0] < 0 and df["d60"].iloc[0] < 0
    assert len(rv.short_crowded(_bundle(), min_sbal=1000)) == 1


def test_family_shares():
    fs = rv.family_shares({"t_rs120": 0.2, "t_lowidio": 0.1, "f_ep": -0.3, "c_x": 0.4}, ["t_rs120", "t_lowidio", "f_ep", "c_x"],
                          ["T", "T", "F", "C"])
    assert fs["T"] == pytest.approx(0.3) and fs["F"] == pytest.approx(0.3) and fs["C"] == pytest.approx(0.4) and fs["K"] == 0
    assert fs["mom"] == pytest.approx(0.2) and fs["top"] == "c_x"
    assert rv.family_shares({}, [], []) == {}


def test_research_constants_consistent():
    data = [e for e in rv.BAN_HISTORY if e["src"] == "data"]
    assert [e["date"] for e in data] == ["2020-03-20", "2022-10-24", "2025-04-07"]
    for e in data:
        assert {"dd", "day0", "r5", "r20", "r60", "r250", "min60", "low"} <= set(e)
        assert e["min60"] <= 0 and e["r60"] > 0
    assert rv.BAN_2026["same_as"] == "2025-04-07" and len(rv.BAN_2026["measures"]) == 5
    assert all(len(x) == 4 and all(len(e) == 4 for e in x[1:]) for x in rv.BAN_XS)
    assert sum(n for _, n, *_ in rv.LEAD_BUCKETS) <= 1870
    assert len(rv.V23_TESTS) == 4 and all("未通過" in t[-1] for t in rv.V23_TESTS)
    assert all(len(x) == 7 for x in rv.IDEAS)
    # v2.0 以後使用者提問引發的想法都沒有改變主模型
    assert not any(x[2] == "使用者提問" and x[5].startswith("✅") for x in rv.IDEAS)


def test_docs_and_app_wired():
    root = Path(__file__).resolve().parents[1]
    proto = (root / "docs" / "v2.3_protocol.md").read_text(encoding="utf-8")
    assert "## 結果" in proto and "主模型不調整" in proto
    assert (root / "docs" / "v2.3_model_review.md").is_file()
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ("def ban_html", "def ban_xs_html", "def lead_section", "def balance_html", "short_crowded_df", "rv.V23_TESTS"):
        assert s in app


# ------------------------------------------------------------------ v2.3.1 漲跌停鎖住、現價來源一致
def test_parse_mis_limit_locked_and_market_orders():
    from kosto.quotes import parse_mis
    msg = {"msgArray": [
        # 漲停鎖住：沒有成交瞬間（z="-"）、委買第一檔是市價（0）、沒有委賣
        {"c": "2344", "n": "華邦電", "z": "-", "y": "120.5000", "b": "0.0000_132.5000_132.0000_", "a": "-", "u": "132.5000",
         "w": "108.5000", "h": "132.5000", "l": "124.0000", "o": "125.0000", "v": "80000", "t": "13:19:15", "d": "20261109"},
        # 只有市價委買
        {"c": "1303", "n": "南亞", "z": "-", "y": "239.5000", "b": "0.0000_", "a": "-", "u": "263.0000", "w": "216.0000"},
        # 跌停鎖住
        {"c": "9999", "n": "測試", "z": "-", "y": "100.0000", "b": "-", "a": "0.0000_90.0000_", "u": "110.0000", "w": "90.0000"},
        # 有成交且剛好在漲停
        {"c": "8888", "n": "測試2", "z": "55.0000", "y": "50.0000", "b": "55.0000_", "a": "-", "u": "55.0000", "w": "45.0000"},
    ]}
    q = parse_mis(msg).set_index("ticker")
    assert q.loc["2344", "price"] == 132.5 and q.loc["2344", "src"] == "limit_up" and q.loc["2344", "lock"] == "up"
    assert q.loc["2344", "bid"] == 132.5 and q.loc["2344", "chg"] == pytest.approx(132.5 / 120.5 - 1)
    assert q.loc["1303", "price"] == 263.0 and q.loc["1303", "lock"] == "up"
    assert q.loc["9999", "price"] == 90.0 and q.loc["9999", "src"] == "limit_down" and q.loc["9999", "lock"] == "down"
    assert q.loc["8888", "src"] == "trade" and q.loc["8888", "lock"] == "up"


def test_limit_prices_and_exact_ticks():
    from kosto.ta import limit_prices, round_tick
    assert limit_prices(120.5) == (132.5, 108.5)
    assert limit_prices(239.5) == (263.0, 216.0)          # 263.45 往下取 0.5 → 263.0
    assert limit_prices(100.0) == (110.0, 90.0)            # 剛好在有效價位：不可被推到下一檔
    assert round_tick(110.0, "down") == 110.0 and round_tick(90.0, "up") == 90.0
    assert round_tick(105.09, "down") == 105.0 and round_tick(118.7, "up") == 119.0


def test_best_quote_fallbacks():
    import pandas as pd
    from kosto import intraday as I
    rd = pd.Timestamp("2026-11-06")
    t = pd.date_range("2026-11-09 09:00", periods=5, freq="2min")
    d = {"t": t, "v": np.array([125, 130, 132.5, 132.5, 132.5]), "vol": np.array([1e6, 2e6, 3e5, 1e5, 1e5]), "prev": 120.5,
         "last": 132.5, "open": 125.0, "high": 132.5, "low": 124.0, "date": pd.Timestamp("2026-11-09").date(), "time": t[-1],
         "start": t[0], "end": t[0] + pd.Timedelta(hours=4.5)}
    # 1. 證交所有價格
    bq = I.best_quote({"price": 132.5, "prev": 120.5, "src": "limit_up", "lock": "up", "open": 125, "high": 132.5, "low": 124,
                       "volume": 80000}, d, 120.5, -0.0549, rd)
    assert bq["live"] and bq["src"] == "limit_up" and bq["lock"] == "up" and bq["chg"] == pytest.approx(0.0996, abs=1e-3)
    # 2. 證交所沒有價格 → 今天的分時（不可退回 11/06 的 120.5 ▼5.49%）
    bq = I.best_quote({"price": np.nan, "prev": 120.5}, d, 120.5, -0.0549, rd)
    assert bq["live"] and bq["src"] == "yahoo" and bq["px"] == 132.5 and bq["chg"] > 0.09 and bq["lock"] == "up"
    assert bq["volume"] == pytest.approx(3500)
    # 3. 都沒有 → 研究資料收盤並標日期
    bq = I.best_quote(None, None, 120.5, -0.0549, rd)
    assert not bq["live"] and bq["px"] == 120.5 and bq["label"] == "11/06 收盤"
    # 分時是研究資料那天（不是今天）→ 不當成今天的價格
    d2 = dict(d, date=rd.date())
    assert I.best_quote(None, d2, 120.5, -0.0549, rd)["src"] == "close"


# ------------------------------------------------------------------ v2.3.2 以前一日收盤（今日參考價）為基準的價位
def test_plan_rebase_to_reference_price():
    from kosto.plan import Plan, rebase
    p = Plan(100.0, 97.0, 101.0, 103.0, 92.0, 95.0, 106.0, 0.02, 55.0, 0.03, False)
    assert rebase(p, 100.0) is p and rebase(p, float("nan")) is p and rebase(None, 99) is None
    r = rebase(p, 96.0)                                # 除權息 4 元：參考價 96
    assert r.close == 96.0 and r.base_from == 100.0
    assert (r.buy_low, r.buy_high, r.chase, r.alert) == (93.1, 96.9, 98.8, 88.3)   # 等比例換算後往下對齊 0.1
    assert r.buy_low <= r.buy_high <= r.chase and r.alert < r.buy_low


def test_best_quote_ignores_quotes_not_newer_than_research():
    import pandas as pd
    from kosto import intraday as I
    rd = pd.Timestamp("2026-11-09")
    q = {"price": 132.5, "prev": 120.5, "src": "trade", "date": "20261109"}
    bq = I.best_quote(q, None, 132.5, 0.0996, rd)      # 晚上研究已更新到 11/09：證交所資料也是 11/09 → 不是新的
    assert not bq["live"] and bq["label"] == "11/09 收盤" and bq["chg"] == pytest.approx(0.0996)
    bq = I.best_quote(dict(q, date="20261110", prev=132.5, price=135.0), None, 132.5, 0.0996, rd)
    assert bq["live"] and bq["chg"] == pytest.approx(135 / 132.5 - 1)


def test_merge_live_does_not_draw_bid_only_new_day():
    import pandas as pd
    from kosto import intraday as I
    q = {"price": 50.0, "prev": 49.0, "src": "bid", "date": "20261110", "time": "08:45:00"}
    assert I.merge_live(None, q, pd.Timestamp("2026-11-10 08:45")) is None
    q2 = dict(q, src="trade", time="09:05:00")
    d = I.merge_live(None, q2, pd.Timestamp("2026-11-10 09:05"))
    assert d is not None and d["last"] == 50.0 and d["prev"] == 49.0


def test_parse_mis_trade_object_fallback():
    from kosto.quotes import parse_mis
    q = parse_mis({"msgArray": [{"c": "6213", "z": "-", "y": "496.0000", "b": "-", "a": "-", "u": "545.0000", "w": "446.5000",
                                 "trade": {"t": "13:30:00", "v": 179, "z": "531.0000"}}]}).set_index("ticker")
    assert q.loc["6213", "price"] == 531.0 and q.loc["6213", "src"] == "trade"
