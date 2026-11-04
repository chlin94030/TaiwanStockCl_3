"""v2.1.1 當日分時走勢：平盤取法（Yahoo ^TWII 昨收錯誤的實例）、MIS 即時補點、迷你圖與大圖。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import intraday as I
from kosto import ui

FIX = Path(__file__).parent / "fixtures" / "yahoo_twii_20261104_2m.json"


def _twii():
    return I.parse_chart(json.loads(FIX.read_text()), "^TWII")


def test_parse_chart_uses_fullday_change_not_wrong_previous_close():
    d = _twii()
    # Yahoo 的 previousClose／chartPreviousClose 給 44,920.1（錯），實際昨收 43,307.54
    assert abs(d["prev"] - 43307.54) < 0.01
    assert d["last"] == 42948.91 and d["high"] == 43741.54 and d["low"] == 41855.82
    assert d["start"] == pd.Timestamp("2026-11-04 09:00") and d["end"] == pd.Timestamp("2026-11-04 13:30")
    assert len(d["v"]) == 136 and str(d["date"]) == "2026-11-04"
    f = I.frac(d)
    assert f[0] == 0 and f[-1] == 1
    assert I.status(d, pd.Timestamp("2026-11-04 20:00")) == "11/04 收盤"
    assert I.status(d, pd.Timestamp("2026-11-04 10:30")).startswith("盤中")


def test_prev_close_fallbacks():
    assert I.prev_close({"regularMarketPrice": 101.0, "regularMarketChangePercent": 1.0}) == 100.0
    assert I.prev_close({"previousClose": 50.0, "chartPreviousClose": 49.0}) == 50.0
    assert I.prev_close({"fulldayPrice": 10.0, "fulldayChange": -1.0, "previousClose": 99}) == 11.0
    assert np.isnan(I.prev_close({}))
    # 回傳空結果、壞資料
    assert I.parse_chart({"chart": {"result": None}}) is None
    assert I.parse_chart({"chart": {"result": [{"meta": {}, "timestamp": [1, 2], "indicators": {"quote": [{"close": [None, None]}]}}]}}) is None


def test_merge_live_appends_point_and_uses_official_prev():
    d = _twii()
    q = {"price": 42990.0, "prev": 43307.54, "open": 43617.0, "high": 43760.0, "low": 41855.82,
         "date": "20261104", "time": "13:30:00"}
    d2 = I.merge_live(d, q)
    assert len(d2["v"]) == 136                        # 13:30 已有資料 → 不重複加點
    assert d2["last"] == 42990.0 and d2["high"] == 43760.0 and d2["prev"] == 43307.54
    # 隔天開盤後 Yahoo 還沒有新資料 → 以開盤價與最新價兩點建立今天的走勢
    q2 = dict(q, date="20261105", time="09:12:05", price=43100.0, prev=42948.91, open=43050.0, high=43120.0, low=43010.0)
    d3 = I.merge_live(d, q2)
    assert str(d3["date"]) == "2026-11-05" and len(d3["v"]) == 2 and d3["prev"] == 42948.91
    assert d3["start"] == pd.Timestamp("2026-11-05 09:00") and d3["high"] == 43120.0
    # 舊的 MIS 資料不覆蓋較新的分時
    assert I.merge_live(d3, dict(q, date="20261104")) is d3
    assert I.merge_live(None, None) is None


def test_yahoo_symbol_and_thin():
    assert I.yahoo_symbol("2330", "TWSE") == "2330.TW" and I.yahoo_symbol("3498", "TPEX") == "3498.TWO"
    x = np.linspace(0, 1, 1000)
    y = np.sin(x * 9)
    xs, ys = I.thin(x, y, 100)
    assert len(ys) <= 104 and ys.max() == y.max() and ys.min() == y.min() and xs[-1] == 1


def test_base_chart_marks_baseline_and_colors():
    d = _twii()
    html = ui.intraday_mini(d, title="11/04 收盤")
    assert "stroke-dasharray" in html                 # 平盤虛線
    assert ui.UP in html and ui.DOWN in html           # 上方紅、下方綠
    assert "平盤" in html and "高 <b>" in html and "低 <b>" in html
    assert ui.intraday_mini(None) == "" and ui.intraday_mini({"v": [1.0]}) == ""
    tr = ui.trend_mini([10, 11, 12, 11.5], period="近1月")
    assert "+15.0%" in tr and "期初" in tr
    assert ui.trend_mini([1, 2]) == ""
    # 對稱縮放：平盤在正中間
    h = ui.base_chart([0, 0.5, 1], [100, 103, 101], 100, h=60, labels={"base": "x"})
    assert "top:50.0%" in h
    assert len(ui.session_ticks(d)) == 4              # 10、11、12、13 點


def test_intraday_figure_builds():
    import pytest
    pytest.importorskip("plotly")
    from kosto.charts import _cross, intraday_figure
    d = _twii()
    fig = intraday_figure(d, height=300, dark=True, volume=False)
    assert fig.layout.yaxis.range[0] < d["low"] and fig.layout.yaxis.range[1] > d["high"]
    mid = (fig.layout.yaxis.range[0] + fig.layout.yaxis.range[1]) / 2
    assert abs(mid - d["prev"]) < 1e-6                 # 平盤置中
    assert any("平盤" in (a.text or "") for a in fig.layout.annotations)
    x, y = _cross(np.array([0, 10, 20], dtype="int64"), np.array([99.0, 101.0, 99.0]), 100.0)
    assert list(y) == [99.0, 100.0, 101.0, 100.0, 99.0] and list(x) == [0, 5, 10, 15, 20]
    # 個股（有成交量）
    t = pd.date_range("2026-11-05 09:00", periods=60, freq="2min")
    v = 100 + np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 0.3, 59)])
    s0, s1 = I.tw_session("2026-11-05")
    d2 = {"t": t, "v": v, "vol": np.full(60, 5000.0), "prev": 100.0, "last": v[-1], "high": v.max(), "low": v.min(),
          "open": v[0], "start": s0, "end": s1, "date": s0.date(), "time": t[-1]}
    fig2 = intraday_figure(d2, dark=False, digits=1)
    assert any(tr.type == "bar" for tr in fig2.data)
