"""v3.13：環境判定卡、站上各均線比例、底線、估值溫度、個股標籤、營收反應分類；App 與文件。"""
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import env as E
from kosto import review as rv
from kosto.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_env_card_states():
    d = pd.bdate_range("2026-01-01", periods=200)
    up = pd.Series(np.linspace(100, 200, 200), index=d)
    c = E.env_card(up)
    assert c["short"] == 3 and c["short_zh"] == "順風" and c["long"] == 3 and c["long_zh"] == "順風"
    dn = pd.Series(np.linspace(200, 100, 200), index=d)
    c2 = E.env_card(dn)
    assert c2["short_zh"] == "逆風" and c2["long_zh"] == "逆風" and c2["short_days"] > 50
    s = pd.Series(np.r_[np.linspace(100, 200, 180), np.linspace(200, 185, 20)], index=d)   # 長多後短拉回
    c3 = E.env_card(s)
    assert c3["short"] == 0 and c3["long"] >= 1
    n = E.card_counts(up, E.SHORT)
    assert n.iloc[:3].isna().all() and n.iloc[-1] == 3 and set(n.dropna().unique()) <= {0, 1, 2, 3}
    assert E.env_card(up.iloc[:10]) == {} and E.state_of(np.nan)[0] == -1


def test_breadth_levels_valuation():
    d = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"a": np.linspace(10, 20, 300), "b": np.linspace(20, 10, 300), "c": 15.0}, index=d)
    valid = pd.DataFrame(True, index=d, columns=px.columns)
    br = E.ma_breadth(px, valid, (5, 240))
    assert abs(br["ab5"].iloc[-1] - 1 / 3) < 1e-9 and np.isnan(br["ab240"].iloc[100]) and abs(br["ab240"].iloc[-1] - 1 / 3) < 1e-9
    tx = pd.Series(np.linspace(100, 130, 300), index=d)
    lv = E.key_levels(tx)
    assert [x["name"][:2] for x in lv[:4]] == ["月線", "季線", "半年", "年線"] and all(x["above"] for x in lv[:4])
    assert abs(lv[-1]["dist"]) < 1e-12                          # 收盤＝近 20 日最高
    pe = pd.Series(np.r_[np.linspace(10, 20, 800), [25.0]], index=pd.bdate_range("2017-01-02", periods=801))
    v = E.valuation(pe)
    assert v["pct"] == 1.0 and v["pe"] == 25.0 and v["max"] == 25.0 and len(v["hist"]) >= 36
    assert np.isnan(E.valuation(pe.iloc[-100:])["pct"])          # 少於 3 年不算百分位


def test_stock_tags_and_reaction():
    d = pd.bdate_range("2026-01-01", periods=80)
    c = pd.Series(np.r_[np.full(70, 50.0), np.linspace(50, 58, 10)], index=d)
    v = pd.Series(np.r_[np.full(70, 1000.0), np.full(10, 3000.0)], index=d)
    fo = pd.Series(np.r_[np.zeros(75), np.full(5, 200.0)], index=d)
    tags = E.stock_tags(c, v, fo, fo)
    assert tags == ["Y1", "Y2", "Y3"]
    assert E.stock_tags(c.iloc[:30], v, fo, fo) == []
    hot = pd.Series(np.r_[np.full(70, 30.0), np.linspace(30, 58, 10)], index=d)      # 60 日漲超過 30% → 不算「剛起漲」
    assert "Y3" not in E.stock_tags(hot, v, fo, fo)
    assert E.revenue_reaction(0.9, 0.8) == "R_UP" and E.revenue_reaction(0.9, 0.1) == "R_DN"
    assert E.revenue_reaction(0.1, 0.9) == "R_RB" and E.revenue_reaction(0.5, 0.5) is None and E.revenue_reaction(np.nan, 0.9) is None


def test_market_series_breadth_columns():
    from kosto.bundle import MKT_COLS
    assert MKT_COLS[-4:] == ["ab5", "ab10", "ab120", "ab240"]
    src = (ROOT / "kosto" / "market.py").read_text(encoding="utf-8")
    assert "ma_breadth(px, valid, (5, 10, 120, 240))" in src
    assert 'meta["valuation"]' in (ROOT / "kosto" / "bundle.py").read_text(encoding="utf-8")


def test_v313_results_docs_and_app():
    assert all(x[-1] == "❌" for x in rv.V313_T4) and {x[0] for x in rv.V313_T4} == {"Y1", "Y2", "Y3", "Y4"}
    t1 = {(a, b): x for a, b, *x in [(r[0], r[1], *r[2:]) for r in rv.V313_T1]}
    assert abs(t1[("短線", "逆風")][1] - t1[("短線", "順風")][1]) < 1.0          # 短線卡不能預測 20 日報酬
    assert [x[0] for x in rv.V313_POS] == ["P0", "P1", "P5", "P6"] and all("❌" in x[-1] for x in rv.V313_POS[2:])
    t5 = {x[0]: x for x in rv.V313_T5}
    assert t5["R_UP"][2] > 1 and t5["R_UP"][4] > 1 and abs(t5["R_DN"][2]) < 1 and abs(t5["R_RB"][2]) < 1
    assert len(rv.V313_COMPARE) >= 7 and [x for x in rv.IDEAS if x[0] == "v3.13"]
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def env_cards", "def env_panel_html", "def env_tab_html", "def stock_tags_html", "def v313_html", "🌡️ 環境與估值"):
        assert fn in app
    assert "v313_html()" in app and "env_panel_html()" in app and "stock_tags_html(t, j)" in app
    doc = (ROOT / "docs" / "v3.13_protocol.md").read_text(encoding="utf-8")
    assert "## 四、結果" in doc and "二之二" in doc
    cfg = load_config()
    assert cfg["research"]["overlay"] == {"f_rev_rel": 0.15}               # 選股模型不變
    assert '__version__ = "3.13.0"' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
