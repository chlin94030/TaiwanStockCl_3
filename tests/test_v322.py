"""v3.22：盤中即時（產業鏈細族群即時強弱、熱力圖自動更新、消息衝擊即時區、總覽「現在盤勢」）。選股模型不變。"""
import ast
import collections
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import live as LV
from kosto import quotes as QT
from kosto import supply as S

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")

SUB = {"a": {"name": "甲", "layer": "L1", "members": [("1001", "x"), ("1002", "y"), ("1003", "z")]},
       "b": {"name": "乙", "layer": "L2", "members": [("2001", "x"), ("2002", "y")]},
       "c": {"name": "丙", "layer": "L2", "members": [("9999", "x")]}}


def _q():
    return pd.DataFrame({"chg": [0.10, 0.02, -0.01, -0.05, -0.03], "volume": [3000, 1000, 500, 800, 400],
                         "lock": ["up", "", "", "", ""]}, index=["1001", "1002", "1003", "2001", "2002"])


def test_session_frac():
    tz = "Asia/Taipei"
    assert LV.session_frac(pd.Timestamp("2026-11-24 09:00", tz=tz)) == 0.05          # 開盤瞬間不會除以 0
    assert abs(LV.session_frac(pd.Timestamp("2026-11-24 11:15", tz=tz)) - 0.5) < 1e-9
    assert LV.session_frac(pd.Timestamp("2026-11-24 14:00", tz=tz)) == 1.0


def test_breadth_and_fallback():
    b = LV.breadth(_q())
    assert (b["n"], b["up"], b["down"], b["lu"], b["ld"]) == (5, 2, 3, 1, 0)
    assert LV.breadth(_q(), ["1001", "nope"])["n"] == 1
    met = pd.DataFrame({"r1": [0.0995, -0.1, 0.01]}, index=["1", "2", "3"])
    fb = LV.fallback_quotes(met)
    assert list(fb["lock"]) == ["up", "down", ""] and fb["volume"].isna().all()
    assert len(LV.fallback_quotes(None)) == 0


def test_group_live_weights_volume_ratio_leaders():
    met = pd.DataFrame({"amt20": [9.0, 1.0, 0.0, 1.0, 1.0], "r20": [0.1, 0.0, 0.0, -0.1, -0.1]}, index=["1001", "1002", "1003", "2001", "2002"])
    vol20 = pd.Series({"1001": 1000.0, "1002": 1000.0, "1003": 1000.0, "2001": 1000.0, "2002": 1000.0})
    gl = LV.group_live(_q(), SUB, met, vol20, frac=0.5, names={"1001": "龍頭"})
    assert list(gl.index[:2]) == ["a", "b"] and gl.loc["c", "nq"] == 0                       # 沒報價的族群排最後、不會出錯
    assert abs(gl.loc["a", "chg"] - (0.10 * 9 + 0.02 * 1) / 10) < 1e-12                      # 成交金額加權（0 權重的不算）
    assert abs(gl.loc["a", "vr"] - 4500 / (3000 * 0.5)) < 1e-12                               # 量比＝到目前的量 ÷（均量 × 已過時間）
    assert gl.loc["a", "lu"] == 1 and gl.loc["a", "lead"][0][:1] == ("1001",) and gl.loc["a", "lead"][0][2] == "龍頭"
    assert gl.loc["b", "lag"][0][0] == "2001" and gl.loc["b", "lead"] == []
    g0 = LV.group_live(_q(), SUB)                                                             # 沒有成交金額 → 等權
    assert abs(g0.loc["b", "chg"] - (-0.04)) < 1e-12 and not np.isfinite(g0.loc["b", "vr"])


def test_chain_live_html_and_strip():
    gl = LV.group_live(_q(), SUB)
    assert gl["up"].dtype.kind == "f"                                                          # 丙族群沒報價 → 欄位是 float
    h = LV.chain_live_html(gl, 0.01, LV.breadth(_q()), S.LAYER_NAME, "證交所即時 10:30:00")
    assert "最強" in h and "最弱" in h and "上下游各層" in h and "./chain?g=a" in h and "漲停 1" in h and "相對加權" in h
    assert "漲 2／跌 1" in h and "2.0／" not in h                                              # 家數是整數（有空族群時欄位會變成小數）
    assert "目前沒有可用的報價" in LV.chain_live_html(pd.DataFrame(), None, {}, S.LAYER_NAME, "")
    s = LV.strip_html({"px": 36812.0, "chg": 0.0216}, {}, LV.breadth(_q()), gl, ["<span>x</span>"], "收盤")
    assert "現在盤勢" in s and "36,812.00" in s and "5 檔" in s and "最強" in s and "<span>x</span>" in s and "櫃買" not in s


def test_real_subgroups_shape():
    tick = [t for d in S.SUBGROUPS.values() for t, _ in d["members"]]
    q = pd.DataFrame({"chg": np.linspace(-0.05, 0.05, len(set(tick)))}, index=list(dict.fromkeys(tick)))
    gl = LV.group_live(q, S.SUBGROUPS)
    assert len(gl) == len(S.SUBGROUPS) and gl["chg"].notna().all() and set(gl["layer"]) <= set(S.LAYER_NAME)
    h = LV.chain_live_html(gl, 0.0, LV.breadth(q), S.LAYER_NAME, "x")
    assert h.count("<tr") >= len(S.SUBGROUPS)


def test_news_impact():
    groups = [{"group": "電子-A", "beta_idio": 0.30}, {"group": "電子-B", "beta_idio": 0.05}, {"group": "電子-C", "beta_idio": np.nan}]
    out = LV.news_impact(groups, 0.33, 0.03, 1.5)
    assert [r["group"] for r in out] == ["電子-A", "電子-B"] and abs(out[0]["impact"] - 0.3 * (0.33 - 0.045)) < 1e-12
    assert LV.news_impact(groups, None, 0.0, 1.5) == [] and LV.news_impact([], 0.1, 0.0, 1.5) == []
    assert abs(LV.news_impact(groups, 0.10, None, None)[0]["impact"] - 0.03) < 1e-12              # 沒有那指 → 全部當輝達特有


def test_quotes_test_overrides(monkeypatch):
    monkeypatch.setenv("KOSTO_FAKE_NOW", "2026-11-24 10:30")
    assert QT.now_tw().hour == 10 and str(QT.now_tw().tz) == "Asia/Taipei" and QT.market_phase() == "盤中"
    monkeypatch.setenv("KOSTO_FAKE_NOW", "2026-11-28 10:30")
    assert QT.market_phase() == "休市"
    monkeypatch.setenv("KOSTO_FAKE_NOW", "not a date")
    assert QT.now_tw().year >= 2026                                                            # 格式錯誤 → 用真的現在時間
    assert QT.MIS.endswith("/stock")


def test_app_wiring():
    for s in ("def chain_live_section", "def home_live_strip", "def news_live_section", "def live_today", "def _heatmap_body",
              '"⚡ 盤中即時"', "chain_live_section()", "home_live_strip()", "news_live_section()", 'order.insert(1, "📰 消息衝擊")',
              "./insight?tab=news", 'st.query_params.get("tab", "") == "news"', "run_every=30 if auto else None",
              "run_every=60 if auto else None", "run_every=30 if live_on else None"):
        assert s in APP, s
    c = collections.Counter(n.name for n in ast.parse(APP).body if isinstance(n, ast.FunctionDef))
    assert not [k for k, v in c.items() if v > 1]
    assert '__version__ = "3.2' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
    doc = (ROOT / "docs" / "v3.22_notes.md").read_text(encoding="utf-8")
    assert "選股模型不變" in doc and "盤中即時" in doc


def test_live_today_and_chain_data_exec():
    """把 App 的 live_today／chain_live_data 抽出來，用假報價執行（盤中、盤後但研究資料已更新、休市三種情況）。"""
    tree = ast.parse(APP)
    fns = {n.name: ast.get_source_segment(APP, n) for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name in ("live_today", "chain_live_data", "chain_members")}
    tick = list(dict.fromkeys(t for d in S.SUBGROUPS.values() for t, _ in d["members"]))
    met = pd.DataFrame({"r1": 0.01, "amt20": 1.0, "r20": 0.0}, index=tick)
    state = {"phase": "盤中", "date": "20261124"}

    def quotes_for(ts):
        return pd.DataFrame({"chg": 0.02, "volume": 100.0, "lock": "", "date": state["date"], "time": "10:30:00"}, index=list(ts)), ""
    g = {"pd": pd, "np": np, "TIDX": {t: i for i, t in enumerate(tick)}, "NAMES": tick, "RDAY": pd.Timestamp("2026-11-23"), "BID": "x",
         "market_phase": lambda: state["phase"], "quotes_for": quotes_for, "chain_metrics": lambda bid: met,
         "vol20_series": lambda bid: pd.Series(1000.0, index=tick), "now_tw": lambda: pd.Timestamp("2026-11-24 11:15", tz="Asia/Taipei")}
    for k in ("chain_members", "live_today", "chain_live_data"):
        exec(fns[k], g)
    gl, q, label, is_live = g["chain_live_data"]()
    assert is_live and "證交所即時" in label and abs(gl["chg"].dropna() - 0.02).max() < 1e-12 and gl["vr"].notna().any()
    state.update(phase="盤後", date="20261123")                                                # 晚上研究資料已更新到今天 → 用研究資料
    gl, q, label, is_live = g["chain_live_data"]()
    assert not is_live and "11/23 收盤" in label and abs(gl["chg"].dropna() - 0.01).max() < 1e-12
    state.update(phase="休市")
    assert not g["live_today"](tick[:3])[2]
