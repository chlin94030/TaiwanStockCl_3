"""v3.23：贏 0050 候選（權值股母池、0050 增強、台積電錨定、200 日線趨勢）、費氏回撤、知名理論、短中長線重疊。"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import beat0050 as BT
from kosto import review as rv
from kosto import ui

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def test_drift_returns_and_costs():
    R = np.array([[0, 0], [0.10, 0.0], [0.0, 0.10], [0.05, -0.05], [0, 0]], float)
    r = BT.drift_returns(R, [0], [{0: 0.5, 1: 0.5}], 5)
    nav = np.cumprod(1 + r)
    v = np.array([0.5, 0.5]) * np.cumprod(1 + R[1:], axis=0)
    assert np.allclose(nav[1:], v.sum(1)) and r[0] == 0                                  # 權重漂移（不是每天再平衡）
    c = BT._trade_cost(0.001, 0.004)
    assert abs(c({0: 1.0}, {0: 0.5, 1: 0.5}) - (0.5 * 0.001 + 0.5 * 0.004)) < 1e-12
    r2 = BT.drift_returns(R, [0, 2], [{0: 1.0}, {1: 1.0}], 5, c)
    assert abs(r2[1] - (0.10 - 0.001)) < 1e-12 and abs(r2[2] - 0.0) < 1e-12 and abs(r2[3] - (-0.05 - 0.005)) < 1e-12


def test_trend_signal_lag_and_switch_cost():
    days = pd.bdate_range("2020-01-01", periods=400)
    bench = np.r_[np.full(250, 0.002), np.full(150, -0.01)]
    B = SimpleNamespace(a={"BENCH": bench.astype(np.float32)}, days=days, sig_pos=np.arange(0, 400, 5))
    st = BT.trend_state(B)
    raw = st.astype(bool)
    first_down = int(np.argmax(~raw.to_numpy()))
    ss = BT.signal_states(B)
    p = next(int(x) for x in B.sig_pos if x >= first_down)
    eff = int(np.argmax(~ss.to_numpy()))
    assert eff == p + 2                                                                  # 換股日收盤看到 → 隔兩天才生效（不偷看）
    a, b = pd.Series(0.01, index=days), pd.Series(0.0, index=days)
    m = BT.mix_returns([(1.0, 0.0, a), (0.0, 1.0, b)], ss)
    assert abs(m.iloc[eff] - (0.0 - BT.SWITCH_COST)) < 1e-12 and abs(m.iloc[eff - 1] - 0.01) < 1e-12
    m5 = BT.mix_returns([(0.5, 0.5, b), (0.5, 0.0, a), (0.0, 0.5, b)], ss)
    assert abs(m5.iloc[eff] - (-0.5 * BT.SWITCH_COST)) < 1e-12                             # 只換掉一半


def test_tilt_weights_keep_anchor():
    base = np.array([0.5, 0.2, 0.2, 0.1])
    w = BT.tilt_weights(base, np.array([0.0, 3.0, 1.0, 2.0]), np.array([True, False, False, False]))
    assert abs(w[0] - 0.5) < 1e-12 and abs(w.sum() - 1) < 1e-12 and w[1] > w[2]           # 台積電不動、分數高的加碼
    assert np.allclose(BT.tilt_weights(base, np.zeros(4), np.zeros(4, bool), 0.0), base)


def test_fib_scan_outcomes_and_peaks():
    p = np.r_[np.linspace(100, 100, 5), np.linspace(99, 80, 20), np.linspace(81, 92, 12), np.linspace(91, 70, 22),
              np.linspace(72, 101, 30), np.full(5, 101.0)]
    ev, peaks = BT.fib_scan(p, 0.10, horizon=250)
    first = [e for e in ev if e["t"] < 40]
    assert {e["level"] for e in first} >= {0.236, 0.382, 0.5} and all(e["out"] == "down" for e in first)   # 第一次反彈到 0.6 又破底
    assert 0.55 < peaks[0] < 0.62
    later = [e for e in ev if e["t"] > 60]
    assert later and all(e["out"] == "up" for e in later) and {0.618, 0.786} <= {e["level"] for e in later}
    tb = BT.fib_table(ev)
    assert {r["level"] for r in tb} == set(BT.GRID) and all(r["rw"] == r["level"] for r in tb)


def test_fib_cluster_neutral_and_detects():
    rng = np.random.default_rng(0)
    smooth = rng.uniform(0.1, 0.95, 20000)
    c = BT.fib_cluster(smooth)
    assert abs(c["ratio"] - 1) < 0.08 and abs(c["z"]) < 3
    spiky = np.r_[smooth, np.full(400, 0.618), np.full(400, 0.382)]
    c2 = BT.fib_cluster(spiky)
    assert c2["z"] > 5 and c2["ratio"] > 1.1
    rw = []                                                                               # 隨機漫步：沒有費氏效果
    for k in range(150):
        e, pk = BT.fib_scan(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1500))), 0.2)
        rw += pk
    assert abs(BT.fib_cluster(rw)["z"]) < 3


def test_stats_and_gates():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 1500)
    assert BT.nw_t(x) > 2 and BT.boot_p(x) > 0.9 and BT.boot_p(-x) < 0.1
    q = BT.bh_q([0.01, 0.04, 0.5, None])
    assert abs(q[0] - 0.03) < 1e-12 and abs(q[1] - 0.06) < 1e-12 and np.isnan(q[3])
    ev = {"full": {"cagr": 0.30, "b_cagr": 0.25, "mdd": -0.25, "b_mdd": -0.30}, "dev": {"cagr": 0.3, "b_cagr": 0.2, "mdd": -0.2, "b_mdd": -0.3},
          "recent": {"cagr": 0.5, "b_cagr": 0.52, "mdd": -0.16, "b_mdd": -0.15}, "years": [(2021, .3, .2), (2022, -.1, -.2), (2023, .1, .2)],
          "p_boot": 0.95}
    g = BT.gates(ev)
    assert g["g1"] and g["g2"] and g["g3"] and g["g4"] and g["g5"] and g["g6"] is None
    g6 = BT.gates(ev, [{"g1": True, "g2": True}] * 3 + [{"g1": False, "g2": True}] * 2)
    assert g6["g6"] is False and g6["g6_n"] == (3, 5)
    out = BT.finalize({"cands": [{"key": "B2", "gates": {**g, "g7": True}}, {"key": "B4", "gates": {**g, "g7": False}}]})
    assert out["cands"][0]["pass"] and not out["cands"][1]["pass"] and out["cands"][1]["observe"]


def _syn(tmp_path):
    from kosto.bundle import build_bundle
    from kosto.config import load_config
    from kosto.panel import build_panel
    from kosto.research import Research
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    st = Store(tmp_path)
    write_demo(st, 120, 3, start="2017-01-02", end="2022-12-30")
    cfg = load_config(overrides={"data": {"dir": str(tmp_path)}, "research": {"start": "2019-01-01", "holdout_months": 6},
                                 "pools": {"sizes": [60, 100], "rules": ["ADV", "MCAP"]}})
    B = build_bundle(build_panel(st), cfg, log=lambda *a: None)
    R = Research(B, cfg, log=lambda m: None)
    return B, R, R.run_all(["short", "mid", "long"], quick=True, progress=lambda *a: None)


def test_summary_on_synthetic(tmp_path):
    B, R, res = _syn(tmp_path)
    out = BT.summary(R, res)
    assert not out["errors"], out["errors"]
    assert [x["key"] for x in out["cands"]] == [k for k, _ in BT.CANDS] and [x["key"] for x in out["refs"]] == [k for k, _ in BT.REFS]
    r0 = next(x for x in out["refs"] if x["key"] == "R0")["full"]
    assert abs(r0["cagr"] - r0["b_cagr"]) < 1e-9                                         # 0050 對自己差 0
    ov = out["overlap"]["pairs"]
    assert {(p["a"], p["b"]) for p in ov} == {("short", "mid"), ("mid", "long"), ("short", "long")} and all(0 <= p["avg"] <= 1 for p in ov)
    assert out["concentration"]["years"] and out["concentration"]["rep_corr"] > 0.5
    assert {x["key"] for x in out["theories"]} == {k for k, _, _ in BT.THEORIES}
    fb = out["fib"]
    assert fb["stocks"]["events"] > 100 and abs(fb["stocks"]["cluster"]["z"]) < 3.5 and "ic" in fb["factor"]
    assert all("pass" in x and "q" in x for x in out["cands"]) and out["current"]["trend"]["sma"] > 0
    row = BT.phase_row(R, res, 1)
    out2 = BT.attach_phases(out, [row])
    assert all(x["gates"]["g6_n"][1] == 2 for x in out2["cands"])
    g = {"ui": ui, "pd": pd, "np": np, "NAMES": list(B.meta["names"]), "TIDX": {t: i for i, t in enumerate(B.tickers)}, "RES": {"beat0050": out2}}
    for fn in ("_pp", "_beat_verdict", "beat_rows", "fib_text", "beat0050_html", "beat_alloc_html"):
        node = next(n for n in ast.parse(APP).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(ast.get_source_segment(APP, node), g)
    h = g["beat0050_html"]()
    for s in ("1. 候選", "為什麼難贏 0050", "短中長線名單重疊", "費氏回撤", "知名理論", "隨機漫步約"):
        assert s in h, s
    assert "8 個組合" in g["beat_alloc_html"]() or "贏 0050 組合" in g["beat_alloc_html"]()
    g["RES"] = {}
    assert "docs/v3.23_protocol.md" in g["beat0050_html"]() and g["beat_alloc_html"]() == ""


def test_v323_wiring_docs():
    for s in ("def beat0050_html", "def fib_panel_html", "def beat_alloc_html", "beat0050_html()", "fib_panel_html()", "beat_alloc_html()"):
        assert s in APP, s
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert 'res["beat0050"] = beat0050.summary(R, res)' in cli and "beat0050.phase_row(Rk, res, k)" in cli and "attach_phases" in cli
    doc = (ROOT / "docs" / "v3.23_protocol.md").read_text(encoding="utf-8")
    for s in ("看任何結果之前", "B6", "BH-FDR", "隨機漫步", "T7", "照實註明"):
        assert s in doc, s
    assert any(x[0] == "v3.23" for x in rv.IDEAS) and rv.V323_NOTE
    assert '__version__ = "3.2' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")


def test_fib_panel_matches_post_levels():
    """貼文的加權指數：49,966.89 → 35,527.69，0.382＝41,043、0.5＝42,747、0.618＝44,451；現在 40,122 → 反彈約 32%。"""
    days = pd.bdate_range("2026-06-01", periods=130)
    px = np.r_[np.linspace(42000, 49966.89, 90), np.linspace(49000, 35527.69, 35), np.linspace(37000, 40122.35, 5)]
    stub = SimpleNamespace(a={"Q_TAIEX": px}, meta={"q_days": [str(d.date()) for d in days]})
    g = {"ui": ui, "pd": pd, "np": np, "B": stub, "RES": {}, "taiex_now": lambda: {"px": 40122.35}}
    for fn in ("fib_text", "fib_panel_html"):
        node = next(n for n in ast.parse(APP).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(ast.get_source_segment(APP, node), g)
    h = g["fib_panel_html"]()
    assert "41,043" in h and "42,747" in h and "44,451" in h and "反彈到跌幅的 32%" in h and "隨機" in h


def test_etf_compare_real_etfs_and_lev():
    days = pd.bdate_range("2021-01-04", periods=700)
    rng = np.random.default_rng(3)
    r0 = rng.normal(0.0006, 0.012, len(days))
    nav = lambda r: np.cumprod(1 + r)                                                         # noqa: E731
    tr = np.column_stack([nav(r0), nav(r0 * 0.8 + 0.0003), nav(2 * r0 - 0.0001), nav(r0 * 1.2), np.r_[np.full(400, np.nan), nav(r0[400:])]])
    meta = {"etf_days": [str(d.date()) for d in days], "etf_tickers": ["0050", "00905", "00631L", "00878", "00999"],
            "etf_table": [{"ticker": "0050", "name": "元大台灣50", "type": "國內股票"}, {"ticker": "00905", "name": "FT臺灣Smart", "type": "國內股票"},
                          {"ticker": "00631L", "name": "元大台灣50正2", "type": "槓桿反向"}, {"ticker": "00878", "name": "高股息", "type": "國內股票"},
                          {"ticker": "00999", "name": "新上市", "type": "國內股票"}]}
    B = SimpleNamespace(a={"ETF_TR": tr.astype(np.float32)}, meta=meta)
    out = BT.etf_compare(B)
    ks = [r["ticker"] for r in out["rows"]]
    assert set(ks) == {"00905", "00631L", "00878"} and "00999" not in ks                       # 不滿約兩年的不比
    r = next(x for x in out["rows"] if x["ticker"] == "00905")
    assert r["mdd"] > r["b_mdd"] and r["years"]
    lv = out["lev"]
    assert lv["sim_corr"] > 0.95 and lv["switches"] >= 0 and np.isfinite(lv["cagr"])
    assert BT.etf_compare(SimpleNamespace(a={}, meta={}))["error"]


def test_beat_html_shows_real_etf_section():
    days = pd.bdate_range("2021-01-04", periods=600)
    r0 = np.random.default_rng(5).normal(0.0005, 0.01, len(days))
    tr = np.column_stack([np.cumprod(1 + r0), np.cumprod(1 + 2 * r0)])
    B = SimpleNamespace(a={"ETF_TR": tr}, meta={"etf_days": [str(d.date()) for d in days], "etf_tickers": ["0050", "00631L"],
                                                 "etf_table": [{"ticker": "00631L", "name": "元大台灣50正2", "type": "槓桿反向"}]})
    g = {"ui": ui, "pd": pd, "np": np, "NAMES": [], "TIDX": {}, "RES": {"beat0050": {"etf": BT.etf_compare(B), "cands": []}}}
    for fn in ("_pp", "_beat_verdict", "beat_rows", "fib_text", "beat0050_html"):
        node = next(n for n in ast.parse(APP).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(ast.get_source_segment(APP, node), g)
    h = g["beat0050_html"]()
    assert "真實 ETF 和 0050 同期間比較" in h and "00631L" in h and "＋200 日線" in h
