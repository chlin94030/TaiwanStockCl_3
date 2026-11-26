"""v3.24：外資賣現股、買主動式 ETF——擁擠研究（crowd.py）、影子模型 W11／W12、關稅情境與事件組、App 說明。"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import crowd as CR
from kosto import impact as I
from kosto import review as rv
from kosto import supply as S
from kosto import ui
from kosto import watch as W

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def _stub(D=400, N=60, seed=0):
    """前 6 檔＝擁擠股（投信買超最高），大跌日跌更多、之後 5 日繼續落後。"""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-06-02", periods=D)
    bench = rng.normal(0.0005, 0.008, D)
    stress = [100, 200, 300, 350]
    bench[stress] = -0.04
    R = bench[:, None] + rng.normal(0, 0.01, (D, N))
    crowd = np.arange(N) < 6
    for d in stress:
        R[d, crowd] -= 0.03
        R[d + 1: d + 6, crowd] -= 0.004
    sp = np.arange(0, D, 5)
    U = np.zeros((len(sp), N, 1), np.uint16)
    U[:, :, 0] = (65535 - np.arange(N) * 1000)[None, :]                      # 第 0 檔投信買超最高
    B = SimpleNamespace(a={"R": R, "BENCH": bench, "U": U, "CAP": np.ones((len(sp), N))}, factors=["c_trust60"], days=days,
                        sig_pos=sp, tickers=[str(1000 + k) for k in range(N)], meta={"names": [f"股{k}" for k in range(N)], "groups": ["電子"] * N})
    return B, stress


def test_crowd_event_study_detects_dumping():
    B, stress = _stub()
    sets = CR.crowd_sets(B, top_n=60)
    assert sets and all(hi.sum() == 6 for _, hi in sets.values())
    dn = CR.event_study(B, -0.03, sets)
    a = dn["summary"][0]
    assert a["n"] == len(stress) and a["d0"] < -0.025 and a["d5"] < -0.015 and a["d0_neg"] == 1.0
    assert {r["period"] for r in dn["summary"]} == {"全部", "2019–2025", "2026（主動式 ETF 時代）"}
    up = CR.event_study(B, 0.03, sets)
    assert up["n"] == 0 or abs(up["summary"][0].get("d0", 0)) < 0.02
    bs = CR.baseline(B, sets)
    assert bs["days"] > 300 and bs["mean"] < 0                                     # 平常日子差很小（大跌日拉低平均）
    lt = CR.latest_crowded(B, 5)
    assert [r["ticker"] for r in lt] == ["1000", "1001", "1002", "1003", "1004"]
    assert CR.crowd_sets(SimpleNamespace(factors=["x"], a={}, sig_pos=[])) is None


def test_tariff_scenario_and_events():
    assert I.EVENT_SETS["tariff"][1] == ["2019-05-06", "2019-08-02", "2025-04-07", "2025-08-01", "2026-11-25"]
    assert "2026-11-25" in I.EVENT_SETS["ai_down"][1]
    sc = I.SCEN_BY_ID["tariff_301"]
    assert I.SCENARIOS[0]["id"] == "tariff_301" and sc["analog"] == "tariff"
    assert all(g in S.SUBGROUPS for g, *_ in sc["impacts"]) and all(-3 <= d <= 3 for _, d, *_ in sc["impacts"])
    deep = {g for g, d, *_ in sc["impacts"] if d <= -2}
    assert {"server", "network", "pc"} <= deep and ("memory", 0) in {(g, d) for g, d, *_ in sc["impacts"]}
    assert any("232" in f for f in sc["facts"]) and len(sc["sources"]) >= 3


def test_watch_items_w11_w12():
    keys = {it["key"]: it for it in W.ITEMS}
    assert {"W11", "W12"} <= set(keys) and keys["W11"]["kind"] == "balanced" and keys["W12"]["since"] == "2026-11-27"


def test_crowd_mask_and_variants_on_synthetic(tmp_path):
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
    if "c_trust60" not in B.factors:
        return
    nn = W.crowd_mask(R, "ADV", 100)
    view = R.view("ADV", 100)
    i = R.first_oos
    share = nn[i, view.members[i]].mean()
    assert 0.05 < share < 0.2
    v = R.model_variant("mid")
    Wt, _ = R.schedule("ADV", 100, "mid", v)
    spec = R.spec("mid", v)
    w11 = W.variant(R, "W11", "mid", "ADV", 100, v, Wt, spec)
    w12 = W.variant(R, "W12", "mid", "ADV", 100, v, Wt, spec)
    assert w11[0] is Wt and w11[1].no_new.sum() >= w12[1].no_new.sum() and np.array_equal(w11[1].no_new | w12[1].no_new, w11[1].no_new)
    out = CR.summary(R, {"watch": {"rows": [{"key": "W11", "name": "x", "h": "mid"}]}})
    assert not out["errors"] and out["down"]["summary"] and out["latest"] and out["watch"][0]["key"] == "W11"


def test_app_panels_render():
    days = pd.bdate_range("2026-11-02", periods=20)
    tick = ["2330", "3017", "2454", "1802"]
    tr = np.zeros((20, 4))
    tr[-1] = [5000, 1200, -800, 300]
    stub = SimpleNamespace(a={"Q_TR": tr, "Q_C": np.full((20, 4), 100.0)}, meta={"q_days": [str(d.date()) for d in days]}, tickers=tick)
    g = {"ui": ui, "pd": pd, "np": np, "B": stub, "NAMES": ["台積電", "奇鋐", "聯發科", "台玻"], "TIDX": {t: i for i, t in enumerate(tick)},
         "HZ": {"short": "短線", "mid": "中線", "long": "長線"}, "RES": {}}
    for fn in ("crowd_html", "trust_footprint", "etf_flow_html"):
        node = next(n for n in ast.parse(APP).body if isinstance(n, ast.FunctionDef) and n.name == fn)
        exec(ast.get_source_segment(APP, node), g)
    h = g["etf_flow_html"]()
    for s in ("溢價套利", "折價套利", "1,236.5 億", "535.3 億", "投信足跡", "台積電 +5,000 張", "聯發科 -800 張", "docs/v3.24_protocol.md"):
        assert s in h, s
    B, _ = _stub()
    cr = CR.summary(SimpleNamespace(B=B), {"watch": {"rows": [{"key": "W12", "name": "急跌", "h": "mid", "dev_d_excess": 0.01, "dev_d_mdd": 0.02,
                                                              "days": 0, "status_zh": "等待新資料"}]}})
    g["RES"] = {"crowd": cr}
    h2 = g["crowd_html"]()
    assert "0050 單日 ≤ −3%" in h2 and "目前擁擠前 15 名" in h2 and "W12" in h2


def test_v324_wiring_docs():
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert 'res["crowd"] = crowd.summary(R, res)' in cli
    for s in ("etf_flow_html()", "crowd_html()", "籌碼擁擠研究（v3.24）"):
        assert s in APP, s
    doc = (ROOT / "docs" / "v3.24_protocol.md").read_text(encoding="utf-8")
    for s in ("看任何結果之前", "W11", "W12", "主模型本版不改", "301", "照實註明"):
        assert s in doc, s
    assert rv.V324_NOTE and len(rv.V324_FACTS) >= 8 and any(x[0] == "v3.24" for x in rv.IDEAS)
    assert '__version__ = "3.2' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
