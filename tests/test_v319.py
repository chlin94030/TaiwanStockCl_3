"""v3.19：全方位驗證——台美連動預測（凍結模型、個股 β、族群／個股開盤預估）、美股第四輪、驗證結果與 App。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import morning as M
from kosto import review as rv

ROOT = Path(__file__).resolve().parents[1]


def _q(sox=-0.02, ixic=-0.01, tsm=-0.015, etf=-0.01):
    q = {"SOX": {"chg": sox, "price": 7000.0}, "IXIC": {"chg": ixic, "price": 22000.0}, "TSM": {"chg": tsm, "price": 400.0}}
    for k in M.LINK_ETF:
        q[k] = {"chg": etf, "price": 100.0}
    return q


def test_link_model_file_and_predict():
    lm = M.link_model()
    assert lm["signals"] == ["SOX", "IXIC", "TSM", "SPY", "XLK", "XLE", "XLF", "XLI", "XLB", "XLY", "XLV", "XLP", "XLU"]
    assert len(lm["groups"]) == 40 and len([g for g in lm["groups"].values() if g["n_test"]]) == 39 and all(g["corr"] is None or g["corr"] == g["corr"] for g in lm["groups"].values()) and sum(g["pass"] for g in lm["groups"].values()) == 12 and lm["market"]["corr_test"] > 0.7
    g = "電子上游-半導體"
    model = {"link": {"stocks": {"2344": {"beta": 0.9, "group": g}, "2883": {"beta": 0.15, "group": "金融-金融"}}, "gbeta": {g: 0.5, "金融-金融": 0.2}}}
    lp = M.link_predict(model, _q(), ["2344", "2883", "9999"])
    mk = lm["market"]
    assert abs(lp["market"] - (mk["const"] + mk["SOX"] * -0.02 + mk["IXIC"] * -0.01 + mk["TSM"] * -0.015)) < 1e-12
    assert lp["groups"]["金融-金融"]["ev"] == "tested" and lp["groups"][g]["ev"] == "ref"
    m = lm["groups"][g]
    assert abs(lp["groups"][g]["pred"] - (m["idx_const"] + m["idx_slope"] * lp["market"])) < 1e-12        # 沒通過 → 大盤 × β
    ridge = m["const"] + sum(m["coef"][k] * _q()[k if k in ("SOX", "IXIC", "TSM") else k]["chg"] for k in lm["signals"])
    assert abs(lp["stocks"]["2344"]["pred"] - (ridge + (0.9 - 0.5) * -0.02)) < 1e-12                   # 個股＝族群脊迴歸＋(β−族群 β)×費半
    assert "9999" not in lp["stocks"] and lp["full"]
    assert M.link_predict(model, {"SOX": {"chg": 0.01}}, ["2344"]) == {}                             # 缺大盤訊號 → 不預估
    part = M.link_predict(model, {k: v for k, v in _q().items() if k not in M.LINK_ETF}, ["2344"])
    assert not part["full"] and all(v["ev"] == "ref" for v in part["groups"].values())                 # 缺類股 ETF → 全部大盤 × β


def test_stock_betas_synthetic():
    rng = np.random.default_rng(0)
    n_days, n = 300, 6
    sox = 5000 * np.cumprod(1 + rng.normal(0, 0.015, n_days))
    sx = np.r_[0, sox[1:] / sox[:-1] - 1]
    betas = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.2])
    gap = sx[:, None] * betas + rng.normal(0, 0.002, (n_days, n))
    OC = rng.normal(0, 0.01, (n_days, n))
    R = (1 + gap) * (1 + OC) - 1
    B = SimpleNamespace(a={"R": R, "OC": OC, "WORLD": sox[:, None], "CAP": np.tile(np.arange(n, 0, -1), (n_days, 1)).astype(float)},
                        meta={"world_keys": ["SOX"], "groups": ["A", "A", "A", "B", "B", "B"]}, tickers=[f"{1000 + k}" for k in range(n)])
    out = M.stock_betas(B, {"horizons": {"mid": {"latest": {"picks": [{"ticker": "1005"}]}}}}, top=3)
    st = out["stocks"]
    assert set(st) >= {"1000", "1001", "1002", "1005"} and "1003" not in st                            # 前 3 大＋名單股
    assert abs(st["1005"]["beta"] - 1.2) < 0.05 and abs(st["1000"]["beta"]) < 0.05
    assert abs(out["gbeta"]["A"] - 0.233) < 0.05 and st["1001"]["group"] == "A"


def test_link_section_in_build():
    model = {"link": {"stocks": {"2330": {"beta": 0.6, "group": "電子上游-半導體"}, "2344": {"beta": 0.9, "group": "電子中游-半導體"}},
                      "gbeta": {"電子上游-半導體": 0.5, "電子中游-半導體": 0.6}}}
    res = {"horizons": {"mid": {"latest": {"picks": [{"ticker": "2344", "name": "華邦電", "kept": True}]}}}}
    rep = M.build(model, {"quotes": _q(sox=-0.03), "night": {}, "news": []}, res, pd.Timestamp("2026-11-23 05:50", tz="Asia/Taipei"))
    g = rep["groups"]
    assert g["groups"] and g["groups"][0]["pred"] >= g["groups"][-1]["pred"]
    st = {r["ticker"]: r for r in g["stocks"]}
    assert st["2344"]["pick"] and not st["2330"]["pick"] and st["2330"]["name"] == "台積電"
    assert "族群" in g["text"] and any("族群（成分股平均" in p["text"] for p in rep["analysis"]["paragraphs"])


def test_v319_results_docs_app():
    assert all(x[-1] == 100 for x in rv.V319_RANDOM) and len(rv.V319_BOOT) == 3
    assert all(b[6] < 0.95 for b in rv.V319_BOOT)                                                     # DSR 照實：未達 0.95
    short25 = [x for x in rv.V319_YEARS if x[0] == "short" and x[1] == 2025][0]
    assert short25[4] < 0 and short25[7] == 0
    assert rv.V319_LINK["groups_pass"] == len(rv.V319_LINK["pass_list"]) == 12 and rv.V319_LINK["l2"]["test"][0] >= 0.10
    assert all(x[-1] == "❌" for x in rv.V319_US) and all(x[3] < 3 for x in rv.V319_US)
    R = json.loads((ROOT / "kosto" / "data" / "us_research.json").read_text(encoding="utf-8"))
    assert {r["cand"] for r in R["round4"]} == {"U1", "U2", "U3", "U4", "U5"} and R["qqq_vs_spy"]["dev"] > 0.04
    from kosto.us import view as UV
    assert "第四輪候選" in UV.round4_html(R) and "QQQ" in UV.round4_html(R)
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def groups_html", "groups_html(rep.get(\"groups\")", "def v319_html", "v319_html()", "UV.round4_html(R)", "短線近兩年"):
        assert fn in app, fn
    doc = (ROOT / "docs" / "v3.19_protocol.md").read_text(encoding="utf-8")
    assert "## 五、結果（2026-11-21）" in doc and "（待補）" not in doc
    import re
    v = re.search(r'__version__ = "3\.(\d+)\.', (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8"))
    assert v and int(v.group(1)) >= 19
