"""v3.21：重大消息衝擊（輝達特有報酬、連動排名、事件區間、消息日研究）、影子模型 W8～W10、盤前重大消息、App。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import morning as M
from kosto import review as rv
from kosto import shock as SH
from kosto import watch as W

ROOT = Path(__file__).resolve().parents[1]


def _bundle(D=900, N=120, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=D)
    ix = 100 * np.cumprod(1 + rng.normal(0, 0.012, D))
    mr = np.r_[0, ix[1:] / ix[:-1] - 1]
    idio = np.clip(rng.normal(0, 0.015, D), -0.04, 0.04)
    ev = [400, 500, 600, 700, 800]
    idio[ev] = [0.12, -0.10, 0.09, -0.11, 0.10]
    nv = 100 * np.cumprod(1 + 1.5 * mr + idio)
    beta = np.linspace(0, 1, N)
    R = 0.6 * mr[:, None] + 0.5 * idio[:, None] * beta[None, :] + rng.normal(0, 0.012, (D, N))
    R[[t + 1 for t in ev], :] -= 0.3 * idio[ev][:, None] * beta[None, :]              # 隔天回吐 60%
    return SimpleNamespace(a={"R": R, "OC": R * 0.3, "WORLD": np.c_[nv, ix], "CAP": np.tile(np.arange(N, 0, -1), (5, 1)).astype(float),
                              "BENCH": 0.6 * mr}, meta={"world_keys": ["NVDA", "IXIC"], "groups": [f"電子-G{k % 4}" for k in range(N)]},
                           days=days, tickers=[str(1000 + k) for k in range(N)], sig_pos=np.arange(0, D, 5)), beta


def test_nvda_news_and_linkage():
    B, beta = _bundle()
    nd = SH.nvda_news(B)
    assert abs(nd["b"].iloc[0] - 1.5) < 0.15 and nd["e"].abs().max() > 0.09
    L = SH.linkage(B, top=100, days=750)
    st = pd.DataFrame(L["stocks"])
    true_b = pd.Series(0.5 * beta, index=[str(1000 + k) for k in range(len(beta))])
    assert st.set_index("ticker")["beta_idio"].corr(true_b.reindex(st["ticker"]).set_axis(st["ticker"])) > 0.8   # 找得到高連動股
    assert L["groups"] and all(g["n"] >= 3 for g in L["groups"])
    assert SH.linkage(SimpleNamespace(a={"WORLD": np.zeros((5, 1))}, meta={"world_keys": ["SOX"]}, days=B.days[:5], tickers=[])).get("error")


def test_shock_study_detects_reversal_and_episode():
    B, _ = _bundle()
    S = SH.shock_study(B, top=100)
    a = S["summary"][0]
    assert a["n"] == 5 and a["d0"] > 0 and a["d5"] < 0                                 # 當天跟著消息、之後回吐
    days = B.days
    E = SH.episode(B, str(days[500].date()), str(days[520].date()))
    assert not E.get("error") and len(E["stocks"]) == 120 and E["rebound_days"] == len(days) - 1 - 520
    assert set(E["stocks"][0]) >= {"down", "up", "recover", "down_rel"}
    assert SH.episode(B, "2030-01-01", "2030-02-01")["error"].startswith("研究資料只到")
    out = SH.summary(B)
    assert {"linkage", "study", "huang", "huang_meta"} <= set(out) and out["huang"].get("error")   # 合成資料沒涵蓋 2026-11


def test_watch_news_candidates(tmp_path):
    from kosto.bundle import build_bundle
    from kosto.config import load_config
    from kosto.panel import build_panel
    from kosto.research import Research
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    st = Store(tmp_path)
    write_demo(st, 120, 3, start="2017-01-02", end="2022-12-30")
    cfg = load_config(overrides={"data": {"dir": str(tmp_path)}, "research": {"start": "2019-01-01", "holdout_months": 6},
                                 "pools": {"sizes": [60, 100], "rules": ["ADV", "KOSTO"]}})
    B = build_bundle(build_panel(st), cfg, log=lambda *a: None)
    D = len(B.days)
    rng = np.random.default_rng(1)
    ix = 100 * np.cumprod(1 + rng.normal(0, 0.01, D))
    idio = rng.normal(0, 0.02, D)
    idio[::60] = 0.08
    nv = 100 * np.cumprod(1 + 1.4 * np.r_[0, ix[1:] / ix[:-1] - 1] + idio)
    B.a["WORLD"] = np.c_[np.asarray(B.a["WORLD"], np.float32).reshape(D, -1), nv, ix].astype(np.float32)
    B.meta["world_keys"] = list(B.meta.get("world_keys") or []) + ["NVDA", "IXIC"]
    W._NEWS.clear()
    beta, sgn = W.news_arrays(B)
    assert beta.shape == (len(B.sig_pos), len(B.tickers)) and set(np.unique(sgn)) <= {-1.0, 0.0, 1.0} and (sgn != 0).any()
    R = Research(B, cfg, log=lambda m: None)
    rule, size, h = "ADV", 100, "mid"
    v = R.model_variant(h)
    Wt, _ = R.schedule(rule, size, h, v)
    spec = R.spec(h, v)
    w8 = W.variant(R, "W8", h, rule, size, v, Wt, spec)
    w9 = W.variant(R, "W9", h, rule, size, v, Wt, spec)
    assert len(w8) == 3 and len(w8[2].Z) == len(Wt) and w8[2].Z[-1].shape[1] == len(B.factors) + 1
    i = next(k for k, w in enumerate(w8[0]) if w is not None and sgn[k] != 0)
    assert abs(w8[0][i][-1] - 0.15) < 1e-6 and abs(np.abs(w8[0][i]).sum() - 1) < 1e-5
    j = next(k for k, w in enumerate(w8[0]) if w is not None and sgn[k] == 0)
    assert w8[0][j][-1] == 0
    zi = w8[2].Z[i][:, -1]
    assert np.allclose(zi, -w9[2].Z[i][:, -1], atol=1e-5) or np.corrcoef(zi, w9[2].Z[i][:, -1])[0, 1] < -0.9      # 反轉＝延續的反方向
    w10 = W.variant(R, "W10", h, rule, size, v, Wt, spec)
    assert w10[1].no_new is not None and w10[1].no_new.any() and w10[0] is Wt
    res = {"protocol": {"last_date": str(B.sig_dates[-1].date())}, "horizons": {h: {"chosen_pool": {"rule": rule, "size": size}}}}
    o = W.evaluate(R, res, since=str(B.sig_dates[-30].date()), need=10)
    ks = {r["key"] for r in o["rows"]}
    assert {"W8", "W9", "W10"} <= ks
    assert all(r["days"] == 0 for r in o["rows"] if r["key"] in ("W8", "W9", "W10"))                 # 2026-11-25 起才算前瞻


def test_morning_news_shock_text():
    res = {"shock": {"linkage": {"b_nasdaq": 1.5, "groups": [{"group": "電子上游-半導體"}, {"group": "電子-半導體設備"}, {"group": "金融"}]},
                     "study": {"summary": [{"side": "利多（輝達特有上漲）", "n": 12, "d0": 0.012, "d5": -0.004, "d20": -0.006}]}}}
    q = {"NVDA": {"chg": 0.33}, "IXIC": {"chg": 0.03}}
    t = M.news_shock_text(q, res)
    assert "輝達前一晚 +33.0%" in t and "+28.5%" in t and "12 次" in t and "電子上游-半導體" in t and "不追漲停開盤" in t and "金融" not in t
    assert M.news_shock_text({"NVDA": {"chg": 0.02}}, res) == ""
    rep = M.build({}, {"quotes": {**q, "SOX": {"chg": 0.05, "price": 1}}, "night": {}, "news": []}, res, pd.Timestamp("2026-11-24 05:50", tz="Asia/Taipei"))
    assert rep["analysis"]["paragraphs"][0]["title"] == "重大消息" and rep["analysis"].get("news_shock")


def test_v321_docs_app():
    assert len(rv.V321_EVENTS) >= 6 and rv.V321_GROUP_SENS[0][1] >= rv.V321_GROUP_SENS[-1][1]
    assert any("不調" in x[-1] for x in rv.V321_CAND) and {"W8", "W9", "W10"} <= {x["key"] for x in W.ITEMS}
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def news_shock_html", "📰 消息衝擊", "news_shock_html()"):
        assert fn in app, fn
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert 'res["shock"] = shock.summary(B)' in cli
    doc = (ROOT / "docs" / "v3.21_protocol.md").read_text(encoding="utf-8")
    assert "照實註明" in doc and "輝達特有報酬" in doc
    assert '__version__ = "3.2' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")


def test_app_has_no_duplicate_functions():
    import ast
    import collections
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    c = collections.Counter(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
    assert not [k for k, v in c.items() if v > 1]                                   # 同名函式會讓前一個被覆蓋（v3.21 差點發生）


def test_news_shock_html_renders():
    import ast
    from kosto import ui
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "news_shock_html")
    B, _ = _bundle()
    sh = SH.summary(B)
    sh["huang"] = SH.episode(B, str(B.days[500].date()), str(B.days[520].date()))
    for RES in ({}, {"shock": sh, "watch": {"rows": [{"key": "W8", "name": "x", "h": "mid", "dev_d_excess": 0.01, "status_zh": "等待新資料"}]}}):
        top = sh["linkage"]["stocks"][0]["ticker"]
        g = {"ui": ui, "pd": pd, "np": np, "NAMES": ["台積電"], "TIDX": {top: 0}, "HZ": {"mid": "中線", "short": "短線", "long": "長線"}, "RES": RES}
        exec(ast.get_source_segment(src, fn), g)
        h = g["news_shock_html"]()
        assert "誰跟輝達連動最高" in h and "跌最深" in h and "消息對短中長線" in h and "模型權重怎麼調" in h
        if RES:
            assert f"{top} 台積電" in h and "反彈最大 15" in h and "W8" in h
