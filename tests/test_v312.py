"""v3.12：回頭檢查（快取鍵、說明文字）、觀察名單前瞻追蹤（影子模型）。"""
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import review as rv
from kosto import watch as W
from kosto.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_forward_and_verdict():
    d = pd.bdate_range("2026-10-01", periods=60)
    nav = pd.Series(np.r_[np.linspace(1.0, 1.1, 16), np.linspace(1.1, 0.99, 22), np.linspace(0.99, 1.21, 22)], index=d)
    f = W.forward(nav, "2026-10-23")
    k = int(d.searchsorted(pd.Timestamp("2026-10-23")))
    assert f["days"] == len(d) - k and f["first"] == "2026-10-23"
    assert abs(f["ret"] - (nav.iloc[-1] / nav.iloc[k - 1] - 1)) < 1e-12 and f["mdd"] < -0.05
    assert W.forward(nav, "2027-01-04")["days"] == 0
    assert W.verdict("return", 0, None, None)[0] == "wait"
    assert W.verdict("return", 50, 0.01, -0.01)[0] == "accum" and "領先" in W.verdict("return", 50, 0.01, -0.01)[1]
    assert W.verdict("return", 120, 0.01, -0.01)[0] == "pass"
    assert W.verdict("return", 120, 0.01, -0.03)[0] == "fail"
    assert W.verdict("balanced", 130, -0.015, 0.025)[0] == "pass"
    assert W.verdict("balanced", 130, -0.03, 0.05)[0] == "fail"


def test_watch_variants_on_synthetic(tmp_path):
    from kosto.bundle import build_bundle
    from kosto.panel import build_panel
    from kosto.research import Research
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    st = Store(tmp_path)
    write_demo(st, 160, 3, start="2017-01-02", end="2023-12-29")
    cfg = load_config(overrides={"data": {"dir": str(tmp_path)}, "research": {"start": "2019-01-01", "holdout_months": 6},
                                 "pools": {"sizes": [60, 100], "rules": ["ADV", "KOSTO"]}})
    B = build_bundle(build_panel(st), cfg, log=lambda *a: None)
    R = Research(B, cfg, log=lambda m: None)
    rule, size = "ADV", 100
    for h in ("mid", "long"):
        v = R.model_variant(h)
        Wt, _ = R.schedule(rule, size, h, v)
        spec = R.spec(h, v)
        for key in ("W1", "W2", "W3", "W4"):
            out = W.variant(R, key, h, rule, size, v, Wt, spec)
            if out is None:
                continue
            w2, sp2 = out
            assert len(w2) == len(Wt)
            if key == "W3":
                assert sp2.max_per_sector == 3 and w2 is Wt
            else:
                for a, b in zip(Wt, w2):
                    assert (a is None) == (b is None)
                    if b is not None:
                        assert abs(float(np.abs(b).sum()) - 1) < 1e-4 or key == "W4"
    if "f_rev_surprise" in B.factors:
        v = R.model_variant("long")
        Wt, _ = R.schedule(rule, size, "long", v)
        w2, _ = W.variant(R, "W1", "long", rule, size, v, Wt, R.spec("long", v))
        F = list(B.factors)
        ws = [w for w in w2 if w is not None]
        if ws and "f_rev_rel" in F:
            assert abs(ws[-1][F.index("f_rev_surprise")] - 0.15) < 1e-4 and abs(ws[-1][F.index("f_rev_rel")] - 0.15) < 1e-4
    res = {"protocol": {"last_date": str(B.sig_dates[-1].date())},
           "horizons": {h: {"chosen_pool": {"rule": rule, "size": size}} for h in ("short", "mid", "long")}}
    o = W.evaluate(R, res, since=str(B.sig_dates[-30].date()), need=10)
    assert o["rows"] and all(r["days"] > 0 for r in o["rows"])
    assert {r["key"] for r in o["rows"]} <= {"W1", "W2", "W3", "W4"}


def test_schedule_cache_key_includes_overlay():
    src = (ROOT / "kosto" / "research.py").read_text(encoding="utf-8")
    i = src.index("def schedule(")
    assert 'r0.get("overlay")' in src[i:i + 1200]


def test_v312_docs_and_app():
    t = {x[0]: x for x in rv.V312_TESTS}
    assert set(t) == {"R1a", "R1b", "R2", "R3", "R4"} and all("❌" in x[-1] for x in rv.V312_TESTS)
    assert t["R2"][2] >= 1.5 and t["R2"][9].startswith("中線 3")        # 報酬門檻過、一致性沒過
    assert t["R4"][6] < -1 and t["R4"][7] > 2                           # 長線：少賺、少跌
    assert len(rv.V312_AUDIT) >= 4 and [x for x in rv.IDEAS if x[0] == "v3.12"]
    assert {it["key"] for it in W.ITEMS} == {"W1", "W2", "W3", "W4"} and W.SINCE == "2026-10-23" and W.NEED_DAYS == 120
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def v312_html" in app and "v312_html()" in app
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert "watch.evaluate(R, res)" in cli
    doc = (ROOT / "docs" / "v3.12_protocol.md").read_text(encoding="utf-8")
    assert "## 四、結果" in doc and "## 五、觀察名單前瞻追蹤" in doc
    tech = (ROOT / "docs" / "technical.md").read_text(encoding="utf-8")
    assert "等權 10 檔、單檔買進額 ≤ 20 日均量 2%" not in tech
    cfg = load_config()
    assert cfg["research"]["overlay"] == {"f_rev_rel": 0.15}             # 主模型不變
    assert cfg["horizons"]["long"]["model"]["hyp_blend"] == 0.5 and cfg["research"]["max_per_sector"] is None
    assert '__version__ = "3.' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
