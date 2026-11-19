"""v3.16：每日台股盤前日報（歷史驗證模型、日報組成、排程、App）。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import morning as M
from kosto import review as rv

ROOT = Path(__file__).resolve().parents[1]


def _fake_bundle(n_days=900, n=140, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-02", periods=n_days)
    sox = rng.normal(0, 0.015, n_days)
    tsm = 0.8 * sox + rng.normal(0, 0.008, n_days)
    W = np.c_[np.cumprod(1 + sox) * 1000, np.cumprod(1 + 0.6 * sox + rng.normal(0, 0.006, n_days)) * 500,
              np.cumprod(1 + tsm) * 100, np.full(n_days, 30.0), 18 + np.cumsum(rng.normal(0, 0.3, n_days)).clip(-8, 30)]
    gap = 0.2 * sox[:, None] + rng.normal(0, 0.004, (n_days, n))                 # 前一夜費半 → 今天開盤（WORLD 已對齊到台股日）
    oc = -0.1 * np.where(sox[:, None] > 0.02, 1, 0) * 0.02 + rng.normal(0, 0.01, (n_days, n))
    R = (1 + gap) * (1 + oc) - 1
    sig = np.arange(0, n_days, 5)
    tick = ["2330"] + [f"{1000 + k}" for k in range(n - 1)]
    a = {"R": R.astype(np.float32), "OC": oc.astype(np.float32), "MCAP": np.tile(np.linspace(1e12, 1e9, n), (len(sig), 1)),
         "WORLD": W.astype(np.float32)}
    meta = {"world_keys": ["SOX", "IXIC", "TSM", "TWD", "VIX"]}
    return SimpleNamespace(a=a, meta=meta, days=days, sig_pos=sig, tickers=tick)


def test_fit_and_score_on_fake():
    B = _fake_bundle()
    m = M.fit(B, split="2021-06-01")
    assert "error" not in m, m
    assert m["m1"]["test"]["corr"] > 0.6 and m["m1"]["tested"] is True and m["m2"]["test"]["corr"] > 0.2
    assert m["m1"]["coef"]["sox"] > 0.1 and len(m["buckets"]) >= 3 and m["heavy"] == {}
    rz = M.realized(B, [B.days[-1], B.days[-2]])
    assert len(rz) == 2
    hist = [{"date": str(B.days[-1].date()), "pred_gap": 0.01}, {"date": "2099-01-01", "pred_gap": -0.01}]
    sc = M.score(hist, B)
    assert sc["n"] == 1 and sc["pending"] == 1
    assert M.fit(SimpleNamespace(a={}, meta={}, days=[], sig_pos=[], tickers=[]))["error"]


def _q(chg, price=100.0, prev=None):
    return {"price": price, "prev": prev if prev is not None else price / (1 + chg), "chg": chg, "bar_date": "2026-11-19"}


def _live():
    q = {"DJI": _q(0.004, 47000), "GSPC": _q(0.006, 6700), "IXIC": _q(0.012, 22900), "SOX": _q(0.031, 7100), "VIX": _q(-0.05, 17.0, 17.9),
         "TNX": _q(0.0, 4.10, 4.12), "TWD": _q(-0.002, 30.6), "TSM": _q(0.028, 452), "UMC": _q(0.01, 7.6), "NVDA": _q(0.04), "AAPL": _q(-0.003),
         "MU": _q(0.052), "NQF": _q(0.004, 25100)}
    return {"quotes": q, "night": {"TX": {"name": "台指期", "last": 23500, "ref": 23300, "chg": 23500 / 23300 - 1, "time": "05:00"}},
            "news": [{"title": "輝達財報優於預期 AI 晶片需求強", "link": "https://example.com/a", "source": "X", "time": "2026-11-20T05:10:00+08:00",
                      "tags": ["AI／半導體", "財報／展望"]}], "fetched_at": "2026-11-20T05:45:00+08:00"}


def test_build_report():
    B = _fake_bundle()
    m = M.fit(B, split="2021-06-01")
    m["heavy"] = {"2330": {"name": "台積電", "date": "2026-11-19", "close": 2385.0, "chg": -0.01, "fi1": -12000.0, "fi5": -60000.0},
                  "2303": {"name": "聯電", "date": "2026-11-19", "close": 46.0, "chg": 0.0, "fi1": 0.0, "fi5": 0.0}}
    res = {"horizons": {"mid": {"latest": {"picks": [{"ticker": "2395", "name": "研華", "kept": True},
                                                     {"ticker": "6213", "name": "聯茂", "kept": False}]}}}}
    now = pd.Timestamp("2026-11-20 05:50", tz="Asia/Taipei")
    rep = M.build(m, _live(), res, now)
    assert rep["date"] == "2026-11-20" and rep["status"] == "complete" and all(rep["sections_ok"].values())
    pr = rep["link"]["pred"]
    assert pr["gap"] > 0.003 and pr["lo"] < pr["gap"] < pr["hi"] and pr["g2330"] > 0
    assert "美股全面上漲" in rep["us"]["summary"] and "費半強於那指" in rep["us"]["summary"]
    assert "台指期夜盤 23,500" in rep["night"]["text"] and "偏多" in rep["night"]["text"]
    tsmc = next(r for r in rep["heavy"]["rows"] if r["ticker"] == "2330")
    assert abs(tsmc["adr"]["implied"] - 452 * 30.6 / 5) < 1e-6 and abs(tsmc["adr"]["premium"] - (452 * 30.6 / 5 / 2385 - 1)) < 1e-9
    assert any(t["theme"] == "記憶體" for t in rep["link"]["themes"])
    adv = rep["advice"]
    assert adv["conservative"][0]["ev"] == "tested" and "七成" in adv["conservative"][0]["text"]
    assert any("6213" in x["text"] for x in adv["aggressive"])
    assert rep["headline"][0].startswith("預估大盤開盤") and rep["disclaimer"]
    s = M.dumps(rep)
    assert "NaN" not in s
    h = M.append_hist([{"date": "2026-11-20", "pred_gap": 0.0}], rep)
    assert len(h) == 1 and h[0]["pred_gap"] == pr["gap"]
    # 夜盤取不到 → 用那指期日線
    lv = _live()
    lv["night"] = {}
    rep2 = M.build(m, lv, res, now)
    assert "那指期" in rep2["night"]["text"] and "相對前一日" in rep2["night"]["text"]
    # 沒有模型 → 沒有預估，但其他段落照常
    rep3 = M.build({}, _live(), res, now)
    assert rep3["link"]["pred"] == {} and rep3["us"]["rows"] and rep3["status"] == "partial"


def test_report_date_and_advice_rules():
    tz = "Asia/Taipei"
    assert str(M.report_date(pd.Timestamp("2026-11-20 05:50", tz=tz)).date()) == "2026-11-20"      # 週五清晨
    assert str(M.report_date(pd.Timestamp("2026-11-20 15:00", tz=tz)).date()) == "2026-11-23"      # 週五收盤後 → 下週一
    assert str(M.report_date(pd.Timestamp("2026-11-22 10:00", tz=tz)).date()) == "2026-11-23"      # 週日
    m = {"m3": {"n": 51, "oc300": -0.0039, "neg": 0.67, "tested": True}, "m4": {"n": 36, "oc300": 0.004, "tested": True}}
    up = M.advice(m, {"gap": 0.015, "tested": True}, {}, None)
    assert any("不追開盤" in x["text"] and x["ev"] == "tested" for x in up["conservative"])
    dn = M.advice(m, {"gap": -0.015, "tested": True}, {"VIX": {"price": 31.0}}, None)
    assert any("恐慌賣出" in x["text"] for x in dn["conservative"]) and any("VIX 31.0" in x["text"] for x in dn["conservative"] + dn["aggressive"])


def test_pipeline_workflow_and_cli():
    wf = (ROOT / ".github" / "workflows" / "morning.yml").read_text(encoding="utf-8")
    assert '"43 21 * * 0-4"' in wf and '"13 22 * * 0-4"' in wf and '"38 22 * * 0-4"' in wf      # 台北 05:43／06:13／06:38（週一～週五）
    assert "python cli.py morning" in wf and "morning_hist.json" in wf
    assert (ROOT / "morning_workflow_COPY.txt").read_text(encoding="utf-8") == wf
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert "def cmd_morning" in cli and '"morning": cmd_morning' in cli and 'res["morning_model"] = morning.fit(B)' in cli


def test_app_and_docs():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def morning_tab", "def morning_html", "def morning_files", "def morning_home_line", "def morning_banner", "morning_banner()", "def v316_html", '"🌅 盤前日報"',
               "morning_tab()", "v316_html()", "b1 += morning_home_line()"):
        assert fn in app, fn
    assert all(x[-1] == "tested" for x in rv.V316_RULES[:4]) and rv.V316_RULES[-1][-1] == "ref"
    assert [x for x in rv.IDEAS if x[0] == "v3.16"]
    doc = (ROOT / "docs" / "v3.16_protocol.md").read_text(encoding="utf-8")
    assert "## 二、歷史驗證規則" in doc and "## 三、結果" in doc
    assert '__version__ = "3.16.0"' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
