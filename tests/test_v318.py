"""v3.18：全天候檢視——模型天氣、觀察名單 W7（美股風險關閉時動能減半）、盤前評析的弱勢狀態說明。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import crash as C
from kosto import morning as M
from kosto import review as rv
from kosto import watch as W

ROOT = Path(__file__).resolve().parents[1]


def _bundle(n_days=320, n=50, crash=True, vix_last=36.0, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    bench = rng.normal(0.0008, 0.008, n_days)
    if crash:
        bench[-25:] = -0.014
    sox = 5000 * np.cumprod(1 + rng.normal(0.001, 0.01, n_days))
    if crash:
        sox[-20:] = sox[-21] * np.linspace(0.97, 0.80, 20)
    vix = np.full(n_days, 16.0)
    vix[-5:] = vix_last
    twd = np.linspace(31.0, 32.5, n_days)
    world = np.c_[sox, vix, twd].astype(np.float32)
    sig_pos = np.arange(0, n_days, 5)
    q = 120
    mkt = np.full((q, 1), 7000.0)
    mkt[-10:, 0] = np.linspace(7000, 5500, 10)
    a = {"BENCH": bench, "R": rng.normal(0, 0.02, (n_days, n)).astype(np.float32), "WORLD": world, "Q_MKT": mkt.astype(np.float32)}
    meta = {"world_keys": ["SOX", "VIX", "TWD"], "q_mkt_cols": ["mbal"], "q_days": [str(d.date()) for d in days[-q:]]}
    return SimpleNamespace(a=a, meta=meta, days=days, tickers=[str(1000 + k) for k in range(n)], sig_pos=sig_pos)


def test_model_weather_states_and_stats():
    wx = C.model_weather(_bundle())
    by = {r["dim"]: r for r in wx["rows"]}
    assert by["美股恐慌"]["state"] == "VIX ≥ 30" and by["美股恐慌"]["mid"] == -22.9 and by["美股恐慌"]["weak"]
    assert by["美股半導體"]["state"] == "費半在 50 日線之下" and by["台股市況"]["state"] == "急跌"
    assert by["槓桿"]["state"] == "融資餘額近 20 日減少" and not by["槓桿"]["weak"]
    assert wx["us_off"] is True and wx["weak"] >= 3
    calm = C.model_weather(_bundle(crash=False, vix_last=15.0))
    assert {r["dim"]: r["state"] for r in calm["rows"]}["美股恐慌"] == "VIX < 20" and calm["us_off"] is False
    live = C.model_weather(_bundle(crash=False, vix_last=15.0), {"vix": 33.0, "dd250": -0.2})
    lb = {r["dim"]: r["state"] for r in live["rows"]}
    assert lb["美股恐慌"] == "VIX ≥ 30" and lb["台股趨勢"] == "0050 在 200 日線之下"


def test_w7_signal_and_variant():
    B = _bundle()
    off = W.us_off_signals(B)
    assert off is not None and len(off) == len(B.sig_pos) and off[-1] and not off[10]
    assert "W7" in {x["key"] for x in W.ITEMS}
    F = list(W.MOM_GROUP) + ["f_ep", "c_trust20"]
    R = SimpleNamespace(B=SimpleNamespace(**{**B.__dict__, "factors": F}))
    w0 = np.r_[np.full(6, 0.1), 0.3, 0.1].astype(np.float32)
    Wl = [None] + [w0] * (len(B.sig_pos) - 1)
    out, spec = W.variant(R, "W7", "mid", "BLEND", 500, None, Wl, "SPEC")
    assert spec == "SPEC" and out[0] is None and np.allclose(out[10], w0)              # 沒觸發 → 不變
    w = out[-1]
    assert abs(np.abs(w).sum() - 1) < 1e-5 and abs(w[:6].sum() - 0.3) < 1e-5 and abs(w[6] / w[7] - 3) < 1e-4   # 動能減半、其餘等比例補回


def test_live_weather_and_analysis_text():
    wx = C.model_weather(_bundle(crash=False, vix_last=15.0))
    q = {"VIX": {"price": 38.0}, "SOX": {"price": 1.0}}
    lw = M.live_weather(wx, q, {})
    by = {r["dim"]: r for r in lw["rows"]}
    assert by["美股恐慌"]["state"] == "VIX ≥ 30" and by["美股恐慌"]["live"] and by["美股半導體"]["state"] == "費半在 50 日線之下" and lw["us_off"]
    model = {"state": {"dd250": -0.05, "r20": -0.02, "panic": False}, "weather": wx}
    an = M.analysis(model, {}, q, {}, {}, [])
    txt = " ".join(p["text"] for p in an["paragraphs"])
    assert "模型天氣" in txt and "VIX ≥ 30 -23%" in txt and "W7" in txt and an["weather"]["weak"] >= 2


def test_v318_results_docs_app():
    assert len(rv.V318_CAND) == 5 and sum("W7" in x[-1] for x in rv.V318_CAND) == 1
    assert all(x[-1].startswith("❌") for x in rv.V318_CAND if x[0] != "A3")
    a3 = [x for x in rv.V318_CAND if x[0] == "A3"][0]
    assert a3[2][1] >= 1.5 and a3[3] >= 0 and a3[10] > 0.10                            # 門檻過、q 未過 → 觀察
    weak = {s for d, s, m, n, t, sh, lg, wk in rv.V318_REGIME if wk}
    assert {"VIX ≥ 30", "急跌", "0050 在 200 日線之下", "費半在 50 日線之下"} <= weak
    assert all(m < 0 for d, s, m, n, t, sh, lg, wk in rv.V318_REGIME if wk)
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def weather_html", "def v318_html", "weather_html(cur)", "weather_html(compact=True)", "v318_html()"):
        assert fn in app, fn
    doc = (ROOT / "docs" / "v3.18_protocol.md").read_text(encoding="utf-8")
    assert "## 一、已經試過" in doc and "## 五、結果" in doc and "（待補）" not in doc
    import re
    v = re.search(r'__version__ = "3\.(\d+)\.', (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8"))
    assert v and int(v.group(1)) >= 18
