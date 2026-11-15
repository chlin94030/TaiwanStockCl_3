"""v3.4：美股選股（模型、回測、Yahoo 解析、時段、資料包、即時重算、App 接線）。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kosto.us import backtest as BT
from kosto.us import data as D
from kosto.us import engine as UE
from kosto.us import model as M
from kosto.us import names as NM

ROOT = Path(__file__).resolve().parents[1]


def _panel(n=420, k=30, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    tk = [f"S{i:02d}" for i in range(k)]
    drift = np.linspace(-0.001, 0.002, k)
    C = pd.DataFrame(50 * np.cumprod(1 + drift + rng.normal(0, 0.01, (n, k)), axis=0), idx, tk)
    O = C.shift(1).fillna(C.iloc[0]) * (1 + rng.normal(0, 0.002, (n, k)))
    V = pd.DataFrame(rng.integers(1e6, 5e6, (n, k)).astype(float), idx, tk)
    return C, O, V


def test_factors_universe_and_pick():
    C, O, V = _panel()
    U = M.universe(C, V, top=20)
    assert not U.iloc[:259].any().any() and (U.iloc[-1].sum() == 20)
    F = M.factors(C, V)
    assert set(F) == set(M.FACTOR_ZH)
    s = M.score(F, U, M.MODELS["MOM"]).iloc[-1]
    assert s.notna().sum() == 20 and s.max() <= 1.0
    # MOM12_1 不看最近一個月：今天大跌不影響
    C2 = C.copy()
    C2.iloc[-1, 0] *= 0.5
    assert M.factors(C2, V, ["MOM12_1"])["MOM12_1"].iloc[-1, 0] == pytest.approx(F["MOM12_1"].iloc[-1, 0])
    sc = pd.Series(np.arange(30, 0, -1, dtype=float), index=C.columns)
    assert M.pick(sc, 5) == list(C.columns[:5])
    assert "S20" in M.pick(sc, 5, held=["S20"], hyst=5) and "S28" not in M.pick(sc, 5, held=["S28"], hyst=5)
    w = M.rank_weights(10)
    assert w.sum() == pytest.approx(1) and w[0] / w[-1] == pytest.approx(3)


def test_backtest_accounting():
    T = 300
    idx = pd.bdate_range("2020-01-01", periods=T)
    g = np.array([0.002, 0.0, -0.001, 0.001])
    C = pd.DataFrame(np.cumprod(np.tile(1 + g, (T, 1)), axis=0) * 100, idx, list("ABCD"))
    O = C.shift(1).fillna(100)
    U = pd.DataFrame(True, idx, C.columns)
    S = pd.DataFrame(np.tile([4, 3, 2, 1], (T, 1)).astype(float), idx, C.columns)
    assert BT.run(C, O, S, U, 20, k=1, cost=0).mean() == pytest.approx(0.002)
    assert BT.run(C, O, S, U, 5, k=2, cost=0.001).iloc[:5].mean() == pytest.approx(0.75 * 0.002, rel=0.01)     # 名次加權 1.5:0.5
    b = BT.run(C, O, None, U, 20, bench=True)
    st = BT.stats(BT.run(C, O, S, U, 20, k=1, cost=0), b)
    assert st["excess"] > 0 and st["fold_win"] == 1.0


def _chart(ts, closes, meta_extra=None, vol=None, adj=None):
    meta = {"currency": "USD", "symbol": "TEST", "gmtoffset": -18000, "regularMarketPrice": closes[-1]}
    meta.update(meta_extra or {})
    q = {"close": closes, "open": closes, "high": closes, "low": closes, "volume": vol or [1000] * len(ts)}
    ind = {"quote": [q]}
    if adj is not None:
        ind["adjclose"] = [{"adjclose": adj}]
    return {"chart": {"result": [{"meta": meta, "timestamp": ts, "indicators": ind}]}}


def test_parse_daily_and_panel_shapes():
    d = pd.bdate_range("2026-11-02", periods=4)
    ts = [int((x + pd.Timedelta(hours=14, minutes=30)).timestamp()) for x in d]
    f = D.parse_daily(_chart(ts, [10.0, None, 11.0, 12.0], adj=[5.0, None, 5.5, 6.0]))
    assert len(f) == 3 and f["adj"].iloc[-1] == 6.0 and f.index[0] == d[0]
    assert D.parse_daily({}).empty
    assert D.ysym("brk.b") == "BRK-B"


def _sessions(day):
    et = lambda h, m: int(pd.Timestamp(f"{day} {h:02d}:{m:02d}", tz="America/New_York").timestamp())  # noqa: E731
    return {"pre": {"start": et(4, 0), "end": et(9, 30)}, "regular": {"start": et(9, 30), "end": et(16, 0)},
            "post": {"start": et(16, 0), "end": et(20, 0)}}, et


def test_snapshot_pre_regular_post():
    per, et = _sessions("2026-11-13")
    meta = {"currentTradingPeriod": per, "chartPreviousClose": 100.0, "regularMarketPrice": 100.0}
    # 盤前：只有盤前資料，regularMarketPrice＝昨收
    ts = [et(8, 0), et(8, 30), et(9, 0)]
    bars, m = D.parse_intraday(_chart(ts, [101.0, 98.0, 95.0], meta))
    assert (bars["session"] == "pre").all()
    now = pd.Timestamp("2026-11-13 09:05", tz="America/New_York")
    s = D.snapshot("TEST", bars, m, now)
    assert s["phase"] == "pre" and s["ext"] == 95.0 and s["ext_chg"] == pytest.approx(-0.05) and s["ext_label"] == "盤前"
    # 盤中
    ts2 = ts + [et(9, 30), et(10, 0)]
    bars2, m2 = D.parse_intraday(_chart(ts2, [101.0, 98.0, 95.0, 96.0, 103.0], {**meta, "regularMarketPrice": 103.0}))
    s2 = D.snapshot("TEST", bars2, m2, pd.Timestamp("2026-11-13 10:01", tz="America/New_York"))
    assert s2["phase"] == "regular" and s2["chg"] == pytest.approx(0.03) and not np.isfinite(s2["ext"])
    # 盤後
    ts3 = ts2 + [et(16, 30)]
    bars3, m3 = D.parse_intraday(_chart(ts3, [101.0, 98.0, 95.0, 96.0, 103.0, 106.09], {**meta, "regularMarketPrice": 103.0}))
    s3 = D.snapshot("TEST", bars3, m3, pd.Timestamp("2026-11-13 17:00", tz="America/New_York"))
    assert s3["phase"] == "post" and s3["ext_label"] == "盤後" and s3["ext_chg"] == pytest.approx(0.03)
    assert D.phase(m3, pd.Timestamp("2026-11-13 21:00", tz="America/New_York")) == "closed"


def test_market_clock_dst():
    w = D.market_clock(pd.Timestamp("2026-11-13 10:00", tz="America/New_York"))
    s = D.market_clock(pd.Timestamp("2026-07-14 08:00", tz="America/New_York"))
    assert w["phase"] == "regular" and not w["dst"] and w["open_tpe"] == "22:30" and w["close_tpe"] == "05:00"
    assert s["phase"] == "pre" and s["dst"] and s["open_tpe"] == "21:30" and s["pre_tpe"] == "16:00"
    assert D.market_clock(pd.Timestamp("2026-11-14 10:00", tz="America/New_York"))["phase"] == "closed"     # 週六


def _bundle():
    info = UE.universe_table(fetch=False, log=None)
    tk = list(info.index[:40]) + ["SPY", "XLK"]
    rng = np.random.default_rng(5)
    n = 320
    idx = pd.bdate_range("2025-08-01", periods=n)
    C = pd.DataFrame(80 * np.cumprod(1 + rng.normal(0.0005, 0.015, (n, len(tk))), axis=0), idx, tk)
    P = {"C": C, "O": C.shift(1).fillna(C.iloc[0]), "H": C * 1.01, "L": C * 0.99,
         "V": pd.DataFrame(1e7, idx, tk), "RAW": C * 2, "fail": ["X"]}
    return UE.build(P, info=info, log=None, now=pd.Timestamp("2026-11-14 12:00", tz="UTC")), C


def test_engine_build_live_and_roundtrip():
    B, C = _bundle()
    assert B["asof"] == str(C.index[-1].date()) and "SPY" in B["etfs"] and "SPY" not in B["tickers"]
    assert set(B["picks"]) == {"short", "mid", "long", "watch"} and all(len(p["names"]) == 10 for p in B["picks"].values())
    assert B["picks"]["short"]["model"] == "MINVOL" and B["picks"]["long"]["model"] == "BAL_REV"     # v3.5 第二輪
    assert B["picks"]["watch"]["model"] == "SPMO"
    B2 = UE.loads(UE.dumps(B))
    assert B2["picks"] == B["picks"] and B2["C"].shape == B["C"].shape
    # 即時重算：沒有報價＝今天平盤；中長線名單會因大漲改變分數
    t = B["picks"]["long"]["names"][-1]
    S0 = UE.live_scores(B, {})
    S1 = UE.live_scores(B, {t: float(B["RAW"][t].iloc[-1]) * 1.3})
    assert S1.loc[t, "live_chg"] == pytest.approx(0.3, rel=1e-4) and S0.loc[t, "live_chg"] == 0
    assert S1.loc[t, "BAL_REV"] != pytest.approx(S0.loc[t, "BAL_REV"])
    assert S1.loc[t, "MOM"] == pytest.approx(S0.loc[t, "MOM"])           # 短線一年動能不看最近一個月
    lp = UE.live_picks(B, {})
    assert set(lp["short"]["names"]) == set(B["picks"]["short"]["names"]) and lp["short"]["new"] == []
    assert len(UE.candidates(B, 5)) <= 60
    # 續抱：前一份資料包的名單仍在前 25 名就保留
    P = {"C": C, "O": C, "H": C, "L": C, "V": pd.DataFrame(1e7, C.index, C.columns), "RAW": C}
    B3 = UE.build(P, info=B["info"], prev=B, log=None, now=pd.Timestamp("2026-11-14 12:00", tz="UTC"))
    assert len(B3["picks_log"]) >= 3 and isinstance(UE.track(B3, "long"), pd.DataFrame)


def test_drop_partial_bar_during_session():
    B, C = _bundle()
    P = {k: B[k] for k in ("C", "O", "H", "L", "V", "RAW")}
    today = pd.Timestamp("2026-11-13")
    P = {k: pd.concat([v, v.iloc[[-1]].set_axis([today])]) for k, v in P.items()}
    out = UE._drop_partial(P, pd.Timestamp("2026-11-13 15:00", tz="UTC"))                 # 美東 10:00 盤中
    assert out["C"].index[-1] < today
    keep = UE._drop_partial(P, pd.Timestamp("2026-11-13 22:00", tz="UTC"))                # 美東 17:00 收盤後
    assert keep["C"].index[-1] == today


def test_names_research_and_wiring():
    assert NM.zh_name("NVDA") == "輝達" and NM.sector_zh("Information Technology") == "資訊科技"
    assert any("2330" in x for x in NM.tw_links("TSM")) and NM.tw_links("ZZZZ") == []
    assert all(t in NM.EXTRA_SECTOR for t in NM.EXTRAS)
    R = UE.research()
    assert R["round1"]["reference"] == {"short": "MOM", "mid": "BAL_REV", "long": "BAL_REV"}
    m = pd.DataFrame(R["round1"]["models"])
    assert len(m) == 3 * 6 * 3                                              # 第一輪 6 個模型
    ref = m[(m.h == "long") & (m.model == "BAL_REV") & (m.period == "dev")].iloc[0]
    assert ref["excess"] < 0.05 and ref["ir"] < 0.4                         # 照實記錄：未達 IR 門檻
    doc = (ROOT / "docs" / "v3.4_us_protocol.md").read_text(encoding="utf-8")
    assert "預先宣告" in doc and "## 二、結果" in doc and "未通過" in doc
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for s in ("def page_us", 'url_path="us"', "UE.live_picks", "us_validation_banner", "us_bundle_daily"):
        assert s in app
    wf = (ROOT / ".github" / "workflows" / "us.yml").read_text(encoding="utf-8")
    assert "python cli.py us-build" in wf and "us_bundle.pkl.gz" in wf
    assert "us-build" in (ROOT / "cli.py").read_text(encoding="utf-8")
    assert json.loads((ROOT / "kosto" / "data" / "us_research.json").read_text(encoding="utf-8"))["round1"]["n_tickers"] > 2000
