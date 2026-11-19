"""v3.14：回應獨立同儕評審——跌停賣不掉、營收相對同業改官方產業別、每日快照雜湊鏈、更新狀態、報價時點、模型驗證頁。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import review as rv
from kosto import snapshot as SN
from kosto.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    from kosto.optimize import PoolView
    D, N = 12, 2
    days = pd.bdate_range("2026-01-05", periods=D)
    R, OC = np.zeros((D, N)), np.zeros((D, N))
    R[3, 0] = R[4, 0] = -0.10                   # 股票 A：連兩天開盤就跌停（開盤＝昨收 −10%）
    sig = np.array([0, 2, 4, 6, 8])
    S = len(sig)
    a = {"R": R, "OC": OC, "OK": np.ones((D, N), bool), "CAP": np.full((S, N), 1e12), "BENCH": np.zeros(D),
         "MKT": np.zeros((S, 8))}
    B = SimpleNamespace(a=a, sig_pos=sig, sig_dates=days[sig], tickers=["A", "B"], days=days)
    za, zb = np.array([[0.5], [-0.5]], np.float32), np.array([[-0.5], [0.5]], np.float32)
    view = PoolView([np.array([0, 1])] * S, [za, zb, zb, zb, zb])     # 第一次選 A，之後換成 B
    return B, view, [np.array([1.0], np.float32)] * S


def test_sell_block_toy():
    from kosto.backtest import Spec, run
    B, view, W = _toy()
    kw = dict(H=2, step=2, top_k=1, hysteresis=1.0, use_gates=False, buy_cost=0, sell_cost=0)
    n0 = run(B, view, W, Spec(**kw), start_i=0).nav
    n1 = run(B, view, W, Spec(**kw, sell_block=True), start_i=0).nav
    assert abs(n0.iloc[-1] - 0.9) < 1e-9                  # 舊假設：跌停開盤照樣賣掉（−10%）
    assert abs(n1.iloc[3] - 0.9) < 1e-9 and abs(n1.iloc[-1] - 0.81) < 1e-9   # 新：賣不掉、隔天又跌停，第三天才賣
    cfg = load_config()
    from kosto.backtest import spec_from_cfg
    assert cfg["research"]["sell_block"] is True and spec_from_cfg(cfg, "mid").sell_block is True


def test_sector_rel_official_groups():
    from kosto.factors import REGISTRY, _sector_rel
    tk = [f"t{k}" for k in range(12)]
    sec = ["半導體業"] * 6 + ["航運業"] * 4 + ["NA"] * 2          # 航運只有 4 檔（<5）→ 不算
    P = SimpleNamespace(info=pd.DataFrame({"sector": sec}, index=tk), tickers=tk)
    x = pd.DataFrame([np.arange(12, dtype=float)], columns=tk)
    out = _sector_rel(SimpleNamespace(P=P), x)
    assert np.allclose(out[tk[:6]].to_numpy(), np.arange(6) - 2.5)
    assert out[tk[6:]].isna().all().all()
    assert "官方產業別" in REGISTRY["f_rev_rel"].zh
    src = (ROOT / "kosto" / "factors.py").read_text(encoding="utf-8")
    assert 'def _f_rev_rel(c): return _sector_rel(c, _rev(c, lambda r: _yoy(r, 3), "rev_yoy3"))' in src


def _res(day="2026-11-18", picks=("2330", "2454")):
    return {"protocol": {"last_date": day},
            "horizons": {"mid": {"latest": {"date": day, "pool": "BLEND500",
                                            "picks": [{"ticker": t, "score": 1.0 - k / 10} for k, t in enumerate(picks)]}}}}


def test_snapshot_chain_and_consistency():
    cfg = load_config()
    meta = {"sig_dates": ["2026-11-17", "2026-11-18"], "built": "2026-11-18T21:50:00+08:00", "version": "3.14.0",
            "data_dates": {"price": "2026-11-18"}}
    r1 = _res()
    s1 = SN.make(r1, meta, cfg, "3.14.0", now=pd.Timestamp("2026-11-18 18:40", tz="Asia/Taipei"))
    assert s1["id"].startswith("20261118-") and len(s1["config_hash"]) == 12 and s1["picks"]["mid"]["tickers"] == ["2330", "2454"]
    ch = SN.append([], s1)
    s2 = SN.make(_res(picks=("2330", "3017")), meta, cfg, "3.14.0", now=pd.Timestamp("2026-11-18 23:05", tz="Asia/Taipei"))
    ch = SN.append(ch, s2)                                   # 同一天重跑：新增一筆，不覆寫
    assert len(ch) == 2 and ch[0]["seq"] == 1 and ch[1]["seq"] == 2 and ch[1]["prev"] == ch[0]["link"]
    assert SN.verify(ch) == (True, None)
    bad = [dict(x) for x in ch]
    bad[0]["hash"] = "0" * 64                                # 改動舊紀錄 → 鏈斷開
    assert SN.verify(bad)[0] is False
    r1["snapshot"] = s1
    assert SN.consistency(meta, r1) == []
    assert SN.consistency({**meta, "built": "2026-11-19T18:40:00+08:00"}, r1)          # 資料包已重建
    assert SN.consistency({**meta, "sig_dates": ["2026-11-19"]}, r1)
    r1["horizons"]["mid"]["latest"]["picks"][0]["ticker"] = "1101"                      # 名單被改
    assert any("名單" in x for x in SN.consistency(meta, r1))
    assert SN.consistency(meta, {}) and SN.config_hash(cfg) == SN.config_hash(load_config())
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert 'snapshot.append(prev.get("snapshots"), res["snapshot"])' in cli


def test_update_states_running_not_red():
    from kosto import freshness as F
    now = pd.Timestamp("2026-11-12 13:08", tz="Asia/Taipei")
    run = {"status": "in_progress", "conclusion": None, "created": "2026-11-12T04:40:00Z", "event": "push", "url": "u"}
    hist = [run] + [{"status": "completed", "conclusion": "failure", "created": f"2026-11-11T1{k}:40:00Z"} for k in range(3)]
    lvl, txt, _ = F.status(now, pd.Timestamp("2026-11-07"), run, hist)
    assert lvl == "amber" and "上一版" in txt and F.state_label(lvl, txt) == "執行中（保留上一版）"
    assert F.state_label("green", "已是最新") == "已完成" and F.state_label("green", "已是最新｜融資券 21:40 補上") == "部分完成"
    assert F.state_label("red", "最近一次更新失敗（點此看原因）") == "失敗" and F.state_label("red", "資料落後：應更新到 11/12") == "延遲"
    assert SN.update_state(run, "2026-11-11", "2026-11-12")[0] == "running"
    assert SN.update_state({"status": "completed", "conclusion": "failure"}, "2026-11-11", "2026-11-12")[0] == "failed"
    assert SN.update_state(None, "2026-11-11", "2026-11-12")[0] == "late" and SN.update_state(None, "2026-11-12", "2026-11-12", True)[0] == "partial"


def test_quote_source_time_stale():
    from kosto import intraday as I
    q = {"price": 100.0, "prev": 98.0, "src": "trade", "time": "10:20:11", "date": "20261119"}
    b = I.best_quote(q, None, 97.0, 0.01, "2026-11-18", today=pd.Timestamp("2026-11-19 10:33"))
    assert b["src_zh"] == "證交所成交" and b["time"] == "10:20" and b["stale_min"] == 13 and "已過 13 分" in I.source_line(b)
    b = I.best_quote(q, None, 97.0, 0.01, "2026-11-18", today=pd.Timestamp("2026-11-19 10:22"))
    assert I.source_line(b) == "證交所成交 10:20"
    b = I.best_quote(None, None, 97.0, 0.01, "2026-11-18", today=pd.Timestamp("2026-11-19 10:22"))
    assert I.source_line(b) == "" and b["label"] == "11/18 收盤" and b["src_zh"] == "收盤資料"


def test_price_fit_and_concentration():
    from kosto import plan as P
    pl = P.Plan(close=100, buy_low=96, buy_high=101, chase=104, alert=90, support=95, resist=110, atr_pct=0.03, rsi=60,
                bias20=0.03, hot=False)
    assert [P.price_fit(x, pl)[0] for x in (95, 100, 103, 106, 89)] == ["dip", "buy", "buy_s", "wait", "warn"]
    assert P.price_fit(100, pl, 100)[0] == "lock" and P.price_fit(np.nan, pl)[0] == "none"
    hot = P.Plan(**{**pl.as_dict(), "hot": True})
    assert "過熱" in P.price_fit(100, hot)[1]
    c = P.concentration(np.linspace(1.5, 0.5, 8), list("aabcadef"))
    assert abs(c["top_w"] - 0.1875) < 1e-9 and abs(c["eff_n"] - 7.2258) < 1e-3 and c["top_group"] == "a" and c["top_group_n"] == 3


def test_v314_results_docs_and_app():
    b1 = {x[0]: x for x in rv.V314_B1}
    assert b1["E2"][2] >= b1["E1"][2] - 1.0 and b1["E3"][2] < b1["E1"][2] - 1.0 and "採用" in b1["E2"][-1]      # 預先宣告規則 1
    assert b1["E2"][6] > b1["E1"][6] and b1["E2"][4] > b1["E1"][4]                       # 整段較好、回撤較淺
    st = {(a, b): x for a, b, *x in rv.V314_STRESS}
    assert st[("中線", "跌停賣不掉")][0] < st[("中線", "原假設（可賣、滑價 0.1%）")][0]
    assert st[("短線", "＋滑價 0.5%")][2] < 0                                           # 短線對成本最敏感
    b2 = rv.V314_B2
    assert b2[1][2] > 0.5 * b2[0][2]                                                     # 延後 5 日沒有掉一半 → 不算時點敏感
    assert all(x[-1] < 0.95 for x in rv.V314_STATS)                                      # Deflated Sharpe 都未達 0.95（照實寫）
    assert len(rv.V314_REVIEW) >= 12 and len(rv.V314_TIERS) == 4 and [x for x in rv.IDEAS if x[0] == "v3.14"]
    doc = (ROOT / "docs" / "v3.14_protocol.md").read_text(encoding="utf-8")
    assert "## 二、規則（寫在看結果之前）" in doc and "## 三、決定" in doc and "## 四、結果" in doc
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def page_validate", "def validate_html", "def snapshot_html", "def concentration_html", "planlib.price_fit(",
               'url_path="validate"', "IDY.source_line(bq)"):
        assert fn in app
    cfg = load_config()
    assert cfg["research"]["overlay"] == {"f_rev_rel": 0.15}
    assert '__version__ = "3.' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
