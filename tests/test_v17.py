"""v1.7：國際市場資料（Yahoo 解析、時間對齊不偷看未來）、新研究選項（減碼、收盤進場、波動倒數）、連動統計。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto.backtest import Spec, book_vol, run
from kosto.panel import align_world
from kosto.sources import world as W
from tests.test_engine import _toy_bundle, world  # noqa: F401  (world = 示範資料 fixture)


def _payload(dates, closes, off=-14400):
    ts = [int((pd.Timestamp(d) + pd.Timedelta(hours=16)).timestamp()) - off for d in dates]   # 16:00 local
    return {"chart": {"result": [{"meta": {"gmtoffset": off, "regularMarketPrice": closes[-1], "regularMarketTime": ts[-1]},
                                  "timestamp": ts, "indicators": {"quote": [{"close": closes}]}}]}}


def test_parse_chart_uses_exchange_local_date_and_drops_nulls():
    f = W.parse_chart(_payload(["2026-10-21", "2026-10-22", "2026-10-23"], [100.0, None, 102.0]), "SOX")
    assert f["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-21", "2026-10-23"]
    assert f["ticker"].unique().tolist() == ["SOX"] and f["close"].tolist() == [100.0, 102.0]


def test_align_world_us_lagged_one_day_asia_same_day():
    tw = pd.DatetimeIndex(["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26"])
    w = pd.DataFrame({"date": pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"] * 2),
                      "ticker": ["SOX"] * 5 + ["KS11"] * 5, "close": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50.0]})
    A = align_world(w, tw)
    # 台股 10/21 收盤後：美股只知道 10/20（10/21 晚上的美股還沒收盤）；韓股知道 10/21
    assert A.loc["2026-10-21", "SOX"] == 2 and A.loc["2026-10-21", "KS11"] == 30
    # 週一 10/26：美股最後是週五 10/23
    assert A.loc["2026-10-26", "SOX"] == 5 and A.loc["2026-10-26", "KS11"] == 50
    stale = align_world(w, pd.DatetimeIndex(["2026-11-30"]))
    assert stale.isna().all().all()                                # 超過 10 天沒有新資料 → 不用


def test_quote_parses_and_skips_failures(monkeypatch):
    class Resp:
        def __init__(self, p):
            self.p = p

        def raise_for_status(self):
            return None

        def json(self):
            return self.p

    def fake_get(url, params=None, headers=None, timeout=None):
        if "SOX" in url:
            return Resp(_payload(["2026-10-21", "2026-10-22"], [100.0, 103.0]))
        raise W.requests.ConnectionError("blocked")
    monkeypatch.setattr(W.requests, "get", fake_get)
    q = W.quote(["SOX", "VIX"])
    assert q["key"].tolist() == ["SOX"] and q["chg"].iloc[0] == pytest.approx(0.03) and len(q["spark"].iloc[0]) == 2


def test_fetch_gives_up_fast_when_host_unreachable(monkeypatch):
    calls = []

    class S:
        headers = {}

        def get(self, *a, **k):
            calls.append(1)
            raise W.requests.ConnectionError("proxy refused")
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    f = W.fetch(session=S(), log=lambda *a: None)
    assert f.empty and len(calls) == 2


def test_exposure_scale_moves_stock_weight_to_cash():
    B, view, Wt = _toy_bundle()
    full = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0))
    half = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, expo_scale=np.full(len(B.sig_pos), 0.5)))
    assert half.tranches["exposure"].iloc[0] == pytest.approx(0.5)
    r_full = full.tranches["ret"].dropna().to_numpy()
    r_half = half.tranches["ret"].dropna().to_numpy()
    assert np.allclose(r_half[0], r_full[0] / 2, rtol=1e-6)       # first tranche: half invested, rest cash (0%)


def test_entry_at_close_skips_open_to_close_of_new_buys():
    B, view, Wt = _toy_bundle()
    S = len(B.sig_pos)
    a = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0))
    b = run(B, view, Wt, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, entry_close=np.ones(S, bool)))
    R, OC = B.a["R"].astype(float), B.a["OC"].astype(float)
    # first tranche is a new buy: open→close of day 1 is skipped; later tranches keep the same stock (no effect)
    assert b.nav.iloc[-1] == pytest.approx(a.nav.iloc[-1] / (1 + OC[1, 0]), rel=1e-6)


def test_invvol_weighting_prefers_calm_stock_and_book_vol():
    B, view, Wt = _toy_bundle()
    S = len(B.sig_pos)
    B.a["VOL60"] = np.tile(np.array([[0.01, 0.04]], np.float32), (S, 1))
    res = run(B, view, Wt, Spec(H=5, step=5, top_k=2, buy_cost=0, sell_cost=0, weighting="invvol"))
    assert res.tranches["n_filled"].iloc[0] == 2
    sv = np.vstack([np.linspace(1, 1.2, 40), np.full(40, np.nan)])
    assert np.isfinite(book_vol(sv, 39)) and np.isnan(book_vol(sv, 5))


def test_linkage_tables_from_bundle():
    from kosto import linkage
    rng = np.random.default_rng(3)
    D, N = 400, 60
    days = pd.bdate_range("2022-01-03", periods=D)
    sox = 1000 * np.cumprod(1 + rng.normal(0, 0.02, D))
    us = np.r_[np.nan, sox[:-1]]                                   # aligned: value known at T = US close of T-1
    us_ret = np.r_[np.nan, us[1:] / us[:-1] - 1]
    R = (0.3 * np.nan_to_num(us_ret)[:, None] + rng.normal(0, 0.01, (D, N))).astype(np.float32)
    OC = (R * 0.2).astype(np.float32)
    sig = np.arange(0, D, 5)
    B = SimpleNamespace(a={"R": R, "OC": OC, "WORLD": np.column_stack([us, us * 2]).astype(np.float32),
                           "RANKS": np.tile(np.arange(1, N + 1, dtype=np.int16), (1, len(sig), 1))},
                        meta={"world_keys": ["SOX", "KS11"], "rules": ["MCAP"], "groups": ["電子上游-半導體"] * N},
                        days=days, sig_pos=sig, tickers=[str(i) for i in range(N)])
    out = linkage.summary(B)
    ov = out["overnight"]
    assert len(ov) == 5 and ov["天數"].sum() > 300
    assert ov.attrs["beta"]["gap"] > 0.15                          # most of the US move shows up in the opening gap
    assert out["yearly"]["相關_費半(前一夜)"].min() > 0.3
    assert out["state"]["light"] in ("green", "amber", "red")


def test_v18_holdings_board(tmp_path_factory):
    from kosto import brief as brieflib
    from kosto.backtest import run as _run
    from kosto.bundle import build_bundle
    from kosto.config import load_config
    from kosto.panel import build_panel
    from kosto.research import Research
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    root = tmp_path_factory.mktemp("v18")
    st = Store(root)
    write_demo(st, 120, 5, start="2018-01-02", end="2022-12-30")
    cfg = load_config(overrides={"data": {"dir": str(root)}, "research": {"start": "2020-01-01", "holdout_months": 6},
                                 "pools": {"sizes": [60], "rules": ["ADV"]}})
    B = build_bundle(build_panel(st), cfg, log=lambda *a: None)
    assert "REV_M" in B.a and "Q_PE" in B.a and len(B.meta["rev_ym"]) == B.a["REV_M"].shape[0]
    R = Research(B, cfg, log=lambda *a: None)
    v = R.model_variant("mid")
    ev = R.evaluate("ADV", 60, "mid", v, keep=True)
    lt = R.latest("mid", "ADV", 60, v, ev["_res"])
    hold = R.holdings("mid", "ADV", 60, v, ev["_res"].tranches, lt)
    assert hold and all(x["status"] in ("keep", "watch", "exit") for x in hold)
    assert all(x["buffer"] == max(x["K"], round(x["K"] * cfg["horizons"]["mid"]["hysteresis"])) for x in hold)
    assert all(x["next_check"] > str(B.sig_dates[-1].date()) for x in hold)
    assert sum(x["slots"] for x in hold) <= 4 * cfg["research"]["top_k"] and all(x["nt"] == 4 for x in hold)
    b = brieflib.brief(B, B.tickers.index(lt["picks"][0]["ticker"]))
    assert b["tone"] in ("正向", "中性", "保守") and b["title"]


def test_v19_index_parsers():
    from kosto.sources.official import parse_twse_fmtqik, parse_twse_index_hist
    a = parse_twse_index_hist({"stat": "OK", "fields": ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"],
                               "data": [["115/10/29", "47,014.87", "47,015.45", "46,055.07", "46,368.33"],
                                        ["115/10/30", "46,139.79", "46,139.79", "44,726.26", "45,211.83"]]})
    b = parse_twse_fmtqik({"stat": "OK", "data": [["115/10/30", "13,112,939,113", "939,605,267,660", "6,365,634", "45,211.83", "-1,156.50"]]})
    assert a["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-29", "2026-10-30"]
    assert a.iloc[1]["low"] == pytest.approx(44726.26) and a.iloc[1]["close"] == pytest.approx(45211.83)
    assert b.iloc[0]["amount"] == pytest.approx(939605267660.0)


def test_v19_tournament_rules():
    from kosto import tournament as T
    from kosto.config import load_config
    cfg = load_config()
    champ = T.base_params(cfg, "mid")
    ch = T.challengers(champ)
    assert ch and all(T.key(c) != T.key(champ) for c in ch)
    assert all(sum(abs(c[d] - champ[d]) > 1e-9 for d in T.GRID) == 1 for c in ch)       # 每次只動一個維度
    eff = T.apply(cfg, {"mid": dict(champ, top_k=6, hysteresis=3.0)})
    assert eff["horizons"]["mid"]["top_k"] == 6 and cfg["horizons"]["mid"].get("top_k") is None
    from kosto.backtest import spec_from_cfg
    assert spec_from_cfg(eff, "mid").top_k == 6 and spec_from_cfg(eff, "short").top_k == cfg["research"]["top_k"]
    better = ch[0]
    res = {"protocol": {"last_date": "2026-10-30"}, "horizons": {"mid": {}}}

    def rows(gain):
        out = []
        for ph in range(5):
            out.append({"phase": ph, "h": "mid", "key": T.key(champ), "params": champ, "dev_excess": 0.20, "dev_ir": 0.7,
                        "hold_cagr": 0.9, "pre_excess": -0.1})
            out.append({"phase": ph, "h": "mid", "key": T.key(better), "params": better, "dev_excess": 0.20 + gain, "dev_ir": 0.8,
                        "hold_cagr": 0.9, "pre_excess": -0.1})
        return out
    st = {}
    for day in range(9):
        res["protocol"]["last_date"] = str((pd.Timestamp("2026-11-02") + pd.offsets.BDay(day)).date())
        st = T.update(st, rows(0.03), res, cfg, T.apply(cfg, st.get("champions")))
    assert st["leader"]["mid"]["streak"] == 9 and T.key(st["champions"]["mid"]) == T.key(champ)
    res["protocol"]["last_date"] = "2026-11-13"
    st = T.update(st, rows(0.03), res, cfg, T.apply(cfg, st.get("champions")))
    assert T.key(st["champions"]["mid"]) == T.key(better) and st["log"][-1]["h"] == "mid"       # 第 10 天換上
    st2 = T.update({}, rows(0.01), res, cfg, cfg)                                                # 差距不足 → 不算領先
    assert not st2["leader"]["mid"]


def test_v19_forward_and_market_pack(world):
    from kosto import forward, tournament as T
    from kosto.bundle import market_pack
    B = world.B
    log = [{"date": str(B.sig_dates[-30].date()), "h": "mid", "days": 20, "picks": B.tickers[:5]},
           {"date": str(B.sig_dates[-1].date()), "h": "mid", "days": 20, "picks": B.tickers[:5]}]
    t = forward.entry_returns(B, log)
    assert len(t) == 2 and bool(t.iloc[0]["matured"]) and not bool(t.iloc[1]["matured"])
    assert forward.summary(B, log).loc["mid", "筆數"] == 1          # 今天的名單還沒有報酬
    arr, cols = market_pack(world.P, world.P.dates[-60:])
    assert arr.shape == (60, len(cols)) and np.isfinite(arr[:, cols.index("close")]).all()
    assert np.nanmax(arr[:, cols.index("up")]) > 0
    st = T.log_picks({}, {"protocol": {"last_date": "2026-10-30"}, "horizons": {"mid": {"days": 20, "latest": {"picks": [{"ticker": "2330"}]}}}})
    st = T.log_picks(st, {"protocol": {"last_date": "2026-10-30"}, "horizons": {"mid": {"days": 20, "latest": {"picks": [{"ticker": "2330"}]}}}})
    assert len(st["picks_log"]) == 1 and st["live_since"] == "2026-10-30"


def test_v19_overlay_weights(world):
    """research.overlay：固定權重疊加 → 原權重歸一後 ×(1−s)，指定因子 +s×方向；消融排除時不疊加。"""
    import copy as _c
    from kosto.research import Research
    B = world.B
    f = "f_rev_rel" if "f_rev_rel" in B.factors else B.factors[0]
    cfg = _c.deepcopy(world.cfg)
    cfg["research"]["overlay"] = {}
    R0 = Research(B, cfg, log=lambda *a: None)
    cfg2 = _c.deepcopy(cfg)
    cfg2["research"]["overlay"] = {f: 0.15}
    R1 = Research(B, cfg2, log=lambda *a: None)
    h = "mid"
    rule, size = cfg["pools"]["rules"][0], cfg["pools"]["sizes"][0]
    v0, v1 = R0.model_variant(h), R1.model_variant(h)
    w0 = np.asarray(R0.final_fit(rule, size, h, v0).weights, float)
    w1 = np.asarray(R1.final_fit(rule, size, h, v1).weights, float)
    j = B.factors.index(f)
    exp = 0.85 * w0 / np.abs(w0).sum()
    exp[j] += 0.15
    assert np.allclose(w1, exp, atol=1e-5)
    v1.exclude = (f,)
    w2 = np.asarray(R1.final_fit(rule, size, h, v1).weights, float)
    assert abs(w2[j]) < 1e-9 or abs(w2[j] - w0[j]) < 1e-9


def test_v19_freshness_status():
    import pandas as pd
    from kosto import freshness as F
    tz = "Asia/Taipei"
    T = lambda s: pd.Timestamp(s, tz=tz)  # noqa: E731
    # 週五 10:00：資料到週四 → 已是最新
    assert F.status(T("2026-10-30 10:00"), pd.Timestamp("2026-10-29"), None)[0] == "green"
    # 週五 15:00：已是最新，提示今日 21:40 更新
    lv, tx, _ = F.status(T("2026-10-30 15:00"), pd.Timestamp("2026-10-29"), None)
    assert lv == "green" and "21:40" in tx
    # 週六 01:00：應到週五；只到週四 → 本機模式黃燈（可能假日），GitHub 有成功紀錄但更早 → 紅燈
    assert F.status(T("2026-10-31 01:00"), pd.Timestamp("2026-10-29"), None)[0] == "amber"
    old = {"status": "completed", "conclusion": "success", "created": "2026-10-29T13:40:00Z", "url": "u"}
    assert F.status(T("2026-10-31 01:00"), pd.Timestamp("2026-10-29"), old)[0] == "red"
    # 週五是假日：週五 21:40 之後有成功執行 → 綠燈
    new = {"status": "completed", "conclusion": "success", "created": "2026-10-30T13:41:00Z", "url": "u"}
    assert F.status(T("2026-10-31 01:00"), pd.Timestamp("2026-10-29"), new)[0] == "green"
    # 執行中／失敗
    assert F.status(T("2026-10-30 22:10"), pd.Timestamp("2026-10-29"), {"status": "in_progress", "url": "u"})[0] == "amber"
    assert F.status(T("2026-10-30 22:10"), pd.Timestamp("2026-10-29"), {"status": "completed", "conclusion": "failure"})[0] == "red"
    # 落後一週 → 紅燈
    assert F.status(T("2026-10-31 01:00"), pd.Timestamp("2026-10-22"), None)[0] == "red"
