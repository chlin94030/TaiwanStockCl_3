"""v3.17：崩跌因應——市況定位、反轉觀察、急跌風險分級、不新買遮罩、盤前評析、觀察名單 W6。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import crash as C
from kosto import morning as M
from kosto import review as rv

ROOT = Path(__file__).resolve().parents[1]


def _bundle(n_days=300, n=900, crash=True, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    bench = rng.normal(0.0005, 0.01, n_days)
    if crash:
        bench[-20:] = -0.017                                          # 最後 20 天急跌約 −29%
    R = rng.normal(0, 0.02, (n_days, n))
    R[-1, :300] = -0.10                                               # 最後一天 300 檔跌停
    q = 120
    mkt = np.full((q, 3), np.nan)
    mkt[:, 0] = np.r_[np.full(q - 10, 7000.0), np.linspace(7000, 4800, 10)]   # 融資餘額
    mkt[:, 1] = np.r_[np.full(q - 5, 10.0), [-300, -250, -400, -100, -200]]   # 外資：連 5 日賣超
    mkt[:, 2] = np.r_[np.full(q - 3, 18.0), [36.0, 41.0, 38.0]]          # VIX
    a = {"BENCH": bench, "R": R.astype(np.float32), "Q_MKT": mkt.astype(np.float32),
         "Q_MB": np.tile(np.linspace(0, 5000, n), (q, 1)).astype(np.float32), "Q_ML": np.full((q, n), 10000.0, np.float32),
         "VOL60": np.tile(np.linspace(0.01, 0.05, n), (3, 1)).astype(np.float32), "CAP": np.tile(np.linspace(1e9, 1e6, n), (3, 1))}
    meta = {"q_mkt_cols": ["mbal", "foreign", "vix"], "q_days": [str(d.date()) for d in days[-q:]]}
    return SimpleNamespace(a=a, meta=meta, days=days, tickers=[f"{1000 + k}" for k in range(n)])


def test_market_state_and_checklist():
    st = C.market_state(_bundle())
    assert st["panic"] is True and st["dd250"] < -0.25 and st["limit_down"] >= 300
    assert st["margin_chg120"] < -0.25 and st["foreign_streak"] == -5 and st["vix_max20"] >= 41
    assert C.regime(st)[0] == "crash" and C.regime({"dd250": -0.12})[0] == "correction" and C.regime({"dd250": -0.02})[0] == "normal"
    ck = {r["name"]: r for r in C.reversal_checklist(st)}
    assert ck["融資餘額自高點減 25%"]["ok"] and not ck["VIX 衝上 35 後回到 30 以下"]["ok"] and not ck["外資連 3 日買超"]["ok"]
    assert ck["VIX 衝上 35 後回到 30 以下"]["r60"] == 20.5
    ck2 = {r["name"]: r for r in C.reversal_checklist({**st, "vix": 28.0})}
    assert ck2["VIX 衝上 35 後回到 30 以下"]["ok"]
    an = C.analogs(-0.28)
    assert [a["peak"][:4] for a in an] == ["2020", "2022", "2025"]


def test_crash_risk_tiers():
    B = _bundle()
    r = C.crash_risk(B)
    assert len(r) == 800 and {v["tier"] for v in r.values()} == {"低", "中", "高"}
    hi = [t for t, v in r.items() if v["tier"] == "高"]
    assert len(hi) == 160 and all(r[t]["score"] > 0.8 for t in hi)
    assert C.recent_limit_downs(B, "1000") >= 1 and C.recent_limit_downs(B, "nope") == 0


def test_no_new_mask():
    from kosto.backtest import Spec, choose
    from kosto.optimize import PoolView
    B = SimpleNamespace(a={"CAP": np.full((1, 4), 1e12)})
    view = PoolView([np.arange(4)], [np.array([[0.4], [0.3], [0.2], [0.1]], np.float32)])
    w = np.array([1.0], np.float32)
    sp = Spec(H=5, step=5, top_k=2, hysteresis=1.0, use_gates=False)
    assert choose(B, view, 0, w, [], sp)[0] == [0, 1]
    nn = np.array([[True, False, False, False]])
    assert choose(B, view, 0, w, [], Spec(**{**sp.__dict__, "no_new": nn}))[0] == [1, 2]          # 不新買第 0 檔
    assert choose(B, view, 0, w, [0], Spec(**{**sp.__dict__, "no_new": nn}))[0] == [0, 1]         # 已持有照樣續抱


def test_analysis_crash_narrative():
    model = {"m1": {"coef": {"const": 0, "sox": 0.2, "ixic": 0.05, "tsm_twd": 0.1, "dvix": -0.0005}, "rmse": 0.007, "tested": True},
             "edges": [-1, 1], "labels": ["全部"], "buckets": [], "m3": {"n": 51, "oc300": -0.004, "neg": 0.67, "tested": True},
             "m4": {"n": 36, "oc300": 0.004, "tested": True},
             "state": {"date": "2026-11-20", "dd250": -0.279, "dd120": -0.279, "r20": -0.21, "panic": True, "margin_chg120": -0.317,
                       "limit_down5": [174, 120, 298, 300, 365], "foreign_streak": -6, "foreign5": -2900.0, "vix": 38.5, "vix_max20": 41.2,
                       "sox_vs_ma20": -0.14},
             "risk": {"2344": {"tier": "中", "ld5": 3}, "6213": {"tier": "高", "ld5": 0}, "2395": {"tier": "低", "ld5": 0}}}
    q = {k: {"price": p, "prev": p / (1 + c), "chg": c, "bar_date": "2026-11-19"} for k, p, c in
         (("SOX", 7000, -0.03), ("IXIC", 22000, -0.015), ("TSM", 400, -0.04), ("TWD", 31.0, 0.002), ("VIX", 38.5, 0.1), ("GSPC", 6500, -0.012))}
    res = {"horizons": {"mid": {"latest": {"picks": [{"ticker": "2395", "name": "研華", "kept": True}, {"ticker": "2344", "name": "華邦電", "kept": True},
                                                     {"ticker": "6213", "name": "聯茂", "kept": False}]}}}}
    rep = M.build(model, {"quotes": q, "night": {}, "news": [{"title": "聯準會釋鷹派訊號", "link": "https://x", "source": "s",
                                                             "time": "2026-11-20T04:00:00+08:00", "tags": ["利率／央行"]}]},
                  res, pd.Timestamp("2026-11-20 05:50", tz="Asia/Taipei"))
    an = rep["analysis"]
    assert an["regime"][0] == "crash" and "急跌延續" in an["oneliner"]                      # 預估開低
    titles = [p["title"] for p in an["paragraphs"]]
    assert titles == ["市況定位", "今晨美股對台股的意義", "消息與風險溫度", "模型名單怎麼看", "資金與部位"]
    txt = " ".join(p["text"] for p in an["paragraphs"])
    assert "2020 年" in txt and "融資" in txt and "華邦電（3 根）" in txt and "65%" in txt and "6213 聯茂" in txt and "不用融資" in txt
    assert any(r["name"].startswith("VIX") for r in an["watch"])
    cons = " ".join(x["text"] for x in rep["advice"]["conservative"])
    aggr = " ".join(x["text"] for x in rep["advice"]["aggressive"])
    assert "分 3～4 份" in cons and "2344" not in cons.split("前 3 名")[-1][:40]                # 急跌時前 3 名排除連續跌停的華邦電
    assert "暫不新買：2344、6213" in aggr
    calm = M.analysis({**model, "state": {"dd250": -0.02, "r20": 0.01, "panic": False}}, {}, q, {}, res, [])
    assert calm["regime"][0] == "normal" and calm["watch"]


def test_v317_results_docs_app():
    assert len(rv.V317_EPISODES) == 6 and all(x[2] <= -15 for x in rv.V317_EPISODES)
    assert all("❌" in x[-1] for x in rv.V317_RULES) and "W6" in rv.V317_RULES[3][-1]
    assert rv.V317_TIERS[0][1] < rv.V317_TIERS[2][1] and rv.V317_TIERS[0][3] > 50 and rv.V317_TWO["next_ld"] > 60
    from kosto import watch as W
    assert "W6" in {x["key"] for x in W.ITEMS}
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def crash_now_html", "def crash_history_html", "def analysis_html", "def risk_badge", "crash_now_html(cur)",
               "crash_history_html()", "analysis_html(rep)", "risk_badge(t)", "原始數據", "def crash_mode_now", "暫不新買</b>"):
        assert fn in app, fn
    doc = (ROOT / "docs" / "v3.17_protocol.md").read_text(encoding="utf-8")
    assert "## 二、急跌時的選股規則" in doc and "## 五、結果" in doc
    import re
    v = re.search(r'__version__ = "3\.(\d+)\.', (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8"))
    assert v and int(v.group(1)) >= 17


def test_live_state_only_more_severe():
    model = {"state": {"date": "2026-10-22", "dd250": -0.08, "dd120": -0.08, "r20": -0.04, "panic": False}}
    st = M.live_state(model, {"taiex": {"date": "2026-11-20", "close": 36033.0, "high250": 49969.0, "dd250": -0.279, "dd120": -0.279, "r20": -0.21}})
    assert st["dd250"] == -0.279 and st["panic"] and C.regime(st)[0] == "crash" and st["taiex_date"] == "2026-11-20"
    same = M.live_state(model, {"taiex": {"date": "2026-10-22", "close": 1.0, "high250": 1.0, "dd250": -0.5, "dd120": -0.5, "r20": -0.3}})
    assert same["dd250"] == -0.08                                                     # 不比研究資料新 → 不動
    up = M.live_state({"state": {**model["state"], "dd250": -0.3}}, {"taiex": {"date": "2026-11-20", "close": 1.0, "high250": 1.0, "dd250": -0.1, "dd120": -0.1, "r20": 0.05}})
    assert up["dd250"] == -0.3                                                         # 只會更嚴重
    an = M.analysis({"state": st}, {}, {}, {}, {}, [])
    assert "加權指數 2026-11-20 收 36,033" in an["paragraphs"][0]["text"] and an["regime"][0] == "crash"
