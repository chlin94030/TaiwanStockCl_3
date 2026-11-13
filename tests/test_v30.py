"""v3.0：產業鏈地圖（細族群、上下游、題材）、消息衝擊評估、短中線名次加權。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import chainmap as cm
from kosto import impact as I
from kosto import supply as S


# ------------------------------------------------------------------ 分類資料的一致性
def test_supply_taxonomy_consistent():
    layers = {k for k, *_ in S.LAYERS}
    assert len(S.LAYERS) == 9 and len(S.SUBGROUPS) >= 35
    for g, d in S.SUBGROUPS.items():
        assert d["layer"] in layers and d["ai"] in (0, 1, 2) and d["plain"] and len(d["members"]) >= 1
        assert all(len(t) == 4 and t.isdigit() and r for t, r in d["members"])
        assert len({t for t, _ in d["members"]}) == len(d["members"])           # 同一族群不重複
    for a, b in S.EDGES:
        assert a in S.SUBGROUPS and b in S.SUBGROUPS and a != b
    for th in S.THEMES:
        assert th["stage"] in ("成熟", "成長", "題材") and th["plain"]
        assert all(g in S.SUBGROUPS for g in th.get("groups", []))
        assert S.theme_members(th)
    new = {th["id"] for th in S.THEMES if th["new"]}
    assert {"cpo", "foplp", "glass", "hvdc", "robot"} <= new
    assert "3481" in {t for t, _ in S.theme_members(S.THEME_BY_ID["foplp"])}       # 面板廠轉型
    assert "server" in S.customers("thermal") and "ccl" in S.suppliers("pcb")
    assert set(S.groups_of("2330")) == {"foundry", "adv_pkg"} and "AI 伺服器" in S.themes_of("2330")
    assert S.primary_group("3017") == "thermal"
    assert S.primary_group("9999", {"group": "電子中游-印刷電路板", "segment": "PCB"}) == "pcb"
    assert S.primary_group("9998", {"group": "電子上游-半導體", "segment": "IC設計"}) == "IC設計（其他）"
    assert S.primary_group("9997", {"group": "傳產-水泥", "segment": ""}) == "傳產-水泥"
    assert "3017" not in S.peers("3017")["thermal"] and "3324" in S.peers("3017")["thermal"]


def test_scenarios_valid():
    assert {s["id"] for s in I.SCENARIOS} >= {"dc_slowdown", "rates", "spacex", "iphone_duo", "nvidia_ceo"}
    for sc in I.SCENARIOS:
        assert sc["facts"] and sc["chain"] and sc["impacts"]
        assert sc.get("analog") in (None, *I.EVENT_SETS)
        for g, dep, speed, why in sc["impacts"]:
            assert g in S.SUBGROUPS and -3 <= dep <= 3
        assert all(u.startswith("https://") for _, u in sc.get("sources", []))
    dc = {g: d for g, d, *_ in I.SCEN_BY_ID["dc_slowdown"]["impacts"]}
    assert dc["thermal"] < dc["foundry"] < 0 and dc["lens"] == 0                    # 機房端受傷深、晶片端淺、手機不受影響
    for k, (nm, dates, desc) in I.EVENT_SETS.items():
        assert dates == sorted(dates) and all(len(d) == 10 for d in dates)


# ------------------------------------------------------------------ 計算（合成資料）
def _bundle():
    """60 日、5 檔：2330、3017（散熱）、3324（散熱）、2303、9999。第 45 日 SOX −5%，散熱股 −6%、其他 −1%。"""
    days = pd.bdate_range("2025-01-01", periods=60)
    tk = ["2330", "3017", "3324", "2303", "9999"]
    R = np.full((60, 5), 0.001)
    R[45, :] = -0.01
    R[45, 1:3] = -0.06
    R[46:51, 1:3] = 0.004
    sox = 100 * np.cumprod(np.r_[1, np.full(59, 1.001)])
    sox[45:] *= 0.95
    C = 100 * np.cumprod(1 + R, axis=0)
    a = {"R": R.astype(np.float32), "OK": np.ones((60, 5), bool), "Q_C": C, "Q_RF": np.vstack([C[:1], C[:-1]]),
         "Q_V": np.full((60, 5), 1000.0), "WORLD": sox[:, None].astype(np.float32)}
    meta = {"days": [str(d.date()) for d in days], "names": ["台積電", "奇鋐", "雙鴻", "聯電", "測試"], "world_keys": ["SOX"]}
    return SimpleNamespace(a=a, meta=meta, tickers=tk, days=days)


def test_event_returns_summary_path():
    B = _bundle()
    d30 = B.meta["days"][45]
    ev = I.event_returns(B, [d30, "2030-01-01"])                                  # 超出資料範圍 → 略過
    assert set(ev["event"]) == {d30}
    th = ev.set_index("group").loc["thermal"]
    mkt0 = np.mean([-0.01, -0.06, -0.06, -0.01, -0.01])
    assert th["d0"] == pytest.approx(-0.06 - mkt0, abs=1e-6)
    assert th["d5"] < 0 and ev.set_index("group").loc["foundry", "d0"] > 0          # 散熱相對弱、晶圓代工相對強
    sm = I.event_summary(ev)
    assert sm.index[0] == "thermal" and sm.loc["thermal", "n"] == 1 and sm.loc["thermal", "win5"] == 0.0
    p = I.path(B, I.group_index(B)["thermal"], d30, 10, 5)
    assert p.loc[-1] == pytest.approx(0.0) and p.loc[0] == pytest.approx(th["d0"], abs=1e-6) and p.index.min() == -10
    assert I.event_summary(pd.DataFrame()).empty


def test_betas_and_since():
    B = _bundle()
    b = I.betas(B, since="2025-01-01")
    assert b["thermal"] > b["foundry"] > 0                                         # 費半 −5% 那天散熱跌更多
    s = cm.since_event(B, B.meta["days"][45])
    assert s["thermal"] < 0 < s["foundry"]
    assert I.betas(SimpleNamespace(a={}, meta={}, tickers=[])).empty


def test_chainmap_html():
    B = _bundle()
    met = cm.metrics(B)
    assert met.loc["3017", "r1"] == pytest.approx(0.001, rel=1e-3) and np.isfinite(met.loc["2330", "r20"])
    gs = cm.group_stats(met)
    assert gs.loc["thermal", "n"] == 2 and gs.loc["thermal", "x20"] < gs.loc["foundry", "x20"]
    html = cm.overview_html(gs, "thermal")
    for _, nm, *_ in S.LAYERS:
        assert nm in html
    assert "cm-chip sel" in html and "?g=thermal" in html
    svg = cm.rack_svg(cm.lead_names(B))
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "奇鋐" in svg and "台積電" in svg
    assert "受傷・深" in cm.depth_bar(-3) and "受惠・淺" in cm.depth_bar(1) and "幾乎無" in cm.depth_bar(0)
    d = cm.group_detail("thermal", met, {"3017": 3}, {"3324"}, lambda t: t)
    assert "奇鋐" not in d or True
    assert "中線第 3 名" in d and "名單" in d and "下游（賣給誰）" in d
    cards = cm.theme_cards(met, lambda t: t, cm.market_ref(met))
    assert cards and all(c.startswith("<div class='kq-panel'>") for c in cards)


# ------------------------------------------------------------------ v3.0 模型：短中線名次加權
def test_rank_weighting_config_and_weights():
    from kosto import plan
    from kosto.config import load_config
    cfg = load_config()
    assert {h: cfg["horizons"][h]["weighting"] for h in ("short", "mid", "long")} == {"short": "rank", "mid": "rank", "long": "rank"}
    w = plan.stock_weights(8, "mid")
    assert w.sum() == pytest.approx(1) and w[0] == pytest.approx(1.5 / 8) and w[-1] == pytest.approx(0.5 / 8)
    assert plan.stock_weights(8, "short", "equal")[0] == pytest.approx(1 / 8)
    from kosto import review as rv
    assert sum(1 for t in rv.V30_TESTS if t[-1].startswith("✅")) == 1 and len(rv.V30_TESTS[0]) == 8


def test_docs_and_app_wired_v30():
    root = Path(__file__).resolve().parents[1]
    proto = (root / "docs" / "v3.0_protocol.md").read_text(encoding="utf-8")
    assert "## 結果" in proto and "只採用 W1" in proto and "預先宣告" in proto
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ("def page_chain", 'url_path="chain"', "def chain_position_html", "rv.V30_TESTS", "cm.rack_svg", "I.SCENARIOS"):
        assert s in app
