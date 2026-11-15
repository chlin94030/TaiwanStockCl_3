"""v3.6：美股第三輪、美股 ETF 推薦順序、新手總覽頁。"""
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import ui
from kosto.us import backtest as BT
from kosto.us import engine as UE
from kosto.us import names as NM
from kosto.us import view as UV

ROOT = Path(__file__).resolve().parents[1]


def test_round3_recorded_and_none_passed():
    R = UE.research()
    g = pd.DataFrame(R["round3"])
    assert set(g.cand) == {"V1", "V2", "V3", "V4", "V1_K30"} and set(g.h) == {"short", "mid", "long"}
    dev = g[(g.period == "dev") & (g.cand != "V1_K30")]
    assert (dev.excess < 0.03).all()                                          # 照實記錄：沒有一個開發期多 3pp
    best = dev.sort_values("excess").iloc[-1]
    assert best.cand in ("V3", "V4")
    assert "巨型股趨勢" in UV.round3_html(R)


def test_chooser_backtest():
    T = 260
    idx = pd.bdate_range("2020-01-01", periods=T)
    C = pd.DataFrame({"A": np.cumprod(np.full(T, 1.002)) * 100, "SPY": np.cumprod(np.full(T, 1.001)) * 100}, index=idx)
    O = C.shift(1).fillna(100)
    U = pd.DataFrame({"A": True, "SPY": False}, index=idx)
    r1 = BT.run(C, O, None, U, 20, cost=0, chooser=lambda s, held: (["SPY"], [1.0]))
    r2 = BT.run(C, O, None, U, 20, cost=0, chooser=lambda s, held: (["A"], [1.0]))
    assert abs(r1.mean() - 0.001) < 1e-9 and abs(r2.mean() - 0.002) < 1e-9


def test_etf_rank_rule_and_info():
    R = UE.research()
    st = R["etf_stats"]
    rk = UV.etf_rank(R)
    assert list(rk) == NM.ROLE_ORDER
    assert rk["core"][0] == "IVV" and rk["growth"][0] == "QQQ" and rk["hedge"][:2] == ["GLD", "IEF"] and rk["sat"][0] == "SOXX"
    for role, xs in rk.items():                                               # 角色內依 Sharpe 由高到低
        sh = [st[t]["sharpe"] for t in xs]
        assert sh == sorted(sh, reverse=True)
        assert all(NM.ETF_INFO[t]["role"] == role for t in xs)
    assert all(t in NM.ETFS for t in NM.ETF_INFO) and all(0 < i["er"] < 1 for i in NM.ETF_INFO.values())
    assert set(NM.MIX) == {"保守型", "穩健型", "積極型"} and all(abs(sum(w.values()) - 1) < 1e-9 for w in NM.MIX.values())
    for nm, w in NM.MIX.items():
        assert R["mix"][nm]["w"] == w
    assert R["mix"]["保守型"]["mdd"] > R["mix"]["100% SPY"]["mdd"]                 # 加債券黃金 → 回撤較淺


def test_etf_views_with_and_without_bundle():
    R = UE.research()
    idx = pd.bdate_range("2025-01-01", periods=300)
    C = pd.DataFrame({"SPY": np.linspace(100, 130, 300), "QQQ": np.linspace(100, 90, 300)}, index=idx)
    B = {"C": C}
    tr = UV.etf_trend(B, "IVV")                                               # IVV 沒資料 → 用 SPY
    assert tr["src"] == "SPY" and tr["above200"] and tr["r6m"] > 0
    assert not UV.etf_trend(B, "QQQ")["above200"] and UV.etf_trend(None, "GLD") == {}
    html = UV.etf_starter_html(R, B)
    assert html.count("class='kq-step'") == 5 and "IVV／VOO／SPY" in html and "00646" in html
    reason = UV.etf_reason("IEF", R["etf_stats"]["IEF"], {})
    assert reason.startswith("2004 年起") and "2008 年 +18%" in reason
    for role in NM.ROLE_ORDER:
        assert "kq-wrap" in UV.etf_role_table(R, None, role)
    assert "穩健型" not in UV.mix_html(R, "穩健型") and "IEF 20%" in UV.mix_html(R, "穩健型")
    assert "100% QQQ" in UV.mix_table(R)


def test_home_page_and_css_wired():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for s in ("def page_home", 'url_path="home", default=True', "def us_etf_tab", '"etf": "💰 ETF 推薦（先看這裡）"',
              "def us_bundle_quick", '("home", "🏠", "總覽")'):
        assert s in app
    css = ui.css(True)
    for c in (".kq-steps", ".kq-step", ".kq-home", ".kq-hcard"):
        assert c in css
    doc = (ROOT / "docs" / "v3.6_protocol.md").read_text(encoding="utf-8")
    assert "預先宣告" in doc and "## 三、結果" in doc and "巨型股趨勢" in doc and "推薦順序" in doc
