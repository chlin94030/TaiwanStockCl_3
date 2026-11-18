"""v3.11：轉折點拆解、持股處理建議、更新時間上限、燈號文字。"""
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import freshness as F
from kosto import review as rv
from kosto import turning as TP
from kosto.config import load_config
from kosto.plan import Plan, entry_advice, holding_advice

ROOT = Path(__file__).resolve().parents[1]


def test_turning_episodes_and_decompose():
    d = pd.bdate_range("2024-01-01", periods=200)
    b = pd.Series(np.r_[np.linspace(100, 120, 50), np.linspace(120, 96, 20), np.linspace(96, 130, 130)], index=d)
    m = b ** 1.5 / b.iloc[0] ** 0.5
    ep = TP.episodes(b)
    assert len(ep) == 1 and ep[0][0] == d[50] and ep[0][1] == d[69] and ep[0][2] is not None
    D = TP.decompose({"mid": pd.DataFrame({"策略": m, "0050": b})})
    r = D.iloc[0]
    assert abs(r["b_dn"] + 0.2) < 1e-9 and r["mid_dn"] < r["b_dn"] and not r["ongoing"]   # 高 β：跌得比 0050 多
    D2 = TP.decompose({"mid": pd.DataFrame({"策略": m.iloc[:80], "0050": b.iloc[:80]})})
    assert bool(D2.iloc[-1]["ongoing"])


def test_holding_and_entry_advice():
    pl = Plan(100, 97, 101, 103, 90, 95, 110, 0.02, 55, 0.02, False)
    assert holding_advice(100, 80, pl, "pick", "buy")[0] == "add"
    assert holding_advice(85, 100, pl, "keep", "warn")[0] == "trim"
    assert holding_advice(70, 100, pl, "out", "none")[1] == "停損換股"
    assert holding_advice(140, 100, pl, "na", "none")[1] == "分批停利"
    assert holding_advice(100, np.nan, pl, "out")[1] == "下次檢查日換股"
    assert holding_advice(100, np.nan, None, "na")[0] == "own"
    assert "103" in entry_advice("wait", pl) and "不追" in entry_advice("wait", pl)


def test_freshness_late_evening_and_budget():
    tz = "Asia/Taipei"
    run = {"status": "completed", "conclusion": "success", "created": "2026-11-17T00:53:00Z", "event": "push", "url": "u"}
    lv, tx, _ = F.status(pd.Timestamp("2026-11-17 23:16", tz=tz), pd.Timestamp("2026-11-16"), run)
    assert lv == "red" and "21:40" in tx and "立即更新" in tx and "會補上" not in tx
    assert F.status(pd.Timestamp("2026-11-17 22:10", tz=tz), pd.Timestamp("2026-11-16"), run)[0] == "amber"
    assert F.status(pd.Timestamp("2026-11-17 21:04", tz=tz), pd.Timestamp("2026-11-16"), run)[0] == "amber"
    assert load_config()["data"]["daily_budget_min"] <= 90
    assert "daily_budget_min" in (ROOT / "cli.py").read_text(encoding="utf-8")


def test_v311_results_and_app():
    t = {x[0]: x for x in rv.V311_OVERLAY}
    for k in ("P2", "P3", "P4"):
        assert t[k][2][0] < t["P0"][2][0] and "❌" in t[k][-1]           # 新規則中線報酬都較低
    assert t["P1"][2][0] >= t["P0"][2][0]
    assert len(rv.V311_ANGLES) >= 6 and [x for x in rv.IDEAS if x[0] == "v3.11"][0][5] == "❌"
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def position_section" in app and "def turning_html" in app and "turning_html()" in app
    assert "## 四、結果" in (ROOT / "docs" / "v3.11_protocol.md").read_text(encoding="utf-8")
    assert '__version__ = "3.' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
