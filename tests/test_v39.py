"""v3.9：前瞻基本面候選（只計算、不加權）、研究結果、App 接線。"""
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import review as rv
from kosto.config import load_config
from kosto.factors import REGISTRY, Ctx

ROOT = Path(__file__).resolve().parents[1]
NEW = ("f_margin_chg", "f_eps_rev", "f_garp")


def test_v39_factors_registered_inactive():
    assert all(f in REGISTRY for f in NEW)
    keys = list(REGISTRY)                                            # 放在既有因子之後註冊，不影響原本的順序
    assert keys.index("f_margin_chg") > keys.index("g_seg_breadth5") and keys.index("f_garp") == keys.index("f_margin_chg") + 2
    off = load_config()["research"]["inactive_factors"]
    assert all(f in off for f in NEW)
    assert load_config()["research"]["overlay"] == {"f_rev_rel": 0.15}   # 主模型不變


def test_v39_factor_math():
    class P:  # 最小 Panel 替身：一檔股票、300 天
        dates = pd.bdate_range("2024-01-01", periods=300)
        tickers = ["A", "B"]
        rev_m = None
        world = None

        def __init__(self):
            n = len(self.dates)
            close = pd.DataFrame({"A": np.linspace(100, 130, n), "B": 50.0}, index=self.dates)
            pe = pd.DataFrame({"A": np.r_[np.full(n - 60, 20.0), np.full(60, 13.0)], "B": np.nan}, index=self.dates)
            self.m = {"close": close, "pe": pe, "pb": pe * 0 + 2, "shares": close * 0 + 1e6,
                      "tradable": close.notna()}

        def __getitem__(self, k):
            return self.m[k]
    c = Ctx(P())
    e = REGISTRY["f_eps_rev"].func(c)
    assert e["A"].iloc[-1] > 0 and np.isnan(e["B"].iloc[-1])      # 本益比下降（EPS 上升）→ 獲利上修；虧損＝缺值


def test_v39_results_and_app():
    assert len(rv.V39_TESTS) == 3 and all("未通過" in x[-1] for x in rv.V39_TESTS)
    assert all(x[1].startswith("−") for x in rv.V39_TESTS)           # 中線開發期三個都是負的
    assert len(rv.V39_IC) == 7 and all(len(x) == 8 for x in rv.V39_IC)
    best = max(rv.V39_IC, key=lambda x: x[1])
    assert best[0].startswith("營收創新高") and best[-1] == "是"
    v39 = [x for x in rv.IDEAS if x[0] == "v3.9"]
    assert len(v39) == 1 and v39[0][5] == "❌"
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def fwd_fund" in app and "rv.V39_TESTS" in app and "rv.V39_IC" in app
    proto = (ROOT / "docs" / "v3.9_protocol.md").read_text(encoding="utf-8")
    assert "## 三、結果" in proto and "2027-10" in proto
    assert "3." in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
