"""v2.3 限空令／空單／權重平衡：研究常數的一致性與即時小工具。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kosto import review as rv


def _bundle():
    """3 檔、70 日：A 融券大且最近 10 日大增；B 融券小；C 融券 0。"""
    D, n = 70, 3
    V = np.full((D, n), 1000.0)                       # 張
    C = np.tile(np.linspace(100, 90, D)[:, None], (1, n)).astype(float)
    C[:, 2] = np.linspace(50, 60, D)
    SB = np.zeros((D, n))
    SB[:, 0] = 2000.0
    SB[-10:, 0] = np.linspace(2100, 3000, 10)         # 10 日內 +1000 張（= 均量 100%）
    SB[:, 1] = 600.0
    MB = np.full((D, n), 4000.0)
    MB[:, 1] = 6000.0
    a = {"Q_SB": SB.astype(np.float32), "Q_MB": MB.astype(np.float32), "Q_V": V.astype(np.float32), "Q_C": C.astype(np.float32)}
    return SimpleNamespace(a=a)


def test_short_snapshot_ratio_and_increase():
    B = _bundle()
    s = rv.short_snapshot(B, [0, 1, 2])
    assert s[0]["ratio"] == pytest.approx(3000 / 4000)
    assert s[0]["inc10"] == pytest.approx((3000 - 2000) / 1000, rel=1e-6)
    assert s[1]["ratio"] == pytest.approx(0.1) and s[1]["inc10"] == pytest.approx(0.0)
    assert s[2]["ratio"] == 0.0
    assert rv.short_snapshot(SimpleNamespace(a={}), [0]) == {}


def test_short_crowded_filters_and_sorts():
    df = rv.short_crowded(_bundle(), n=5, top=3, min_sbal=500)
    assert list(df["j"]) == [0, 1]                    # C 沒有融券 → 排除；A 券資比最高排第一
    assert df["r20"].iloc[0] < 0 and df["d60"].iloc[0] < 0
    assert len(rv.short_crowded(_bundle(), min_sbal=1000)) == 1


def test_family_shares():
    fs = rv.family_shares({"t_rs120": 0.2, "t_lowidio": 0.1, "f_ep": -0.3, "c_x": 0.4}, ["t_rs120", "t_lowidio", "f_ep", "c_x"],
                          ["T", "T", "F", "C"])
    assert fs["T"] == pytest.approx(0.3) and fs["F"] == pytest.approx(0.3) and fs["C"] == pytest.approx(0.4) and fs["K"] == 0
    assert fs["mom"] == pytest.approx(0.2) and fs["top"] == "c_x"
    assert rv.family_shares({}, [], []) == {}


def test_research_constants_consistent():
    data = [e for e in rv.BAN_HISTORY if e["src"] == "data"]
    assert [e["date"] for e in data] == ["2020-03-20", "2022-10-24", "2025-04-07"]
    for e in data:
        assert {"dd", "day0", "r5", "r20", "r60", "r250", "min60", "low"} <= set(e)
        assert e["min60"] <= 0 and e["r60"] > 0
    assert rv.BAN_2026["same_as"] == "2025-04-07" and len(rv.BAN_2026["measures"]) == 5
    assert all(len(x) == 4 and all(len(e) == 4 for e in x[1:]) for x in rv.BAN_XS)
    assert sum(n for _, n, *_ in rv.LEAD_BUCKETS) <= 1870
    assert len(rv.V23_TESTS) == 4 and all("未通過" in t[-1] for t in rv.V23_TESTS)
    assert all(len(x) == 7 for x in rv.IDEAS)
    # v2.0 以後使用者提問引發的想法都沒有改變主模型
    assert not any(x[2] == "使用者提問" and x[5].startswith("✅") for x in rv.IDEAS)


def test_docs_and_app_wired():
    root = Path(__file__).resolve().parents[1]
    proto = (root / "docs" / "v2.3_protocol.md").read_text(encoding="utf-8")
    assert "## 結果" in proto and "主模型不調整" in proto
    assert (root / "docs" / "v2.3_model_review.md").is_file()
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ("def ban_html", "def ban_xs_html", "def lead_section", "def balance_html", "short_crowded_df", "rv.V23_TESTS"):
        assert s in app
