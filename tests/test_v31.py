"""v3.1：名次×波動資金分配（選用）、台股熱力圖、今日行動看板、導覽分組。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import chainmap as cm
from kosto import plan
from kosto import ui


def test_rank_invvol_weights():
    w = plan.stock_weights(4, "mid", "rank_invvol", vols=[0.02, 0.02, 0.02, 0.02])
    assert w.sum() == pytest.approx(1) and np.all(np.diff(w) < 0)                  # 波動相同 → 等同名次加權
    assert w == pytest.approx(plan.stock_weights(4, "mid"))
    w2 = plan.stock_weights(4, "mid", "rank_invvol", vols=[0.06, 0.01, 0.02, 0.02])
    assert w2.sum() == pytest.approx(1) and w2[0] < w2[1] and w2.max() <= 2 / 4 + 1e-9     # 波動大少放、單檔上限 2/n
    w3 = plan.stock_weights(3, "mid", "rank_invvol", vols=[np.nan, 0.02, 0.02])
    assert np.isfinite(w3).all() and w3.sum() == pytest.approx(1)
    assert plan.stock_weights(3, "mid", "rank_invvol", vols=None) == pytest.approx(plan.stock_weights(3, "mid"))


def test_backtest_rank_invvol_runs():
    import importlib
    te = importlib.import_module("tests.test_engine") if Path("tests/test_engine.py").exists() else None
    if te is None:
        pytest.skip("engine toy bundle not importable")
    from kosto.backtest import Spec, run
    B, view, W = te._toy_bundle()
    B.a["VOL60"] = np.full((len(B.sig_pos), B.a["R"].shape[1]), 0.02, np.float32)
    eq = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0))
    rv = run(B, view, W, Spec(H=5, step=5, top_k=1, buy_cost=0, sell_cost=0, weighting="rank_invvol"))
    assert rv.nav.iloc[-1] == pytest.approx(eq.nav.iloc[-1], rel=1e-9)            # 一檔 → 與等權相同


def test_treemap_frame_and_figure():
    tk = ["2330", "3017", "2881", "9999"]
    met = pd.DataFrame({"r1": [0.01, -0.02, 0.0, 0.03], "r5": [0.02, -0.05, 0.01, 0.0], "r20": [0.1, -0.1, 0.0, 0.05],
                        "r60": [0.2, 0.0, 0.0, 0.1], "amt20": [300.0, 50.0, 20.0, 5.0], "name": ["台積電", "奇鋐", "富邦金", "測試-KY"]},
                       index=tk)
    B = SimpleNamespace(tickers=tk, meta={"sectors": ["半導體業", "電腦及週邊設備業", "金融保險業", "電子零組件業"]})
    labels = {"9999": {"group": "電子中游-印刷電路板", "segment": "PCB"}, "2881": {"group": "金融-金融", "segment": ""}}
    df = cm.treemap_frame(B, met, labels, top=10)
    assert len(df) == 4
    d = df.set_index("t")
    assert d.loc["2330", "board"].startswith("電子｜") and d.loc["3017", "group"] == "散熱（氣冷・液冷）"
    assert d.loc["2881", "board"] == "非電子" and d.loc["9999", "group"] == "印刷電路板（PCB）"
    assert len(cm.treemap_frame(B, met, labels, top=10, elec_only=True)) == 3
    assert cm.treemap_frame(B, pd.DataFrame(), labels).empty
    pytest.importorskip("plotly")
    fig = cm.treemap_figure(df, "r20", dark=True)
    tr = fig.data[0]
    assert tr.type == "treemap" and "" in tr.parents and len(tr.ids) == len(set(tr.ids))
    assert "測試" in tr.labels and all(not str(x).endswith("-KY") for x in tr.labels)


def test_action_board_and_schip():
    c = ui.schip("2330", "台積電", "資金 18.8%")
    assert "href='./stock?s=2330'" in c and "<b>2330</b>" in c and "資金 18.8%" in c
    h = ui.action_board("今天開盤要做的事", "名單日", [("買進", "#3b82f6", [c], "無"), ("賣出", "#f0464b", [], "今天沒有要賣的")])
    assert h.count("class='col'") == 2 and "今天沒有要賣的" in h and "1 檔" in h and "0 檔" in h


def test_app_wiring_v31():
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ('"更多": [', "def heatmap_section", "ui.action_board(", "rank_invvol", "rv.V31_TESTS"):
        assert s in app
    proto = (root / "docs" / "v3.1_protocol.md").read_text(encoding="utf-8")
    assert "## 結果" in proto and "主模型維持 v3.0" in proto
