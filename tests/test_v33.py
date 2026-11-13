"""v3.3：科斯托蘭尼雞蛋階段（B3 修正）、對消息的反應、貨幣＋心理溫度計、原著原則對照、利率資料。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import egg
from kosto.config import STAGES


def _bundle(n=400):
    """n 日：前 250 日緩漲（A2），之後 40 日重挫 25%（原分類 B2），最後 30 日反彈；WORLD：SOX、TNX、TWD、VIX。"""
    days = pd.bdate_range("2024-01-01", periods=n)
    r = np.r_[np.full(250, 0.001), np.full(40, -0.0072), np.full(n - 290, 0.004)]
    idx = np.cumprod(1 + r)
    dd = idx / np.maximum.accumulate(idx) - 1
    md = np.zeros((n, 11))
    md[:, 6] = 0.5                                                  # confidence
    md[:, 7] = idx; md[:, 8] = dd; md[:, 9] = 1.0                  # index, drawdown, turnover_ratio
    md[:, 10] = np.where(dd < -0.2, 0.15, 0.6)                     # breadth
    code = np.where(dd < -0.1, 5, 2).astype(np.int8)               # 重挫期原分類＝B2
    R = np.tile(r[:, None], (1, 4)).astype(np.float32)
    hit = (np.arange(n) % 17 == 0) & (np.arange(n) > 0)
    sox = 100 * np.cumprod(1 + np.where(hit, -0.03, 0.002))
    R[hit] = 0.001                                                  # 費半大跌那幾天台股沒跌（利空不跌）
    tnx = np.linspace(3.5, 5.5, n); twd = np.linspace(29, 34, n); vix = np.full(n, 35.0)
    a = {"MKT_DAILY": md.astype(np.float32), "MKT_DAILY_CODE": code, "R": R,
         "WORLD": np.c_[sox, tnx, twd, vix].astype(np.float32)}
    meta = {"days": [str(d.date()) for d in days], "world_keys": ["SOX", "TNX", "TWD", "VIX"], "rules": []}
    return SimpleNamespace(a=a, meta=meta, tickers=["1", "2", "3", "4"], sig_pos=np.arange(0, n, 5))


def test_market_stage_b3_fix_and_forward():
    B = _bundle()
    md = egg.market_stage(B)
    assert set(md["stage_model"].unique()) <= {"A2", "B2"}
    assert md["b3_fix"].any() and (md.loc[md["b3_fix"], "stage"] == "B3").all()
    assert (md.loc[md["b3_fix"], "stage_model"] == "B2").all()                  # 原分類把重挫判成 B2
    assert not md.loc[md["drawdown"] > -0.15, "b3_fix"].any()
    fw = egg.stage_forward(B, since="2024-01-01")
    assert list(fw.index) == list(STAGES) + ["全部"] and fw.loc["全部", "n"] > 0
    assert fw.loc["B3", "n"] > 0 and fw.loc["B3", "f20"] > 0                     # 恐慌之後反彈
    assert egg.market_stage(SimpleNamespace(a={}, meta={})).empty


def test_news_reaction_and_money_psych():
    B = _bundle()
    nr = egg.news_reaction(B, window=40)
    assert nr["bad_days"] >= 2 and len(nr["bad_no_react"]) == nr["bad_days"]
    assert nr["good_no_react"] == [] and isinstance(nr["oversold_signal"], bool)
    assert egg.news_reaction(SimpleNamespace(a={}, meta={})) == {}
    mp = {nm: (k, v, t) for k, nm, v, txt, t in egg.money_psych(B)}
    assert mp["美國 10 年期公債殖利率"][0] == "貨幣" and mp["美國 10 年期公債殖利率"][2] == -1   # 殖利率上升 → 偏空
    assert mp["美元／台幣"][2] == -1 and mp["VIX 恐慌指數"][2] == 1                          # 台幣貶值偏空、VIX 恐慌偏多（逆向）
    assert "廣度（站上季線的股票）" in mp


def test_egg_svg_and_principles():
    svg = egg.egg_svg("B3", {"B3": 0.7, "A2": 0.2})
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "drop-shadow" in svg and "0.70" in svg
    assert svg.count("<circle") == 6
    assert set(egg.STAGE_INFO) == set(STAGES)
    assert [egg.STAGE_INFO[k]["act"] for k in ("A1", "B3")] == ["買入", "買入"]
    assert all(len(p) == 4 for p in egg.PRINCIPLES) and len(egg.PRINCIPLES) >= 10
    assert any("B3" in p[0] for p in egg.PRINCIPLES) and any(p[3].startswith("❌") for p in egg.PRINCIPLES)


def test_rates_in_world_fetch():
    from kosto.sources import world as W
    assert {"TNX", "IRX", "DXY"} <= set(W.RATE_KEYS) and W.SYMBOLS["TNX"] == "^TNX" and "TNX" in W.US
    src = (Path(__file__).resolve().parents[1] / "kosto" / "build.py").read_text(encoding="utf-8")
    assert "W.RATE_KEYS" in src


def test_docs_and_app_wired_v33():
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "v3.3_kostolany_review.md").read_text(encoding="utf-8")
    assert "## 二、結果" in doc and "B3" in doc and "預先宣告" in doc
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ("def kostolany_tab", '"🥚 科斯托蘭尼"', "def stock_egg_html", "max_entries=1)\ndef load_github", "live_n"):
        assert s in app
    assert "其餘放現金" not in app
