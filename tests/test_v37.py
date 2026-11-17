"""v3.7：週刊觀點驗證（只報告）、資安題材、美股資安股範圍、那斯達克夜盤時鐘。"""
from pathlib import Path

import pandas as pd

from kosto import review as rv
from kosto import supply as S
from kosto.us import data as UD
from kosto.us import engine as UE
from kosto.us import names as NM
from kosto.us import view as UV

ROOT = Path(__file__).resolve().parents[1]


def test_v37_research_constants():
    assert len(rv.V37_CYBER) == 4 and all(len(x) == 6 for x in rv.V37_CYBER)
    assert len(rv.V37_IPO) == 11 and all(len(x) == 8 for x in rv.V37_IPO)
    avg = next(x for x in rv.V37_IPO if x[0] == "平均"); base = next(x for x in rv.V37_IPO if x[0].startswith("平常"))
    assert abs(avg[4] - base[4]) < 3          # 美股 IPO 後的回落與平常差不到 3 個百分點 → 不成立
    mag = rv.V37_REBOUND[0]; app_ = rv.V37_REBOUND[1]; etf = rv.V37_REBOUND[-1]
    assert mag[2] < app_[2] and mag[4] < app_[4] and mag[2] < etf[2]   # 週刊組合 60 日兩段都輸反彈候選 → 不採用
    v37 = [x for x in rv.IDEAS if x[0] == "v3.7"]
    assert len(v37) == 3 and not any(x[5].startswith("✅") for x in v37)


def test_cyber_theme_and_us_set():
    th = S.THEME_BY_ID["cyber"]
    assert "7765" in dict(th["members"]) and "6690" in dict(th["members"])
    assert set(NM.CYBER_NOW) >= {"CRWD", "PANW", "FTNT", "ZS", "NET"} and "cyber" in NM.SETS and "cyber" in UV.SET_ZH
    for t in NM.CYBER_NOW:
        assert NM.zh_name(t) and NM.zh_name(t) != t or t in NM.EXTRAS
    assert {"HACK", "CIBR", "BUG"} <= set(NM.ETFS)
    tk = pd.Index(["CRWD", "AAPL", "ZS"])
    B = {"tickers": tk, "universe": pd.Series(True, tk), "info": pd.DataFrame({"sp500": [True, True, False]}, index=tk), "umask": {}}
    m = UE.mask(B, "cyber")                    # 舊資料包沒有 cyber 欄位 → 用名單
    assert m.tolist() == [True, False, True]


def test_overnight_clock():
    ny = "America/New_York"
    before = UD.market_clock(pd.Timestamp("2026-11-17 22:00", tz=ny))
    assert before["phase"] == "closed" and not before["overnight_live"]
    on = UD.market_clock(pd.Timestamp("2026-12-07 23:30", tz=ny))           # 週一晚上
    assert on["phase"] == "overnight" and UD.PHASE_ZH["overnight"] == "夜盤"
    assert UD.market_clock(pd.Timestamp("2026-12-06 21:30", tz=ny))["phase"] == "overnight"   # 週日晚上開始
    assert UD.market_clock(pd.Timestamp("2026-12-11 22:00", tz=ny))["phase"] == "closed"      # 週五晚上休息
    assert UD.market_clock(pd.Timestamp("2026-12-08 03:30", tz=ny))["phase"] == "overnight"
    assert UD.market_clock(pd.Timestamp("2026-12-08 05:00", tz=ny))["phase"] == "pre"
    assert on["night_tpe"] == "10:00"                                        # 冬令：台北 10:00


def test_docs_and_app_wired():
    proto = (ROOT / "docs" / "v3.7_protocol.md").read_text(encoding="utf-8")
    assert "## 四、結果" in proto and "不改主模型" in proto
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def v37_html" in app and "v37_html()" in app and "overnight" in app
    assert (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8").count("3.") >= 1
