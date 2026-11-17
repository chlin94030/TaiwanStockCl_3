"""v3.8：多檔對照、核能／綠電題材、美股電力範圍、選舉研究（只報告）。"""
from pathlib import Path

import pandas as pd

from kosto import review as rv
from kosto import supply as S
from kosto.us import engine as UE
from kosto.us import names as NM
from kosto.us import view as UV

ROOT = Path(__file__).resolve().parents[1]


def test_v38_themes_and_sets():
    assert {"smr", "green"} <= set(S.THEME_BY_ID)
    assert "1605" in dict(S.THEME_BY_ID["smr"]["members"]) and "2308" in dict(S.THEME_BY_ID["green"]["members"])
    assert S.THEME_BY_ID["smr"]["stage"] == "題材"
    assert "power" in NM.SETS and "power" in UV.SET_ZH and {"CEG", "OKLO", "BWXT", "FSLR"} <= set(NM.POWER_NOW)
    assert {"BWXT", "LEU", "NNE", "FLNC"} <= set(NM.EXTRAS)
    tk = pd.Index(["CEG", "AAPL", "OKLO"])
    B = {"tickers": tk, "universe": pd.Series(True, tk), "info": pd.DataFrame({"sp500": [True, True, False]}, index=tk), "umask": {}}
    assert UE.mask(B, "power").tolist() == [True, False, True]


def test_v38_election_report_only():
    d = rv.V38_ELECT[0]; r = rv.V38_ELECT[1]
    assert d[1] == 4 and r[1] == 7 and d[4] > r[4]
    v38 = [x for x in rv.IDEAS if x[0] == "v3.8"]
    assert len(v38) == 1 and not v38[0][5].startswith("✅") and len(v38[0]) == 7


def test_v38_app_wired():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def parse_codes" in app and "def multi_check_html" in app and "多檔對照" in app and "V38_ELECT" in app
    assert "3." in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
    assert "## 二、結果" in (ROOT / "docs" / "v3.8_protocol.md").read_text(encoding="utf-8")
