"""v3.2：AI 晶片陣營（輝達／AMD／ASIC）與陣營輪動。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import chainmap as cm
from kosto import impact as I
from kosto import supply as S


def test_camps_consistent():
    assert set(S.CAMPS) == {"nv", "amd", "asic"}
    ex = {c: {t for t, _ in S.camp_members(c, exclusive=True)} for c in S.CAMPS}
    assert not (ex["nv"] & ex["amd"]) and not (ex["nv"] & ex["asic"]) and not (ex["amd"] & ex["asic"])
    sh = set(S.shared_members())
    assert {"2330", "3711", "3037", "2383", "2368"} <= sh and not (sh & ex["nv"])
    assert {"2317", "3017"} <= ex["nv"] and {"3661", "2449"} <= ex["asic"] and "3533" in ex["amd"]
    assert S.camps_of("2330") == ["amd", "asic"] and S.camps_of("2317") == ["nv"] and S.camps_of("1101") == []
    assert {"camp_nv", "camp_amd", "camp_asic"} <= set(S.THEME_BY_ID)
    for c in S.CAMPS.values():
        assert c["color"].startswith("#") and all(len(t) == 4 for t, _ in c["members"])


def test_nvidia_scenario_has_winners_and_losers():
    sc = I.SCEN_BY_ID["nvidia_ceo"]
    assert sc["analog"] == "rival"
    dep = {g: d for g, d, *_ in sc["impacts"]}
    assert dep["server"] < 0 and dep["asic"] > 0 and dep["foundry"] == 0          # 輝達專屬受傷、ASIC 受惠、台積電中性
    camps = {c: d for c, d, *_ in sc["camps"]}
    assert camps["nv"] < 0 < camps["amd"] <= camps["asic"]
    assert "2025-11-26" in I.EVENT_SETS["rival"][1]


def _bundle():
    """80 日；第 70 日：輝達專屬股 −3%、ASIC 專屬股 +3%、其他 0；WORLD 有 NVDA、AMD。"""
    days = pd.bdate_range("2025-06-02", periods=80)
    tk = ["2317", "3017", "3661", "2449", "2330", "3711", "9999"]
    R = np.zeros((80, len(tk)))
    R[70, 0:2] = -0.03
    R[70, 2:4] = 0.03
    nv = np.r_[np.full(70, 100.0), np.full(10, 90.0)]
    amd = np.r_[np.full(70, 100.0), np.full(10, 106.0)]
    a = {"R": R.astype(np.float32), "OK": np.ones_like(R, bool), "WORLD": np.c_[nv, amd].astype(np.float32)}
    meta = {"days": [str(d.date()) for d in days], "names": tk, "world_keys": ["NVDA", "AMD"]}
    return SimpleNamespace(a=a, meta=meta, tickers=tk)


def test_camp_paths_events_and_us_moves():
    B = _bundle()
    ci = I.camp_index(B)
    assert set(ci) == {"nv", "asic", "shared"} and len(ci["nv"]) == 2           # 9999 不在任何陣營；AMD 專屬股不在資料裡
    p = I.camp_paths(B, 30)
    assert len(p) == 30 and p["asic"].iloc[-1] > 0 > p["nv"].iloc[-1]
    ev = I.event_returns(B, [B.meta["days"][70]], ci)
    sm = I.event_summary(ev)
    assert sm.loc["asic", "d0"] > 0 > sm.loc["nv", "d0"]
    us = I.us_moves(B)
    assert us.loc["NVDA", "r20"] == pytest.approx(-0.10) and us.loc["AMD", "r1"] == pytest.approx(0.0)
    assert I.us_moves(SimpleNamespace(a={}, meta={}, tickers=[])).empty
    rows, mrows = cm.camp_rows(I.SCEN_BY_ID["nvidia_ceo"]["camps"], sm, p, lambda t: t)
    assert len(rows) == 4 and "三陣營共用" in rows[-1][0] and "受傷" in rows[0][1] and len(mrows) == 4


def test_docs_and_app_wired_v32():
    root = Path(__file__).resolve().parents[1]
    notes = (root / "docs" / "v3.2_notes.md").read_text(encoding="utf-8")
    assert "模型有沒有納入" in notes and "2025-11-26" in notes
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ("def camp_section", "camp_cached", 'sc.get("camps")', "S.camps_of(t)"):
        assert s in app
