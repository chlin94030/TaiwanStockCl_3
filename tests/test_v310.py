"""v3.10：修正期 ETF 比較（只報告）。"""
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import etf as E
from kosto import review as rv

ROOT = Path(__file__).resolve().parents[1]


def test_correction_events_and_perf():
    d = pd.bdate_range("2022-01-03", periods=600)
    m = np.r_[np.full(100, 0.002), np.full(40, -0.006), np.full(460, 0.001)]
    tr = pd.DataFrame({"0050": np.cumprod(1 + m), "00713": np.cumprod(1 + 0.5 * m), "XYZ": 1.0}, index=d)
    ev = E.correction_events(tr["0050"])
    assert len(ev) == 1 and 100 < ev[0] < 140                       # 只算一段、在下跌途中
    perf, dates = E.correction_perf(tr, tr["0050"], codes=["0050", "00713", "00646"])
    assert dates and list(perf["ticker"]) == ["0050", "00713"]        # 沒有資料的代號略過
    a, b = perf.set_index("ticker").loc["0050"], perf.set_index("ticker").loc["00713"]
    assert b["mdd"] > a["mdd"]                                       # 低 β 回落較淺
    assert E.correction_perf(tr.iloc[:50], tr["0050"].iloc[:50])[0].empty


def test_v310_results_and_app():
    t = {x[0]: x for x in rv.V310_ETF}
    assert t["0050"][2] == max(x[2] for x in rv.V310_ETF)            # 0050 之後 60 日最好
    assert t["高息低波（00713 代理）"][2] < t["0050"][2] and t["留 40% 現金（0050 30／高息低波 20／00646 10）"][4] > t["0050"][4]
    assert all(len(xs) == 6 for _, xs in rv.V310_REAL)
    assert [x for x in rv.IDEAS if x[0] == "v3.10"][0][5] == "只報告"
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def etf_correction_section" in app and "📉 修正期怎麼選" in app
    assert "## 三、結果" in (ROOT / "docs" / "v3.10_protocol.md").read_text(encoding="utf-8")
    assert "3." in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
