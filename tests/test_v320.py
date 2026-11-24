"""v3.20：挑戰架構結果、五種換股日共識組合（phases.current_book／consensus）、App。"""
from pathlib import Path

import pandas as pd

from kosto import phases as PH
from kosto import review as rv

ROOT = Path(__file__).resolve().parents[1]


def test_current_book_and_consensus():
    tr = pd.DataFrame({"i": [0, 1, 2, 3, 4, 5], "picks": [["A", "B"], ["C"], ["A", "D"], ["E", "F"], ["A", "B"], []]})
    b = PH.current_book(tr, 4)                                  # 最近 4 批：[A,D]、[E,F]、[A,B]、[]（空批不計）
    assert set(b) == {"A", "B", "D", "E", "F"} and abs(sum(b.values()) - 1) < 1e-9
    assert abs(b["A"] - 1 / 3) < 1e-9 and abs(b["E"] - 1 / 6) < 1e-9
    assert PH.current_book(None, 4) == {} and PH.current_book(tr.iloc[:0], 4) == {}
    cs = PH.consensus({"mid": {0: {"A": 0.5, "B": 0.5}, 1: {"A": 1.0}, 2: {}}, "long": {0: {}}})
    assert "long" not in cs and cs["mid"]["n_phases"] == 2
    rows = {r["ticker"]: r for r in cs["mid"]["rows"]}
    assert rows["A"]["weight"] == 0.75 and rows["A"]["phases"] == 2 and rows["B"]["weight"] == 0.25 and cs["mid"]["rows"][0]["ticker"] == "A"


def test_phase_check_records_books_signature():
    import inspect
    assert "books" in inspect.signature(PH.phase_check).parameters
    src = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert 'res["consensus"] = consensus(books)' in src and "books=books" in src


def test_v320_results_docs_app():
    assert [x[0] for x in rv.V320_TW] == ["M1", "M2", "M4", "S1"]
    assert all(x[-1].startswith("❌") for x in rv.V320_TW[:3]) and all(x[2][1] < 0 for x in rv.V320_TW[:3])
    assert all(x[2] < 0 for x in rv.V320_US)
    for h, pool, avg, worst, blend, *_ in rv.V320_BLEND:
        assert abs(blend - avg) <= 1.0 and blend >= worst                 # 共識＝平均，且不會比最差的換股日差
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def v320_html", "v320_html()", "def consensus_html", "consensus_html(h)"):
        assert fn in app, fn
    doc = (ROOT / "docs" / "v3.20_protocol.md").read_text(encoding="utf-8")
    assert "## 四之二、第二階段" in doc and "## 五、結果（2026-11-22）" in doc and "（待補）" not in doc
    import re
    v = re.search(r'__version__ = "3\.(\d+)\.', (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8"))
    assert v and int(v.group(1)) >= 20
