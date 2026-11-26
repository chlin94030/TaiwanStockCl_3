"""v3.25：⚡ 現在最值得買（模型排名 × 即時價位）、研究時間預算、標籤修正、總檢視文件。"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from kosto import ui

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def _exec(names, g):
    tree = ast.parse(APP)
    for fn in names:
        node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign)) and getattr(n, "name", None) == fn)
        exec(ast.get_source_segment(APP, node), g)


def test_best_now_html_orders_and_explains():
    g = {"ui": ui, "pd": pd, "np": np, "NAMES": ["台積電", "奇鋐", "長榮航"], "TIDX": {"2330": 0, "3017": 1, "2618": 2},
         "RDAY": pd.Timestamp("2026-11-25"), "stock_cell": lambda t: t, "regime_line": lambda: "穩健型今天：ETF 40%"}
    exec("FIT_ORDER = " + APP.split("FIT_ORDER = ", 1)[1].split("\n", 1)[0], g)
    _exec(["best_now_html"], g)
    rows = [{"n": 1, "t": "2330", "w": 0.19, "px": 2100.0, "chg": -0.05, "lock": "", "code": "wait", "fit": "偏高・等拉回", "col": "#b45309",
             "advice": "不追", "kept": True, "flags": []},
            {"n": 2, "t": "3017", "w": 0.17, "px": 900.0, "chg": -0.08, "lock": "", "code": "dip", "fit": "便宜（拉回買點）", "col": "#15803d",
             "advice": "可承接", "kept": False, "flags": ["投信擁擠"]}]
    b = {"rows": sorted(rows, key=lambda r: (g["FIT_ORDER"][r["code"]] > 2, r["n"])), "live": True, "date": "2026-11-25", "h": "mid"}
    assert b["rows"][0]["t"] == "3017"                                                    # 能買的排前面
    h = g["best_now_html"](b)
    assert "先買" in h and "3017 奇鋐" in h and "投信擁擠" in h and "證交所即時" in h and "v1.7" in h and "ETF 40%" in h
    assert h.index("便宜（拉回買點）") < h.index("偏高・等拉回")                                  # 表格也是能買的在前
    c = g["best_now_html"](b, compact=True)
    assert "現在最值得買" in c and "./picks" in c
    b2 = {**b, "rows": [rows[0]], "live": False}
    assert "目前沒有在買進區" in g["best_now_html"](b2) and "11/25 收盤" in g["best_now_html"](b2)
    assert "今天沒有名單" in g["best_now_html"]({})


def test_wiring_budget_and_labels():
    for s in ("def best_now(", "def best_now_section", "best_now_section(h)", 'best_now_html(best_now("mid"), compact=True)',
              "def ref_flags", "def regime_line", "成交金額前 {len(top)} 大", '(RES or {}).get("skipped")'):
        assert s in APP, s
    assert "市值前 {len(top)} 大" not in APP
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    for s in ("def room(need", "KOSTO_RESEARCH_BUDGET_MIN", 'room(35, "換股日敏感度', 'room(12, "贏 0050 研究")', 'room(3, "擁擠研究")'):
        assert s in cli, s
    cfg = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    assert "time_budget_min: 120" in cfg
    doc = (ROOT / "docs" / "v3.25_notes.md").read_text(encoding="utf-8")
    for s in ("即時", "選股模型不變", "收盤進場", "時間預算", "現在最值得買"):
        assert s in doc, s
    assert '__version__ = "3.25.0"' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")


def test_room_budget_logic():
    src = (ROOT / "cli.py").read_text(encoding="utf-8")
    body = src.split("    def room(need: float, what: str) -> bool:", 1)[1].split("    hz = a.horizons", 1)[0]
    import textwrap
    import time
    code = textwrap.dedent("    def room(need: float, what: str) -> bool:" + body)
    logs, res = [], {}
    g = {"time": time, "t0": time.time() - 100 * 60, "budget": 120.0, "log": logs.append, "res": res}
    exec(code, g)
    assert g["room"](10, "A") is True and g["room"](30, "B") is False and res["skipped"] == ["B"] and "略過：B" in logs[0]
