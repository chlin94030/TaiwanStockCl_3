"""v3.15：綜合第二、三份評審——產業調整後動能（前瞻觀察）、三級晉級、異常診斷、證據徽章、缺值透明、歷史名單重現。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kosto import review as rv
from kosto import tournament as T
from kosto.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_momentum_split_and_factor():
    from kosto.factors import REGISTRY
    from kosto.research import Research
    assert "t_mom_secrel" in REGISTRY and "產業" in REGISTRY["t_mom_secrel"].zh
    cfg = load_config()
    ms = cfg["research"]["momentum_split"]
    assert ms["enabled"] is False and ms["to"] == "t_mom_secrel" and ms["share"] == 0.5        # 前瞻觀察中，主模型不變
    assert "t_mom_secrel" in cfg["research"]["inactive_factors"]
    F = ["t_rs120", "t_mom120x20", "f_ep", "t_mom_secrel"]
    R = SimpleNamespace(B=SimpleNamespace(factors=F), cfg=cfg)
    w = np.array([0.4, 0.2, 0.4, 0.0], np.float32)
    assert Research.split_momentum(R, w) is w                                                   # 關閉 → 原權重
    w2 = Research.split_momentum(R, w, {**ms, "enabled": True})
    assert np.allclose(w2, [0.2, 0.1, 0.4, 0.3]) and abs(float(np.abs(w2).sum()) - 1) < 1e-6
    assert Research.split_momentum(R, None, {**ms, "enabled": True}) is None
    src = (ROOT / "kosto" / "research.py").read_text(encoding="utf-8")
    assert 'repr(sorted((r0.get("momentum_split") or {}).items()))' in src and "w = self.split_momentum(w)" in src


def test_watch_w5():
    from kosto import watch as W
    it = {x["key"]: x for x in W.ITEMS}
    assert "W5" in it and it["W5"]["hz"] == ("short", "mid", "long") and it["W5"]["kind"] == "return"
    assert W.SPLIT["to"] == "t_mom_secrel" and W.SPLIT["enabled"] is True


def _rows(champ_fwd, chal_fwd, days, champ, chal, d_ex=0.03):
    out = []
    for ph in range(3):
        out.append({"phase": ph, "h": "mid", "key": T.key(champ), "params": champ, "champion": True, "dev_excess": 0.30, "dev_ir": 0.9,
                    "dev_mdd": -0.40, "hold_cagr": 0.5, "fwd_ret": champ_fwd, "fwd_days": days})
        out.append({"phase": ph, "h": "mid", "key": T.key(chal), "params": chal, "champion": False, "dev_excess": 0.30 + d_ex,
                    "dev_ir": 0.95, "dev_mdd": -0.40, "hold_cagr": 0.5, "fwd_ret": chal_fwd, "fwd_days": days})
    return out


def test_three_tier_promotion():
    cfg = load_config()
    assert cfg["research"]["auto_forward_days"] == 60 and cfg["research"]["auto_reset"] is False
    champ = T.base_params(cfg, "mid")
    chal = T.challengers(champ)[0]
    res = {"protocol": {"last_date": "2026-12-01"}, "horizons": {"mid": {}}}
    st = {"leader": {"mid": {"key": T.key(chal), "params": chal, "streak": 9, "d_excess": 0.03, "since": "2026-11-17"}},
          "champions": {"mid": champ}, "updated": "2026-11-30"}
    st = T.update(st, _rows(np.nan, np.nan, 0, champ, chal), res, cfg, cfg)
    assert st["champions"]["mid"] == champ                                   # 不直接換上
    assert st["observe"]["mid"]["key"] == T.key(chal) and st["observe"]["mid"]["since"] == "2026-12-01"
    assert st["log"][-1]["event"] == "observe"
    res2 = {"protocol": {"last_date": "2027-03-01"}, "horizons": {"mid": {}}}
    st2 = T.update(st, _rows(0.05, 0.08, 61, champ, chal), res2, cfg, cfg)   # 前瞻 61 天領先 → 換上
    assert st2["champions"]["mid"] == chal and st2["log"][-1]["event"] == "promote" and "mid" not in st2["observe"]
    st3 = T.update(st, _rows(0.08, 0.05, 61, champ, chal), res2, cfg, cfg)   # 前瞻落後 → 退回
    assert st3["champions"]["mid"] == champ and st3["log"][-1]["event"] == "drop" and "mid" not in st3["observe"]
    st4 = T.update(st, _rows(0.05, 0.08, 30, champ, chal), res2, cfg, cfg)   # 還沒滿 60 天 → 繼續觀察
    assert st4["observe"]["mid"]["fwd_days"] == 30 and abs(st4["observe"]["mid"]["fwd_diff"] - 0.03) < 1e-9
    nav = pd.Series([1.0, 1.1, 1.2, 1.32], index=pd.bdate_range("2026-12-01", periods=4))
    r, d = T.forward_since(nav, "2026-12-02")
    assert abs(r - 0.2) < 1e-9 and d == 3 and T.forward_since(nav, "2027-01-01") == (np.nan, 0) or np.isnan(T.forward_since(nav, "2027-01-01")[0])
    cli = (ROOT / "cli.py").read_text(encoding="utf-8")
    assert 'observe=prev.get("observe")' in cli and 'cfg["research"].get("auto_reset")' in cli


def test_diagnostics():
    from kosto import diagnose as D
    picks = [{"ticker": f"{k}", "name": f"n{k}", "sector": "半導體業" if k < 5 else "航運業", "missing": ["營收年增", "盈餘殖利率"] if k == 0 else []}
             for k in range(8)]
    res = {"protocol": {"last_date": "2026-11-19"}, "data_status": {"final": False, "text": "融資券待補"},
           "horizons": {"mid": {"latest": {"picks": picks}, "chosen_pool": {"rule": "ADV", "size": 500}}}}
    log = [{"date": "2026-11-18", "h": "mid", "picks": ["0", "1", "x", "y", "z", "w", "v", "u"]}]
    fw = pd.DataFrame({"已到期": [30], "平均超額": [-0.08]}, index=["mid"])
    out = D.run(res, log, {"mid": {"rule": "BLEND", "size": 500}}, fw)
    codes = {x["code"] for x in out}
    assert codes == {"turnover", "missing", "concentration", "pool", "forward", "data"}
    assert D.run({"protocol": {"last_date": "2026-11-19"}, "horizons": {}}, [], {}, None) == []


def test_evidence_badge_and_missing():
    from kosto import ui
    assert "已回測" in ui.evidence("tested") and "未回測" in ui.evidence("ref") and "前瞻觀察" in ui.evidence("observe")
    src = (ROOT / "kosto" / "research.py").read_text(encoding="utf-8")
    assert '"missing": [fz[f] for f in np.argsort(-np.abs(fit.weights))[:12]' in src
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for fn in ("def v315_html", "def diagnostics_html", "def replay_section", "ui.evidence('tested')", "ui.evidence('ref')",
               '"missing": list(p.get("missing") or [])', "v315_html()", "replay_section()", "diagnostics_html()"):
        assert fn in app, fn


def test_v315_results_and_docs():
    c = {x[0]: x for x in rv.V315_CAND}
    assert len(c) == 10 and all("❌" in x[-1] for k, x in c.items() if k != "N3")
    n3 = c["N3"]
    assert n3[4][1] >= 1.5 and n3[5] >= 0 and min(n3[6]) > 0 and min(n3[7]) >= -1 and n3[10] > 0.10 and "前瞻觀察" in n3[-1]
    assert c["N2"][8] < -50                                                   # 動態權重 50%：最近 12 個月崩落
    assert all(min(x[6]) >= 0 and min(x[4]) < -1.5 for k, x in c.items() if k.startswith("R"))   # 少跌但少賺太多
    core = {x[0]: x for x in rv.V315_CORE}
    assert core[0.7][1][1] > core[1.0][1][1] and core[0.7][1][0] > core[1.0][1][0]             # 七成 0050：回撤較淺、年化較高
    assert 0 < rv.V315_STATS["pbo"] < 0.5 and len(rv.V315_COMPARE) >= 6 and [x for x in rv.IDEAS if x[0] == "v3.15"]
    doc = (ROOT / "docs" / "v3.15_protocol.md").read_text(encoding="utf-8")
    assert "## 二、候選（全部事先固定，不做網格）" in doc and "## 五、結果" in doc and "q > 0.10" in doc
    cfg = load_config()
    assert cfg["research"]["overlay"] == {"f_rev_rel": 0.15} and cfg["research"]["sell_block"] is True
    assert '__version__ = "3.' in (ROOT / "kosto" / "__init__.py").read_text(encoding="utf-8")
