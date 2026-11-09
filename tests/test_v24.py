"""v2.4 驗效與偏移檢查：母池沿用、自動晉級風險護欄、族群占比含疊加。"""
import numpy as np
import pandas as pd
import pytest

from kosto import tournament as T
from kosto.research import choose_pool


def _pt():
    return pd.DataFrame({"rule": ["MCAP", "BLEND", "ADV", "MCAP"], "size": [500, 500, 500, 300], "score": [0.935, 0.917, 0.90, 0.80]})


def test_choose_pool_keeps_incumbent_within_margin():
    r, s, n = choose_pool(_pt(), {"rule": "BLEND", "size": 500}, 0.05)
    assert (r, s) == ("BLEND", 500) and n["kept"] and n["best"] == "MCAP500"
    r, s, n = choose_pool(_pt(), {"rule": "MCAP", "size": 300}, 0.05)      # 差 0.135 ≥ 0.05 → 換成第一名
    assert (r, s) == ("MCAP", 500) and not n["kept"]
    assert choose_pool(_pt(), None, 0.05)[:2] == ("MCAP", 500)
    assert choose_pool(_pt(), {"rule": "BLEND", "size": 500}, 0.0)[:2] == ("MCAP", 500)   # 0＝舊規則
    assert choose_pool(_pt(), {"rule": "KOSTO", "size": 150}, 0.05)[:2] == ("MCAP", 500)  # 前一天的母池今天不在表內


def _rows(d_mdd):
    champ = {"k": 5, "hyp_blend": 0.5, "growth_blend": 0.3, "hysteresis": 2.0, "top_k": 8}
    chal = dict(champ, top_k=6)
    rows = []
    for ph in range(5):
        rows.append({"phase": ph, "h": "mid", "key": T.key(champ), "params": champ, "champion": True, "dev_excess": 0.30,
                     "dev_ir": 1.0, "dev_cagr": 0.5, "dev_mdd": -0.40, "hold_cagr": 0.3})
        rows.append({"phase": ph, "h": "mid", "key": T.key(chal), "params": chal, "champion": False, "dev_excess": 0.36,
                     "dev_ir": 1.06, "dev_cagr": 0.56, "dev_mdd": -0.40 + d_mdd, "hold_cagr": 0.32})
    return rows, champ, chal


def _cfg(tol):
    return {"research": {"auto_promote": True, "auto_min_gain": 0.02, "auto_streak": 10, "auto_cooldown": 20, "auto_max_mdd_worse": tol, "top_k": 8},
            "horizons": {"mid": {"model": {"k": 5, "hyp_blend": 0.5, "growth_blend": 0.3}, "hysteresis": 2.0, "top_k": 8}}}


def test_tournament_risk_guard_blocks_deeper_drawdown():
    res = {"protocol": {"last_date": "2026-11-06"}, "horizons": {"mid": {}}}
    rows, champ, chal = _rows(-0.031)                     # 6 檔：超額 +6pp、IR 較高，但回撤深 3.1pp
    prev = {"leader": {"mid": {"key": T.key(chal), "params": chal, "streak": 6, "d_excess": 0.083, "since": "2026-10-29"}}}
    st = T.update(prev, rows, res, _cfg(0.02), _cfg(0.02))
    assert st["leader"]["mid"] == {}                      # 不再是領先者，連續天數歸零
    assert st["risk_blocked"]["mid"][0]["d_mdd"] == pytest.approx(-0.031)
    assert st["rule"]["max_mdd_worse"] == 0.02
    st2 = T.update(prev, rows, res, _cfg(None), _cfg(None))  # null＝v1.9 原規則
    assert st2["leader"]["mid"]["streak"] == 7
    rows3, _, _ = _rows(-0.01)                            # 回撤只深 1pp → 仍合格
    st3 = T.update(prev, rows3, res, _cfg(0.02), _cfg(0.02))
    assert st3["leader"]["mid"]["key"] == T.key(chal)


def test_pool_state_since():
    res = {"protocol": {"last_date": "2026-11-09"}, "horizons": {"mid": {"chosen_pool": {"rule": "BLEND", "size": 500}},
                                                                  "short": {"chosen_pool": {"rule": "ADV", "size": 500}}}}
    st = T.pool_state({"mid": {"rule": "BLEND", "size": 500, "since": "2026-11-02"}, "short": {"rule": "BLEND", "size": 500, "since": "2026-11-05"}}, res)
    assert st["mid"]["since"] == "2026-11-02" and st["short"] == {"rule": "ADV", "size": 500, "since": "2026-11-09"}


def test_audit_constants():
    from kosto import review as rv
    assert rv.TOPK_6_VS_8["dev_mdd"] < -0.02 < rv.TOPK_6_VS_8["dev_excess"]
    assert [x[0] for x in rv.MDD_DRIFT_LONG][0] == "v1.4" and rv.MDD_DRIFT_LONG[-1][3] < rv.MDD_DRIFT_LONG[0][3]
    assert all(len(x) == 7 for x in rv.STYLE_DRIFT_MID)
