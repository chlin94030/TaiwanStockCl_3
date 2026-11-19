"""v3.15 更新後自動異常診斷（第二份評審 9C）：每日研究完成後檢查名單是否異常——只產生警告，不改名單、不改模型。

檢查項目：
  * 名單突變：與前一個公布日相比，換掉的比例超過門檻（短線 60%、中長線 50%）。
  * 缺值進榜：名單中有權重的因子缺 2 個以上（缺值在排名裡被當成中性，可能是資料源暫時缺漏）。
  * 族群過度集中：同一官方產業別的資金比重 ≥ 50%（名次加權）。
  * 前瞻長期落後：前瞻紀錄已到期的名單平均落後 0050 超過 5 個百分點（至少 20 筆）。
  * 資料未完整：融資券等資料尚未更新（名單可能在完整版改變）。
  * 母池更換：每日研究選定的母池與前一天不同。
"""
from __future__ import annotations

import numpy as np

CHANGE_MAX = {"short": 0.6, "mid": 0.5, "long": 0.5}
HZ = {"short": "短線", "mid": "中線", "long": "長線"}


def _rank_weights(n: int) -> np.ndarray:
    lin = np.linspace(1.5, 0.5, n) if n > 1 else np.ones(max(n, 1))
    return lin / lin.sum()


def run(res: dict, prev_log: list | None = None, prev_pools: dict | None = None, forward_table=None) -> list[dict]:
    """回傳 [{"level": "warn"|"info", "code", "h", "text"}]。prev_log＝前瞻紀錄（含今天以前公布的名單）。"""
    out = []
    today = (res.get("protocol") or {}).get("last_date")
    log = [x for x in (prev_log or []) if x.get("date") and x["date"] < str(today)]
    for h, hb in (res.get("horizons") or {}).items():
        picks = (hb.get("latest") or {}).get("picks") or []
        tk = [p["ticker"] for p in picks]
        prev = [x for x in log if x.get("h") == h]
        if prev and tk:
            last = set(prev[-1]["picks"])
            chg = 1 - len(last & set(tk)) / max(len(tk), 1)
            if chg >= CHANGE_MAX.get(h, 0.5):
                out.append({"level": "warn", "code": "turnover", "h": h,
                            "text": f"{HZ.get(h, h)}名單換掉 {chg:.0%}（{prev[-1]['date']} → {today}），超過平常水準，請先確認資料是否完整"})
        miss = [(p["ticker"], p.get("name", ""), p.get("missing") or []) for p in picks if len(p.get("missing") or []) >= 2]
        for t, nm, m in miss:
            out.append({"level": "warn", "code": "missing", "h": h,
                        "text": f"{HZ.get(h, h)} {t} {nm} 缺 {len(m)} 個有權重的因子（{'、'.join(m[:3])}），排名可能偏高或偏低"})
        if len(picks) >= 4:
            w = _rank_weights(len(picks))
            secs = {}
            for p, x in zip(picks, w):
                secs[p.get("sector") or "NA"] = secs.get(p.get("sector") or "NA", 0) + float(x)
            s, v = max(secs.items(), key=lambda kv: kv[1])
            if v >= 0.5 and s != "NA":
                out.append({"level": "warn", "code": "concentration", "h": h,
                            "text": f"{HZ.get(h, h)}名單 {s} 資金比重 {v:.0%}（≥50%），同產業會一起漲跌"})
        cp = hb.get("chosen_pool") or {}
        old = (prev_pools or {}).get(h) or {}
        if old and cp and (old.get("rule") != cp.get("rule") or int(old.get("size", -1)) != int(cp.get("size", -1))):
            out.append({"level": "info", "code": "pool", "h": h,
                        "text": f"{HZ.get(h, h)}母池由 {old.get('rule')}{old.get('size')} 換成 {cp.get('rule')}{cp.get('size')}"})
    if forward_table is not None and len(forward_table):
        for h, r in forward_table.iterrows():
            if r.get("已到期", 0) >= 20 and np.isfinite(r.get("平均超額", np.nan)) and r["平均超額"] < -0.05:
                out.append({"level": "warn", "code": "forward", "h": h,
                            "text": f"{HZ.get(h, h)}前瞻紀錄 {int(r['已到期'])} 筆到期名單平均落後 0050 {r['平均超額']:.1%}，模型可能失效，請看模型驗證頁"})
    ds = res.get("data_status") or {}
    if ds and not ds.get("final", True):
        out.append({"level": "info", "code": "data", "h": "",
                    "text": "資料尚未完整：" + str(ds.get("text", "")) + "（完整版可能小幅改變名單）"})
    return out
