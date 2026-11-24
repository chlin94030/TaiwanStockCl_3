"""v3.22 盤中即時分析（純函式，App 共用；報價來源見 quotes.py：證交所基本市況報導，約 5 秒延遲）。

* group_live：產業鏈細族群的即時強弱——成交金額加權漲跌、中位數、漲跌家數、漲跌停、量比、領漲／領跌。
* breadth：一籃股票的漲跌家數。
* chain_live_html／strip_html：產業鏈地圖「盤中即時」分頁與總覽「現在盤勢」列的 HTML。
* news_impact：輝達（或其他美股）即時漲跌 × 台股族群的輝達特有 β → 族群的預估反應（v3.21 研究的 β）。
沒有即時報價（非交易時間、連不上證交所）時，用研究資料最後一天的收盤（標日期），不會出錯。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ui

OPEN, CLOSE = (9, 0), (13, 30)


def session_frac(now: pd.Timestamp) -> float:
    """台股盤中已經過的比例（9:00→0、13:30→1）；用來把「到目前的成交量」換算成全天的量比。"""
    t = now.hour * 60 + now.minute
    a, b = OPEN[0] * 60 + OPEN[1], CLOSE[0] * 60 + CLOSE[1]
    return float(min(max((t - a) / (b - a), 0.05), 1.0))


def fallback_quotes(met: pd.DataFrame) -> pd.DataFrame:
    """非交易時間：用研究資料最後一天（收盤）的漲跌當作「今日」。"""
    if met is None or "r1" not in met:
        return pd.DataFrame(columns=["chg", "volume", "lock"])
    r = met["r1"].astype(float)
    return pd.DataFrame({"chg": r, "volume": np.nan, "lock": np.where(r >= 0.095, "up", np.where(r <= -0.095, "down", ""))}, index=met.index)


def breadth(q: pd.DataFrame, tickers=None) -> dict:
    x = q if tickers is None else q.reindex([t for t in tickers if t in q.index])
    c = x["chg"].astype(float).dropna() if "chg" in x else pd.Series(dtype=float)
    lk = x["lock"].astype(str) if "lock" in x else pd.Series("", index=x.index)
    return {"n": int(len(c)), "up": int((c > 0).sum()), "down": int((c < 0).sum()), "flat": int((c == 0).sum()),
            "lu": int((lk == "up").sum()), "ld": int((lk == "down").sum()), "med": float(c.median()) if len(c) else np.nan}


def group_live(q: pd.DataFrame, subgroups: dict, met: pd.DataFrame | None = None, vol20: pd.Series | None = None,
               frac: float = 1.0, names: dict | None = None) -> pd.DataFrame:
    """每個細族群：有報價的檔數、成交金額加權漲跌（權重＝近 20 日均成交金額）、中位數、漲跌家數、漲跌停、量比、領漲／領跌各 2 檔。"""
    names = names or {}
    rows = []
    for g, d in subgroups.items():
        ts = [t for t, _ in d["members"]]
        x = q.reindex([t for t in ts if t in q.index])
        c = x["chg"].astype(float) if "chg" in x else pd.Series(dtype=float)
        c = c.dropna()
        if not len(c):
            rows.append({"gid": g, "name": d["name"], "layer": d["layer"], "n": len(ts), "nq": 0})
            continue
        w = (met["amt20"].reindex(c.index).astype(float).fillna(0) if met is not None and "amt20" in met else pd.Series(1.0, index=c.index))
        w = w.where(w > 0, 0)
        wavg = float(np.average(c, weights=w)) if w.sum() > 0 else float(c.mean())
        vr = np.nan
        if vol20 is not None and "volume" in x and x["volume"].notna().any():
            v, a = x["volume"].astype(float), vol20.reindex(x.index).astype(float)
            ok = v.notna() & a.notna() & (a > 0)
            if ok.any():
                vr = float(v[ok].sum() / (a[ok].sum() * max(frac, 0.05)))
        lk = x["lock"].astype(str) if "lock" in x else pd.Series("", index=x.index)
        srt = c.sort_values()
        rows.append({"gid": g, "name": d["name"], "layer": d["layer"], "n": len(ts), "nq": int(len(c)), "chg": wavg,
                     "med": float(c.median()), "up": int((c > 0).sum()), "down": int((c < 0).sum()),
                     "lu": int((lk == "up").sum()), "ld": int((lk == "down").sum()), "vr": vr,
                     "lead": [(t, float(srt[t]), names.get(t, "")) for t in srt.index[::-1][:2] if srt[t] > 0],
                     "lag": [(t, float(srt[t]), names.get(t, "")) for t in srt.index[:2] if srt[t] < 0],
                     "r20": float(met["r20"].reindex(c.index).astype(float).mean()) if met is not None and "r20" in met else np.nan})
    df = pd.DataFrame(rows).set_index("gid")
    return df.sort_values("chg", ascending=False, na_position="last") if "chg" in df else df


def _who(xs) -> str:
    return "、".join(f"{ui.esc(n or t)} {c * 100:+.1f}%" for t, c, n in (xs or [])) or "–"


def chain_live_html(gl: pd.DataFrame, idx_chg: float | None, br: dict, layer_name: dict, label: str, top: int = 5) -> str:
    """產業鏈地圖「盤中即時」：最強／最弱細族群卡片、各層平均、完整表。"""
    if gl is None or not len(gl) or "chg" not in gl or gl["chg"].notna().sum() == 0:
        return "<div class='kq-muted'>目前沒有可用的報價（盤中 9:00～13:30 每 30 秒更新；非交易時間顯示最近收盤）。</div>"
    g = gl[gl["chg"].notna()].copy()
    g[["up", "down", "lu", "ld"]] = g[["up", "down", "lu", "ld"]].fillna(0).astype(int)
    rel = (lambda v: v - idx_chg) if idx_chg is not None and np.isfinite(idx_chg) else (lambda v: np.nan)
    h = (f"<div class='kq-muted' style='margin-bottom:6px'>{ui.esc(label)}"
         + (f"｜加權 {ui.pct(idx_chg, 2, color=True)}" if idx_chg is not None and np.isfinite(idx_chg) else "")
         + f"｜產業鏈代表股 {br['n']} 檔：上漲 {br['up']}、下跌 {br['down']}"
         + (f"、漲停 {br['lu']}" if br.get("lu") else "") + (f"、跌停 {br['ld']}" if br.get("ld") else "") + "</div>")

    def card(r, gid):
        col = ui.color_of(r["chg"])
        return (f"<div class='kq-panel' style='padding:8px 10px;margin:0'><div><a href='./chain?g={gid}' target='_self'><b>{ui.esc(r['name'])}</b></a> "
                f"<small class='kq-muted'>{ui.esc(layer_name.get(r['layer'], ''))}</small></div>"
                f"<div style='font-size:20px;font-weight:700;color:{col}'>{r['chg'] * 100:+.2f}%</div>"
                f"<div class='kq-muted' style='font-size:12px'>漲 {int(r['up'])}／跌 {int(r['down'])}"
                + (f"｜量比 {r['vr']:.1f}" if np.isfinite(r.get("vr", np.nan)) else "") + "</div>"
                f"<div style='font-size:12px'>{_who(r['lead'] if r['chg'] >= 0 else r['lag'])}</div></div>")
    grid = "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;margin:4px 0 10px'>{}</div>"
    h += "<div class='kq-muted'><b>最強</b></div>" + grid.format("".join(card(r, gid) for gid, r in g.head(top).iterrows()))
    h += "<div class='kq-muted'><b>最弱</b></div>" + grid.format("".join(card(r, gid) for gid, r in g.tail(top).iloc[::-1].iterrows()))
    lay = g.groupby("layer")["chg"].mean().reindex(list(layer_name)).dropna()
    if len(lay):
        h += ("<div class='kq-muted' style='margin:2px 0 4px'><b>上下游各層</b>（細族群平均）："
              + "　".join(f"{ui.esc(layer_name[k])} <b style='color:{ui.color_of(v)}'>{v * 100:+.2f}%</b>" for k, v in lay.items()) + "</div>")
    rows = [[f"<a href='./chain?g={gid}' target='_self'>{ui.esc(r['name'])}</a>", ui.esc(layer_name.get(r["layer"], "")),
             ui.pct(r["chg"], 2, color=True), ui.pct(rel(r["chg"]), 2, color=True) if idx_chg is not None else "–",
             f"{int(r['up'])}／{int(r['down'])}", f"{int(r['lu'])}／{int(r['ld'])}", f"{r['vr']:.1f}" if np.isfinite(r.get("vr", np.nan)) else "–",
             _who(r["lead"]), _who(r["lag"]), ui.pct(r.get("r20"), 1, color=True)] for gid, r in g.iterrows()]
    h += ui.table(["細族群", "層級", "今日（成交額加權）", "相對加權", "漲／跌", "漲停／跌停", "量比", "領漲", "領跌", "近 20 日"], rows, left=(0, 1, 7, 8))
    return h


def news_impact(groups: list, nvda: float | None, ixic: float | None, b: float | None, top: int = 8) -> list:
    """v3.21 族群輝達特有 β × 輝達「特有」漲跌（扣那指）→ 族群的預估反應（只當參考：β 是歷史平均，事件時常被放大）。"""
    if nvda is None or not np.isfinite(nvda) or not groups:
        return []
    e = nvda - (b if b is not None and np.isfinite(b) else 1.5) * (ixic if ixic is not None and np.isfinite(ixic) else 0.0)
    out = [{"group": g["group"], "beta_idio": g["beta_idio"], "impact": g["beta_idio"] * e} for g in groups if np.isfinite(g.get("beta_idio", np.nan))]
    return sorted(out, key=lambda r: -abs(r["impact"]))[:top]


def strip_html(idx: dict, otc: dict, br: dict, gl: pd.DataFrame | None, extra: list, label: str) -> str:
    """總覽最上方「現在盤勢」一列：加權、櫃買、前 300 大漲跌家數、最強／最弱細族群、美股期貨、重大消息提醒。"""
    items = []
    for nm, d in (("加權", idx), ("櫃買", otc)):
        if d and np.isfinite(d.get("chg", np.nan)):
            items.append(f"<span class='kq-item'>{nm}<b>{ui.num(d['px'], 2) if np.isfinite(d.get('px', np.nan)) else ''}</b> {ui.chg(d['chg'])}</span>")
    if br and br.get("n"):
        who = br.get("label") or f"{br['n']} 檔"
        items.append(f"<span class='kq-item'>{ui.esc(who)}<b style='color:#f0464b'>漲 {br['up']}</b>／<b style='color:#16a35f'>跌 {br['down']}</b>"
                     + (f"｜漲停 {br['lu']}" if br.get("lu") else "") + (f"｜跌停 {br['ld']}" if br.get("ld") else "") + "</span>")
    if gl is not None and len(gl) and "chg" in gl and gl["chg"].notna().any():
        g = gl[gl["chg"].notna()]
        items.append("<span class='kq-item'>最強 " + "、".join(f"<a href='./chain?g={gid}' target='_self'>{ui.esc(r['name'])}</a> {r['chg'] * 100:+.1f}%"
                                                         for gid, r in g.head(3).iterrows()) + "</span>")
        items.append("<span class='kq-item'>最弱 " + "、".join(f"<a href='./chain?g={gid}' target='_self'>{ui.esc(r['name'])}</a> {r['chg'] * 100:+.1f}%"
                                                         for gid, r in g.tail(3).iloc[::-1].iterrows()) + "</span>")
    items += extra
    return ("<div class='kq-panel' style='margin:4px 0 10px'><div class='kq-muted' style='margin-bottom:4px'>⚡ <b>現在盤勢</b>"
            f"｜{ui.esc(label)}</div><div style='display:flex;flex-wrap:wrap;gap:6px 16px'>{''.join(items)}</div></div>")
