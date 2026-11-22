"""v3.4／v3.5 美股頁面的 HTML／圖表組件（不依賴 Streamlit，方便測試）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ui
from . import model as M
from . import names as NM

H_LABEL = {"short": "短線（約 1 週）", "long": "中長線（1～3 個月）", "mid": "中線（約 1 個月）", "watch": "動能觀察名單"}
H_SHOW = ["short", "long"]                       # 中線與長線參考名單相同（SP200 × BAL_REV）→ 合併顯示為「中長線」
UNI_ZH = {100: "標普 500 前 100 大", 200: "標普 500 前 200 大", 300: "標普 500 前 300 大", 0: "標普 500"}
SET_ZH = {"sp500": "標普 500", "ndx": "那斯達克 100", "sox": "費城半導體", "cyber": "資安股", "power": "電力・核能", "all": "全部（含熱門股、ADR）"}
G1_DOWN, G1_UP = -0.04, 0.04


def money(x) -> str:
    if not ui.ok(x):
        return "–"
    x = float(x)
    return f"{x:,.2f}" if x < 1000 else f"{x:,.1f}"


def big_money(x) -> str:
    """美元金額 → 億／百萬美元。"""
    if not ui.ok(x):
        return "–"
    x = float(x)
    return f"{x / 1e8:,.1f} 億美元" if x >= 1e8 else f"{x / 1e6:,.0f} 百萬美元"


def label(B: dict, t: str) -> str:
    s = B["summary"]
    zh = s.at[t, "zh"] if t in s.index and isinstance(s.at[t, "zh"], str) else NM.zh_name(t)
    en = s.at[t, "name_en"] if t in s.index and isinstance(s.at[t, "name_en"], str) else ""
    return zh or en or t


def chip(B: dict, t: str, extra: str = "") -> str:
    return ui.schip(t, label(B, t), extra, href=f"./us?u={t}")


def reasons(B: dict, t: str, model: str, n: int = 2) -> str:
    """入選理由：該模型用到的因子中，這檔排名最前面的 n 個（母池百分位）。"""
    fr = B["factor_rank"]
    if t not in fr.index:
        return ""
    w = M.MODELS[model]
    xs = sorted(((fr.at[t, k], k) for k in w if k in fr.columns and ui.ok(fr.at[t, k])), reverse=True)[:n]
    return "、".join(f"{M.FACTOR_ZH[k]} 前 {max(1, round((1 - v) * 100))}%" for v, k in xs)


def g1_flag(h: str, ext_chg) -> str:
    """盤前跳空提示（docs/v3.5 C）：大型股很少跳空 4% 以上、歷史樣本太少 → 只做提醒，不給方向。"""
    if not ui.ok(ext_chg):
        return ""
    if ext_chg >= G1_UP:
        return "<span class='kq-tag gold'>盤前大漲：先看消息、別追價</span>"
    if ext_chg <= G1_DOWN:
        return "<span class='kq-tag gold'>盤前大跌：先看消息</span>"
    return ""


def list_title(B: dict, h: str) -> str:
    p = B["picks"][h]
    top = int(p.get("top", 0) or 0)
    uni = UNI_ZH.get(top, "") if "top" in p else "母池"
    return f"{H_LABEL.get(h, h)}｜{uni} × {M.MODEL_ZH.get(p['model'], p['model'])}"


def pick_table(B: dict, h: str, q: pd.DataFrame | None = None, live: dict | None = None, mode: str = "pre") -> str:
    """名單表（電腦＋手機兩種排版）。q＝即時摘要（index＝代號）；live＝engine.live_picks 的結果（即時名單）。"""
    p = B["picks"][h]
    model = p["model"]
    names = live[h]["names"] if live and h in live else p["names"]
    w = M.rank_weights(len(names))
    s = B["summary"]
    rows, cards = [], []
    for i, t in enumerate(names):
        r = s.loc[t] if t in s.index else pd.Series(dtype=float)
        qq = q.loc[t] if q is not None and len(q) and t in q.index else None
        px, chg, ext, ext_chg, ext_lab = r.get("close", np.nan), r.get("r1", np.nan), np.nan, np.nan, ""
        if qq is not None:
            px = qq.get("price", px)
            chg = qq.get("chg", chg)
            ext, ext_chg, ext_lab = qq.get("ext", np.nan), qq.get("ext_chg", np.nan), qq.get("ext_label", "")
        new = live is not None and h in live and t in live[h].get("new", [])
        tag = "<span class='kq-tag gold'>新進</span>" if new else ""
        rk = f"{i + 1}" + (f"<br><small class='kq-muted'>即時第 {live[h]['rank'].get(t, '–')} 名</small>" if live and h in live else "")
        ext_html = (f"{money(ext)} {ui.chg(ext_chg)}<br><small class='kq-muted'>{ext_lab}</small>" if ui.ok(ext) else "–")
        why = reasons(B, t, model)
        flag = g1_flag(h, ext_chg) if ext_lab == "盤前" else ""
        rows.append([rk, chip(B, t) + tag, ui.esc(r.get("sector", "") or "–"), money(px), ui.chg(chg), ext_html,
                     ui.pct(r.get("r20"), color=True), ui.pct(r.get("r120"), color=True), ui.pct(r.get("hi52", np.nan) - 1, color=True),
                     f"{w[i] * 100:.1f}%", f"<small>{ui.esc(why)}</small>{('<br>' + flag) if flag else ''}"])
        mid = (f"<small class='kq-muted'>{ui.esc(r.get('sector', '') or '')}｜半年 {ui.pct(r.get('r120'), color=True)}"
               f"｜距一年高 {ui.pct(r.get('hi52', np.nan) - 1, color=True)}｜資金 {w[i] * 100:.1f}%</small>"
               f"<br><small>{ui.esc(why)}</small>" + (f"<br>{flag}" if flag else ""))
        right = f"{money(px)} {ui.chg(chg)}" + (f"<br><small class='kq-muted'>{ext_lab} {money(ext)} {ui.chg(ext_chg)}</small>"
                                                 if ui.ok(ext) else "")
        cards.append(ui.mcard(f"<b>{i + 1}.</b> {chip(B, t)}{tag}", right, mid))
    head = ["名次", "股票", "產業", "價格", "今日", "盤前／盤後", "近 20 日", "近半年", "距一年高", "資金", "入選理由"]
    return ui.resp(ui.table(head, rows, left=(1, 2, 10)), "".join(cards))


def movers(B: dict, q: pd.DataFrame, n: int = 8, thr: float = 0.02) -> str:
    """盤前／盤後異動：候選股中延長交易時段漲跌最大的幾檔。"""
    if q is None or not len(q) or "ext_chg" not in q:
        return "<div class='kq-muted'>目前沒有延長時段報價（盤前台北 17:00 起、夏令 16:00 起）。</div>"
    x = q[q["ext_chg"].abs() >= thr].copy()
    if x.empty:
        return f"<div class='kq-muted'>候選股中沒有漲跌超過 {thr:.0%} 的股票。</div>"
    x = x.reindex(x["ext_chg"].abs().sort_values(ascending=False).index).head(n)
    chips = [chip(B, t, f"{r['ext_label']} {r['ext_chg'] * 100:+.1f}%") for t, r in x.iterrows()]
    return f"<div class='kq-chips'>{''.join(chips)}</div>"


def validation_html(R: dict, h: str = "long") -> str:
    """第二輪研究（docs/v3.5_protocol.md）：某個期間的 9 模型 × 4 母池，對 SPY 年化超額（開發期／驗證期／2025 後）。"""
    if not R or "grid" not in R:
        return "<div class='kq-muted'>沒有研究摘要檔。</div>"
    g = pd.DataFrame(R["grid"])
    g = g[g.h == h]
    unis = ["SP100", "SP200", "SP300", "SP500"]
    ref = R["reference"].get(h) or ["", 0]
    ref_u = f"SP{ref[1] or 500}"
    rows = []
    for mod in M.MODELS:
        cells = []
        for u in unis:
            x = g[(g.model == mod) & (g.uni == u)].set_index("period")
            if x.empty:
                cells.append("–")
                continue
            star = "⭐ " if (mod == ref[0] and u == ref_u) else ""
            cells.append(f"{star}{ui.pct(x.at['dev', 'excess'], color=True)}"
                         f"<br><small class='kq-muted'>驗 {ui.pct(x.at['val', 'excess'])}｜25後 {ui.pct(x.at['blind', 'excess'])}</small>")
        rows.append([f"{ui.esc(M.MODEL_ZH[mod])}<br><small class='kq-muted'>{mod}</small>", *cells])
    return ui.table(["模型（開發期 2004–21 對 SPY 超額；小字＝驗證期、2025 後）", "前 100 大", "前 200 大", "前 300 大", "標普 500"],
                    rows, left=(0,))


def etf_html(R: dict) -> str:
    e = pd.DataFrame(R.get("etf", []))
    if e.empty:
        return ""
    lab = {"R1": "類股輪動（6 個月最強 3 類）", "R2": "雙動能（SPY／中期美債）", "R3": "200 日線趨勢（SPY／中期美債）",
           "R4": "科技／大盤（QQQ／SPY）", "R5": "半導體／大盤（SOXX／SPY）", "QQQ": "一直抱 QQQ（對照）",
           "SOXX": "一直抱 SOXX（對照）", "RSP": "一直抱等權 RSP（對照）"}
    rows = []
    for k, nm in lab.items():
        x = e[e.strat == k].set_index("period")
        if x.empty:
            continue
        rows.append([ui.esc(nm), f"{ui.pct(x.at['dev', 'excess'], color=True)}<br><small class='kq-muted'>IR {x.at['dev', 'ir']:.2f}</small>",
                     f"{ui.pct(x.at['dev', 'mdd'])}<br><small class='kq-muted'>SPY {ui.pct(x.at['dev', 'bench_mdd'])}</small>",
                     ui.pct(x.at["val", "excess"], color=True), ui.pct(x.at["blind", "excess"], color=True)])
    return ui.table(["ETF 策略（每 20 個交易日檢查）", "開發期超額", "開發期最大回撤", "驗證期超額", "2025 後超額"], rows, left=(0,))


def set_table(B: dict, key: str, model: str, q: pd.DataFrame | None = None, n: int = 15) -> str:
    """瀏覽：某個成分股集合內，用某個模型排名的前 n 名（附即時漲跌）。"""
    from . import engine as E
    rk = E.set_rank(B, key, model, n)
    if rk.empty:
        return "<div class='kq-muted'>這個範圍目前沒有股票（資料包可能是舊版）。</div>"
    s = B["summary"]
    rows, cards = [], []
    for t, r in rk.iterrows():
        x = s.loc[t] if t in s.index else pd.Series(dtype=float)
        qq = q.loc[t] if q is not None and len(q) and t in q.index else None
        px = qq.get("price") if qq is not None else x.get("close", np.nan)
        chg = qq.get("chg") if qq is not None else x.get("r1", np.nan)
        why = reasons(B, t, model)
        rows.append([f"{int(r['rank'])}", chip(B, t), ui.esc(x.get("sector", "") or "–"), money(px), ui.chg(chg),
                     ui.pct(x.get("r20"), color=True), ui.pct(x.get("r120"), color=True), f"<small>{ui.esc(why)}</small>"])
        cards.append(ui.mcard(f"<b>{int(r['rank'])}.</b> {chip(B, t)}", f"{money(px)} {ui.chg(chg)}",
                              f"<small class='kq-muted'>{ui.esc(x.get('sector', '') or '')}｜半年 {ui.pct(x.get('r120'), color=True)}</small>"
                              f"<br><small>{ui.esc(why)}</small>"))
    return ui.resp(ui.table(["名次", "股票", "產業", "價格", "今日", "近 20 日", "近半年", "排名理由"], rows, left=(1, 2, 7)), "".join(cards))


def round1_html(R: dict) -> str:
    """第一輪（v3.4，倖存者偏差資料、對等權母池）摘要。"""
    r1 = R.get("round1") or {}
    m = pd.DataFrame(r1.get("models", []))
    if m.empty:
        return ""
    rows = []
    for h, hz in (("short", "短線"), ("mid", "中線"), ("long", "長線")):
        x = m[(m.h == h) & (m.period == "dev")].sort_values("excess", ascending=False).head(2)
        for _, r in x.iterrows():
            v = m[(m.h == h) & (m.model == r.model) & (m.period == "val")]
            rows.append([hz, ui.esc(M.MODEL_ZH.get(r.model, r.model)), ui.pct(r.excess, color=True),
                         ui.pct(v.iloc[0].excess, color=True) if len(v) else "–"])
    return ui.table(["期間", "第一輪開發期前 2 名", "開發期 2014–21（對等權）", "驗證期"], rows, left=(0, 1))


def ic_html(R: dict) -> str:
    ic = pd.DataFrame(R.get("ic", []))
    if ic.empty:
        return ""
    rows = []
    order = ic[(ic.H == 60) & (ic.period == "dev")].sort_values("ic", ascending=False)["factor"].tolist()
    for f in order:
        cells = []
        for H in (5, 20, 60):
            x = ic[(ic.factor == f) & (ic.H == H)].set_index("period")
            cells.append(" ／ ".join(f"{x.at[p, 'ic']:+.3f}" if p in x.index else "–" for p in ("dev", "val", "blind")))
        rows.append([ui.esc(M.FACTOR_ZH.get(f, f)) + f"<br><small class='kq-muted'>{f}</small>", *cells])
    return ui.table(["因子", "5 日（開發／驗證／盲測）", "20 日", "60 日"], rows, left=(0,))


def gap_html(R: dict) -> str:
    g = pd.DataFrame(R.get("gap", []))
    if g.empty or "list" not in g:
        return ""
    lab = {"short_ref": "短線參考名單", "mid_ref": "中長線參考名單", "watch": "動能觀察名單"}
    glab = {"up": "跳空 ≥ +4%", "down": "跳空 ≤ −4%"}
    rows = []
    for li, ln in lab.items():
        for grp, gl in glab.items():
            x = g[(g.list == li) & (g.grp == grp)].set_index("period")
            rows.append([f"{ln}｜{gl}", *[f"{ui.pct(x.at[p, 'mean'], color=True)}<br><small class='kq-muted'>{int(x.at[p, 'n'])} 次</small>"
                                          if p in x.index else "–" for p in ("dev", "val", "blind")]])
    return ui.table(["入選股盤前跳空之後 5 日（對母池等權）", "開發期", "驗證期", "2025 後"], rows, left=(0,))


def basics_html(clk: dict) -> str:
    season = "夏令時間" if clk["dst"] else "冬令時間"
    return ui.bullets([
        f"<b>交易時間（台北時間，目前{season}）</b>：盤前 {clk['pre_tpe']}～{clk['open_tpe']}、正常交易 {clk['open_tpe']}～隔天 {clk['close_tpe']}、"
        f"盤後 {clk['close_tpe']}～{clk['post_end_tpe']}。每年 3 月第二個週日～11 月第一個週日為夏令，全部提早 1 小時。",
        "<b>沒有漲跌停</b>：個股一天可以漲跌 20%、50% 以上（財報、藥證結果）。只有「個股暫停交易（LULD，5 分鐘內波動過大）」與"
        "「大盤熔斷（標普 500 跌 7%／13%／20%）」。",
        "<b>交割</b>：T+1（成交隔一個營業日交割）。<b>最小單位</b>：1 股（部分券商可買零股到小數股）。",
        "<b>盤前、盤後</b>：成交量小、買賣價差大，價格容易失真；複委託與海外券商是否開放、可下哪些委託，請以券商規定為準。",
        "<b>稅</b>：台灣投資人買美股，美國不課資本利得稅；股息預扣 30%（台美租稅協定／雙重課稅減免法案截至本版查到的資料仍未生效，以券商公告為準）。"
        "海外所得在台灣適用最低稅負制（一年海外所得超過 100 萬元才需申報，詳見國稅局）。",
        "<b>財報季</b>（1、4、7、10 月中下旬起約一個月）：個股常在盤後公布財報，隔天開盤大跳空；名單內股票遇到財報週，風險比平常大。",
        "<b>ADR</b>：外國公司在美國掛牌的存託憑證（例如台積電 TSM、日月光 ASX、聯電 UMC），價格與台股本尊連動但有溢價／折價。",
        "<b>配色</b>：本頁沿用台股習慣──<span style='color:#f0464b'>紅漲</span>、<span style='color:#16a35f'>綠跌</span>（美國網站多半相反）。",
    ])


def sector_table(B: dict) -> str:
    """類股 ETF 表現（資料包收盤），依 20 日漲跌排序。"""
    C = B["C"]
    items = []
    for t, zh in NM.SECTOR_ETF.items():
        if t not in C.columns or C[t].notna().sum() < 61:
            continue
        c = C[t].dropna().astype(float)
        items.append((c.iloc[-1] / c.iloc[-21] - 1, t, zh, c))
    if not items:
        return "<div class='kq-muted'>資料包沒有類股 ETF 資料。</div>"
    rows = [[f"<b>{t}</b> {ui.esc(zh)}", ui.chg(c.iloc[-1] / c.iloc[-2] - 1), ui.pct(c.iloc[-1] / c.iloc[-6] - 1, color=True),
             ui.pct(r20, color=True), ui.pct(c.iloc[-1] / c.iloc[-61] - 1, color=True)]
            for r20, t, zh, c in sorted(items, key=lambda x: -x[0])]
    return ui.table(["類股", "最近一日", "5 日", "20 日", "60 日"], rows, left=(0,))


def breadth(B: dict) -> dict:
    s = B["summary"]
    u = s[s["in_u"].fillna(False).astype(bool)]
    if u.empty:
        return {}
    return {"n": len(u), "above50": float((u["ma50"] > 0).mean()), "above200": float((u["ma200"] > 0).mean()),
            "near_high": float((u["hi52"] >= 0.95).mean()), "up1": float((u["r1"] > 0).mean())}


def treemap(B: dict, period: str = "r1", dark: bool = True, height: int = 560, live: pd.Series | None = None):
    """母池熱力圖：類股 → 個股，面積＝60 日平均成交金額，顏色＝漲跌。"""
    import plotly.graph_objects as go
    s = B["summary"]
    u = s[s["in_u"].fillna(False).astype(bool)].copy()
    if u.empty:
        return go.Figure()
    v = u[period].astype(float)
    if live is not None and period == "r1":
        v = live.reindex(u.index).fillna(v)
    u["v"] = v
    u["lab"] = [f"{t}<br>{(label(B, t))[:6]}" for t in u.index]
    sec = u.groupby("sector")
    ids, labels, parents, values, colors, text = [], [], [], [], [], []
    for sname, g in sec:
        ids.append("S:" + sname)
        labels.append(sname)
        parents.append("")
        values.append(float(g["adv60"].sum()))
        wavg = float(np.average(g["v"].fillna(0), weights=g["adv60"].fillna(0) + 1))
        colors.append(wavg)
        text.append(f"{wavg * 100:+.2f}%")
    for t, r in u.iterrows():
        ids.append(t)
        labels.append(r["lab"])
        parents.append("S:" + r["sector"])
        values.append(float(r["adv60"]) if ui.ok(r["adv60"]) else 0.0)
        colors.append(float(r["v"]) if ui.ok(r["v"]) else 0.0)
        text.append(f"{r['v'] * 100:+.2f}%" if ui.ok(r["v"]) else "–")
    lim = {"r1": 0.03, "r5": 0.06, "r20": 0.12, "r60": 0.2}.get(period, 0.05)
    bg = "#0f1726" if dark else "#ffffff"
    fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", text=text,
                               texttemplate="%{label}<br>%{text}", hovertemplate="%{label}<br>%{text}<extra></extra>",
                               marker=dict(colors=colors, cmin=-lim, cmid=0, cmax=lim,
                                           colorscale=[[0, "#16a35f"], [0.5, "#334155" if dark else "#e2e8f0"], [1, "#f0464b"]],
                                           line=dict(width=1, color=bg)), maxdepth=2, tiling=dict(pad=2)))
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor=bg,
                      font=dict(color="#d6deeb" if dark else "#1e293b", size=12))
    return fig


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """日線 → K 線指標（沿用台股 ta.indicators，另加 50／200 日線）。"""
    from .. import ta
    x = df.rename(columns={"adj": "adj_close"}).copy()
    out = ta.indicators(x[["open", "high", "low", "close", "volume"]])
    for n in (50, 200):
        out[f"ma{n}"] = out["close"].rolling(n, min_periods=n).mean()
    return out


def analysis_rows(ind: pd.DataFrame, spy: pd.Series | None = None) -> list[tuple[str, str]]:
    """技術面摘要（給不熟美股的人看的白話）。"""
    if ind is None or len(ind) < 30:
        return []
    c = ind["close"]
    last = float(c.iloc[-1])
    out = []

    def ret(n):
        return c.iloc[-1] / c.iloc[-1 - n] - 1 if len(c) > n else np.nan
    rel = ""
    if spy is not None and len(spy) > 130:
        rs = ret(126) - (spy.iloc[-1] / spy.iloc[-127] - 1)
        rel = f"（比標普 500 {'多' if rs >= 0 else '少'} {abs(rs) * 100:.0f} 個百分點）"
    out.append(("近 1／3／6／12 個月", " ／ ".join(ui.pct(ret(n), color=True) for n in (21, 63, 126, 252)) + rel))
    m50, m200 = ind["ma50"].iloc[-1], ind["ma200"].iloc[-1]
    if ui.ok(m200):
        st = ("多頭（股價 > 50 日線 > 200 日線）" if last > m50 > m200 else "空頭（股價 < 50 日線 < 200 日線）" if last < m50 < m200
              else "整理（均線交錯）")
        out.append(("趨勢", f"{st}｜50 日線 {money(m50)}、200 日線 {money(m200)}"))
    hi, lo = c.iloc[-252:].max(), c.iloc[-252:].min()
    out.append(("一年區間", f"{money(lo)} ～ {money(hi)}｜距高點 {ui.pct(last / hi - 1, color=True)}"))
    if "rsi" in ind and ui.ok(ind["rsi"].iloc[-1]):
        r = float(ind["rsi"].iloc[-1])
        out.append(("RSI（14）", f"{r:.0f}　{'過熱（>70）' if r > 70 else '超賣（<30）' if r < 30 else '中性'}"))
    vol = (c / c.shift(1) - 1).iloc[-60:].std() * np.sqrt(252)
    atr = ind["atr"].iloc[-1] / last if "atr" in ind and ui.ok(ind["atr"].iloc[-1]) else np.nan
    out.append(("波動", f"年化 {vol * 100:.0f}%｜平均一天振幅約 {atr * 100:.1f}%" if ui.ok(atr) else f"年化 {vol * 100:.0f}%"))
    s20, r20 = ind["low"].iloc[-20:].min(), ind["high"].iloc[-20:].max()
    out.append(("近 20 日支撐／壓力", f"{money(s20)} ／ {money(r20)}"))
    return out


def model_rows(B: dict, t: str) -> str:
    """個股在各候選模型的母池排名＋各因子百分位。"""
    sc = B["scores"]
    u = B["universe"]
    if t not in sc.index or not bool(u.get(t, False)):
        return "<div class='kq-muted'>不在今日母池（股價 < 5 美元、上市未滿一年或成交金額不夠大），模型不評分。</div>"
    rows = []
    n = int(u.sum())
    for m in M.MODELS:
        rk = sc[m].where(u.reindex(sc.index).fillna(False).astype(bool)).rank(ascending=False)
        r = rk.get(t, np.nan)
        from . import engine as E
        ref = [h for h in B["reference"] if E.ref(B, h)[0] == m]
        rows.append([f"{'⭐ ' if ref else ''}{ui.esc(M.MODEL_ZH[m])}", f"{int(r)} ／ {n}" if ui.ok(r) else "–",
                     ui.esc("、".join({'short': '短線', 'mid': '中線', 'long': '長線'}[h] for h in ref) or "—")])
    fr = B["factor_rank"].loc[t] if t in B["factor_rank"].index else pd.Series(dtype=float)
    frows = [[ui.esc(M.FACTOR_ZH[k]), f"前 {max(1, round((1 - fr[k]) * 100))}%" if ui.ok(fr.get(k)) else "–"] for k in M.FACTOR_ZH]
    return (ui.table(["模型", "今日母池排名", "參考名單"], rows, left=(0, 2))
            + "<div style='height:8px'></div>" + ui.table(["因子", "母池位置"], frows, left=(0,)))


def tw_link_html(t: str, tw_names: dict | None = None) -> str:
    lines = NM.tw_links(t)
    if not lines:
        return ""
    if tw_names:
        import re
        lines = [re.sub(r"\b(\d{4})\b", lambda m: f"{m.group(1)}{tw_names.get(m.group(1), '')}", x) for x in lines]
    return ui.bullets([ui.esc(x) for x in lines])


# ------------------------------------------------------------------ v3.6 美股 ETF 推薦（docs/v3.6_protocol.md 第二節）
def etf_rank(R: dict) -> dict[str, list[str]]:
    """角色內依長期 Sharpe 排（同分看費用率）；角色順序固定：核心 → 成長 → 避險 → 衛星。"""
    st = R.get("etf_stats") or {}
    out = {}
    for role in NM.ROLE_ORDER:
        xs = [t for t, i in NM.ETF_INFO.items() if i["role"] == role and t in st]
        xs.sort(key=lambda t: (-(st[t].get("sharpe") or 0), NM.ETF_INFO[t]["er"]))
        out[role] = xs
    return out


def etf_label(t: str) -> str:
    return "IVV／VOO／SPY" if t == "IVV" else ("QQQ／QQQM" if t == "QQQ" else ("SOXX／SMH" if t == "SOXX" else t))


def etf_trend(B: dict | None, t: str) -> dict:
    """資料包內最近的走勢（Yahoo 日線）：1 個月、6 個月、1 年、今年以來、是否站上 200 日線。IVV 沒資料時用 VOO／SPY。"""
    if B is None:
        return {}
    C = B.get("C")
    for x in ([t, "VOO", "SPY"] if t == "IVV" else [t]):
        if C is not None and x in C.columns and C[x].notna().sum() > 210:
            c = C[x].dropna().astype(float)
            ytd = c[c.index >= f"{c.index[-1].year}-01-01"]
            return {"r1m": c.iloc[-1] / c.iloc[-22] - 1, "r6m": c.iloc[-1] / c.iloc[-127] - 1,
                    "r1y": c.iloc[-1] / c.iloc[-253] - 1 if len(c) > 253 else np.nan,
                    "ytd": c.iloc[-1] / c[c.index < ytd.index[0]].iloc[-1] - 1 if len(ytd) and (c.index < ytd.index[0]).any() else np.nan,
                    "above200": bool(c.iloc[-1] > c.iloc[-200:].mean()), "src": x}
    return {}


def etf_reason(t: str, s: dict, tr: dict) -> str:
    """一句話理由（全部來自統計數字）。"""
    i = NM.ETF_INFO[t]
    y0 = int(s["start"][:4]) + (1 if s["start"][5:7] == "12" else 0)
    parts = [f"{y0} 年起年化 {s['cagr'] * 100:.1f}%" + (f"、近 10 年 {s['cagr10'] * 100:.1f}%" if s.get("cagr10") is not None else "")]
    if s.get("y2008") is not None:
        parts.append(f"2008 年 {s['y2008'] * 100:+.0f}%")
    if s.get("y2022") is not None:
        parts.append(f"2022 年 {s['y2022'] * 100:+.0f}%")
    parts.append(f"最大回撤 {s['mdd'] * 100:.0f}%")
    if s.get("corr0050") is not None:
        c = s["corr0050"]
        parts.append(f"與 0050 相關 {c:.2f}（{'分散效果好' if c < 0.3 else '有分散效果' if c < 0.55 else '和台股很像'}）")
    parts.append(f"費用 {i['er']:.2f}%")
    if tr:
        parts.append(f"目前{'站上' if tr['above200'] else '跌破'} 200 日線、近 6 個月 {tr['r6m'] * 100:+.0f}%")
    return "，".join(parts) + "。"


def etf_starter_html(R: dict, B: dict | None) -> str:
    """第一次買美股的順序：核心第一名 → 成長第一名 → 避險前兩名 → 衛星第一名。"""
    rk, st = etf_rank(R), R.get("etf_stats") or {}
    steps = [("①", "core", 0, "先買核心，佔美股資金的大部分"), ("②", "growth", 0, "想多一點報酬，再加成長型"),
             ("③", "hedge", 0, "怕大跌，加一點避險"), ("③", "hedge", 1, ""), ("④", "sat", 0, "看好特定產業，小量當衛星")]
    out = ["<div class='kq-steps'>"]
    for no, role, k, head in steps:
        if len(rk.get(role, [])) <= k:
            continue
        t = rk[role][k]
        i, s = NM.ETF_INFO[t], st[t]
        tr = etf_trend(B, t)
        trend = (f"<span class='kq-tag {'g' if tr['above200'] else 'gold'}'>{'多頭' if tr['above200'] else '跌破 200 日線'}</span>"
                 if tr else "")
        tw = f"<div class='kq-muted'>台股同類（台幣買）：{ui.esc(i['tw'])}</div>" if i["tw"] else ""
        alt = f"<div class='kq-muted'>同類替代：{ui.esc(i['alt'])}</div>" if i["alt"] else ""
        out.append(f"<div class='kq-step'><div class='no'>{no}</div><div class='bd'>"
                   f"<div class='hd'><a href='./us?u={t}' target='_self'><b>{ui.esc(etf_label(t))}</b></a> {ui.esc(i['zh'])}"
                   f"<span class='kq-tag'>{ui.esc(NM.ROLE_ZH[role].split('（')[0])}</span>{trend}</div>"
                   + (f"<div class='why'><b>{ui.esc(head)}</b></div>" if head else "")
                   + f"<div class='why'>{ui.esc(etf_reason(t, s, tr))}</div>{alt}{tw}</div></div>")
    out.append("</div>")
    return "".join(out)


def etf_role_table(R: dict, B: dict | None, role: str) -> str:
    rk, st = etf_rank(R), R.get("etf_stats") or {}
    rows, cards = [], []
    for n, t in enumerate(rk.get(role, []), 1):
        i, s = NM.ETF_INFO[t], st[t]
        tr = etf_trend(B, t)
        trend = (f"{'▲ 站上' if tr['above200'] else '▼ 跌破'} 200 日線<br><small class='kq-muted'>6 個月 {ui.pct(tr['r6m'], 0)}｜1 年 {ui.pct(tr['r1y'], 0)}</small>"
                 if tr else "–")
        sub = " ／ ".join(x for x in (i["alt"], ("台股：" + i["tw"]) if i["tw"] else "") if x)
        rows.append([f"{n}", f"<a href='./us?u={t}' target='_self'><b>{ui.esc(etf_label(t))}</b></a><br><small class='kq-muted'>{ui.esc(i['zh'])}</small>",
                     f"<small>{ui.esc(i['idx'])}</small>",
                     f"{ui.pct(s['cagr'], 1, color=True)}<br><small class='kq-muted'>{int(s['start'][:4]) + (1 if s['start'][5:7] == '12' else 0)} 起</small>",
                     ui.pct(s.get("cagr10"), 1, color=True), ui.pct(s["mdd"], 0), f"{ui.pct(s['worst_year'], 0)}<br><small class='kq-muted'>{s['worst_year_y']}</small>",
                     f"{s['sharpe']:.2f}", f"{i['er']:.2f}%", f"{s['corr0050']:.2f}" if s.get("corr0050") is not None else "–", trend,
                     f"<small>{ui.esc(sub)}</small>"])
        cards.append(ui.mcard(f"<b>{n}.</b> <a href='./us?u={t}' target='_self'><b>{ui.esc(etf_label(t))}</b></a> {ui.esc(i['zh'])}",
                              f"年化 {ui.pct(s['cagr'], 1, color=True)}",
                              f"<small class='kq-muted'>回撤 {ui.pct(s['mdd'], 0)}｜最差一年 {ui.pct(s['worst_year'], 0)}｜Sharpe {s['sharpe']:.2f}｜費用 {i['er']:.2f}%</small>"
                              + (f"<br><small>{'站上' if tr['above200'] else '跌破'} 200 日線｜6 個月 {ui.pct(tr['r6m'], 0)}</small>" if tr else ""),
                              f"<small class='kq-muted'>{ui.esc(sub)}</small>" if sub else ""))
    return ui.resp(ui.table(["#", "ETF", "追蹤", "長期年化", "近 10 年", "最大回撤", "最差一年", "Sharpe", "費用率", "與 0050 相關", "目前趨勢",
                             "同類／台股替代"], rows, left=(1, 2, 11)), "".join(cards))


def mix_html(R: dict, prof: str) -> str:
    mx = (R.get("mix") or {}).get(prof)
    if not mx:
        return ""
    w = mx["w"]
    chips = "".join(f"<span class='kq-tag'>{ui.esc(etf_label(t))} {x:.0%}</span>" for t, x in w.items())
    return ui.cards([("配置（每年調回一次）", f"<span style='font-size:14px'>{chips}</span>", "2004–2026 回測，只是示意"),
                     ("年化報酬", ui.pct(mx["cagr"], 1, color=True), f"100% SPY {ui.pct(R['mix']['100% SPY']['cagr'], 1)}"),
                     ("最大回撤", ui.pct(mx["mdd"], 0), f"100% SPY {ui.pct(R['mix']['100% SPY']['mdd'], 0)}"),
                     ("最差一年", ui.pct(mx["worst_year"], 0), f"{mx['worst_year_y']} 年｜2022 年 {ui.pct(mx['y2022'], 0)}")], raw=True)


def mix_table(R: dict) -> str:
    rows = []
    for nm, mx in (R.get("mix") or {}).items():
        rows.append([f"<b>{ui.esc(nm)}</b><br><small class='kq-muted'>{'、'.join(f'{etf_label(t)} {x:.0%}' for t, x in mx['w'].items())}</small>",
                     ui.pct(mx["cagr"], 1, color=True), ui.pct(mx["mdd"], 0), ui.pct(mx["worst_year"], 0), ui.pct(mx["y2008"], 0),
                     ui.pct(mx["y2022"], 0), f"{mx['sharpe']:.2f}"])
    return ui.table(["組合（2004–2026、每年再平衡）", "年化", "最大回撤", "最差一年", "2008", "2022", "Sharpe"], rows, left=(0,))


def round3_html(R: dict) -> str:
    g = pd.DataFrame(R.get("round3", []))
    if g.empty:
        return ""
    zh = R.get("round3_zh", {})
    rows = []
    for c in ["V1", "V2", "V3", "V4"]:
        cells = []
        for h in ("short", "mid", "long"):
            x = g[(g.cand == c) & (g.h == h)].set_index("period")
            if x.empty:
                cells.append("–")
                continue
            cells.append(f"{ui.pct(x.at['dev', 'excess'], color=True)}<br><small class='kq-muted'>驗 {ui.pct(x.at['val', 'excess'])}｜25後 {ui.pct(x.at['blind', 'excess'])}</small>")
        rows.append([f"{ui.esc(zh.get(c, c))}<br><small class='kq-muted'>{c}</small>", *cells])
    return ui.table(["第三輪候選（開發期對 SPY 超額；小字＝驗證期、2025 後）", "短線", "中線", "長線"], rows, left=(0,))



def round4_html(R: dict) -> str:
    """v3.19 美股第四輪：每個候選取開發期較好的母池；小字＝驗證期、2025 後、對 QQQ（開發期）。"""
    g = pd.DataFrame(R.get("round4", []))
    if g.empty:
        return ""
    zh = R.get("round4_zh", {})
    rows = []
    for c in ["U1", "U2", "U3", "U4", "U5"]:
        cells = []
        for h in ("short", "mid", "long"):
            x = g[(g.cand == c) & (g.h == h)]
            if x.empty:
                cells.append("–")
                continue
            dev = x[x.period == "dev"].sort_values("excess", ascending=False).iloc[0]
            y = x[x.pool == dev["pool"]].set_index("period")
            cells.append(f"{ui.pct(y.at['dev', 'excess'], color=True)}<small class='kq-muted'>（IR {y.at['dev', 'ir']:.2f}，{dev['pool']}）</small><br>"
                         f"<small class='kq-muted'>驗 {ui.pct(y.at['val', 'excess'])}｜25後 {ui.pct(y.at['blind', 'excess'])}｜對 QQQ {ui.pct(y.at['dev', 'vs_qqq'])}</small>")
        rows.append([f"{ui.esc(zh.get(c, c))}<br><small class='kq-muted'>{c}</small>", *cells])
    q = R.get("qqq_vs_spy") or {}
    rows.append(["<b>參考：一直抱 QQQ</b>", *([f"{ui.pct(q.get('dev'), color=True)}<br><small class='kq-muted'>驗 {ui.pct(q.get('val'))}｜25後 {ui.pct(q.get('blind'))}</small>"] * 3)])
    return ui.table(["第四輪候選（開發期對 SPY 超額）", "短線", "中線", "長線"], rows, left=(0,))
