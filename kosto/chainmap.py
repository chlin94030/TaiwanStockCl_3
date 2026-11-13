"""v3.0「產業鏈地圖」頁的資料與 HTML（全景圖、細族群拆解、AI 伺服器拆解圖、題材地圖、消息衝擊評估）。

只產生 HTML 字串與表格資料；Streamlit 版面在 app.py。顏色依台股習慣：紅＝漲（相對大盤強）、綠＝跌（相對弱）。
"""
from __future__ import annotations

import html as _h

import numpy as np
import pandas as pd

from kosto import impact, supply
from kosto import ui

AI_TAG = {2: ("AI 核心", "gold"), 1: ("部分 AI", ""), 0: ("", "")}
STAGE_KIND = {"成熟": "g", "成長": "", "題材": "gold"}


def esc(x) -> str:
    return _h.escape(str(x))


# ------------------------------------------------------------------ 數據
def metrics(B) -> pd.DataFrame:
    """每檔：今日（最後一日）、5／20／60 日漲跌，近 20 日均成交金額（億）。資料＝看盤資料包 Q_C、Q_RF、Q_V。"""
    if "Q_C" not in B.a:
        return pd.DataFrame(index=list(B.tickers))
    C = pd.DataFrame(B.a["Q_C"].astype(float), columns=list(B.tickers)).ffill()
    RF = pd.DataFrame(B.a["Q_RF"].astype(float), columns=list(B.tickers)) if "Q_RF" in B.a else C.shift(1)
    V = pd.DataFrame(B.a["Q_V"].astype(float), columns=list(B.tickers)) if "Q_V" in B.a else C * np.nan
    last = C.iloc[-1]
    out = pd.DataFrame({"r1": last / RF.iloc[-1].where(RF.iloc[-1] > 0) - 1})
    for n in (5, 20, 60):
        out[f"r{n}"] = last / C.iloc[-1 - n] - 1 if len(C) > n else np.nan
    out["amt20"] = (V * C).tail(20).mean() * 1000 / 1e8          # 張 × 元 → 億元
    out["name"] = list(B.meta["names"])
    return out


def market_ref(met: pd.DataFrame, top: int = 800) -> dict:
    m = met[met["amt20"].rank(ascending=False) <= top] if "amt20" in met and met["amt20"].notna().any() else met
    return {k: float(m[k].mean()) for k in ("r1", "r5", "r20", "r60") if k in m}


def group_stats(met: pd.DataFrame, universe=None) -> pd.DataFrame:
    mk = market_ref(met)
    rows = []
    for g, d in supply.SUBGROUPS.items():
        ts = [t for t, _ in d["members"] if t in met.index and (universe is None or t in universe)]
        x = met.loc[ts] if ts else met.iloc[:0]
        rows.append({"gid": g, "layer": d["layer"], "name": d["name"], "ai": d["ai"], "n": len(ts),
                     **{k: float(x[k].mean()) if len(x) else np.nan for k in ("r1", "r5", "r20", "r60")},
                     **{f"x{k[1:]}": float(x[k].mean()) - mk.get(k, 0) if len(x) else np.nan for k in ("r1", "r5", "r20", "r60")}})
    return pd.DataFrame(rows).set_index("gid")


def heat(x: float, scale: float = 0.10) -> str:
    """相對大盤的超額報酬 → 背景色（紅強、綠弱）。"""
    if x is None or not np.isfinite(x):
        return "var(--panel)"
    a = min(abs(x) / scale, 1.0) * 0.55 + 0.06
    return f"rgba(240,70,75,{a:.2f})" if x > 0 else f"rgba(22,163,95,{a:.2f})"


# ------------------------------------------------------------------ 全景圖
CSS = """<style>
.cm-layer{display:flex;gap:12px;align-items:stretch;margin:0}
.cm-lab{flex:0 0 190px;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:8px 10px}
.cm-lab .n{font-weight:700;color:var(--strong);font-size:14px}.cm-lab .d{font-size:12px;color:var(--muted);margin-top:2px;line-height:1.35}
.cm-chips{flex:1;display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start}
.cm-chip{display:block;min-width:150px;flex:0 1 auto;border:1px solid var(--line);border-radius:12px;padding:7px 10px;text-decoration:none!important;
 color:var(--text)!important;box-shadow:var(--shadow)}
.cm-chip:hover{border-color:var(--accent)}.cm-chip.sel{outline:2px solid var(--accent)}
.cm-chip .n{font-weight:600;color:var(--strong);font-size:13.5px}.cm-chip .s{font-size:11.5px;color:var(--muted);margin-top:2px}
.cm-arrow{text-align:left;padding-left:84px;color:var(--faint);font-size:18px;line-height:20px;min-height:20px}
.cm-arrow small{font-size:12.5px;margin-left:6px}
.cm-cust{display:flex;flex-wrap:wrap;gap:8px}.cm-cust span{border:1px dashed var(--line);border-radius:12px;padding:6px 10px;font-size:12.5px}
.cm-leg{font-size:12px;color:var(--muted);margin:2px 0 8px}
.cm-step{display:flex;gap:10px;align-items:flex-start;margin:4px 0}.cm-step .k{flex:0 0 26px;height:26px;border-radius:50%;background:var(--accent);
 color:#111;font-weight:700;text-align:center;line-height:26px;font-size:13px}.cm-step .t{flex:1;font-size:13.5px;line-height:1.5}
.cm-bar{display:inline-block;height:10px;border-radius:3px;vertical-align:middle}
.cm-svg text{font-family:'Noto Sans TC',sans-serif}
@media (max-width:760px){.cm-layer{flex-direction:column;gap:6px}.cm-lab{flex:none}.cm-arrow{padding-left:12px}
 .cm-chip{flex:1 1 calc(50% - 8px);min-width:0;padding:6px 8px}.cm-chip .n{font-size:12.5px}.cm-chip .s{font-size:10.5px}
 .cm-chip .kq-tag{font-size:10px;padding:0 5px}}
</style>"""


def overview_html(gs: pd.DataFrame, sel: str | None = None, base_url: str = "./chain") -> str:
    out = [CSS, "<div class='cm-leg'>方塊顏色＝細族群代表股近 20 日相對大盤（紅＝強、綠＝弱，越深差越多）；點方塊看拆解。"
                "<b>AI 核心</b>＝營收大多來自 AI／資料中心。箭頭＝上游供應給下游。</div>"]
    for k, (lid, lname, desc, analogy) in enumerate(supply.LAYERS):
        chips = []
        for gid, d in supply.SUBGROUPS.items():
            if d["layer"] != lid:
                continue
            r = gs.loc[gid] if gid in gs.index else None
            x20 = float(r["x20"]) if r is not None else np.nan
            tg = AI_TAG[d["ai"]]
            badge = f" {ui.tag(tg[0], tg[1])}" if tg[0] else ""
            sub = (f"近20日 {ui.pct(r['r20'], 1)}（相對大盤 {ui.pct(x20, 1)}）" if r is not None and np.isfinite(r["r20"]) else "")
            chips.append(f"<a class='cm-chip{' sel' if gid == sel else ''}' href='{base_url}?g={gid}' target='_self' "
                         f"style='background:{heat(x20)}'><div class='n'>{esc(d['name'])}{badge}</div><div class='s'>{sub}"
                         f"{'｜' if sub else ''}{int(r['n']) if r is not None else 0} 檔</div></a>")
        out.append(f"<div class='cm-layer'><div class='cm-lab'><div class='n'>{k + 1}. {esc(lname)}</div><div class='d'>{esc(desc)}"
                   f"<br><span style='color:var(--faint)'>比喻：{esc(analogy)}</span></div></div><div class='cm-chips'>{''.join(chips)}</div></div>")
        if k < len(supply.LAYERS) - 2:
            out.append("<div class='cm-arrow'>↓</div>")
        elif k == len(supply.LAYERS) - 2:
            out.append("<div class='cm-arrow'>↓<small>賣給國外終端客戶</small></div><div class='cm-cust' style='margin:0 0 10px 0'>"
                       + "".join(f"<span><b>{esc(n)}</b>：{esc(d)}</span>" for n, d in supply.CUSTOMERS) + "</div>"
                       "<div class='cm-leg' style='margin-top:8px'>延伸族群（不在晶片→伺服器主線上，但跟 AI 機房與新應用有關）：</div>")
    return "".join(out)


def overview_mobile_list(gs: pd.DataFrame) -> pd.DataFrame:
    return gs


# ------------------------------------------------------------------ 細族群拆解
def group_detail(gid: str, met: pd.DataFrame, rank_mid: dict, picks: set, cell) -> str:
    d = supply.SUBGROUPS[gid]
    tg = AI_TAG[d["ai"]]
    head = (f"<div style='font-size:13.5px;line-height:1.6'><b>{esc(d['name'])}</b> {ui.tag(supply.LAYER_NAME[d['layer']])}"
            f"{' ' + ui.tag(tg[0], tg[1]) if tg[0] else ''}<br>{esc(d['plain'])}</div>")
    rows, mrows = [], []
    for t, role in d["members"]:
        if t not in met.index:
            continue
        r = met.loc[t]
        rk = rank_mid.get(t)
        tag = ui.tag("名單", "gold") if t in picks else (ui.tag(f"中線第 {rk} 名") if rk and rk <= 50 else "")
        others = [supply.group_name(g) for g in supply.groups_of(t) if g != gid]
        th = supply.themes_of(t)
        rows.append([cell(t) + (" " + tag if tag else ""), esc(role), ui.pct(r["r1"], 2, color=True), ui.pct(r["r20"], 1, color=True),
                     ui.pct(r["r60"], 1, color=True), f"{r['amt20']:.1f}" if np.isfinite(r["amt20"]) else "–",
                     esc("、".join(others + th)) or "–"])
        mrows.append(ui.mcard(cell(t) + (" " + tag if tag else ""), ui.pct(r["r1"], 2, color=True), esc(role),
                              f"20 日 {ui.pct(r['r20'], 1)}｜60 日 {ui.pct(r['r60'], 1)}" + (f"｜也在：{esc('、'.join(others + th))}" if others or th else "")))
    tbl = ui.resp(ui.table(["公司", "在這個族群做什麼", "今日", "20 日", "60 日", "成交額(億/日)", "也屬於（橫向）"], rows, left=(0, 1, 6)),
                  "".join(mrows))
    up = [supply.group_name(g) for g in supply.suppliers(gid)]
    down = [supply.group_name(g) for g in supply.customers(gid)]
    rel = ui.kv([("上游（向誰買）", esc("、".join(up)) or "原料商、國外設備"), ("下游（賣給誰）", esc("、".join(down)) or "國外終端客戶"),
                 ("同業（橫向競爭）", f"{len(d['members'])} 檔代表股互為同業"),
                 ("相關題材", esc("、".join(th["name"] for th in supply.THEMES
                                         if gid in th.get("groups", []) or any(t in {m for m, _ in d['members']} for t, _ in th.get("members", [])))) or "–")])
    return head + "<div style='margin:8px 0'>" + rel + "</div>" + tbl


# ------------------------------------------------------------------ AI 伺服器拆解圖（SVG）
RACK_PARTS = [  # (標籤, 白話, 細族群)
    ("整櫃組裝", "把下面所有東西裝成一整櫃、測試、出貨", "server"),
    ("交換器・光模組", "伺服器之間傳資料；CPO 把光模組封進晶片旁", "optical"),
    ("運算托盤", "每盤放 4 顆 GPU＋CPU，一櫃 18 盤", "pcb"),
    ("水冷板・CDU", "冷卻液流過晶片上方把熱帶走", "thermal"),
    ("電源層・BBU", "整櫃供電＋停電時的備援電池", "power"),
    ("機殼・滑軌", "托盤可抽拉維修", "chassis"),
    ("線材・連接器", "托盤之間的高速銅線與插槽", "conn"),
    ("BMC 管理晶片", "每台伺服器的遠端管家", "bmc"),
]
CHIP_PARTS = [  # 由上到下的剖面
    ("GPU 晶片＋HBM 記憶體", "輝達設計、台積電 3 奈米製造；HBM 由韓美廠供應", "foundry"),
    ("中介層（CoWoS）", "把 GPU 與 HBM 並排連線的「矽轉接板」", "adv_pkg"),
    ("ABF 載板", "把晶片的極細線路放大到主機板", "abf"),
    ("主機板板材（CCL）", "做高層數主機板的高速低損耗銅箔基板", "ccl"),
]


def rack_svg(lead: dict[str, list[str]]) -> str:
    """lead：細族群 → 代表公司名稱（前 3 名）。左：機櫃分層；右：一顆 AI 晶片的剖面。"""
    W, H = 1000, 520
    s = [f"<svg class='cm-svg' viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='AI 伺服器拆解圖' style='max-width:1000px'>"]
    s.append("<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
             "<path d='M0,0 L10,5 L0,10 z' fill='var(--muted)'/></marker></defs>")
    # 機櫃
    x0, y0, rw, rh = 30, 40, 200, 440
    s.append(f"<text x='{x0}' y='26' fill='var(--strong)' font-size='15' font-weight='700'>① 一整櫃 AI 伺服器（約 1.5 公噸）</text>")
    s.append(f"<rect x='{x0}' y='{y0}' width='{rw}' height='{rh}' rx='10' fill='var(--panel2)' stroke='var(--line)' stroke-width='2'/>")
    n = len(RACK_PARTS)
    hh = (rh - 20) / n
    for i, (nm, plain, g) in enumerate(RACK_PARTS):
        y = y0 + 10 + i * hh
        col = "var(--accent2)" if g in ("pcb", "thermal") else "var(--accent)" if g in ("server", "optical") else "var(--faint)"
        s.append(f"<rect x='{x0 + 12}' y='{y + 4:.0f}' width='{rw - 24}' height='{hh - 8:.0f}' rx='5' fill='none' stroke='{col}' stroke-width='1.5'/>")
        s.append(f"<text x='{x0 + rw / 2}' y='{y + hh / 2 + 5:.0f}' text-anchor='middle' fill='var(--text)' font-size='13'>{esc(nm)}</text>")
        ty = y + hh / 2
        names = "、".join(lead.get(g, [])[:3])
        s.append(f"<line x1='{x0 + rw - 12}' y1='{ty:.0f}' x2='{x0 + rw + 28}' y2='{ty:.0f}' stroke='var(--muted)' stroke-width='1' marker-end='url(#ah)'/>")
        s.append(f"<text x='{x0 + rw + 34}' y='{ty - 2:.0f}' fill='var(--strong)' font-size='12.5' font-weight='600'>{esc(names)}</text>")
        s.append(f"<text x='{x0 + rw + 34}' y='{ty + 13:.0f}' fill='var(--muted)' font-size='11'>{esc(plain)}</text>")
    # 晶片剖面
    cx = 600
    s.append(f"<text x='{cx}' y='26' fill='var(--strong)' font-size='15' font-weight='700'>② 放大：一顆 AI 晶片的封裝剖面</text>")
    layers = [(cx + 40, 70, 300, 46), (cx + 20, 140, 340, 30), (cx, 194, 380, 34), (cx - 20, 252, 420, 40)]
    fills = ["rgba(233,185,73,.35)", "rgba(90,169,255,.30)", "rgba(240,70,75,.22)", "rgba(22,163,95,.25)"]
    for k, ((x, y, w, h), (nm, plain, g), f) in enumerate(zip(layers, CHIP_PARTS, fills)):
        if k == 0:                                   # 最上層：HBM｜GPU｜HBM 三塊並排
            for bx, bw, lab in ((x, 70, "HBM"), (x + 80, 140, "GPU 晶片"), (x + 230, 70, "HBM")):
                s.append(f"<rect x='{bx}' y='{y}' width='{bw}' height='{h}' rx='4' fill='{f}' stroke='var(--line)'/>")
                s.append(f"<text x='{bx + bw / 2}' y='{y + h / 2 + 5}' text-anchor='middle' fill='var(--strong)' font-size='13' font-weight='600'>{lab}</text>")
            continue
        s.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='4' fill='{f}' stroke='var(--line)'/>")
        s.append(f"<text x='{x + w / 2}' y='{y + h / 2 + 5}' text-anchor='middle' fill='var(--strong)' font-size='13' font-weight='600'>{esc(nm)}</text>")
    yy = 318
    for nm, plain, g in CHIP_PARTS:
        names = "、".join(lead.get(g, [])[:3])
        s.append(f"<text x='{cx - 20}' y='{yy}' fill='var(--strong)' font-size='12.5' font-weight='600'>{esc(nm)}：{esc(names)}</text>")
        s.append(f"<text x='{cx - 20}' y='{yy + 15}' fill='var(--muted)' font-size='11'>{esc(plain)}</text>")
        yy += 38
    s.append(f"<text x='{cx - 20}' y='{yy + 6}' fill='var(--muted)' font-size='11'>封裝後還要測試：探針卡、測試座（旺矽、穎崴）；設備：弘塑、辛耘、萬潤</text>")
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ 題材地圖
def theme_cards(met: pd.DataFrame, cell, mk: dict) -> str:
    out = []
    for th in supply.THEMES:
        ms = supply.theme_members(th, set(met.index))
        if not ms:
            continue
        r20 = float(np.nanmean([met.loc[t, "r20"] for t, _ in ms]))
        tags = ui.tag(th["stage"], STAGE_KIND.get(th["stage"], "")) + (" " + ui.tag("新題材／轉型", "gold") if th["new"] else "")
        chips = "".join(f"<div style='display:flex;justify-content:space-between;gap:8px;font-size:12.5px;padding:2px 0;border-bottom:1px dashed var(--line)'>"
                        f"<span>{cell(t)} <span class='kq-muted'>{esc(role)}</span></span><span>{ui.pct(met.loc[t, 'r20'], 1, color=True)}</span></div>"
                        for t, role in ms[:14])
        more = f"<div class='kq-muted' style='font-size:12px'>另有 {len(ms) - 14} 檔</div>" if len(ms) > 14 else ""
        out.append(f"<div class='kq-panel'><h4>{esc(th['name'])} {tags}</h4><div style='font-size:13px;line-height:1.55'>{esc(th['plain'])}</div>"
                   f"<div class='kq-muted' style='font-size:12px;margin:4px 0'>美國指標股：{esc(th['us'])}｜代表股近 20 日平均 {ui.pct(r20, 1)}"
                   f"（大盤 {ui.pct(mk.get('r20', np.nan), 1)}）</div>{chips}{more}</div>")
    return out


# ------------------------------------------------------------------ 消息衝擊評估
def depth_bar(x: int) -> str:
    if x == 0:
        return "<span class='kq-muted'>幾乎無</span>"
    w = abs(int(x)) * 16
    col = "#16a35f" if x < 0 else "#f0464b"
    return (f"<span style='white-space:nowrap'><span class='cm-bar' style='width:{w}px;background:{col}'></span> <b style='color:{col}'>"
            f"{'受傷' if x < 0 else '受惠'}・{['', '淺', '中', '深'][min(abs(int(x)), 3)]}</b></span>")


def steps_html(lines) -> str:
    return "".join(f"<div class='cm-step'><div class='k'>{i + 1}</div><div class='t'>{esc(x)}</div></div>" for i, x in enumerate(lines))


def impact_rows(sc: dict, hist: pd.DataFrame | None, since: pd.DataFrame | None, met: pd.DataFrame, lead: dict) -> tuple[list, list]:
    rows, mrows = [], []
    for gid, dep, speed, why in sc["impacts"]:
        h = hist.loc[gid] if hist is not None and gid in hist.index else None
        s_ = since.loc[gid] if since is not None and gid in since.index else None
        hs = f"{ui.pct(h['d5'], 1, color=True)} / {ui.pct(h['d20'], 1, color=True)}" if h is not None else "–"
        ss = ui.pct(s_, 1, color=True) if s_ is not None and np.isfinite(s_) else "–"
        names = "、".join(lead.get(gid, [])[:4])
        rows.append([f"<b>{esc(supply.group_name(gid))}</b>", depth_bar(dep), esc(speed), esc(why), hs, ss, esc(names)])
        mrows.append(ui.mcard(f"<b>{esc(supply.group_name(gid))}</b>", depth_bar(dep), f"{esc(speed)}｜{esc(why)}",
                              f"歷史同類 5／20 日 {hs}｜事件後至今 {ss}｜{esc(names)}"))
    return rows, mrows


def since_event(B, date: str, gids=None) -> pd.Series:
    """事件日前一日收盤到資料最後一天，細族群代表股相對全市場的累積超額。"""
    days = pd.DatetimeIndex(B.meta["days"])
    i = int(days.searchsorted(pd.Timestamp(date)))
    if i >= len(days) or i == 0:
        return pd.Series(dtype=float)
    out = {}
    gi = impact.group_index(B, gids)
    for g, js in gi.items():
        p = impact.path(B, js, str(days[i].date()), before=1, after=len(days) - 1 - i)
        out[g] = float(p.iloc[-1]) if len(p) else np.nan
    return pd.Series(out)


def lead_names(B, met: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """細族群 → 依成交金額排序的代表公司簡稱。"""
    nm = dict(zip(B.tickers, B.meta["names"]))
    out = {}
    for g, d in supply.SUBGROUPS.items():
        ts = [t for t, _ in d["members"] if t in nm]                       # 代表個股已依重要性排序（龍頭在前）
        out[g] = [nm[t].replace("*", "").replace("-KY", "") for t in ts]
    out["foundry"] = [x for x in out.get("foundry", []) if x.startswith("台積")] or out.get("foundry", [])   # AI 晶片只有台積電做
    return out


# ------------------------------------------------------------------ v3.1 台股熱力圖（treemap）
PERIODS = {"r1": ("今日", 0.03), "r5": ("5 日", 0.08), "r20": ("20 日", 0.15), "r60": ("60 日", 0.25)}


def treemap_frame(B, met: pd.DataFrame, labels: dict, top: int = 400, elec_only: bool = False) -> pd.DataFrame:
    """成交金額前 top 名：板塊（電子＝產業鏈層級；其他＝官方產業）→ 細族群 → 個股；面積＝近 20 日均成交金額。"""
    if met is None or not len(met) or "amt20" not in met:
        return pd.DataFrame()
    m = met[met["amt20"].notna()].sort_values("amt20", ascending=False).head(top)
    sec = dict(zip(B.tickers, B.meta.get("sectors") or [""] * len(B.tickers)))
    rows = []
    for t, r in m.iterrows():
        lab = labels.get(t) or {}
        g = supply.primary_group(t, lab)
        if g in supply.SUBGROUPS:
            d = supply.SUBGROUPS[g]
            board, grp = f"電子｜{supply.LAYER_NAME[d['layer']]}", d["name"]
        elif str(lab.get("group", "")).startswith("電子"):
            board, grp = "電子｜其他", str(lab.get("group")).split("-", 1)[-1]
        else:
            if elec_only:
                continue
            board, grp = "非電子", (sec.get(t) or "其他").replace("業", "") or "其他"
        rows.append({"t": t, "name": r["name"], "board": board, "group": grp, "amt": float(r["amt20"]),
                     **{k: float(r[k]) if k in r and np.isfinite(r[k]) else np.nan for k in PERIODS}})
    return pd.DataFrame(rows)


def treemap_figure(df: pd.DataFrame, period: str = "r1", dark: bool = True, height: int = 620):
    import plotly.graph_objects as go
    lim = PERIODS[period][1]
    mid = "#2b3140" if dark else "#e3e7ee"
    ids, labels, parents, values, colors, texts = [], [], [], [], [], []

    def wavg(x):
        v = x[period].to_numpy(float); w = x["amt"].to_numpy(float)
        ok = np.isfinite(v)
        return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
    for b, xb in df.groupby("board", sort=False):
        ids.append(b); labels.append(b); parents.append(""); values.append(float(xb["amt"].sum())); c = wavg(xb); colors.append(c)
        texts.append(ui.pct(c, 1) if np.isfinite(c) else "")
        for g, xg in xb.groupby("group", sort=False):
            gid = f"{b}/{g}"
            ids.append(gid); labels.append(g); parents.append(b); values.append(float(xg["amt"].sum())); c = wavg(xg); colors.append(c)
            texts.append(ui.pct(c, 1) if np.isfinite(c) else "")
            for r in xg.itertuples():
                ids.append(f"{gid}/{r.t}"); labels.append(str(r.name).replace("-KY", "").replace("*", "")); parents.append(gid); values.append(r.amt)
                v = getattr(r, period); colors.append(v); texts.append(f"{v * 100:+.1f}%" if np.isfinite(v) else r.t)
    col = [0.0 if not np.isfinite(c) else c for c in colors]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total", text=texts,
        texttemplate="<b>%{label}</b><br>%{text}", hovertemplate="<b>%{label}</b><br>%{text}<br>成交額 %{value:.1f} 億/日<extra></extra>",
        marker=dict(colors=col, colorscale=[[0, "#0f7a45"], [0.5, mid], [1, "#c62a33"]], cmin=-lim, cmax=lim, cmid=0,
                    line=dict(width=1, color="#0b1020" if dark else "#ffffff")),
        tiling=dict(pad=2), pathbar=dict(visible=True, thickness=22), maxdepth=3, insidetextfont=dict(size=12),
        textfont=dict(family="Noto Sans TC, Inter, sans-serif", color="#ffffff" if dark else "#0d1626")))
    fig.update_layout(height=height, margin=dict(l=2, r=2, t=26, b=2), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Noto Sans TC, Inter, sans-serif"))
    return fig
