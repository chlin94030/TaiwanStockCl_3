"""看盤圖（Plotly）：K 線＋均線＋布林通道＋成交量＋KD＋MACD＋三大法人＋融資券＋大戶持股。台股配色：漲紅跌綠。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .ta import MA_LINES

UP, DOWN = "#f0464b", "#16a35f"
THEME = {True: ("#0f1726", "#1e2a3f", "#d6deeb", "#8a97ad"), False: ("#ffffff", "#e5eaf1", "#1e293b", "#64748b")}
MA_COLOR = {5: "#f5b942", 10: "#e879f9", 20: "#38bdf8", 60: "#a3e635", 120: "#fb923c", 240: "#c084fc"}
PANELS = {"volume": "成交量", "kd": "KD", "macd": "MACD", "inst": "三大法人", "margin": "融資券", "holders": "大戶持股"}
PANEL_H = {"volume": 0.13, "kd": 0.12, "macd": 0.12, "inst": 0.14, "margin": 0.12, "holders": 0.11}


def kline(ind: pd.DataFrame, mas=(5, 20, 60), boll: bool = True, panels=("volume", "kd", "macd"),
          plan=None, bars: int = 120, height: int | None = None, title: str = "", dark: bool = True,
          unit: str = "張", vol_label: str = "成交量(張)") -> go.Figure:
    BG, GRID, TXT, MUTED = THEME[bool(dark)]
    df = ind.tail(bars).copy()
    if df.empty:
        return go.Figure()
    panels = [p for p in panels if p in PANELS]
    if "holders" in panels and ("big400" not in df or df["big400"].notna().sum() == 0):
        panels.remove("holders")
    x = [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(df.index)]
    rows = 1 + len(panels)
    hs = [max(0.30, 1 - sum(PANEL_H[p] for p in panels))] + [PANEL_H[p] for p in panels]
    s = sum(hs)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.018, row_heights=[h / s for h in hs],
                        specs=[[{"secondary_y": p == "margin"}] for p in ["price"] + panels])
    fig.add_trace(go.Candlestick(x=x, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K線", showlegend=False,
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
                                 hoverlabel=dict(namelength=0)), row=1, col=1)
    for n in mas:
        c = f"ma{n}"
        if c in df and df[c].notna().any():
            fig.add_trace(go.Scatter(x=x, y=df[c], name=MA_LINES.get(n, f"MA{n}"), mode="lines",
                                     line=dict(color=MA_COLOR.get(n, "#ccc"), width=1.2)), row=1, col=1)
    if boll and "bb_up" in df:
        fig.add_trace(go.Scatter(x=x, y=df["bb_up"], name="布林通道", mode="lines",
                                 line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["bb_dn"], showlegend=False, name="布林下軌", mode="lines", fill="tonexty",
                                 fillcolor="rgba(148,163,184,0.06)",
                                 line=dict(color="rgba(148,163,184,0.7)", width=1, dash="dot")), row=1, col=1)
    if plan is not None:
        for y, nm, col, dash in ((plan.buy_high, "買上", "#3b82f6", "solid"), (plan.buy_low, "拉回", "#06b6d4", "dot"),
                                 (plan.chase, "追價", "#f59e0b", "dash"), (plan.alert, "警戒", "#ef4444", "dash")):
            if y and np.isfinite(y):
                fig.add_hline(y=y, line=dict(color=col, width=1, dash=dash), row=1, col=1,
                              annotation_text=f"{nm} {y:g}", annotation_position="top left",
                              annotation_font=dict(color=col, size=10), annotation_bgcolor=BG)
    r, mrow = 2, None
    up = (df["close"] >= df["open"]).to_numpy()
    for p in panels:
        if p == "volume":
            fig.add_trace(go.Bar(x=x, y=df["volume"], showlegend=False, name=vol_label, marker_color=np.where(up, UP, DOWN),
                                 marker_line_width=0, opacity=0.85), row=r, col=1)
            if "vma20" in df:
                fig.add_trace(go.Scatter(x=x, y=df["vma20"], showlegend=False, name="20日均量", mode="lines",
                                         line=dict(color="#f5b942", width=1)), row=r, col=1)
        elif p == "kd":
            fig.add_trace(go.Scatter(x=x, y=df["k"], showlegend=False, name="K(9)", mode="lines", line=dict(color="#f59e0b", width=1.2)), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["d"], showlegend=False, name="D(9)", mode="lines", line=dict(color="#38bdf8", width=1.2)), row=r, col=1)
            for lv in (20, 80):
                fig.add_hline(y=lv, line=dict(color=GRID, width=1, dash="dot"), row=r, col=1)
        elif p == "macd":
            fig.add_trace(go.Bar(x=x, y=df["osc"], showlegend=False, name="OSC", marker_color=np.where(df["osc"] >= 0, UP, DOWN),
                                 marker_line_width=0, opacity=0.8), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["dif"], showlegend=False, name="DIF", mode="lines", line=dict(color="#f5b942", width=1.2)), row=r, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["dem"], showlegend=False, name="MACD", mode="lines", line=dict(color="#38bdf8", width=1.2)), row=r, col=1)
        elif p == "inst":
            for col, nm, cc in (("foreign", "外資", "#60a5fa"), ("trust", "投信", "#f472b6"), ("dealer", "自營商", "#facc15")):
                if col in df:
                    fig.add_trace(go.Bar(x=x, y=df[col], name=f"{nm}({unit})", marker_color=cc, marker_line_width=0), row=r, col=1)
        elif p == "margin":
            if "mbal" in df:
                fig.add_trace(go.Scatter(x=x, y=df["mbal"], showlegend=False, name=f"融資餘額({unit})", mode="lines",
                                         line=dict(color="#fb7185", width=1.4)), row=r, col=1, secondary_y=False)
            if "sbal" in df:
                fig.add_trace(go.Scatter(x=x, y=df["sbal"], showlegend=False, name=f"融券餘額({unit})", mode="lines",
                                         line=dict(color="#34d399", width=1.4)), row=r, col=1, secondary_y=True)
        elif p == "holders":
            fig.add_trace(go.Scatter(x=x, y=df["big400"], showlegend=False, name="大戶≥400張(%)", mode="lines", connectgaps=True,
                                     line=dict(color="#f59e0b", width=1.6, shape="hv")), row=r, col=1)
            if "retail50" in df:
                fig.add_trace(go.Scatter(x=x, y=df["retail50"], showlegend=False, name="散戶<50張(%)", mode="lines", connectgaps=True,
                                         line=dict(color="#94a3b8", width=1.2, shape="hv")), row=r, col=1)
        fig.update_yaxes(title_text=vol_label.split("(")[0] if p == "volume" else PANELS[p], title_font=dict(size=10, color=MUTED),
                         row=r, col=1)
        if p == "margin":
            mrow = r
        r += 1
    if "inst" in panels:
        fig.update_layout(barmode="relative")
    fig.update_xaxes(type="category", showgrid=False, rangeslider_visible=False, nticks=8, tickfont=dict(color=MUTED, size=10),
                     linecolor=GRID, showspikes=True, spikemode="across", spikethickness=1, spikecolor="#64748b")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED, size=10), side="right")
    if mrow is not None:                                   # 融券用左側座標軸，避免和融資的刻度疊在一起
        fig.update_yaxes(side="left", showgrid=False, title_text="", tickfont=dict(color="#34d399", size=9), row=mrow, col=1,
                         secondary_y=True)
        fig.update_yaxes(tickfont=dict(color="#fb7185", size=10), row=mrow, col=1, secondary_y=False)
    fig.update_layout(height=height or int(380 + 120 * len(panels)), paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color=TXT, family="Inter, Noto Sans TC, PingFang TC, Microsoft JhengHei, sans-serif"),
                      margin=dict(l=40 if mrow is not None else 8, r=64, t=36 if title else 10, b=28), hovermode="x unified",
                      title=dict(text=title, font=dict(size=14)) if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                                  traceorder="normal"),
                      xaxis_rangeslider_visible=False, dragmode="pan")
    return fig


def nav_figure(navs: dict, height: int = 300, log: bool = True, dark: bool = True) -> go.Figure:
    BG, GRID, TXT, MUTED = THEME[bool(dark)]
    colors = ["#f5b942" if dark else "#b7791f", "#38bdf8", "#a3e635", "#f472b6", "#94a3b8"]
    fig = go.Figure()
    for k, (nm, sr) in enumerate(navs.items()):
        sr = sr.dropna()
        if len(sr):
            fig.add_trace(go.Scatter(x=sr.index, y=sr / sr.iloc[0], name=nm, mode="lines",
                                     line=dict(width=2 if k == 0 else 1.4, color=colors[k % len(colors)])))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
                      margin=dict(l=8, r=48, t=10, b=26), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, tickfont=dict(color=MUTED), side="right", type="log" if log else "linear")
    return fig


# ------------------------------------------------------------------ v2.1.1 當日分時走勢（平盤線、紅漲綠跌）
def _cross(x: np.ndarray, y: np.ndarray, base: float) -> tuple[np.ndarray, np.ndarray]:
    """在穿越平盤的地方插入交點，讓紅、綠兩段剛好在平盤線上銜接。x：int64 奈秒。"""
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(y)):
        a, b = y[i - 1] - base, y[i] - base
        if a * b < 0:
            f = a / (a - b)
            xs.append(x[i - 1] + (x[i] - x[i - 1]) * f)
            ys.append(base)
        xs.append(x[i])
        ys.append(y[i])
    return np.asarray(xs, dtype="int64"), np.asarray(ys, float)


def _polys(x, y, base, up: bool):
    """平盤上方（或下方）每一段的填色多邊形，以 None 分段（fill='toself' 會各自封閉）。"""
    keep = (y >= base) if up else (y <= base)
    px, py, i, n = [], [], 0, len(y)
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and keep[j + 1]:
            j += 1
        if j > i:
            px += list(x[i:j + 1]) + [x[j], x[i], None]
            py += list(y[i:j + 1]) + [base, base, None]
        i = j + 1
    return px, py


def _nice_step(amp_pct: float) -> float:
    for s in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10):
        if amp_pct / s <= 4.5:
            return s
    return 10.0


def intraday_figure(d: dict, height: int = 300, dark: bool = True, digits: int = 2, volume: bool = True,
                    vol_div: float = 1000.0, vol_label: str = "量(張)") -> go.Figure:
    """分時走勢：平盤虛線置中（上下對稱），右側價格、左側漲跌幅；線在平盤上方為紅、下方為綠；
    標出最高、最低與最新價；有成交量時下方畫量（紅＝比前一筆漲、綠＝跌）。"""
    BG, GRID, TXT, MUTED = THEME[bool(dark)]
    t = pd.DatetimeIndex(d["t"])
    y = np.asarray(d["v"], float)
    base = float(d["prev"])
    vol = np.asarray(d.get("vol", np.full(len(y), np.nan)), float) / vol_div
    has_vol = volume and np.nansum(np.nan_to_num(vol)) > 0 and len(y) > 5
    fig = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.78, 0.22] if has_vol else [1.0])
    xs, ys = _cross(t.values.astype("datetime64[ns]").astype("int64"), y, base)
    xdt = pd.to_datetime(xs)
    up_y = np.where(ys >= base, ys, np.nan)
    dn_y = np.where(ys <= base, ys, np.nan)
    for upside, col, yy in ((True, UP, up_y), (False, DOWN, dn_y)):
        px, py = _polys(xs, ys, base, upside)
        if px:
            fig.add_trace(go.Scatter(x=[pd.Timestamp(v) if v is not None else None for v in px], y=py, mode="lines",
                                     fill="toself", fillcolor=(f"rgba(240,70,75,{0.16 if dark else 0.12})" if upside
                                                               else f"rgba(22,163,95,{0.16 if dark else 0.12})"),
                                     line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=xdt, y=yy, mode="lines", line=dict(color=col, width=2), connectgaps=False,
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
    pct = (y / base - 1) * 100
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"), showlegend=False,
                             customdata=np.c_[pct], name="",
                             hovertemplate=f"%{{x|%H:%M}}　%{{y:,.{digits}f}}　(%{{customdata[0]:+.2f}}%)<extra></extra>"),
                  row=1, col=1)
    hi, lo = float(d["high"]), float(d["low"])
    amp = max(abs(hi - base), abs(lo - base), abs(y - base).max(), base * 0.003) * 1.3
    step = _nice_step(amp / base * 100)
    k = int(np.floor(amp / base * 100 / step + 1e-9))
    tv = [base * (1 + i * step / 100) for i in range(-k, k + 1)]

    def pcol(i):
        return UP if i > 0 else (DOWN if i < 0 else MUTED)
    fig.update_yaxes(range=[base - amp, base + amp], tickvals=tv, side="right", gridcolor=GRID, zeroline=False,
                     ticktext=[f"<span style='color:{pcol(i)}'>{v:,.{digits}f}</span>" for i, v in zip(range(-k, k + 1), tv)],
                     tickfont=dict(size=10, color=MUTED), row=1, col=1)
    # 左側漲跌幅（同一組刻度）
    fig.add_trace(go.Scatter(x=[t[0], t[0]], y=[base - amp, base + amp], yaxis="y3", mode="markers",
                             marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    fig.update_layout(yaxis3=dict(overlaying="y", side="left", range=[-amp / base * 100, amp / base * 100],
                                  tickvals=[i * step for i in range(-k, k + 1)], showgrid=False, zeroline=False,
                                  ticktext=[f"<span style='color:{pcol(i)}'>{i * step:+.{1 if step < 1 else 0}f}%</span>"
                                            if i else "0%" for i in range(-k, k + 1)],
                                  tickfont=dict(size=10, color=MUTED)))
    fig.add_hline(y=base, line=dict(color=MUTED, dash="dash", width=1.2), row=1, col=1)
    fig.add_annotation(x=d["end"], y=base, xref="x", yref="y", text=f"平盤 {base:,.{digits}f}", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(size=10.5, color=MUTED), bgcolor=BG, borderpad=1)
    span_s = max((d["end"] - d["start"]).total_seconds(), 1.0)
    # 最高、最低（放在線的最高、最低點；數值用當日官方高低）
    for val, idx, sym, ya, col in ((hi, int(np.argmax(y)), "最高", "bottom", UP if hi > base else DOWN),
                                   (lo, int(np.argmin(y)), "最低", "top", UP if lo > base else DOWN)):
        yy_ = val if abs(val / y[idx] - 1) < 0.01 else y[idx]
        fx = (t[idx] - d["start"]).total_seconds() / span_s
        fig.add_annotation(x=t[idx], y=yy_, text=f"{sym} {val:,.{digits}f}（{(val / base - 1) * 100:+.2f}%）",
                           showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=col, ax=0, ay=-10 if ya == "bottom" else 10,
                           xanchor="left" if fx < 0.15 else ("right" if fx > 0.85 else "center"),
                           font=dict(size=10.5, color=col), bgcolor=BG, borderpad=1, row=1, col=1)
    last = float(y[-1])
    lc = UP if last > base else (DOWN if last < base else MUTED)
    fig.add_trace(go.Scatter(x=[t[-1]], y=[last], mode="markers", marker=dict(size=8, color=lc, line=dict(color=BG, width=2)),
                             hoverinfo="skip", showlegend=False, cliponaxis=False), row=1, col=1)
    if has_vol:
        vv = np.nan_to_num(vol)
        cc = [UP if i == 0 and y[0] >= base or i > 0 and y[i] >= y[i - 1] else DOWN for i in range(len(y))]
        fig.add_trace(go.Bar(x=t, y=vv, marker=dict(color=cc, line=dict(width=0)), opacity=0.75, name=vol_label,
                             hovertemplate=f"%{{x|%H:%M}}　{vol_label} %{{y:,.0f}}<extra></extra>", showlegend=False), row=2, col=1)
        fig.update_yaxes(side="right", gridcolor=GRID, tickfont=dict(size=9, color=MUTED), row=2, col=1,
                         title=dict(text=vol_label, font=dict(size=9, color=MUTED)))
    span = (d["end"] - d["start"]).total_seconds()
    fig.update_xaxes(range=[d["start"], d["end"]], tickformat="%H:%M", showgrid=True, gridcolor=GRID,
                     dtick=3600000 if span <= 9 * 3600 else 4 * 3600000, tick0=d["start"].floor("h"),
                     tickfont=dict(size=10, color=MUTED), showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor="#64748b", spikedash="dot")
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, hovermode="x",
                      font=dict(color=TXT, family="Inter, Noto Sans TC, PingFang TC, Microsoft JhengHei, sans-serif"),
                      margin=dict(l=46, r=64, t=8, b=24), dragmode=False, showlegend=False,
                      hoverlabel=dict(bgcolor=BG, font=dict(color=TXT, size=12)))
    return fig
