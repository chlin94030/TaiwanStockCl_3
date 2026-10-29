"""介面樣式與 HTML 小元件（v1.7：專業看盤風格、手機／電腦自適應；漲紅跌綠、等寬數字）。

自適應做法：寬表格用 resp(桌機HTML, 手機HTML) 同時輸出兩種版面，CSS 依螢幕寬度只顯示一種
（≥ 760px 顯示表格，< 760px 顯示卡片），不需要偵測裝置。
"""
from __future__ import annotations

import html

import numpy as np

UP, DOWN, FLAT = "#f0464b", "#16a35f", "#94a3b8"

PALETTE = {
    True: {"bg": "#080d18", "bg2": "#0b1222", "panel": "#0f1729", "panel2": "#0c1424", "line": "#1c2740", "line2": "#151f33",
           "hover": "#132036", "text": "#d5deec", "strong": "#f4f7fb", "muted": "#8391a8", "faint": "#5d6b82",
           "accent": "#e9b949", "accent2": "#5aa9ff", "tagbg": "#141f35", "tagline": "#24324f", "tagtx": "#b3c1d8",
           "gbg": "#10281d", "gtx": "#78d3a6", "gline": "#1d4632", "ybg": "#29200d", "ytx": "#f2c35a", "yline": "#4b3b16",
           "shadow": "0 1px 0 rgba(255,255,255,.03) inset, 0 10px 30px -18px rgba(0,0,0,.8)",
           "zone_buy": "rgba(90,169,255,.30)", "zone_chase": "rgba(233,185,73,.28)", "zone_alert": "rgba(240,70,75,.22)",
           "track": "#1a253b"},
    False: {"bg": "#f3f5f9", "bg2": "#eef1f6", "panel": "#ffffff", "panel2": "#ffffff", "line": "#e0e6ef", "line2": "#eef2f7",
            "hover": "#f5f8fc", "text": "#1f2a3c", "strong": "#0d1626", "muted": "#647086", "faint": "#94a0b4",
            "accent": "#b07612", "accent2": "#1f6fe5", "tagbg": "#eef2f8", "tagline": "#d7dfea", "tagtx": "#334155",
            "gbg": "#e7f6ee", "gtx": "#12704a", "gline": "#bde5cf", "ybg": "#fcf2dd", "ytx": "#87590a", "yline": "#efd7a3",
            "shadow": "0 1px 2px rgba(16,24,40,.05), 0 8px 24px -16px rgba(16,24,40,.25)",
            "zone_buy": "rgba(31,111,229,.18)", "zone_chase": "rgba(176,118,18,.18)", "zone_alert": "rgba(220,38,38,.14)",
            "track": "#e9eef5"},
}


def css(dark: bool = True) -> str:
    v = PALETTE[bool(dark)]
    root = ":root{" + "".join(f"--{k}:{c};" for k, c in v.items()) + "}"
    return CSS_BODY.replace("<style>", "<style>" + root, 1)


CSS_BODY = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Roboto+Mono:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea, select {
  font-family: 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif; }
[data-testid="stAppViewContainer"] { background: radial-gradient(1200px 500px at 15% -10%, var(--bg2), var(--bg)) fixed; color: var(--text); }
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p, [data-testid="stExpander"] summary p { color: var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
.block-container { padding-top: 3.4rem; padding-bottom: 2.5rem; max-width: 1440px; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line); }
h1, h2, h3 { letter-spacing: .3px; }
h3 { font-size: 1.05rem !important; color: var(--strong); margin-top: .5rem !important; }
.stMarkdown p { line-height: 1.7; }
/* ---------- top bar */
.kq-bar { display:flex; flex-wrap:wrap; gap:6px 18px; align-items:center; background:var(--panel2); border:1px solid var(--line);
  border-radius:14px; padding:10px 16px; margin-bottom:12px; box-shadow:var(--shadow); }
.kq-brand { font-weight:700; font-size:17px; color:var(--accent); letter-spacing:2.5px; margin-right:4px; }
.kq-brand small { color:var(--muted); font-weight:400; letter-spacing:.5px; margin-left:6px; font-size:12px; }
.kq-item { color:var(--muted); font-size:12.5px; white-space:nowrap; }
.kq-item b { color:var(--strong); font-family:'Roboto Mono', monospace; font-weight:600; font-size:14px; margin-left:4px; }
.kq-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:500; white-space:nowrap; }
.kq-sep { width:1px; height:18px; background:var(--line); }
/* ---------- section title */
.kq-h { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin:18px 0 8px; }
.kq-h .t { font-size:17px; font-weight:700; color:var(--strong); padding-left:10px; border-left:3px solid var(--accent); line-height:1.1; }
.kq-h .s { color:var(--muted); font-size:12.5px; }
/* ---------- stat cards */
.kq-cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin: 4px 0 12px; }
.kq-card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }
.kq-card .l { color:var(--muted); font-size:12px; }
.kq-card .v { font-family:'Roboto Mono', monospace; font-size:21px; font-weight:600; color:var(--strong); margin-top:3px; }
.kq-card .s { color:var(--muted); font-size:11.5px; margin-top:3px; }
/* ---------- tables */
.kq-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:var(--panel2); box-shadow:var(--shadow); }
table.kq { width:100%; border-collapse:collapse; font-size:13.5px; white-space:nowrap; }
table.kq th { color:var(--muted); font-weight:500; font-size:12px; text-align:right; padding:9px 10px; border-bottom:1px solid var(--line);
  background:var(--panel2); position:sticky; top:0; }
table.kq td { padding:9px 10px; border-bottom:1px solid var(--line2); text-align:right; color:var(--text); vertical-align:middle;
  font-family:'Roboto Mono', 'Noto Sans TC', monospace; }
table.kq tr:last-child td { border-bottom:none; }
table.kq th.l, table.kq td.l { text-align:left; }
table.kq tr:hover td { background:var(--hover); }
table.kq td .nm { font-family:'Noto Sans TC', sans-serif; font-weight:500; color:var(--strong); }
table.kq td .cd, .kq-cd { color:var(--accent); font-weight:600; margin-right:6px; font-family:'Roboto Mono', monospace; }
table.kq td .sub, .kq-sub { color:var(--muted); font-size:11.5px; font-family:'Noto Sans TC', sans-serif; }
/* ---------- tags & badges */
.kq-tag { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11.5px; background:var(--tagbg); color:var(--tagtx);
  border:1px solid var(--tagline); margin:1px 3px 1px 0; font-family:'Noto Sans TC', sans-serif; white-space:nowrap; }
.kq-tag.g { background:var(--gbg); color:var(--gtx); border-color:var(--gline); }
.kq-tag.gold { background:var(--ybg); color:var(--ytx); border-color:var(--yline); }
.kq-badge { display:inline-block; padding:2px 9px; border-radius:7px; font-size:12px; color:#fff; font-family:'Noto Sans TC', sans-serif;
  white-space:nowrap; }
/* ---------- panels */
.kq-panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; margin-bottom:10px;
  box-shadow:var(--shadow); }
.kq-panel h4 { margin:0 0 8px; font-size:14px; color:var(--accent); font-weight:600; letter-spacing:.3px; }
.kq-panel ul { margin:0; padding-left:18px; } .kq-panel li { margin:3px 0; color:var(--text); font-size:13.5px; line-height:1.65; }
.kq-verdict { border-left:4px solid var(--accent); background:var(--panel); padding:13px 16px; border-radius:10px; margin:6px 0 12px;
  font-size:14.5px; box-shadow:var(--shadow); line-height:1.75; }
.kq-mono { font-family:'Roboto Mono', monospace; }
.kq-muted { color:var(--muted); font-size:12.5px; }
.kq-kv { display:flex; justify-content:space-between; gap:10px; padding:4px 0; border-bottom:1px dashed var(--line); font-size:13.5px; }
.kq-kv:last-child { border-bottom:none; }
.kq-step { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px; height:100%; box-shadow:var(--shadow); }
.kq-step .n { color:var(--bg); background:var(--accent); border-radius:999px; width:24px; height:24px; display:inline-flex;
  align-items:center; justify-content:center; font-weight:700; font-size:13px; margin-right:8px; }
.kq-step .t { font-weight:600; font-size:15px; color:var(--strong); }
.kq-step p, .kq-step li { color:var(--text); font-size:13.5px; line-height:1.65; }
/* ---------- world tiles */
.kq-tiles { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:10px; margin:4px 0 12px; }
.kq-tile { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px 13px; box-shadow:var(--shadow);
  position:relative; overflow:hidden; }
.kq-tile .l { color:var(--muted); font-size:12px; display:flex; justify-content:space-between; gap:6px; }
.kq-tile .v { font-family:'Roboto Mono', monospace; font-size:19px; font-weight:600; color:var(--strong); margin-top:2px; }
.kq-tile .c { font-family:'Roboto Mono', monospace; font-size:13px; font-weight:600; }
.kq-tile .s { color:var(--faint); font-size:11px; margin-top:2px; }
.kq-tile svg { display:block; margin-top:6px; opacity:.9; }
.kq-light { display:inline-flex; align-items:center; gap:8px; font-weight:600; }
.kq-light i { width:11px; height:11px; border-radius:50%; display:inline-block; box-shadow:0 0 0 3px color-mix(in srgb, currentColor 22%, transparent); }
/* ---------- price ladder */
.kq-lad { min-width:180px; }
.kq-lad .tr { position:relative; height:8px; border-radius:6px; background:var(--track); margin:6px 0 4px; }
.kq-lad .z { position:absolute; top:0; height:8px; }
.kq-lad .m { position:absolute; top:-4px; width:3px; height:16px; border-radius:2px; background:var(--strong);
  box-shadow:0 0 0 2px var(--panel); transform:translateX(-1px); }
.kq-lad .lb { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); font-family:'Roboto Mono', monospace; gap:6px; }
.kq-lad .lb b { color:var(--text); font-weight:600; }
/* ---------- v1.8 pick cards grid (desktop & mobile, never scrolls sideways) */
.kq-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:12px; margin:4px 0 10px; }
.kq-pcard { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 14px 11px; box-shadow:var(--shadow);
  display:flex; flex-direction:column; gap:6px; min-width:0; }
.kq-pcard:hover { border-color:var(--tagline); }
.kq-pcard .hd { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
.kq-pcard .rk { display:inline-flex; width:20px; height:20px; border-radius:6px; background:var(--tagbg); color:var(--muted);
  font-size:11.5px; align-items:center; justify-content:center; margin-right:7px; font-family:'Roboto Mono', monospace; }
.kq-pcard .nm { font-weight:700; color:var(--strong); font-size:15.5px; }
.kq-pcard .px { text-align:right; font-family:'Roboto Mono', monospace; font-weight:600; color:var(--strong); font-size:16px; line-height:1.1; }
.kq-pcard .px small { display:block; font-size:12px; margin-top:2px; }
.kq-pcard .grp { line-height:1.5; min-height:22px; }
.kq-pcard .sp svg { display:block; }
.kq-pcard .ft { display:flex; flex-wrap:wrap; gap:5px; align-items:center; font-size:12px; color:var(--muted); }
.kq-pcard .ft .why { flex:1 1 100%; color:var(--muted); line-height:1.45; }
table.kq td.l { white-space:normal; }
/* ---------- mobile cards */
.kq-mob { display:none; }
.kq-mcard { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 13px; margin-bottom:10px;
  box-shadow:var(--shadow); }
.kq-mcard .hd { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.kq-mcard .nm { font-weight:700; color:var(--strong); font-size:15.5px; }
.kq-mcard .rk { color:var(--faint); font-family:'Roboto Mono', monospace; font-size:12px; margin-right:6px; }
.kq-mcard .px { text-align:right; font-family:'Roboto Mono', monospace; font-weight:600; color:var(--strong); font-size:15.5px; line-height:1.15; }
.kq-mcard .px small { display:block; font-size:12px; }
.kq-mcard .mid { margin:6px 0 2px; }
.kq-mcard .ft { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:6px; font-size:12px; color:var(--muted); }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel2); }
div[data-testid="stMetricValue"] { font-family:'Roboto Mono', monospace; }
@media (max-width: 760px) {
  .block-container { padding-left: .55rem; padding-right: .55rem; padding-top: 3rem; }
  .kq-desk { display:none !important; }
  .kq-mob { display:block; }
  .kq-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px; }
  .kq-card { padding:10px 11px; } .kq-card .v { font-size:17px; } .kq-card .s { font-size:11px; }
  .kq-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px; }
  .kq-tile .v { font-size:16px; }
  .kq-bar { gap:4px 12px; padding:8px 11px; border-radius:12px; }
  .kq-bar .kq-hide-m { display:none; }
  .kq-h .t { font-size:16px; }
  table.kq { font-size:12.5px; }
  .kq-verdict { font-size:14px; padding:11px 13px; }
}
</style>
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def ok(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def price(x) -> str:
    if not ok(x):
        return "–"
    x = float(x)
    d = 2 if x < 50 else (1 if x < 500 else 0)          # 依台股升降單位顯示小數位
    return f"{x:,.{d}f}"


def num(x, d=2) -> str:
    return f"{float(x):,.{d}f}" if ok(x) else "–"


def color_of(x) -> str:
    return FLAT if not ok(x) or float(x) == 0 else (UP if float(x) > 0 else DOWN)


def chg(x, d=2) -> str:
    """漲跌幅：紅漲綠跌。"""
    if not ok(x):
        return "<span>–</span>"
    x = float(x)
    arrow = "▲" if x > 0 else ("▼" if x < 0 else "")
    return f"<span style='color:{color_of(x)}'>{arrow}{abs(x)*100:.{d}f}%</span>"


def pct(x, d=1, sign=True, color=False) -> str:
    if not ok(x):
        return "–"
    x = float(x)
    s = f"{x*100:+.{d}f}%" if sign else f"{x*100:.{d}f}%"
    if color:
        return f"<span style='color:{color_of(x)}'>{s}</span>"
    return s


def lots_fmt(x) -> str:
    if not ok(x):
        return "–"
    x = float(x)
    return f"<span style='color:{color_of(x)}'>{x:+,.0f}</span>"


def tag(t, kind="") -> str:
    return f"<span class='kq-tag {kind}'>{esc(t)}</span>"


def badge(text, color) -> str:
    return f"<span class='kq-badge' style='background:{color}'>{esc(text)}</span>"


def tone_badge(tone: str) -> str:
    col = {"偏多": "#b4232a", "偏空": "#167a4b", "中性": "#475569"}.get(tone, "#475569")
    return badge(tone, col)


def cards(items) -> str:
    """items: [(label, value_html, sub_html), ...]"""
    out = ["<div class='kq-cards'>"]
    for lab, val, sub in items:
        out.append(f"<div class='kq-card'><div class='l'>{esc(lab)}</div><div class='v'>{val}</div>"
                   f"<div class='s'>{sub or ''}</div></div>")
    out.append("</div>")
    return "".join(out)


def table(headers, rows, left=()) -> str:
    """headers: list[str]; rows: list[list[html]]; left: indices of left-aligned columns."""
    th = "".join(f"<th class='{'l' if i in left else ''}'>{esc(h)}</th>" for i, h in enumerate(headers))
    body = []
    for r in rows:
        body.append("<tr>" + "".join(f"<td class='{'l' if i in left else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    return f"<div class='kq-wrap'><table class='kq'><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def panel(title, body_html) -> str:
    return f"<div class='kq-panel'><h4>{esc(title)}</h4>{body_html}</div>"


def bullets(lines) -> str:
    return "<ul>" + "".join(f"<li>{x}</li>" for x in lines) + "</ul>"


def kv(rows) -> str:
    """rows: [(label, value_html), ...] → 左標籤右數值的清單。"""
    return "".join(f"<div class='kq-kv'><span class='kq-muted'>{esc(a)}</span><span class='kq-mono'>{b}</span></div>" for a, b in rows)


def section(title, sub="") -> str:
    return f"<div class='kq-h'><span class='t'>{esc(title)}</span><span class='s'>{sub}</span></div>"


def resp(desktop_html: str, mobile_html: str) -> str:
    """寬螢幕顯示 desktop_html、手機顯示 mobile_html（純 CSS 切換）。"""
    return f"<div class='kq-desk'>{desktop_html}</div><div class='kq-mob'>{mobile_html}</div>"


def spark(vals, w: int = 92, h: int = 26, color: str | None = None, fill: bool = True, fluid: bool = False) -> str:
    """迷你走勢圖（SVG）；顏色依期間漲跌自動紅／綠。"""
    v = np.asarray([x for x in (vals if vals is not None else []) if ok(x)], float)
    if len(v) < 2:
        return ""
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo or abs(hi) * 0.01 or 1.0
    xs = np.linspace(1, w - 1, len(v))
    ys = h - 2 - (v - lo) / rng * (h - 4)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    c = color or (UP if v[-1] >= v[0] else DOWN)
    area = (f"<polygon points='1,{h - 1} {pts} {w - 1},{h - 1}' fill='{c}' fill-opacity='.12'/>" if fill else "")
    if fluid:
        w2 = 300
        xs = np.linspace(1, w2 - 1, len(v))
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        area = (f"<polygon points='1,{h - 1} {pts} {w2 - 1},{h - 1}' fill='{c}' fill-opacity='.12'/>" if fill else "")
        return (f"<svg width='100%' height='{h + 8}' viewBox='0 0 {w2} {h}' preserveAspectRatio='none'>{area}"
                f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='1.6' vector-effect='non-scaling-stroke'/></svg>")
    return (f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' style='vertical-align:middle'>{area}"
            f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='1.6' stroke-linejoin='round'/>"
            f"<circle cx='{xs[-1]:.1f}' cy='{ys[-1]:.1f}' r='2' fill='{c}'/></svg>")


def ladder(px, low, high, chase, alert, compact: bool = False) -> str:
    """價位尺：警戒價 ─ 買進區(藍) ─ 追價區(金) ─ 追價上限，白線＝現價／收盤。"""
    if not all(ok(x) for x in (low, high, chase, alert)):
        return "–"
    lo_end = float(alert) * 0.985
    hi_end = float(chase) * 1.015
    span = max(hi_end - lo_end, 1e-9)

    def pos(x):
        return float(np.clip((float(x) - lo_end) / span * 100, 0, 100))
    zones = (f"<div class='z' style='left:0;width:{pos(alert):.1f}%;background:var(--zone_alert);border-radius:6px 0 0 6px'></div>"
             f"<div class='z' style='left:{pos(low):.1f}%;width:{pos(high) - pos(low):.1f}%;background:var(--zone_buy)'></div>"
             f"<div class='z' style='left:{pos(high):.1f}%;width:{pos(chase) - pos(high):.1f}%;background:var(--zone_chase)'></div>")
    mark = f"<div class='m' style='left:{pos(px):.1f}%'></div>" if ok(px) else ""
    if compact:
        lab = (f"<div class='lb'><span>警 <b>{price(alert)}</b></span><span>買 <b>{price(low)}–{price(high)}</b></span>"
               f"<span>追≤ <b>{price(chase)}</b></span></div>")
    else:
        lab = (f"<div class='lb'><span>警戒 <b>{price(alert)}</b></span><span>買進區 <b>{price(low)}～{price(high)}</b></span>"
               f"<span>追價上限 <b>{price(chase)}</b></span></div>")
    return f"<div class='kq-lad'><div class='tr'>{zones}{mark}</div>{lab}</div>"


def mcard(head_left: str, head_right: str, middle: str = "", foot: str = "") -> str:
    return (f"<div class='kq-mcard'><div class='hd'><div>{head_left}</div><div class='px'>{head_right}</div></div>"
            + (f"<div class='mid'>{middle}</div>" if middle else "") + (f"<div class='ft'>{foot}</div>" if foot else "") + "</div>")


def tile(label: str, value: str, change=None, sub: str = "", sparkline: str = "", note: str = "") -> str:
    ch = (f"<span class='c' style='color:{color_of(change)}'>{('▲' if change > 0 else '▼' if change < 0 else '')}"
          f"{abs(change)*100:.2f}%</span>") if ok(change) else ""
    return (f"<div class='kq-tile'><div class='l'><span>{esc(label)}</span>{ch}</div><div class='v'>{value}</div>"
            f"<div class='s'>{sub}</div>{sparkline}{note}</div>")


def light(level: str, text: str) -> str:
    c = {"green": "#16a35f", "amber": "#e9a23b", "red": "#f0464b"}.get(level, "#94a3b8")
    return f"<span class='kq-light' style='color:{c}'><i style='background:{c}'></i>{esc(text)}</span>"
