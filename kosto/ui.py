"""介面樣式與 HTML 小元件（v1.7：專業看盤風格、手機／電腦自適應；漲紅跌綠、等寬數字）。

自適應做法：寬表格用 resp(桌機HTML, 手機HTML) 同時輸出兩種版面，CSS 依螢幕寬度只顯示一種
（≥ 760px 顯示表格，< 760px 顯示卡片），不需要偵測裝置。
"""
from __future__ import annotations

import html
import itertools

import numpy as np
import pandas as pd

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
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea, select {
  font-family: 'Inter', 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif; font-variant-numeric: tabular-nums; }
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
.kq-item b { color:var(--strong); font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; font-weight:600; font-size:14px; margin-left:4px; }
.kq-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:500; white-space:nowrap; }
.kq-sep { width:1px; height:18px; background:var(--line); }
.kq-small { font-size:12.5px; }
.kq-small .kq-light { font-weight:500; gap:6px; } .kq-small .kq-light i { width:8px; height:8px; }
/* ---------- section title */
.kq-h { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin:18px 0 8px; }
.kq-h .t { font-size:17px; font-weight:700; color:var(--strong); padding-left:10px; border-left:3px solid var(--accent); line-height:1.1; }
.kq-h .s { color:var(--muted); font-size:12.5px; }
/* ---------- stat cards */
.kq-cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:10px; margin: 4px 0 12px; }
.kq-card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }
.kq-card .l { color:var(--muted); font-size:12px; }
.kq-card .v { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.4px; font-size:24px; font-weight:700; color:var(--strong); margin-top:3px; }
.kq-card .s { color:var(--muted); font-size:11.5px; margin-top:3px; }
/* ---------- tables */
.kq-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:var(--panel2); box-shadow:var(--shadow); }
table.kq { width:100%; border-collapse:collapse; font-size:13.5px; white-space:nowrap; }
table.kq th { color:var(--muted); font-weight:500; font-size:12px; text-align:right; padding:9px 10px; border-bottom:1px solid var(--line);
  background:var(--panel2); position:sticky; top:0; }
table.kq td { padding:9px 10px; border-bottom:1px solid var(--line2); text-align:right; color:var(--text); vertical-align:middle;
  font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; }
table.kq tr:last-child td { border-bottom:none; }
table.kq th.l, table.kq td.l { text-align:left; }
table.kq tr:hover td { background:var(--hover); }
table.kq td .nm { font-family:'Noto Sans TC', sans-serif; font-weight:500; color:var(--strong); }
table.kq td .cd, .kq-cd { color:var(--accent); font-weight:600; margin-right:6px; font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; }
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
.kq-mono { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; }
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
.kq-tile .v { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; font-size:19px; font-weight:600; color:var(--strong); margin-top:2px; }
.kq-tile .c { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; font-size:13px; font-weight:600; }
.kq-tile .s { color:var(--faint); font-size:11px; margin-top:2px; }
.kq-tile svg { display:block; margin-top:6px; opacity:.9; }
.kq-light { display:inline-flex; align-items:center; gap:8px; font-weight:600; }
.kq-strip { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:-2px 0 10px; font-size:12px; }
.kq-chip { font-family:'Inter',sans-serif; font-variant-numeric:tabular-nums; padding:2px 8px; border-radius:999px;
  border:1px solid var(--line); color:var(--muted); background:var(--panel); white-space:nowrap; }
.kq-chip.late { color:#e9a23b; border-color:#e9a23b; }
.kq-warn { border:1px solid #e9a23b; border-left-width:4px; border-radius:10px; padding:8px 12px; margin:0 0 10px;
  font-size:13px; line-height:1.6; background:color-mix(in srgb, #e9a23b 8%, transparent); }
.kq-warn a { color:var(--accent); font-weight:600; }
.kq-tag { display:inline-block; font-size:11px; padding:1px 6px; border-radius:6px; margin-left:4px;
  background:color-mix(in srgb, var(--accent) 18%, transparent); color:var(--accent); font-weight:600; }
.kq-light i { width:11px; height:11px; border-radius:50%; display:inline-block; box-shadow:0 0 0 3px color-mix(in srgb, currentColor 22%, transparent); }
/* ---------- price ladder */
.kq-lad { min-width:180px; }
.kq-lad .tr { position:relative; height:8px; border-radius:6px; background:var(--track); margin:6px 0 4px; }
.kq-lad .z { position:absolute; top:0; height:8px; }
.kq-lad .m { position:absolute; top:-4px; width:3px; height:16px; border-radius:2px; background:var(--strong);
  box-shadow:0 0 0 2px var(--panel); transform:translateX(-1px); }
.kq-lad .lb { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; gap:6px; }
.kq-lad .lb b { color:var(--text); font-weight:600; }
/* ---------- v1.8 pick cards grid (desktop & mobile, never scrolls sideways) */
.kq-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:12px; margin:4px 0 10px; }
.kq-pcard { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 14px 11px; box-shadow:var(--shadow);
  display:flex; flex-direction:column; gap:6px; min-width:0; }
.kq-pcard:hover { border-color:var(--tagline); }
.kq-pcard .hd { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
.kq-pcard .rk { display:inline-flex; width:20px; height:20px; border-radius:6px; background:var(--tagbg); color:var(--muted);
  font-size:11.5px; align-items:center; justify-content:center; margin-right:7px; font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; }
.kq-pcard .nm { font-weight:700; color:var(--strong); font-size:15.5px; }
.kq-pcard .px { text-align:right; font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; font-weight:600; color:var(--strong); font-size:16px; line-height:1.1; }
.kq-pcard .px small { display:block; font-size:12px; margin-top:2px; }
.kq-pcard .grp { line-height:1.5; min-height:22px; }
.kq-pcard .sp svg { display:block; }
.kq-alert { border:1px solid #f0464b; border-left-width:5px; border-radius:12px; padding:10px 14px; margin:0 0 10px;
  background:linear-gradient(90deg, rgba(240,70,75,.14), rgba(240,70,75,.03)); }
.kq-alert .t { font-weight:700; color:#f0464b; font-size:16px; margin-bottom:2px; }
.kq-alert .s { color:var(--text); font-size:13.5px; } .kq-alert .s2 { color:var(--muted); font-size:12.5px; margin-top:4px; }
.kq-lock { display:inline-block; font-size:10.5px; font-weight:700; line-height:1; padding:2px 5px; border-radius:4px; margin-left:4px; color:#fff; vertical-align:middle; } .kq-lock.up { background:#e5383b; } .kq-lock.dn { background:#16a35f; }
.kq-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; background:#94a3b8; }
.kq-dot.red { background:#f0464b; box-shadow:0 0 0 3px rgba(240,70,75,.18); } .kq-dot.amber { background:#e9a23b; box-shadow:0 0 0 3px rgba(233,162,59,.18); }
.kq-dot.green { background:#16a35f; box-shadow:0 0 0 3px rgba(22,163,95,.16); }
.kq-flags { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 2px; }
.kq-chkgrid { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:6px; }
.kq-chk { border:1px solid var(--line); border-radius:10px; padding:7px 9px; background:var(--panel); font-size:12px; color:var(--muted); line-height:1.35; }
.kq-chk b { display:block; color:var(--muted); font-size:13px; margin-bottom:1px; } .kq-chk.on { border-color:rgba(22,163,95,.55); }
.kq-chk.on b { color:#16a35f; }
.kq-ch { position:relative; }
.kq-ch svg { position:absolute; left:0; top:0; display:block; margin:0 !important; opacity:1 !important; overflow:visible; }
.kq-ch .dt { position:absolute; width:7px; height:7px; border-radius:50%; transform:translate(-50%,-50%);
  box-shadow:0 0 0 2px var(--panel); pointer-events:none; }
.kq-ch .lb { position:absolute; font-size:10.5px; line-height:1.1; color:var(--muted); white-space:nowrap; pointer-events:none;
  font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px;
  text-shadow:0 0 3px var(--panel), 0 0 3px var(--panel), 0 0 2px var(--panel), 0 0 1px var(--panel); }
.kq-ch .lb b { color:var(--text); font-weight:600; }
.kq-ch .tl { left:3px; top:1px; } .kq-ch .tr { right:3px; top:1px; } .kq-ch .br { right:3px; bottom:0; }
.kq-ch .bl { left:3px; bottom:0; color:var(--faint); }
.kq-ch .bsw { display:inline-block; width:12px; border-top:1px dashed var(--muted); vertical-align:middle; margin-right:4px; } .kq-ch .bs { right:3px; transform:translateY(-50%); color:var(--faint); font-size:10px; }
.kq-pcard .sp { margin:6px 0 4px; }
.kq-tile .kq-ch { margin-top:7px; }
.kq-imini { display:flex; align-items:center; gap:8px; padding:5px 0; border-bottom:1px dashed var(--line); font-size:13px; }
.kq-imini:last-child { border-bottom:0; }
.kq-imini .n { flex:0 0 92px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--text); }
.kq-imini .c { flex:0 0 62px; text-align:right; font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; font-weight:600; }
.kq-imini .g { flex:1 1 auto; min-width:60px; }
.kq-istat { display:flex; flex-wrap:wrap; gap:4px 16px; align-items:baseline; margin:2px 0 6px; font-size:12.5px; color:var(--muted); }
.kq-istat .big { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; font-size:22px; font-weight:700; color:var(--strong); }
.kq-istat b { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; color:var(--text); font-weight:600; }
.kq-pcard .ft { display:flex; flex-wrap:wrap; gap:5px; align-items:center; font-size:12px; color:var(--muted); }
.kq-pcard .ft .why { flex:1 1 100%; color:var(--muted); line-height:1.45; }
table.kq td.l { white-space:normal; }
/* ---------- mobile cards */
.kq-mob { display:none; }
.kq-mcard { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px 13px; margin-bottom:10px;
  box-shadow:var(--shadow); }
.kq-mcard .hd { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.kq-mcard .nm { font-weight:700; color:var(--strong); font-size:15.5px; }
.kq-mcard .rk { color:var(--faint); font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; font-size:12px; margin-right:6px; }
.kq-mcard .px { text-align:right; font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; font-weight:600; color:var(--strong); font-size:15.5px; line-height:1.15; }
.kq-mcard .px small { display:block; font-size:12px; }
.kq-mcard .mid { margin:6px 0 2px; }
.kq-mcard .ft { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:6px; font-size:12px; color:var(--muted); }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel2); }
div[data-testid="stMetricValue"] { font-family:'Inter', 'Noto Sans TC', sans-serif; font-variant-numeric: tabular-nums; letter-spacing:-.1px; }
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
  /* v2.2 手機：頁首一行、說明字最多兩行、面板與表格更緊湊 */
  .kq-top { font-size:12px; gap:3px 10px; padding:7px 10px; }
  .kq-top .kq-brand small, .kq-top .kq-sub { display:none; }
  .kq-top .kq-pill { margin-left:0 !important; padding:1px 7px; font-size:11px; }
  .kq-strip { gap:4px; margin:-4px 0 8px; font-size:11px; } .kq-strip .kq-chip { padding:1px 6px; }
  .kq-h { margin:14px 0 6px; } .kq-h .s { font-size:11.5px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .kq-panel { padding:11px 12px; border-radius:12px; } .kq-panel li { font-size:13px; line-height:1.55; }
  .kq-warn, .kq-alert { font-size:12.5px; padding:8px 11px; } .kq-alert .t { font-size:15px; }
  .kq-istat { gap:2px 10px; font-size:12px; } .kq-istat .big { font-size:19px; }
  .kq-imini .n { flex-basis:78px; font-size:12px; } .kq-imini .c { flex-basis:54px; font-size:12px; }
  .kq-mcard { padding:10px 11px; margin-bottom:8px; }
  [data-testid="stTabs"] button p { font-size:13px; }
  [data-testid="stExpander"] summary p { font-size:13px; }
}
/* ---------- v2.2.2 不同裝置：電腦（≥1101px）／平板（761–1100px）／手機（≤760px）＋觸控 */
.kq-qnav { display:none; }
@media (max-width: 1100px) {
  /* 快速切換頁面：可左右滑動，固定在上方 */
  .kq-qnav { display:flex; gap:6px; overflow-x:auto; -webkit-overflow-scrolling:touch; scrollbar-width:none; margin:-2px 0 10px;
    padding:4px 0 6px; position:sticky; top:2.85rem; z-index:20; background:linear-gradient(var(--bg) 70%, transparent); }
  .kq-qnav::-webkit-scrollbar { display:none; }
  .kq-qnav a { flex:0 0 auto; text-decoration:none; color:var(--muted); font-size:13px; padding:6px 12px; border-radius:999px;
    border:1px solid var(--line); background:var(--panel); white-space:nowrap; }
  .kq-qnav a.on { color:var(--strong); border-color:var(--accent); background:var(--tagbg); font-weight:600; }
}
@media (min-width: 761px) and (max-width: 1100px) {
  /* 平板：欄位太窄時自動換行（例：三欄面板 → 兩欄＋一欄；圖＋側欄 → 上下排） */
  div[data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:.75rem; }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width:300px; }
  .block-container { padding-left:1.1rem; padding-right:1.1rem; }
  .kq-cards { grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); }
  .kq-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
  table.kq { font-size:12.5px; } table.kq th, table.kq td { padding:7px 8px; }
  .kq-bar { gap:4px 14px; }
}
@media (max-width: 900px) {
  /* 直式平板與手機：寬表格改用卡片 */
  .kq-desk { display:none !important; }
  .kq-mob { display:block; }
}
@media (min-width: 600px) and (max-width: 900px) {
  .kq-mob:has(> .kq-mcard) { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; align-items:start; }
  .kq-mob > .kq-mcard { margin-bottom:0; }
  .kq-chkgrid { grid-template-columns:repeat(4, minmax(0, 1fr)); }
}
@media (pointer: coarse) {
  /* 觸控：按鈕、分頁加大，方便手指點選 */
  [data-testid="stTabs"] button { min-height:42px; }
  [data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button { min-height:40px; }
  [data-testid="stExpander"] summary { min-height:44px; }
  .kq-qnav a { padding:8px 14px; }
}
@media (max-height: 500px) and (orientation: landscape) {
  /* 手機橫放：減少上方留白，讓圖表多一點空間 */
  .block-container { padding-top:2.6rem; }
  .kq-qnav { position:static; }
}
@media (min-width: 1600px) {
  .block-container { max-width:1520px; }
  .kq-grid { grid-template-columns:repeat(4, minmax(0, 1fr)); }
}
/* ---------- v3.1 美化：卡片懸停、今日行動看板、股票籤、分頁底線、區塊標題 */
.kq-card, .kq-pcard, .kq-tile, .kq-panel { transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease; }
@media (hover:hover) { .kq-card:hover, .kq-pcard:hover, .kq-tile:hover { border-color:color-mix(in srgb, var(--accent) 55%, var(--line)); transform:translateY(-1px); } }
.kq-h .t { background:linear-gradient(90deg, var(--strong), color-mix(in srgb, var(--strong) 70%, var(--accent))); -webkit-background-clip:text; background-clip:text; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:linear-gradient(90deg, var(--accent), var(--accent2)) !important; height:3px !important; border-radius:3px; }
[data-testid="stTabs"] button[aria-selected="true"] p { color:var(--strong) !important; font-weight:600; }
.kq-act { display:grid; grid-template-columns: 1.25fr 1fr 1fr; gap:10px; margin:6px 0 12px; }
.kq-act .col { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:11px 13px; box-shadow:var(--shadow); }
.kq-act .hd { display:flex; align-items:center; gap:8px; font-weight:700; color:var(--strong); font-size:14px; margin-bottom:8px; }
.kq-act .hd .dot { width:10px; height:10px; border-radius:50%; }
.kq-act .hd .n { margin-left:auto; font-size:12px; color:var(--muted); font-weight:500; }
.kq-act .empty { color:var(--muted); font-size:12.5px; }
.kq-act .title { grid-column:1 / -1; display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 12px; font-size:15px; font-weight:700; color:var(--strong); }
.kq-act .title small { font-size:12px; color:var(--muted); font-weight:400; }
.kq-chips { display:flex; flex-wrap:wrap; gap:6px; }
a.kq-schip { display:inline-flex; align-items:baseline; gap:5px; padding:4px 9px; border-radius:10px; border:1px solid var(--line);
  background:var(--panel2); color:var(--text) !important; text-decoration:none !important; font-size:13px; }
a.kq-schip:hover { border-color:var(--accent); }
a.kq-schip b { font-family:'Inter', sans-serif; font-variant-numeric: tabular-nums; color:var(--accent); font-weight:600; }
a.kq-schip small { color:var(--muted); font-size:11px; }
@media (max-width: 760px) { .kq-act { grid-template-columns: 1fr; gap:8px; } .kq-act .col { padding:10px 11px; } a.kq-schip { font-size:12.5px; } }

</style>
"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def ok(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def price(x, etf: bool = False) -> str:
    """股價顯示：先對齊到最接近的有效升降單位（避免出現 31.83、103.8 這類不能下單的價格），再依級距決定小數位。
    股票：<50 兩位（0.01／0.05）、50–500 一位（0.1／0.5）、≥500 整數（1／5）；ETF 一律兩位（0.01／0.05）。"""
    if not ok(x):
        return "–"
    from .ta import round_tick
    x = float(x)
    if x > 0:
        x = round_tick(x, "near", etf)
    d = 2 if etf or x < 50 else (1 if x < 500 else 0)
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


def cards(items, raw: bool = False) -> str:
    """items: [(label, value_html, sub_html), ...]；raw=True 時 label 可含 HTML（儀表板燈號）。"""
    out = ["<div class='kq-cards'>"]
    for lab, val, sub in items:
        out.append(f"<div class='kq-card'><div class='l'>{lab if raw else esc(lab)}</div><div class='v'>{val}</div>"
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


_CHART_IDS = itertools.count(1)


def base_chart(xf, vals, base, *, h: int = 64, fluid: bool = True, w: int = 300, sym: bool = True,
               min_amp: float = 0.004, ticks=(), labels: dict | None = None, dot: bool = True) -> str:
    """v2.1.1 有「參考線」的迷你走勢圖（SVG＋HTML 標籤）：
    參考線（分時＝平盤／昨收；多日＝期初價）以虛線標出，線在上方為紅、下方為綠，並以漸層填色到參考線。
    sym=True：以參考線為中心上下對稱（分時，一眼看出漲跌幅）；False：依高低點自動縮放（多日）。
    xf：每一點的水平位置 0～1（分時用交易時段位置，盤中未走完的部分留白）。"""
    v = np.asarray(vals if vals is not None else [], float)
    xf = np.asarray(xf if xf is not None else [], float)
    if len(xf) != len(v):
        xf = np.linspace(0, 1, len(v))
    m = np.isfinite(v) & np.isfinite(xf)
    v, xf = v[m], xf[m]
    if len(v) < 2 or not ok(base) or float(base) <= 0:
        return ""
    base = float(base)
    W = 300 if fluid else int(w)
    if sym:
        amp = max(float(np.abs(v - base).max()), base * min_amp) * 1.12
        lo, hi = base - amp, base + amp
    else:
        lo, hi = min(float(v.min()), base), max(float(v.max()), base)
        sp = (hi - lo) or base * 0.01
        lo, hi = lo - sp * 0.08, hi + sp * 0.08
    pt = pb = (11.0 if labels else 3.0)            # 有標籤時上下留白給文字，線不會壓到「高／低」

    def Y(x):
        return pt + (hi - x) / (hi - lo) * (h - pt - pb)
    xs, ys, by = 1 + xf * (W - 2), Y(v), float(Y(base))
    line = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    area = line + f" L{xs[-1]:.1f},{by:.1f} L{xs[0]:.1f},{by:.1f} Z"
    i = next(_CHART_IDS)
    nse = "vector-effect='non-scaling-stroke'"
    grid = "".join(f"<line x1='{1 + f * (W - 2):.1f}' x2='{1 + f * (W - 2):.1f}' y1='0' y2='{h}' style='stroke:var(--line)' "
                   f"stroke-width='1' {nse}/>" for f in ticks)
    svg = (f"<svg width='{'100%' if fluid else W}' height='{h}' viewBox='0 0 {W} {h}' preserveAspectRatio='none'>"
           f"<defs><clipPath id='kcu{i}'><rect x='0' y='0' width='{W}' height='{by:.2f}'/></clipPath>"
           f"<clipPath id='kcd{i}'><rect x='0' y='{by:.2f}' width='{W}' height='{max(h - by, 0):.2f}'/></clipPath>"
           f"<linearGradient id='kgu{i}' gradientUnits='userSpaceOnUse' x1='0' y1='{pt}' x2='0' y2='{by:.2f}'>"
           f"<stop offset='0' stop-color='{UP}' stop-opacity='.34'/><stop offset='1' stop-color='{UP}' stop-opacity='.04'/></linearGradient>"
           f"<linearGradient id='kgd{i}' gradientUnits='userSpaceOnUse' x1='0' y1='{by:.2f}' x2='0' y2='{h - pb}'>"
           f"<stop offset='0' stop-color='{DOWN}' stop-opacity='.04'/><stop offset='1' stop-color='{DOWN}' stop-opacity='.34'/></linearGradient>"
           f"</defs>{grid}"
           f"<path d='{area}' fill='url(#kgu{i})' clip-path='url(#kcu{i})'/>"
           f"<path d='{area}' fill='url(#kgd{i})' clip-path='url(#kcd{i})'/>"
           f"<line x1='0' x2='{W}' y1='{by:.2f}' y2='{by:.2f}' style='stroke:var(--muted)' stroke-width='1' "
           f"stroke-dasharray='4 3' opacity='.8' {nse}/>"
           f"<path d='{line}' fill='none' stroke='{UP}' stroke-width='1.7' stroke-linejoin='round' clip-path='url(#kcu{i})' {nse}/>"
           f"<path d='{line}' fill='none' stroke='{DOWN}' stroke-width='1.7' stroke-linejoin='round' clip-path='url(#kcd{i})' {nse}/>"
           "</svg>")
    c = UP if v[-1] > base else (DOWN if v[-1] < base else FLAT)
    out = [f"<div class='kq-ch' style='height:{h}px;width:{'100%' if fluid else f'{W}px'}'>", svg]
    if dot:
        out.append(f"<i class='dt' style='left:{xs[-1] / W * 100:.2f}%;top:{ys[-1] / h * 100:.2f}%;background:{c}'></i>")
    for k, txt in (labels or {}).items():
        if not txt:
            continue
        if k == "base":
            out.append(f"<span class='lb bs' style='top:{by / h * 100:.1f}%'>{txt}</span>")
        else:
            out.append(f"<span class='lb {k}'>{txt}</span>")
    out.append("</div>")
    return "".join(out)


def session_ticks(d) -> list:
    """分時圖的整點格線位置（0～1）。"""
    s, e = d["start"], d["end"]
    span = (e - s).total_seconds()
    if span <= 0 or span > 26 * 3600:
        return []
    step = 3600 if span <= 8 * 3600 else 4 * 3600
    t = s.ceil("h")
    out = []
    while t < e:
        f = (t - s).total_seconds() / span
        if 0.02 < f < 0.98:
            out.append(f)
        t += pd.Timedelta(seconds=step)
    return out


def intraday_mini(d, *, h: int = 84, fluid: bool = True, w: int = 300, labels: bool = True, digits=None,
                  title: str = "") -> str:
    """分時迷你圖：平盤虛線、紅漲綠跌，盤中未走完的時段留白。d：kosto.intraday 的資料。"""
    if not d or len(d.get("v", [])) < 2:
        return ""
    from . import intraday as ID
    x, y = ID.thin(ID.frac(d), np.asarray(d["v"], float), 180 if fluid else 90)
    fmt = (lambda z: num(z, digits)) if digits is not None else price
    lab = None
    if labels:
        lab = {"tl": title, "tr": f"高 <b>{fmt(d['high'])}</b>", "br": f"低 <b>{fmt(d['low'])}</b>",
               "bl": f"<i class='bsw'></i>平盤 {fmt(d['prev'])}"}
    return base_chart(x, y, d["prev"], h=h, fluid=fluid, w=w, sym=True, min_amp=0.003, ticks=session_ticks(d), labels=lab)


def trend_mini(vals, *, h: int = 84, fluid: bool = True, w: int = 300, labels: bool = True, period: str = "",
               digits=None) -> str:
    """多日迷你圖：以期初價為參考虛線（高於期初＝紅、低於＝綠），依高低點自動縮放。"""
    v = np.asarray([x for x in (vals if vals is not None else []) if ok(x)], float)
    if len(v) < 3:
        return ""
    fmt = (lambda z: num(z, digits)) if digits is not None else price
    r = v[-1] / v[0] - 1
    lab = None
    if labels:
        lab = {"tl": f"{period} <b style='color:{color_of(r)}'>{r * 100:+.1f}%</b>", "tr": f"高 <b>{fmt(v.max())}</b>",
               "br": f"低 <b>{fmt(v.min())}</b>", "bl": f"<i class='bsw'></i>期初 {fmt(v[0])}"}
    return base_chart(np.linspace(0, 1, len(v)), v, v[0], h=h, fluid=fluid, w=w, sym=False, labels=lab)


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


def schip(code: str, name: str, extra: str = "", href: str | None = None) -> str:
    """可點的股票籤（代號＋名稱＋小字），預設連到個股診斷。"""
    href = href or f"./stock?s={esc(code)}"
    return (f"<a class='kq-schip' href='{href}' target='_self'><b>{esc(code)}</b>{esc(name)}"
            + (f"<small>{esc(extra)}</small>" if extra else "") + "</a>")


def action_board(title: str, note: str, cols) -> str:
    """今日行動看板：cols＝[(標題, 顏色, [股票籤 html…], 空白時文字)]。"""
    out = [f"<div class='kq-act'><div class='title'>{esc(title)}<small>{note}</small></div>"]
    for head, color, chips, empty in cols:
        body = f"<div class='kq-chips'>{''.join(chips)}</div>" if chips else f"<div class='empty'>{esc(empty)}</div>"
        out.append(f"<div class='col' style='border-top:3px solid {color}'><div class='hd'><span class='dot' style='background:{color}'></span>"
                   f"{esc(head)}<span class='n'>{len(chips)} 檔</span></div>{body}</div>")
    out.append("</div>")
    return "".join(out)
