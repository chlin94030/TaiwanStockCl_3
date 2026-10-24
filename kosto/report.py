"""Full downloadable test report: report.html + report.xlsx + CSVs + config_used.yaml → ZIP."""
from __future__ import annotations

import html
import io
import json
import zipfile

import numpy as np
import pandas as pd

from . import APP_NAME, __version__
from .config import STAGE_ZH, dump_yaml
from .svg import bar_chart, heat_table, line_chart, stage_strip
from .universe import RULE_ZH

CSS = """
:root{--bg:#ffffff;--fg:#1f2937;--mut:#6b7280;--line:#e5e7eb;--card:#f9fafb;--acc:#2563eb;--bad:#dc2626;--good:#16a34a}
@media (prefers-color-scheme: dark){:root{--bg:#111827;--fg:#e5e7eb;--mut:#9ca3af;--line:#374151;--card:#1f2937;--acc:#60a5fa}}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif;
max-width:1000px;margin:0 auto;padding:16px;line-height:1.55}
h1{font-size:1.5rem;margin:.2rem 0}h2{font-size:1.2rem;border-bottom:2px solid var(--acc);padding-bottom:4px;margin-top:2rem}
h3{font-size:1.05rem;margin-top:1.4rem}.mut{color:var(--mut);font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin:.5rem 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}th{background:var(--card)}
td.l,th.l{text-align:left;white-space:normal}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:.6rem 0}
.banner{background:#fef3c7;color:#92400e;border:1px solid #f59e0b;border-radius:8px;padding:10px;margin:10px 0;font-weight:600}
.pos{color:var(--good)}.neg{color:var(--bad)}svg.chart{width:100%;height:auto;margin:.4rem 0}
.ct{font-size:13px;font-weight:600;fill:var(--fg)}.ax{font-size:10px;fill:var(--mut)}.lg{font-size:11px;fill:var(--fg)}
.grid{stroke:var(--line);stroke-width:1}.legend{font-size:.75rem;color:var(--mut)}.sw{display:inline-block;width:10px;height:10px;margin:0 3px 0 8px}
.heat td{color:#111}ul{padding-left:1.2rem}.kpi{display:flex;flex-wrap:wrap;gap:8px}.kpi div{flex:1 1 140px;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:8px}.kpi b{display:block;font-size:1.2rem}
"""


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x*100:+.{d}f}%"


def num(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def cls(x):
    return "" if x is None or not np.isfinite(x) else ("pos" if x > 0 else "neg")


def table(df: pd.DataFrame, fmts: dict | None = None, left=()) -> str:
    fmts = fmts or {}
    out = ["<table><tr>" + "".join(f"<th class='{'l' if c in left else ''}'>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmts.get(c)
            if f == "pct":
                s, k = pct(v), cls(v)
            elif f == "pct0":
                s, k = ("–" if v is None or not np.isfinite(v) else f"{v*100:.1f}%"), ""
            elif f and isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                s, k = f.format(v), ""
            else:
                s, k = ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)), ""
            cells.append(f"<td class='{k} {'l' if c in left else ''}'>{html.escape(s)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) + "</table>"


KOSTO_MAP = [
    ("雞蛋理論 A1/B3 買、A2 等待、A3/B1 賣、B2 持現金（個股）", "個股雞蛋階段閘門＋k_egg 因子",
     ["移除個股雞蛋階段閘門", "移除 k_egg"]),
    ("雞蛋理論（大盤）：過熱減碼、恐慌加碼", "市場階段持股水位（none/soft/strict 三組比較）", ["移除市場階段持股水位"]),
    ("固執的投資者 vs 猶豫的投資者（股票是否已付清）", "k_hands 籌碼轉移、k_paid 融資占股本、KOSTO 母池",
     ["移除 k_hands", "移除 k_paid"]),
    ("量價四象限（量小價漲有利、量增價跌近底…）", "k_vp 量價因子", ["移除 k_vp"]),
    ("利空不跌＝近底、利多不漲＝近頂", "k_resilience", ["移除 k_resilience"]),
    ("M 頭 / W 底", "k_mw 因子 + M 頭禁止買進 + B3 需 W 底/止跌確認", ["移除 k_mw", "移除M頭閘門"]),
    ("戒1 不跟著建議跑、媒體過度樂觀", "k_calm 避開爆量急漲與融資暴增", ["移除 k_calm"]),
    ("律3 耐心、戒6 不對風吹草動反應", "遲滯續抱（名次仍在前 K×遲滯倍數 就不換股）", ["移除耐心續抱(遲滯)"]),
    ("戒3 不要想把賠掉的再賺回來", "持有期停損（中線12%、長線18%，可調）", ["移除停損(戒3)"]),
    ("律6 新局面就賣出", "持股轉為 A3/B1/B2 或出現 M 頭 → 下次換股時不續抱", ["移除個股雞蛋階段/M頭閘門"]),
    ("整體：科斯托蘭尼因子族群", "K 族群最低權重 λ", ["移除科斯托蘭尼因子(K族群)"]),
    ("律2 足夠資金免受壓力、律9 考慮所有風險", "不融資、等權10檔、流動性容量限制（買進金額≤20日均量2%）", []),
    ("戒8 不要只想獲利就賣掉", "不設停利；只要階段與分數仍佳就續抱", []),
]


ZH = {"variant": "模型", "dev_cagr": "開發期CAGR", "dev_sharpe": "開發期Sharpe", "dev_mdd": "開發期MDD",
      "dev_excess_cagr": "開發期超額", "hold_cagr": "盲測CAGR", "hold_mdd": "盲測MDD", "turnover": "周轉率",
      "score": "綜合分數", "case": "情境", "factor": "因子", "family": "族群", "coverage": "覆蓋率", "ic_dev": "開發期IC",
      "t_dev": "t值", "ic_holdout": "盲測IC", "ic_all": "全期IC", "weight": "權重", "stage": "市場階段", "n": "批數",
      "avg_ret": "平均報酬", "avg_bench": "0050同期", "avg_excess": "平均超額", "avg_exposure": "平均水位",
      "from": "生效日", "n_train": "訓練期數", "alpha": "資料權重α"}


def hints(H: dict) -> list[str]:
    out = []
    for h, r in H.items():
        lab = r["label"]
        ev = r["eval"]
        if r.get("data_beats_kostolany"):
            out.append(f"{lab}：純數據模型(DATA)在開發期綜合分數高於所有科斯托蘭尼版本 → 可試 λ=0.25 或檢視 K 因子 IC。")
        ab = r["ablation"]
        for _, row in ab.iloc[1:].sort_values("貢獻(開發期CAGR)").head(3).iterrows():
            if np.isfinite(row["貢獻(開發期CAGR)"]) and row["貢獻(開發期CAGR)"] < -0.01:
                out.append(f"{lab}：「{row['case']}」後開發期 CAGR 反而提高 {(-row['貢獻(開發期CAGR)'])*100:.1f}% → 此規則在本期間拖累績效，可考慮關閉或放寬。")
        dex, hex_ = ev.get("dev_excess_cagr"), ev.get("hold_excess_cagr")
        if dex is not None and hex_ is not None and np.isfinite(dex) and np.isfinite(hex_) and hex_ < dex - 0.10:
            out.append(f"{lab}：盲測期超額報酬比開發期低 {(dex-hex_)*100:.0f} 個百分點 → 可能過度配適或市場風格改變，建議提高 n0_shrink 或縮小候選。")
        rb = r["robust"]
        if rb.get("dsr") is not None and np.isfinite(rb["dsr"]) and rb["dsr"] < 0.5:
            out.append(f"{lab}：考慮 {rb['n_trials']} 次測試後的 Deflated Sharpe = {rb['dsr']:.2f}（<0.5）→ 優勢在統計上尚不顯著，請以小資金或紙上交易驗證。")
        ft = r["factor_table"]
        bad = ft[(ft["weight"].abs() > 0.02) & (ft["t_dev"] * np.sign(ft["weight"]) < -1.5)]
        for _, row in bad.iterrows():
            out.append(f"{lab}：因子「{row['中文']}」有權重但開發期 IC 方向相反(t={row['t_dev']:.1f}) → 檢查資料或移除。")
        st = r["stage_table"]
        if len(st) and {"A1", "B3", "A3", "B1"} & set(st["stage"]):
            buy = st[st["stage"].isin(["A1", "B3"])]["avg_bench"].mean()
            sell = st[st["stage"].isin(["A3", "B1"])]["avg_bench"].mean()
            if np.isfinite(buy) and np.isfinite(sell) and buy < sell:
                out.append(f"{lab}：市場判為 A1/B3(買點) 後的大盤報酬 {buy*100:.1f}% 低於 A3/B1(賣點) 後的 {sell*100:.1f}% → 市場雞蛋階段判斷在本資料不具預測力，建議水位用 none/soft。")
        if h == "short" and r["turnover"] > 0.5:
            out.append(f"{lab}：每次換股周轉率 {r['turnover']:.0%}，交易成本吃掉不少報酬 → 可把 hysteresis 提高到 2.0~2.5。")
    return out or ["本次未偵測到需要優先調整的項目。"]


def build_html(res: dict, cfg: dict, comparison: tuple | None = None) -> str:
    H = res["horizons"]
    meta, proto = res["meta"], res["protocol"]
    p = [f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Kosto Lab 回測報告</title><style>{CSS}</style></head><body>",
         f"<h1>{APP_NAME}｜完整測試報告</h1>",
         f"<div class='mut'>版本 {__version__}｜建置 {meta.get('built')}｜訊號 {res['protocol']['signals']} 期（每 {cfg['research']['signal_step']} 交易日）"
         f"｜樣本外起點 {proto['first_oos']}｜盲測起點 {proto['holdout_start']}｜最新資料 {proto['last_date']}</div>"]
    if meta.get("synthetic"):
        p.append("<div class='banner'>⚠ 示範模式：本報告使用「合成資料」驗證程式流程，數字不代表真實台股，請勿據以投資。"
                 "連上 GitHub Actions 的真實資料後重新產生報告。</div>")
    if proto.get("quick"):
        p.append("<div class='banner'>快速模式：只測試部分母池（ADV/KOSTO/INST2Y × 150/300）。完整測試請用 full 模式。</div>")
    # ---------------- summary
    p.append("<h2>1. 總覽：哪個模型、哪個母池</h2>")
    rows = []
    for h, r in H.items():
        ev, rb = r["eval"], r["robust"]
        rows.append({"期間": r["label"], "精選母池": f"{r['chosen_pool']['rule']}{r['chosen_pool']['size']}（{RULE_ZH[r['chosen_pool']['rule']]}）",
                     "選定模型": r["chosen"]["variant"], "開發期CAGR": ev.get("dev_cagr"), "0050同期": ev.get("dev_bench_cagr"),
                     "開發期Sharpe": ev.get("dev_sharpe"), "開發期MDD": ev.get("dev_mdd"), "盲測CAGR": ev.get("hold_cagr"),
                     "盲測0050": ev.get("hold_bench_cagr"), "盲測MDD": ev.get("hold_mdd"), "PSR": rb.get("psr"), "DSR": rb.get("dsr"),
                     "超額>0機率": rb.get("p_positive")})
    p.append(table(pd.DataFrame(rows), {"開發期CAGR": "pct", "0050同期": "pct", "開發期MDD": "pct", "盲測CAGR": "pct",
                                        "盲測0050": "pct", "盲測MDD": "pct", "開發期Sharpe": "{:.2f}", "PSR": "{:.2f}",
                                        "DSR": "{:.2f}", "超額>0機率": "pct0"}, left=("期間", "精選母池", "選定模型")))
    p.append("<p class='mut'>開發期＝樣本外起點到盲測起點（用來挑母池與模型）；盲測＝最後 "
             f"{cfg['research']['holdout_months']} 個月，挑選時完全沒看。PSR＝夏普比大於 0 的機率；DSR＝扣除多次測試運氣後的機率；"
             "超額>0機率＝區塊自助法估計年化超額報酬為正的機率。</p>")
    # ---------------- today's picks
    first = next(iter(H.values()))
    lt0 = first["latest"]
    p.append(f"<h2>2. 今日推薦（{lt0['date']} 收盤後）</h2>")
    p.append(f"<div class='card'>市場雞蛋階段：<b>{STAGE_ZH.get(lt0['market_stage'], lt0['market_stage'])}</b></div>")
    for h, r in H.items():
        lt = r["latest"]
        p.append(f"<h3>{r['label']}｜母池 {lt['pool']}｜建議持股水位 {lt['exposure']:.0%}</h3>")
        if lt["picks"]:
            df = pd.DataFrame([{"代號": x["ticker"], "名稱": x["name"], "產業": x["sector"], "收盤": x["close"],
                                "分數": x["score"], "個股階段": x["stage"], "續抱": "✓" if x["kept"] else "",
                                "主要理由": "；".join(x["reasons"]), "停損價": x["stop_price"]} for x in lt["picks"]])
            p.append(table(df, {"收盤": "{:.2f}", "分數": "{:.3f}", "停損價": "{:.2f}"}, left=("名稱", "產業", "主要理由")))
        if lt["blocked"]:
            p.append("<p class='mut'>高分但被科斯托蘭尼規則排除：" + "、".join(f"{b['ticker']}{b['name']}（{b['why']}）" for b in lt["blocked"][:8]) + "</p>")
    # ---------------- market regime
    mk = res["market"]
    p.append("<h2>3. 市場雞蛋階段（科斯托蘭尼）</h2>")
    p.append(line_chart({"市場指數": mk["index"]}, "大盤指數與階段", log=True))
    p.append(stage_strip(mk["stage"]))
    p.append("<p class='mut'>階段由「指數位置與趨勢、成交量(20/250日)、融資餘額(持有人數代理)、市場廣度」判定；持有人數與新聞反應無法直接觀察，以上為代理變數。</p>")
    # ---------------- per horizon
    for h, r in H.items():
        p.append(f"<h2>4-{r['label']}：回測細節</h2>")
        nav = r["nav"]
        p.append(line_chart({c: nav[c] for c in nav.columns}, f"{r['label']} 淨值（樣本外，含交易成本）", log=True))
        yt = r["yearly"]
        p.append(bar_chart([str(i) for i in yt.index], {c: yt[c].tolist() for c in yt.columns if c != "超額"}, "年度報酬"))
        ev = r["eval"]
        kp = [("開發期 CAGR", pct(ev.get("dev_cagr"))), ("開發期超額", pct(ev.get("dev_excess_cagr"))),
              ("盲測 CAGR", pct(ev.get("hold_cagr"))), ("盲測超額", pct(ev.get("hold_excess_cagr"))),
              ("每次周轉率", f"{r['turnover']:.0%}"), ("年交易成本(占資金)", f"{r['cost_drag']:.1%}"),
              ("單批獲利率", f"{ev.get('tr_hit_rate', float('nan')):.0%}"),
              ("單批勝0050", f"{ev.get('tr_beat_bench_rate', float('nan')):.0%}")]
        p.append("<div class='kpi'>" + "".join(f"<div>{k}<b>{v}</b></div>" for k, v in kp) + "</div>")
        pt = r["pool_table"]
        if len(pt) <= 1:
            p.append(f"<p class='mut'>單一設定報告：母池固定為 {r['chosen_pool']['rule']}{r['chosen_pool']['size']}，未做母池比較。</p>")
            pt = pt.assign(score=pt["score"].fillna(1.0))
        hm = pt.pivot_table(index="rule", columns="size", values="score")
        hm.index = [f"{i} {RULE_ZH.get(i, '')}" for i in hm.index]
        p.append("<h3>母池測試（開發期綜合分數，越綠越好）</h3>" + heat_table(hm, "{:.2f}"))
        hc = pt.pivot_table(index="rule", columns="size", values="dev_cagr")
        hc.index = [f"{i} {RULE_ZH.get(i, '')}" for i in hc.index]
        p.append("<p class='mut'>同一模型在各母池的開發期 CAGR：</p>" + heat_table(hc))
        vt = r["variant_table"][["variant", "dev_cagr", "dev_sharpe", "dev_mdd", "dev_excess_cagr", "hold_cagr", "hold_mdd", "turnover", "score"]]
        p.append("<h3>模型比較（同一母池）</h3>" + table(vt.rename(columns=ZH), {ZH[k]: v for k, v in {
            "dev_cagr": "pct", "dev_mdd": "pct", "dev_excess_cagr": "pct", "hold_cagr": "pct", "hold_mdd": "pct",
            "dev_sharpe": "{:.2f}", "turnover": "pct0", "score": "{:.2f}"}.items()}, left=("模型",)))
        p.append("<p class='mut'>挑選規則：只在科斯托蘭尼版本中挑開發期綜合分數最高者；DATA/PRIOR 為對照組。</p>")
        p.append("<h3>科斯托蘭尼規則消融（關掉一項看績效變化）</h3>" + table(r["ablation"].rename(columns=ZH), {ZH.get(k, k): v for k, v in {
            "dev_cagr": "pct", "dev_mdd": "pct", "hold_cagr": "pct", "hold_mdd": "pct", "dev_sharpe": "{:.2f}",
            "turnover": "pct0", "貢獻(開發期CAGR)": "pct"}.items()}, left=("情境",)))
        fw = r["family_final"]
        p.append("<h3>因子與權重（最新一次估計）</h3><p>族群權重：" + "、".join(f"{k}={v:.0%}" for k, v in fw.items())
                 + "（T技術、C籌碼、F基本、K科斯托蘭尼）</p>")
        ft = r["factor_table"]
        ft = ft[ft["horizon_ok"]][["factor", "中文", "family", "coverage", "ic_dev", "t_dev", "ic_holdout", "ic_all", "weight"]]
        p.append(table(ft.rename(columns=ZH), {ZH[k]: v for k, v in {"coverage": "pct0", "ic_dev": "{:.3f}", "t_dev": "{:.1f}",
                 "ic_holdout": "{:.3f}", "ic_all": "{:.3f}", "weight": "{:+.3f}"}.items()}, left=("中文",)))
        p.append("<p class='mut'>IC＝因子排名與未來報酬排名的相關係數（台股常見 0.02~0.06 已屬有用）；t 值絕對值 >2 較可信。"
                 "權重 0 代表訓練資料與先驗方向相反或不顯著，系統不會硬把方向翻轉。</p>")
        st = r["stage_table"]
        if len(st):
            p.append("<h3>依市場階段分組的單批績效</h3>" + table(st.rename(columns=ZH), {ZH[k]: v for k, v in {
                "avg_ret": "pct", "avg_bench": "pct", "avg_excess": "pct", "avg_exposure": "pct0"}.items()}))
            p.append("<p class='mut'>檢查雞蛋階段是否有預測力：A1/B3（買點）之後的 0050 報酬應高於 A3/B1（賣點）之後。</p>")
        hist = pd.DataFrame(r["fit_history"]).drop(columns=["weights"]).rename(columns=ZH)
        p.append("<h3>權重重估歷程（走勢前進）</h3>" + table(hist, {c: "{:.2f}" for c in hist.columns if c.startswith("fam_") or c == "資料權重α"}))
        p.append("<p class='mut'>α 越接近 1 越相信資料、越接近 0 越貼近先驗權重；長線獨立樣本少，α 自動較低。</p>")
    # ---------------- comparison
    if comparison is not None and comparison[1] is not None and len(comparison[1]):
        p.append("<h2>5. 新舊模型比較（同一引擎、同一進出場規則）</h2>")
        s = comparison[1]
        p.append(table(s, {c: "pct" for c in s.columns if "報酬" in c or "平均" in c or "新減舊" in c} |
                       {c: "pct0" for c in s.columns if "勝" in c or "涵蓋" in c} | {"差異t值": "{:.2f}"}))
        p.append("<p class='mut'>新模型取「舊模型訊號日之前最近一次」的推薦，以相同進場日計算；t 值 > 2 才算顯著差異。</p>")
    # ---------------- Kostolany map
    p.append("<h2>6. 《一個投機者的告白》觀念 → 程式規則 → 實測貢獻</h2>")
    rows = []
    for idea, impl, cases in KOSTO_MAP:
        contrib = []
        for h, r in H.items():
            ab = r["ablation"].set_index("case")
            vals = []
            for c in cases:
                hit = [k for k in ab.index if k == c or (c.startswith("移除 k_") and k.startswith(c + "（"))]
                vals += [ab.loc[k, "貢獻(開發期CAGR)"] for k in hit]
            if vals:
                contrib.append(f"{r['label'][:2]} {pct(np.nansum(vals))}")
        rows.append({"書中觀念": idea, "程式實作": impl, "關掉後CAGR少了(正=有幫助)": "；".join(contrib) or "（不可單獨關閉）"})
    p.append(table(pd.DataFrame(rows), left=("書中觀念", "程式實作", "關掉後CAGR少了(正=有幫助)")))
    # ---------------- hints
    p.append("<h2>7. 自動調整建議</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in hints(H)) + "</ul>")
    p.append("<p class='mut'>調整方式：修改 config/default.yaml（或 Streamlit「快速調整」頁），重新執行即可；每次報告都附 config_used.yaml 方便回溯。</p>")
    # ---------------- method
    p.append("<h2>8. 方法與限制</h2><ul>"
             "<li>資料：證交所/櫃買中心/公開資訊觀測站逐日全市場（或 FinMind）；含已下市股票，無存活者偏誤（FinMind 逐檔模式除外）。</li>"
             "<li>報酬：以官方『除權息參考價』還原，股利再投入；訊號日收盤決策、次日開盤成交、持有 H 日於開盤出場；開盤漲停買不到視為現金。</li>"
             f"<li>成本：手續費 {cfg['costs']['fee']*100:.4f}%×{cfg['costs']['fee_discount']} 折、證交稅 {cfg['costs']['tax']*100:.1f}%、滑價單邊 {cfg['costs']['slippage']*100:.1f}%。</li>"
             "<li>權重：每 6 個月以過去已到期資料估計 IC-IR 權重（不翻轉先驗方向、向先驗收縮、科斯托蘭尼族群下限 λ）。</li>"
             "<li>月營收以法定申報期限（次月10日）後才可使用；本益比/淨值比為交易所每日公布值。</li>"
             "<li>雞蛋理論的「持有人數」以融資餘額代理、「媒體氣氛」以爆量急漲與融資暴增代理；新聞反應與貨幣供給未觀察。</li>"
             "<li>歷史回測不保證未來績效；本工具僅供研究，不構成投資建議，也不會自動下單。</li></ul>")
    p.append("<h2>附錄：本次設定</h2><pre style='font-size:.75rem;overflow-x:auto'>" + html.escape(dump_yaml(cfg)) + "</pre>")
    p.append("</body></html>")
    return "".join(p)


def _xls_name(s: str) -> str:
    for ch in "[]:*?/\\":
        s = s.replace(ch, "_")
    return s[:31]


def build_excel(res: dict, comparison=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summ = []
        for h, r in res["horizons"].items():
            row = {"期間": r["label"], "母池": f"{r['chosen_pool']['rule']}{r['chosen_pool']['size']}", "模型": r["chosen"]["variant"]}
            row.update({k: v for k, v in r["eval"].items() if isinstance(v, (int, float, np.floating, str, bool))})
            row.update({f"robust_{k}": v for k, v in r["robust"].items()})
            summ.append(row)
        pd.DataFrame(summ).to_excel(xw, sheet_name="總覽", index=False)
        for h, r in res["horizons"].items():
            tag = {"short": "短", "mid": "中", "long": "長"}[h]
            lt = r["latest"]
            pd.DataFrame([dict(x, reasons="；".join(x["reasons"]), risk="；".join(x["risk"])) for x in lt["picks"]]).to_excel(
                xw, sheet_name=_xls_name(f"{tag}_今日推薦"), index=False)
            pd.DataFrame([dict(x, reasons="；".join(x["reasons"]), risk="；".join(x["risk"])) for x in lt["alternates"]]).to_excel(
                xw, sheet_name=_xls_name(f"{tag}_備選"), index=False)
            r["pool_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_母池測試"), index=False)
            r["variant_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_模型比較"), index=False)
            r["ablation"].to_excel(xw, sheet_name=_xls_name(f"{tag}_消融"), index=False)
            r["factor_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_因子IC權重"), index=False)
            wh = pd.DataFrame([{"from": x["from"], **x["weights"]} for x in r["fit_history"]])
            wh.to_excel(xw, sheet_name=_xls_name(f"{tag}_權重歷程"), index=False)
            r["stage_table"].to_excel(xw, sheet_name=_xls_name(f"{tag}_階段績效"), index=False)
            r["yearly"].to_excel(xw, sheet_name=_xls_name(f"{tag}_年度"))
            tr = r["tranches"].copy()
            tr["picks"] = tr["picks"].map(lambda x: "|".join(x))
            tr.to_excel(xw, sheet_name=_xls_name(f"{tag}_每期推薦紀錄"), index=False)
            r["nav"].to_excel(xw, sheet_name=_xls_name(f"{tag}_淨值"))
        mk = res["market"].copy()
        mk.to_excel(xw, sheet_name="市場階段")
        if comparison is not None and comparison[0] is not None and len(comparison[0]):
            comparison[1].to_excel(xw, sheet_name="新舊比較_摘要", index=False)
            comparison[0].to_excel(xw, sheet_name="新舊比較_逐期", index=False)
    return buf.getvalue()


def summary_json(res: dict) -> dict:
    out = {"meta": res["meta"], "protocol": res["protocol"], "horizons": {}}
    for h, r in res["horizons"].items():
        out["horizons"][h] = {
            "label": r["label"], "chosen_pool": r["chosen_pool"], "chosen": r["chosen"],
            "eval": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r["eval"].items()
                     if isinstance(v, (int, float, np.floating, str, bool))},
            "robust": r["robust"], "family_final": r["family_final"], "latest": r["latest"],
        }
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def build_zip(res: dict, cfg: dict, comparison=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.html", build_html(res, cfg, comparison))
        z.writestr("report.xlsx", build_excel(res, comparison))
        z.writestr("config_used.yaml", dump_yaml(cfg))
        z.writestr("results_summary.json", json.dumps(summary_json(res), ensure_ascii=False, indent=1))
        for h, r in res["horizons"].items():
            tr = r["tranches"][["date", "stage", "exposure", "ret", "bench", "picks"]].copy()
            tr.insert(1, "horizon", r["days"])
            tr["selected"] = tr.pop("picks").map(lambda x: "|".join(x))
            z.writestr(f"picks_history_{h}.csv", tr.to_csv(index=False).encode("utf-8-sig"))
            z.writestr(f"nav_{h}.csv", r["nav"].to_csv().encode("utf-8-sig"))
        z.writestr("market_stage.csv", res["market"].to_csv().encode("utf-8-sig"))
        z.writestr("README.txt", (
            "Kosto Lab 測試報告\n"
            "report.html          完整報告（手機/電腦瀏覽器開啟）\n"
            "report.xlsx          所有表格（母池測試、模型比較、因子IC、權重歷程、每期推薦…）\n"
            "config_used.yaml     本次設定；修改後貼回 config/default.yaml 即可重跑\n"
            "picks_history_*.csv  每期推薦清單（可匯入其他系統或 Kosto Lab 的「新舊比較」）\n"
            "nav_*.csv            每日淨值；market_stage.csv 市場雞蛋階段\n").encode("utf-8-sig"))
    return buf.getvalue()
