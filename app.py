"""Kosto Lab — Streamlit front end.

Heavy work (data download, full pool/model research) runs in GitHub Actions; this app loads the
precomputed research bundle and re-runs single configurations in seconds.
"""
from __future__ import annotations

import copy
import io
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from kosto import APP_NAME, __version__
from kosto.bundle import Bundle, build_bundle
from kosto.compare import Daily, head_to_head, parse_external
from kosto.config import HORIZON_KEYS, STAGE_ZH, dump_yaml, load_config
from kosto.remote import RESEARCH_TAG, dispatch, fetch_asset, latest_runs, release_info
from kosto.report import build_zip
from kosto.research import Research, Variant
from kosto.setup_page import (detect_repo, local_progress, local_running, on_streamlit_cloud, read_status,
                              render_progress, start_local_build, tail_log)
from kosto.universe import RULE_ZH, RULES

st.set_page_config(page_title="Kosto Lab 台股選股", page_icon="🥚", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;padding-left:1rem;padding-right:1rem}
div[data-testid="stMetricValue"]{font-size:1.25rem}
</style>""", unsafe_allow_html=True)

HZ = {"short": "短線 5日", "mid": "中線 20日", "long": "長線 60日"}
MODES = {"KOSTO 最佳化(λ下限)": "opt", "純數據(無科斯托蘭尼)": "data", "只用科斯托蘭尼因子": "kpure", "先驗權重(不最佳化)": "prior"}
EXPO = {"none": "不調整(全持股)", "soft": "溫和(A3 60%、B1 50%、B2 40%)", "strict": "嚴格(A3 30%、B1/B2 0%)"}


def secret(key, default=""):
    try:
        return str(st.secrets.get(key, os.getenv(key, default))).strip()
    except Exception:  # noqa: BLE001
        return os.getenv(key, default).strip()


TOKEN = secret("GITHUB_TOKEN") or None
REPO, REPO_HOW = detect_repo(secret("GITHUB_REPO"), st.query_params.get("repo", ""))
CLOUD = on_streamlit_cloud()


@st.cache_data(ttl=300, show_spinner=False)
def cached_status(repo, token):
    return read_status(repo, token)


@st.cache_data(ttl=300, show_spinner=False)
def research_stamp(repo, token):
    info = release_info(repo, RESEARCH_TAG, token)
    stamp = (info.get("assets", {}).get("bundle.npz") or {}).get("updated_at", "")
    if not stamp and not info:  # GitHub API rate-limited → use the status file instead
        stamp = (read_status(repo, token).get("research") or {}).get("built", "")
    return stamp


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="下載研究資料（約 30~80 MB）…")
def load_github(repo, token, stamp):
    B = Bundle.load(io.BytesIO(fetch_asset(repo, RESEARCH_TAG, "bundle.npz", token)))
    try:
        res = pickle.loads(fetch_asset(repo, RESEARCH_TAG, "results.pkl", token))
    except Exception:  # noqa: BLE001
        res = None
    return B, res


@st.cache_resource(show_spinner="讀取本機 artifacts…")
def load_local(stamp):
    B = Bundle.load("artifacts/bundle.npz")
    res = None
    if Path("artifacts/results.pkl").is_file():
        with open("artifacts/results.pkl", "rb") as f:
            res = pickle.load(f)
    return B, res


@st.cache_resource(show_spinner="產生合成示範資料（約 20 秒）…")
def load_demo(n=320):
    from kosto.panel import build_panel
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    d = Path(tempfile.mkdtemp(prefix="kosto_demo_"))
    stt = Store(d)
    write_demo(stt, n, 11)
    cfg = load_config(overrides={"data": {"dir": str(d)}})
    B = build_bundle(build_panel(stt), cfg, log=lambda *a: None)
    return B, None


@st.cache_resource
def research_obj(bundle_id: str, _B: Bundle):
    return Research(_B, load_config())


def setup_page():
    """Shown until real research results exist: progress of the first build + what to do."""
    st.title("🥚 Kosto Lab｜台股短中長線選股")
    if not CLOUD:
        st.subheader("💻 本機模式（資料下載到這台電腦）")
        prog = local_progress()
        running = local_running()
        prog["stage"] = "downloading" if running else ""
        render_progress(st, prog, "", True)
        if running:
            st.info("⏳ 背景下載中。可以關掉這個網頁，但**電腦請保持開機、不要睡眠**。完成後會自動做研究。")
            st.code(tail_log() or "（啟動中…）", language="text")
            if st.button("🔄 重新整理進度", width="stretch"):
                st.rerun()
        else:
            st.write("第一次需要從 2017 年下載全市場資料，約 6~8 小時（可以分好幾次，中斷後再按一次會接著下載）。")
            if st.button("▶ 開始／繼續建立資料", type="primary", width="stretch"):
                start_local_build()
                time.sleep(2)
                st.rerun()
    elif REPO:
        st.subheader("☁️ GitHub 正在自動建立資料庫")
        st.caption(f"Repo：{REPO}（{REPO_HOW}）")
        status = cached_status(REPO, TOKEN)
        if not status or status.get("stage") == "not_started":
            st.info("尚未收到第一次執行的結果。上傳程式後約 1~10 分鐘會自動開始。")
            st.markdown(f"""若超過 15 分鐘仍沒有進度，請檢查：
1. 打開 [github.com/{REPO}](https://github.com/{REPO})，檔案列表裡要看得到 **`.github`** 資料夾（Mac 上傳前要先按 ⌘+Shift+. 顯示隱藏檔）。
2. 打開 [Actions 頁面](https://github.com/{REPO}/actions)，若出現綠色按鈕「I understand my workflows, go ahead and enable them」請按下。
3. 在 Actions 頁點左側「Kosto Lab 全自動流程」→ 右側「Run workflow」→ 綠色「Run workflow」。""")
        else:
            render_progress(st, status, REPO, False)
            st.info("這段期間不需要做任何事，GitHub 會自己一輪接一輪下載，完成後自動產生研究與報告。")
        if st.button("🔄 重新整理進度", width="stretch"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.warning("找不到您的 GitHub repo 名稱。")
        val = st.text_input("請輸入 GitHub 帳號/repo（例如 yangty/kosto-lab）")
        if val and st.button("確定"):
            st.query_params["repo"] = val.strip()
            st.rerun()
        st.caption("輸入後請把瀏覽器網址加入書籤（網址會記住 repo）。")
    st.divider()
    if st.button("👀 先用示範資料體驗操作（合成資料）", width="stretch"):
        st.session_state["demo"] = True
        st.rerun()
    st.stop()


def get_source():
    st.sidebar.title("🥚 Kosto Lab")
    st.sidebar.caption(f"科斯托蘭尼量化選股實驗室 v{__version__}")
    demo = st.session_state.get("demo", False)
    B = res = None
    src = ""
    if not demo:
        if REPO:
            try:
                stamp = research_stamp(REPO, TOKEN)
                if stamp:
                    B, res = load_github(REPO, TOKEN, stamp)
                    src = "GitHub 每日研究"
            except Exception as e:  # noqa: BLE001
                st.sidebar.caption(f"雲端結果讀取失敗：{e}")
        if B is None and Path("artifacts/bundle.npz").is_file():
            B, res = load_local(Path("artifacts/bundle.npz").stat().st_mtime)
            src = "本機研究"
        if B is None:
            setup_page()
    if B is None:
        B, res = load_demo()
        src = "示範（合成資料）"
    st.sidebar.write(f"來源：**{src}**")
    st.sidebar.write(f"資料至 **{B.sig_dates[-1].date()}**｜{len(B.tickers)} 檔｜{len(B.factors)} 因子")
    st.sidebar.caption(f"建置：{B.meta.get('built')}")
    if demo:
        if st.sidebar.button("回到正式資料"):
            st.session_state["demo"] = False
            st.rerun()
    elif B.meta.get("synthetic"):
        st.sidebar.warning("目前為合成資料，數字不代表真實台股")
    if not CLOUD and src == "本機研究":
        if local_running():
            st.sidebar.info("⏳ 背景更新中…")
        elif st.sidebar.button("🔄 更新到最新交易日（背景）"):
            start_local_build()
            st.rerun()
    return B, (res or {}).get("res") if isinstance(res, dict) and "res" in res else res


B, RES = get_source()
BID = B.meta.get("built", "") + str(len(B.tickers))
R = research_obj(BID, B)
CFG0 = load_config()


@st.cache_data(show_spinner=False, max_entries=64)
def cached_latest(bid, h, rule, size, vkey):
    R.cfg = CFG0
    return R.latest(h, rule, size, Variant(*vkey))


@st.cache_data(show_spinner=False, max_entries=64)
def cached_factor_table(bid, h, rule, size, vkey):
    R.cfg = CFG0
    lt = cached_latest(bid, h, rule, size, vkey)
    return R.factor_table(h, rule, size, lt["weights"])


def vkey(v):
    return (v.name, v.mode, v.lam, v.exposure, v.gates)

st.title("台股短中長線選股｜科斯托蘭尼模型")
if B.meta.get("synthetic"):
    st.warning("⚠ 示範模式（合成資料）：用於體驗流程，結果不可用於投資。")

tabs = st.tabs(["📌 今日推薦", "⚡ 快速驗證", "🧪 母池測試", "📊 因子", "⚖️ 新舊比較", "📥 報告/系統"])


def default_choice(h):
    if RES and h in RES.get("horizons", {}):
        hb = RES["horizons"][h]
        c = hb["chosen"]
        return hb["chosen_pool"]["rule"], hb["chosen_pool"]["size"], Variant(c["variant"], c["mode"], c["lam"], c["exposure"], c["gates"])
    return "KOSTO", 300, Variant("KOSTO λ=0.40 水位=soft", "opt", 0.40, "soft", True)


def picks_frame(items):
    return pd.DataFrame([{"代號": x["ticker"], "名稱": x["name"], "產業": x["sector"], "收盤": round(x["close"], 2),
                          "分數": round(x["score"], 3), "階段": x["stage"], "續抱": "✓" if x.get("kept") else "",
                          "理由": "；".join(x["reasons"]), "停損價": x.get("stop_price")} for x in items])


# ================================================================== 今日推薦
with tabs[0]:
    latest = {}
    for h in HORIZON_KEYS:
        if RES and h in RES.get("horizons", {}):
            latest[h] = RES["horizons"][h]["latest"]
    if len(latest) < 3:
        with st.spinner("計算今日推薦…"):
            for h in HORIZON_KEYS:
                if h not in latest:
                    rule, size, v = default_choice(h)
                    latest[h] = cached_latest(BID, h, rule, size, vkey(v))
    first = latest["short"]
    stage = first["market_stage"]
    c1, c2 = st.columns([2, 3])
    c1.metric("市場雞蛋階段", STAGE_ZH.get(stage, stage))
    c2.caption("A1、B3 買進；A2 續抱；A3、B1 賣出；B2 持現金等待（《一個投機者的告白》雞蛋理論）。"
               f"判斷日：{first['date']} 收盤")
    sub = st.tabs([HZ[h] for h in HORIZON_KEYS])
    for k, h in enumerate(HORIZON_KEYS):
        with sub[k]:
            lt = latest[h]
            st.write(f"母池 **{lt['pool']}**｜模型 **{lt['variant']}**｜建議持股水位 **{lt['exposure']:.0%}**")
            if lt["picks"]:
                st.dataframe(picks_frame(lt["picks"]), hide_index=True, width="stretch")
            else:
                st.info("目前沒有符合條件的標的（可能因市場階段或個股閘門全部排除）。")
            with st.expander("備選名單（下一順位）"):
                if lt["alternates"]:
                    st.dataframe(picks_frame(lt["alternates"]), hide_index=True, width="stretch")
            with st.expander("高分但被科斯托蘭尼規則排除"):
                if lt["blocked"]:
                    st.dataframe(pd.DataFrame(lt["blocked"]).rename(columns={"ticker": "代號", "name": "名稱", "why": "原因"}),
                                 hide_index=True, width="stretch")
            with st.expander("目前因子權重"):
                fz = dict(zip(B.factors, B.meta["factor_zh"]))
                w = pd.DataFrame([{"因子": f, "中文": fz.get(f, f), "權重": round(v, 4)} for f, v in lt["weights"].items()])
                st.dataframe(w.sort_values("權重", ascending=False), hide_index=True, width="stretch")
    st.caption("僅供研究參考，非投資建議；系統不會自動下單。")

# ================================================================== 快速驗證
with tabs[1]:
    st.write("調整設定後按「執行」，直接使用預先算好的因子庫，數秒內完成走勢前進回測（含交易成本）。")
    with st.form("quick_form"):
        a1, a2, a3 = st.columns(3)
        h = a1.selectbox("期間", HORIZON_KEYS, format_func=lambda x: HZ[x], index=1)
        d_rule, d_size, d_v = default_choice(h)
        rule = a2.selectbox("母池規則", list(RULES), index=list(RULES).index(d_rule), format_func=lambda r: f"{r} {RULE_ZH[r]}")
        size = a3.select_slider("母池大小", [100, 150, 200, 300, 400, 500], value=d_size)
        b1, b2, b3 = st.columns(3)
        mode_lab = b1.selectbox("模型", list(MODES), index=list(MODES.values()).index(d_v.mode) if d_v.mode in MODES.values() else 0)
        lam = b2.slider("科斯托蘭尼族群最低權重 λ", 0.0, 0.9, float(d_v.lam if d_v.mode == "opt" else 0.4), 0.05)
        expo = b3.selectbox("市場階段持股水位", list(EXPO), index=list(EXPO).index(d_v.exposure), format_func=lambda x: EXPO[x])
        c1, c2, c3, c4 = st.columns(4)
        gates = c1.checkbox("個股雞蛋/M頭閘門", value=d_v.gates)
        hyst = c2.slider("續抱遲滯倍數", 1.0, 4.0, float(CFG0["horizons"][h]["hysteresis"]), 0.25)
        stop0 = CFG0["horizons"][h]["stop_loss"] or 0.0
        stop = c3.slider("停損（0=不停損）", 0.0, 0.4, float(stop0), 0.01)
        topk = c4.slider("持股檔數", 5, 20, CFG0["research"]["top_k"])
        with st.expander("進階：族群先驗權重 / 成本"):
            fp = CFG0["horizons"][h]["family_prior"]
            e1, e2, e3, e4, e5 = st.columns(5)
            pT = e1.number_input("技術T", 0.0, 1.0, float(fp["T"]), 0.05)
            pC = e2.number_input("籌碼C", 0.0, 1.0, float(fp["C"]), 0.05)
            pF = e3.number_input("基本F", 0.0, 1.0, float(fp["F"]), 0.05)
            pK = e4.number_input("科斯托蘭尼K", 0.0, 1.0, float(fp["K"]), 0.05)
            disc = e5.number_input("手續費折扣", 0.1, 1.0, float(CFG0["costs"]["fee_discount"]), 0.05)
        go = st.form_submit_button("▶ 執行", type="primary", width="stretch")
    if go:
        cfg = copy.deepcopy(CFG0)
        tot = pT + pC + pF + pK or 1
        cfg["horizons"][h]["family_prior"] = {"T": pT / tot, "C": pC / tot, "F": pF / tot, "K": pK / tot}
        cfg["horizons"][h]["hysteresis"] = hyst
        cfg["horizons"][h]["stop_loss"] = stop or None
        cfg["research"]["top_k"] = int(topk)
        cfg["costs"]["fee_discount"] = disc
        v = Variant(f"自訂 {mode_lab} λ={lam:.2f} 水位={expo}", MODES[mode_lab], lam, expo, gates)
        R.cfg = cfg
        with st.spinner("回測中…"):
            ev = R.evaluate(rule, size, h, v, keep=True)
            ew = R.pool_ew(rule, size, h)
            lt = R.latest(h, rule, size, v, ev["_res"])
        st.session_state["quick_res"] = {"cfg": cfg, "h": h, "rule": rule, "size": size, "v": v, "ev": ev, "ew": ew, "lt": lt}
    q = st.session_state.get("quick_res")
    if q:
        ev, res = q["ev"], q["ev"]["_res"]
        m = st.columns(4)
        m[0].metric("開發期 CAGR", f"{ev.get('dev_cagr', np.nan):.1%}", f"0050 {ev.get('dev_bench_cagr', np.nan):.1%}")
        m[1].metric("開發期 Sharpe / MDD", f"{ev.get('dev_sharpe', np.nan):.2f}", f"MDD {ev.get('dev_mdd', np.nan):.1%}",
                    delta_color="off")
        m[2].metric("盲測 CAGR", f"{ev.get('hold_cagr', np.nan):.1%}", f"0050 {ev.get('hold_bench_cagr', np.nan):.1%}")
        m[3].metric("每次周轉率", f"{res.turnover:.0%}", f"年成本 {res.cost_drag:.1%}", delta_color="off")
        nav = pd.DataFrame({"策略": res.nav, "0050": res.bench, "母池等權": q["ew"].nav.reindex(res.nav.index)})
        st.line_chart(nav, height=320)
        st.caption(f"樣本外 {R.B.sig_dates[R.first_oos].date()} 起；盲測 {R.B.sig_dates[R.holdout].date()} 起（盲測不參與任何挑選）。")
        from kosto.metrics import yearly_table
        st.dataframe(yearly_table(res.nav, res.bench).style.format("{:.1%}"), width="stretch")
        with st.expander("今日推薦（此設定）"):
            st.dataframe(picks_frame(q["lt"]["picks"]), hide_index=True, width="stretch")
        with st.expander("每期推薦紀錄"):
            tr = res.tranches.copy()
            tr["picks"] = tr["picks"].map(lambda x: " ".join(x))
            st.dataframe(tr[["date", "stage", "exposure", "n_filled", "ret", "bench", "picks"]], hide_index=True,
                         width="stretch")
        cA, cB = st.columns(2)
        cA.download_button("下載此設定 (config_used.yaml)", dump_yaml(q["cfg"]).encode("utf-8"), "config_used.yaml",
                           width="stretch")
        if cB.button("產生此設定完整報告（含消融，約 30 秒）", width="stretch"):
            R.cfg = q["cfg"]
            with st.spinner("產生報告…"):
                rep = R.single(q["h"], q["rule"], q["size"], q["v"])
                st.session_state["quick_zip"] = build_zip(rep, q["cfg"])
        if st.session_state.get("quick_zip"):
            st.download_button("⬇ 下載報告 ZIP", st.session_state["quick_zip"], "kosto_quick_report.zip", "application/zip",
                               width="stretch")

# ================================================================== 母池測試
with tabs[2]:
    st.write("比較 5 種母池規則 × 6 種大小（同一模型、只看開發期）。")
    if RES:
        for h in [x for x in HORIZON_KEYS if x in RES["horizons"]]:
            hb = RES["horizons"][h]
            st.subheader(f"{HZ[h]}：選定 {hb['chosen_pool']['rule']}{hb['chosen_pool']['size']}")
            pt = hb["pool_table"]
            piv = pt.pivot_table(index="rule", columns="size", values="score")
            piv.index = [f"{i} {RULE_ZH.get(i, '')}" for i in piv.index]
            st.dataframe(piv.style.format("{:.2f}"), width="stretch")
            cols = ["rule", "size", "dev_cagr", "dev_sharpe", "dev_mdd", "dev_excess_cagr", "pool_ew_dev_cagr", "avg_members", "score"]
            st.dataframe(pt[[c for c in cols if c in pt]].style.format({c: "{:.1%}" for c in ("dev_cagr", "dev_mdd", "dev_excess_cagr", "pool_ew_dev_cagr")}
                                                                        | {"dev_sharpe": "{:.2f}", "score": "{:.2f}", "avg_members": "{:.0f}"}),
                         hide_index=True, width="stretch")
    else:
        st.info("尚無完整研究結果（GitHub Actions 每日自動產生）。可在下方即時跑一次。")
    with st.expander("即時重跑母池測試（約 1~3 分鐘）"):
        hh = st.selectbox("期間 ", HORIZON_KEYS, format_func=lambda x: HZ[x])
        quick = st.checkbox("快速（只測 150/300）", value=True)
        if st.button("開始母池測試"):
            R.cfg = CFG0
            prog = st.empty()
            sizes = [150, 300] if quick else CFG0["pools"]["sizes"]
            t = R.pool_study(hh, CFG0["pools"]["rules"], sizes, progress=lambda m: prog.write(m))
            st.dataframe(t[["rule", "size", "dev_cagr", "dev_sharpe", "dev_mdd", "score"]], hide_index=True)

# ================================================================== 因子
with tabs[3]:
    f1, f2, f3 = st.columns(3)
    hf = f1.selectbox("期間  ", HORIZON_KEYS, format_func=lambda x: HZ[x], index=1)
    d_rule, d_size, d_v = default_choice(hf)
    rf = f2.selectbox("母池  ", list(RULES), index=list(RULES).index(d_rule))
    sf = f3.select_slider("大小  ", [100, 150, 200, 300, 400, 500], value=d_size)
    with st.spinner("計算因子 IC…"):
        ft = cached_factor_table(BID, hf, rf, sf, vkey(d_v))
    ft = ft[ft["horizon_ok"]]
    st.bar_chart(ft.set_index("中文")["ic_dev"], height=320)
    st.dataframe(ft[["factor", "中文", "family", "coverage", "ic_dev", "t_dev", "ic_holdout", "ic_all", "weight"]]
                 .style.format({"coverage": "{:.0%}", "ic_dev": "{:.3f}", "t_dev": "{:.1f}", "ic_holdout": "{:.3f}",
                                "ic_all": "{:.3f}", "weight": "{:+.3f}"}), hide_index=True, width="stretch")
    st.caption("IC＝因子排名與未來報酬排名的相關；台股 0.02~0.06 已屬有用。權重 0＝資料與先驗方向相反或不顯著（不翻轉）。")

# ================================================================== 新舊比較
with tabs[4]:
    st.write("上傳舊版 App（Alpha Radar V22）的報告 ZIP 或 `monthly_decisions.csv` / `final_confirmation_records.csv`，"
             "或任何含「日期＋股票代號」的 CSV。兩邊用同一引擎、同一進出場與成本計算。")
    up = st.file_uploader("上傳 CSV / ZIP", type=["csv", "zip"])
    if up is not None:
        try:
            ext = parse_external(up.name, up.getvalue())
            st.success(f"讀到 {len(ext)} 筆推薦（期間 {sorted(ext['horizon'].unique().tolist())} 日）")
            ours = {}
            R.cfg = CFG0
            with st.spinner("計算新模型同期推薦…"):
                for h in HORIZON_KEYS:
                    H = CFG0["horizons"][h]["days"]
                    if H not in set(ext["horizon"]):
                        continue
                    if RES and h in RES["horizons"]:
                        ours[H] = RES["horizons"][h]["tranches"]
                    else:
                        rule, size, v = default_choice(h)
                        ours[H] = R.evaluate(rule, size, h, v, keep=True)["_res"].tranches
            per, summ = head_to_head(Daily.from_bundle(B), ext, ours, CFG0["costs"])
            st.dataframe(summ.style.format({c: "{:.2%}" for c in summ.columns if c not in ("horizon", "signals", "paired", "差異t值")}
                                           | {"差異t值": "{:.2f}"}), hide_index=True, width="stretch")
            st.caption("涵蓋率<100% 代表舊模型有部分股票不在本 App 的資料範圍（只含曾進入母池的股票）；"
                       "GitHub Actions 版的比較會使用全市場資料。")
            st.dataframe(per, hide_index=True, width="stretch")
            st.download_button("下載逐期比較 CSV", per.to_csv(index=False).encode("utf-8-sig"), "head_to_head.csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"無法解析：{e}")

# ================================================================== 報告 / 系統
with tabs[5]:
    st.subheader("完整測試報告")
    zip_bytes = None
    if REPO:
        try:
            zip_bytes = fetch_asset(REPO, RESEARCH_TAG, "report.zip", TOKEN)
        except Exception:  # noqa: BLE001
            zip_bytes = None
    if zip_bytes is None and Path("artifacts/report.zip").is_file():
        zip_bytes = Path("artifacts/report.zip").read_bytes()
    if zip_bytes:
        st.download_button("⬇ 下載完整報告 ZIP（HTML＋Excel＋CSV＋設定）", zip_bytes, "kosto_full_report.zip",
                           "application/zip", type="primary", width="stretch")
    elif RES:
        if st.button("由目前結果產生報告"):
            st.session_state["full_zip"] = build_zip(RES, CFG0)
        if st.session_state.get("full_zip"):
            st.download_button("⬇ 下載報告", st.session_state["full_zip"], "kosto_full_report.zip")
    else:
        st.info("尚無完整報告。可在「快速驗證」產生單一設定的報告，或執行 GitHub Actions。")
    st.subheader("GitHub Actions")
    if REPO:
        st.write(f"Repo：`{REPO}`")
        runs = latest_runs(REPO, TOKEN)
        if runs:
            st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
        if TOKEN:
            mode = st.selectbox("模式", ["full", "quick", "data-only"])
            if st.button("▶ 立即在 GitHub 執行（資料更新＋研究）"):
                try:
                    dispatch(REPO, "pipeline.yml", TOKEN, inputs={"mode": mode})
                    st.success("已啟動，約 20~60 分鐘後重新整理本頁。")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
        else:
            st.markdown(f"手動重跑：到 [Actions 頁面](https://github.com/{REPO}/actions/workflows/pipeline.yml) → "
                        "Run workflow → 綠色 Run workflow（平常不需要，系統每天會自動執行）。")
    else:
        st.caption("本機模式：資料與報告存在這台電腦的 kosto-lab 資料夾（data/、artifacts/）。")
    st.subheader("資料品質")
    st.json(B.meta.get("data_quality", {}))
    with st.expander("目前設定 (config/default.yaml)"):
        st.code(dump_yaml(CFG0), language="yaml")
    st.caption(f"{APP_NAME} v{__version__}")
