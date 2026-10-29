"""KOSTO 量化選股 — Streamlit 前端（v1.8：卡片式選股、持股追蹤（何時賣）、基本面短評、自動研究說明）。

每日研究（資料下載、選股模型、回測）在 GitHub Actions 完成；本 App 讀取研究結果與看盤資料包，提供：
今日選股、盤中監控、國際盤（美股／日韓連動）、個股診斷、資金配置、操作指南，以及給進階使用者的「研究室」。
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
from kosto import chain as chainlib
from kosto import linkage
from kosto import plan as planlib
from kosto import ta
from kosto import ui
from kosto.bundle import Bundle, build_bundle
from kosto.compare import Daily, head_to_head, parse_external
from kosto.config import HORIZON_KEYS, STAGE_ZH, dump_yaml, load_config
from kosto.metrics import perf
from kosto.quotes import INDEX_CH, Quotes, channel, market_phase
from kosto.remote import RESEARCH_TAG, dispatch, fetch_asset, latest_runs, release_info
from kosto.report import build_zip
from kosto.research import Research, Variant
from kosto.setup_page import (detect_repo, local_progress, local_running, on_streamlit_cloud, read_status,
                              render_add_workflow, render_progress, start_local_build, tail_log, workflow_present)
from kosto.universe import RULE_ZH, RULES

st.set_page_config(page_title="KOSTO 量化選股", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
def _is_dark() -> bool:
    """設定檔有指定主題（.streamlit/config.toml 的 theme.base）→ 以設定檔為準（此時 st.context.theme 回報的是
    作業系統偏好、不是畫面實際主題）；沒有設定檔 → 跟隨使用者的系統／選單設定。"""
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base == "dark"
    except Exception:  # noqa: BLE001
        pass
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t == "dark"
    except Exception:  # noqa: BLE001
        pass
    return False


DARK = _is_dark()
st.markdown(ui.css(DARK), unsafe_allow_html=True)

HZ = {"short": "短線", "mid": "中線", "long": "長線"}
HZ_SUB = {"short": "每週調整・持有約 1–4 週", "mid": "每週換 1/4 資金・持有約 1–2 個月", "long": "每 4 週換 1/3 資金・持有約 3 個月"}
HZ_STYLE = {"short": "強勢動能＋營收成長（每週檢查）", "mid": "強勢動能＋營收成長", "long": "產業領先股＋長期趨勢"}
MODES = {"精簡因子(v1.2 中長線預設)": "sparse", "KOSTO 最佳化(λ下限)": "opt", "純數據(無科斯托蘭尼)": "data", "只用科斯托蘭尼因子": "kpure", "先驗權重(不最佳化)": "prior"}
EXPO = {"none": "不調整(全持股)", "soft": "溫和(A3 60%、B1 50%、B2 40%)", "strict": "嚴格(A3 30%、B1/B2 0%)"}
BOARD_ZH = {"TWSE": "上市", "TPEX": "上櫃"}


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


@st.cache_data(ttl=120, show_spinner=False)
def cached_workflow_present(repo, token):
    return workflow_present(repo, token)


@st.cache_data(ttl=300, show_spinner=False)
def research_stamp(repo, token):
    info = release_info(repo, RESEARCH_TAG, token)
    stamp = (info.get("assets", {}).get("bundle.npz") or {}).get("updated_at", "")
    if not stamp and not info:  # GitHub API rate-limited → use the status file instead
        stamp = (read_status(repo, token).get("research") or {}).get("built", "")
    return stamp


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="下載每日研究資料（約 50~90 MB，第一次較久）…")
def load_github(repo, token, stamp):
    B = Bundle.load(io.BytesIO(fetch_asset(repo, RESEARCH_TAG, "bundle.npz", token)))
    try:
        res = pickle.loads(fetch_asset(repo, RESEARCH_TAG, "results.pkl", token))
    except Exception:  # noqa: BLE001
        res = None
    return B, res


@st.cache_resource(show_spinner="讀取本機資料…")
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
    st.title("KOSTO 量化選股｜系統建置中")
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
            if cached_workflow_present(REPO, TOKEN) is False:
                render_add_workflow(st, REPO)
            else:
                st.info("尚未收到第一次執行的結果。上傳程式後約 1~10 分鐘會自動開始。")
                st.markdown(f"""若超過 15 分鐘仍沒有進度：打開 [Actions 頁面](https://github.com/{REPO}/actions)，
若出現綠色按鈕「I understand my workflows, go ahead and enable them」請按下；
然後點左側「Kosto Lab 全自動流程」→ 右側「Run workflow」→ 綠色「Run workflow」。""")
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
    st.sidebar.markdown(f"**KOSTO 量化選股** v{__version__}")
    st.sidebar.write(f"資料來源：{src}")
    st.sidebar.write(f"資料至 **{B.sig_dates[-1].date()}**｜{len(B.tickers)} 檔")
    st.sidebar.caption(f"研究更新：{B.meta.get('built')}")
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
    if st.sidebar.button("🔄 重新載入最新研究", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    return B, (res or {}).get("res") if isinstance(res, dict) and "res" in res else res


B, RES = get_source()
BID = B.meta.get("built", "") + str(len(B.tickers))
R = research_obj(BID, B)
CFG0 = load_config()
TIDX = {t: j for j, t in enumerate(B.tickers)}
NAMES, SECTORS, BOARDS = B.meta["names"], B.meta["sectors"], B.meta.get("boards", [""] * len(B.tickers))
HAS_Q = "Q_C" in B.a and "q_days" in B.meta
FIDX = {f: k for k, f in enumerate(B.factors)}


# ================================================================== shared helpers
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


def default_choice(h):
    if RES and h in RES.get("horizons", {}):
        hb = RES["horizons"][h]
        c = hb["chosen"]
        return hb["chosen_pool"]["rule"], hb["chosen_pool"]["size"], Variant(
            c["variant"], c["mode"], c["lam"], c["exposure"], c["gates"], k=int(c.get("k", 5)), tmin=float(c.get("tmin", 1.0)),
            kK=int(c.get("kK", 0)), lamK=float(c.get("lamK", 0.0)), weighting=c.get("weighting"),
            anchor=float(c.get("anchor", 0.0)), growth_blend=float(c.get("growth_blend", 0.0)),
            hyp_blend=float(c.get("hyp_blend", 0.0)), hypothesis=c.get("hypothesis"), label_h=c.get("label_h"))
    return "KOSTO", 300, Variant("KOSTO λ=0.40 水位=soft", "opt", 0.40, "soft", True)


@st.cache_data(show_spinner=False)
def chain_all(bid):
    return {t: chainlib.label(t, s) for t, s in zip(B.tickers, SECTORS)}


CH = chain_all(BID)


def hblock(h):
    return RES["horizons"].get(h) if RES and "horizons" in RES else None


def latest(h):
    hb = hblock(h)
    if hb:
        return hb["latest"]
    rule, size, v = default_choice(h)
    with st.spinner("計算今日名單…"):
        return cached_latest(BID, h, rule, size, vkey(v))


def prev_picks(h) -> list[str]:
    hb = hblock(h)
    if not hb or "tranches" not in hb:
        return []
    nt = max(1, CFG0["horizons"][h]["days"] // CFG0["research"]["signal_step"])
    tr = hb["tranches"]
    row = tr[tr["i"] == len(B.sig_dates) - 1 - nt]
    return list(row.iloc[0]["picks"]) if len(row) else []


@st.cache_data(show_spinner="計算全市場排名…", max_entries=8)
def ranks(bid, h) -> pd.DataFrame:
    rule, size, v = default_choice(h)
    R.cfg = CFG0
    return R.rank_table(h, rule, size, v).set_index("ticker")


@st.cache_data(show_spinner=False, max_entries=600)
def raw_frame(bid, j):
    return ta.stock_frame(B, j)


@st.cache_data(show_spinner=False, max_entries=600)
def ind_frame(bid, j, period="D"):
    return ta.indicators(ta.resample(raw_frame(bid, j), period))


@st.cache_data(show_spinner=False, max_entries=600)
def chips(bid, j):
    return planlib.chip_summary(raw_frame(bid, j))


def plan_of(j, h):
    return planlib.trade_plan(ind_frame(BID, j, "D"), h) if HAS_Q else None


def last_close(j):
    df = raw_frame(BID, j)
    if len(df) >= 2:
        return float(df["close"].iloc[-1]), float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
    if len(df) == 1:
        return float(df["close"].iloc[-1]), np.nan
    return float(B.a["CLOSE"][-1, j]), np.nan


def chain_html(t, themes=True) -> str:
    c = CH.get(t, {})
    out = ui.tag(c.get("group", ""), "gold") if c.get("group") else ""
    if c.get("segment"):
        out += f"<span class='sub'>{ui.esc(c['segment'])}</span>"
    if themes and c.get("themes"):
        out += " " + "".join(ui.tag(x, "g") for x in c["themes"][:3])
    return out


def stock_cell(t) -> str:
    j = TIDX[t]
    return f"<span class='cd'>{t}</span><span class='nm'>{ui.esc(NAMES[j])}</span>" \
           f"<span class='sub'> {BOARD_ZH.get(BOARDS[j], '')}</span>"


def clean_reason(s: str) -> str:
    """'相對0050強度(120日)(+0.097)' → '相對0050強度'。"""
    import re
    return re.sub(r"[（(][^）)]*[）)]", "", s).strip()


@st.cache_resource
def qclient():
    return Quotes()


@st.cache_data(ttl=15, show_spinner=False)
def live_quotes(chs: tuple):
    q = qclient()
    df = q.fetch(list(chs))
    return df, q.error


def quotes_for(tickers) -> tuple[pd.DataFrame, str]:
    chs = tuple(channel(t, BOARDS[TIDX[t]]) for t in tickers if t in TIDX)
    df, err = live_quotes(chs)
    return (df.set_index("ticker") if len(df) else pd.DataFrame()), err


def is_live_time():
    return market_phase() in ("盤中", "盤前試撮")


K = CFG0["research"]["top_k"]


# ================================================================== 國際盤（美股／日韓／匯率）
WORLD_ORDER = ["SOX", "IXIC", "NQF", "ESF", "TSM", "VIX", "N225", "KS11", "TWD"]
WORLD_SUB = {"SOX": "台股電子最相關", "IXIC": "科技股大盤", "NQF": "近 24 小時交易，台股盤中可看", "ESF": "近 24 小時交易",
             "TSM": "1 ADR = 5 股 2330", "VIX": "> 30 恐慌", "N225": "8:00 開盤（早台股 1 小時）", "KS11": "8:00 開盤（早台股 1 小時）",
             "TWD": "數字變小＝台幣升值"}


@st.cache_data(ttl=60, show_spinner=False)
def world_live() -> pd.DataFrame:
    from kosto.sources import world as W
    try:
        return W.quote(WORLD_ORDER)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def world_snapshot() -> dict:
    """{key: {price, chg, time, spark, live}}；即時抓不到時用研究資料（前一個美股交易日收盤）。"""
    out = {}
    q = world_live()
    for r in q.to_dict("records") if len(q) else []:
        out[r["key"]] = {"price": r["price"], "chg": r["chg"], "time": f"{r['time']:%m/%d %H:%M}", "spark": r["spark"], "live": True,
                         "bar_date": r["bar_date"]}
    W = linkage._world(B)
    for k in W.columns:
        if k in out:
            continue
        s = W[k].dropna()
        s = s[s.diff().fillna(1) != 0] if len(s) > 2 else s       # 去掉對齊造成的重複值
        if len(s) >= 2:
            out[k] = {"price": float(s.iloc[-1]), "chg": float(s.iloc[-1] / s.iloc[-2] - 1), "time": "最近收盤",
                      "spark": s.tail(22).tolist(), "live": False}
    return out


@st.cache_data(show_spinner=False)
def link_stats(bid) -> dict:
    if RES and RES.get("world") and len(RES["world"].get("overnight", [])):
        return RES["world"]
    return linkage.summary(B)


def overnight_read(sox_chg: float) -> tuple[str, dict, str]:
    """把最近一個美股交易日的費半漲跌，對照歷史上台股大型股隔日的平均表現 → (級距, 統計, 建議)。"""
    ov = link_stats(BID).get("overnight")
    if ov is None or not len(ov) or not ui.ok(sox_chg):
        return "", {}, ""
    k = int(np.searchsorted(linkage.BUCKETS, sox_chg, side="left")) - 1
    k = int(np.clip(k, 0, len(linkage.BUCKET_ZH) - 1))
    lab = linkage.BUCKET_ZH[k]
    row = ov[ov["級距"].astype(str) == lab]
    row = row.iloc[0].to_dict() if len(row) else {}
    if not row:
        return lab, {}, ""
    g, oc, ge, oce = row["台股開盤跳空"], row["開盤到收盤"], row["電子股跳空"], row["電子股盤中"]
    stat = (f"歷史 {int(row['天數'])} 次：台股大型股平均開盤 {ui.pct(g, 2)}（電子 {ui.pct(ge, 2)}），"
            f"開盤→收盤 {ui.pct(oc, 2)}（電子 {ui.pct(oce, 2)}），隔一日 {ui.pct(row['隔一日'], 2)}")
    if k == 0:
        adv = "美股重挫的影響幾乎都在開盤一次反映，盤中平均小幅收斂 → 不要在開盤恐慌賣出；名單上的新買進照常掛在買進區（回測：改收盤買反而較差）。"
    elif k == 4:
        adv = "美股大漲隔天台股開高、盤中平均回落 → 開盤不追價，新買進掛在買進區或等開盤 30 分鐘後分批（回測差異約 +0.5%/年，屬小幅優化）。"
    elif k == 3:
        adv = "開高後盤中平均小幅回落 → 不追開盤，掛買進區即可。"
    elif k == 1:
        adv = "開盤小跌、盤中仍偏弱 → 新買進不必搶開盤，盤中回到買進區分批。"
    else:
        adv = "美股平穩，照計畫執行。"
    return lab, {"stat": stat, **row}, adv


def risk_light() -> tuple[str, str, list[str]]:
    st_ = link_stats(BID).get("state") or {}
    snap = world_snapshot()
    risk = list(st_.get("risk") or [])
    vix = snap.get("VIX", {}).get("price", st_.get("VIX"))
    if ui.ok(vix) and vix > 30 and "VIX > 30" not in risk:
        risk.append("VIX > 30")
    lvl = "red" if len(risk) >= 2 else ("amber" if risk else "green")
    txt = {"green": "美股環境正常", "amber": "美股轉弱，留意風險", "red": "美股高風險"}[lvl]
    return lvl, txt, risk


def world_strip() -> str:
    """頁首下方的國際盤小條（費半、那指期、台積電ADR、燈號）。"""
    snap = world_snapshot()
    if not snap:
        return ""
    items = []
    for k, lab in (("SOX", "費半"), ("NQF", "那指期"), ("TSM", "台積ADR"), ("N225", "日經"), ("KS11", "韓股")):
        if k in snap:
            hide = " kq-hide-m" if k in ("N225", "KS11") else ""
            items.append(f"<span class='kq-item{hide}'>{lab}<b>{ui.chg(snap[k]['chg'])}</b></span>")
    lvl, txt, _ = risk_light()
    return ("<div class='kq-bar' style='padding:7px 14px;margin-top:-4px'>" + "".join(items) + ui.light(lvl, txt) + "</div>")


# ================================================================== header
def header(world: bool = True):
    d_last = B.sig_dates[-1].date()
    lt = latest("short")
    stage = lt.get("market_stage", "UNKNOWN")
    ph = market_phase()
    items = []
    if HAS_Q:
        tx = pd.Series(B.a["Q_TAIEX"], index=pd.DatetimeIndex(B.meta["q_days"])).dropna()
        if len(tx) >= 2:
            items.append(("加權指數", f"{tx.iloc[-1]:,.0f} {ui.chg(tx.iloc[-1] / tx.iloc[-2] - 1)}", ""))
    if is_live_time():
        q, _ = live_quotes((INDEX_CH["加權指數"],))
        if len(q) and ui.ok(q["price"].iloc[0]):
            items = [("加權(即時)", f"{q['price'].iloc[0]:,.0f} {ui.chg(q['chg'].iloc[0])}", "")]
    items += [("資料日", str(d_last), " kq-hide-m"), ("市場階段", ui.esc(STAGE_ZH.get(stage, stage).split(" ")[0]), " kq-hide-m")]
    col = {"盤中": "#b4232a", "盤前試撮": "#a16207", "盤前": "#334155", "盤後": "#1e3a8a", "休市": "#334155"}[ph]
    now = pd.Timestamp.now(tz="Asia/Taipei")
    html = ["<div class='kq-bar'><span class='kq-brand'>KOSTO<small class='kq-hide-m'>量化選股</small></span>"]
    html += [f"<span class='kq-item{c}'>{k}<b>{v}</b></span>" for k, v, c in items]
    html.append(f"<span class='kq-pill' style='background:{col};color:#fff'>{ph} {now:%H:%M}</span>")
    if B.meta.get("synthetic"):
        html.append("<span class='kq-pill' style='background:#7c2d12;color:#fff'>示範資料</span>")
    html.append("</div>")
    st.markdown("".join(html) + (world_strip() if world else ""), unsafe_allow_html=True)


# ================================================================== stock view (chart + plan + chips)
MA_OPTS = {5: "週線 5", 10: "10日", 20: "月線 20", 60: "季線 60", 120: "半年線 120", 240: "年線 240"}
IND_OPTS = {"boll": "布林通道", "volume": "成交量", "kd": "KD", "macd": "MACD", "inst": "三大法人", "margin": "融資券", "holders": "大戶持股"}
RANGE = {"3個月": 66, "6個月": 130, "1年": 250, "2年": 500}
PERIOD = {"D": "日K", "W": "週K", "M": "月K"}


def rev_bars(rv: pd.DataFrame, n=13) -> str:
    """近 13 個月營收柱狀圖（紅＝年增、綠＝年減）。"""
    if rv is None or not len(rv):
        return ""
    r = rv.tail(n)
    mx = float(r["rev"].max()) or 1.0
    W, H, bw = 300, 54, 300 / len(r)
    bars = []
    for k, (per, x) in enumerate(r.iterrows()):
        hh = max(2.0, float(x["rev"]) / mx * (H - 14))
        col = ui.UP if ui.ok(x["yoy"]) and x["yoy"] > 0 else (ui.DOWN if ui.ok(x["yoy"]) else ui.FLAT)
        bars.append(f"<rect x='{k * bw + 2:.1f}' y='{H - 12 - hh:.1f}' width='{bw - 4:.1f}' height='{hh:.1f}' rx='2' fill='{col}' "
                    f"fill-opacity='.8'><title>{per.year}/{per.month:02d} {x['rev']:,.1f} 億 年增 {x['yoy'] if ui.ok(x['yoy']) else 0:+.0%}</title></rect>")
        if k in (0, len(r) - 1) or per.month == 1:
            bars.append(f"<text x='{k * bw + bw / 2:.1f}' y='{H - 1}' font-size='9' fill='currentColor' opacity='.6' "
                        f"text-anchor='middle'>{per.month}月</text>")
    return (f"<div class='kq-muted' style='margin:6px 0 2px'>近 {len(r)} 個月營收（紅＝年增、綠＝年減）</div>"
            f"<svg width='100%' height='{H}' viewBox='0 0 {W} {H}' preserveAspectRatio='none' style='color:var(--muted)'>{''.join(bars)}</svg>")


def brief_html(j) -> str:
    from kosto import brief as brieflib
    try:
        b = brieflib.brief(B, j)
    except Exception:  # noqa: BLE001
        return ""
    col = {"正向": "#b4232a", "中性": "#475569", "保守": "#167a4b"}[b["tone"]]
    out = f"<div style='margin-bottom:6px'>{ui.badge('總評：' + b['tone'], col)} <b style='color:var(--strong)'>{ui.esc(b['title'])}</b></div>"
    lines = [f"<span style='color:{ui.UP}'>＋</span> {ui.esc(x)}" for x in b["good"]] + \
            [f"<span style='color:{ui.DOWN}'>－</span> {ui.esc(x)}" for x in b["bad"]]
    if lines:
        out += "<div style='font-size:13px;line-height:1.75'>" + "<br>".join(lines) + "</div>"
    out += rev_bars(b["rev"])
    lk = "　".join(f"<a href='{u}' target='_blank' rel='noopener'>{ui.esc(n)}</a>"
                  for n, u in brieflib.links(B.tickers[j], BOARDS[j]))
    out += (f"<div class='kq-muted' style='margin-top:6px'>以上由月營收、近四季 EPS、估值自動整理；"
            f"法說會展望與新聞請看：{lk}</div>")
    return out


def fundamentals_html(j) -> str:
    if "LATEST_RAW" not in B.a:
        return ""
    raw = B.a["LATEST_RAW"][j]

    def g(f):
        k = FIDX.get(f)
        return float(raw[k]) if k is not None and np.isfinite(raw[k]) else np.nan
    ep, bp, dy, roe = g("f_ep"), g("f_bp"), g("f_dy"), g("f_roe")
    yoy, yoy3, hi, eps = g("f_rev_yoy"), g("f_rev_yoy3"), g("f_rev_high"), g("f_eps_yoy")
    rows = [("本益比", f"{1/ep:.1f}" if ep > 0 else ("虧損" if ui.ok(ep) else "–")),
            ("股價淨值比", f"{1/bp:.2f}" if bp > 0 else "–"),
            ("殖利率", f"{dy:.2f}%" if ui.ok(dy) else "–"),
            ("ROE(推估)", ui.pct(roe, 1, sign=False) if ui.ok(roe) else "–"),
            ("月營收年增", ui.pct(yoy, 1, color=True)), ("近3月營收年增", ui.pct(yoy3, 1, color=True)),
            ("EPS 年增", ui.pct(eps, 0, color=True)),
            ("營收動能", "近 3 月創兩年新高" if ui.ok(hi) and hi >= 0.999 else (f"為兩年高點的 {hi:.0%}" if ui.ok(hi) else "–"))]
    beta = g("g_us_beta")
    if ui.ok(beta):
        rows.append(("對美股費半連動 β", f"{beta:.2f}（費半漲跌 1%，此股隔日平均 {beta:+.2f}%）"))
    return ui.kv(rows)


def plan_html(p, h) -> str:
    if p is None:
        return "<span class='kq-muted'>看盤資料包尚未產生（下次每日更新後提供）</span>"
    rows = [("建議買進區", f"{ui.price(p.buy_low)} ～ {ui.price(p.buy_high)}", "#60a5fa"),
            ("追價上限", ui.price(p.chase), "#f59e0b"),
            ("警戒價", ui.price(p.alert), "#f87171"),
            ("支撐／壓力", f"{ui.price(p.support)} ／ {ui.price(p.resist)}", "var(--text)"),
            ("RSI／月線乖離", f"{p.rsi:.0f} ／ {ui.pct(p.bias20, 1)}" if ui.ok(p.rsi) else "–", "var(--text)")]
    out = "".join(f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed var(--line)'>"
                  f"<span class='kq-muted'>{a}</span><span class='kq-mono' style='color:{c}'>{b}</span></div>" for a, b, c in rows)
    if p.hot:
        out += "<div style='margin-top:6px;color:#fbbf24;font-size:12.5px'>⚠ 短線過熱：只用小部位分批，避免一次追滿</div>"
    out += (f"<div class='kq-muted' style='margin-top:6px'>出場：{HZ[h]}依模型{planlib.CHECK_EVERY[h]}檢查；"
            f"跌出前 {BUF(h)} 名才賣（不設固定停損、讓獲利奔跑）。跌破警戒價時重新檢視基本面與籌碼。</div>")
    return out


def chips_html(c) -> str:
    if not c or c.get("table") is None:
        return "<span class='kq-muted'>無籌碼資料</span>"
    t = c["table"]
    rows = [[ui.esc(r["法人"]), ui.lots_fmt(r["今日(張)"]), ui.lots_fmt(r["近5日(張)"]), ui.lots_fmt(r["近20日(張)"]),
             (f"{'買' if r['連續'] > 0 else '賣'}{abs(r['連續'])}日" if r["連續"] else "–")] for _, r in t.iterrows()]
    out = ui.table(["法人", "今日", "近5日", "近20日", "連續"], rows, left=(0,))
    extra = []
    if ui.ok(c.get("mbal")):
        extra.append(f"融資餘額 {c['mbal']:,.0f} 張")
    if ui.ok(c.get("sbal")):
        extra.append(f"融券餘額 {c['sbal']:,.0f} 張")
    if ui.ok(c.get("usage")):
        extra.append(f"融資使用率 {c['usage']:.1%}")
    if ui.ok(c.get("m20")):
        extra.append(f"融資20日 {ui.pct(c['m20'], 1)}")
    if ui.ok(c.get("short_ratio")):
        extra.append(f"券資比 {c['short_ratio']:.1%}")
    if ui.ok(c.get("big400")):
        ch = c.get("big400_ch")
        extra.append(f"大戶≥400張 {c['big400']:.1f}%" + (f"（週 {ch:+.1f}pp）" if ui.ok(ch) else ""))
    if ui.ok(c.get("conc20")):
        extra.append(f"法人20日集中度 {c['conc20']:+.0%}")
    out += f"<div class='kq-muted' style='margin:8px 0 4px'>{'｜'.join(extra)}</div>" if extra else ""
    out += f"<div style='margin-top:6px'>{ui.tone_badge(c['tone'])}</div>" + ui.bullets([ui.esc(x) for x in c["lines"]])
    return out


def stock_view(j, h="mid", key="sv"):
    t = B.tickers[j]
    c = CH.get(t, {})
    close, ch = last_close(j)
    live_html = ""
    if is_live_time():
        q, _ = quotes_for([t])
        if t in q.index and ui.ok(q.loc[t, "price"]):
            live_html = (f"<span class='kq-item'>即時<b>{ui.price(q.loc[t, 'price'])}</b> {ui.chg(q.loc[t, 'chg'])}"
                         f" <span class='kq-muted'>{q.loc[t, 'time']}</span></span>")
    themes = "".join(ui.tag(x, "g") for x in c.get("themes", []))
    st.markdown(f"<div class='kq-bar'><span style='font-size:20px;font-weight:700;color:var(--strong)'>{ui.esc(NAMES[j])}</span>"
                f"<span class='kq-mono' style='color:var(--accent);font-size:18px'>{t}</span>"
                f"<span class='kq-item'>{BOARD_ZH.get(BOARDS[j], '')}</span>"
                f"<span class='kq-item'>收盤<b>{ui.price(close)}</b> {ui.chg(ch)}</span>{live_html}"
                f"<span>{ui.tag(c.get('group', SECTORS[j]), 'gold')}<span class='kq-muted'> {ui.esc(c.get('segment', ''))}</span> {themes}</span>"
                "</div>", unsafe_allow_html=True)
    if not HAS_Q:
        st.info("看盤資料包（K 線、籌碼）會在更新到 v1.6 後的第一次每日研究產生，約 1 小時後重新整理即可。")
        return
    p = plan_of(j, h)
    c1, c2, c3 = st.columns([1.05, 1.25, 1])
    c1.markdown(ui.panel(f"交易計畫（{HZ[h]}）", plan_html(p, h)), unsafe_allow_html=True)
    c2.markdown(ui.panel("籌碼動向｜三大法人・融資券・大戶", chips_html(chips(BID, j))), unsafe_allow_html=True)
    c3.markdown(ui.panel("基本面快照與短評", brief_html(j) + "<div style='margin-top:8px'></div>" + fundamentals_html(j)),
                unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns([0.8, 1.6, 2.4, 0.9])
    per = a1.segmented_control("週期", list(PERIOD), format_func=PERIOD.get, default="D", key=f"{key}_per") or "D"
    mas = a2.multiselect("均線", list(MA_OPTS), default=[5, 20, 60], format_func=MA_OPTS.get, key=f"{key}_ma")
    inds = a3.multiselect("指標", list(IND_OPTS), default=["boll", "volume", "kd", "macd", "inst"],
                          format_func=IND_OPTS.get, key=f"{key}_ind")
    rng = a4.selectbox("區間", list(RANGE), index=1, key=f"{key}_rng")
    from kosto.charts import kline
    ind = ind_frame(BID, j, per)
    bars = max(20, RANGE[rng] // {"D": 1, "W": 5, "M": 21}[per])
    fig = kline(ind, dark=DARK, mas=tuple(mas), boll="boll" in inds, panels=[x for x in inds if x != "boll"],
                plan=p if per == "D" else None, bars=bars)
    st.plotly_chart(fig, width="stretch", theme=None, key=f"{key}_fig",
                    config={"displaylogo": False, "scrollZoom": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
    st.caption("圖上藍線＝買進區上緣、青點線＝拉回買點、橘虛線＝追價上限、紅虛線＝警戒價。大戶持股為集保每週資料（每週累積）。"
               "券商分點「主力進出」屬付費資料，本系統以「大戶持股＋法人集中度」呈現主力動向。")


# ================================================================== page: 今日選股
def spark_of(j, n=60, fluid=False) -> str:
    if not HAS_Q:
        return ""
    df = raw_frame(BID, j)
    return ui.spark(df["close"].tail(n).tolist(), fluid=fluid) if len(df) > 5 else ""


def phase_stats(h):
    t = (RES or {}).get("phases")
    if t is None or not len(t):
        return None
    from kosto.phases import summary
    sm = summary(t)
    return sm.loc[h].to_dict() if h in sm.index and sm.loc[h, "起點數"] >= 3 else None


def BUF(h) -> int:
    """續抱門檻：排名在前 round(K×續抱倍數) 名內就不賣（與回測相同）。"""
    return max(K, int(round(K * float(CFG0["horizons"][h]["hysteresis"]))))


def pick_items(h, lt) -> list[dict]:
    out, prev = [], set(prev_picks(h))
    for n, p in enumerate(lt["picks"], 1):
        t = p["ticker"]
        if t not in TIDX:
            continue
        j = TIDX[t]
        close, ch = last_close(j)
        out.append({"n": n, "t": t, "j": j, "close": close, "ch": ch, "pl": plan_of(j, h), "cs": chips(BID, j) if HAS_Q else {},
                    "kept": bool(p.get("kept") or (t in prev)),
                    "reasons": "、".join(clean_reason(x) for x in p.get("reasons", [])[:3])})
    return out


def pick_card(it) -> str:
    t, j, pl = it["t"], it["j"], it["pl"]
    c = CH.get(t, {})
    status = ui.badge("續抱", "#5b21b6") if it["kept"] else ui.badge("新進・明日買", "#1d4ed8")
    tone = ui.tone_badge(it["cs"].get("tone", "中性")) if it["cs"] else ""
    lad = ui.ladder(it["close"], pl.buy_low, pl.buy_high, pl.chase, pl.alert, compact=True) if pl else ""
    themes = "".join(ui.tag(x, "g") for x in c.get("themes", [])[:2])
    return (f"<div class='kq-pcard'><div class='hd'><div><span class='rk'>{it['n']}</span><span class='kq-cd'>{t}</span>"
            f"<span class='nm'>{ui.esc(NAMES[j])}</span><span class='kq-sub'> {BOARD_ZH.get(BOARDS[j], '')}</span></div>"
            f"<div class='px'>{ui.price(it['close'])}<small>{ui.chg(it['ch'])}</small></div></div>"
            f"<div class='grp'>{ui.tag(c.get('group', SECTORS[j]), 'gold')}<span class='kq-sub'>{ui.esc(c.get('segment', ''))}</span>{themes}</div>"
            f"<div class='sp'>{spark_of(j, fluid=True)}</div>{lad}"
            f"<div class='ft'>{status} {tone}<span class='why'>{ui.esc(it['reasons'])}</span></div></div>")


def render_picks(items) -> str:
    return "<div class='kq-grid'>" + "".join(pick_card(it) for it in items) + "</div>"


@st.cache_data(show_spinner=False, max_entries=8)
def holdings_of(bid, h) -> list[dict]:
    hb = hblock(h)
    if hb and hb.get("holdings") is not None:
        return hb["holdings"]
    rule, size, v = default_choice(h)
    R.cfg = CFG0
    tr = hb["tranches"] if hb else None
    try:
        return R.holdings(h, rule, size, v, tr, latest(h))
    except Exception:  # noqa: BLE001
        return []


def rank_trail(hist) -> str:
    xs = [x for x in hist if ui.ok(x)]
    if not xs:
        return "–"
    parts = [str(int(x)) if ui.ok(x) else "–" for x in hist[-5:]]
    return "→".join(parts)


def holdings_board(h, lt) -> str:
    """持股追蹤：模型目前持有哪些股票、何時檢查、預計續抱或賣出。"""
    rows, cards_, n = [], [], {"exit": 0, "watch": 0, "keep": 0}
    buf = BUF(h)
    now_set = [p["ticker"] for p in lt["picks"]]
    sold = [t for t in prev_picks(h) if t not in now_set and t in TIDX]
    style = {"exit": ("預計賣出", "#dc2626"), "watch": ("觀察", "#d97706"), "keep": ("續抱", "#16a34a"), "sold": ("明日開盤賣出", "#991b1b")}
    hold = holdings_of(BID, h)
    held = {x["ticker"] for x in hold}
    for t in sold:
        if t in held:
            continue                                       # 其他份資金仍持有 → 併入下方該股那一列
        rows.append(("sold", t, "–", np.nan, "–", "今天（本期換股）", f"本期換出：排名已跌出前 {buf} 名或不符合條件"))
    for x in hold:
        if x["ticker"] not in TIDX:
            continue
        n[x["status"]] += 1
        if x["ticker"] in sold:
            x = dict(x, why=f"本期這份資金已換出（明日開盤賣出）；其餘 {x['slots']} 份待各自檢查日。{x['why']}")
        entry = f"{x['entry_day'] or x['entry_signal'] or '–'}" + (f"<br><span class='kq-sub'>@{ui.price(x['entry_price'])}</span>"
                                                                    if ui.ok(x.get("entry_price")) else "")
        share = f"<br><span class='kq-sub'>持有 {x['slots']}/{x['nt']} 份資金</span>" if x["nt"] > 1 else ""
        rows.append((x["status"], x["ticker"], entry, x.get("ret"), rank_trail(x.get("rank_hist", [])),
                     f"約 {x['next_check']}{share}", x["why"]))
    trs, mc = [], []
    for status, t, entry, ret, trail, nxt, why in rows:
        j = TIDX[t]
        lab, col = style[status]
        trs.append([stock_cell(t), entry, ui.pct(ret, 1, color=True) if ui.ok(ret) else "–", f"<span class='kq-mono'>{trail}</span>",
                    nxt, f"{ui.badge(lab, col)}<div class='kq-sub' style='margin-top:3px;white-space:normal'>{ui.esc(why)}</div>"])
        mc.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[j])}</span>", ui.badge(lab, col), "",
                           f"進場 {entry.split('<br>')[0]}　報酬 {ui.pct(ret, 1, color=True) if ui.ok(ret) else '–'}　排名 {trail}"
                           f"<br>下次檢查 {nxt.split('<br>')[0]}｜{ui.esc(why)}"))
    n_exit, n_watch, n_keep = n["exit"] + len([t for t in sold if t not in held]), n["watch"], n["keep"]
    head = (f"<div class='kq-muted' style='margin:2px 0 8px'>{ui.badge('預計賣出 %d' % n_exit, '#dc2626')} "
            f"{ui.badge('觀察 %d' % n_watch, '#d97706')} {ui.badge('續抱 %d' % n_keep, '#16a34a')}　"
            f"規則：每份資金到「下次檢查日」時，排名仍在前 {buf} 名就續抱，跌出就在隔天開盤賣出、換成當時的新名單；"
            f"不設固定停利停損，所以賣點會隨排名浮動。排名走勢＝最近 5 次（每 5 個交易日）的名次。</div>")
    if not trs:
        return head + "<div class='kq-muted'>目前沒有持股紀錄。</div>"
    desk = ui.table(["股票", "進場日／價", "持有報酬", "排名走勢", "下次檢查日", "判斷"], trs, left=(0, 1, 4, 5))
    return head + ui.resp(desk, "".join(mc))


def preopen_box() -> str:
    snap = world_snapshot()
    if "SOX" not in snap:
        return ""
    sox = snap["SOX"]
    lab, row, adv = overnight_read(sox["chg"])
    if not adv:
        return ""
    when = f"即時 {sox['time']}" if sox.get("live") else "前一個美股交易日收盤"
    nq = snap.get("NQF")
    nq_txt = f"｜那指期 {ui.chg(nq['chg'])}" if nq else ""
    return (f"<div class='kq-verdict'><b>開盤前看美股</b>　費半 {ui.chg(sox['chg'])}（{ui.esc(when)}）{nq_txt}"
            f"　<span class='kq-tag'>{ui.esc(lab)}</span><br><span class='kq-muted'>{row.get('stat', '')}</span><br>{ui.esc(adv)}</div>")


def page_picks():
    header()
    h = st.segmented_control("選股期間", list(HZ), format_func=lambda x: f"{HZ[x]}", default="mid", key="pick_h") or "mid"
    lt = latest(h)
    hb = hblock(h)
    items = []
    ph = phase_stats(h)
    if ph is not None:
        items += [("回測年化 2021 起（五種換股日平均）", ui.pct(ph["樣本外年化_平均"], 1, color=True),
                   f"範圍 {ui.pct(ph['樣本外年化_最低'], 0)}～{ui.pct(ph['樣本外年化_最高'], 0)}｜0050 {ui.pct(ph['0050樣本外'], 1)}"),
                  ("近一年（未參與建模）", ui.pct(ph["近一年_平均"], 1, color=True),
                   f"範圍 {ui.pct(ph['近一年_最低'], 0)}～{ui.pct(ph['近一年_最高'], 0)}｜0050 {ui.pct(ph['0050近一年'], 1)}"),
                  ("最大回撤（平均）", ui.pct(ph["最大回撤_平均"], 1), "股價下跌時帳面可能先虧到這個幅度")]
    elif hb:
        ev = hb["eval"]
        items += [("回測年化（2021 至今）", ui.pct(ev.get("oos_cagr"), 1, color=True), f"同期 0050 {ui.pct(ev.get('oos_bench_cagr'), 1)}"),
                  ("近一年（未參與建模）", ui.pct(ev.get("hold_cagr"), 1, color=True), f"同期 0050 {ui.pct(ev.get('hold_bench_cagr'), 1)}"),
                  ("最大回撤", ui.pct(ev.get("oos_mdd"), 1), f"Sharpe {ev.get('oos_sharpe', np.nan):.2f}")]
    items += [("策略", f"<span style='font-size:15px'>{HZ_STYLE[h]}</span>", HZ_SUB[h])]
    st.markdown(ui.cards(items), unsafe_allow_html=True)
    st.markdown(preopen_box(), unsafe_allow_html=True)
    its = pick_items(h, lt)
    now_set = [p["ticker"] for p in lt["picks"]]
    new = [it["t"] for it in its if not it["kept"]]
    sold = [t for t in prev_picks(h) if t not in now_set and t in TIDX]
    def fmt(xs):
        return "、".join(f"{t} {NAMES[TIDX[t]]}" for t in xs if t in TIDX) or "無"
    st.markdown(f"<div class='kq-verdict'><b>明日開盤要做的事（{HZ[h]}）</b><br>"
                f"🟦 買進：{ui.esc(fmt(new))}<br>🟥 賣出：{ui.esc(fmt(sold))}<br>"
                f"🟪 續抱：其餘 {len(its) - len(new)} 檔不動　<span class='kq-muted'>名單日 {lt['date']} 收盤後產生</span></div>",
                unsafe_allow_html=True)
    st.markdown(ui.section(f"{HZ[h]}推薦 {len(its) or K} 檔", f"市場階段 {ui.esc(STAGE_ZH.get(lt.get('market_stage'), '').split(' ')[0])}"
                           f"｜價位尺：紅＝警戒價、藍＝建議買進區、金＝可小量追、白線＝收盤價"), unsafe_allow_html=True)
    if its:
        st.markdown(render_picks(its), unsafe_allow_html=True)
    else:
        st.info("目前沒有符合條件的標的。")
    st.markdown(ui.section("持股追蹤：什麼時候賣？", f"模型目前持有的股票、下次檢查日與預計續抱／賣出（續抱門檻：前 {BUF(h)} 名）"),
                unsafe_allow_html=True)
    st.markdown(holdings_board(h, lt), unsafe_allow_html=True)
    with st.expander(f"候補名單（未列入推薦但排名在前 {BUF(h)}：已持有可續抱；標「新資金可買」者為排名前 {K}）"):
        try:
            rk = ranks(BID, h)
            alt = rk[rk["valid"] & (rk["rank"] > 0)].sort_values("rank")
            alt = alt[~alt.index.isin(now_set) & (alt["rank"] <= BUF(h))]
            rows2, cards2 = [], []
            for t, r in alt.iterrows():
                cl, ch = last_close(TIDX[t])
                why = "、".join(clean_reason(x) for x in r["reasons"][:3])
                badge_ = " " + ui.tag("新資金可買", "g") if r["rank"] <= K else ""
                rows2.append([f"{int(r['rank'])}{badge_}", stock_cell(t), chain_html(t, themes=False), ui.price(cl), ui.chg(ch),
                              f"<span class='sub'>{ui.esc(why)}</span>"])
                cards2.append(ui.mcard(f"<span class='rk'>#{int(r['rank'])}</span><span class='kq-cd'>{t}</span>"
                                       f"<span class='nm'>{ui.esc(NAMES[TIDX[t]])}</span>{badge_}",
                                       f"{ui.price(cl)}<small>{ui.chg(ch)}</small>", "", ui.esc(why)))
            st.markdown(ui.resp(ui.table(["排名", "股票", "族群", "收盤", "漲跌", "主要加分"], rows2, left=(1, 2, 5)), "".join(cards2)),
                        unsafe_allow_html=True)
        except Exception as e:  # noqa: BLE001
            st.caption(f"無法計算候補：{e}")
    if lt.get("blocked"):
        with st.expander("高分但被風控規則排除"):
            st.markdown(ui.table(["股票", "原因"], [[stock_cell(b["ticker"]), ui.esc(b["why"])] for b in lt["blocked"]
                                                  if b["ticker"] in TIDX], left=(0, 1)), unsafe_allow_html=True)
    st.markdown(ui.section("個股看盤", "K 線、價位計畫、基本面短評、三大法人、融資券、大戶"), unsafe_allow_html=True)
    opts = [p["ticker"] for p in lt["picks"] if p["ticker"] in TIDX]
    if opts:
        sel = st.selectbox("選擇個股", opts, format_func=lambda t: f"{t} {NAMES[TIDX[t]]}｜{CH.get(t, {}).get('group', '')}",
                           key=f"pick_sel_{h}")
        stock_view(TIDX[sel], h, key=f"pv_{h}")
    st.caption("僅供研究參考，非投資建議；系統不會自動下單。回測以隔日開盤價成交並計入手續費、證交稅與滑價；"
               "價位計畫（買進區、追價上限、警戒價）為執行風控建議，未納入回測。")


# ================================================================== page: 盤中監控
def holdings_param() -> list[str]:
    raw = st.query_params.get("w", "")
    return [x for x in raw.replace("，", ",").replace(" ", ",").split(",") if x]


def resolve_codes(text: str) -> list[str]:
    out = []
    by_name = {n: t for t, n in zip(B.tickers, NAMES)}
    for x in text.replace("，", ",").replace("、", ",").replace(" ", ",").split(","):
        x = x.strip()
        if not x:
            continue
        if x in TIDX:
            out.append(x)
        elif x in by_name:
            out.append(by_name[x])
    return list(dict.fromkeys(out))


def page_live():
    header()
    ph = market_phase()
    st.markdown(ui.section("盤中監控", "報價：證交所基本市況報導（約 5 秒延遲）"), unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 2])
    lists = c1.pills("監控名單", list(HZ), selection_mode="multi", default=list(HZ), format_func=HZ.get, key="live_lists") or []
    hold_txt = c2.text_input("我的持股／自選股（代號或名稱，逗號分隔；會記在網址，可加入書籤）",
                             value=",".join(holdings_param()), key="live_hold", placeholder="例：2330, 2603, 聯發科")
    holds = resolve_codes(hold_txt)
    if ",".join(holds) != ",".join(holdings_param()):
        st.query_params["w"] = ",".join(holds)
    auto = st.toggle("盤中每 30 秒自動更新", value=ph in ("盤中", "盤前試撮"), key="live_auto")

    @st.fragment(run_every=30 if auto else None)
    def board():
        member = {}
        for h in lists:
            for p in latest(h)["picks"]:
                member.setdefault(p["ticker"], []).append(h)
        uni = list(dict.fromkeys(list(member) + holds))
        uni = [t for t in uni if t in TIDX]
        if not uni:
            st.info("請選擇名單或輸入持股。")
            return
        q, err = quotes_for(uni) if ph != "休市" else (pd.DataFrame(), "")
        if err:
            st.caption(f"⚠ {err}")
        rows, chances, mcards = [], [], []
        for t in uni:
            j = TIDX[t]
            hs = member.get(t, [])
            h = hs[0] if hs else "mid"
            pl = plan_of(j, h)
            close, ch = last_close(j)
            live = t in q.index and ui.ok(q.loc[t, "price"])
            px = float(q.loc[t, "price"]) if live else close
            chg = float(q.loc[t, "chg"]) if live else ch
            lu = float(q.loc[t, "limit_up"]) if live else np.nan
            role = "pick" if hs else "none"
            if not hs:
                for hh in HZ:
                    try:
                        rk = ranks(BID, hh)
                        if t in rk.index and ui.ok(rk.loc[t, "rank"]) and rk.loc[t, "rank"] <= BUF(hh):
                            role = "keep"
                            break
                    except Exception:  # noqa: BLE001
                        pass
            if live:
                code, txt = planlib.live_status(px, pl, role, lu)
            elif pl and px <= pl.alert:
                code, txt = "warn", "跌破警戒"
            else:
                code, txt = "none", ("開盤後判斷" if ph in ("盤前", "盤前試撮") else "收盤價參考")
            if code in ("buy", "dip"):
                chances.append(t)
            vol = f"{q.loc[t, 'volume']:,.0f}" if live and ui.ok(q.loc[t, "volume"]) else "–"
            rows.append([stock_cell(t), chain_html(t, themes=False), f"<b>{ui.price(px)}</b>", ui.chg(chg),
                         (f"{ui.price(q.loc[t, 'open'])}/{ui.price(q.loc[t, 'high'])}/{ui.price(q.loc[t, 'low'])}" if live else "–"),
                         vol, f"{ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}" if pl else "–",
                         ui.price(pl.chase) if pl else "–", ui.price(pl.alert) if pl else "–",
                         ui.badge(txt, planlib.STATUS_STYLE[code]),
                         " ".join(ui.tag(HZ[x]) for x in hs) + (ui.tag("持股", "g") if t in holds else "")])
            mcards.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[j])}</span> "
                                   + " ".join(ui.tag(HZ[x]) for x in hs) + (ui.tag("持股", "g") if t in holds else ""),
                                   f"{ui.price(px)}<small>{ui.chg(chg)}</small>",
                                   ui.ladder(px, pl.buy_low, pl.buy_high, pl.chase, pl.alert, compact=True) if pl else "",
                                   ui.badge(txt, planlib.STATUS_STYLE[code]) + (f" <span>量 {vol} 張</span>" if vol != "–" else "")))
        tm = q["time"].iloc[0] if len(q) else "收盤價"
        st.markdown(f"<div class='kq-muted'>更新 {pd.Timestamp.now(tz='Asia/Taipei'):%H:%M:%S}｜報價時間 {ui.esc(tm)}｜{ph}</div>",
                    unsafe_allow_html=True)
        st.markdown(ui.resp(ui.table(["股票", "族群", "現價", "漲跌", "開/高/低", "量(張)", "買進區", "追價上限", "警戒價", "狀態", "名單"],
                                     rows, left=(0, 1, 9, 10)), "".join(mcards)), unsafe_allow_html=True)
        st.markdown("<div class='kq-muted' style='margin-top:6px'>" + "　".join(
            f"{ui.badge(k_, planlib.STATUS_STYLE[c_])} {ui.esc(planlib.STATUS_HELP[c_].split('→')[-1].strip())}"
            for c_, k_ in (("buy", "可進場"), ("dip", "拉回買點"), ("wait", "等拉回"), ("warn", "跌破警戒"), ("lock", "漲停不追")))
            + "</div>", unsafe_allow_html=True)
        if chances:
            st.success("盤中可進場：" + "、".join(f"{t} {NAMES[TIDX[t]]}" for t in chances))
        if holds:
            st.markdown(ui.section("持股健檢", "仍在任一期間的續抱門檻內 → 續抱"), unsafe_allow_html=True)
            hr, hc = [], []
            for t in holds:
                if t not in TIDX:
                    continue
                cells, acts = [], []
                for hh in HZ:
                    try:
                        rk = ranks(BID, hh)
                    except Exception:  # noqa: BLE001
                        rk = pd.DataFrame()
                    picks = [p["ticker"] for p in latest(hh)["picks"]]
                    if t in picks:
                        cells.append(ui.badge("推薦中", "#1d4ed8"))
                        acts.append("hold")
                    elif t in rk.index and ui.ok(rk.loc[t, "rank"]):
                        r = int(rk.loc[t, "rank"])
                        cells.append(f"第 {r} 名" + (" ✓" if r <= BUF(hh) else ""))
                        acts.append("hold" if r <= BUF(hh) else "out")
                    elif t in rk.index:
                        cells.append(f"<span class='sub'>{ui.esc(rk.loc[t, 'why'] or '排除')}</span>")
                        acts.append("out")
                    else:
                        cells.append("<span class='sub'>不在選股範圍</span>")
                        acts.append("na")
                pl = plan_of(TIDX[t], "mid")
                px = float(q.loc[t, "price"]) if t in q.index and ui.ok(q.loc[t, "price"]) else last_close(TIDX[t])[0]
                if pl and px <= pl.alert:
                    act = ui.badge("跌破警戒，重新檢視", "#dc2626")
                elif "hold" in acts:
                    act = ui.badge("續抱", "#5b21b6")
                elif "out" in acts:
                    act = ui.badge("下次檢查日換股", "#d97706")
                else:
                    act = ui.badge("非模型標的", "#6b7280")
                hr.append([stock_cell(t), ui.price(px)] + cells + [act])
                hc.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[TIDX[t]])}</span>", ui.price(px), "",
                                   act + " " + "　".join(f"{HZ[hh]}：{c_}" for hh, c_ in zip(HZ, cells))))
            st.markdown(ui.resp(ui.table(["股票", "現價", "短線", "中線", "長線", "建議"], hr, left=(0, 5)), "".join(hc)),
                        unsafe_allow_html=True)
            st.caption(f"規則：仍在任一期間的續抱門檻內（短線前 {BUF('short')}、中線前 {BUF('mid')}、長線前 {BUF('long')} 名）→ 續抱；"
                       "全部跌出 → 在你的下一個檢查日換成名單上的新股票。")

    board()
    with st.expander("盤後：決定明日標的（每晚自動更新）", expanded=ph in ("盤後", "休市", "盤前")):
        st.markdown(f"""
- 系統每個交易日 **23:10** 自動抓取當日收盤、法人、融資券資料並重新選股，約 30–60 分鐘完成；隔天開盤前打開「今日選股」即可看到明日名單（目前名單日：**{latest('short')['date']}**）。
- **明日要買**：名單中標「新進」的股票；**續抱**：標「續抱」或仍在續抱門檻內；**要賣**：手上有、但已跌出續抱門檻（看「今日選股」的持股追蹤或下方「持股健檢」）。
- 隔日 **開盤前** 先看「國際盤」：昨夜費半大跌 → 台股多半開低後盤中小幅收斂，不要開盤殺低；大漲 → 開高後常回落，不追開盤。
- 以「建議買進區上緣」以內掛單；**開盤跳空超過追價上限就不追**，盤中回到買進區再分批買。
- 13:25 前仍未成交 → 當天放棄，下一個交易日再看；不要用市價追。""")


# ================================================================== page: 個股診斷
def verdict_for(t, h, live_px=np.nan):
    """回傳 (等級, 標題, 細節)；等級 buy/hold/watch/no/na。"""
    j = TIDX[t]
    try:
        rk = ranks(BID, h)
    except Exception:  # noqa: BLE001
        rk = pd.DataFrame()
    picks = [p["ticker"] for p in latest(h)["picks"]]
    pl = plan_of(j, h)
    px = live_px if ui.ok(live_px) else last_close(j)[0]
    if t not in rk.index:
        return "na", "不在選股範圍", "成交量或規模未進入模型股票池，模型不評分"
    r = rk.loc[t]
    n = int(r["n_valid"]) if ui.ok(r["n_valid"]) else 0
    top = f"前 {max(1, int(round((1 - r['pct']) * 100)))}%"
    if not r["valid"]:
        return "no", "暫不建議（風控排除）", f"原因：{r['why']}｜分數位於股票池{top}"
    rank = int(r["rank"])
    if t in picks or rank <= K:
        code, txt = planlib.live_status(px, pl, "pick")
        note = "" if t in picks else f"（排名前 {K}；本期名單保留了續抱股，空手者可列為首選候補）"
        if code in ("buy", "dip"):
            return "buy", f"可以買進（第 {rank} 名）", planlib.STATUS_HELP[code] + note
        if code == "wait":
            return "watch", f"推薦但先別追（第 {rank} 名）", planlib.STATUS_HELP["wait"] + note
        if code == "warn":
            return "watch", f"推薦但股價轉弱（第 {rank} 名）", planlib.STATUS_HELP["warn"] + note
        return "buy", f"推薦（第 {rank} 名）", f"{txt}{note}"
    if rank <= BUF(h):
        return "hold", f"持有可續抱（第 {rank}/{n} 名）", f"未進前 {K} 名：空手者不新買，等它進入推薦名單"
    return "no", f"不建議（第 {rank}/{n} 名）", f"分數位於股票池{top}；持有者可在下次檢查日換股"


def page_stock():
    header()
    st.markdown(ui.section("個股診斷", "輸入代號或名稱，三個模型同時評分"), unsafe_allow_html=True)
    opts = sorted(B.tickers)
    q0 = st.query_params.get("s", "")
    idx = opts.index(q0) if q0 in TIDX else None
    t = st.selectbox("輸入股票代號或名稱", opts, index=idx, placeholder="例：2330 或 台積電（可直接打字搜尋）",
                     format_func=lambda x: f"{x} {NAMES[TIDX[x]]}", key="diag_sel")
    if not t:
        st.markdown(ui.panel("使用方式", ui.bullets([
            "輸入代號或公司名，系統會用短、中、長三個模型為這檔股票評分與排名，告訴你「現在能不能買、買在哪裡、跌到哪裡要檢視」。",
            "結果包含：模型排名與理由、建議買進區／追價上限／警戒價、K 線與技術指標、三大法人、融資券、大戶持股。",
            "只有成交量與規模達標、進入模型股票池的股票才會評分（約 500 檔）。"])), unsafe_allow_html=True)
        return
    if q0 != t:
        st.query_params["s"] = t
    j = TIDX[t]
    live_px = np.nan
    if is_live_time():
        q, _ = quotes_for([t])
        if t in q.index:
            live_px = float(q.loc[t, "price"])
    vs = {h: verdict_for(t, h, live_px) for h in HZ}
    color = {"buy": "#1d4ed8", "hold": "#5b21b6", "watch": "#d97706", "no": "#475569", "na": "#475569"}
    items = []
    for h, (lvl, title, det) in vs.items():
        items.append((f"{HZ[h]}｜{HZ_SUB[h].split('・')[0]}", f"<span style='font-size:16px;color:var(--strong)'>{ui.badge(title, color[lvl])}</span>",
                       ui.esc(det)))
    best = next((h for h in ("mid", "short", "long") if vs[h][0] == "buy"), None)
    pl = plan_of(j, best or "mid")
    if best:
        msg = (f"✅ <b>現在可以投資</b>：{HZ[best]}模型推薦。建議價位 <b class='kq-mono'>{ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}</b>，"
               f"高於 <b class='kq-mono'>{ui.price(pl.chase)}</b> 不追；跌破 <b class='kq-mono'>{ui.price(pl.alert)}</b> 重新檢視。") if pl else \
            f"✅ <b>現在可以投資</b>：{HZ[best]}模型推薦。"
    elif any(v[0] == "watch" for v in vs.values()):
        h2 = next(h for h in HZ if vs[h][0] == "watch")
        pl = plan_of(j, h2)
        msg = (f"⏳ <b>可以買但先等價位</b>：{vs[h2][2].split('（')[0]}。等回到 <b class='kq-mono'>{ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}</b> 再分批。"
               if pl else f"⏳ <b>可以買但先等價位</b>：{vs[h2][2]}")
    elif any(v[0] == "hold" for v in vs.values()):
        msg = "🟣 <b>持有者續抱、空手者先觀望</b>：排名在續抱門檻內但未進推薦名單。"
    elif all(v[0] == "na" for v in vs.values()):
        msg = "⛔ <b>不在模型選股範圍</b>：成交量或規模未進入模型股票池，模型不評分；可參考下方技術與籌碼自行判斷。"
    else:
        why = next((v[2].split("｜")[0].replace("原因：", "") for v in vs.values() if v[1].startswith("暫不建議（風控")), "")
        msg = ("⛔ <b>目前不建議買進</b>：" + (f"被風控規則排除（{ui.esc(why)}），等型態修復後再評估。" if why
                                            else "三個模型都沒有把它排進續抱門檻內。"))
    st.markdown(f"<div class='kq-verdict'>{msg}</div>", unsafe_allow_html=True)
    st.markdown(ui.cards(items), unsafe_allow_html=True)
    try:
        rk = ranks(BID, best or "mid")
        if t in rk.index and rk.loc[t, "reasons"]:
            rs = [clean_reason(x) for x in rk.loc[t, "reasons"]]
            rk_ = [clean_reason(x) for x in rk.loc[t, "risk"]]
            st.markdown(f"<div class='kq-muted'>模型加分：{ui.esc('、'.join(rs))}　｜　扣分：{ui.esc('、'.join(rk_) or '無')}</div>",
                        unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass
    stock_view(j, best or "mid", key="diag")


# ================================================================== page: 資金配置
def no50(w: dict) -> dict:
    """拿掉 0050、其餘按比例放大。"""
    ww = {k: v for k, v in w.items() if k != "0050" and v > 0}
    tot = sum(ww.values()) or 1.0
    return {k: v / tot for k, v in ww.items()}


@st.cache_data(show_spinner=False)
def mix_stats(bid, weights: tuple, cash: bool = False):
    """cash=True：模型沒買滿時空出的資金放現金（完全不含 0050）。"""
    if not RES:
        return None
    col = "策略(不含0050)"
    navs = {h: (RES["horizons"][h]["nav"][col] if cash and col in RES["horizons"][h]["nav"] else RES["horizons"][h]["nav"]["策略"])
            for h in HZ if h in RES["horizons"]}
    any_h = next(iter(RES["horizons"].values()))
    navs["0050"] = any_h["nav"]["0050"]
    w = dict(weights)
    nav = planlib.blend_nav(navs, w)
    if nav.empty:
        return None
    b = navs["0050"].reindex(nav.index)
    hs = pd.Timestamp(RES["protocol"]["holdout_start"])
    full, last = perf(nav, b), perf(nav[nav.index >= hs], b[b.index >= hs])
    return {"nav": nav, "bench": b, "full": full, "last": last}


def page_alloc():
    header()
    st.markdown(ui.section("資金配置", "輸入金額 → 每檔張數；可選擇是否包含 0050"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    cap = c1.number_input("投資金額（元）", min_value=50_000, max_value=1_000_000_000, value=1_000_000, step=50_000)
    prof = c2.segmented_control("投資風格", list(planlib.PROFILES) + ["自訂"], default="穩健型", key="alloc_prof") or "穩健型"
    use50 = c3.toggle("包含 0050", value=True, key="alloc_50", help="關閉＝全部資金投入模型股票；模型沒買滿時空出的部分留現金")
    if prof == "自訂":
        s1, s2, s3, s4 = st.columns(4)
        w = {"0050": s1.slider("0050", 0, 100, 30, 5) / 100, "short": s2.slider("短線", 0, 100, 0, 5) / 100,
             "mid": s3.slider("中線", 0, 100, 40, 5) / 100, "long": s4.slider("長線", 0, 100, 30, 5) / 100}
        tot = sum(w.values()) or 1
        w = {k: v / tot for k, v in w.items()}
        st.caption("短線與中線名單重疊約 8 成，同時配置等於重複持有同一批股票，建議兩者擇一。")
    else:
        w = dict(planlib.PROFILES[prof])
        st.markdown(f"<div class='kq-muted'>{planlib.PROFILE_NOTE[prof]}</div>", unsafe_allow_html=True)
    if not use50:
        w = no50(w) or {"mid": 0.5, "long": 0.5}
    mix_lbl = "＋".join(f"{('0050' if k == '0050' else HZ[k])} {v:.0%}" for k, v in w.items() if v > 0)
    ms = mix_stats(BID, tuple(sorted(w.items())), not use50)
    if ms:
        f, l = ms["full"], ms["last"]
        st.markdown(ui.cards([
            ("配置", f"<span style='font-size:15px'>{ui.esc(mix_lbl)}</span>", "每月再平衡" + ("" if use50 else "｜不含 0050")),
            ("回測年化（2021 至今）", ui.pct(f.get("cagr"), 1, color=True), f"0050 {ui.pct(f.get('bench_cagr'), 1)}"),
            ("近一年", ui.pct(l.get("cagr"), 1, color=True), f"0050 {ui.pct(l.get('bench_cagr'), 1)}"),
            ("最大回撤", ui.pct(f.get("mdd"), 1), f"Sharpe {f.get('sharpe', np.nan):.2f}")]), unsafe_allow_html=True)
        from kosto.charts import nav_figure
        st.plotly_chart(nav_figure({f"配置：{mix_lbl}": ms["nav"], "0050": ms["bench"]}, height=260, dark=DARK), width="stretch",
                        theme=None, config={"displaylogo": False})
    with st.expander("一定要買 0050 嗎？（各配置含／不含 0050 的回測比較）", expanded=not use50):
        rows, mc, pairs = [], [], {}
        for name, ww in planlib.PROFILES.items():
            for flag in (True, False):
                if not flag and "0050" not in ww:
                    continue
                wx = ww if flag else no50(ww)
                m = mix_stats(BID, tuple(sorted(wx.items())), not flag)
                if not m:
                    continue
                if "0050" in ww:
                    pairs.setdefault(name, {})[flag] = m
                lbl = name + ("" if flag else "　→ 拿掉 0050")
                rows.append([ui.esc(lbl), ui.esc("＋".join(f"{('0050' if k == '0050' else HZ[k])} {v:.0%}" for k, v in wx.items() if v > 0)),
                             ui.pct(m["full"].get("cagr"), 1, color=True), ui.pct(m["full"].get("mdd"), 1), ui.num(m["full"].get("sharpe")),
                             ui.pct(m["last"].get("cagr"), 1, color=True), ui.pct(m["last"].get("mdd"), 1)])
                mc.append(ui.mcard(f"<span class='nm' style='font-size:14px'>{ui.esc(lbl)}</span>", ui.pct(m["full"].get("cagr"), 1, color=True),
                                   "", f"回撤 {ui.pct(m['full'].get('mdd'), 1)}　Sharpe {ui.num(m['full'].get('sharpe'))}　近一年 "
                                       f"{ui.pct(m['last'].get('cagr'), 1)}（回撤 {ui.pct(m['last'].get('mdd'), 1)}）"))
        st.markdown(ui.resp(ui.table(["配置", "比例", "年化(2021起)", "最大回撤", "Sharpe", "近一年", "近一年回撤"], rows, left=(0, 1)),
                            "".join(mc)), unsafe_allow_html=True)
        facts = []
        for name, pr in pairs.items():
            if True in pr and False in pr:
                a, b_ = pr[True]["full"], pr[False]["full"]
                facts.append(f"{name}：含 0050 年化 {ui.pct(a.get('cagr'), 1, sign=False)}、最大回撤 {ui.pct(a.get('mdd'), 1)}；"
                             f"拿掉 0050 → {ui.pct(b_.get('cagr'), 1, sign=False)}、{ui.pct(b_.get('mdd'), 1)}。")
        st.markdown(ui.bullets([
            "<b>不一定要包含 0050。</b>0050 的角色是「不要落後大盤太多、降低單一模型的風險」，不是提高報酬。"] + [ui.esc(x) for x in facts] + [
            "0050 本身也會大跌（2022 年最大回撤約 −34%），和模型股票高度同向，所以加進來全期回撤改善有限；"
            "但在權值股獨強、模型落後大盤的期間（例如 2019–2020）可以縮小與大盤的落差，近一年含 0050 的配置回撤也較淺。",
            f"模型沒買滿 {K} 檔時（漲停買不到、名單不足），回測預設把空出的資金放 0050；關閉 0050 後改留現金，報酬差異很小。",
            "建議：能承受與大盤不同步、追求報酬 → 可以不買 0050；希望穩定、不想連續幾個月輸大盤 → 保留 2–5 成 0050。"]),
            unsafe_allow_html=True)
    st.markdown(ui.section("下單明細", "以最近收盤價換算，實際以成交價為準"), unsafe_allow_html=True)
    agg, sleeves = {}, []
    for h in HZ:
        if w.get(h, 0) <= 0:
            continue
        picks = [p for p in latest(h)["picks"] if p["ticker"] in TIDX]
        ws = planlib.stock_weights(len(picks), h)
        sleeves.append(f"{HZ[h]} {w[h]:.0%}：{HZ_SUB[h]}")
        for p, wt in zip(picks, ws):
            t = p["ticker"]
            a = agg.setdefault(t, {"amt": 0.0, "src": []})
            a["amt"] += cap * w[h] * wt
            a["src"].append(f"{HZ[h]} {wt:.0%}")
    rows, mc = [], []
    for t, a in sorted(agg.items(), key=lambda kv: -kv[1]["amt"]):
        j = TIDX[t]
        px = last_close(j)[0]
        lot, odd = planlib.lots(a["amt"], px)
        pl = plan_of(j, "mid")
        sh = (f"<b>{lot}</b> 張" + (f" + {odd} 股" if odd else "")) if lot else f"<b>{odd}</b> 股（零股）"
        rows.append([stock_cell(t), chain_html(t, themes=False), " ".join(ui.tag(s) for s in a["src"]), f"{a['amt']:,.0f}",
                     f"{a['amt']/cap:.1%}", ui.price(px), sh, f"{ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}" if pl else "–"])
        mc.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[j])}</span>", sh,
                           "", f"{a['amt']:,.0f} 元（{a['amt']/cap:.1%}）　買進區 {ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}" if pl
                           else f"{a['amt']:,.0f} 元"))
    if w.get("0050", 0) > 0:
        rows.append(["<span class='cd'>0050</span><span class='nm'>元大台灣50</span>", ui.tag("核心 ETF", "gold"), ui.tag("核心"),
                     f"{cap * w['0050']:,.0f}", f"{w['0050']:.1%}", "–", "依現價換算", "可定期定額"])
        mc.append(ui.mcard("<span class='kq-cd'>0050</span><span class='nm'>元大台灣50</span>", f"{w['0050']:.0%}", "",
                           f"{cap * w['0050']:,.0f} 元，可定期定額"))
    if rows:
        st.markdown(ui.resp(ui.table(["股票", "族群", "來源（子帳戶權重）", "金額", "占總資金", "參考價", "建議股數", "建議買進區"],
                                     rows, left=(0, 1, 2)), "".join(mc)), unsafe_allow_html=True)
    st.markdown(ui.panel("執行方式", ui.bullets([ui.esc(s) for s in sleeves] + [
        "中線：把中線資金分成 4 份，每週用 1 份換成當週最新名單（每份約抱 4 週）；長線：分成 3 份，每 4 週換 1 份。",
        "股價太高買不到一張時，用零股（盤後零股或盤中零股）湊足金額。",
        f"模型名單不足 {K} 檔時，空出的比例" + ("放 0050（回測就是這樣計算）。" if use50 else "留現金（已關閉 0050）。")])),
        unsafe_allow_html=True)
    st.caption("回測為每月再平衡、含交易成本的歷史模擬，不代表未來報酬。")


# ================================================================== page: 國際盤
def page_world():
    header(world=False)
    snap = world_snapshot()
    ls = link_stats(BID)
    st.markdown(ui.section("國際盤", "美股收盤、美股期貨、日韓早盤、匯率｜每 60 秒更新"), unsafe_allow_html=True)
    if not snap:
        st.info("國際市場資料尚未建立：下一次每日更新（23:10）會自動抓取，或目前網路無法連到 Yahoo Finance。")
        return
    tiles = []
    for k in WORLD_ORDER:
        if k not in snap:
            continue
        d = snap[k]
        val = ui.num(d["price"], 2 if d["price"] < 1000 else 0)
        sub = ui.esc(WORLD_SUB.get(k, "")) + f"<br>{'即時 ' if d.get('live') else ''}{ui.esc(d['time'])}"
        if k == "TSM" and "TWD" in snap and HAS_Q and "2330" in TIDX:
            px2330 = last_close(TIDX["2330"])[0]
            prem = d["price"] * snap["TWD"]["price"] / 5 / px2330 - 1 if ui.ok(px2330) else np.nan
            if ui.ok(prem):
                sub = f"ADR 溢價 {ui.pct(prem, 1)}（vs 2330 收盤）<br>{'即時 ' if d.get('live') else ''}{ui.esc(d['time'])}"
        name = {"NQF": "那斯達克期貨", "ESF": "標普期貨", "TSM": "台積電 ADR", "TWD": "美元／台幣"}.get(k, linkage_name(k))
        tiles.append(ui.tile(name, val, d["chg"], sub, ui.spark(d.get("spark"), 70, 22, fill=False)))
    st.markdown("<div class='kq-tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
    lvl, txt, risk = risk_light()
    stt = ls.get("state") or {}
    c1, c2 = st.columns([1, 1.25])
    with c1:
        rows = [("費半 vs 百日線", ui.pct(stt.get("SOX_vs_ma100"), 1, color=True)),
                ("VIX 恐慌指數", ui.num(snap.get("VIX", {}).get("price", stt.get("VIX")), 1)),
                ("費半 近 20 日", ui.pct(stt.get("SOX_d20"), 1, color=True)),
                ("韓股 近 20 日", ui.pct(stt.get("KS11_d20"), 1, color=True)),
                ("台幣 近 20 日", ui.pct(-stt.get("TWD_d20", np.nan) if ui.ok(stt.get("TWD_d20")) else np.nan, 1, color=True) + " 升值為正")]
        body = (f"<div style='font-size:16px;margin-bottom:6px'>{ui.light(lvl, txt)}</div>" + ui.kv(rows)
                + "<div class='kq-muted' style='margin-top:8px'>燈號只供參考，不會改變模型名單。實測：費半跌破百日線時把個股降到一半，"
                  "最大回撤可少約 6 個百分點，但年化報酬少約 2.3 個百分點、近一年少 13 個百分點 → 未達採用門檻；"
                  "保守者可自行在紅燈時降低持股。</div>")
        st.markdown(ui.panel("美股風險燈號（研究時資料）", body), unsafe_allow_html=True)
    with c2:
        sox = snap.get("SOX", {})
        lab, row, adv = overnight_read(sox.get("chg", np.nan))
        if adv:
            nq = snap.get("NQF")
            extra = f"；那斯達克期貨目前 {ui.chg(nq['chg'])}" if nq else ""
            body = (f"<div style='margin-bottom:6px'>最近一個美股交易日 費半 <b>{ui.chg(sox['chg'])}</b>{extra}　"
                    f"<span class='kq-tag'>{ui.esc(lab)}</span></div><div class='kq-muted'>{row.get('stat', '')}</div>"
                    f"<div style='margin-top:8px'>{ui.esc(adv)}</div>")
            n225, ks = snap.get("N225"), snap.get("KS11")
            if n225 or ks:
                body += ("<div class='kq-muted' style='margin-top:8px'>日韓 8:00 開盤（比台股早 1 小時）："
                         + "、".join(f"{nm} {ui.chg(x['chg'])}" for nm, x in (("日經", n225), ("韓股", ks)) if x)
                         + "；台股與日韓同日報酬相關係數約 0.6–0.7，開盤前可當領先參考。</div>")
            st.markdown(ui.panel("開盤前判讀（依歷史統計）", body), unsafe_allow_html=True)
    yl = ls.get("yearly")
    if yl is not None and len(yl):
        st.markdown(ui.section("台股越來越受美股影響嗎？", "台股市值前 300 大等權日報酬 vs 前一夜美股／同日日韓"), unsafe_allow_html=True)
        import plotly.graph_objects as go
        from kosto.charts import THEME
        BG, GRID, TXT, MUTED = THEME[bool(DARK)]
        fig = go.Figure()
        for col, nm, c in (("相關_費半(前一夜)", "費半（前一夜）", "#e9b949"), ("相關_那斯達克(前一夜)", "那斯達克（前一夜）", "#5aa9ff"),
                           ("相關_韓股(同日)", "韓股（同日）", "#78d3a6"), ("相關_日經(同日)", "日經（同日）", "#c084fc")):
            if col in yl:
                fig.add_trace(go.Scatter(x=yl["年"].astype(str), y=yl[col], name=nm, mode="lines+markers", line=dict(color=c, width=2.4)))
        fig.update_layout(height=280, margin=dict(l=48, r=10, t=10, b=10), paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TXT),
                          legend=dict(orientation="h", y=-0.2), yaxis=dict(title="相關係數", gridcolor=GRID, range=[0, 1], dtick=0.2,
                                                                         tickformat=".1f"),
                          xaxis=dict(gridcolor=GRID))
        st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "displayModeBar": False})
        r0, r1 = yl.iloc[0], yl.iloc[-2] if len(yl) > 2 else yl.iloc[-1]
        st.markdown(ui.bullets([
            f"台股對前一夜費半的相關係數由 {int(r0['年'])} 年 {r0.get('相關_費半(前一夜)', np.nan):.2f} 升到 {int(r1['年'])} 年 "
            f"{r1.get('相關_費半(前一夜)', np.nan):.2f}；β 由 {r0.get('β_費半(前一夜)', np.nan):.2f} 升到 {r1.get('β_費半(前一夜)', np.nan):.2f}"
            "（費半漲跌 1%，台股大型股隔日平均跟漲跌的幅度）。",
            "連動主要反映在<b>開盤跳空</b>：美股的影響幾乎在台股開盤一次反映完，盤中反而小幅回吐，隔一日已無延續 → "
            "用美股漲跌去「預測」台股接下來幾週哪些股票會贏，實測沒有幫助（見下方研究結論）。",
            "美股 23 小時交易：Nasdaq 與 NYSE Arca 預計 2026-12-06 起週日至週五 23×5 交易（SEC 已核准 Nasdaq 方案）。"
            "夜盤時段 美東 21:00–04:00 ＝ 台北 10:00–17:00（冬令）／09:00–16:00（夏令），會與台股交易時間重疊。"
            "影響：台積電 ADR、美股科技股將在台股盤中即時交易，「隔夜跳空」可能縮小、改為盤中同步波動；"
            "本系統每日收盤後選股、隔日開盤執行的設計不受影響，盤中可多參考本頁的期貨／ADR 報價。"]), unsafe_allow_html=True)
    ov = ls.get("overnight")
    if ov is not None and len(ov):
        with st.expander("明細：前一夜費半漲跌 → 台股隔日平均表現", expanded=False):
            rows = [[ui.esc(r["級距"]), f"{int(r['天數'])}", ui.pct(r["台股開盤跳空"], 2, color=True), ui.pct(r["開盤到收盤"], 2, color=True),
                     ui.pct(r["全日"], 2, color=True), ui.pct(r["隔一日"], 2, color=True), ui.pct(r["電子股跳空"], 2, color=True),
                     ui.pct(r["電子股盤中"], 2, color=True)] for _, r in ov.iterrows()]
            st.markdown(ui.table(["前一夜費半", "天數", "開盤跳空", "開盤→收盤", "全日", "隔一日", "電子跳空", "電子盤中"], rows, left=(0,)),
                        unsafe_allow_html=True)
            st.caption(f"期間 {ov.attrs.get('period', '')}；台股市值前 300 大等權平均。")
    sd = ls.get("study")
    if sd is not None and len(sd):
        st.markdown(ui.section("研究結論：國際連動能不能提高選股報酬？", "v1.7 預先宣告的十面向測試（開發期挑選、盲測與 2018–2020 驗證）"),
                    unsafe_allow_html=True)
        rows, mc = [], []
        for _, r in sd.iterrows():
            good = str(r["結論"]).startswith("✅")
            rows.append([ui.esc(r["面向"]), ui.esc(r["候選"]), ui.pct(r["中線開發期超額Δ"], 1, color=True), ui.pct(r["短線開發期Δ"], 1, color=True),
                         ui.pct(r["長線開發期Δ"], 1, color=True), ui.pct(r["中線回撤Δ"], 1, color=True),
                         ("<b>" if good else "") + ui.esc(r["結論"]) + ("</b>" if good else "")])
            mc.append(ui.mcard(f"<span class='nm' style='font-size:14px'>{ui.esc(r['候選'])}</span><div class='kq-sub'>{ui.esc(r['面向'])}</div>",
                               ui.pct(r["中線開發期超額Δ"], 1, color=True), "", ui.esc(r["結論"])))
        st.markdown(ui.resp(ui.table(["面向", "候選", "中線超額Δ", "短線Δ", "長線Δ", "回撤Δ", "結論"], rows, left=(0, 1, 6)), "".join(mc)),
                    unsafe_allow_html=True)
        st.caption("Δ＝相對 v1.6 的年化超額報酬差（正＝較好）；回撤Δ 正＝回撤較淺。只有「持股 8 檔」通過，已納入 v1.7 模型。")


def linkage_name(k):
    from kosto.sources.world import NAME_ZH
    return NAME_ZH.get(k, k)


# ================================================================== page: 操作指南
def page_guide():
    header()
    st.markdown(ui.section("操作指南", "盤後選股 → 開盤前看國際盤 → 盤中執行 → 定期檢查"), unsafe_allow_html=True)
    steps = [
        ("盤後｜決定明日標的", "每個交易日 23:10 系統自動更新（約 30–60 分鐘）。", [
            "打開「今日選股」，選擇短／中／長線。",
            "標「新進」＝明天要買；標「續抱」＝已持有就不動。",
            "看「價位計畫」：藍色＝買進區、金色＝可小量、超過追價上限不追。"]),
        ("開盤前｜看國際盤", "8:00–9:00 打開「國際盤」。", [
            "昨夜費半大跌：台股多半開低、盤中小幅收斂 → 不要開盤殺低。",
            "昨夜費半大漲：開高後常回落 → 新買進不追開盤。",
            "日韓 8:00 開盤，可當台股開盤的領先參考。"]),
        ("盤中｜執行與監控", "9:00–13:30 使用「盤中監控」，每 30 秒更新。", [
            "現價在建議買進區 → 可進場（分 2–3 筆）。",
            "高於追價上限 → 不追，等拉回；漲停鎖住 → 放棄。",
            "回到拉回買點 → 適合加碼承接。"]),
        ("持有｜何時賣出", "模型不設固定停利停損，讓獲利奔跑。", [
            "短線每週、中線每週一份、長線每 4 週一份做檢查。",
            f"排名仍在續抱門檻內（短 {BUF('short')}／中 {BUF('mid')}／長 {BUF('long')} 名）→ 續抱；跌出 → 隔天開盤換成新名單（「今日選股」的持股追蹤會直接列出）。",
            "跌破警戒價 → 重新檢視籌碼與基本面，不一定要賣。"]),
    ]
    for row in (steps[:2], steps[2:]):
        cols = st.columns(2)
        for k, (title, sub, pts) in enumerate(row):
            n = steps.index((title, sub, pts)) + 1
            cols[k].markdown(f"<div class='kq-step'><span class='n'>{n}</span><span class='t'>{title}</span>"
                             f"<p class='kq-muted' style='margin:6px 0'>{sub}</p>{ui.bullets(pts)}</div>", unsafe_allow_html=True)
    st.markdown(ui.panel("資金怎麼分", ui.bullets([
        "到「資金配置」輸入金額，系統算出每檔張數；每個期間持有 8 檔。",
        "保守：0050 50%＋中線 20%＋長線 30%｜穩健：0050 20%＋中線 40%＋長線 40%｜積極：短線 60%＋長線 40%。",
        "0050 不是必要：關閉「包含 0050」即為全股票配置（報酬較高、波動較大）。短線與中線名單重疊約 8 成，兩者擇一。"])),
        unsafe_allow_html=True)
    st.markdown(ui.section("價位怎麼算"), unsafe_allow_html=True)
    st.markdown(ui.table(["項目", "計算方式", "用途"], [
        ["建議買進區", "上緣＝收盤價＋0.3×ATR（最多 +1.5%）；下緣＝5日線或收盤－1×ATR（不低於 −5%）", "開盤前掛單、盤中分批"],
        ["追價上限", "收盤價＋0.6×ATR（最多 +3.5%）", "開高超過就不追"],
        ["警戒價", "短線 收盤－2×ATR（最多 −7%）／中線 −3×ATR（−10%）／長線 −4×ATR（−15%）", "跌破時重新檢視（非強制停損）"],
        ["支撐／壓力", "月線、20日低點、季線／布林上軌、60日高點", "看圖參考"],
    ], left=(0, 1, 2)), unsafe_allow_html=True)
    st.markdown(ui.section("名詞"), unsafe_allow_html=True)
    st.markdown(ui.bullets([
        "<b>族群</b>：依櫃買中心產業價值鏈平台標示「電子上游／中游／下游－產業鏈｜區段」，綠色標籤為相關題材（AI、雲端、低軌衛星…）。",
        "<b>週線／月線／季線／半年線／年線</b>：5、20、60、120、240 日均線。",
        "<b>KD</b> 9 日（台股慣用）、<b>MACD</b> 12/26/9、<b>布林通道</b> 20 日 ±2 標準差。",
        "<b>籌碼偏多／偏空</b>：綜合三大法人同步買賣、連續買賣超、融資增減與股價方向、大戶持股週變化。"
        "融資增而股價跌＝股票流向「猶豫者」（散戶），法人與大戶增加＝流向「固執者」（科斯托蘭尼）。",
        "<b>市場階段（雞蛋理論）</b>：A1 打底、A2 上漲、A3 過熱、B1 下跌、B2 續跌、B3 恐慌；只做參考，回測顯示依階段調整持股反而降低報酬。",
        "<b>美股風險燈號</b>：費半跌破百日線（黃）、再加上 VIX>30（紅）；只供參考，不改變模型名單。",
    ]), unsafe_allow_html=True)
    st.markdown(ui.section("風險提醒"), unsafe_allow_html=True)
    st.markdown(ui.bullets([
        "回測期間 2021–2026 為台股多頭，模型在 2019–2020 這段（權值股獨強）落後 0050；任何模型都可能連續幾個月落後大盤。",
        "建議先用小資金或紙上交易跑 3–6 個月，確認自己能照規則執行。",
        "本系統僅供研究，不構成投資建議；系統不會自動下單。"]), unsafe_allow_html=True)


# ================================================================== page: 研究室（進階）
def picks_frame(items):
    return pd.DataFrame([{"代號": x["ticker"], "名稱": x["name"], "產業": x["sector"], "收盤": round(x["close"], 2),
                          "分數": round(x["score"], 3), "階段": x["stage"], "續抱": "✓" if x.get("kept") else "",
                          "理由": "；".join(x["reasons"]), "警戒價": x.get("stop_price")} for x in items])


FZH = dict(zip(B.factors, B.meta.get("factor_zh", B.factors)))


def auto_research_tab():
    st.markdown(ui.panel("今日選股用的就是這裡的結果，不需要手動調權重", ui.bullets([
        "每個交易日 23:10，GitHub Actions 自動：①下載最新資料 ②在 30 個母池（5 種規則 × 6 種大小）中依開發期分數<b>自動挑出最佳母池</b> "
        "③用走勢前進法<b>自動估計因子權重</b>（每 6 個月重估一次，只用當時已知的資料）④用全部資料重估一次權重，選出今天的名單 ⑤產生完整報告。",
        "「模型結構」（精簡因子數、假說混合比例、續抱倍數、持股檔數）是每一版研究依預先宣告的多窗口規則決定，寫在設定檔並鎖定"
        "（lock_model）；每天不會因為最近幾週的結果自動換結構，以免追著雜訊跑。",
        "下方「模型監控」若列出挑戰者（開發期與盲測期都勝過目前模型 2 個百分點以上），代表值得在下一版研究檢討 → 把完整報告 ZIP 交給開發者即可。",
        "「自行試驗」分頁只給想自己測參數的人用，不會影響每日名單。"])), unsafe_allow_html=True)
    if not RES:
        st.info("尚無每日研究結果。")
        return
    pr = RES.get("protocol", {})
    st.markdown(f"<div class='kq-muted'>資料至 {pr.get('last_date')}｜樣本外起點 {pr.get('first_oos')}｜盲測起點 {pr.get('holdout_start')}｜"
                f"權重重估日：{'、'.join(pr.get('retrain_points', [])[-4:])}…</div>", unsafe_allow_html=True)
    for h in [x for x in HORIZON_KEYS if x in RES["horizons"]]:
        hb = RES["horizons"][h]
        ev, cp = hb["eval"], hb["chosen_pool"]
        gates = (hb.get("validation") or {}).get("gates")
        npass = f"{int(gates['通過'].sum())}/{len(gates)}" if gates is not None and len(gates) else "–"
        w = (hb.get("latest") or {}).get("weights", {})
        ws = sorted(w.items(), key=lambda kv: -abs(kv[1]))
        wtxt = "、".join(f"{FZH.get(f, f)} {x:+.0%}" for f, x in ws[:8])
        mon = hb.get("monitor")
        mtxt = ("沒有挑戰者同時在開發期與盲測期勝出 → 維持目前模型" if not mon else
                "；".join(f"{ui.esc(m['variant'])}（開發 {m['d_dev']:+.1%}、盲測 {m['d_hold']:+.1%}）" for m in mon)) \
            if mon is not None else "（下次每日研究後提供）"
        ph = phase_stats(h)
        phtxt = (f"{ui.pct(ph['樣本外年化_平均'], 1)}（{ui.pct(ph['樣本外年化_最低'], 0)}～{ui.pct(ph['樣本外年化_最高'], 0)}）／"
                 f"{ui.pct(ph['近一年_平均'], 1)}" if ph else "（下次每日研究後提供）")
        st.markdown(ui.panel(f"{HZ[h]}｜自動選出母池 {cp['rule']}{cp['size']}（{RULE_ZH.get(cp['rule'], '')}）", ui.kv([
            ("模型", ui.esc(hb["chosen"]["variant"])),
            ("本次回測 樣本外年化／近一年／最大回撤", f"{ui.pct(ev.get('oos_cagr'), 1)}／{ui.pct(ev.get('hold_cagr'), 1)}／{ui.pct(ev.get('oos_mdd'), 1)}"),
            ("五種換股日平均 樣本外（範圍）／近一年", phtxt),
            ("驗收閘門通過", npass),
            ("目前因子權重（前 8）", f"<span style='white-space:normal'>{ui.esc(wtxt)}</span>"),
            ("模型監控", f"<span style='white-space:normal'>{mtxt}</span>")])), unsafe_allow_html=True)


def page_lab():
    header()
    st.markdown(ui.section("研究室", "模型回測、母池測試、因子與完整報告（進階）"), unsafe_allow_html=True)
    tabs = st.tabs(["🤖 自動研究結果", "⚡ 自行試驗（選用）", "🧪 母池測試", "📊 因子", "⚖️ 新舊比較", "📥 報告/系統"])
    with tabs[0]:
        auto_research_tab()
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
            c0, c1, c2, c3, c4 = st.columns(5)
            stage_on = c0.checkbox("個股雞蛋階段閘門", value=bool(CFG0["kostolany"]["block_stages"]),
                                   help="A3過熱/B1/B2 不買。v1.1 實測拖累績效，預設關閉")
            dtop_on = c1.checkbox("M頭閘門", value=d_v.gates)
            hyst = c2.slider("續抱遲滯倍數", 1.0, 4.0, float(CFG0["horizons"][h]["hysteresis"]), 0.25)
            stop0 = CFG0["horizons"][h]["stop_loss"] or 0.0
            stop = c3.slider("停損（0=不停損）", 0.0, 0.4, float(stop0), 0.01)
            topk = c4.slider("持股檔數", 5, 20, CFG0["research"]["top_k"])
            s1, s2, s3, s4 = st.columns(4)
            sk = s1.slider("精簡因子數 k", 2, 12, int(getattr(d_v, "k", 5)))
            stmin = s2.select_slider("因子 t 值門檻", [0.5, 1.0, 1.5, 2.0, 2.5], value=float(getattr(d_v, "tmin", 1.0)))
            kplus = s3.checkbox("科斯托蘭尼加強(另保留2個K因子30%)", value=bool(getattr(d_v, "kK", 0)))
            WMODES = {"equal": "等權", "rank": "名次加權(1.5→0.5)", "sqrt_cap": "市值平方根加權", "invvol": "波動倒數加權"}
            w0 = getattr(d_v, "weighting", None) or CFG0["horizons"][h].get("weighting", "equal")
            wmode = s4.selectbox("持股權重", list(WMODES), index=list(WMODES).index(w0), format_func=lambda x: WMODES[x])
            t1, t2, t3, t4 = st.columns(4)
            anc = t1.slider("趨勢錨比例(相對強度/動能)", 0.0, 0.6, float(getattr(d_v, "anchor", 0.0)), 0.05)
            gbl = t2.slider("成長動能假說比例(V23.1)", 0.0, 1.0, float(getattr(d_v, "growth_blend", 0.0)), 0.1)
            to50 = t3.checkbox("沒買股票的資金放 0050", value=CFG0["horizons"][h].get("residual", "cash") == "bench")
            ohot = t4.checkbox("不追高(RSI>84/乖離>14%不買)", value=bool(CFG0["kostolany"].get("overheat_gate", False)))
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
            v = Variant(f"自訂 {mode_lab} λ={lam:.2f} 水位={expo}", MODES[mode_lab], lam, expo, stage_on or dtop_on,
                        stage_gate=stage_on, dtop_gate=dtop_on, k=int(sk), tmin=float(stmin), kK=2 if kplus else 0,
                        lamK=0.3 if kplus else 0.0, weighting=wmode, anchor=float(anc), growth_blend=float(gbl),
                        residual="bench" if to50 else "cash", overheat=bool(ohot),
                        hyp_blend=float(getattr(d_v, "hyp_blend", 0.0)), hypothesis=getattr(d_v, "hypothesis", None),
                        label_h=getattr(d_v, "label_h", None))
            if v.mode == "sparse":
                v.name = (f"自訂 精簡{sk}因子 t>{stmin:g}{'＋K' if kplus else ''}{f'＋趨勢錨{anc:.0%}' if anc else ''}"
                          f"{f'＋成長假說{gbl:.0%}' if gbl else ''} 水位={expo}")
            R.cfg = cfg
            with st.spinner("回測中…"):
                ev = R.evaluate(rule, size, h, v, keep=True)
                ew = R.pool_ew(rule, size, h)
                lt = R.latest(h, rule, size, v, ev["_res"])
            R.cfg = CFG0
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
                R.cfg = CFG0
            if st.session_state.get("quick_zip"):
                st.download_button("⬇ 下載報告 ZIP", st.session_state["quick_zip"], "kosto_quick_report.zip", "application/zip",
                                   width="stretch")
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
        else:
            st.info("尚無完整研究結果（GitHub Actions 每日自動產生）。")
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
        st.caption("IC＝因子排名與未來報酬排名的相關；台股 0.02~0.06 已屬有用。")
    with tabs[4]:
        st.write("上傳其他選股系統的推薦 CSV／ZIP（含「日期＋股票代號」），用同一引擎、同一進出場與成本比較。")
        up = st.file_uploader("上傳 CSV / ZIP", type=["csv", "zip"])
        if up is not None:
            try:
                ext = parse_external(up.name, up.getvalue())
                st.success(f"讀到 {len(ext)} 筆推薦（期間 {sorted(ext['horizon'].unique().tolist())} 日）")
                ours = {}
                with st.spinner("計算本系統同期推薦…"):
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
                st.dataframe(summ, hide_index=True, width="stretch")
                st.dataframe(per, hide_index=True, width="stretch")
                st.download_button("下載逐期比較 CSV", per.to_csv(index=False).encode("utf-8-sig"), "head_to_head.csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"無法解析：{e}")
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


# ================================================================== navigation
pages = [st.Page(page_picks, title="今日選股", icon="📋", url_path="picks", default=True),
         st.Page(page_live, title="盤中監控", icon="⏱️", url_path="live"),
         st.Page(page_world, title="國際盤", icon="🌏", url_path="world"),
         st.Page(page_stock, title="個股診斷", icon="🔍", url_path="stock"),
         st.Page(page_alloc, title="資金配置", icon="⚖️", url_path="alloc"),
         st.Page(page_guide, title="操作指南", icon="📖", url_path="guide"),
         st.Page(page_lab, title="研究室", icon="🧪", url_path="lab")]
st.navigation(pages, position="top").run()
