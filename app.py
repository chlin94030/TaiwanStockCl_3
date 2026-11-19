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
from kosto import freshness
from kosto import intraday as IDY
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


def _is_mobile() -> bool:
    """v2.2 手機版面：依瀏覽器 User-Agent 判斷（手機縮短圖表、減少預設指標；版面本身另由 CSS 依螢幕寬度調整）。"""
    try:
        ua = (st.context.headers or {}).get("User-Agent", "") or ""
    except Exception:  # noqa: BLE001
        return False
    return any(k in ua for k in ("iPhone", "Android", "Mobile", "iPod")) and "iPad" not in ua


IS_MOBILE = _is_mobile()
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


@st.cache_data(ttl=180, show_spinner=False)
def research_stamp(repo, token):
    info = release_info(repo, RESEARCH_TAG, token)
    stamp = (info.get("assets", {}).get("bundle.npz") or {}).get("updated_at", "")
    if not stamp and not info:  # GitHub API rate-limited → use the status file instead
        stamp = (read_status(repo, token).get("research") or {}).get("built", "")
    return stamp


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="下載每日研究資料（約 50~90 MB，第一次較久）…", max_entries=1)
def load_github(repo, token, stamp):
    B = Bundle.load(io.BytesIO(fetch_asset(repo, RESEARCH_TAG, "bundle.npz", token)))
    try:
        res = pickle.loads(fetch_asset(repo, RESEARCH_TAG, "results.pkl", token))
    except Exception:  # noqa: BLE001
        res = None
    return B, res


@st.cache_resource(show_spinner="讀取本機資料…", max_entries=1)
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


def effective_config(res) -> dict:
    """設定檔＋每日研究自動晉級後的冠軍設定（與 GitHub 當天選股用的完全相同）。"""
    from kosto import tournament
    return tournament.apply(load_config(), ((res or {}).get("auto") or {}).get("champions"))


@st.cache_resource
def research_obj(bundle_id: str, _B: Bundle, _cfg: dict):
    return Research(_B, _cfg)


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
CFG0 = effective_config(RES)
R = research_obj(BID, B, CFG0)
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


def exec_day() -> str:
    """名單的執行日：研究資料日之後的平日＝今天（例：11/06 名單、11/09 盤中看 → 今天開盤）；否則下一個交易日。"""
    now = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None)
    return f"今天 {now:%m/%d}" if now.normalize() > RDAY.normalize() and now.weekday() < 5 else "下個交易日"


def plan_today(j, h, bq=None):
    """v2.3.2 交易計畫以「前一個交易日收盤」為基準：研究資料就是前一個交易日時原樣使用；
    今天有新報價、且證交所參考價與計畫收盤不同（除權息、或研究資料落後）→ 依參考價等比例換算（plan.rebase）。"""
    pl = plan_of(j, h)
    if pl is not None and bq and bq.get("live") and ui.ok(bq.get("prev")):
        pl = planlib.rebase(pl, bq["prev"])
    return pl


def last_close(j):
    """研究資料最後一天的收盤與漲跌幅。v2.3.2：有參考價（Q_RF）時用「收盤÷參考價」，除權息日不會顯示成假跌幅。"""
    df = raw_frame(BID, j)
    if len(df) >= 1 and "ref" in df and ui.ok(df["ref"].iloc[-1]) and df["ref"].iloc[-1] > 0:
        return float(df["close"].iloc[-1]), float(df["close"].iloc[-1] / df["ref"].iloc[-1] - 1)
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


RDAY = pd.Timestamp(B.meta["q_days"][-1]) if HAS_Q else pd.Timestamp(B.sig_dates[-1])     # 研究資料（收盤價）的日期


def now_quote(t, d=None, q=None) -> dict:
    """v2.3.1 所有頁面共用的現價：證交所即時 → 今天的分時 → 研究資料收盤（標日期）。見 intraday.best_quote。"""
    j = TIDX[t]
    close, ch = last_close(j)
    df = raw_frame(BID, j)
    rd = df.index[-1] if len(df) else RDAY                     # 這檔最後有收盤的日期（暫停交易的股票可能比研究資料日舊）
    row = q.loc[t].to_dict() if q is not None and len(q) and t in q.index else None
    bq = IDY.best_quote(row, d, close, ch, rd, etf=False)
    if bq["live"] and market_phase() == "盤前試撮":            # 8:30–9:00 是試撮，不是成交
        bq["label"] = "盤前試撮"
    return bq


def quote_tags(bq) -> str:
    """現價旁的小標籤：漲停／跌停、委買價／委賣價、分時（可能延遲）、MM/DD 收盤。"""
    out = ""
    if bq.get("lock") == "up":
        out += f"<span class='kq-lock up'>漲停</span>"
    elif bq.get("lock") == "down":
        out += f"<span class='kq-lock dn'>跌停</span>"
    if bq.get("label"):
        out += (f"<span class='kq-sub' style='display:block;font-size:10.5px;font-weight:400;line-height:1.3'>"
                f"{ui.esc(bq['label'])}</span>")
    sl = IDY.source_line(bq)                                      # v3.14 報價來源、時間、是否過期
    if sl:
        col = ";color:#b45309" if bq.get("stale_min") else ""
        out += (f"<span class='kq-sub' style='display:block;font-size:10px;font-weight:400;line-height:1.3{col}'>"
                f"{ui.esc(sl)}</span>")
    return out


K = int(CFG0["horizons"]["mid"].get("top_k") or CFG0["research"]["top_k"])


def KH(h) -> int:
    """各期間持股檔數（自動晉級可能讓短、中、長線不同）。"""
    return int(CFG0["horizons"][h].get("top_k") or CFG0["research"]["top_k"])


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


def world_tile_chart(di, spark) -> str:
    """國際盤小卡的圖：有分時用分時（平盤虛線）；沒有就用近 1 月收盤（期初虛線）。"""
    if di is not None:
        html = ui.intraday_mini(di, h=44, labels=False)
        if html:
            return html
    return ui.trend_mini(spark, h=40, labels=False) if spark else ""


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


@st.cache_data(ttl=600, show_spinner=False)
def world_live_state() -> dict:
    """v2.2.2 最新美股收盤的國際狀態（Yahoo 日線，免金鑰）：週末／盤前打開 App 時，用週五（前一晚）的美股收盤，
    不必等下一次每日研究。欄位與 linkage.world_state 相同（SOX、SOX_ma100、SOX_vs_ma100、SOX_d20、VIX、KS11_d20、TWD_d20）。"""
    from concurrent.futures import ThreadPoolExecutor

    import requests
    from kosto.sources import world as W
    tzs = {"SOX": "America/New_York", "VIX": "America/Chicago", "KS11": "Asia/Seoul", "TWD": "Europe/London"}

    def get(k):
        try:
            r = requests.get(W.YAHOO + requests.utils.quote(W.SYMBOLS[k], safe=""), params={"range": "1y", "interval": "1d"},
                             headers=W.UA, timeout=6)
            r.raise_for_status()
            m = r.json()["chart"]["result"][0]
            c = pd.Series((m.get("indicators") or {}).get("quote", [{}])[0].get("close") or [],
                          index=pd.to_datetime(m.get("timestamp") or [], unit="s", utc=True), dtype=float).dropna()
            return k, c
        except Exception:  # noqa: BLE001
            return k, pd.Series(dtype=float)
    out = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, c in ex.map(get, list(tzs)):
            if len(c) < 25:
                continue
            out[k] = float(c.iloc[-1])
            out[f"{k}_d1"] = float(c.iloc[-1] / c.iloc[-2] - 1)
            out[f"{k}_d20"] = float(c.iloc[-1] / c.iloc[-21] - 1)
            if k == "SOX":
                out["us_date"] = str(c.index[-1].tz_convert(tzs[k]).date())
                ma = c.rolling(100, min_periods=80).mean()
                if np.isfinite(ma.iloc[-1]):
                    out["SOX_ma100"] = float(ma.iloc[-1])
                    out["SOX_vs_ma100"] = out["SOX"] / out["SOX_ma100"] - 1
    return out if "us_date" in out else {}


def world_state_now() -> dict:
    """研究時的國際狀態，若 Yahoo 已有更新的美股收盤（例如週五晚上 → 週一開盤前）就改用最新的；src＝live／research。"""
    st_ = dict(link_stats(BID).get("state") or {})
    st_["src"] = "research"
    lv = world_live_state()
    if lv and lv.get("us_date", "") >= str(st_.get("date", ""))[:10]:
        st_.update(lv)
        st_["src"] = "live"
        risk = []
        if ui.ok(st_.get("SOX_vs_ma100")) and st_["SOX_vs_ma100"] < 0:
            risk.append("費半跌破百日線")
        if ui.ok(st_.get("VIX")) and st_["VIX"] > 30:
            risk.append("VIX > 30")
        st_["risk"] = risk
    return st_


def world_label(stt: dict) -> str:
    if stt.get("src") == "live":
        return f"{pd.Timestamp(stt['us_date']):%m/%d}（{'一二三四五六日'[pd.Timestamp(stt['us_date']).weekday()]}）美股收盤"
    return "研究時資料"


def risk_light() -> tuple[str, str, list[str]]:
    st_ = world_state_now()
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
@st.cache_data(ttl=180, show_spinner=False)
def actions_runs(repo, token):
    try:
        runs = latest_runs(repo, token, 10)
    except Exception:  # noqa: BLE001
        return []
    return [r for r in runs if "全自動" in (r.get("name") or "") or "pipeline" in (r.get("name") or "").lower()] or runs


def data_status() -> tuple[str, str, str, str]:
    """(燈號, 文字, 連結, 滑鼠提示)：見 kosto/freshness.py。"""
    runs = actions_runs(REPO, TOKEN) if REPO else []
    run = runs[0] if runs else None
    lvl, txt, url = freshness.status(pd.Timestamp.now(tz="Asia/Taipei"), pd.Timestamp(B.sig_dates[-1]), run, runs)
    dd = B.meta.get("data_dates") or {}
    if lvl == "green" and dd.get("margin") and dd.get("price") and dd["margin"] < dd["price"]:
        txt += "｜融資券 21:40 補上"
    return lvl, txt, url, freshness.run_desc(run)


def update_now_row():
    """資料落後時：有 GITHUB_TOKEN → 「立即更新」按鈕；沒有 → 告訴使用者到 GitHub 按 Run workflow。"""
    if not REPO:
        return
    link = f"https://github.com/{REPO}/actions/workflows/pipeline.yml"
    if TOKEN:
        if st.session_state.get("kq_dispatched"):
            st.markdown("<div class='kq-muted'>已啟動自動更新，約 30～60 分鐘完成；完成後重新整理頁面即可。</div>",
                        unsafe_allow_html=True)
            return
        c1, c2 = st.columns([1, 4], vertical_alignment="center")
        if c1.button("🔄 立即更新", key="kq_update_now", type="primary"):
            err = None
            for ref in ("main", "master"):
                try:
                    dispatch(REPO, "pipeline.yml", TOKEN, ref=ref, inputs={"mode": "auto"})
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = e
            if err is None:
                st.session_state["kq_dispatched"] = True
                actions_runs.clear()
                st.toast("已啟動 GitHub 自動更新（約 30～60 分鐘）")
            else:
                c2.warning(f"啟動失敗：{err}。請改到 GitHub 手動執行。")
        c2.markdown(f"<div class='kq-muted'>或到 <a href='{link}' target='_blank'>GitHub Actions</a> 按「Run workflow」。</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='kq-muted'>手動更新：到 <a href='{link}' target='_blank'>GitHub Actions</a> → 右側「Run workflow」→ 綠色「Run workflow」，"
                    "約 30～60 分鐘完成（之後也有 23:17、00:47、06:23 備援排程自動補跑）。</div>", unsafe_allow_html=True)


def taiex_now() -> dict:
    out = {}
    if "Q_TAIEX" in B.a and "q_days" in B.meta:
        tx = pd.Series(B.a["Q_TAIEX"], index=pd.DatetimeIndex(B.meta["q_days"])).dropna()
        if len(tx) >= 2:
            out = {"px": float(tx.iloc[-1]), "chg": float(tx.iloc[-1] / tx.iloc[-2] - 1), "date": tx.index[-1], "live": False}
    ph = market_phase()
    if ph in ("盤前試撮", "盤中", "盤後"):
        # v2.5 頁首與「加權指數｜今日走勢」用同一份資料（分時＋證交所最新價合併），不會一個 43,236、一個 43,444
        d = taiex_intraday()
        if d is not None and (not out or pd.Timestamp(d["date"]) > out["date"]) and d["prev"] > 0:
            return {"px": float(d["last"]), "chg": float(d["last"] / d["prev"] - 1), "live": True,
                    "date": pd.Timestamp(d["date"]), "time": f"{d['time']:%H:%M}", "delayed": not d.get("live_point")}
        q, _ = live_quotes((INDEX_CH["加權指數"],))
        if len(q) and ui.ok(q["price"].iloc[0]):
            out = {"px": float(q["price"].iloc[0]), "chg": float(q["chg"].iloc[0]), "live": True,
                   "date": pd.Timestamp.now(tz="Asia/Taipei").normalize().tz_localize(None), "time": q["time"].iloc[0]}
    return out


# ================================================================== v2.1.1 當日分時走勢（平盤線、紅漲綠跌）
def _tw_live() -> bool:
    return market_phase() in ("盤中", "盤前試撮")


def _slot(fast: bool) -> str:
    """快取時間槽：盤中每分鐘更新，其他時間每 10 分鐘。"""
    now = pd.Timestamp.now(tz="Asia/Taipei")
    return f"{now:%Y%m%d%H%M}" if fast else f"{now:%Y%m%d%H}{now.minute // 10}"


@st.cache_data(ttl=1800, show_spinner=False, max_entries=64)
def intraday_raw(syms: tuple, slot: str, interval: str = "2m") -> dict:
    try:
        return IDY.fetch_many(list(syms), interval)
    except Exception:  # noqa: BLE001
        return {}


def tw_intraday(tickers, interval: str = "2m") -> dict:
    """台股個股分時（Yahoo）＋盤中以證交所即時價補最新一點、以官方昨收為平盤。"""
    tk = [t for t in dict.fromkeys(tickers) if t in TIDX]
    if not tk:
        return {}
    sym = {t: IDY.yahoo_symbol(t, BOARDS[TIDX[t]]) for t in tk}
    live = _tw_live()
    raw = intraday_raw(tuple(sorted(set(sym.values()))), _slot(live), interval)
    out = {t: raw.get(s) for t, s in sym.items()}
    if live:
        q, _ = quotes_for(tk)
        for t in tk:
            if len(q) and t in q.index:
                out[t] = IDY.merge_live(out[t], q.loc[t].to_dict())
    return {t: IDY.snap_ticks(d) for t, d in out.items() if d is not None and len(d["v"]) >= 2}


def taiex_intraday() -> dict | None:
    live = _tw_live()
    d = intraday_raw((IDY.TAIEX,), _slot(live), "1m").get(IDY.TAIEX)
    if live:
        q, _ = live_quotes((INDEX_CH["加權指數"],))
        if len(q):
            d = IDY.merge_live(d, q.iloc[0].to_dict())
    return d if d is not None and len(d["v"]) >= 2 else None


def world_intraday(keys) -> dict:
    """國際指數／期貨／個股的最近一個交易時段分時（美股晚上盤中每分鐘更新）。"""
    from kosto.sources import world as W
    sym = {k: W.SYMBOLS[k] for k in keys if k in W.SYMBOLS}
    raw = intraday_raw(tuple(sorted(set(sym.values()))), _slot(True), "5m")
    return {k: raw[s] for k, s in sym.items() if s in raw and len(raw[s]["v"]) >= 2}


def intraday_title(d) -> str:
    r = d["last"] / d["prev"] - 1 if d["prev"] else np.nan
    return f"{IDY.status(d)} <b style='color:{ui.color_of(r)}'>{r * 100:+.2f}%</b>" if ui.ok(r) else IDY.status(d)


def intraday_stats(d, digits: int = 2, unit: str = "") -> str:
    """分時圖上方的數字列：現價、漲跌點與幅度、開高低、振幅、平盤、時間。"""
    f = (lambda x: ui.num(x, digits)) if digits is not None else ui.price
    diff = d["last"] - d["prev"]
    r = diff / d["prev"]
    c = ui.color_of(r)
    arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "")
    amp = (d["high"] - d["low"]) / d["prev"]
    return (f"<div class='kq-istat'><span class='big' style='color:{c}'>{f(d['last'])}</span>"
            f"<span style='color:{c};font-weight:600'>{arrow}{f(abs(diff))}（{r * 100:+.2f}%）</span>"
            f"<span>開 <b>{f(d['open'])}</b></span><span>高 <b style='color:{ui.color_of(d['high'] - d['prev'])}'>{f(d['high'])}</b></span>"
            f"<span>低 <b style='color:{ui.color_of(d['low'] - d['prev'])}'>{f(d['low'])}</b></span>"
            f"<span>振幅 <b>{amp * 100:.2f}%</b></span><span>平盤 <b>{f(d['prev'])}</b></span>"
            f"<span>{ui.esc(IDY.status(d))}{unit}</span></div>")


def intraday_chart(d, key: str, height: int = 300, digits: int = 2, volume: bool = True, snap: bool = False):
    from kosto.charts import intraday_figure
    st.plotly_chart(intraday_figure(d, height=height, dark=DARK, digits=digits, volume=volume, snap=snap), width="stretch", theme=None,
                    key=key, config={"displaylogo": False, "displayModeBar": False, "scrollZoom": False})


def mini_rows(rows) -> str:
    """[(名稱, 分時資料, 補充)] → 一列一檔的小分時圖清單。"""
    out = []
    for name, d, extra in rows:
        if d is None:
            continue
        r = d["last"] / d["prev"] - 1
        out.append(f"<div class='kq-imini'><span class='n' title='{ui.esc(name)}'>{ui.esc(name)}</span>"
                   f"<span class='g'>{ui.intraday_mini(d, h=30, labels=False)}</span>"
                   f"<span class='c' style='color:{ui.color_of(r)}'>{r * 100:+.2f}%</span></div>"
                   + (f"<div class='kq-muted' style='margin:-4px 0 4px'>{extra}</div>" if extra else ""))
    return "".join(out)


def header(world: bool = True):
    ph = market_phase()
    now = pd.Timestamp.now(tz="Asia/Taipei")
    tx = taiex_now()
    html = ["<div class='kq-bar kq-top'><span class='kq-brand'>KOSTO<small class='kq-hide-m'>量化選股</small></span>"]
    if tx:
        tag = (("延遲" if tx.get("delayed") and ph == "盤中" else "即時") if ph in ("盤中", "盤前試撮") else "今日收盤") \
            if tx.get("live") else f"{tx['date']:%m/%d} 收盤"
        html.append(f"<span class='kq-item'>加權<b>{tx['px']:,.0f}</b> {ui.chg(tx['chg'])} <span class='kq-sub'>{tag}</span></span>")
    lvl, txt, url, tip = data_status()
    st_html = ui.light(lvl, txt)
    if url:
        st_html = (f"<a href='{url}' target='_blank' rel='noopener' title='{ui.esc(tip)}' style='text-decoration:none'>"
                   f"{st_html}</a>")
    html.append(f"<span class='kq-item'>研究資料日<b>{B.sig_dates[-1]:%m/%d}</b></span><span class='kq-small'>{st_html}</span>")
    if world:
        snap = world_snapshot()
        for k, lab in (("SOX", "費半"), ("NQF", "那指期"), ("TSM", "台積ADR")):
            if k in snap:
                html.append(f"<span class='kq-item kq-hide-m'>{lab}<b>{ui.chg(snap[k]['chg'])}</b></span>")
        rl, rt, _ = risk_light()
        html.append(f"<span class='kq-small kq-hide-m'>{ui.light(rl, rt)}</span>")
    col = {"盤中": "#b4232a", "盤前試撮": "#a16207", "盤前": "#334155", "盤後": "#1e3a8a", "休市": "#334155"}[ph]
    html.append(f"<span class='kq-pill' style='background:{col};color:#fff;margin-left:auto'>{ph} {now:%H:%M}</span>")
    if B.meta.get("synthetic"):
        html.append("<span class='kq-pill' style='background:#7c2d12;color:#fff'>示範資料</span>")
    html.append("</div>")
    st.markdown("".join(html) + quick_nav(), unsafe_allow_html=True)
    if lvl == "red":
        update_now_row()
    if lvl != "green" and REPO:
        rr = freshness.run_rows(actions_runs(REPO, TOKEN))
        if rr:
            with st.expander("最近幾次自動更新（研究資料為什麼還沒更新？）"):
                rows = [[ui.esc(r["start"]), ui.esc(r["event"]),
                         f"<b style='color:{ui.DOWN if r['result'] == '成功' else ui.UP if r['result'] == '失敗' else 'inherit'}'>{ui.esc(r['result'])}</b>",
                         f"{r['minutes']} 分" if r["minutes"] is not None else "–",
                         f"<a href='{ui.esc(r['url'])}' target='_blank' rel='noopener'>看細節</a>" if r["url"] else ""] for r in rr]
                st.markdown(ui.table(["開始（台北）", "觸發", "結果", "耗時", "GitHub"], rows, left=(0, 1)), unsafe_allow_html=True)
                st.caption("每天 17:23 初版、21:40 完整版，23:17／00:47／06:23 為備援；同一時間只會跑一個，後到的會排隊（再後到的會被取消）。"
                           "連續「失敗」：點「看細節」→ 紅色 ✗ 的步驟看錯誤訊息；「成功」但資料日沒前進：多半是資料源（證交所）當天沒有給資料。"
                           "上傳新程式會觸發一次更新，Streamlit 也會重新啟動（所以打開時會看到 reboot）。")


QNAV = [("home", "🏠", "總覽"), ("picks", "📋", "選股"), ("insight", "🎛️", "儀表板"), ("live", "⏱️", "盤中"), ("stock", "🔍", "個股"), ("chain", "🗺️", "產業鏈"),
        ("us", "🗽", "美股"), ("world", "🌏", "國際"),
        ("market", "📈", "大盤"), ("etf", "💰", "ETF"), ("alloc", "⚖️", "配置"), ("validate", "🧪", "驗證"), ("guide", "📖", "指南")]


def quick_nav() -> str:
    """v2.2.2 手機／平板快速切換頁面（寬度 ≤1100px 才顯示；電腦用上方選單）：可左右滑動，目前頁面反白。"""
    cur = globals().get("CUR_PAGE", "home") or "home"
    return ("<nav class='kq-qnav'>" + "".join(
        f"<a class='{'on' if k == cur else ''}' href='./{'' if k == 'home' else k}' target='_self'>{ic} {nm}</a>" for k, ic, nm in QNAV) + "</nav>")


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
                  for n, u in brieflib.links(B.tickers[j], BOARDS[j], CH.get(B.tickers[j])))
    out += (f"<div class='kq-muted' style='margin-top:6px'>以上由月營收、每季財報、估值自動整理；"
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
    if ui.ok(getattr(p, "base_from", np.nan)):
        out += (f"<div style='margin-top:6px;color:#fbbf24;font-size:12.5px'>已依今日參考價 {ui.price(p.close)} 換算"
                f"（研究資料收盤 {ui.price(p.base_from)}；除權息或研究資料不是前一個交易日）</div>")
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


# ================================================================== v2.1 同族群比較、國際同業
@st.cache_data(show_spinner=False)
def world_frame(bid) -> pd.DataFrame:
    keys = B.meta.get("world_keys") or []
    if "WORLD" not in B.a or not keys:
        return pd.DataFrame()
    return pd.DataFrame(B.a["WORLD"].astype(float), index=B.days, columns=keys)


def _seg_of(t):
    lab = CH.get(t) or {}
    return lab.get("segment") or lab.get("group") or "其他"


@st.cache_data(show_spinner=False)
def seg_members(bid) -> dict:
    out = {}
    for k, t in enumerate(B.tickers):
        out.setdefault(_seg_of(t), []).append(k)
    return out


def _ret_n(arr, n):
    a = arr[np.isfinite(arr)]
    return float(a[-1] / a[-1 - n] - 1) if len(a) > n and a[-1 - n] > 0 else np.nan


def group_members(j) -> tuple[str, list]:
    """同族群成員（細產業 → 不足時用主題 → 再不足用產業鏈族群），依市值取前 10，本股排第一或在內。"""
    from kosto import peers as peerslib
    t = B.tickers[j]
    seg = _seg_of(t)
    mem = seg_members(BID).get(seg, [])
    th = peerslib.themes_for(CH.get(t) or {})
    if len(mem) < 4 and th:                                       # 細產業太少 → 用主題（例：記憶體＝DRAM、Flash、模組、控制 IC…）
        name, _, pat = next(x for x in peerslib.THEMES if x[0] == th[0][0])
        mem = [TIDX[x] for x in peerslib.tw_members(pat, CH) if x in TIDX]
        seg = f"{name}（主題）"
    if len(mem) < 3:                                              # 仍太少 → 用產業鏈族群
        g = (CH.get(t) or {}).get("group")
        mem = [k for k, x in enumerate(B.tickers) if (CH.get(x) or {}).get("group") == g]
        seg = g or seg
    cap = B.a["CAP"][-1] if "CAP" in B.a else np.zeros(len(B.tickers))
    mem = sorted(set(mem), key=lambda k: -np.nan_to_num(cap[k]))[:10]
    if j not in mem:
        mem = [j] + mem[:9]
    return seg, mem


def group_compare_html(j) -> tuple[str, str]:
    from kosto import brief as brieflib
    seg, mem = group_members(j)
    C = B.a.get("Q_C")
    rows, mrows = [], []
    for k in mem:
        c = C[:, k].astype(float) if C is not None else np.array([])
        fs = brieflib.fin_summary(B, k)
        rv = brieflib.revenue_frame(B, k)
        y3 = float(rv["rev"].iloc[-3:].sum() / rv["rev"].iloc[-15:-12].sum() - 1) if len(rv) >= 15 and rv["rev"].iloc[-15:-12].sum() > 0 else np.nan
        pe = float(B.a["Q_PE"][-1, k]) if "Q_PE" in B.a and np.isfinite(B.a["Q_PE"][-1, k]) and B.a["Q_PE"][-1, k] > 0 else np.nan
        gm, gml = fs.get("gm", np.nan), fs.get("gm_ly", np.nan)
        dgm = (gm - gml) if ui.ok(gm) and ui.ok(gml) else np.nan
        nm = (f"<b style='white-space:nowrap'>{B.tickers[k]} {ui.esc(NAMES[k])}</b>" if k == j
              else f"<span style='white-space:nowrap'>{B.tickers[k]} {ui.esc(NAMES[k])}</span>")
        r1, r5, r20 = _ret_n(c, 1), _ret_n(c, 5), _ret_n(c, 20)
        rows.append([nm, ui.pct(r1, 1, color=True), ui.pct(r5, 1, color=True), ui.pct(r20, 1, color=True),
                     f"{pe:.1f}" if ui.ok(pe) else "–", ui.pct(y3, 0, color=True),
                     f"{fs['eps_ttm']:.2f}" if ui.ok(fs.get("eps_ttm")) else "–",
                     ui.pct(gm, 1, sign=False) + (f"<span class='kq-muted'> {dgm * 100:+.1f}</span>" if ui.ok(dgm) else ""),
                     ui.pct(fs.get("om"), 1, sign=False)])
        pe_s = f"{pe:.1f}" if ui.ok(pe) else "–"
        eps_s = f"{fs['eps_ttm']:.2f}" if ui.ok(fs.get("eps_ttm")) else "–"
        mrows.append([f"{nm}<span class='sub' style='display:block'>本益比 {pe_s}｜EPS {eps_s}｜毛利 {ui.pct(gm, 0, sign=False)}</span>",
                      ui.pct(r1, 1, color=True), ui.pct(r20, 1, color=True), ui.pct(y3, 0, color=True)])
    q = next((brieflib.fin_summary(B, k).get("q") for k in mem if brieflib.fin_summary(B, k)), None)
    head = ["股票", "1日", "5日", "20日", "本益比", "營收年增", "EPS", "毛利率", "營益率"]
    html = ui.resp(ui.table(head, rows, left=(0,)), ui.table(["股票", "1日", "20日", "營收年增"], mrows, left=(0,)))   # 手機：精簡 4 欄表
    if "FIN_Q" not in B.a:
        html += "<div class='kq-muted'>每季財報（EPS、毛利率）會在下一次每日更新後提供。</div>"
    else:
        html += ("<div class='kq-muted' style='margin-top:4px'>營收年增＝近 3 個月合計對去年同期；EPS＝近四季合計；"
                 f"毛利率、營益率＝{ui.esc(q or '最新一季')}單季，灰字為較去年同季增減（百分點）。</div>")
    return seg, html


def peers_html(j) -> str:
    from kosto import peers as peerslib
    lab = CH.get(B.tickers[j]) or {}
    th = peerslib.themes_for(lab)
    W = world_frame(BID)
    keys = [k for k in peerslib.peers_for(lab) if k in W and W[k].notna().sum() > 60]
    if not keys:
        return ("<span class='kq-muted'>" + ("國際同業資料將在下一次每日更新後提供（第一次會補 10 年歷史）。" if th else "此族群沒有明確對應的國際同業。")
                + "</span>")
    r = pd.Series(B.a["R"][:, j].astype(float), index=B.days)
    rows, mrows = [], []
    for k in keys:
        s = W[k].dropna()
        x = s.pct_change().where(lambda v: v.abs() < 0.3)
        z = pd.concat([r, x], axis=1).dropna().tail(250)
        cor = float(z.iloc[:, 0].corr(z.iloc[:, 1])) if len(z) > 60 else np.nan
        a = s.to_numpy()
        r1, r5, r20 = _ret_n(a, 1), _ret_n(a, 5), _ret_n(a, 20)
        rows.append([ui.esc(peerslib.name_of(k)), ui.pct(r1, 1, color=True), ui.pct(r5, 1, color=True), ui.pct(r20, 1, color=True),
                     f"{cor:.2f}" if ui.ok(cor) else "–"])
        mrows.append(ui.mcard(f"<span class='nm'>{ui.esc(peerslib.name_of(k))}</span>", ui.pct(r1, 1, color=True),
                              f"5日 {ui.pct(r5, 1, color=True)}｜20日 {ui.pct(r20, 1, color=True)}", f"連動係數 {cor:.2f}" if ui.ok(cor) else ""))
    out = (f"<div class='kq-muted' style='margin-bottom:4px'>主題：{ui.esc('、'.join(n for n, _ in th))}｜"
           f"資料至 {W.index[-1]:%m/%d}（美股＝前一晚）</div>")
    out += ui.resp(ui.table(["同業", "1日", "5日", "20日", "連動(近一年)"], rows, left=(0,)), "".join(mrows))
    out += ("<div class='kq-muted' style='margin-top:4px'>連動＝近一年此股日報酬與同業（前一晚）報酬的相關係數；"
            "研究顯示同業的消息多半在台股<b>開盤就反映完</b>，適合當開盤前參考，不適合追價（見國際盤「連動研究」）。</div>")
    return out


def _px_digits(x) -> int:
    return 2 if ui.ok(x) and float(x) < 50 else (1 if ui.ok(x) and float(x) < 500 else 0)


def stock_today(j, key: str = "sv"):
    """個股今日走勢（分時＋量）與同族群今日走勢（每列以平盤為中線）。"""
    t = B.tickers[j]
    seg, mem = group_members(j)
    peers = [B.tickers[k] for k in mem if k != j][:7]
    intr = tw_intraday([t] + peers)
    d = intr.get(t)
    i1, i2 = st.columns([1.75, 1])
    with i1:
        st.markdown(f"<div class='kq-h' style='margin:8px 0 4px'><span class='t'>今日走勢｜{ui.esc(NAMES[j])}</span>"
                    "<span class='s'>分時｜虛線＝平盤（昨收），線在上方紅、下方綠；下方為每 2 分鐘成交量（張）</span></div>",
                    unsafe_allow_html=True)
        if d is None:
            st.markdown("<div class='kq-muted'>分時資料暫時無法取得（Yahoo Finance），請稍後重新整理；下方 K 線不受影響。</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(intraday_stats(d, None), unsafe_allow_html=True)
            intraday_chart(d, key=f"{key}_intra_{t}", height=260 if IS_MOBILE else 300, digits=_px_digits(d["prev"]), volume=True,
                           snap=True)
    rows = [(f"{x} {NAMES[TIDX[x]]}", intr.get(x), "") for x in peers if x in TIDX]
    body = mini_rows(rows)
    up = sum(1 for _, x, _ in rows if x is not None and x["last"] > x["prev"])
    dn = sum(1 for _, x, _ in rows if x is not None and x["last"] < x["prev"])
    note = (f"<div class='kq-muted' style='margin-top:6px'>同族群 {up} 漲 {dn} 跌" + ("，與本股同向" if d is not None and (
        (d["last"] > d["prev"] and up > dn) or (d["last"] < d["prev"] and dn > up)) else "") + "。族群共漲共跌多半在開盤就反映（見國際盤「連動研究」）。</div>"
            if body else "")
    i2.markdown(ui.panel(f"同族群今日走勢｜{seg}", (body or "<div class='kq-muted'>分時資料暫時無法取得。</div>") + note),
                unsafe_allow_html=True)


def stock_view(j, h="mid", key="sv"):
    t = B.tickers[j]
    c = CH.get(t, {})
    close, ch = last_close(j)
    live_html = ""
    bq = None
    if market_phase() in ("盤前試撮", "盤中", "盤後"):                 # 收盤後到晚上研究更新前也顯示今天的收盤
        q, _ = quotes_for([t])
        bq = now_quote(t, tw_intraday([t]).get(t), q)
        if bq["live"]:
            nm = {"bid": "委買", "ask": "委賣", "yahoo": "分時"}.get(bq["src"], "即時" if is_live_time() else "今日收盤")
            tm = q.loc[t, "time"] if bq["src"] not in ("yahoo",) and t in q.index else "可能延遲 20 分"
            lock = {"up": " <span class='kq-lock up'>漲停</span>", "down": " <span class='kq-lock dn'>跌停</span>"}.get(bq["lock"], "")
            live_html = (f"<span class='kq-item'>{nm}<b>{ui.price(bq['px'])}</b> {ui.chg(bq['chg'])}{lock}"
                         f" <span class='kq-muted'>{ui.esc(tm)}</span></span>")
    themes = "".join(ui.tag(x, "g") for x in c.get("themes", []))
    st.markdown(f"<div class='kq-bar'><span style='font-size:20px;font-weight:700;color:var(--strong)'>{ui.esc(NAMES[j])}</span>"
                f"<span class='kq-mono' style='color:var(--accent);font-size:18px'>{t}</span>"
                f"<span class='kq-item'>{BOARD_ZH.get(BOARDS[j], '')}</span>"
                f"<span class='kq-item'>{RDAY:%m/%d} 收盤<b>{ui.price(close)}</b> {ui.chg(ch)}</span>{live_html}"
                f"<span>{ui.tag(c.get('group', SECTORS[j]), 'gold')}<span class='kq-muted'> {ui.esc(c.get('segment', ''))}</span> {themes}</span>"
                "</div>", unsafe_allow_html=True)
    if not HAS_Q:
        st.info("看盤資料包（K 線、籌碼）會在更新到 v1.6 後的第一次每日研究產生，約 1 小時後重新整理即可。")
        return
    p = plan_today(j, h, bq)
    c1, c2, c3 = st.columns([1.05, 1.25, 1])
    c1.markdown(ui.panel(f"交易計畫（{HZ[h]}）", plan_html(p, h)), unsafe_allow_html=True)
    c2.markdown(ui.panel("籌碼動向｜三大法人・融資券・大戶", chips_html(chips(BID, j))), unsafe_allow_html=True)
    c3.markdown(ui.panel("基本面快照與短評", brief_html(j) + "<div style='margin-top:8px'></div>" + fundamentals_html(j)),
                unsafe_allow_html=True)
    stock_today(j, key)
    nw1, nw2 = st.columns([1.6, 1])
    from urllib.parse import quote as _q
    gq = f"https://news.google.com/search?q={_q(NAMES[j] + ' ' + t)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    nw1.markdown(ui.panel(f"最新新聞｜{ui.esc(NAMES[j])}（Google 新聞，近 3 天）", headlines_html(
        news_headlines(f"{NAMES[j]} {t}", 6, 3),
        empty=f"目前抓不到新聞（網路或來源暫時無法連線），可直接開 <a href='{gq}' target='_blank' rel='noopener'>Google 新聞搜尋</a>。")),
        unsafe_allow_html=True)
    nw2.markdown(ui.panel("今天的重大訊息", material_html([t])), unsafe_allow_html=True)
    st.markdown(ui.panel(f"產業鏈位置｜{ui.esc(NAMES[j])}", chain_position_html(t)), unsafe_allow_html=True)
    eg = stock_egg_html(j)
    if eg:
        st.markdown(ui.panel(f"科斯托蘭尼雞蛋階段｜{ui.esc(NAMES[j])}", eg), unsafe_allow_html=True)
    g1, g2 = st.columns([1.6, 1])
    seg, grp_html = group_compare_html(j)
    g1.markdown(ui.panel(f"同族群比較｜{seg}", grp_html), unsafe_allow_html=True)
    g2.markdown(ui.panel("國際同業｜美股為前一晚收盤", peers_html(j)), unsafe_allow_html=True)
    per, rng, mas, inds = chart_controls(key, [5, 20, 60], IND_OPTS, ["boll", "volume", "kd", "macd", "inst"], 1)
    from kosto.charts import kline
    ind = ind_frame(BID, j, per)
    bars = max(20, RANGE[rng] // {"D": 1, "W": 5, "M": 21}[per])
    pn = [x for x in inds if x != "boll"]
    fig = kline(ind, dark=DARK, mas=tuple(mas), boll="boll" in inds, panels=pn,
                plan=p if per == "D" else None, bars=bars, height=(300 + 95 * len(pn)) if IS_MOBILE else None)
    if IS_MOBILE:                                  # 觸控：單指拖曳用來捲動頁面，不要拖動圖表
        fig.update_layout(dragmode=False)
    st.plotly_chart(fig, width="stretch", theme=None, key=f"{key}_fig",
                    config={"displaylogo": False, "scrollZoom": not IS_MOBILE, "displayModeBar": not IS_MOBILE,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
    st.caption("圖上藍線＝買進區上緣、青點線＝拉回買點、橘虛線＝追價上限、紅虛線＝警戒價。大戶持股為集保每週資料（每週累積）。"
               "券商分點「主力進出」屬付費資料，本系統以「大戶持股＋法人集中度」呈現主力動向。")


# ================================================================== page: 今日選股
def phase_stats(h):
    t = (RES or {}).get("phases")
    if t is None or not len(t):
        return None
    from kosto.phases import summary
    sm = summary(t)
    return sm.loc[h].to_dict() if h in sm.index and sm.loc[h, "起點數"] >= 3 else None


def BUF(h) -> int:
    """續抱門檻：排名在前 round(K×續抱倍數) 名內就不賣（與回測相同）。"""
    return max(KH(h), int(round(KH(h) * float(CFG0["horizons"][h]["hysteresis"]))))


def pick_items(h, lt) -> list[dict]:
    out, prev = [], set(prev_picks(h))
    for n, p in enumerate(lt["picks"], 1):
        t = p["ticker"]
        if t not in TIDX:
            continue
        j = TIDX[t]
        close, ch = last_close(j)
        out.append({"n": n, "t": t, "j": j, "close": close, "ch": ch, "pl": plan_of(j, h), "cs": chips(BID, j) if HAS_Q else {},
                    "kept": bool(p.get("kept") or (t in prev)), "missing": list(p.get("missing") or []),
                    "reasons": "、".join(clean_reason(x) for x in p.get("reasons", [])[:3])})
    ws = planlib.stock_weights(len(out), h, CFG0["horizons"][h].get("weighting"))        # v3.0 名次加權：與回測相同的資金比重
    for it, w_ in zip(out, ws):
        it["w"] = float(w_)
    return out


CARD_CHART = {"today": "今日分時", "m1": "近 1 月", "m3": "近 3 月"}


def card_chart(j, mode: str, d=None) -> str:
    """選股卡片的走勢圖：今日分時（平盤虛線、紅漲綠跌）或近 1／3 月（期初價虛線）；分時取不到時改顯示近 3 月。"""
    if mode == "today" and d is not None:
        html = ui.intraday_mini(d, title=intraday_title(d))
        if html:
            return html
        mode = "m3"
    if not HAS_Q:
        return ""
    n = 21 if mode == "m1" else 63
    df = raw_frame(BID, j)
    lab = {"m1": "近1月", "m3": "近3月"}.get(mode, "近3月")
    return ui.trend_mini(df["close"].tail(n + 1).tolist(), period=lab) if len(df) > 5 else ""


def pick_card(it, mode: str = "m3", d=None, q=None) -> str:
    t, j, pl = it["t"], it["j"], it["pl"]
    c = CH.get(t, {})
    status = ui.badge("續抱", "#5b21b6") if it["kept"] else ui.badge("新進・" + ("今天買" if exec_day().startswith("今天") else "下個交易日買"), "#1d4ed8")
    tone = ui.tone_badge(it["cs"].get("tone", "中性")) if it["cs"] else ""
    # v2.3.1 現價：證交所即時 → 今天的分時 → 研究資料收盤（標日期）；漲跌停另外標示
    bq = IDY.best_quote(q.loc[t].to_dict() if q is not None and len(q) and t in q.index else None, d, it["close"], it["ch"], RDAY)
    px, ch = bq["px"], bq["chg"]
    if pl is not None and bq["live"] and ui.ok(bq.get("prev")):
        pl = planlib.rebase(pl, bq["prev"])
    when = (IDY.status(d) if d is not None and bq["src"] != "close" and d["date"] > RDAY.date() else
            ("盤中" if bq["live"] else "")) or bq["label"]
    if bq["src"] in ("bid", "ask"):
        when += "・" + bq["label"]
    lock = {"up": "<span class='kq-lock up'>漲停</span>", "down": "<span class='kq-lock dn'>跌停</span>"}.get(bq["lock"], "")
    tag = (lock + "<span class='kq-sub' style='display:block;font-size:10.5px;font-weight:400'>" + ui.esc(when) + "</span>") if (when or lock) else ""
    lad = ui.ladder(px, pl.buy_low, pl.buy_high, pl.chase, pl.alert, compact=True) if pl else ""
    # v3.14（同儕評審第 5 章）：「模型排名」（值不值得買）與「價格適合度」（現在這個價位適不適合進場）分開標示
    _, fit_txt, fit_col = planlib.price_fit(px, pl, px if bq.get("lock") == "up" else np.nan)
    lad += (f"<div class='kq-fit' style='font-size:12px;margin:2px 0 4px'><span class='kq-sub'>模型排名</span> <b>第 {it['n']} 名</b>{ui.evidence('tested')}"
            f"<span class='kq-sub'>｜價格適合度</span> <b style='color:{fit_col}'>{ui.esc(fit_txt)}</b>{ui.evidence('ref')}</div>")
    if it.get("missing"):                                          # v3.15 缺值透明（評審二 2.8）
        lad += (f"<div style='font-size:11px;color:#b45309;margin:0 0 4px'>缺值：{ui.esc('、'.join(it['missing'][:3]))}"
                "（排名時當成中性）</div>")
    themes = "".join(ui.tag(x, "g") for x in c.get("themes", [])[:2])
    return (f"<div class='kq-pcard'><div class='hd'><div><span class='rk'>{it['n']}</span><span class='kq-cd'>{t}</span>"
            f"<span class='nm'>{ui.esc(NAMES[j])}</span><span class='kq-sub'> {BOARD_ZH.get(BOARDS[j], '')}</span></div>"
            f"<div class='px'>{ui.price(px)}<small>{ui.chg(ch)}</small>{tag}</div></div>"
            f"<div class='grp'>{ui.tag(c.get('group', SECTORS[j]), 'gold')}<span class='kq-sub'>{ui.esc(c.get('segment', ''))}</span>{themes}</div>"
            f"<div class='sp'>{card_chart(j, mode, d)}</div>{lad}"
            f"<div class='ft'>{status} {tone}" + (f"<span class='kq-tag' title='名次加權：分數越高放越多（與回測相同）'>資金 {it['w'] * 100:.1f}%</span>"
                                                    if it.get("w") else "") + f"<span class='why'>{ui.esc(it['reasons'])}</span></div></div>")


def render_picks(items, mode: str = "m3") -> str:
    tks = [it["t"] for it in items]
    ph = market_phase()
    # 交易日盤中／盤後（研究資料還沒更新前）：不論圖表選哪一種，價格都用今天的，避免顯示前一個交易日的漲跌
    intr = tw_intraday(tks) if mode == "today" or ph in ("盤前試撮", "盤中", "盤後") else {}
    q = quotes_for(tks)[0] if _tw_live() else None
    return concentration_html(items) + "<div class='kq-grid'>" + "".join(pick_card(it, mode, intr.get(it["t"]), q) for it in items) + "</div>"


def concentration_html(items) -> str:
    """v3.14 組合集中度（同儕評審 1.4）：最大單檔、有效檔數、最大族群。"""
    if not items or not all(it.get("w") for it in items):
        return ""
    cc = planlib.concentration([it["w"] for it in items], [CH.get(it["t"], {}).get("group", SECTORS[it["j"]]) for it in items])
    if not cc:
        return ""
    warn = cc["top_group_w"] >= 0.5
    sty = " style='color:#b45309'" if warn else ""
    return (f"<div class='kq-muted' style='margin:2px 0 6px;font-size:12.5px'>組合集中度：最大單檔 {cc['top_w']:.1%}｜有效檔數 {cc['eff_n']:.1f}／{cc['n']}"
            f"｜{cc['n_groups']} 個族群，最多的是 <b{sty}>{ui.esc(cc['top_group'])}</b>"
            f" {cc['top_group_n']} 檔、資金 {cc['top_group_w']:.0%}" + ("（同族群過半，波動會一起放大）" if warn else "") + "</div>")


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
    style = {"exit": ("預計賣出", "#dc2626"), "watch": ("觀察", "#d97706"), "keep": ("續抱", "#16a34a"), "sold": (f"{exec_day().split(' ')[0]}開盤賣出", "#991b1b")}
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
            x = dict(x, why=f"本期這份資金已換出（{exec_day().split(' ')[0]}開盤賣出）；其餘 {x['slots']} 份待各自檢查日。{x['why']}")
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
            f"不設固定停利停損，所以賣點會隨排名浮動。排名走勢＝最近 5 次（每 5 個交易日）的名次；持有報酬算到 {RDAY:%m/%d} 收盤。</div>")
    if not trs:
        return head + "<div class='kq-muted'>目前沒有持股紀錄。</div>"
    desk = ui.table(["股票", "進場日／價", "持有報酬", "排名走勢", "下次檢查日", "判斷"], trs, left=(0, 1, 4, 5))
    return head + ui.resp(desk, "".join(mc))


def us_session_label(q: dict) -> str:
    """美股報價的時間說明：美股盤中（台北約 22:30～05:00）＝「盤中即時」，其餘＝「MM/DD（週X）收盤，下一個台股交易日開盤參考」。"""
    if not q.get("live"):
        return "前一個美股交易日收盤"
    now = pd.Timestamp.now(tz="Asia/Taipei")
    hm = now.hour * 60 + now.minute
    us_open = (hm >= 22 * 60 + 30 and now.weekday() < 5) or (hm < 5 * 60 and 0 < now.weekday() <= 5)
    if us_open:
        return f"美股盤中 即時 {q.get('time', '')}"
    bd = q.get("bar_date")
    if bd is not None and not pd.isna(bd):
        bd = pd.Timestamp(bd)
        return f"{bd:%m/%d}（週{'一二三四五六日'[bd.weekday()]}）美股收盤 → 下一個台股交易日開盤參考"
    return f"即時 {q.get('time', '')}"


def preopen_box() -> str:
    snap = world_snapshot()
    if "SOX" not in snap:
        return ""
    sox = snap["SOX"]
    lab, row, adv = overnight_read(sox["chg"])
    if not adv:
        return ""
    when = us_session_label(sox)
    nq = snap.get("NQF")
    nq_txt = f"｜那指期 {ui.chg(nq['chg'])}" if nq else ""
    return (f"<div class='kq-verdict'><b>開盤前看美股</b>　費半 {ui.chg(sox['chg'])}（{ui.esc(when)}）{nq_txt}"
            f"　<span class='kq-tag'>{ui.esc(lab)}</span><br><span class='kq-muted'>{row.get('stat', '')}</span><br>{ui.esc(adv)}</div>")


def correction_banner():
    """v2.0／v2.2：大盤修正（距高點 ≥10%）或觸發急跌條件時，提醒先看盤勢儀表板。"""
    df, _, sto = insight_frame(BID)
    if not sto:
        return
    cur = crash_now(df) if not df.empty else {}
    on = [t for _, t, ok_ in cur.get("flags", []) if ok_]
    if not on and not sto.get("corr"):
        return
    col = "#f0464b" if on or sto["score"] <= 2 else "#e9a23b"
    head = ("🚨 急跌警報：" + "｜".join(on) + "。") if on else f"大盤修正中：距近半年高點 {ui.pct(sto['dd'], 1)}。"
    st.markdown(f"<div class='kq-warn' style='border-color:{col}'>{ui.esc(head)}止穩訊號 {sto['score']}/8，"
                f"{'新資金建議分批、小量，不要在恐慌時殺低' if sto['score'] <= 2 else '可開始分批布局'}；"
                "<a href='./insight' target='_self'>看盤勢儀表板（急跌應變、分批計畫、反彈候選）→</a></div>", unsafe_allow_html=True)


def picks_status_banner():
    """v2.1：17:20 初版／21:40 完整版的名單狀態。"""
    ds = (RES or {}).get("data_status") or {}
    if not ds or ds.get("complete"):
        return
    col = "#16a35f" if ds.get("final") else "#e9a23b"
    used = ds.get("margin_factors") or {}
    extra = ("；用到融資因子的期間：" + "、".join(f"{HZ.get(h, h)}（{'、'.join(FZH.get(f, f) for f in fs)}）" for h, fs in used.items())) if used else ""
    st.markdown(f"<div class='kq-warn' style='border-color:{col}'>{'✅ 名單已定版' if ds.get('final') else '🕒 初版名單'}：{ui.esc(ds.get('text', ''))}{ui.esc(extra)}</div>",
                unsafe_allow_html=True)


def page_picks():
    header()
    data_strip()
    picks_status_banner()
    correction_banner()
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
    wmap = {it["t"]: it.get("w") for it in its}
    kept = [it["t"] for it in its if it["kept"]]
    chip = lambda t, x="": ui.schip(t, NAMES[TIDX[t]], x)                     # noqa: E731
    st.markdown(ui.action_board(f"{exec_day()}開盤要做的事（{HZ[h]}）", f"名單日 {lt['date']} 收盤後產生｜點股票看診斷", [
        ("買進", "#3b82f6", [chip(t, f"資金 {wmap[t] * 100:.1f}%" if wmap.get(t) else "") for t in new if t in TIDX], "今天沒有新買進"),
        ("賣出", "#f0464b", [chip(t) for t in sold if t in TIDX], "今天沒有要賣的"),
        ("續抱", "#8b5cf6", [chip(t, f"{wmap[t] * 100:.1f}%" if wmap.get(t) else "") for t in kept if t in TIDX], "—")]),
        unsafe_allow_html=True)
    secs = pd.Series([SECTORS[TIDX[t]] for t in now_set if t in TIDX]).value_counts()
    if len(secs) and secs.iloc[0] >= 3:                    # v2.2 產業集中提醒（研究：同產業 ≤2～3 檔可降低回撤，但報酬也略降）
        st.markdown(f"<div class='kq-muted' style='margin:-4px 0 8px'>⚠ 名單中有 {secs.iloc[0]} 檔{ui.esc(secs.index[0])}"
                    f"（占 {secs.iloc[0] / max(len(now_set), 1):.0%}），同族群容易一起漲跌；想降低波動可只買其中 2 檔，"
                    "其餘名額改用下一名（見「盤勢儀表板 → 特別時期選股」）。</div>", unsafe_allow_html=True)
    hold = holdings_of(BID, h)
    n_sell = sum(1 for x in hold if x["status"] == "exit")
    tabs = st.tabs([f"📋 推薦 {len(its) or KH(h)} 檔", f"🧭 持股追蹤（何時賣）{'・' + str(n_sell) + ' 檔預計賣出' if n_sell else ''}",
                    "🔎 候補與排除"])
    with tabs[0]:
        mode = st.segmented_control("卡片走勢圖", list(CARD_CHART), format_func=CARD_CHART.get, default="today",
                                    key="pick_chart") or "today"
        st.markdown(f"<div class='kq-muted' style='margin:2px 0 8px'>市場階段 {ui.esc(STAGE_ZH.get(lt.get('market_stage'), '').split(' ')[0])}"
                    "｜走勢圖：虛線＝" + ("平盤（昨收），線在虛線上方紅、下方綠" if mode == "today" else "期初價，高於期初紅、低於綠")
                    + "｜價位尺：<span style='color:#f0464b'>紅＝警戒價</span>、<span style='color:#5aa9ff'>藍＝建議買進區</span>、"
                    "<span style='color:#e9b949'>金＝可小量追</span>、白線＝最新價</div>", unsafe_allow_html=True)
        if its:
            st.markdown(render_picks(its, mode), unsafe_allow_html=True)
        else:
            st.info("目前沒有符合條件的標的。")
    with tabs[1]:
        st.markdown(holdings_board(h, lt), unsafe_allow_html=True)
    with tabs[2]:
        try:
            rk = ranks(BID, h)
            alt = rk[rk["valid"] & (rk["rank"] > 0)].sort_values("rank")
            alt = alt[~alt.index.isin(now_set) & (alt["rank"] <= BUF(h))]
            rows2, cards2 = [], []
            for t, r in alt.iterrows():
                cl, ch = last_close(TIDX[t])
                why = "、".join(clean_reason(x) for x in r["reasons"][:3])
                badge_ = " " + ui.tag("新資金可買", "g") if r["rank"] <= KH(h) else ""
                rows2.append([f"{int(r['rank'])}{badge_}", stock_cell(t), chain_html(t, themes=False), ui.price(cl), ui.chg(ch),
                              f"<span class='sub'>{ui.esc(why)}</span>"])
                cards2.append(ui.mcard(f"<span class='rk'>#{int(r['rank'])}</span><span class='kq-cd'>{t}</span>"
                                       f"<span class='nm'>{ui.esc(NAMES[TIDX[t]])}</span>{badge_}",
                                       f"{ui.price(cl)}<small>{ui.chg(ch)}</small>", "", ui.esc(why)))
            st.markdown(f"<div class='kq-muted' style='margin:2px 0 8px'>排名在前 {BUF(h)} 名、但不在推薦名單：已持有可續抱；"
                        f"標「新資金可買」者排名在前 {KH(h)}。</div>", unsafe_allow_html=True)
            st.markdown(ui.resp(ui.table(["排名", "股票", "族群", "收盤", "漲跌", "主要加分"], rows2, left=(1, 2, 5)), "".join(cards2)),
                        unsafe_allow_html=True)
        except Exception as e:  # noqa: BLE001
            st.caption(f"無法計算候補：{e}")
        if lt.get("blocked"):
            st.markdown(ui.section("高分但被風控規則排除"), unsafe_allow_html=True)
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
        market_today(compact=True)
        intr = tw_intraday(uni)
        rows, chances, mcards = [], [], []
        for t in uni:
            j = TIDX[t]
            hs = member.get(t, [])
            h = "mid" if "mid" in hs or not hs else hs[0]              # 價位以中線計畫為準（不在中線名單時用所在名單）
            di = intr.get(t)
            bq = now_quote(t, di, q)
            pl = plan_today(j, h, bq)
            live, px, chg = bq["live"], bq["px"], bq["chg"]
            lu = px if bq["lock"] == "up" else (float(q.loc[t, "limit_up"]) if bq["src"] not in ("yahoo", "close") and "limit_up" in q.columns
                                                 else np.nan)
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
                code, txt = "none", ("開盤後判斷" if ph in ("盤前", "盤前試撮") else f"{bq['label']}參考")
            if code in ("buy", "dip"):
                chances.append(t)
            vol = f"{bq['volume']:,.0f}" if live and ui.ok(bq["volume"]) else "–"
            src_tag = quote_tags(bq)
            rows.append([stock_cell(t), chain_html(t, themes=False), f"<b>{ui.price(px)}</b>{src_tag}", ui.chg(chg),
                         ui.intraday_mini(di, h=34, fluid=False, w=120, labels=False) or "–",
                         (f"{ui.price(bq['open'])}/{ui.price(bq['high'])}/{ui.price(bq['low'])}" if live and ui.ok(bq["open"]) else "–"),
                         vol, f"{ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}" if pl else "–",
                         ui.price(pl.chase) if pl else "–", ui.price(pl.alert) if pl else "–",
                         ui.badge(txt, planlib.STATUS_STYLE[code]),
                         " ".join(ui.tag(HZ[x]) for x in hs) + (ui.tag("持股", "g") if t in holds else "")])
            mcards.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[j])}</span> "
                                   + " ".join(ui.tag(HZ[x]) for x in hs) + (ui.tag("持股", "g") if t in holds else ""),
                                   f"{ui.price(px)}<small>{ui.chg(chg)}</small>{quote_tags(bq)}",
                                   ui.intraday_mini(di, h=44, labels=False)
                                   + (ui.ladder(px, pl.buy_low, pl.buy_high, pl.chase, pl.alert, compact=True) if pl else ""),
                                   ui.badge(txt, planlib.STATUS_STYLE[code]) + (f" <span>量 {vol} 張</span>" if vol != "–" else "")))
        tm = q["time"].iloc[0] if len(q) else ("分時報價（證交所即時報價暫停，可能延遲約 20 分鐘）" if any(
            d_ is not None and d_["date"] > RDAY.date() for d_ in intr.values()) else f"{RDAY:%m/%d} 收盤價")
        st.markdown(f"<div class='kq-muted'>更新 {pd.Timestamp.now(tz='Asia/Taipei'):%H:%M:%S}｜報價時間 {ui.esc(tm)}｜{ph}</div>",
                    unsafe_allow_html=True)
        st.markdown(ui.resp(ui.table(["股票", "族群", "現價", "漲跌", "今日走勢（虛線＝平盤）", "開/高/低", "量(張)", "買進區", "追價上限",
                                      "警戒價", "狀態", "名單"], rows, left=(0, 1, 10, 11)), "".join(mcards)), unsafe_allow_html=True)
        st.markdown("<div class='kq-muted' style='margin-top:6px'>" + "　".join(
            f"{ui.badge(k_, planlib.STATUS_STYLE[c_])} {ui.esc(planlib.STATUS_HELP[c_].split('→')[-1].strip())}"
            for c_, k_ in (("buy", "可進場"), ("dip", "拉回買點"), ("wait", "等拉回"), ("warn", "跌破警戒"), ("lock", "漲停不追")))
            + f"<br>價位（買進區、追價上限、警戒價）以前一個交易日（{RDAY:%m/%d}）收盤計算，同時在多個名單時以中線為準；"
              "除權息日或研究資料不是前一個交易日時，自動依證交所今日參考價等比例換算。</div>", unsafe_allow_html=True)
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
                bq_h = now_quote(t, intr.get(t), q)
                pl = plan_today(TIDX[t], "mid", bq_h)
                px = bq_h["px"]
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
- 系統每個交易日 **17:23** 先用收盤、法人、本益比出初版（約 18:30 完成），**21:40** 補上融資券出完整版（約 23:00 完成）；隔天開盤前打開「今日選股」即可看到明日名單（目前名單日：**{latest('short')['date']}**）。
- **明日要買**：名單中標「新進」的股票；**續抱**：標「續抱」或仍在續抱門檻內；**要賣**：手上有、但已跌出續抱門檻（看「今日選股」的持股追蹤或下方「持股健檢」）。
- 隔日 **開盤前** 先看「國際盤」：昨夜費半大跌 → 台股多半開低後盤中小幅收斂，不要開盤殺低；大漲 → 開高後常回落，不追開盤。
- 以「建議買進區上緣」以內掛單；**開盤跳空超過追價上限就不追**，盤中回到買進區再分批買。
- 13:25 前仍未成交 → 當天放棄，下一個交易日再看；不要用市價追。""")


# ================================================================== page: 個股診斷
def verdict_for(t, h, live_px=np.nan, bq=None):
    """回傳 (等級, 標題, 細節)；等級 buy/hold/watch/no/na。"""
    j = TIDX[t]
    try:
        rk = ranks(BID, h)
    except Exception:  # noqa: BLE001
        rk = pd.DataFrame()
    picks = [p["ticker"] for p in latest(h)["picks"]]
    pl = plan_today(j, h, bq)
    px = live_px if ui.ok(live_px) else last_close(j)[0]
    lu = px if bq and bq.get("lock") == "up" else np.nan
    if t not in rk.index:
        return "na", "不在選股範圍", "成交量或規模未進入模型股票池，模型不評分"
    r = rk.loc[t]
    n = int(r["n_valid"]) if ui.ok(r["n_valid"]) else 0
    top = f"前 {max(1, int(round((1 - r['pct']) * 100)))}%"
    if not r["valid"]:
        return "no", "暫不建議（風控排除）", f"原因：{r['why']}｜分數位於股票池{top}"
    rank = int(r["rank"])
    if t in picks or rank <= KH(h):
        code, txt = planlib.live_status(px, pl, "pick", lu)
        note = "" if t in picks else f"（排名前 {KH(h)}；本期名單保留了續抱股，空手者可列為首選候補）"
        if code in ("buy", "dip"):
            return "buy", f"可以買進（第 {rank} 名）", planlib.STATUS_HELP[code] + note
        if code == "wait":
            return "watch", f"推薦但先別追（第 {rank} 名）", planlib.STATUS_HELP["wait"] + note
        if code == "warn":
            return "watch", f"推薦但股價轉弱（第 {rank} 名）", planlib.STATUS_HELP["warn"] + note
        if code == "lock":
            return "watch", f"推薦但今天漲停（第 {rank} 名）", planlib.STATUS_HELP["lock"] + note
        return "buy", f"推薦（第 {rank} 名）", f"{txt}{note}"
    if rank <= BUF(h):
        return "hold", f"持有可續抱（第 {rank}/{n} 名）", f"未進前 {KH(h)} 名：空手者不新買，等它進入推薦名單"
    return "no", f"不建議（第 {rank}/{n} 名）", f"分數位於股票池{top}；持有者可在下次檢查日換股"


VCOLOR = {"buy": "#1d4ed8", "hold": "#5b21b6", "watch": "#d97706", "no": "#475569", "na": "#475569"}
MODEL_OF = {"buy": "pick", "watch": "pick", "hold": "keep", "no": "out", "na": "na"}


def position_section(t: str, j: int, vs: dict, bq, live_px):
    """v3.11 個股診斷：空手怎麼做／持有怎麼做（可輸入平均成本、張數）。"""
    order = ["pick", "keep", "out", "na"]
    model = min((MODEL_OF[v[0]] for v in vs.values()), key=order.index)
    h_use = next((h for h in ("mid", "short", "long") if MODEL_OF[vs[h][0]] == model), "mid")
    pl = plan_today(j, h_use, bq)
    px = live_px if ui.ok(live_px) else last_close(j)[0]
    lu = px if bq and bq.get("lock") == "up" else np.nan
    role = {"pick": "pick", "keep": "keep"}.get(model, "none")
    code, txt = planlib.live_status(px, pl, role, lu)
    st.markdown(ui.section("怎麼做：空手 vs 持有", f"依 {HZ[h_use]}模型與交易計畫；現價 {ui.price(px)}"
                           + ("（即時）" if ui.ok(live_px) else f"（{RDAY:%m/%d} 收盤）") + ui.evidence("ref")), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    cost = c1.number_input("我的平均成本（元，選填）", min_value=0.0, value=float(st.session_state.get(f"kq_cost_{t}", 0.0)),
                           step=0.05, format="%.2f", key=f"cost_in_{t}")
    lots_ = c2.number_input("持有張數（選填）", min_value=0, value=int(st.session_state.get(f"kq_lots_{t}", 0)), step=1, key=f"lots_in_{t}")
    st.session_state[f"kq_cost_{t}"], st.session_state[f"kq_lots_{t}"] = cost, lots_
    if cost > 0 and ui.ok(px):
        pnl = px / cost - 1
        amt = (px - cost) * lots_ * 1000 if lots_ else None
        c3.markdown(f"<div class='kq-panel' style='padding:8px 12px'>損益 <b style='color:{ui.color_of(pnl)}'>{pnl:+.1%}</b>"
                    + (f"｜<b style='color:{ui.color_of(pnl)}'>{amt:+,.0f}</b> 元" if amt is not None else "")
                    + (f"<br><span class='kq-muted'>距警戒價 {pl.alert / px - 1:+.1%}</span>" if pl else "") + "</div>", unsafe_allow_html=True)
    hcode, head, notes = planlib.holding_advice(px, cost if cost > 0 else np.nan, pl, model, code)
    left = (f"{ui.badge(txt, planlib.STATUS_STYLE.get(code, '#6b7280'))}<div style='margin-top:6px'>{ui.esc(planlib.entry_advice(code, pl))}</div>"
            + (ui.ladder(px, pl.buy_low, pl.buy_high, pl.chase, pl.alert, compact=True) if pl else ""))
    right = (f"{ui.badge(head, planlib.HOLD_STYLE[hcode])}" + ui.bullets([ui.esc(x) for x in notes])
             + ("" if cost > 0 else "<div class='kq-muted'>輸入平均成本，會加上損益與停利／停損提醒。</div>"))
    a, b = st.columns(2)
    a.markdown(ui.panel("🟢 空手的人", left), unsafe_allow_html=True)
    b.markdown(ui.panel("🟣 已經持有的人", right), unsafe_allow_html=True)
    st.caption("第一層照模型規則（回測驗證過：排名在續抱門檻內就續抱、跌出門檻在下次檢查日換股；機械式停損在 v1.1 測試中拖累績效，所以不設）。"
               "成本、損益、停利停損那一層是資金控管紀律（未經回測）。成本只在這次開啟的頁面裡有效（關掉就清除），不會存到 GitHub。")


def parse_codes(txt: str) -> tuple[list[str], list[str]]:
    """v3.8 多檔對照：從一段文字找出股票（代號或公司名），回傳（找到的代號, 找不到的字）。"""
    import re
    found, miss = [], []
    name_idx = {str(n): t for t, n in zip(B.tickers, NAMES)}
    for tok in [x for x in re.split(r"[\s,，、;；/（）()]+", txt) if x]:
        code = re.match(r"^(\d{4,6}[A-Z]?)", tok)
        t = code.group(1) if code and code.group(1) in TIDX else name_idx.get(tok)
        if t is None and len(tok) >= 2 and not tok.isdigit():
            t = next((tt for nm, tt in name_idx.items() if tok in nm), None)
        if t and t not in found:
            found.append(t)
        elif not t and tok not in miss:
            miss.append(tok)
    return found[:40], miss


def fwd_fund(t: str) -> str:
    """v3.9 季報反推的前瞻基本面（只供參考、不在模型裡）：淨利率一年變化、最新一季獲利上修、成長合理價——
    資料包存的是在全市場（可交易股票）中的百分位；舊資料包沒有這三個因子就空白。"""
    fs = list(B.factors)
    if "f_margin_chg" not in fs or t not in TIDX:
        return ""
    U = B.a["U"][-1, TIDX[t]]
    out = []
    for k, lab in (("f_margin_chg", "淨利率改善"), ("f_eps_rev", "獲利上修"), ("f_garp", "成長合理價")):
        u = int(U[fs.index(k)])
        if u > 0:
            out.append(f"{lab} 前 {max(1, round(100 - (u - 1) / 65534 * 100))}%")
    return "｜".join(out)


def multi_check_html(txt: str) -> str:
    """v3.8 多檔對照：貼上雜誌或別人的名單 → 每檔在三個模型的判斷、題材、是否在反彈候選。"""
    from kosto import supply as S
    ts, miss = parse_codes(txt)
    if not ts:
        return "<div class='kq-muted'>沒有找到股票代號或名稱。</div>"
    rb = {x["ticker"]: k for k, x in enumerate(B.meta.get("rebound") or [], 1)}
    order = ["buy", "watch", "hold", "low", "risk", "na"]
    rows, mrows, cnt = [], [], {k: 0 for k in order}
    for t in ts:
        j = TIDX[t]
        vs = {h: verdict_for(t, h) for h in HZ}
        cat = [("risk" if v[1].startswith("暫不建議") else "low") if v[0] == "no" else v[0] for v in vs.values()]
        cnt[min(cat, key=order.index)] += 1
        lab = CH.get(t) or chainlib.label(t, SECTORS[j])
        grp = lab.get("segment") or lab.get("group") or ""
        th = "、".join(S.themes_of(t)[:2])
        px, chg = last_close(j)
        why = lambda v: (v[2].split("｜")[0].replace("原因：", "") if v[1].startswith("暫不建議") else "")   # noqa: E731
        cells = [ui.badge(vs[h][1], VCOLOR[vs[h][0]]) + (f"<div class='kq-muted' style='font-size:11px'>{ui.esc(why(vs[h]))}</div>"
                                                          if why(vs[h]) else "") for h in HZ]
        reb = f"第 {rb[t]} 名" if t in rb else "–"
        ff = fwd_fund(t)
        rows.append([stock_cell(t), ui.esc(grp) + (f"<div class='kq-muted' style='font-size:12px'>{ui.esc(th)}</div>" if th else "")
                     + (f"<div class='kq-muted' style='font-size:12px'>{ui.esc(ff)}</div>" if ff else ""),
                     *cells, reb, ui.num(px) + f"<div style='font-size:12px'>{ui.pct(chg, 1, color=True)}</div>"])
        mrows.append(ui.mcard(f"<span class='nm'>{t} {ui.esc(NAMES[j])}</span>", ui.num(px),
                              "".join(f"<div style='margin:2px 0'>{HZ[h]} {ui.badge(vs[h][1], VCOLOR[vs[h][0]])}"
                                      + (f" <span class='kq-muted' style='font-size:11px'>{ui.esc(why(vs[h]))}</span>" if why(vs[h]) else "")
                                      + "</div>" for h in HZ),
                              ui.esc(grp) + (f"｜{ui.esc(th)}" if th else "") + (f"｜反彈候選 {reb}" if t in rb else "")
                              + (f"<br>{ui.esc(ff)}" if ff else "")))
    head = (f"<div class='kq-verdict'>{len(ts)} 檔（取三個期間中最好的判斷）：<b>{cnt['buy']}</b> 檔模型現在推薦、"
            f"{cnt['watch']} 檔推薦但先等價位、{cnt['hold']} 檔持有可續抱、{cnt['low']} 檔排名不夠前面、{cnt['risk']} 檔被風控排除、"
            f"{cnt['na']} 檔不在選股範圍。雜誌名單和模型重疊的，才是兩邊都看好的。</div>")
    tail = (f"<div class='kq-muted'>找不到：{ui.esc('、'.join(miss[:20]))}</div>" if miss else "")
    return head + ui.resp(ui.table(["股票", "族群／題材", "短線", "中線", "長線", "反彈候選", "收盤"], rows, left=(0, 1)),
                          "".join(mrows)) + tail + (
        f"<div class='kq-muted'>依研究資料 {RDAY:%m/%d} 收盤；點代號可看完整診斷。判斷說明：推薦＝在名單或排名前段；"
        "風控排除＝M 頭、過熱、乖離過大等；不在選股範圍＝模型不評分。"
        "族群欄下方的「淨利率改善／獲利上修／成長合理價」是全市場排名（前 x%，越小越好），由交易所本益比與月營收反推（v3.9），用來檢查雜誌說的三率三升、獲利成長是否屬實；"
        "歷史驗證顯示這些季報訊號公布時股價多半已反映，所以只供參考、不在模型裡。</div>")


def page_stock():
    header()
    st.markdown(ui.section("個股診斷", "輸入代號或名稱，三個模型同時評分"), unsafe_allow_html=True)
    with st.expander("📋 多檔對照：貼上雜誌、投顧或朋友的名單，一次看模型怎麼評", expanded=bool(st.session_state.get("multi_q"))):
        mq = st.text_area("股票代號或名稱（空白、逗號或換行分隔，最多 40 檔）", key="multi_q", height=80,
                          placeholder="例：6669 2382 2317 2330 3017 2059 2308（理財周刊 AI 供應鏈估值表）")
        if mq.strip():
            st.markdown(multi_check_html(mq), unsafe_allow_html=True)
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
    live_px, bq = np.nan, None
    if market_phase() in ("盤前試撮", "盤中", "盤後"):
        q, _ = quotes_for([t])
        bq = now_quote(t, tw_intraday([t]).get(t), q)
        if bq["live"]:
            live_px = bq["px"]
    vs = {h: verdict_for(t, h, live_px, bq) for h in HZ}
    color = {"buy": "#1d4ed8", "hold": "#5b21b6", "watch": "#d97706", "no": "#475569", "na": "#475569"}
    items = []
    for h, (lvl, title, det) in vs.items():
        items.append((f"{HZ[h]}｜{HZ_SUB[h].split('・')[0]}", f"<span style='font-size:16px;color:var(--strong)'>{ui.badge(title, color[lvl])}</span>",
                       ui.esc(det)))
    best = next((h for h in ("mid", "short", "long") if vs[h][0] == "buy"), None)
    pl = plan_today(j, best or "mid", bq)
    if best:
        msg = (f"✅ <b>現在可以投資</b>：{HZ[best]}模型推薦。建議價位 <b class='kq-mono'>{ui.price(pl.buy_low)}～{ui.price(pl.buy_high)}</b>，"
               f"高於 <b class='kq-mono'>{ui.price(pl.chase)}</b> 不追；跌破 <b class='kq-mono'>{ui.price(pl.alert)}</b> 重新檢視。") if pl else \
            f"✅ <b>現在可以投資</b>：{HZ[best]}模型推薦。"
    elif any(v[0] == "watch" for v in vs.values()):
        h2 = next(h for h in HZ if vs[h][0] == "watch")
        pl = plan_today(j, h2, bq)
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
    ff = fwd_fund(t)
    if ff:
        st.markdown(f"<div class='kq-muted'>季報反推（v3.9，只供參考、不在模型裡）：{ui.esc(ff)}</div>", unsafe_allow_html=True)
    try:
        tg = stock_tags_html(t, j)                                   # v3.13 Yoda 式標籤＋營收公布反應
        if tg:
            st.markdown(tg, unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass
    position_section(t, j, vs, bq, live_px)
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
    _, _, sto = insight_frame(BID)
    if use50 and prof != "自訂" and sto and sto.get("corr") and sto.get("score", 9) <= 2:
        from kosto import market as mklib
        if st.toggle(f"套用修正期調整：0050 提高 20 個百分點（大盤距高點 {ui.pct(sto['dd'], 1)}、止穩訊號 {sto['score']}/8）",
                     value=True, key="alloc_tilt", help="止穩訊號 ≥3 項或回落小於 10% 時，系統會自動恢復基本比例"):
            w = mklib.tilt_weights(w, mklib.TILT)
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
    wmode = st.segmented_control("每檔資金分配", ["rank", "rank_invvol"], default="rank", key="alloc_wmode",
                                 format_func=lambda k: {"rank": "名次加權（預設、報酬較高）", "rank_invvol": "名次×波動（回撤較小）"}[k]) or "rank"
    if wmode == "rank_invvol":
        st.caption("v3.1 研究（五種換股日、2021–25 開發期）：同一份名單，波動大的股票少放。中線最大回撤少 6.7 個百分點，"
                   "但年化超額少 5.1 個百分點；短線回撤少 7.4、超額少 4.0。適合在意帳面起伏、不追求最高報酬的人。")
    vol_now = B.a["VOL60"][-1] if "VOL60" in B.a else None
    agg, sleeves = {}, []
    for h in HZ:
        if w.get(h, 0) <= 0:
            continue
        picks = [p for p in latest(h)["picks"] if p["ticker"] in TIDX]
        ws = planlib.stock_weights(len(picks), h, wmode if wmode == "rank_invvol" else CFG0["horizons"][h].get("weighting"),
                                   vols=[float(vol_now[TIDX[p["ticker"]]]) for p in picks] if vol_now is not None else None)
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
        st.markdown(ui.resp(ui.table(["股票", "族群", "來源（子帳戶權重）", "金額", "占總資金", f"參考價（{RDAY:%m/%d} 收盤）", "建議股數", "建議買進區"],
                                     rows, left=(0, 1, 2)), "".join(mc)), unsafe_allow_html=True)
    st.markdown(ui.panel("執行方式", ui.bullets([ui.esc(s) for s in sleeves] + [
        "中線：把中線資金分成 4 份，每週用 1 份換成當週最新名單（每份約抱 4 週）；長線：分成 3 份，每 4 週換 1 份。",
        "股價太高買不到一張時，用零股（盤後零股或盤中零股）湊足金額。",
        f"模型名單不足 {K} 檔時，空出的比例" + ("放 0050（回測就是這樣計算）。" if use50 else "留現金（已關閉 0050）。")])),
        unsafe_allow_html=True)
    st.caption("回測為每月再平衡、含交易成本的歷史模擬，不代表未來報酬。")


# ================================================================== page: 大盤
def chart_controls(key: str, ma_default: list, ind_opts: dict, ind_default: list, rng_index: int):
    """K 線控制列（v1.9 精簡）：週期、區間一列；均線與指標收進「圖表設定」彈出選單。"""
    a1, a2, a3 = st.columns([1.1, 1, 1.3], vertical_alignment="bottom")
    per = a1.segmented_control("週期", list(PERIOD), format_func=PERIOD.get, default="D", key=f"{key}_per") or "D"
    rng = a2.selectbox("區間", list(RANGE), index=rng_index, key=f"{key}_rng")
    if IS_MOBILE:                                  # v2.2 手機：預設只開最常用的兩個副圖，圖比較不擁擠
        ind_default = [x for x in ind_default if x in ("boll", "volume", "inst")][:3]
    with a3.popover("均線・指標設定", width="stretch"):
        mas = st.multiselect("均線", list(MA_OPTS), default=ma_default, format_func=MA_OPTS.get, key=f"{key}_ma")
        inds = st.multiselect("指標", list(ind_opts), default=ind_default, format_func=ind_opts.get, key=f"{key}_ind")
    return per, rng, mas, inds


@st.cache_data(show_spinner=False)
def market_frame(bid) -> tuple[pd.DataFrame, bool]:
    """加權指數日資料：開高低收、成交金額(億)、三大法人(億)、融資券(億)、漲跌家數；第二個值＝是否有官方開高低。"""
    if "Q_MKT" in B.a and B.meta.get("q_mkt_cols"):
        df = pd.DataFrame(B.a["Q_MKT"].astype(float), index=pd.DatetimeIndex(B.meta["q_days"]), columns=B.meta["q_mkt_cols"])
    elif "Q_TAIEX" in B.a:
        df = pd.DataFrame({"close": B.a["Q_TAIEX"].astype(float)}, index=pd.DatetimeIndex(B.meta["q_days"]))
    else:
        return pd.DataFrame(), False
    df = df[df["close"].notna()]
    for c in ("open", "high", "low"):
        if c not in df:
            df[c] = np.nan
    real = df["open"].notna()
    has_ohlc = bool(real.tail(60).mean() > 0.8)
    prev = df["close"].shift(1).fillna(df["close"])            # 缺開高低的日子：以前一日收盤近似開盤
    df["open"] = df["open"].where(real, prev)
    df["high"] = df["high"].where(real, np.maximum(df["open"], df["close"]))
    df["low"] = df["low"].where(real, np.minimum(df["open"], df["close"]))
    df["volume"] = df["amount"] if "amount" in df else np.nan
    return df, has_ohlc


MARKET_REF_TW = ["2330", "2454", "2317", "2308", "2382"]       # 權值股：台積電、聯發科、鴻海、台達電、廣達


def market_today(compact: bool = False):
    """大盤今日走勢：加權指數分時（平盤線置中、紅漲綠跌）＋盤中參考（權值股、那斯達克期貨、櫃買）。"""
    d = taiex_intraday()
    c1, c2 = st.columns([1.75, 1]) if not compact else (st.container(), None)
    with c1:
        st.markdown("<div class='kq-h' style='margin:8px 0 4px'><span class='t'>加權指數｜今日走勢</span>"
                    "<span class='s'>分時｜虛線＝平盤（昨收），線在上方紅、下方綠</span></div>", unsafe_allow_html=True)
        if d is None:
            st.markdown("<div class='kq-muted'>分時資料暫時無法取得（Yahoo Finance），請稍後重新整理；K 線與統計不受影響。</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(intraday_stats(d, 2), unsafe_allow_html=True)
            intraday_chart(d, key=f"tx_intra_{'c' if compact else 'm'}", height=230 if (compact or IS_MOBILE) else 300, digits=2, volume=False)
    if c2 is None:
        return
    rows = []
    tw = tw_intraday(MARKET_REF_TW)
    rows += [(f"{t} {NAMES[TIDX[t]]}", tw.get(t), "") for t in MARKET_REF_TW if t in TIDX]
    wd = world_intraday(["NQF", "SOX"])
    rows += [("那斯達克期貨", wd.get("NQF"), ""), ("費城半導體", wd.get("SOX"), "")]
    otc = ""
    if market_phase() in ("盤中", "盤後", "盤前試撮"):
        q, _ = live_quotes((INDEX_CH["櫃買指數"],))
        if len(q) and ui.ok(q["price"].iloc[0]):
            otc = (f"<div class='kq-imini'><span class='n'>櫃買指數</span><span class='g kq-muted'>{ui.num(q['price'].iloc[0], 2)}</span>"
                   f"<span class='c' style='color:{ui.color_of(q['chg'].iloc[0])}'>{q['chg'].iloc[0] * 100:+.2f}%</span></div>")
    body = mini_rows(rows) + otc
    c2.markdown(ui.panel("盤中參考｜權值股・美股期貨", (body or "<div class='kq-muted'>分時資料暫時無法取得。</div>")
                         + "<div class='kq-muted' style='margin-top:6px'>每列小圖都以平盤為中線。v2.1 研究：今年台股盤中跟著那斯達克期貨走的程度明顯變高"
                           "（每小時相關 0.34→0.54），盤中大跌時可先看期貨是否止跌。</div>"), unsafe_allow_html=True)


def page_market():
    header()
    data_strip(("price", "institutional", "margin", "mkt"))
    st.markdown(ui.section("大盤", "加權指數 K 線、成交金額、三大法人、融資、漲跌家數｜法人與融資＝個股張數×收盤價加總估算（億元）"),
                unsafe_allow_html=True)
    df, has_ohlc = market_frame(BID)
    if df.empty:
        st.info("大盤資料將在下一次每日研究後提供。")
        return
    last, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    tx = taiex_now()
    inst = sum(float(last.get(c, np.nan)) for c in ("foreign", "trust", "dealer") if ui.ok(last.get(c)))
    mchg = last.get("mflow", np.nan) if ui.ok(last.get("mflow")) else last.get("mbal", np.nan) - prev.get("mbal", np.nan)
    stage = latest("mid").get("market_stage", "UNKNOWN")
    items = [("加權指數" + ("（即時）" if tx.get("live") else ""), f"{tx.get('px', last['close']):,.0f}",
              f"{ui.chg(tx.get('chg', last['close'] / prev['close'] - 1))}　{(tx.get('date') or df.index[-1]):%m/%d}"),
             ("成交金額", f"{last.get('volume', np.nan):,.0f} 億" if ui.ok(last.get("volume")) else "–",
              f"20 日均 {df['volume'].tail(20).mean():,.0f} 億" if "volume" in df else ""),
             (("三大法人（上市官方）", f"{ui.lots_fmt(last['o_foreign'] + last['o_trust'] + last['o_dealer'])} 億",
               f"外資 {last['o_foreign']:+,.0f}｜投信 {last['o_trust']:+,.0f}｜自營 {last['o_dealer']:+,.0f}")
              if all(ui.ok(last.get(c)) for c in ("o_foreign", "o_trust", "o_dealer")) else
              ("三大法人合計（估）", f"{ui.lots_fmt(inst)} 億" if ui.ok(inst) else "–",
               "外資 " + (f"{last['foreign']:+,.0f}" if ui.ok(last.get("foreign")) else "–") + "｜投信 "
               + (f"{last['trust']:+,.0f}" if ui.ok(last.get("trust")) else "–"))),
             (("融資金額（上市官方）", f"{last['o_margin']:,.0f} 億",
               f"今日 {last['o_margin'] - prev.get('o_margin', np.nan):+,.0f} 億｜張數 {ui.pct(last['o_mlots'] / prev.get('o_mlots', np.nan) - 1, 1)}"
               if ui.ok(prev.get("o_margin")) else "")
              if ui.ok(last.get("o_margin")) else
              ("融資餘額（市值估）", f"{last.get('mbal', np.nan):,.0f} 億" if ui.ok(last.get("mbal")) else "–",
               f"今日增減 {mchg:+,.0f} 億（以張數變化計）" if ui.ok(mchg) else "")),
             ("上漲／下跌家數", f"{last.get('up', np.nan):,.0f}／{last.get('down', np.nan):,.0f}" if ui.ok(last.get("up")) else "–",
              f"市場階段 {ui.esc(STAGE_ZH.get(stage, stage).split(' ')[0])}")]
    st.markdown(ui.cards(items), unsafe_allow_html=True)
    market_today()
    opts = {k: v for k, v in IND_OPTS.items() if k != "holders"}
    opts["volume"] = "成交金額"
    per, rng, mas, inds = chart_controls("mk", [5, 20, 60, 240], opts, ["boll", "volume", "kd", "macd", "inst", "margin"], 2)
    from kosto.charts import kline
    ind = ta.indicators(ta.resample(df[[c for c in ("open", "high", "low", "close", "volume", "foreign", "trust", "dealer", "mbal", "sbal")
                                        if c in df]], per))
    bars = max(20, RANGE[rng] // {"D": 1, "W": 5, "M": 21}[per])
    fig = kline(ind, dark=DARK, mas=tuple(mas), boll="boll" in inds, panels=[x for x in inds if x != "boll"], bars=bars,
                height=(300 + 95 * len([x for x in inds if x != "boll"])) if IS_MOBILE else None,
                unit="億", vol_label="成交金額(億)")
    if IS_MOBILE:                                  # 觸控：單指拖曳用來捲動頁面，不要拖動圖表
        fig.update_layout(dragmode=False)
    st.plotly_chart(fig, width="stretch", theme=None, key="mk_fig",
                    config={"displaylogo": False, "scrollZoom": not IS_MOBILE, "displayModeBar": not IS_MOBILE,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]})
    if not has_ohlc:
        st.caption("加權指數的開高低將在下一次每日研究後補齊（目前以前一日收盤近似開盤）。")
    st.markdown(ui.section("台股熱力圖", "面積＝近 20 日成交金額、顏色＝漲跌（紅漲綠跌，越深越多）；電子依產業鏈層級與細族群分組"),
                unsafe_allow_html=True)
    heatmap_section("mk_heat", default_top=150)
    if "up" in df:
        brd = df[["up", "down"]].tail(60)
        import plotly.graph_objects as go
        from kosto.charts import THEME
        BG, GRID, TXT, MUTED = THEME[bool(DARK)]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=[d.strftime("%m/%d") for d in brd.index], y=brd["up"], name="上漲家數", marker_color=ui.UP))
        fig2.add_trace(go.Bar(x=[d.strftime("%m/%d") for d in brd.index], y=-brd["down"], name="下跌家數", marker_color=ui.DOWN))
        fig2.update_layout(barmode="relative", height=200, margin=dict(l=8, r=40, t=8, b=8), paper_bgcolor=BG, plot_bgcolor=BG,
                           font=dict(color=TXT, family="Inter, Noto Sans TC, sans-serif"), legend=dict(orientation="h", y=1.15),
                           xaxis=dict(type="category", nticks=10, gridcolor=GRID), yaxis=dict(gridcolor=GRID, side="right"))
        st.markdown(ui.section("多空家數（近 60 日）"), unsafe_allow_html=True)
        st.plotly_chart(fig2, width="stretch", theme=None, config={"displaylogo": False, "displayModeBar": False})


# ================================================================== v2.0 資料依據（每頁顯示用了哪一天的資料）
DD_LABEL = {"price": "收盤", "institutional": "法人", "margin": "融資券", "valuation": "本益比", "revenue": "月營收",
            "world": "美股", "etf": "ETF", "mkt": "官方合計"}


def data_strip(keys=("price", "institutional", "margin", "valuation", "revenue", "world")):
    """「分析依據」：各資料集最新日期；比收盤日舊的標黃。研究資料日落後應有交易日 → 紅色提醒。"""
    dd = B.meta.get("data_dates") or {"price": str(B.sig_dates[-1].date())}
    base = dd.get("price")
    chips = []
    for k in keys:
        v = dd.get(k)
        if not v:
            continue
        late = k in ("institutional", "margin", "valuation", "etf", "mkt") and base and v < base
        txt = v[5:] if len(v) == 10 else (v[:4] + "/" + v[5:] if len(v) == 7 else v)
        chips.append(f"<span class='kq-chip{' late' if late else ''}'>{DD_LABEL.get(k, k)} {txt}</span>")
    exp = freshness.expected_data_day(pd.Timestamp.now(tz="Asia/Taipei"))
    warn = ""
    if pd.Timestamp(base) < exp:
        warn = (f"<div class='kq-warn'>⚠ 分析依據 {pd.Timestamp(base):%m/%d} 收盤；最新交易日應為 {exp:%m/%d}，"
                "資料還沒更新（看頁首燈號）。盤中請搭配「盤中監控」的即時報價。</div>")
    lates = [c for c in chips if " late'" in c]
    mob = (f"<div class='kq-strip'><span class='kq-muted'>分析依據</span><span class='kq-chip'>收盤 {base[5:] if base else '–'}</span>"
           + ("".join(lates) if lates else "<span class='kq-chip'>各資料同步</span>") + "</div>")
    st.markdown(ui.resp(f"<div class='kq-strip'><span class='kq-muted'>分析依據</span>{''.join(chips)}</div>", mob) + warn, unsafe_allow_html=True)


# ================================================================== v2.0 市場解讀
@st.cache_data(show_spinner=False)
def insight_frame(bid):
    from kosto import market as mklib
    df, _ = market_frame(bid)
    if df.empty:
        return df, pd.DataFrame(), {}
    if "foreign" in df and df["foreign"].notna().any():
        c = mklib.checklist(df)
        sto = mklib.state(df, c, latest("mid").get("market_stage", ""))
    else:
        c, sto = pd.DataFrame(), {}
    return df, c, sto


@st.cache_data(show_spinner=False)
def group_moves(bid, n=5):
    """類股（細產業）近 5／20 日中位數報酬與上漲家數比例（成交金額前 500 名）。"""
    if "Q_C" not in B.a:
        return pd.DataFrame()
    C = pd.DataFrame(B.a["Q_C"].astype(float), index=pd.DatetimeIndex(B.meta["q_days"]), columns=B.tickers).ffill()
    g = pd.Series({t: (CH.get(t) or {}).get("group", "其他") for t in B.tickers})
    cap = B.a["CAP"][-1] if "CAP" in B.a else np.ones(len(B.tickers))
    top = pd.Index(B.tickers)[np.argsort(-np.nan_to_num(cap))[:500]]
    r5 = C.iloc[-1] / C.iloc[-6] - 1
    r20 = C.iloc[-1] / C.iloc[-21] - 1
    t = pd.DataFrame({"g": g, "r5": r5, "r20": r20}).loc[top]
    t = t[t["g"].astype(str).str.len() > 0]
    out = t.groupby("g").agg(n=("r5", "size"), r5=("r5", "median"), r20=("r20", "median"), up=("r5", lambda s: float((s > 0).mean())))
    return out[out["n"] >= 4].sort_values("r5", ascending=False)


# ================================================================== v2.2 盤勢儀表板（原「市場解讀」＋急跌應變＋特別時期選股＋模型體檢）
V22_TESTS = [  # docs/v2.2_crash_protocol.md：五種換股日、開發期 2021–2025（資料到 2026-10-22）
    ("B1 急跌時動能因子權重減半", "+0.6", "+0.02", "+1.2", "+3.2", "−0.2", "未通過（需 ≥ +1.5）"),
    ("B2 同上、急跌後 40 日內維持", "−0.3", "+0.01", "+1.4", "+0.7", "−3.3", "未通過"),
    ("C2 同產業最多 2 檔", "−3.8", "−0.05", "+4.7", "−4.2", "−3.5", "未通過（超額降太多）"),
    ("C3 同產業最多 3 檔", "−1.1", "−0.01", "+2.4", "+0.0", "−0.4", "未通過（差 0.1，列入觀察）"),
]


def crash_now(df) -> dict:
    """急跌狀態：研究資料（B.meta crash_stats.current）＋盤中即時（加權分時、前一晚費半即時報價）。"""
    from kosto import market as mklib
    cs = B.meta.get("crash_stats") or {}
    cur = dict(cs.get("current") or {})
    tx = df["close"].astype(float).dropna() if "close" in df else pd.Series(dtype=float)
    if not cur and len(tx) > 30:
        r = tx.pct_change().to_numpy()
        k = 0
        while k < len(r) - 1 and r[-1 - k] < 0:
            k += 1
        hi = float(tx.tail(250).max())
        cur = {"date": str(tx.index[-1].date()), "close": float(tx.iloc[-1]), "high250": hi, "dd250": float(tx.iloc[-1] / hi - 1),
               "streak": int(k), "streak_ret": float(tx.iloc[-1] / tx.iloc[-1 - k] - 1) if k else 0.0}
        if "sox" in df and df["sox"].notna().sum() > 2:
            sx = df["sox"].ffill()
            cur["sox_day"] = float(sx.iloc[-1] / sx.iloc[-2] - 1)
    if not cur:
        return {}
    cur["live"] = False
    d = taiex_intraday()
    if d is not None and str(d["date"]) > cur.get("date", ""):
        cur["live"] = True
        cur["live_status"] = IDY.status(d)
        hi = max(float(cur.get("high250") or 0), float(d["high"]))
        cur["dd250"] = d["last"] / hi - 1 if hi else cur.get("dd250")
        if abs(d["prev"] / cur["close"] - 1) < 0.005:          # 研究資料就是昨天 → 連跌天數可以接上今天
            if d["last"] < d["prev"]:
                cur["streak_ret"] = (1 + cur.get("streak_ret", 0.0)) * (d["last"] / d["prev"]) - 1
                cur["streak"] = cur.get("streak", 0) + 1
            else:
                cur["streak"], cur["streak_ret"] = 0, 0.0
        cur["close_live"] = d["last"]
    sx = world_snapshot().get("SOX")
    if sx and sx.get("live") and ui.ok(sx.get("chg")):
        cur["sox_day"] = float(sx["chg"])
        cur["sox_live"] = True
    cur["flags"] = mklib.crash_flags(cur)
    return cur


def crash_summary() -> tuple[dict, str, bool]:
    from kosto import market as mklib
    cs = B.meta.get("crash_stats") or {}
    if cs.get("summary"):
        return cs["summary"], cs.get("asof", ""), True
    return mklib.CRASH_FALLBACK["summary"], mklib.CRASH_FALLBACK["asof"], False


def crash_banner(cur) -> str:
    on = [f for f in cur.get("flags", []) if f[2]]
    if not on:
        return ""
    summ, _, _ = crash_summary()
    k = "E2" if any(f[0] == "E2" for f in on) else on[0][0]
    sm = summ.get(k) or {}
    lines = "｜".join(ui.esc(t) for _, t, ok_ in cur["flags"] if ok_)
    base = (f"歷史上同類事件 {sm.get('n', 0)} 次：當天就是之後 60 日最低點的只有 {sm.get('low_hits', 0)} 次；"
            f"之後 20 日內平均還會再跌 {abs(sm.get('min20') or 0):.0%}，但 60 日後平均 {sm.get('r60', 0):+.0%}"
            f"（{sm.get('win60', 0):.0%} 上漲）。" if sm else "")
    return (f"<div class='kq-alert'><div class='t'>🚨 急跌警報</div><div class='s'>{lines}</div>"
            f"<div class='s2'>{ui.esc(base)}→ 不要在恐慌時殺低，也不要一次重壓；照下方「急跌應變」分批。</div></div>")


def crash_rates_html(summ: dict) -> str:
    from kosto import market as mklib
    rows, mrows = [], []
    for k, nm in mklib.CRASH_TYPES.items():
        r = summ.get(k)
        if not r:
            continue
        low = f"{r['low_hits']}/{r['low_n']}" if r.get("low_n") else "–"
        rows.append([f"<b>{k}</b> {ui.esc(nm)}", f"{r['n']}", f"{ui.pct(r['r20'], 1, color=True)}<span class='sub'>（{ui.pct(r['win20'], 0, sign=False)}漲）</span>",
                     f"{ui.pct(r['r60'], 1, color=True)}<span class='sub'>（{ui.pct(r['win60'], 0, sign=False)}漲）</span>",
                     ui.pct(r["min20"], 1), f"{ui.pct(r['min60'], 1)}<span class='sub'>（最深 {ui.pct(r['worst60'], 0)}）</span>", low,
                     f"{ui.pct(r['lump60'], 1)} / {ui.pct(r['staged60'], 1)}"])
        mrows.append(ui.mcard(f"<span class='nm'>{k} {ui.esc(nm)}</span>", f"{r['n']} 次",
                              f"之後 20 日 {ui.pct(r['r20'], 1, color=True)}｜60 日 {ui.pct(r['r60'], 1, color=True)}（{ui.pct(r['win60'], 0, sign=False)}漲）",
                              f"20 日內再跌 {ui.pct(r['min20'], 1)}｜60 日內 {ui.pct(r['min60'], 1)}（最深 {ui.pct(r['worst60'], 0)}）｜當天是低點 {low}"))
    return ui.resp(ui.table(["事件", "次數", "之後 20 日", "之後 60 日", "20 日內再跌", "60 日內再跌", "當天＝60日低點",
                             "60 日後：一次買／分三批"], rows, left=(0,)), "".join(mrows))


def ban_html() -> str:
    """v2.3 限空令：歷次措施之後的大盤與個股（docs/v2.3_protocol.md A）。"""
    from kosto import review as rv
    rows, mrows = [], []
    for e in rv.BAN_HISTORY:
        src = "研究資料" if e["src"] == "data" else "新聞"
        r250 = ui.pct(e.get("r250"), 0, color=True) if e.get("r250") is not None else "–"
        low = f"{ui.pct(e['min60'], 1)}<span class='sub'>（第 {e['low']} 天）</span>" if e.get("min60") is not None else "–"
        rows.append([f"<b>{ui.esc(e['date'])}</b><div class='sub'>{ui.esc(e['what'])}</div>", ui.pct(e.get("dd"), 0) if e.get("dd") is not None else "–",
                     ui.pct(e.get("day0"), 1, color=True) if e.get("day0") is not None else "–",
                     ui.pct(e["r20"], 1, color=True), ui.pct(e["r60"], 1, color=True), r250, low, src])
        mrows.append(ui.mcard(f"<span class='nm'>{ui.esc(e['date'])}</span>", f"1 季 {ui.pct(e['r60'], 1, color=True)}",
                              ui.esc(e["what"]),
                              (f"生效當天 {ui.pct(e['day0'], 1, color=True)}｜" if e.get("day0") is not None else "")
                              + f"1 個月 {ui.pct(e['r20'], 1, color=True)}" + (f"｜之後再跌 {ui.pct(e['min60'], 1)}" if e.get("min60") is not None else "")))
    ref = rv.CRASH_E1_REF
    rows.append(["<span class='kq-muted'>對照：一般急跌（回落 15%、沒有限空，6 次）</span>", "", "", ui.pct(ref["r20"], 1, color=True),
                 ui.pct(ref["r60"], 1, color=True), "", ui.pct(ref["min60"], 1), "研究資料"])
    return ui.resp(ui.table(["生效日／措施", "生效前距高點", "生效當天", "之後 1 個月", "之後 1 季", "之後 1 年", "1 季內再跌（最低在第幾天）", "來源"],
                            rows, left=(0,)), "".join(mrows))


def ban_xs_html() -> str:
    from kosto import review as rv
    rows, mrows = [], []
    for nm, *evs in rv.BAN_XS:
        cells = []
        for a, b_, c_, t in evs:
            tt = f"<span class='sub'>（t {t:+.1f}）</span>" if t is not None else ""
            cells.append(f"{a * 100:+.0f}／{b_ * 100:+.0f}／<b>{c_ * 100:+.0f}</b>{tt}")
        rows.append([f"<b>{ui.esc(nm)}</b>"] + cells)
        mrows.append(ui.mcard(f"<span class='nm'>{ui.esc(nm)}</span>", "", "　".join(
            f"{e}：{x[2] * 100:+.0f}" for e, x in zip(rv.BAN_XS_EVENTS, evs)), "之後 60 日超額（百分點）"))
    return ui.resp(ui.table(["最高 20% 減最低 20%（百分點）", "2020-03　5／20／60 日", "2022-10", "2025-04（與這次相同）"], rows, left=(0,)),
                   "".join(mrows))


@st.cache_data(ttl=600, show_spinner=False)
def news_headlines(query: str, n: int = 6, days: int = 3) -> list:
    from kosto import news
    return news.google_news(query, n=n, days=days)


@st.cache_data(ttl=600, show_spinner=False)
def material_today_df() -> pd.DataFrame:
    from kosto import news
    return news.material_today()


def headlines_html(items, empty="目前抓不到新聞（網路或來源暫時無法連線），請用下方「可靠的即時消息來源」。") -> str:
    if not items:
        return f"<div class='kq-muted'>{empty}</div>"
    rows = []
    for x in items:
        tm = f"{x['time']:%m/%d %H:%M}" if x.get("time") is not None else ""
        rows.append(f"<div style='padding:4px 0;border-bottom:1px dashed var(--line)'><a href='{ui.esc(x['link'])}' target='_blank' rel='noopener'>"
                    f"{ui.esc(x['title'])}</a><div class='kq-sub'>{ui.esc(x.get('source', ''))}　{tm}</div></div>")
    return "".join(rows)


def sources_html() -> str:
    from kosto import news
    rows = [[ui.tag(k, "gold" if k == "官方" else ""), f"<a href='{u}' target='_blank' rel='noopener'>{ui.esc(n)}</a>", ui.esc(d)]
            for k, n, u, d in news.SOURCES]
    mrows = [ui.mcard(f"<a href='{u}' target='_blank' rel='noopener'>{ui.esc(n)}</a>", ui.tag(k), "", ui.esc(d)) for k, n, u, d in news.SOURCES]
    return ui.resp(ui.table(["類別", "來源", "用途"], rows, left=(0, 1, 2)), "".join(mrows))


def material_html(tickers) -> str:
    """名單／自選股今天的重大訊息（證交所、櫃買中心 OpenAPI）。"""
    df = material_today_df()
    if df is None or not len(df):
        return ("<div class='kq-muted'>今天的重大訊息暫時無法取得（或尚未公布），可到 <a href='https://mops.twse.com.tw/mops/web/t05sr01_1' "
                "target='_blank' rel='noopener'>公開資訊觀測站｜即時重大訊息</a> 查詢。</div>")
    x = df[df["ticker"].isin(list(tickers))]
    if not len(x):
        return f"<div class='kq-muted'>名單中的股票今天沒有重大訊息（全市場今天共 {len(df)} 則）。</div>"
    rows = [[stock_cell(r.ticker) if r.ticker in TIDX else ui.esc(f"{r.ticker} {r.name}"), f"{ui.esc(r.date[5:])} {ui.esc(r.time)}",
             f"<b>{ui.esc(r.title)}</b><div class='kq-sub' style='white-space:normal'>{ui.esc(r.detail[:160])}</div>"] for r in x.itertuples()]
    return ui.table(["股票", "時間", "主旨"], rows, left=(0, 2))


@st.cache_data(show_spinner=False)
def shock_df(bid) -> pd.DataFrame:
    from kosto import review as rv
    return rv.shock_events(B)


def shock_html(df: pd.DataFrame, cnt: dict) -> str:
    """近 10 日消息面急漲／急跌股＋歷史上同類事件之後的表現（v2.5 研究）。df：每檔只留最近一次事件。"""
    ranks_mid = {}
    try:
        rk = ranks(BID, "mid")
        ranks_mid = {t: int(r) for t, r in rk["rank"].dropna().items()}
    except Exception:  # noqa: BLE001
        pass
    picks = {p["ticker"] for h in HZ for p in latest(h).get("picks", [])}
    rows, mrows = [], []
    for r in df.itertuples():
        t = B.tickers[r.j]
        tag = ui.tag("名單", "gold") if t in picks else (ui.tag(f"中線第 {ranks_mid[t]} 名") if ranks_mid.get(t, 999) <= 50 else "")
        kind = f"<b style='color:{ui.UP if r.up else ui.DOWN}'>{'急漲' if r.up else '急跌'}</b>"
        rep = f"<span class='sub'>（10 日內 {cnt[r.j]} 次）</span>" if cnt.get(r.j, 1) > 1 else ""
        rows.append([stock_cell(t) + " " + tag, f"{r.date:%m/%d}{rep}", f"{kind} {ui.pct(r.ret, 1, color=True)}", f"{r.volx:.1f} 倍",
                     ui.pct(r.inst, 0, color=True) if ui.ok(r.inst) else "–", ui.pct(r.marg, 0) if ui.ok(r.marg) else "–",
                     ui.pct(r.rev3, 0, color=True) if ui.ok(r.rev3) else "–", ui.esc(r.group), ui.pct(r.since, 1, color=True) if r.days_ago > 0 else "<span class='sub'>最新一天</span>",
                     f"{ui.pct(r.hist_x20, 1, color=True)} / {ui.pct(r.hist_x60, 1, color=True)}"])
        mrows.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[r.j])}</span> {tag}",
                              f"{kind} {ui.pct(r.ret, 1, color=True)}",
                              f"{r.date:%m/%d}{rep}｜量 {r.volx:.1f} 倍｜法人 {ui.pct(r.inst, 0) if ui.ok(r.inst) else '–'}｜融資 {ui.pct(r.marg, 0) if ui.ok(r.marg) else '–'}",
                              f"{ui.esc(r.group)}：歷史之後 20／60 日超額 {ui.pct(r.hist_x20, 1)}／{ui.pct(r.hist_x60, 1)}"
                              + (f"｜事件後至今 {ui.pct(r.since, 1)}" if r.days_ago > 0 else "｜最新一天")))
    return ui.resp(ui.table(["股票", "最近事件", "當天", "成交量", "法人買賣／量", "融資增減／量", "營收年增(3月)", "歸類", "事件後至今",
                             "歷史同類之後 20／60 日超額"], rows, left=(0, 7)), "".join(mrows))


def shock_section(n_show: int = 15):
    df = shock_df(BID)
    if df is None or not len(df):
        st.markdown("<div class='kq-muted'>近 10 個交易日沒有符合條件的事件（或資料包尚未更新）。</div>", unsafe_allow_html=True)
        return
    cnt = df.groupby("j").size().to_dict()
    d1 = df.drop_duplicates("j", keep="first").reset_index(drop=True)          # 已依日期新到舊排序 → 每檔留最近一次
    st.markdown(shock_html(d1.head(n_show), cnt), unsafe_allow_html=True)
    if len(d1) > n_show:
        with st.expander(f"其餘 {len(d1) - n_show} 檔（共 {len(d1)} 檔、{len(df)} 次事件）"):
            st.markdown(shock_html(d1.iloc[n_show:], cnt), unsafe_allow_html=True)


def shock_study_html() -> str:
    from kosto import review as rv
    def tbl(data):
        rows = [[ui.esc(g), f"{n:,}", ui.pct(a, 1, color=True), ui.pct(b_, 1, color=True), ui.pct(c, 1, color=True), ui.pct(d, 1, color=True),
                 f"{w20:.0%}／{w60:.0%}"] for g, n, a, b_, c, d, w20, w60 in data]
        return ui.table(["分組", "次數", "隔天", "5 日", "20 日", "60 日", "勝率 20／60 日"], rows, left=(0,))
    pk = rv.SHOCK_PEAK
    return ("<div class='kq-muted' style='margin:4px 0'>急漲（單日 ≥ +7% 且量 ≥ 20 日均量 3 倍）之後的超額報酬（減成交金額前 800 名平均）</div>" + tbl(rv.SHOCK_UP)
            + "<div class='kq-muted' style='margin:8px 0 4px'>急跌（單日 ≤ −7% 且量 ≥ 3 倍）</div>" + tbl(rv.SHOCK_DOWN)
            + ui.panel("怎麼解讀", ui.bullets([
                f"<b>多數急漲之後會修正，但少數大漲把平均拉高</b>：20 日內最高點有 {pk['day0'] + pk['day1']:.0%} 出現在當天或隔天，中位數第 {pk['median_day']} 天；"
                "60% 的事件 20 日後輸給大盤，但平均仍是正的（少數股票持續大漲）。",
                "<b>分辨「會延續」與「會修正」看三件事</b>：法人當天有沒有買（外資＋投信買超 > 當日量 5%）、營收是否真的成長、"
                "是不是融資追價而法人在賣。法人買＋營收年增 > 20% 之後 60 日超額 +4.1%；融資追＋法人賣 −3.0%。",
                "<b>近兩年消息面延續性變強</b>：急漲之後 60 日超額 2021–2024 介於 −1.4%～+1.1%，2025 年 +3.1%、2026 年 +4.9%；"
                "題材股比過去更容易續漲，但只有兩年、過去方向曾反覆。",
                "<b>急跌</b>：法人當天逢低承接的，之後 5 日小幅反彈；法人賣、營收也衰退的，之後 60 日再落後 3.5%。",
                "<b>模型怎麼用</b>：模型每天收盤後重算，急漲股要等「動能＋營收」都到位才會進名單，所以不會追第一天的消息；"
                "v2.5 測試把「急漲＋法人買」「急漲＋融資未追」當因子加進模型，2021–2025 開發期反而少 1～4 個百分點 → 不改模型；"
                "「急漲＋法人買」在最近 12 個月較好，列入觀察（不能用最近一年挑模型）。"])))


@st.cache_data(show_spinner=False)
def nv_chain_set(bid) -> set:
    from kosto import review as rv
    return rv.nv_chain(CH)


def nv_event_html() -> str:
    """v2.5 輝達執行長請辭（2026-11-12）：歷史上美股半導體重挫時台灣供應鏈怎麼走＋名單曝險。"""
    from kosto import review as rv
    ev = rv.NV_EVENT
    out = ui.panel(f"重大事件：輝達執行長請辭（{ev['date']}）", ui.bullets([ui.esc(x) for x in ev["facts"]]))
    for (thr, n), data in rv.NV_CHAIN_SHOCK.items():
        rows = [[f"<b>{ui.esc(g)}</b>", ui.pct(a, 1, color=True), ui.pct(b_, 1, color=True), ui.pct(c, 1, color=True), ui.pct(d, 1, color=True),
                 ui.pct(e, 1, color=True) + (f"<span class='sub'>（20／60 日上漲比例 {w20:.0%}／{w60:.0%}）</span>" if w20 is not None else "")]
                for g, a, b_, c, d, e, w20, w60 in data]
        out += (f"<div class='kq-muted' style='margin:8px 0 2px'>費半前一晚 {thr}（2019–2026 有之後 60 日資料的 {n} 次）：台股市值前 300 大等權；之後 5／20／60 日自當天收盤起算</div>"
                + ui.table(["組別", "開盤跳空", "當天", "之後 5 日", "之後 20 日", "之後 60 日"], rows, left=(0,)))
    chain_set = nv_chain_set(BID)
    expo, nh, nt = [], 0, 0
    for h in HZ:
        xs = [p["ticker"] for p in latest(h).get("picks", []) if p["ticker"] in TIDX]
        hit = [t for t in xs if t in chain_set]
        nh, nt = nh + len(hit), nt + len(xs)
        expo.append(f"{HZ[h]} {len(hit)}/{len(xs)} 檔" + (f"（{'、'.join(f'{t} {NAMES[TIDX[t]]}' for t in hit)}）" if hit else ""))
    share = nh / nt if nt else 0.0
    out += ui.panel("怎麼看", ui.bullets([
        "<b>歷史上美股半導體重挫後，台灣輝達供應鏈並沒有「直接完蛋」</b>：開盤跳空約 −3～−4%、當天收斂一些，1～2 週內常再探底，"
        "但之後 20／60 日平均反彈，而且反彈幅度大於其他類股（費半 ≤ −7% 的 14 次：60 日平均 +18%）。",
        "<b>這次不同的地方</b>：不是景氣或總體利空，而是領導人與治理風險（婚外情對象疑涉外國情報、出口管制疑慮）。"
        "關鍵是訂單與資本支出有沒有改變——11/18 清晨的輝達財報與財測、四大雲端業者的資本支出、台積電與鴻海的月營收（12/10 前公布）。",
        "<b>輝達失分、誰補位</b>：AMD、博通與雲端自研晶片（TPU）可能分走份額；台灣多數 AI 供應商三個陣營都做（一進一出），"
        "只做輝達整櫃的環節相對受傷、ASIC／AMD 專屬環節相對受惠——見「產業鏈地圖 → 消息衝擊評估 → 輝達執行長」的陣營輪動。",
        "<b>開盤怎麼應對</b>：前一晚美股的影響幾乎在開盤一次反映（v2.3），不必在開盤恐慌賣出；分批計畫與止穩檢查照常；"
        "名單中與輝達直接相關的股票，若跌破警戒價再重新檢視。",
        "<b>模型名單的曝險</b>：" + "；".join(expo) + "。" + ("名單中輝達供應鏈占比不高，主要風險是整體電子股同步下跌。" if share < 0.34 else
                                                       "名單中輝達供應鏈占比偏高，留意財報與接班消息；不建議因單一事件全數出清（歷史上 20／60 日多半反彈）。")]))
    return out


@st.cache_data(show_spinner=False)
def egg_cached(bid):
    from kosto import egg
    md = egg.market_stage(B)
    return md, egg.stage_forward(B), egg.news_reaction(B), egg.money_psych(B)


def kostolany_tab():
    """v3.3 科斯托蘭尼：雞蛋圖與目前階段、各階段歷史表現、對消息的反應、貨幣＋心理溫度計、原著原則對照。"""
    from kosto import egg
    md, fw, nr, mp = egg_cached(BID)
    if md is None or md.empty:
        st.info("大盤階段資料將在下一次每日研究後提供。")
        return
    cur = md["stage"].iloc[-1]
    info = egg.STAGE_INFO.get(cur, {"name": "階段不明", "book": "", "act": "—", "color": "#94a3b8"})
    scores = {k: float(md[k].iloc[-1]) for k in egg.STAGE_INFO if k in md}
    st.markdown(ui.section("科斯托蘭尼雞蛋理論：大盤現在在哪裡", f"依 {md.index[-1]:%Y-%m-%d} 收盤；成交量比、融資水位（代替持股人數）、廣度、位置"),
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    c1.markdown(egg.egg_svg(cur, scores), unsafe_allow_html=True)
    f = fw.loc[cur] if fw is not None and cur in fw.index else None
    hist = (f"這個階段歷史上（2019-07 起 {int(f['n'])} 天）之後 20 日大盤平均 {ui.pct(f['f20'], 1, color=True)}、60 日 {ui.pct(f['f60'], 1, color=True)}，"
            f"60 日後上漲機率 {f['w60']:.0%}。") if f is not None and f["n"] > 0 else ""
    prev = md["stage"].iloc[-21] if len(md) > 21 else cur
    since = int((md["stage"][::-1] == cur).cumprod().sum())
    c2.markdown(ui.panel(f"目前：{cur} {info['name']}", ui.bullets([
        f"<b>原著定義</b>：{ui.esc(info['book'])}；<b>原著做法</b>：<b style='color:{info['color']}'>{ui.esc(info['act'])}</b>",
        f"<b>已持續</b> {since} 個交易日（20 日前是 {prev}）；判斷信心 {float(md['confidence'].iloc[-1]):.0%}",
        hist or "歷史樣本不足。",
        "<b>本系統怎麼用</b>：長線在 A3／B1／B2 自動降低持股（60%／50%／40%，其餘放 0050）；"
        "短中線不依階段減碼（回測顯示台股過熱後常續漲，照書減碼年化少 4～7 個百分點）。"]
        + ([f"<b>B3 為 v3.3 依原著修正的判定</b>（原分類：{md['stage_model'].iloc[-1]}）；模型持股水位仍用原分類，B2 減碼的資金放 0050，總持股不變。"]
           if bool(md["b3_fix"].iloc[-1]) else []))), unsafe_allow_html=True)
    if fw is not None and len(fw):
        rows = [[f"<b style='color:{egg.STAGE_INFO[k]['color']}'>{k}</b> {ui.esc(egg.STAGE_INFO[k]['name'])}", ui.esc(egg.STAGE_INFO[k]['act']),
                 f"{int(r['n'])}", ui.pct(r["f20"], 1, color=True), ui.pct(r["f60"], 1, color=True), f"{r['w60']:.0%}" if np.isfinite(r["w60"]) else "–"]
                for k, r in fw.iterrows() if k in egg.STAGE_INFO]
        r = fw.loc["全部"]
        rows.append(["<b>全部日子</b>", "", f"{int(r['n'])}", ui.pct(r["f20"], 1, color=True), ui.pct(r["f60"], 1, color=True), f"{r['w60']:.0%}"])
        st.markdown(ui.section("原著說的對不對？各階段之後大盤怎麼走", "市值前 300 大等權；2019-07 起"), unsafe_allow_html=True)
        mrows = [ui.mcard(f"<b style='color:{egg.STAGE_INFO[k]['color']}'>{k}</b> {ui.esc(egg.STAGE_INFO[k]['name'])}", ui.esc(egg.STAGE_INFO[k]["act"]),
                          f"之後 20 日 {ui.pct(r['f20'], 1, color=True)}｜60 日 {ui.pct(r['f60'], 1, color=True)}",
                          f"60 日上漲機率 {r['w60']:.0%}｜{int(r['n'])} 天" if np.isfinite(r["w60"]) else f"{int(r['n'])} 天")
                 for k, r in fw.iterrows() if k in egg.STAGE_INFO]
        st.markdown(ui.resp(ui.table(["階段", "原著做法", "天數", "之後 20 日", "之後 60 日", "60 日上漲機率"], rows, left=(0, 1)), "".join(mrows)),
                    unsafe_allow_html=True)
        st.caption("B3（恐慌出清）與 A1（量小打底）之後最好、B1（高檔初跌）之後最差、B2（下跌途中）之後平平——與原著一致；"
                   "只有 A3（過熱）不同：台股過熱後常續漲。B3 依原著修正判定（v3.3）：原分類把 2020-03、2022-10、2025-04 的重挫都判成 B2。")
    if nr:
        lines = [f"近 20 個交易日：費半大漲（≥ +2%）{nr['good_days']} 天，其中台股沒跟漲 {len(nr['good_no_react'])} 天"
                 + (f"（{'、'.join(d[5:] for d in nr['good_no_react'])}）" if nr["good_no_react"] else "") + "；"
                 f"費半大跌（≤ −2%）{nr['bad_days']} 天，其中台股沒跟跌 {len(nr['bad_no_react'])} 天"
                 + (f"（{'、'.join(d[5:] for d in nr['bad_no_react'])}）" if nr["bad_no_react"] else "") + "。"]
        if nr.get("oversold_signal"):
            lines.append("<b style='color:#22c55e'>超賣訊號（原著 133 題：利空不跌就進場）</b>：大盤距高點已逾 10%，且出現利空不跌。"
                         "歷史上這種情況之後 60 日大盤平均 +11.4%（一般下跌期 +7.7%），22 次。")
        elif nr.get("overbought_watch"):
            lines.append("<b style='color:#f59e0b'>原著的超買警訊（長期上漲後利多不漲）</b>：但台股 2019 年以來這種情況之後 60 日平均仍 +6.7%（48 次），"
                         "只當提醒，不當賣出訊號。")
        else:
            lines.append("目前沒有原著定義的超買或超賣訊號（需「長期上漲後利多不漲」或「長期下跌後利空不跌」）。")
        st.markdown(ui.section("市場對消息的反應（原著：好消息不漲就退場、壞消息不跌就進場）"), unsafe_allow_html=True)
        st.markdown(ui.panel("最近的反應", ui.bullets(lines)), unsafe_allow_html=True)
    if mp:
        rows = [[ui.esc(k), ui.esc(nm), ui.esc(txt), {1: "<b style='color:#22c55e'>偏多</b>", -1: "<b style='color:#ef4444'>偏空</b>"}.get(t, "中性")]
                for k, nm, v, txt, t in mp]
        sc = sum(t for *_, t in mp)
        st.markdown(ui.section("貨幣＋心理＝趨勢（原著公式）", f"偏多 {sum(1 for *_, t in mp if t > 0)} 項、偏空 {sum(1 for *_, t in mp if t < 0)} 項；"
                               "只做溫度計參考，沒有經過回測驗證"), unsafe_allow_html=True)
        st.markdown(ui.table(["類別", "指標", "現況", "傾向"], rows, left=(0, 1, 2)), unsafe_allow_html=True)
        st.caption("利率與美元指數在下一次每日更新後開始提供（v3.3 新增）；持股人數（集保）每週更新，歷史持續累積中。"
                   + (f" 綜合：{'偏多' if sc > 0 else '偏空' if sc < 0 else '中性'}。" if mp else ""))
    st.markdown(ui.section("原著原則與本系統的對照", "哪些照書做、哪些被台股資料否決（docs/v3.3_kostolany_review.md）"), unsafe_allow_html=True)
    rows = [[ui.esc(a), ui.esc(b_), ui.esc(c), ("<b>" + ui.esc(d) + "</b>") if d.startswith("✅") else ui.esc(d)] for a, b_, c, d in egg.PRINCIPLES]
    st.markdown(ui.resp(ui.table(["原著", "本系統怎麼做", "台股歷史驗證", "狀態"], rows, left=(0, 1, 2, 3)),
                        "".join(ui.mcard(f"<b>{ui.esc(a)}</b>", ui.esc(d), ui.esc(b_), ui.esc(c)) for a, b_, c, d in egg.PRINCIPLES)),
                unsafe_allow_html=True)


def stock_egg_html(j: int) -> str:
    """個股雞蛋階段（研究資料最後一個換股日）＋原著做法；個股階段閘門已關閉（回測傷報酬），只做提醒。"""
    from kosto import egg
    from kosto.config import STAGES
    if "STAGE" not in B.a:
        return ""
    code = int(B.a["STAGE"][-1, j])
    if not 1 <= code <= 6:
        return "<div class='kq-muted'>這檔目前的雞蛋階段不明確（各階段分數都不高）。</div>"
    k = STAGES[code - 1]
    info = egg.STAGE_INFO[k]
    hist = [STAGES[c - 1] if 1 <= c <= 6 else "–" for c in B.a["STAGE"][-6:, j].astype(int)]
    return (f"<div style='font-size:13.5px;line-height:1.7'><b style='color:{info['color']}'>{k} {ui.esc(info['name'])}</b>"
            f"｜原著：{ui.esc(info['book'])} → <b>{ui.esc(info['act'])}</b><br>"
            f"<span class='kq-muted'>近 6 個換股日：{' → '.join(hist)}。個股階段只做提醒：回測顯示照個股階段不買 A3／B1／B2，"
            "開發期年化少 23～50 個百分點（台股強勢股常續強），所以模型不用它擋買進。</span></div>")


def audit_html() -> str:
    """v2.4 驗效摘要（每日研究結果）＋偏移檢查（docs/v2.4_model_audit.md）。"""
    from kosto import review as rv
    from kosto import tournament as T
    rows, mrows = [], []
    au = (RES or {}).get("auto") or {}
    for h in HZ:
        hb = ((RES or {}).get("horizons") or {}).get(h)
        if not hb:
            continue
        ev = hb.get("eval") or {}
        g = (hb.get("validation") or {}).get("gates")
        npass = f"{int(g['通過'].sum())}/{len(g)}" if g is not None and len(g) else "–"
        pre = np.nan
        if g is not None and len(g):
            m = g[g["閘門"].astype(str).str.contains("未用期間")]
            pre = float(m["數值"].iloc[0]) if len(m) else np.nan
        vs = (hb.get("vs_pool_ew") or {}).get("holdout", np.nan)
        cp = hb.get("chosen_pool") or {}
        pn = hb.get("pool_note") or {}
        since = ((au.get("pools") or {}).get(h) or {}).get("since")
        pool = f"{cp.get('rule', '')}{cp.get('size', '')}" + ("<span class='sub'>（沿用）</span>" if pn.get("kept") else "") \
            + (f"<span class='sub'>　自 {since}</span>" if since else "")
        rows.append([f"<b>{HZ[h]}</b>", pool, ui.pct(ev.get("dev_excess_cagr"), 1, color=True), f"{ev.get('dev_info_ratio', np.nan):.2f}",
                     ui.pct(ev.get("hold_excess_cagr"), 1, color=True), ui.pct(pre, 1, color=True), ui.pct(vs, 1, color=True), npass])
        mrows.append(ui.mcard(f"<span class='nm'>{HZ[h]}</span>", f"閘門 {npass}",
                              f"開發期超額 {ui.pct(ev.get('dev_excess_cagr'), 1, color=True)}｜盲測期 {ui.pct(ev.get('hold_excess_cagr'), 1, color=True)}"
                              f"｜2019–21 {ui.pct(pre, 1, color=True)}",
                              f"母池 {pool}｜盲測期勝母池等權 {ui.pct(vs, 1, color=True)}"))
    out = ""
    if rows:
        out += ui.resp(ui.table(["期間", "母池", "開發期超額", "資訊比率", "盲測期超額", "2019–21 超額", "盲測期勝母池等權", "閘門"], rows, left=(0, 1)),
                       "".join(mrows))
    ld = {h: x for h, x in (au.get("leader") or {}).items() if x}
    rb = {h: x for h, x in (au.get("risk_blocked") or {}).items() if x}
    lead_txt = "；".join(f"{HZ.get(h, h)} {ui.esc(T.describe(x['params'], (au.get('champions') or {}).get(h) or T.base_params(CFG0, h)))}"
                        f"（連續 {x['streak']} 天）" for h, x in ld.items()) or "沒有"
    k = rv.TOPK_6_VS_8
    lines = [
        "<b>原來的目標</b>：開發期以「贏 0050」（超額報酬、資訊比率）挑母池與模型，並用 8 道預先宣告的閘門檢查。"
        "開發期三個期間都明顯勝過 0050；盲測期（最近 12 個月）與 2019–21 都落後——兩段都是台積電等權值股獨強，"
        "模型仍勝過自己的母池（中小型股等權），落後來自選股範圍，不是選股失靈。",
        "<b>權重沒有偏移</b>：12 次重估技術／基本約 48／52，長線技術 55～61%（見下方「因子權重平衡」）。"
        "v2.4 起報告的族群占比也改成含「營收相對同業」15% 疊加的實際權重（舊報告寫 57／43）。",
        "<b>母池不再每天換</b>：前一天的母池與今天第一名綜合分數差不到 0.05 就沿用（舊版短中線在 ADV500／BLEND500／MCAP500 間來回切換）。",
        f"<b>自動晉級加風險護欄</b>：中線「持股 8→6 檔」在五種換股日 × 4 個母池平均開發期超額 {k['dev_excess'] * 100:+.1f}、"
        f"但最大回撤深 {abs(k['dev_mdd']) * 100:.1f} 個百分點、2019–20 {k['pre_excess'] * 100:+.1f} → 超過 2 個百分點的回撤上限，不自動換上。"
        f"目前領先的挑戰者：{lead_txt}" + ("；被護欄擋下：" + "、".join(HZ.get(h, h) for h in rb) if rb else "") + "。",
        "<b>累積的風險偏移</b>：長線開發期最大回撤隨每次通過門檻的改良由 −39% 累積到 −45%（下表），已超過「比 0050 深不超過 10 個百分點」的閘門；"
        "降低回撤最有效的仍是 0050 核心（中線 0050 30%：全期最大回撤 −41% → −37%）。",
        "<b>選股風格</b>跟著市場領漲股走：名單市值越來越大、電子占比 2025 下半年後約 75%，這是動能模型的本質，不是參數改變；"
        "同產業 ≥3 檔時名單頁會提醒。"]
    out += ui.panel("檢查結論", ui.bullets(lines))
    mrow = [[ui.esc(v), ui.esc(w), ui.pct(x, 1, color=True), ui.pct(y, 1)] for v, w, x, y in rv.MDD_DRIFT_LONG]
    out += ("<div class='kq-muted' style='margin:8px 0 2px'>長線（MCAP500）每次改良後的開發期超額與最大回撤（研究報告）</div>"
            + ui.table(["版本", "改了什麼", "開發期超額", "開發期最大回撤"], mrow, left=(0, 1)))
    srow = [[ui.esc(a), f"{b:.2f}", f"{c:.2f}", f"{d:.2f}", f"{e:.0%}", f"{f:.0%}", f"{g_:.1f}"] for a, b, c, d, e, f, g_ in rv.STYLE_DRIFT_MID]
    out += ("<div class='kq-muted' style='margin:8px 0 2px'>中線名單特性（每半年平均；市值分位＝全市場百分位，動能／波動分位＝成交金額前 500 名中的百分位）</div>"
            + ui.table(["半年", "市值分位", "120 日動能分位", "60 日波動分位", "金融占比", "電子占比", "同產業最多檔數"], srow, left=(0,)))
    return out


def balance_html() -> str:
    from kosto import review as rv
    rows, mrows = [], []
    fams, facs = B.meta.get("families") or [], B.factors
    for h in HZ:
        w = ((RES or {}).get("horizons", {}).get(h, {}).get("latest") or {}).get("weights") or latest(h).get("weights") or {}
        fs = rv.family_shares(w, facs, fams)
        if not fs:
            continue
        hist = rv.FAMILY_HIST.get(h, {})

        def rng(k):
            v = hist.get(k)
            if v is None:
                return ""
            lo, hi = (v, v) if not isinstance(v, tuple) else v
            return f"<span class='sub'>（{lo:.0%}～{hi:.0%}）</span>" if abs(hi - lo) > 0.005 else f"<span class='sub'>（{lo:.0%}）</span>"
        rows.append([f"<b>{HZ[h]}</b>", f"{fs['T']:.0%}{rng('T')}", f"{fs['F']:.0%}{rng('F')}", f"{fs['C']:.0%}", f"{fs['K']:.0%}",
                     f"{fs['mom']:.0%}{rng('mom')}", f"{ui.esc(FZH.get(fs['top'], fs['top']))} {fs['top_w']:.0%}"])
        mrows.append(ui.mcard(f"<span class='nm'>{HZ[h]}</span>", f"技術 {fs['T']:.0%}／基本 {fs['F']:.0%}",
                              f"籌碼 {fs['C']:.0%}｜科斯托蘭尼 {fs['K']:.0%}｜其中動能 {fs['mom']:.0%}",
                              f"最大單一因子：{ui.esc(FZH.get(fs['top'], fs['top']))} {fs['top_w']:.0%}"))
    if not rows:
        return "<div class='kq-muted'>（下一次每日研究後提供）</div>"
    return ui.resp(ui.table(["期間", "技術面", "基本面", "籌碼", "科斯托蘭尼", "其中動能", "最大單一因子"], rows, left=(0,)), "".join(mrows))


@st.cache_data(show_spinner=False)
def short_crowded_df(bid) -> pd.DataFrame:
    from kosto import review as rv
    return rv.short_crowded(B)


def lead_html() -> str:
    """v2.3 美股前一晚 → 台股（市值前 300 等權）。"""
    from kosto import review as rv
    rows = [[f"<b>{ui.esc(b)}</b>", f"{n}", ui.pct(g, 1, color=True), ui.pct(oc, 1, color=True), ui.pct(n5, 1, color=True),
             f"{ui.pct(n20, 1, color=True)}<span class='sub'>（{u:.0%} 漲）</span>"] for b, n, g, oc, n5, n20, u in rv.LEAD_BUCKETS]
    mrows = [ui.mcard(f"<span class='nm'>費半 {ui.esc(b)}</span>", f"開盤 {ui.pct(g, 1, color=True)}",
                      f"開盤→收盤 {ui.pct(oc, 1, color=True)}｜之後 5 日 {ui.pct(n5, 1, color=True)}", f"之後 20 日 {ui.pct(n20, 1, color=True)}（{u:.0%} 漲）｜{n} 天")
             for b, n, g, oc, n5, n20, u in rv.LEAD_BUCKETS]
    return ui.resp(ui.table(["費半前一晚", "天數", "台股開盤跳空", "開盤→收盤", "之後 5 日", "之後 20 日"], rows, left=(0,)), "".join(mrows))


def lead_section(compact: bool = False) -> None:
    """v2.3 美股、期貨、ADR 的領先意義（儀表板「觀察重點」與國際盤共用）。"""
    from kosto import review as rv
    st.markdown(ui.section("美股、期貨、台積電 ADR：開盤前看什麼", "v2.3 研究：2019-02～2026-10，1,870 個交易日；台股＝市值前 300 等權"),
                unsafe_allow_html=True)
    if not compact:
        st.markdown(lead_html(), unsafe_allow_html=True)
    ap = adr_premium(BID)
    adr_txt = ""
    if ap:
        dev = ap["last"] - ap["mean60"]
        adr_txt = (f"目前（{ap['date'][5:].replace('-', '/')} 研究資料）ADR 溢價 {ui.pct(ap['last'], 1, sign=False)}、60 日平均 {ui.pct(ap['mean60'], 1, sign=False)}"
                   f"（{'偏高' if dev > 0.01 else '偏低' if dev < -0.01 else '接近平均'} {ui.pct(dev, 1)}）。")
    c = rv.LEAD_CORR[0]
    st.markdown(ui.panel("結論", ui.bullets([
        f"<b>前一晚美股幾乎在開盤一次反映</b>：費半與台股開盤跳空的相關 {c[1]:.2f}，開盤後反而小幅回吐（{c[2]:+.2f}），對之後 5～20 日幾乎沒有預測力（{c[3]:+.2f}／{c[4]:+.2f}）。"
        "→ 模型在隔日開盤成交，美股資訊已在開盤價裡，所以不放進選股；大跌後不必追殺開盤。",
        "<b>費半大跌（≤ −5%）的隔天</b>：大型股平均開低 2.2%、盤中反而 +0.6%、之後 20 日 +2.0%（59% 上漲，一般日子 67%）→ 開盤殺低後盤中平均小幅回升，但之後一個月比一般日子弱。",
        "<b>那斯達克期貨</b>：台股盤中與那指期的每小時相關 2026 年升到 0.55～0.62（β 約 1.4，v2.1），盤中可看那指期判斷權值股方向（本頁「盤中止跌觀察」已納入）；"
        "但這是同步，不是領先。",
        "<b>台指期夜盤</b>：15:00～隔日 05:00 交易，收在美股收盤之後，幾乎包含前一晚美股全部資訊 → 隔天開盤跳空≈夜盤漲跌，"
        "是「開盤會怎麼開」最直接的參考；研究資料沒有夜盤歷史，無法另外驗證對選股的價值。",
        f"<b>台積電 ADR 溢價</b>（平均約 {rv.ADR_PREM['mean']:.0%}）：相對 60 日平均偏高時，台積電隔天開盤平均 {ui.pct(rv.ADR_PREM['hi_gap'], 1)}、之後 20 日 "
        f"{ui.pct(rv.ADR_PREM['hi_n20'], 1)}（偏低時 {ui.pct(rv.ADR_PREM['lo_gap'], 1)}／{ui.pct(rv.ADR_PREM['lo_n20'], 1)}）；關係弱、只適用台積電。" + adr_txt])),
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def adr_premium(bid) -> dict:
    """台積電 ADR 溢價（研究資料）：最新值與 60 日平均（v2.3：溢價相對平均偏高 → 台積電之後偏強，關係弱）。"""
    keys = B.meta.get("world_keys") or []
    if "WORLD" not in B.a or "TSM" not in keys or "TWD" not in keys or not HAS_Q or "2330" not in TIDX:
        return {}
    W = pd.DataFrame(B.a["WORLD"].astype(float), index=B.days, columns=keys)
    qd = pd.DatetimeIndex(B.meta["q_days"])
    c = pd.Series(B.a["Q_C"][:, TIDX["2330"]].astype(float), index=qd).ffill()
    w = W.reindex(qd)
    prem = (w["TSM"] * w["TWD"] / 5 / c.shift(1) - 1).where(lambda x: x.abs() < 0.8).dropna()
    if len(prem) < 40:
        return {}
    return {"last": float(prem.iloc[-1]), "mean60": float(prem.tail(60).mean()), "date": str(prem.index[-1].date())}


def staged_plan_html(cur, sto, live_checks) -> str:
    """分批計畫：每一批的觸發條件與目前狀態。"""
    dd = cur.get("dd250")
    sc = sto.get("score", 0)
    ma20 = (sto.get("ma") or {}).get(20)
    px = cur.get("close_live") or sto.get("close")
    live_ok = sum(1 for _, _, x in live_checks if x)
    steps = [
        ("第 1 批（計畫部位的 1/3）", "回落 ≥ 10% 且盤中出現止跌（右側「盤中止跌觀察」至少 2 項成立）",
         ui.ok(dd) and dd <= -0.10 and live_ok >= 2, f"回落 {ui.pct(dd, 1)}｜盤中止跌 {live_ok} 項"),
        ("第 2 批（1/3）", "止穩訊號 ≥ 3／8（特別是「外資賣壓減輕」「融資去槓桿」）", sc >= 3, f"止穩 {sc}/8"),
        ("第 3 批（1/3）", "加權站回月線（20 日線）且止穩 ≥ 5", bool(ui.ok(px) and ui.ok(ma20) and px > ma20 and sc >= 5),
         f"{'站上' if ui.ok(px) and ui.ok(ma20) and px > ma20 else '低於'}月線 {ma20:,.0f}" if ui.ok(ma20) else "–"),
        ("暫停加碼", "跌破前波低點再跌 5%，或費半再創新低 → 等新的止穩訊號", None, "依盤勢判斷"),
    ]
    active = ui.ok(dd) and dd <= -0.10                       # 回落未達 10%：不是急跌／修正期，照模型名單操作即可
    rows, mrows = [], []
    for nm, cond, ok_, now in steps:
        if not active:
            ok_ = None
        mark = ("<b style='color:#16a35f'>✓ 條件成立</b>" if ok_ else "<span class='kq-muted'>等待</span>") if ok_ is not None else \
            ("<span class='kq-muted'>—</span>" if active else "<span class='kq-muted'>非急跌期</span>")
        rows.append([f"<b>{nm}</b>", ui.esc(cond), ui.esc(now), mark])
        mrows.append(ui.mcard(f"<span class='nm'>{nm}</span>", mark, ui.esc(cond), ui.esc(now)))
    return ui.resp(ui.table(["分批", "觸發條件", "目前", "狀態"], rows, left=(0, 1, 2)), "".join(mrows))


@st.cache_data(show_spinner=False)
def model_health(bid) -> tuple[pd.DataFrame, dict]:
    """各期間模型 vs 0050：自高點回落、近 1／6／12 月、今年；以及 0050 核心（0／30／50%）的效果（中線）。"""
    rows, mix = [], {}
    for h in HZ:
        hb = hblock(h)
        if not hb or "nav" not in hb:
            continue
        n = hb["nav"].dropna(subset=["策略"])
        s, b = n["策略"].astype(float), n["0050"].astype(float)

        def back(x, k):
            return float(x.iloc[-1] / x.iloc[-1 - k] - 1) if len(x) > k else np.nan
        y0 = s.index[-1].year
        ys = s[s.index.year < y0]
        ytd = float(s.iloc[-1] / ys.iloc[-1] - 1) if len(ys) else np.nan
        ytd_b = float(b.iloc[-1] / b[b.index.year < y0].iloc[-1] - 1) if len(ys) else np.nan
        pool = n["母池等權"].astype(float) if "母池等權" in n else None
        ytd_p = float(pool.iloc[-1] / pool[pool.index.year < y0].iloc[-1] - 1) if pool is not None and len(ys) else np.nan
        rows.append({"h": h, "dd": float(s.iloc[-1] / s.max() - 1), "peak": str(s.idxmax().date()), "dd_b": float(b.iloc[-1] / b.max() - 1),
                     "m1": back(s, 21), "m1_b": back(b, 21), "m6": back(s, 126), "m6_b": back(b, 126), "y1": back(s, 250), "y1_b": back(b, 250),
                     "ytd": ytd, "ytd_b": ytd_b, "ytd_pool": ytd_p, "asof": str(s.index[-1].date())})
        if h == "mid":
            r = n[["策略", "0050"]].astype(float).pct_change().fillna(0)
            for w in (0.0, 0.3, 0.5):
                val, out, prev = np.array([1 - w, w]), [], None
                for d, x in r.iterrows():
                    pm = d.to_period("M")
                    if prev is not None and pm != prev:
                        val = val.sum() * np.array([1 - w, w])
                    val = val * (1 + x.to_numpy())
                    out.append(val.sum())
                    prev = pm
                v = pd.Series(out, index=r.index)
                yrs = max((v.index[-1] - v.index[0]).days / 365.25, 0.1)
                mix[w] = {"cagr": float(v.iloc[-1] ** (1 / yrs) - 1), "mdd": float((v / v.cummax() - 1).min()),
                          "y1": float(v.iloc[-1] / v.iloc[-251] - 1) if len(v) > 251 else np.nan, "dd": float(v.iloc[-1] / v.max() - 1)}
    return pd.DataFrame(rows), mix


def picks_risk() -> tuple[str, list[str]]:
    """三期間名單合併：在哪些名單、族群、近 5 日、距 20 日高、對費半 β；並檢查重疊與產業集中。"""
    lists = {h: [p["ticker"] for p in latest(h).get("picks", []) if p["ticker"] in TIDX] for h in HZ}
    union = list(dict.fromkeys(sum(lists.values(), [])))
    notes = []
    s_, m_, l_ = (set(lists.get(k, [])) for k in ("short", "mid", "long"))
    if lists.get("mid"):
        notes.append(f"三個期間的名單合計只有 <b>{len(union)}</b> 檔不同股票（短中重疊 {len(s_ & m_)} 檔、中長重疊 {len(m_ & l_)} 檔、三者都有 {len(s_ & m_ & l_)} 檔）"
                     "→ 同時照三份名單買，實際上是集中押同一批股票；資金請以一個期間為主。")
    for h, xs in lists.items():
        cnt = pd.Series([SECTORS[TIDX[t]] for t in xs]).value_counts()
        if len(cnt) and cnt.iloc[0] >= 3:
            notes.append(f"{HZ[h]}名單有 <b>{cnt.iloc[0]} 檔{ui.esc(cnt.index[0])}</b>（占 {cnt.iloc[0] / max(len(xs), 1):.0%}）："
                         "v2.2 研究顯示同產業最多 3 檔、2 檔可讓最大回撤少約 2～5 個百分點，但開發期報酬也少 1～4 個百分點；"
                         "想降低波動可自行把同產業限制在 2～3 檔。")
    C = B.a.get("Q_C")
    beta_k = FIDX.get("g_us_beta")
    from kosto import review as rv
    shs = rv.short_snapshot(B, [TIDX[t] for t in union])
    rows, mrows = [], []
    for t in union:
        j = TIDX[t]
        sh = shs.get(j) or {}
        s_ratio, s_inc = sh.get("ratio", np.nan), sh.get("inc10", np.nan)
        c = C[:, j].astype(float) if C is not None else np.array([])
        c = c[np.isfinite(c)]
        r5 = c[-1] / c[-6] - 1 if len(c) > 6 else np.nan
        d20 = c[-1] / c[-20:].max() - 1 if len(c) >= 20 else np.nan
        beta = float(B.a["LATEST_RAW"][j][beta_k]) if beta_k is not None and "LATEST_RAW" in B.a else np.nan
        tags = "".join(ui.tag(HZ[h]) for h in HZ if t in lists.get(h, []))
        lab = CH.get(t) or {}
        rows.append([stock_cell(t), tags, ui.esc(lab.get("segment") or lab.get("group") or SECTORS[j]), ui.pct(r5, 1, color=True),
                     ui.pct(d20, 1), f"{beta:.2f}" if ui.ok(beta) else "–", ui.pct(s_ratio, 1, sign=False),
                     ui.pct(s_inc, 1) if ui.ok(s_inc) else "–"])
        mrows.append(ui.mcard(f"<span class='kq-cd'>{t}</span><span class='nm'>{ui.esc(NAMES[j])}</span> {tags}", ui.pct(r5, 1, color=True),
                              (f"距 20 日高 {ui.pct(d20, 1)}｜對費半 β {beta:.2f}" if ui.ok(beta) else f"距 20 日高 {ui.pct(d20, 1)}")
                              + (f"｜券資比 {ui.pct(s_ratio, 1, sign=False)}" if ui.ok(s_ratio) else ""),
                              ui.esc(lab.get("segment") or SECTORS[j])))
    html = ui.resp(ui.table(["股票", "名單", "族群", "近 5 日", "距 20 日高", "對費半 β", "券資比", "融券 10 日增減／均量"], rows, left=(0, 1, 2)),
                   "".join(mrows))
    return html, notes


def insight_checklist(c) -> str:
    from kosto import market as mklib
    hs = B.meta.get("mkt_stats") or {}
    dev_items = (hs.get("items") or {}).get("dev", {})
    rows, chips = [], []
    for k, nm, desc in mklib.CHECKS:
        on = bool(c[k].iloc[-1]) if k in c else False
        it = dev_items.get(k) or {}
        hist = (f"{ui.pct(it.get('on'), 1, color=True)} vs {ui.pct(it.get('off'), 1, color=True)}" if it else "–")
        mark = "<b style='color:#16a35f'>✓ 成立</b>" if on else "<span class='kq-muted'>✗ 未成立</span>"
        rows.append([f"<b>{nm}</b>", ui.esc(desc), mark, hist])
        chips.append(f"<div class='kq-chk{' on' if on else ''}'><b>{'✓' if on else '✗'} {nm}</b><span>{ui.esc(desc)}</span></div>")
    return ui.resp(ui.table(["項目", "定義", "今天", "歷史（成立 vs 未成立）"], rows, left=(0, 1)),
                   "<div class='kq-chkgrid'>" + "".join(chips) + "</div>")


def insight_buckets() -> str:
    hs = B.meta.get("mkt_stats") or {}
    bk = hs.get("buckets") or {}
    if not bk:
        return ""
    brow, mrow = [], []
    for nm, lab in (("dev", "2017–2023"), ("val", "2024–2026")):
        for r in bk.get(nm, []):
            brow.append([lab, f"<b>{r['bucket']}</b>", f"{r['n']}", ui.pct(r["f20"], 1, color=True), ui.pct(r["win20"], 0, sign=False),
                         ui.pct(r["low20"], 1), ui.pct(r["f60"], 1, color=True)])
            mrow.append(ui.mcard(f"<span class='nm'>{lab}｜分數 {r['bucket']}</span>", f"{r['n']} 天",
                                 f"之後 20 日 {ui.pct(r['f20'], 1, color=True)}（{ui.pct(r['win20'], 0, sign=False)}漲）｜60 日 {ui.pct(r['f60'], 1, color=True)}",
                                 f"20 日內再跌 {ui.pct(r['low20'], 1)}"))
    html = ui.resp(ui.table(["期間", "止穩分數", "天數", "之後 20 日", "上漲機率", "20 日內再跌", "之後 60 日"], brow, left=(0, 1)), "".join(mrow))
    eps = hs.get("episodes") or []
    if eps:
        html += ("<div class='kq-muted' style='margin:4px 0 2px'>2017 年以來回落 ≥10% 的修正：" + "；".join(
            f"{e['start'][:7]} 最深 {e['max_dd']:.0%}（低點 {e['low_date'][5:]}" + (f"，之後 60 日 {e['after_low_60d']:+.0%}" if e.get("after_low_60d") is not None else "")
            + "）" for e in eps) + "</div>")
    html += ("<div class='kq-muted'>怎麼用：分數高時之後再跌的幅度明顯較小（兩段期間都成立）；但 2024 年後的修正都是 V 型反彈，"
             "分數低時進場報酬反而不差 → 檢查表適合用來<b>控制分批的速度</b>，不是預測漲跌。</div>")
    return html


def insight_focus(df, sto, f5, t5, mchg, sox_dd) -> str:
    last = df.iloc[-1]
    ma = sto["ma"]
    trend = ("均線多頭排列" if ma[5] > ma[20] > ma[60] else "均線空頭排列" if ma[5] < ma[20] < ma[60] else "均線糾結")
    amt20 = df["volume"].tail(20).mean() if "volume" in df else np.nan
    pe_now = last.get("pe_mkt", np.nan)
    pe_hist = df["pe_mkt"].dropna() if "pe_mkt" in df else pd.Series(dtype=float)
    pe_pct = float((pe_hist < pe_now).mean()) if len(pe_hist) > 100 and ui.ok(pe_now) else np.nan
    gm = group_moves(BID)
    lines = [f"<b>趨勢</b>：收盤 {sto['close']:,.0f}，{'站上' if sto['close'] > ma[20] else '跌破'}月線（{ma[20]:,.0f}）、"
             f"{'站上' if sto['close'] > ma[60] else '跌破'}季線（{ma[60]:,.0f}）、年線 {ma[240]:,.0f}；{trend}。",
             f"<b>廣度</b>：近 5 日上漲 {df['up'].tail(5).sum():,.0f} 家／下跌 {df['down'].tail(5).sum():,.0f} 家；"
             f"站上月線的股票 {ui.pct(last.get('ab20'), 0, sign=False)}，20 日新高 {last.get('hi20', 0):,.0f} 家、新低 {last.get('lo20', 0):,.0f} 家。",
             f"<b>資金</b>：成交金額 {last.get('volume', np.nan):,.0f} 億（20 日均 {amt20:,.0f} 億）；外資近 5 日 {f5:+,.0f} 億、"
             f"投信 {t5:+,.0f} 億；融資張數 10 日 {ui.pct(mchg, 1)}（{'去槓桿中' if ui.ok(mchg) and mchg <= -0.03 else '尚未明顯去槓桿'}）。"]
    if ui.ok(sox_dd):
        lines.append(f"<b>國際</b>：費半距一年高點 {ui.pct(sox_dd, 1)}，VIX {last.get('vix', np.nan):.1f}；"
                     f"美元／台幣 {last.get('twd', np.nan):.2f}（{'台幣轉弱、外資匯出壓力' if ui.ok(last.get('twd')) and df['twd'].ffill().iloc[-1] > df['twd'].ffill().iloc[-21] else '台幣持穩'}）。")
    if ui.ok(pe_now):
        lines.append(f"<b>估值</b>：大盤本益比約 {pe_now:.1f} 倍（近兩年 {ui.pct(pe_pct, 0, sign=False)} 分位）、現金殖利率 {last.get('dy_mkt', np.nan):.2f}%。")
    if len(gm):
        lines.append("<b>類股</b>：近 5 日相對強 " + "、".join(f"{ui.esc(g)} {r.r5:+.1%}" for g, r in gm.head(4).iterrows())
                     + "；最弱 " + "、".join(f"{ui.esc(g)} {r.r5:+.1%}" for g, r in gm.tail(4).iloc[::-1].iterrows()) + "。")
    lines.append("<b>資金觀望或進場</b>：" + ("修正未止穩時，歷史上之後 20 日內平均還會再跌 3～5%；想進場用「分批」：例如先投入 1/3，"
                                       "止穩訊號 ≥3（特別是融資去槓桿、外資賣壓減輕）再投入 1/3，站回月線再投入最後 1/3。"
                                       if sto["corr"] and sto["score"] <= 2 else
                                       "止穩訊號增加中，可依計畫分批布局；跌破前波低點則暫停加碼。" if sto["corr"] else
                                       "非修正期：照模型名單操作即可。"))
    return ui.panel("今天的重點", ui.bullets(lines))


def turning_html() -> str:
    """v3.11 大跌大漲時模型準不準：重要轉折點（0050 回落 ≥10%）每段的下跌段與反彈段，依每日研究的回測淨值自動更新。"""
    from kosto import review as rv
    from kosto import turning as TP
    navs = {h: RES["horizons"][h]["nav"] for h in HZ if RES and h in RES.get("horizons", {}) and "nav" in RES["horizons"][h]}
    h = ui.section("大跌大漲時，模型準不準？（v3.11 重要轉折點驗證）", "0050 回落 ≥10% 的每一段；依每日研究的回測淨值，每天自動更新")
    D = TP.decompose(navs) if navs else pd.DataFrame()
    if len(D):
        f = lambda v: "–" if not ui.ok(v) else ui.pct(v, 1, color=True)       # noqa: E731
        rows = []
        for _, r in D.iterrows():
            seg = f"{r['peak']:%Y-%m-%d} → {r['trough']:%m-%d}" + ("（進行中）" if r["ongoing"] else "")
            rows.append([f"<b>{seg}</b>", f(r["b_dn"]), *[f(r.get(f"{k}_dn")) for k in navs], f(r["b_up60"]),
                         *[f(r.get(f"{k}_up60")) for k in navs]])
        h += ui.table(["轉折（高點 → 低點）", "0050 下跌段", *[f"{HZ[k]}下跌段" for k in navs], "0050 低點後 60 日", *[f"{HZ[k]} 60 日" for k in navs]],
                      rows, left=(0,))
    rows = [[f"<b>{a}</b>", ui.esc(b), f"{m[0]:+.1f}%／{m[1]:.1f}%", f"{s[0]:+.1f}%／{s[1]:.1f}%", f"{l[0]:+.1f}%／{l[1]:.1f}%", ui.esc(v)]
            for a, b, m, s, l, v in rv.V311_OVERLAY]
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>大跌時要不要減個股部位？（2021–2026，年化／最大回撤）</b></div>"
    h += ui.table(["", "規則", "中線", "短線", "長線", "判定"], rows, left=(0, 1, 5))
    h += f"<div class='kq-muted'>{ui.esc(rv.V311_NOTE)}</div>"
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>歷次「大跌時怎麼辦」的檢討</b></div>"
    h += ui.table(["做法", "版本", "結果", "現在怎麼做"], [[ui.esc(a), b, ui.esc(c), ui.esc(d)] for a, b, c, d in rv.V311_ANGLES], left=(0, 2, 3))
    return h


# ================================================================== v3.13 環境判定卡、估值溫度、各均線比例、底線、個股標籤（docs/v3.13_protocol.md）
ENV_WORLD = [("標普 500", "GSPC"), ("那斯達克", "IXIC"), ("費城半導體", "SOX")]


@st.cache_data(show_spinner=False)
def env_cards(bid) -> list:
    """台股加權＋美股三個指數的環境判定卡（美股依台股交易日對齊＝前一晚收盤）。"""
    from kosto import env as E
    out = []
    df, _ = market_frame(bid)
    if not df.empty:
        out.append(("台股加權", E.env_card(df["close"])))
    W = world_frame(bid)
    for nm, k in ENV_WORLD:
        if k in W:
            s = W[k].dropna()
            out.append((nm, E.env_card(s[s.diff().fillna(1) != 0])))       # 去掉美股休市時往前補的重複值
    return [(n, c) for n, c in out if c]


def env_panel_html() -> str:
    from kosto import env as E
    cards = env_cards(BID)
    if not cards:
        return ""

    def chip(k, zh):
        col = ui.UP if k >= 2 else ui.DOWN if k >= 0 else ui.FLAT
        return f"<span class='kq-tag' style='background:{col};color:#fff;border-color:{col}'>{ui.esc(zh)}</span>"
    rows, mrows = [], []
    for nm, c in cards:
        s_ = chip(c["short"], c["short_zh"]) + f" {ui.steps(c['short_hist'], 3)} <span class='kq-muted' style='font-size:11px'>{c['short_days']} 天</span>"
        l_ = chip(c["long"], c["long_zh"]) + f" {ui.steps(c['long_hist'], 3)} <span class='kq-muted' style='font-size:11px'>{c['long_days']} 天</span>"
        rows.append([f"<b>{ui.esc(nm)}</b><div class='kq-muted' style='font-size:12px'>{c['close']:,.2f}（{c['date'][5:]}）</div>", s_, l_])
        mrows.append(ui.mcard(f"<span class='nm'>{ui.esc(nm)}</span>", f"{c['close']:,.0f}",
                              f"<span style='font-size:13px'>短線 {chip(c['short'], c['short_zh'])} {ui.steps(c['short_hist'], 3, w=110)}</span>",
                              f"<span style='font-size:13px'>長線 {chip(c['long'], c['long_zh'])} {ui.steps(c['long_hist'], 3, w=110)}</span>"))
    return (ui.section("環境判定卡（v3.13）", "收盤站上幾條均線：短線 5／10／20 日、長線 20／60／120 日；3 條＝順風、2＝偏強、1＝偏弱、0＝逆風"
                       + ui.evidence("ref"))
            + ui.resp(ui.table(["指數", "短線（近 60 天）", "長線（近 60 天）"], rows, left=(0, 1, 2)), "".join(mrows))
            + "<div class='kq-muted' style='margin-bottom:16px'>參考 Yoda Research 綜合報告的版面重建（對方另有「拉扯」一級）。歷史檢查（2017–2026）：各狀態之後 20 日的大盤報酬差不多"
              "（短線逆風 +1.4%、順風 +1.7%）→ 用來看<b>現在的趨勢位置</b>，不是買賣訊號；美股為前一晚收盤。</div>")


# ================================================================== v3.16 每日台股盤前日報（kosto/morning.py；docs/v3.16_protocol.md）
@st.cache_data(ttl=300, show_spinner=False)
def morning_files() -> tuple[dict, list]:
    """(最新日報, 日報歷史)：GitHub 模式讀 kosto-research 的 morning.json；本機讀 artifacts/。"""
    import json
    rep, hist = {}, []
    if REPO:
        for nm in ("morning.json", "morning_hist.json"):
            try:
                x = json.loads(fetch_asset(REPO, RESEARCH_TAG, nm, TOKEN).decode("utf-8"))
                rep, hist = (x, hist) if nm == "morning.json" else (rep, x)
            except Exception:  # noqa: BLE001
                pass
    for nm in ("morning.json", "morning_hist.json"):
        p = Path("artifacts") / nm
        if p.is_file() and not (rep if nm == "morning.json" else hist):
            try:
                x = json.loads(p.read_text(encoding="utf-8"))
                rep, hist = (x, hist) if nm == "morning.json" else (rep, x)
            except Exception:  # noqa: BLE001
                pass
    return rep, hist


@st.cache_data(show_spinner=False)
def morning_score(bid, hist_json: str) -> dict:
    import json
    from kosto import morning as M
    return M.score(json.loads(hist_json or "[]"), B)


def morning_now() -> dict:
    """今天的日報：GitHub 排程產出的那一份；沒有或不是今天的 → 用本頁「立即產生」的那一份。"""
    from kosto import morning as M
    rep, _ = morning_files()
    day = str(M.report_date(pd.Timestamp.now(tz="Asia/Taipei")).date())
    live = st.session_state.get("kq_morning_live") or {}
    if live.get("date") == day and (rep.get("date") != day or live.get("generated_at", "") > rep.get("generated_at", "")):
        return live
    return rep


def _ev(kind: str) -> str:
    return ui.evidence("tested" if kind == "tested" else "ref")


def morning_html(rep: dict, hist: list | None = None) -> str:
    from kosto import morning as M
    pc = lambda x, d=1: "–" if x is None else ui.pct(x, d, color=True)          # noqa: E731
    p0 = lambda v: "–" if v is None else f"{v:.0%}"                              # noqa: E731
    f2 = lambda v: "–" if v is None else f"{v:.2f}"                              # noqa: E731
    gen = pd.Timestamp(rep.get("generated_at")).tz_convert("Asia/Taipei") if rep.get("generated_at") else None
    d = pd.Timestamp(rep["date"])
    st_txt = "完整" if rep.get("status") == "complete" else "部分資料缺（" + "、".join(
        {"us": "美股", "night": "夜盤", "heavy": "權值股", "news": "新聞", "link": "開盤預估"}[k] for k, v in (rep.get("sections_ok") or {}).items() if not v) + "）"
    h = ui.section(f"🌅 台股盤前日報 {d:%m/%d}（週{'一二三四五六日'[d.weekday()]}）",
                   f"產出 {gen:%m/%d %H:%M}｜美股資料日 {ui.esc(rep.get('us_date') or '–')}｜研究資料到 {ui.esc(rep.get('model_asof') or '–')}｜{ui.esc(st_txt)}"
                   if gen is not None else "")
    if rep.get("headline"):
        h += "<div class='kq-verdict'>" + "<br>".join(f"<b>{ui.esc(x)}</b>" if k == 0 else ui.esc(x) for k, x in enumerate(rep["headline"])) + "</div>"
    lk = rep.get("link") or {}
    pr = lk.get("pred") or {}
    m1 = lk.get("m1") or {}
    cards = []
    if pr:
        t1 = m1.get("test") or {}
        cards.append(("預估大盤開盤" + _ev("tested" if pr.get("tested") else "ref"), pc(pr["gap"]),
                      f"區間 {ui.pct(pr['lo'], 1)}～{ui.pct(pr['hi'], 1)}｜2024 年後方向命中 {p0(t1.get('hit'))}"))
    tsm = next((r for r in (rep.get("heavy") or {}).get("rows", []) if r["ticker"] == "2330"), {})
    if pr.get("g2330") is not None:
        adr = tsm.get("adr") or {}
        cards.append(("台積電預估開盤" + _ev("tested" if pr.get("g2330_tested") else "ref"), pc(pr["g2330"]),
                      f"ADR {pc(adr.get('chg'), 2)}｜換算溢價 {ui.pct(adr.get('premium'), 1)}" if adr else ""))
    us = {r["key"]: r for r in (rep.get("us") or {}).get("rows", [])}
    if "SOX" in us:
        cards.append(("費城半導體", pc(us["SOX"]["chg"], 2), f"那指 {pc(us.get('IXIC', {}).get('chg'), 2)}｜VIX {f2(us.get('VIX', {}).get('price'))}"))
    nt = rep.get("night") or {}
    tx = nt.get("tx") or nt.get("nq") or nt.get("nq_day")
    cards.append(("夜盤／期貨" + _ev("ref"), pc((tx or {}).get("chg"), 2) if tx else "–",
                  "台指期夜盤" if nt.get("tx") else "台股收盤後那指期" if nt.get("nq") else "那指期（相對前一日）" if nt.get("nq_day") else "見下方說明"))
    h += ui.cards(cards, raw=True)
    # 1 美股
    h += ui.section("1. 美股收盤總評", ui.evidence("ref"))
    h += f"<div style='margin:2px 0 6px'>{ui.esc((rep.get('us') or {}).get('summary', ''))}</div>"
    rows = [[ui.esc(r["name"]), f"{r['price']:,.2f}" if r.get("price") is not None else "–", pc(r.get("chg"), 2), ui.esc(r.get("bar_date") or "")]
            for r in (rep.get("us") or {}).get("rows", [])]
    if rows:
        h += ui.table(["指數／商品", "收盤", "漲跌", "資料日"], rows, left=(0,))
    # 2 夜盤
    h += ui.section("2. 夜盤趨勢", "台指期夜盤 15:00～05:00；連不到期交所時改看台股收盤後的美股期貨" + ui.evidence("ref"))
    h += f"<div>{ui.esc(nt.get('text', ''))}</div>"
    # 3 權值股
    h += ui.section("3. 權值股動態", "昨收與外資（研究資料）、ADR 換算價（ADR 收盤 × 匯率 ÷ 換股比例）、對應的美股同業" + ui.evidence("ref"))
    rows = []
    for r in (rep.get("heavy") or {}).get("rows", []):
        adr = r.get("adr") or {}
        rows.append([f"<b>{r['ticker']}</b> {ui.esc(r['name'])}", f"{r['close']:,.1f}" if r.get("close") else "–", pc(r.get("chg"), 2),
                     (f"{r['fi1']:+,.0f}／{r['fi5']:+,.0f}" if r.get("fi1") is not None else "–"),
                     (f"{ui.esc(adr['name'])} {pc(adr['chg'], 2)}｜換算 {adr['implied']:,.1f}（溢價 {ui.pct(adr['premium'], 1)}）" if adr else "–"),
                     "、".join(f"{ui.esc(p['name'])} {pc(p['chg'], 1)}" for p in r.get("peers", []) if p["name"] != adr.get("name")) or "–"])
    if rows:
        h += ui.table(["權值股", "昨收", "漲跌", "外資 今日／5日（張）", "ADR", "美股同業"], rows, left=(0, 4, 5))
    # 4 消息
    h += ui.section("4. 重大消息", "近 24 小時新聞標題（美股、台股、台積電、聯準會、輝達），依關鍵字標主題" + ui.evidence("ref"))
    items = (rep.get("news") or {}).get("items") or []
    if items:
        h += "<ul style='margin:4px 0 6px 18px;padding:0'>" + "".join(
            f"<li style='margin:2px 0'><a href='{ui.esc(x['link'])}' target='_blank' rel='noopener'>{ui.esc(x['title'])}</a>"
            f"<span class='kq-sub'>　{ui.esc(x.get('source') or '')} {pd.Timestamp(x['time']).tz_convert('Asia/Taipei'):%m/%d %H:%M}</span>"
            + "".join(ui.tag(t, "g") for t in x.get("tags") or []) + "</li>" if x.get("time") else
            f"<li><a href='{ui.esc(x['link'])}' target='_blank' rel='noopener'>{ui.esc(x['title'])}</a></li>" for x in items) + "</ul>"
    else:
        h += "<div class='kq-muted'>新聞來源暫時取不到（不影響其他段落）。</div>"
    # 5 關聯性
    h += ui.section("5. 影響台股關聯性", "前一夜美股 → 今天台股開盤的歷史關係（2024 年起為檢驗期，不參與估計）")
    if pr:
        x = pr.get("x") or {}
        t1, t2 = m1.get("test") or {}, ((lk.get("m2") or {}).get("test") or {})
        h += (f"<div style='margin:2px 0 6px'>依前一夜費半 {pc(x.get('sox'), 2)}、那指 {pc(x.get('ixic'), 2)}、台積電 ADR（台幣計）{pc(x.get('tsm_twd'), 2)}、"
              f"VIX {x.get('dvix', 0):+.1f} → <b>預估大盤開盤 {pc(pr['gap'])}</b>（誤差帶 ±{(m1.get('rmse') or 0) * 100:.1f}%）{_ev('tested' if pr.get('tested') else 'ref')}<br>"
              f"<span class='kq-muted'>檢驗：2024 年後 {t1.get('n', 0)} 天，預估與實際相關 {f2(t1.get('corr'))}、"
              f"預估 |跳空|≥0.3% 的 {t1.get('n_hit', 0)} 天方向命中 {p0(t1.get('hit'))}；台積電開盤相關 {f2(t2.get('corr'))}。"
              "大盤＝市值前 50 名市值加權開盤，與加權指數接近。</span></div>")
        bks = lk.get("buckets") or []
        if bks:
            cur = pr.get("bucket")
            h += ui.table(["預估開盤", "天數（2024 年後）", "實際開盤平均", "開盤→收盤（大型股）"],
                          [[("<b>▶ " if b["bucket"] == cur else "") + ui.esc(b["bucket"]) + ("</b>" if b["bucket"] == cur else ""), f"{int(b['n'])}",
                            pc(b["gap"], 2), pc(b["oc"], 2)] for b in bks], left=(0,))
        m3, m4 = lk.get("m3") or {}, lk.get("m4") or {}
        h += (f"<div class='kq-muted'>開高不追：預估開高 ≥1% 的 {m3.get('n', 0)} 天，大型股開盤→收盤平均 {ui.pct(m3.get('oc300'), 2)}"
              f"（{(m3.get('neg') or 0):.0%} 收在開盤價之下）{_ev('tested' if m3.get('tested') else 'ref')}；"
              f"大跌不恐慌：預估開低 ≤−1% 的 {m4.get('n', 0)} 天，開盤→收盤平均 {ui.pct(m4.get('oc300'), 2)}{_ev('tested' if m4.get('tested') else 'ref')}</div>")
    else:
        h += "<div class='kq-muted'>缺少研究模型或美股報價，今天沒有開盤預估。</div>"
    th = lk.get("themes") or []
    if th:
        h += "<div style='margin:8px 0 2px'><b>美股族群大漲跌 → 台股相關族群</b>" + ui.evidence("ref") + "</div>"
        h += ui.table(["主題", "美股", "台股相關"], [[f"<b>{ui.esc(t['theme'])}</b> {pc(t['avg'], 1)}",
                                                   "、".join(f"{ui.esc(u['name'])} {pc(u['chg'], 1)}" for u in t["us"]), ui.esc(t["tw"])] for t in th],
                      left=(0, 1, 2))
    # 6 建議
    adv = rep.get("advice") or {}
    h += ui.section("6. 今日操作建議", "只用經過歷史檢驗的規則與模型名單；每點標證據等級")
    col = lambda nm, xs, c: (f"<div class='kq-panel' style='border-top:3px solid {c}'><h4>{nm}</h4><ol style='margin:4px 0 0 18px;padding:0'>"   # noqa: E731
                             + "".join(f"<li style='margin:4px 0'>{ui.esc(x['text'])}{_ev(x['ev'])}</li>" for x in xs) + "</ol></div>")
    h += ("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px'>"
          + col("🛡️ 保守型", adv.get("conservative") or [], "#2563eb") + col("🚀 積極型", adv.get("aggressive") or [], "#dc2626") + "</div>")
    if hist:
        sc = morning_score(BID, M.dumps(hist))
        if sc.get("n"):
            h += (f"<div class='kq-muted' style='margin-top:8px'>日報前瞻紀錄：{sc['n']} 天已有實際開盤，方向命中 "
                  f"{p0(sc.get('hit'))}（{sc['n_dir']} 天），平均誤差 {sc['mae'] * 100:.2f}%" + (f"；{sc['pending']} 天待研究資料更新" if sc.get("pending") else "") + "</div>")
    h += f"<div class='kq-muted' style='margin-top:6px'>{ui.esc(rep.get('disclaimer', ''))}</div>"
    return h


def morning_tab():
    from kosto import morning as M
    now = pd.Timestamp.now(tz="Asia/Taipei")
    day = str(M.report_date(now).date())
    rep, hist = morning_files()
    cur = morning_now()
    if cur.get("date") != day:
        st.markdown(ui.section("🌅 台股盤前日報", f"每個交易日台北 05:43 自動產出（06:13、06:38 備援），07:00 前可看；{day} 的日報還沒有產出"),
                    unsafe_allow_html=True)
        if cur:
            st.caption(f"以下是最近一份（{cur.get('date')}）。")
    if st.button("立即產生今日盤前日報（約 20～40 秒）", key="kq_morning_btn",
                 help="抓最新的美股收盤、期貨、ADR 與新聞，用每日研究的模型做開盤預估（不會上傳，只在這次瀏覽使用）"):
        with st.spinner("抓取美股、期貨、ADR 與新聞…"):
            st.session_state["kq_morning_live"] = M.build((RES or {}).get("morning_model") or {}, M.fetch_live(now), RES or {}, now)
        cur = morning_now()
    if cur:
        st.markdown(morning_html(cur, hist), unsafe_allow_html=True)
    else:
        st.info("還沒有任何盤前日報：第一次請按上方按鈕，或到 GitHub Actions 執行「Kosto 盤前日報」。")


def morning_banner() -> str:
    """盤勢儀表板頂端：今日盤前日報的重點與兩種建議的第一點（全文在「🌅 盤前日報」分頁）。"""
    try:
        from kosto import morning as M
        cur = morning_now()
        if cur.get("date") != str(M.report_date(pd.Timestamp.now(tz="Asia/Taipei")).date()):
            return ""
        hd = cur.get("headline") or []
        adv = cur.get("advice") or {}
        pick = lambda k: next((x["text"] for x in adv.get(k) or [] if "預估" in x["text"] or "開" in x["text"][:4]), "")   # noqa: E731
        c, g = pick("conservative"), pick("aggressive")
        return ("<div class='kq-verdict' style='border-left-color:#f59e0b'><b>🌅 今日盤前日報</b>　" + "｜".join(ui.esc(x) for x in hd[:2])
                + (f"<br><span class='kq-muted'>保守：{ui.esc(c)}</span>" if c else "") + (f"<br><span class='kq-muted'>積極：{ui.esc(g)}</span>" if g else "")
                + "<br><span class='kq-sub'>全文（美股、夜盤、權值股、消息、關聯性、建議）在下方第一個分頁「🌅 盤前日報」</span></div>")
    except Exception:  # noqa: BLE001
        return ""


def morning_home_line() -> str:
    try:
        from kosto import morning as M
        cur = morning_now()
        if cur.get("date") != str(M.report_date(pd.Timestamp.now(tz="Asia/Taipei")).date()):
            return ""
        pr = (cur.get("link") or {}).get("pred") or {}
        hd = cur.get("headline") or []
        return ("<br><span class='kq-muted'>🌅 今日盤前日報：" + ui.esc(hd[0] if hd else "")
                + ("" if pr else "") + "（<a href='./insight' target='_self'>盤勢儀表板 → 盤前日報</a>）</span>")
    except Exception:  # noqa: BLE001
        return ""


def env_tab_html(df: pd.DataFrame) -> str:
    """🌡️ 環境與估值：估值溫度、站上各均線比例、底線（關鍵價位）、歷史檢查。"""
    from kosto import env as E
    from kosto import review as rv
    h = ""
    v = B.meta.get("valuation") or (E.valuation(df["pe_mkt"]) if "pe_mkt" in df else {})
    if v:
        pct = v.get("pct")
        hist = [x[1] for x in v.get("hist", [])] or df["pe_mkt"].dropna().tolist()
        txt = (f"比 {v['since'][:4]} 年以來 <b>{pct:.0%}</b> 的時間貴" if ui.ok(pct) else "歷史百分位：下一次每日研究後提供")
        h += ui.section("估值溫度（台股加權）", f"市值加權本益比（排除虧損股），資料日 {v['asof']}")
        h += ui.cards([("本益比", f"{v['pe']:.1f} 倍", txt), ("距近一年高點", ui.pct(v.get("off_high_12m"), 1), f"一年高點 {v.get('high_12m', np.nan):.1f} 倍"),
                       ("2017 年以來", f"{v.get('min', np.nan):.1f}～{v.get('max', np.nan):.1f} 倍", f"中位數 {v.get('median', np.nan):.1f} 倍")], raw=True)
        h += f"<div style='margin:4px 0'>{ui.spark(hist, w=600, h=60, fluid=True)}</div>"
        h += ui.table(["本益比歷史百分位", "天數", "之後 60 日", "之後一年", "一年後上漲比例"],
                      [[f"<b>{a}</b>", str(n), f"{b:+.1f}%", f"{c:+.1f}%", f"{d:.0f}%"] for a, n, b, c, d in rv.V313_T3], left=(0,))
        h += ("<div class='kq-muted'>2017–2026 只有約 10 年、一次大循環，樣本少：估值高（≥90%）之後一年平均仍 +20%，只是比 50～90% 的 +43% 少；"
              "估值低（<50%）多在 2018–2019、2022 這種空頭，之後也只有 +11%。估值告訴你「貴不貴」，不告訴你「什麼時候跌」。</div>")
    cols = [c for c in ("ab5", "ab10", "ab20", "ab60", "ab120", "ab240") if c in df and df[c].notna().any()]
    if cols:
        lab = {"ab5": "5 日線", "ab10": "10 日線", "ab20": "月線（20）", "ab60": "季線（60）", "ab120": "半年線（120）", "ab240": "年線（240）"}
        rows = []
        for c in cols:
            s_ = df[c].dropna()
            x = float(s_.iloc[-1])
            zone = "偏多" if x >= 0.6 else "拉鋸" if x >= 0.4 else "偏空"
            col = ui.UP if x >= 0.6 else ui.DOWN if x < 0.4 else ui.FLAT
            ch5 = x - float(s_.iloc[-6]) if len(s_) > 6 else np.nan
            rows.append([f"<b>{lab[c]}</b>", f"<b style='color:{col}'>{x:.0%}</b>", ui.esc(zone),
                         ("–" if not ui.ok(ch5) else f"{ch5 * 100:+.0f} 個百分點"), ui.spark(s_.tail(120).tolist(), w=160, h=26)])
        h += ui.section("站上各條均線的股票比例", "上市櫃可交易股票；≥60% 偏多、40～60% 拉鋸、<40% 偏空")
        h += ui.table(["均線", "目前", "區間", "近 5 日變化", "近半年"], rows, left=(0, 2))
        if len(cols) < 6:
            h += "<div class='kq-muted'>5／10／120／240 日線比例：下一次每日研究（v3.13 起）後提供。</div>"
    lv = E.key_levels(df["close"])
    if lv:
        px = float(df["close"].dropna().iloc[-1])
        h += ui.section("底線：加權指數的關鍵價位", f"{df['close'].dropna().index[-1]:%m/%d} 收盤 {px:,.0f}；跌破代表那個期間的趨勢判斷要重新檢視")
        h += ui.table(["價位", "點數", "收盤距離", "狀態"],
                      [[ui.esc(x["name"]), f"{x['level']:,.0f}", ui.pct(x["dist"], 1, color=True), "站上" if x["above"] else "<b>跌破</b>"] for x in lv],
                      left=(0, 3))
    h += ui.section("環境判定卡的歷史檢查（加權指數，2017–2026）", "各狀態之後的大盤報酬；最後一欄＝中線模型之後 20 日比 0050 多賺")
    h += ui.table(["卡", "狀態", "天數", "之後 20 日", "20 日上漲比例", "之後 60 日", "2017–23 之後 20 日", "2024–26 之後 20 日", "中線模型超額"],
                  [[a, f"<b>{b}</b>", str(n), f"{c:+.1f}%", f"{d:.0f}%", f"{e:+.1f}%", f"{f:+.1f}%", f"{g:+.1f}%", f"{i:+.1f}%"]
                   for a, b, n, c, d, e, f, g, i in rv.V313_T1], left=(0, 1))
    h += ui.table(["部位規則", "", "中線 年化／回撤", "短線", "長線", "判定"],
                  [[f"<b>{a}</b>", ui.esc(b), f"{m[0]:+.1f}%／{m[1]:.1f}%", f"{s_[0]:+.1f}%／{s_[1]:.1f}%", f"{l_[0]:+.1f}%／{l_[1]:.1f}%", ui.esc(j)]
                   for a, b, m, s_, l_, j in rv.V313_POS], left=(0, 1, 5))
    h += ui.table(["跌破 50 或 120 日線時的「止跌條件」", "次數", "之後 20 日", "上漲比例", "之後 60 日", "20 日內再跌"],
                  [[ui.esc(a), str(n), f"{b:+.1f}%", f"{c:.0f}%", f"{d:+.1f}%", f"{e:+.1f}%"] for a, n, b, c, d, e in rv.V313_T2], left=(0,))
    h += ("<div class='kq-muted'>結論：環境判定卡與止跌條件都只能描述「現在在哪裡」，之後的大盤報酬和沒有訊號時差不多；"
          "依卡片減碼（P5、P6）也沒有改善回撤。本系統沒有 VIX 期貨與信用利差資料，止跌條件以那斯達克、VIX 均線代替。</div>")
    return h


@st.cache_data(show_spinner=False)
def tag_context(bid) -> dict:
    """個股標籤需要的全市場資料：族群帶頭名單、最近一次營收公布的反應分類。"""
    from kosto.sources.official import revenue_available_date
    from kosto import env as E
    out = {"lead": set(), "react": {}, "react_ym": None}
    if "Q_C" not in B.a:
        return out
    days = pd.DatetimeIndex(B.meta["q_days"])
    C = pd.DataFrame(B.a["Q_C"].astype(float), index=days, columns=B.tickers).ffill()
    cap = B.a["CAP"][-1] if "CAP" in B.a else np.ones(len(B.tickers))
    top = pd.Index(B.tickers)[np.argsort(-np.nan_to_num(cap))[:800]]
    r20 = (C.iloc[-1] / C.iloc[-21] - 1).reindex(top)
    g = pd.Series({t: ((CH.get(t) or {}).get("segment") or (CH.get(t) or {}).get("group") or "") for t in top})
    t_ = pd.DataFrame({"g": g, "r": r20}).dropna()
    t_ = t_[t_["g"].astype(str).str.len() > 0]
    gm = t_.groupby("g")["r"].agg(["median", "size"])
    gm = gm[gm["size"] >= 5]
    if len(gm):
        for gname in gm[gm["median"] >= gm["median"].quantile(0.8)].index:
            out["lead"].add(t_[t_["g"] == gname]["r"].idxmax())
    if "REV_M" in B.a and B.meta.get("rev_ym"):
        rm = pd.DataFrame(B.a["REV_M"].astype(float), index=B.meta["rev_ym"], columns=B.tickers)
        s3 = rm.rolling(3, min_periods=3).sum()
        for ym in reversed(list(rm.index)[15:]):
            k = int(days.searchsorted(revenue_available_date(ym)))
            if k + 2 >= len(days) or k < 3:
                continue
            base = s3.loc[ym]
            prev = s3.iloc[list(rm.index).index(ym) - 12] if list(rm.index).index(ym) >= 12 else None
            if prev is None:
                continue
            yoy = (base / prev - 1).where(prev > 0).reindex(top)
            rr = (C.iloc[k + 2] / C.iloc[k - 3] - 1).reindex(top)
            rr = rr - rr.mean()
            ok = yoy.notna() & rr.notna()
            if ok.sum() < 200:
                continue
            yr, rk = yoy[ok].rank(pct=True), rr[ok].rank(pct=True)
            out["react"] = {t: E.revenue_reaction(float(yr[t]), float(rk[t])) for t in yr.index}
            out["react_ym"] = int(ym)
            break
    return out


def stock_tags_html(t: str, j: int) -> str:
    """個股診斷：Yoda 式標籤＋營收公布反應（只供參考，附歷史驗證）。"""
    from kosto import env as E
    from kosto import review as rv
    if "Q_C" not in B.a:
        return ""
    days = pd.DatetimeIndex(B.meta["q_days"])
    col = lambda k: pd.Series(B.a[k][:, j].astype(float), index=days) if k in B.a else pd.Series(np.nan, index=days)   # noqa: E731
    fo = col("Q_FI")
    tot = fo.fillna(0) + col("Q_TR").fillna(0) + col("Q_DL").fillna(0)
    tags = E.stock_tags(col("Q_C"), col("Q_V"), fo, tot.where(fo.notna()))
    ctx = tag_context(BID)
    if t in ctx["lead"]:
        tags.append("LEAD")
    ev = {k: (a, b) for k, a, b, _ in rv.V313_T4E}
    parts = [f"<span class='kq-tag'>{ui.esc(E.TAGS[k])}</span>" for k in tags]
    note = "；".join(f"{E.TAGS[k]}：之後 20 日比大盤 2018–23 {ev[k][0]:+.1f}%、2024–26 {ev[k][1]:+.1f}%" for k in tags if k in ev)
    rc = ctx["react"].get(t)
    t5 = {k: (n, a, c) for k, n, a, _, c, _, _ in rv.V313_T5}
    rtxt = ""
    if rc and ctx.get("react_ym"):
        ym = ctx["react_ym"]
        n, a, c = t5[rc]
        rtxt = (f"<div class='kq-muted'>{ym // 100}/{ym % 100:02d} 營收公布反應：<b>{ui.esc(n)}</b>（歷史上之後 20 日比大盤 2019–23 {a:+.1f}%、2024–26 {c:+.1f}%）</div>")
    if not parts and not rtxt:
        return ""
    head = ("<div class='kq-muted'>Yoda 式標籤（v3.13）" + ui.evidence("ref") + "：" + (" ".join(parts) if parts else "無") + "</div>")
    tail = (f"<div class='kq-muted' style='font-size:12px'>{ui.esc(note)}。這些標籤加進模型三個期間都變差，"
            "代表「籌碼熱、剛創高」之後不一定比較好；模型名單與排名仍以上方為準。</div>" if note else "")
    return head + tail + rtxt


def v313_html() -> str:
    """模型體檢：v3.13 和 Yoda 綜合報告截長補短的歷史檢查。"""
    from kosto import review as rv
    h = ui.section("和 Yoda 綜合報告截長補短（v3.13）", "對方的判讀方法先用 2017–2026 資料檢查；預先宣告：docs/v3.13_protocol.md")
    h += ui.table(["Yoda 的做法", "歷史檢查", "App 怎麼用"], [[ui.esc(a), ui.esc(b), ui.esc(c)] for a, b, c in rv.V313_COMPARE], left=(0, 1, 2))
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>選股標籤加進模型（各 15%，五種換股日，百分點）</b></div>"
    f = lambda v: "0" if abs(v) < 0.005 else f"{v:+.2f}"          # noqa: E731
    h += ui.table(["", "標籤", "中線超額", "中線 IR", "中線回撤", "短線超額", "長線超額", "整段樣本外 短／中／長", "中線為正", "判定"],
                  [[f"<b>{a}</b>", ui.esc(b), f(c), f"{d:+.3f}", f(e), f(g), f(i), ui.esc(k), ui.esc(m), ui.esc(n)]
                   for a, b, c, d, e, g, i, k, m, n in rv.V313_T4], left=(0, 1, 9))
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>營收公布後的反應（成交金額前 800 名，每月一次）：之後 20／60 日比大盤多</b></div>"
    h += ui.table(["分類", "2019–23 20 日", "60 日", "2024–26 20 日", "60 日", "每月平均檔數"],
                  [[f"<b>{ui.esc(n)}</b>", f"{a:+.1f}%", f"{b:+.1f}%", f"{c:+.1f}%", f"{d:+.1f}%", str(m)] for _, n, a, b, c, d, m in rv.V313_T5], left=(0,))
    h += f"<div class='kq-muted'>{ui.esc(rv.V313_NOTE)}</div>"
    return h


def v312_html() -> str:
    """v3.12 回頭檢查：稽核、觀察清單重測、觀察名單前瞻追蹤（每日影子模型，kosto/watch.py）。"""
    from kosto import review as rv
    from kosto import watch as wt
    f = lambda v: "0" if abs(v) < 0.005 else f"{v:+.2f}"          # noqa: E731
    h = ui.section("回頭檢查：有沒有被遺漏或改掉的好策略？（v3.12）",
                   "逐版比對設定與程式；重測歷次「列入觀察」的候選；之後用新資料每天追蹤")
    h += "<div class='kq-muted' style='margin:6px 0 2px'><b>A. 稽核：設定、程式、頁面有沒有被不小心改掉</b></div>"
    h += ui.table(["項目", "方法", "結果"], [[f"<b>{ui.esc(a)}</b>", ui.esc(b), ui.esc(c)] for a, b, c in rv.V312_AUDIT], left=(0, 1, 2))
    h += ("<div class='kq-muted' style='margin:10px 0 2px'><b>B. 重測歷次「列入觀察」的候選</b>（五種換股日、2021–2025 開發期；百分點，"
          "回撤改善為正；預先宣告：docs/v3.12_protocol.md）</div>")
    rows = [[f"<b>{a}</b>", ui.esc(b), f(c), f(d), f(e), f(g), f(i), f(j), ui.esc(k), ui.esc(m), ui.esc(n)]
            for a, b, c, d, e, g, i, j, k, m, n in rv.V312_TESTS]
    h += ui.table(["", "候選", "中線超額", "中線 IR", "中線回撤", "短線超額", "長線超額", "長線回撤", "整段樣本外 短／中／長", "一致性", "判定"],
                  rows, left=(0, 1, 10))
    h += f"<div class='kq-muted'>{ui.esc(rv.V312_NOTE)}</div>"
    w = (RES or {}).get("watch") or {}
    h += (f"<div class='kq-muted' style='margin:10px 0 2px'><b>C. 觀察名單前瞻追蹤</b>（影子模型：同資料、同母池、同權重流程，只多一個改動；"
          f"比較 {wt.SINCE} 之後的新資料，滿 {wt.NEED_DAYS} 個交易日判定；不影響選股）</div>")
    if w.get("rows"):
        rows = []
        for r in w["rows"]:
            pct = lambda v: "–" if v is None or not ui.ok(v) else ui.pct(v, 1, color=True)       # noqa: E731
            st_ = f"<b>{ui.esc(r['status_zh'])}</b>" if r["status"] in ("pass", "fail") else ui.esc(r["status_zh"])
            rows.append([f"<b>{r['key']}</b> {ui.esc(r['name'])}", HZ.get(r["h"], r["h"]), wt.KIND_ZH[r["kind"]], ui.esc(r["pool"]),
                         f"{r['days']} 日" + (f"（{r['first']}～{r['last']}）" if r.get("first") else ""),
                         f"{pct(r.get('ret_var'))}／{pct(r.get('ret_base'))}", pct(r.get("d_ret")), pct(r.get("d_mdd")), st_])
        h += ui.table(["影子模型", "期間", "類型", "母池", "前瞻期", "累積報酬 影子／現行", "差", "最大回撤差", "狀態"], rows, left=(0, 8))
        if w.get("errors"):
            h += "<div class='kq-muted'>" + ui.esc("；".join(w["errors"])) + "</div>"
    else:
        h += ui.table(["影子模型", "期間", "類型", "為什麼在名單上"],
                      [[f"<b>{it['key']}</b> {ui.esc(it['name'])}", "、".join(HZ.get(x, x) for x in it["hz"]), wt.KIND_ZH[it["kind"]],
                        ui.esc(it["why"])] for it in wt.ITEMS], left=(0, 3))
        h += "<div class='kq-muted'>下一次每日研究（v3.12 起）開始計算。</div>"
    h += ("<div class='kq-muted'>判定規則（預先寫好）：報酬型＝" + ui.esc(wt.RULE["return"]) + "；平衡型＝" + ui.esc(wt.RULE["balanced"])
          + "。「前瞻支持」才在下一版正式採用；「前瞻不支持」就移出觀察、不再重測。</div>")
    return h


def v37_html() -> str:
    """v3.7 週刊觀點的歷史驗證（資安事件、巨型 IPO、搶反彈三條件）。"""
    from kosto import review as rv
    f = lambda v: "–" if v is None else f"{v:+.1f}%"          # noqa: E731
    h = ui.section("週刊觀點的歷史驗證（v3.7～v3.9）", "先寫規則再跑；只報告，主模型不變")
    h += "<div class='kq-muted' style='margin:6px 0 2px'><b>A. 重大駭客事件後，資安股有沒有超額報酬？</b></div>"
    h += ui.table(["市場", "事件數", "事件後", "平均多大盤", "平常同期", "上漲比例 事件／平常"],
                  [[ui.esc(a), str(n), b, c, d, e] for a, n, b, c, d, e in rv.V37_CYBER], left=(0,))
    h += f"<div class='kq-muted'>{ui.esc(rv.V37_CYBER_NOTE)}</div>"
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>B. 巨型 IPO 掛牌後，大盤會不會回檔？</b></div>"
    h += ui.table(["掛牌日", "IPO（募資，美元）", "標普 20 日", "60 日", "60 日內最大回落", "加權 20 日", "60 日", "最大回落"],
                  [[f"<b>{a}</b>", ui.esc(b), f(c), f(d), f(e), f(g), f(i), f(j)] for a, b, c, d, e, g, i, j in rv.V37_IPO], left=(0, 1))
    h += f"<div class='kq-muted'>{ui.esc(rv.V37_IPO_NOTE)}</div>"
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>C. 搶反彈「跌深＋融資減＋低本益比」（台灣 50 成分股，修正期）</b></div>"
    h += ui.table(["選法（各取前 5 名）", "開發期 20 日", "開發期 60 日", "驗證期 20 日", "驗證期 60 日"],
                  [[ui.esc(a), f(b), f(c), f(d), f(e)] for a, b, c, d, e in rv.V37_REBOUND], left=(0,))
    h += f"<div class='kq-muted'>{ui.esc(rv.V37_REBOUND_NOTE)}</div>"
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>D. 美國選舉後，電力／綠能股會不會比較強？（v3.8）</b></div>"
    h += ui.table(["勝選", "次數", "5 日", "20 日", "60 日", "公用事業 XLU 60 日", "60 日贏 SPY"],
                  [[ui.esc(a), "–" if n is None else str(n), f(b), f(c), f(d), f(e), w] for a, n, b, c, d, e, w in rv.V38_ELECT], left=(0,))
    h += f"<div class='kq-muted'>{ui.esc(rv.V38_ELECT_NOTE)}</div>"
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>E. 週刊的「前瞻」選股理由，歷史上準不準？（v3.9：三率三升、獲利上修、成長合理價）</b></div>"
    g = lambda v: f"{v:+.3f}"          # noqa: E731
    h += ui.table(["因子（之後 60 日報酬的 IC）", "2019–26", "2022", "2023", "2024", "2025", "2026", "模型"],
                  [[ui.esc(a), f"<b>{g(b)}</b>", g(c), g(d), g(e), g(f_), g(i), ui.esc(j)] for a, b, c, d, e, f_, i, j in rv.V39_IC], left=(0, 7))
    h += f"<div class='kq-muted'>{ui.esc(rv.V39_NOTE)}</div>"
    return h


def insight_rebound() -> str:
    rb = B.meta.get("rebound") or []
    head = ui.section("修正期反彈候選（與主模型分開）",
                      f"依 {B.meta.get('rebound_asof', '')} 收盤；成交金額前 500 名中「近 20 日跌多、距 60 日高點跌深、營收年增優於同業、投信沒有追高」等權排名")
    if not rb:
        return head + "<div class='kq-muted'>下一次每日研究後提供。</div>"
    inpick = {}
    for h in HORIZON_KEYS:
        for p in latest(h).get("picks", []):
            inpick.setdefault(p["ticker"], []).append(HZ[h])
    rows, mrows = [], []
    for k, x in enumerate(rb[:20], 1):
        lab = CH.get(x["ticker"]) or chainlib.label(x["ticker"], x.get("sector", ""))
        g = lab.get("segment") or lab.get("group") or ""
        also = "、".join(inpick.get(x["ticker"], []))
        name = f"<b>{x['ticker']} {ui.esc(x['name'])}</b>" + (f" <span class='kq-tag'>也在{also}名單</span>" if also else "")
        rows.append([f"{k}", name, ui.esc(g), ui.num(x["close"]), ui.pct(x["ret20"], 1, color=True), ui.pct(x["dd60"], 1),
                     f"{x['rev_rel'] * 100:+.0f}pp" if ui.ok(x.get("rev_rel")) else "–", ui.pct(x["trust20"], 2), f"{x['score']:.2f}"])
        mrows.append(ui.mcard(f"<span class='nm'>{k}. {x['ticker']} {ui.esc(x['name'])}</span>", ui.num(x["close"]),
                              f"近20日 {ui.pct(x['ret20'], 1, color=True)}｜距60日高 {ui.pct(x['dd60'], 1)}｜{ui.esc(g)}",
                              (f"營收年增比同業 {x['rev_rel'] * 100:+.0f}pp｜" if ui.ok(x.get("rev_rel")) else "") + f"分數 {x['score']:.2f}"
                              + (f"｜也在{also}名單" if also else "")))
    return (head + ui.resp(ui.table(["#", "股票", "族群", "收盤", "近20日", "距60日高", "近3月營收年增 vs 同業", "投信20日/股本", "反彈分數"],
                                    rows, left=(1, 2)), "".join(mrows))
            + "<div class='kq-muted'>研究（docs/v2.0_rebound_protocol.md）：2017–2026 共 6 次修正中，這個分數的前 20 名之後 20 日平均 +6%、"
              "勝過同期大型股平均約 3 個百分點（2024–2026 驗證期 +6.3% vs 平均 +3.3%、0050 +5.2%）；但 2024 年 8 月那次輸給 0050。"
              "跌深股波動也大，請小量、分批，並設停損。兩份名單重疊的股票最值得優先看。"
              "v3.7 另測週刊常見的「跌深＋融資減＋低本益比」：低本益比在修正期最弱，60 日落後 0050，所以沒有改這份名單（見模型體檢）。</div>")


def dash_today(cur):
    """今日盤中：加權分時大圖＋盤中止跌觀察（加權、那指期、台積電）＋權值股小圖。"""
    from kosto import market as mklib
    d = taiex_intraday()
    wd = world_intraday(["NQF", "SOX"])
    tw = tw_intraday(["2330"] + [t for t in MARKET_REF_TW if t != "2330"])
    checks = mklib.intraday_checks(d, wd.get("NQF"), tw.get("2330")) if d is not None else []
    c1, c2 = st.columns([1.75, 1])
    with c1:
        st.markdown("<div class='kq-h' style='margin:8px 0 4px'><span class='t'>加權指數｜今日走勢</span>"
                    "<span class='s'>分時｜虛線＝平盤（昨收），線在上方紅、下方綠</span></div>", unsafe_allow_html=True)
        if d is None:
            st.markdown("<div class='kq-muted'>分時資料暫時無法取得（Yahoo Finance），請稍後重新整理。</div>", unsafe_allow_html=True)
        else:
            st.markdown(intraday_stats(d, 2), unsafe_allow_html=True)
            intraday_chart(d, key="dash_tx", height=230 if IS_MOBILE else 300, digits=2, volume=False)
    body = ""
    if checks:
        body += "".join(f"<div class='kq-kv'><span>{'✅' if ok_ else ('⬜' if ok_ is not None else '·')} {ui.esc(nm)}</span>"
                        f"<span class='kq-mono'>{ui.esc(v)}</span></div>" for nm, v, ok_ in checks)
        body += (f"<div class='kq-muted' style='margin:6px 0 8px'>成立 {sum(1 for *_, x in checks if x)}/{len(checks)} 項。"
                 "只看當天，用來決定「今天要不要先買第一批」；收盤前仍可能翻轉。</div>")
    rows = [("那指期", wd.get("NQF"), ""), ("費半(前一晚)", wd.get("SOX"), "")]
    rows += [(f"{t} {NAMES[TIDX[t]]}", tw.get(t), "") for t in MARKET_REF_TW if t in TIDX]
    body += mini_rows(rows)
    c2.markdown(ui.panel("盤中止跌觀察", body or "<div class='kq-muted'>盤中資料暫時無法取得。</div>"), unsafe_allow_html=True)
    return checks


def page_insight():
    from kosto import market as mklib
    header()
    data_strip()
    df, c, sto = insight_frame(BID)
    st.markdown(ui.section("盤勢儀表板", "盤前日報、現在在哪裡、今天盤中、止穩了沒、急跌怎麼應對、特別時期怎麼選股、模型體檢"), unsafe_allow_html=True)
    st.markdown(morning_banner(), unsafe_allow_html=True)            # v3.16 今日盤前日報摘要（全文在下方第一個分頁）
    if df.empty or not sto:
        st.info("盤勢儀表板需要新版每日研究資料（下一次每日更新後提供）。")
        return
    last = df.iloc[-1]
    cur = crash_now(df)
    st.markdown(crash_banner(cur), unsafe_allow_html=True)
    tone_col = {"red": "#f0464b", "amber": "#e9a23b", "green": "#16a35f"}[sto["tone"]]
    al = mklib.allocation("穩健型", sto)
    st.markdown(
        f"<div class='kq-panel' style='border-left:4px solid {tone_col}'>"
        f"<div style='font-size:19px;font-weight:700;color:{tone_col};margin-bottom:6px'>{ui.esc(sto['view'])}</div>"
        f"<div class='kq-muted' style='font-size:13px'>{ui.esc(sto['date'])} 收盤 {sto['close']:,.0f}，距近半年高點 {ui.pct(sto['dd'], 1)}"
        f"（{ui.esc(sto['regime'])}）｜止穩訊號 <b>{sto['score']}/8</b>｜科斯托蘭尼階段 {ui.esc(STAGE_ZH.get(sto['stage'], sto['stage']).split(' ')[0])}"
        f"｜穩健型配置參考：ETF {al['etf']:.0%}／個股 {al['stock']:.0%}</div>"
        + ("<div class='kq-muted' style='margin-top:4px;color:#f0464b'>↑ 這是依前一個收盤日的判斷；今天觸發急跌條件，請以上方急跌警報與「急跌應變」為準。</div>"
           if any(f[2] for f in cur.get("flags", [])) and sto["tone"] != "red" else "") + "</div>", unsafe_allow_html=True)
    f5 = df["foreign"].tail(5).sum() if "foreign" in df else np.nan
    t5 = df["trust"].tail(5).sum() if "trust" in df else np.nan
    o_f = last.get("o_foreign", np.nan)
    mchg = (df["mlots"].iloc[-1] / df["mlots"].iloc[-11] - 1) if "mlots" in df and len(df) > 11 else np.nan
    sox = df["sox"].ffill() if "sox" in df else pd.Series(dtype=float)
    sox_dd = float(sox.iloc[-1] / sox.tail(250).max() - 1) if len(sox.dropna()) else np.nan
    wsn = world_state_now()
    vix_now = float(wsn.get("VIX", np.nan) if wsn.get("src") == "live" else last.get("vix", np.nan))
    if wsn.get("src") == "live" and len(sox.dropna()):             # 最新美股收盤比研究資料新 → 距高點用最新收盤
        sox_dd = float(wsn["SOX"] / max(float(sox.tail(250).max()), wsn["SOX"]) - 1)
    fs = df["foreign"].dropna() if "foreign" in df else pd.Series(dtype=float)
    nsell = 0
    while nsell < len(fs) and fs.iloc[-1 - nsell] < 0:
        nsell += 1

    def dot(level):
        return f"<i class='kq-dot {level}'></i>"
    dd_now = cur.get("dd250", sto["dd"])
    items = [(dot("red" if dd_now <= -0.15 else "amber" if dd_now <= -0.05 else "green") + "距一年高點", ui.pct(dd_now, 1),
              (f"{cur.get('live_status', '')}｜" if cur.get("live") else "") + f"高點 {cur.get('high250', sto['high120']):,.0f}"),
             (dot("green" if sto["score"] >= 5 else "amber" if sto["score"] >= 3 else "red") + "止穩訊號", f"{sto['score']} / 8", "≥3 開始分批、≥5 多數止穩"),
             (dot("red" if cur.get("streak", 0) >= 5 else "amber" if cur.get("streak", 0) >= 3 else "green") + "連跌", f"{cur.get('streak', 0)} 天",
              f"累計 {ui.pct(cur.get('streak_ret'), 1)}"),
             (dot("red" if nsell >= 5 else "amber" if nsell >= 3 else "green") + "外資", f"連賣 {nsell} 天" if nsell else "買超",
              f"近 5 日 {f5:+,.0f} 億" + (f"｜今日官方 {o_f:+,.0f}" if ui.ok(o_f) else "")),
             (dot("green" if ui.ok(mchg) and mchg <= -0.03 else "amber") + "融資 10 日", ui.pct(mchg, 1), "≤ −3% 代表去槓桿"),
             (dot("red" if (cur.get("sox_day") or 0) <= -0.05 else "amber" if ui.ok(sox_dd) and sox_dd <= -0.10 else "green") + "費半",
              ui.pct(cur.get("sox_day"), 1) if cur.get("sox_day") is not None else ui.pct(sox_dd, 1),
              ("前一晚（即時）" if cur.get("sox_live") else "最近一日") + f"｜距高點 {ui.pct(sox_dd, 0)}｜VIX {vix_now:.1f}")]
    st.markdown(ui.cards(items, raw=True), unsafe_allow_html=True)
    st.markdown(env_panel_html(), unsafe_allow_html=True)
    live_checks = dash_today(cur)

    in_crash = any(f[2] for f in cur.get("flags", [])) or sto["corr"]
    names = ["🚨 急跌應變", "✅ 止穩檢查", "🎯 特別時期選股", "🩺 模型體檢", "🧭 觀察重點", "🥚 科斯托蘭尼", "🌡️ 環境與估值"]
    order = names if in_crash else [names[4], names[6], names[5], names[1], names[0], names[2], names[3]]
    order = ["🌅 盤前日報"] + order                                     # v3.16 每日台股盤前日報（07:00 前）
    tabs = dict(zip(order, st.tabs(order)))
    with tabs["🌅 盤前日報"]:
        morning_tab()

    with tabs["🚨 急跌應變"]:
        summ, asof, fresh = crash_summary()
        flags = cur.get("flags", [])
        st.markdown("<div class='kq-flags'>" + "".join(
            f"<span class='kq-chip{' late' if ok_ else ''}'>{'⚠' if ok_ else '·'} {k} {ui.esc(t)}</span>" for k, t, ok_ in flags) + "</div>",
            unsafe_allow_html=True)
        st.markdown(ui.section("今天會是低點嗎？歷史上急跌之後怎麼走",
                               f"加權指數 2017 年以來（資料到 {asof}{'' if fresh else '，研究時的結果；下一次每日更新後自動重算'}）；樣本少，只當量級參考"),
                    unsafe_allow_html=True)
        st.markdown(crash_rates_html(summ), unsafe_allow_html=True)
        e2 = summ.get("E2") or {}
        st.markdown(ui.panel("怎麼解讀", ui.bullets([
            f"<b>急跌當天通常不是最低點</b>：費半大跌後的 {e2.get('n', 0)} 次中，台股那天收盤就是之後 60 日最低點的只有 {e2.get('low_hits', 0)} 次；"
            f"之後 20 日內平均還會再跌 {abs(e2.get('min20') or 0):.0%}、最深 {abs(e2.get('worst60') or 0):.0%}。",
            f"<b>但 1～3 個月後多半比較高</b>：之後 60 日平均 {e2.get('r60', 0):+.0%}、{e2.get('win60', 0):.0%} 的次數上漲 → 恐慌時殺低通常不划算。",
            f"<b>分三批（第 0、5、10 天）和一次買進的 60 日報酬幾乎一樣</b>（{ui.pct(e2.get('staged60'), 1)} vs {ui.pct(e2.get('lump60'), 1)}），"
            "但分批可以大幅降低「買在半山腰」的風險 → 用分批，依止穩訊號決定速度。",
            "<b>利空是否出盡沒有人能確定</b>：能做的是用下面的條件分批，並保留「暫停加碼」的紀律。"])), unsafe_allow_html=True)
        st.markdown(ui.section("分批計畫", "每一批的觸發條件與目前狀態（第 1 批看盤中，第 2、3 批看收盤後的止穩訊號）"), unsafe_allow_html=True)
        st.markdown(staged_plan_html(cur, sto, live_checks), unsafe_allow_html=True)
        stage = latest("long").get("market_stage", "")
        exp_l = latest("long").get("exposure", 1.0)
        st.markdown(ui.panel("模型怎麼因應", ui.bullets([
            f"<b>長線</b>：科斯托蘭尼階段 {ui.esc(STAGE_ZH.get(stage, stage).split(' ')[0])}，模型持股水位 {exp_l:.0%}（A3／B1／B2 自動降到 60%／50%／40%，其餘放 0050）。",
            "<b>短線、中線</b>：維持 100% 持股——v1.1、v1.7 實測「依階段或費半跌破百日線降水位」兩段期間都拖累報酬，所以不自動降。",
            "<b>v2.2、v2.3 測試</b>（見「模型體檢」）：急跌時動能權重減半、同產業檔數上限、技術面上限、加入籌碼或券資比，"
            "都沒有通過預先宣告的門檻 → 主模型不改。",
            "<b>你可以做的</b>：提高 0050 等大盤 ETF 的比例（「資金配置」頁可一鍵套用修正期 +20 個百分點）、名單中同產業不超過 2～3 檔、"
            "新資金照分批計畫進場，並優先看「反彈候選」與模型名單重疊的股票。"])), unsafe_allow_html=True)
        from kosto import review as rv
        bn = rv.BAN_2026
        st.markdown(ui.section("限空令（禁空令）：歷史上有用嗎？",
                               f"金管會 {bn['announce'][5:].replace('-', '/')} 宣布、{bn['effective'][5:].replace('-', '/')} 起實施（與 2025-04-07 相同）；"
                               "大盤報酬從生效日收盤起算（與新聞相同基準）"), unsafe_allow_html=True)
        st.markdown(ui.panel("這次的措施", ui.bullets([ui.esc(m) for m in bn["measures"]])
                             + f"<div class='kq-muted' style='margin-top:4px'>結束日：{ui.esc(bn['end'])}</div>"), unsafe_allow_html=True)
        st.markdown(ban_html(), unsafe_allow_html=True)
        st.markdown(ui.panel("怎麼解讀限空", ui.bullets([
            "<b>2017 年後 3 次限空，之後 1 季都上漲（+16%～+25%）</b>，比一般急跌（+8%）好；但每次同時有更大的事件"
            "（2020 聯準會無限 QE、2022 美國通膨見頂、2025 關稅暫緩＋國安基金），而且都在回落 20～30% 的恐慌高點才出手 →"
            "<b>限空本身不是原因，政府通常在最恐慌時出手，而那常接近低點</b>。",
            "<b>限空不保證隔天止跌</b>：2025 年同樣的措施生效當天仍跌 9.7%，兩天後才見低點（再跌 9.6%）；2008 年全面禁空後一個月仍跌 20%"
            "（金融海嘯，基本面持續惡化時擋不住）。",
            "→ 把限空當成「賣壓會減輕」的加分項，而不是買進訊號；仍照上方分批計畫與止穩檢查進場。"])), unsafe_allow_html=True)
        st.markdown(ui.section("被空單集中的個股，限空後比較會止穩嗎？",
                               "成交金額前 500 名，依生效前的融券狀況分五組，比較之後的超額報酬（減 500 檔平均）；t＝控制前 20 日漲跌與市值"),
                    unsafe_allow_html=True)
        st.markdown(ban_xs_html(), unsafe_allow_html=True)
        st.markdown(ui.panel("結論", ui.bullets([
            "<b>融券餘額大（券資比高）的股票</b>：2020、2022 兩次限空後 1 季多漲約 10 個百分點；<b>但 2025 年（措施與這次相同）幾乎沒有差別</b>。"
            "2025、2026 的措施主要限制「借券賣出」（法人的主要放空管道），研究資料只有融券（散戶為主），借券的效果無法驗證。",
            "<b>最近 10 天才被大量放空（被空單狙擊）的股票</b>：沒有比較容易止穩，2020 年反而落後 22 個百分點——空方通常是放空基本面轉弱的股票。",
            "<b>跌最深的股票</b>：2020、2022 之後反彈最多；2025 年前 1 個月仍是抗跌股較強，3 個月後才換跌深股反彈。",
            f"<b>平時</b>：券資比高低之後 20 日只差 {rv.SHORT_RATIO_NORMAL['q5q1_20'] * 100:+.1f} 個百分點（勝率 {rv.SHORT_RATIO_NORMAL['win']:.0%}），"
            "逐年方向正負交替 → 不放進選股模型（v2.3 測試 S1、S2 都未通過）。名單的券資比列在「特別時期選股」供參考。"])),
            unsafe_allow_html=True)
        cs_ev = (B.meta.get("crash_stats") or {}).get("events") or []
        if cs_ev:
            with st.expander("過去的急跌事件明細"):
                rows = [[e["date"], e["type"], f"{e['close']:,.0f}", ui.pct(e.get("dd250"), 1), ui.pct(e.get("r20"), 1, color=True),
                         ui.pct(e.get("r60"), 1, color=True), ui.pct(e.get("min60"), 1), "✓" if e.get("is_low60") else ""] for e in cs_ev[::-1]]
                st.markdown(ui.table(["日期", "事件", "收盤", "距一年高點", "之後 20 日", "之後 60 日", "60 日內再跌", "當天＝低點"], rows,
                                     left=(0, 1)), unsafe_allow_html=True)

    with tabs["✅ 止穩檢查"]:
        st.markdown(ui.section("止穩檢查表", "每項只用當天收盤後已知的資料；歷史欄＝2017–2023 修正期中，該項成立 vs 未成立時加權指數之後 20 日平均報酬"),
                    unsafe_allow_html=True)
        st.markdown(insight_checklist(c), unsafe_allow_html=True)
        bh = insight_buckets()
        if bh:
            st.markdown(ui.section("歷史上修正 10% 以上時：止穩分數 vs 之後走勢", "加權指數；之後 20 日內再跌＝這 20 日中最低點相對當天的幅度"),
                        unsafe_allow_html=True)
            st.markdown(bh, unsafe_allow_html=True)

    with tabs["🎯 特別時期選股"]:
        st.markdown(ui.section("模型名單在急跌時的風險", "三個期間合併；β＝費半漲跌 1% 時此股隔天平均漲跌（越高越容易跟著美股科技股大跌）"),
                    unsafe_allow_html=True)
        html, notes = picks_risk()
        if notes:
            st.markdown(ui.panel("提醒", ui.bullets(notes)), unsafe_allow_html=True)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<div class='kq-muted' style='margin:2px 2px 10px'>券資比＝融券餘額÷融資餘額；融券 10 日增減以 20 日平均成交量為分母"
                    "（+20% 代表 10 天內新增的融券相當於兩成的日均量）。v2.3 研究：限空後融券集中股只有 2/3 次較強，近期融券大增的股票沒有比較好。</div>",
                    unsafe_allow_html=True)
        sc_df = short_crowded_df(BID)
        if len(sc_df):
            with st.expander(f"融券集中股（成交金額前 500 名、融券 ≥500 張，券資比最高 {len(sc_df)} 檔）— 只供參考，不是推薦"):
                rows = [[stock_cell(B.tickers[r.j]), ui.pct(r.ratio, 1, sign=False), f"{r.sbal:,.0f}", ui.pct(r.inc10, 1) if ui.ok(r.inc10) else "–",
                         ui.pct(r.r20, 1, color=True), ui.pct(r.d60, 1)] for r in sc_df.itertuples()]
                st.markdown(ui.table(["股票", "券資比", "融券（張）", "融券 10 日增減／均量", "近 20 日", "距 60 日高"], rows, left=(0,)),
                            unsafe_allow_html=True)
                st.caption(f"融資券資料 {(B.meta.get('data_dates') or {}).get('margin') or '–'}。歷史上限空後這類股票 2020、2022 年較強、2025 年沒有差別；"
                           "資料不含借券賣出餘額。")
        st.markdown(ui.section("近 10 日消息面急漲／急跌股", "成交金額前 800 名；單日 ≥ ±7% 且量 ≥ 20 日均量 3 倍；依法人、融資、營收歸類，附歷史上同類事件之後的表現"),
                    unsafe_allow_html=True)
        shock_section()
        with st.expander("歷史研究：消息面急漲急跌之後怎麼走（2019-07～2026-10，v2.5）"):
            st.markdown(shock_study_html(), unsafe_allow_html=True)
        tick_all = sorted({p["ticker"] for h in HZ for p in latest(h).get("picks", [])})
        st.markdown(ui.panel("名單今天的重大訊息（證交所／櫃買中心）", material_html(tick_all)), unsafe_allow_html=True)
        gm = group_moves(BID)
        if len(gm):
            g2 = gm.sort_values("r20", ascending=False)
            strong = "、".join(f"{ui.esc(g)} {r.r20:+.0%}" for g, r in g2.head(6).iterrows())
            weak = "、".join(f"{ui.esc(g)} {r.r20:+.0%}" for g, r in g2.tail(6).iloc[::-1].iterrows())
            st.markdown(ui.panel("族群：近 20 日抗跌 vs 重災（成交金額前 500 名、中位數）",
                                 f"<div style='margin-bottom:4px'><b style='color:{ui.UP}'>抗跌</b>　{strong}</div>"
                                 f"<div><b style='color:{ui.DOWN}'>重災</b>　{weak}</div>"
                                 "<div class='kq-muted' style='margin-top:6px'>v2.0 研究：修正期間「近期跌深、營收仍優於同業、不是投信擁擠股」反彈最好；"
                                 "修正前的強勢股之後 20 日反彈較弱。</div>"), unsafe_allow_html=True)
        st.markdown(insight_rebound(), unsafe_allow_html=True)

    with tabs["🩺 模型體檢"]:
        mh, mix = model_health(BID)
        if len(mh):
            rows, mrows = [], []
            for r in mh.to_dict("records"):
                rows.append([f"<b>{HZ[r['h']]}</b>", f"{ui.pct(r['dd'], 1)}<span class='sub'>（高點 {r['peak'][2:]}）</span>", ui.pct(r["dd_b"], 1),
                             f"{ui.pct(r['m1'], 1, color=True)} / {ui.pct(r['m1_b'], 1, color=True)}",
                             f"{ui.pct(r['m6'], 1, color=True)} / {ui.pct(r['m6_b'], 1, color=True)}",
                             f"{ui.pct(r['y1'], 1, color=True)} / {ui.pct(r['y1_b'], 1, color=True)}",
                             f"{ui.pct(r['ytd'], 1, color=True)} / {ui.pct(r['ytd_b'], 1, color=True)} / {ui.pct(r['ytd_pool'], 1, color=True)}"])
                mrows.append(ui.mcard(f"<span class='nm'>{HZ[r['h']]}</span>", f"自高點 {ui.pct(r['dd'], 1)}",
                                      f"近1月 {ui.pct(r['m1'], 1, color=True)}（0050 {ui.pct(r['m1_b'], 1)}）｜近6月 {ui.pct(r['m6'], 1, color=True)}（{ui.pct(r['m6_b'], 1)}）",
                                      f"近12月 {ui.pct(r['y1'], 1, color=True)}（0050 {ui.pct(r['y1_b'], 1)}）｜今年 {ui.pct(r['ytd'], 1, color=True)}（0050 {ui.pct(r['ytd_b'], 1)}、母池 {ui.pct(r['ytd_pool'], 1)}）"))
            st.markdown(ui.section("模型 vs 0050", f"回測淨值到 {mh['asof'].iloc[0]}；欄位為「模型 / 0050」，今年多一欄同母池等權（中小型股平均）"),
                        unsafe_allow_html=True)
            st.markdown(ui.resp(ui.table(["期間", "模型自高點", "0050 自高點", "近 1 月", "近 6 月", "近 12 月", "今年（模型/0050/母池）"], rows, left=(0,)),
                                "".join(mrows)), unsafe_allow_html=True)
            r = mh.set_index("h").loc["mid"] if "mid" in set(mh["h"]) else None
            lines = []
            if r is not None:
                gap = r["y1"] - r["y1_b"]
                lines.append(f"<b>中線</b>：自 {r['peak']} 高點回落 {ui.pct(r['dd'], 1)}（0050 {ui.pct(r['dd_b'], 1)}）；"
                             f"近 12 月 {ui.pct(r['y1'], 1)} vs 0050 {ui.pct(r['y1_b'], 1)}（{'落後' if gap < 0 else '領先'} {abs(gap) * 100:.0f} 個百分點）。")
                if ui.ok(r["ytd_pool"]) and r["ytd_b"] - r["ytd_pool"] > 0.05:
                    lines.append(f"<b>今年的落差主要來自「選股範圍」</b>：0050 由台積電等權值股帶動（今年 {ui.pct(r['ytd_b'], 0)}），"
                                 f"中小型股平均（母池等權）只有 {ui.pct(r['ytd_pool'], 0)}；模型今年 {ui.pct(r['ytd'], 0)}"
                                 + ("，和母池平均差不多 → 選股本身沒有失效，但追不上權值股集中的行情。" if abs(r["ytd"] - r["ytd_pool"]) < 0.08 else
                                    "，勝過母池平均 → 選股仍有效，只是權值股漲更多。" if r["ytd"] > r["ytd_pool"] else "，也落後母池平均。"))
                if r["dd"] < r["dd_b"] - 0.05:
                    lines.append("<b>回落比大盤深</b>：模型偏好「相對強勢、接近新高、營收動能」的股票，這類股票在急跌與反轉時跌得比大盤多（動能反轉）。"
                                 "v2.2 測試「急跌時動能權重減半」在 2021–2025 只多 0.6 個百分點，未達門檻，所以不改模型；想降低波動請提高 0050 比例（下表）。")
            lines.append("<b>三個期間的名單高度重疊</b>（短線與中線用同一套模型），同時買三份名單等於加碼同一批股票。")
            st.markdown(ui.panel("體檢結論", ui.bullets(lines)), unsafe_allow_html=True)
        if mix:
            rows = [[f"<b>{w:.0%}</b>", ui.pct(m["cagr"], 1), ui.pct(m["mdd"], 1), ui.pct(m["y1"], 1, color=True), ui.pct(m["dd"], 1)]
                    for w, m in mix.items()]
            st.markdown(ui.section("搭配 0050 當核心（中線，每月再平衡）", "0050 比例越高：長期報酬較低，但回撤較小、在權值股行情中不容易落後"),
                        unsafe_allow_html=True)
            st.markdown(ui.table(["0050 比例", "年化（2021 起）", "最大回撤", "近 12 月", "目前自高點"], rows, left=(0,)), unsafe_allow_html=True)
        from kosto import review as rv
        st.markdown(ui.section("驗效與偏移檢查（v2.4）", "研究報告的預先宣告閘門、盲測期、未用期間；模型有沒有往某個方向偏"),
                    unsafe_allow_html=True)
        st.markdown(audit_html(), unsafe_allow_html=True)
        st.markdown(ui.section("因子權重平衡", "目前權重（每日研究）中各家族的占比；括號為 2021 年以來每 6 個月重估一次的範圍"),
                    unsafe_allow_html=True)
        st.markdown(balance_html(), unsafe_allow_html=True)
        st.markdown(ui.panel("平衡檢查結論", ui.bullets([
            "<b>沒有隨時間往某一邊偏移</b>：2021 年以來 12 次重估，短中線都是技術面約 48%、基本面約 52%；長線技術面 55～61%。"
            "每次重估都用到當時為止的全部歷史（2019 年起）並向先驗收縮，所以不會追著最近一兩季的行情跑。",
            "<b>最大的集中是「動能」</b>（相對強度、120 日動能、一年新高，中線約 41%、長線約 60%）——這是 7 月與 11 月初回落比大盤深的原因。"
            "v2.2「急跌時動能減半」與 v2.3「技術面上限 45%」都只多不到 1 個百分點、回撤改善不到 2 個百分點 → 歷史資料不支持調低。",
            "<b>籌碼面權重為 0</b>：外資、投信、融資券在中線的 IC 多半是反向且不穩定；v2.3 加入籌碼 15% 中線變好、但長線少 8 個百分點 → 未通過。",
            "<b>想要更平衡，比改因子更有效的是配置</b>：中線搭配 0050 30%，最大回撤由 −37% 降到 −33%（上表）。"])), unsafe_allow_html=True)
        st.markdown(ui.section("急跌時哪些因子還有效？", "中線因子的平均 IC：一般時期 vs 急跌狀態（0050 近 20 日 ≤ −8% 或距 120 日高 ≥ 12%，2019 年後 48 個換股日）"),
                    unsafe_allow_html=True)
        rows = [[ui.esc(nm), rv.FAMILY_ZH[f], f"{a:+.3f}", f"<b style='color:{ui.UP if b_ > 0 else ui.DOWN}'>{b_:+.3f}</b>"] for nm, f, a, b_ in rv.PANIC_IC]
        st.markdown(ui.table(["因子", "家族", "一般時期", "急跌時"], rows, left=(0, 1)), unsafe_allow_html=True)
        st.caption("急跌時「接近新高、低波動、高殖利率」失效，「營收動能」更有效。這正是 B1（急跌時動能減半）想利用的現象，但開發期只多 0.6 個百分點、未達門檻。")
        st.markdown(ui.section("預先宣告測試（docs/v2.2_crash_protocol.md、v2.3、v2.5、v3.0、v3.1、v3.5_protocol.md）",
                               "五種換股日、2021–2025 開發期；單位：百分點（資訊比率為差值）；與 v1.9 相同的採用門檻"), unsafe_allow_html=True)
        st.markdown(ui.table(["候選", "中線超額", "中線 IR", "中線回撤改善", "短線超額", "長線超額", "判定"],
                             [[f"<b>{a}</b>", b, c_, d, e, f, g] for a, b, c_, d, e, f, g in V22_TESTS + rv.V23_TESTS]
                             + [[f"<b>{a}</b>", b, c_, d, e, f, f"{g_}（盲測期中線 {bl}，只報告）"] for a, b, c_, d, e, f, bl, g_ in rv.V25_TESTS + rv.V30_TESTS + rv.V31_TESTS + rv.V35_TESTS + rv.V39_TESTS],
                             left=(0, 6)),
                    unsafe_allow_html=True)
        st.markdown(v37_html(), unsafe_allow_html=True)
        st.markdown(turning_html(), unsafe_allow_html=True)
        st.markdown(v312_html(), unsafe_allow_html=True)
        st.markdown(v313_html(), unsafe_allow_html=True)
        st.markdown(ui.section("模型的時效（v2.5）", "中線分數與之後 60 日報酬的相關（IC，越高越準）；消息面因子逐年方向"), unsafe_allow_html=True)
        rows = [[f"<b>{y}</b>", f"{v:+.3f}"] for y, v in rv.SCORE_IC.items()] + [[f"<b>{q}</b>", f"{v:+.3f}"] for q, v in rv.SCORE_IC_Q.items()]
        rows2 = [[f"<b>{y}</b>", f"{rv.SHOCK_IC['t_limitup20'].get(y, np.nan):+.3f}", f"{rv.SHOCK_IC['t_max20'].get(y, np.nan):+.3f}"]
                 for y in rv.SHOCK_IC["t_limitup20"]]
        c1, c2 = st.columns(2)
        c1.markdown(ui.table(["期間", "模型分數 IC"], rows, left=(0,)), unsafe_allow_html=True)
        c2.markdown(ui.table(["年", "近 20 日漲停次數", "近 20 日沒有暴漲"], rows2, left=(0,)), unsafe_allow_html=True)
        st.caption("模型分數的預測力逐年上升（2026 年 0.079），但 2026 第二、三季明顯轉弱（0.012、0.036）——權值股獨強、題材輪動快的行情。"
                   "「漲停次數」以前是反向（追漲停股之後較差），2025 年起轉正；「沒有暴漲」以前是正向，2025 年起轉為零或負："
                   "最近兩年題材股比較會延續。方向還不穩定、只有兩年，依規則不加入模型，持續觀察。")
        with st.expander("歷次想法總檢視（v1.1～v3.14）：每個想法對模型的影響"):
            rows = [[ui.esc(v), ui.esc(i), ui.esc(src), ui.esc(tp), ui.esc(res), ("<b>" if st_.startswith("✅") else "") + ui.esc(st_)
                     + ("</b>" if st_.startswith("✅") else ""), ui.esc(imp)] for v, i, src, tp, res, st_, imp in rv.IDEAS]
            st.markdown(ui.table(["版本", "想法", "來源", "類型", "驗證結果", "狀態", "對模型的影響"], rows, left=(1, 2, 3, 4, 5, 6)),
                        unsafe_allow_html=True)
            n_user = sum(1 for x in rv.IDEAS if x[2] == "使用者提問")
            n_user_ok = sum(1 for x in rv.IDEAS if x[2] == "使用者提問" and x[5].startswith("✅"))
            st.caption(f"使用者提問引發的 {n_user} 組想法中，{n_user_ok} 組改變了主模型；其餘都轉為儀表板、燈號、提醒等資訊顯示。"
                       "主模型最近一次調整是 v3.14（回應同儕評審：跌停賣不掉納入回測；「營收優於同業」的同業改用官方產業別）；"
                       "之前是 v3.0（短中線改名次加權）、v1.9（營收優於同業 15%）。"
                       "v3.5 從美股借來的「風險調整動能」開發期過門檻，但整段樣本外沒有進步，未採用。")

    with tabs["🥚 科斯托蘭尼"]:
        kostolany_tab()

    with tabs["🌡️ 環境與估值"]:
        st.markdown(env_tab_html(df), unsafe_allow_html=True)

    with tabs["🧭 觀察重點"]:
        st.markdown(insight_focus(df, sto, f5, t5, mchg, sox_dd), unsafe_allow_html=True)
        st.markdown(ui.section("重大事件與即時消息", "事件的歷史參考、可靠的即時消息來源、最新頭條（10 分鐘更新）"), unsafe_allow_html=True)
        st.markdown(nv_event_html(), unsafe_allow_html=True)
        n1, n2 = st.columns(2)
        n1.markdown(ui.panel("台股頭條（Google 新聞，近 2 天）", headlines_html(news_headlines("台股 盤勢", 6, 2))), unsafe_allow_html=True)
        n2.markdown(ui.panel("輝達／AI 供應鏈頭條（近 2 天）", headlines_html(news_headlines("輝達 供應鏈 台股", 6, 2))), unsafe_allow_html=True)
        with st.expander("可靠的即時消息來源（官方優先）"):
            st.markdown(sources_html(), unsafe_allow_html=True)
            st.caption("媒體標題只是線索；公司是否「打入供應鏈」「取得專利」等，以公開資訊觀測站的重大訊息為準。")
        lead_section()
        with st.expander("模型裡原本就有的「跌深／止跌」設計"):
            st.markdown(ui.bullets([
                "短期反轉因子「近 5 日跌多」（t_rev5）：在短線候選池中，由每日研究決定是否使用。",
                "科斯托蘭尼雞蛋階段：B3（下跌末段）要有「止跌確認」（W 底支撐或站回 5 日線且 5 日上漲）才會買。",
                "長線持股水位：B1／B2 階段自動降到 50%／40%；短線、中線不降（實測降水位拖累績效）。",
                "v2.0 研究：修正期間「近期跌深、營收仍優於同業、不是投信擁擠股」反彈最好，所以修正期另外提供反彈候選，而不是改主模型。"]),
                unsafe_allow_html=True)


# ================================================================== v2.0 精選 ETF
@st.cache_data(show_spinner=False)
def etf_table(bid) -> pd.DataFrame:
    t = pd.DataFrame(B.meta.get("etf_table") or [])
    return t


def page_etf():
    from kosto import etf as etflib
    from kosto import market as mklib
    header()
    data_strip(("price", "etf"))
    st.markdown(ui.section("精選 ETF", "含息報酬（配息再投入）；配息與不配息分開排名，依長期年化報酬（近 3 年，不足 3 年以近 1 年）由高到低"),
                unsafe_allow_html=True)
    _, _, sto = insight_frame(BID)
    prof = st.segmented_control("你的風險屬性（與「資金配置」頁相同）", list(mklib.PROFILES), default="穩健型", key="etf_prof") or "穩健型"
    al = mklib.allocation(prof, sto or {})
    mix = None
    hb = (RES or {}).get("horizons", {}).get("mid") if RES else None
    if hb is not None and isinstance(hb.get("mix"), pd.DataFrame):
        mix = hb["mix"]
    st.markdown(ui.cards([("ETF（核心，長期持有）", f"{al['etf']:.0%}", "以 0050 等大盤型為主，配息與否不影響總報酬"),
                          ("個股（衛星，照模型換股）", f"{al['stock']:.0%}", "中線名單為主，短長線依喜好"),
                          ("市場狀態調整", f"{al['tilt']:+.0%}" if al["tilt"] else "不調整", ui.esc(al["why"][:40]) + "…" if len(al["why"]) > 40 else ui.esc(al["why"]))]),
                unsafe_allow_html=True)
    if mix is not None and len(mix):
        rows = [[ui.esc(r["組合"]), ui.pct(r.get("oos_cagr"), 1, color=True), ui.pct(r.get("oos_mdd"), 1), f"{r.get('oos_sharpe', np.nan):.2f}",
                 ui.pct(r.get("holdout_cagr"), 1, color=True)] for _, r in mix.iterrows()]
        st.markdown(ui.table(["0050 核心＋中線模型（回測 2021 起）", "年化", "最大回撤", "Sharpe", "近一年"], rows, left=(0,)), unsafe_allow_html=True)
    st.markdown("<div class='kq-muted'>建議：ETF 當核心長期持有（不需擇時）；個股部位照模型換股。修正期且止穩訊號 ≤2 時把 ETF 提高 20 個百分點，"
                "回測 2021–2026 年化約 +1.5～2 個百分點、回撤略小（樣本只有約 5 次修正，參考就好）。配息與不配息的總報酬比較看「年化」即可；"
                "配息另有所得稅與二代健保補充保費（單次 ≥2 萬元扣 2.11%），在意的話優先不配息或低配息。</div>", unsafe_allow_html=True)

    t = etf_table(BID)
    if t.empty:
        st.info("ETF 資料將在下一次每日更新後開始累積：最新日期優先，約 1 個晚上有近 1 年報酬、4 個晚上補滿 5 年。")
        return
    st.markdown(f"<div class='kq-muted'>資料至 {B.meta.get('etf_asof', '')}｜上市＋上櫃 ETF {len(t)} 檔｜"
                "排除：上市未滿 1 年、近 60 日平均成交金額 < 0.1 億；槓桿反向、期貨商品型另列（不列入建議）</div>", unsafe_allow_html=True)
    tabs = st.tabs(["🏛️ 大盤型（核心）", "💰 配息", "📈 不配息", "🏦 債券", "⚠️ 槓桿反向", "🔍 全部／搜尋", "⚖️ 與選股模型比較",
                    "📉 修正期怎麼選"])
    with tabs[7]:
        etf_correction_section()

    def show(x: pd.DataFrame, key: str):
        if x.empty:
            st.markdown("<div class='kq-muted'>沒有符合條件的 ETF。</div>", unsafe_allow_html=True)
            return
        rows, mrows = [], []
        for k, (_, r) in enumerate(x.iterrows(), 1):
            nm = f"<b>{r['ticker']} {ui.esc(r['name'])}</b>"
            rows.append([f"{k}", nm, ui.esc(r["type"]), ui.pct(r["長期年化"], 1, color=True) + f"<span class='kq-muted'> {r['長期依據']}</span>",
                         ui.pct(r["1月"], 1, color=True), ui.pct(r["3月"], 1, color=True), ui.pct(r["6月"], 1, color=True),
                         ui.pct(r["1年"], 1, color=True), ui.pct(r["3年"], 1, color=True), ui.pct(r["5年"], 1, color=True),
                         (ui.pct(r["殖利率"], 1, sign=False) + f" {ui.esc(r['頻率'])}") if r["配息"] else ui.esc(r["頻率"]),
                         ui.pct(r["波動"], 0, sign=False), ui.pct(r["最大回撤"], 0), f"{r['成交額']:,.1f}" if ui.ok(r["成交額"]) else "–"])
            mrows.append(ui.mcard(f"<span class='nm'>{k}. {r['ticker']} {ui.esc(r['name'])}</span>",
                                  ui.pct(r["長期年化"], 1, color=True),
                                  f"1月 {ui.pct(r['1月'], 1, color=True)}｜3月 {ui.pct(r['3月'], 1, color=True)}｜1年 {ui.pct(r['1年'], 1, color=True)}｜3年年化 {ui.pct(r['3年'], 1, color=True)}",
                                  f"{ui.esc(r['type'])}｜{ui.esc(r['頻率'])}" + (f" 殖利率 {ui.pct(r['殖利率'], 1, sign=False)}" if r["配息"] else "")
                                  + f"｜回撤 {ui.pct(r['最大回撤'], 0)}"))
        st.markdown(ui.resp(ui.table(["#", "ETF", "類型", "長期年化", "1月", "3月", "6月", "1年", "3年年化", "5年年化", "殖利率／頻率",
                                      "波動", "最大回撤", "成交額(億)"], rows, left=(1, 2)), "".join(mrows)), unsafe_allow_html=True)
        sel = x["ticker"].head(5).tolist()
        tr = etf_tr(BID)
        if not tr.empty and sel:
            yrs = st.segmented_control("走勢", ["1年", "3年", "5年"], default="1年", key=f"etf_rng_{key}") or "1年"
            n = {"1年": 252, "3年": 756, "5年": 1260}[yrs]
            cols = [c for c in sel + (["0050"] if "0050" not in sel else []) if c in tr]
            seg = tr[cols].iloc[-n:]
            navs = {f"{c} {etf_name(c)}": seg[c] / seg[c].dropna().iloc[0] for c in cols if seg[c].notna().sum() > 20}
            from kosto.charts import nav_figure
            st.plotly_chart(nav_figure(navs, height=300, log=False, dark=DARK), width="stretch", theme=None, key=f"etf_nav_{key}",
                            config={"displaylogo": False, "displayModeBar": False})

    with tabs[0]:
        st.markdown("<div class='kq-muted'>和 0050 走勢幾乎同步（近一年日報酬相關係數 ≥0.95）的國內股票型 ETF：最適合當「核心」長期持有，"
                    "差別主要在費用、配息方式與追蹤誤差。</div>", unsafe_allow_html=True)
        show(etflib.picks(t, "大盤型", 12), "core")
    with tabs[1]:
        st.markdown("<div class='kq-muted'>提醒：近年報酬最高的多是半導體／科技主題 ETF，和 0050（台積電權重最大）高度重疊、波動也大；"
                    "可以當衛星，不建議取代大盤型核心。高股息 ETF 在下跌時通常跌得較少。</div>", unsafe_allow_html=True)
        show(etflib.picks(t, "配息", 15), "div")
    with tabs[2]:
        show(etflib.picks(t, "不配息", 15), "nodiv")
    with tabs[3]:
        show(etflib.picks(t, "債券", 10), "bond")
    with tabs[4]:
        st.markdown("<div class='kq-warn'>槓桿／反向 ETF 每天重設倍數，長期持有會有「波動耗損」，報酬不等於指數的 2 倍；"
                    "大跌時回撤可達 50% 以上。只適合短線、看對方向時使用，不列入配置建議。</div>", unsafe_allow_html=True)
        show(etflib.picks(t, "槓桿反向", 10, min_amt=0.0), "lev")
    with tabs[5]:
        q = st.text_input("搜尋代號或名稱（例：0056、高股息、美國）", key="etf_q")
        x = t.copy()
        if q:
            x = x[x["ticker"].str.contains(q, case=False) | x["name"].str.contains(q, case=False)]
        show(x.head(40), "all")
    with tabs[6]:
        st.markdown("<div class='kq-muted'>ETF 是一籃子股票、不用換股；選股模型是每週換股的 8 檔組合。把它們放在一起比，只能比「同一段時間的報酬與風險」：</div>",
                    unsafe_allow_html=True)
        rows = []
        for h in HORIZON_KEYS:
            ph = phase_stats(h)
            if ph:
                rows.append([f"<b>選股模型・{HZ[h]}</b>（回測）", ui.pct(ph["近一年_平均"], 1, color=True), ui.pct(ph["樣本外年化_平均"], 1, color=True) + "（2021 起）",
                             ui.pct(ph["最大回撤_平均"], 0), "每週換股、交易成本已扣"])
        for _, r in pd.concat([etflib.picks(t, "配息", 3), etflib.picks(t, "不配息", 3)]).iterrows():
            rows.append([f"{r['ticker']} {ui.esc(r['name'])}", ui.pct(r["1年"], 1, color=True), ui.pct(r["3年"], 1, color=True) + "（近 3 年）",
                         ui.pct(r["最大回撤"], 0), "買進持有、含息"])
        st.markdown(ui.table(["標的", "近一年", "年化", "最大回撤", "說明"], rows, left=(0, 4)), unsafe_allow_html=True)
        st.markdown("<div class='kq-muted'>模型的回測年化較高，但 2019–2020（研究沒用到的期間）輸 0050；ETF 不會有選錯股的風險。"
                    "這就是建議「ETF 核心＋個股衛星」而不是二選一的原因。</div>", unsafe_allow_html=True)


def etf_correction_section():
    """v3.10 修正期 1～3 個月：ETF 怎麼選、要不要留現金（docs/v3.10_protocol.md）。"""
    from kosto import etf as etflib
    from kosto import review as rv
    st.markdown(ui.section("修正期 1～3 個月：ETF 怎麼選、要不要留現金", "歷史驗證（v3.10）；只報告，不是預測"), unsafe_allow_html=True)
    tr = etf_tr(BID)
    base = tr["0050"] if "0050" in tr else pd.Series(dtype=float)
    if len(base.dropna()) > 130:
        b = base.ffill()
        dd = float(b.iloc[-1] / b.iloc[-120:].max() - 1)
        st.markdown(f"<div class='kq-verdict'>目前 0050（含息）距 120 日高點 <b>{dd * 100:+.1f}%</b>："
                    + ("<b>在修正期</b>（≥ 10%）。下表是過去每次修正第一天買進之後的表現。" if dd <= -0.10 else
                       "不在修正期（< 10%）。下表供下一次修正時參考。") + "</div>", unsafe_allow_html=True)
        perf, dates = etflib.correction_perf(tr, base)
        if len(perf):
            rows = [[f"<b>{r['ticker']}</b> {ui.esc(etf_name(r['ticker']))}", str(int(r["n"])), ui.pct(r["r20"], 1, color=True),
                     ui.pct(r["r60"], 1, color=True), "–" if not ui.ok(r["up60"]) else f"{r['up60'] * 100:.0f}%", ui.pct(r["mdd"], 1),
                     ui.pct(r["worst"], 1), "–" if r["ticker"] == "0050" or not ui.ok(r.get("beat")) else f"{r['beat'] * 100:.0f}%"]
                    for _, r in perf.iterrows()]
            st.markdown("<div class='kq-muted' style='margin:6px 0 2px'><b>實際 ETF（含息，資料包近 5 年）</b>：修正期第一天＝"
                        + "、".join(dates) + "</div>", unsafe_allow_html=True)
            st.markdown(ui.table(["ETF", "次數", "之後 20 日", "之後 60 日", "60 日上漲比例", "期間最大回落（平均）", "最差回落", "60 日贏 0050"],
                                 rows, left=(0,)), unsafe_allow_html=True)
    f = lambda v: f"{v:+.1f}%"          # noqa: E731
    st.markdown("<div class='kq-muted' style='margin:10px 0 2px'><b>較長期間（2020–2026 共 7 次修正；台股 ETF 上市較晚的用成分股代理）</b></div>",
                unsafe_allow_html=True)
    st.markdown(ui.table(["對象", "之後 20 日", "之後 60 日", "60 日上漲", "期間最大回落（平均）", "最差回落", "60 日贏 0050"],
                         [[ui.esc(a), f(b), f"<b>{f(c)}</b>", d, f(e), f(g), h] for a, b, c, d, e, g, h in rv.V310_ETF], left=(0,)),
                unsafe_allow_html=True)
    rows = []
    for dt, xs in rv.V310_REAL:
        rows.append([f"<b>{dt}</b>"] + [f"{r:+.1f}%／{m:.1f}%" for _, r, m in xs])
    st.markdown("<div class='kq-muted' style='margin:10px 0 2px'><b>實際行情檢查</b>（之後 60 日報酬／期間最大回落；價格、不含配息）</div>",
                unsafe_allow_html=True)
    st.markdown(ui.table(["修正第一天"] + [c for c, _, _ in rv.V310_REAL[0][1]], rows, left=(0,)), unsafe_allow_html=True)
    st.markdown(ui.panel("怎麼用", ui.bullets([
        ui.esc(rv.V310_NOTE),
        "分批節奏看「盤勢儀表板 → 止穩檢查」：0～2 分小量、3～4 分正常、5 分以上可以加快；急跌應變頁有一次買 vs 分批的歷史比較（差不多）。",
        "反向 ETF（00632R 等）每天重設倍數，震盪時會耗損，不適合持有 1～3 個月；買賣價位（支撐、目標價）無法用歷史驗證，只當參考。"])),
        unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def etf_tr(bid) -> pd.DataFrame:
    if "ETF_TR" not in B.a or not B.meta.get("etf_tickers"):
        return pd.DataFrame()
    return pd.DataFrame(B.a["ETF_TR"].astype(float), index=pd.DatetimeIndex(B.meta["etf_days"]), columns=B.meta["etf_tickers"])


def etf_name(code: str) -> str:
    t = etf_table(BID)
    m = t[t["ticker"] == code] if len(t) else t
    return str(m["name"].iloc[0]) if len(m) else ""


# ================================================================== page: 國際盤
def link_oc_text(yl) -> str:
    """v2.1：前一夜費半 → 台股開盤跳空 vs 開盤到收盤（盤中延續）的逐年相關。"""
    c1, c2 = "相關_費半→開盤跳空", "相關_費半→開盤到收盤"
    if yl is None or c2 not in yl or yl[c2].notna().sum() < 3:
        return ("連動主要反映在<b>開盤跳空</b>：美股的影響多半在台股開盤就反映 → 用美股漲跌去「預測」台股接下來幾週哪些股票會贏，"
                "實測沒有幫助（見下方研究結論）。")
    z = yl.dropna(subset=[c2])
    old, new = z.iloc[:-2][c2].mean(), z.iloc[-1][c2]
    return (f"以前美股的影響幾乎在台股<b>開盤</b>一次反映（{int(z.iloc[0]['年'])}–{int(z.iloc[-3]['年'])} 年 費半→開盤到收盤 相關約 {old:.2f}）；"
            f"{int(z.iloc[-1]['年'])} 年升到 <b>{new:.2f}</b>，代表盤中仍持續跟著美股（期貨）走，像 11/4 跌 1,400 點後隨美股期貨翻紅急拉的 V 型走勢。"
            "週頻選股仍不受影響；盤中操作請同時看那斯達克期貨。")


@st.cache_data(ttl=120, show_spinner=False)
def peers_live() -> pd.DataFrame:
    from kosto.sources import world as W
    try:
        return W.quote(W.PEER_KEYS, workers=8)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def theme_section():
    """v2.1 台股連動精選：依主題的國際指標股（即時）＋台股對應族群與近一年連動係數。"""
    from kosto import peers as peerslib
    Wd = world_frame(BID)
    q = peers_live()
    live = {r["key"]: r for r in q.to_dict("records")} if len(q) else {}
    st.markdown(ui.section("台股連動精選", "國際指標股即時報價（美股含盤前盤後最新價）；台股族群為研究日收盤；連動＝近一年日報酬相關"),
                unsafe_allow_html=True)
    R = pd.DataFrame(B.a["R"].astype(float), index=B.days, columns=B.tickers) if "R" in B.a else pd.DataFrame()
    cap = B.a["CAP"][-1] if "CAP" in B.a else np.zeros(len(B.tickers))
    tabs = st.tabs([n for n, _, _ in peerslib.THEMES])
    for tab, (name, keys, pat) in zip(tabs, peerslib.THEMES):
        with tab:
            tiles = []
            wi = world_intraday(keys)
            for k in keys:
                d = live.get(k)
                di = wi.get(k)
                if di is not None:
                    tiles.append(ui.tile(peerslib.name_of(k), ui.num(di["last"], 2), di["last"] / di["prev"] - 1, IDY.status(di),
                                         world_tile_chart(di, None)))
                elif d:
                    tiles.append(ui.tile(peerslib.name_of(k), ui.num(d["price"], 2), d["chg"], f"即時 {d['time']:%m/%d %H:%M}",
                                         world_tile_chart(None, d.get("spark"))))
                elif k in Wd and Wd[k].notna().sum() > 2:
                    s_ = Wd[k].dropna()
                    s_ = s_[s_.diff().fillna(1) != 0]
                    if len(s_) >= 2:
                        tiles.append(ui.tile(peerslib.name_of(k), ui.num(s_.iloc[-1], 2), float(s_.iloc[-1] / s_.iloc[-2] - 1), "最近收盤",
                                             world_tile_chart(None, s_.tail(22).tolist())))
            if tiles:
                st.markdown("<div class='kq-tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='kq-muted'>國際同業資料將在下一次每日更新後提供。</div>", unsafe_allow_html=True)
            mem = [t for t in peerslib.tw_members(pat, CH) if t in TIDX]
            mem = sorted(mem, key=lambda t: -np.nan_to_num(cap[TIDX[t]]))[:8]
            if not mem or R.empty:
                continue
            basket = pd.concat([Wd[k].pct_change(fill_method=None) for k in keys if k in Wd], axis=1).mean(axis=1) if any(k in Wd for k in keys) else None
            rows = []
            for t in mem:
                r = R[t]
                cor = float(pd.concat([r, basket], axis=1).dropna().tail(250).corr().iloc[0, 1]) if basket is not None else np.nan
                c = B.a["Q_C"][:, TIDX[t]].astype(float) if "Q_C" in B.a else np.array([])
                rows.append([f"{t} {ui.esc(NAMES[TIDX[t]])}", ui.esc(_seg_of(t)), ui.pct(_ret_n(c, 1), 1, color=True),
                             ui.pct(_ret_n(c, 5), 1, color=True), ui.pct(_ret_n(c, 20), 1, color=True), f"{cor:.2f}" if ui.ok(cor) else "–"])
            st.markdown(ui.table(["台股", "細產業", "1日", "5日", "20日", "與國際同業連動"], rows, left=(0, 1)), unsafe_allow_html=True)


def intraday_section():
    il = B.meta.get("intraday_link") or []
    if not il:
        return
    t = pd.DataFrame(il)
    rows = [[r["季"], f"{int(r['小時數'])}", f"{r['盤中每小時相關']:.2f}" if ui.ok(r.get("盤中每小時相關")) else "–",
             f"{r['盤中β']:.2f}" if ui.ok(r.get("盤中β")) else "–",
             f"{r['開盤到收盤相關']:.2f}" if ui.ok(r.get("開盤到收盤相關")) else "–"] for _, r in t.iterrows()]
    st.markdown(ui.section("盤中連動：台股 vs 那斯達克期貨（逐季）", "台股交易時段內兩者同一小時報酬的相關係數；β＝期貨漲跌 1% 時加權指數同時段平均漲跌"),
                unsafe_allow_html=True)
    st.markdown(ui.table(["季", "小時數", "每小時相關", "盤中β", "開盤到收盤相關"], rows, left=(0,)), unsafe_allow_html=True)
    h = t["盤中每小時相關"].dropna()
    if len(h) >= 8:
        a, b = h.iloc[-4:].mean(), h.iloc[-8:-4].mean()
        verdict = "連動變緊" if a - b >= 0.1 else ("差不多" if abs(a - b) < 0.1 else "連動變鬆")
        st.markdown(f"<div class='kq-muted'>最近 4 季平均 {a:.2f}，前 4 季 {b:.2f} → <b>{verdict}</b>（預先宣告：差距 ≥0.1 才算變緊）。</div>",
                    unsafe_allow_html=True)


def linkage_ic_section():
    li = (RES or {}).get("linkage_ic") or {}
    ic = li.get("ic") or {}
    if not ic:
        return
    years = sorted({y for v in ic.values() for y in v})[-8:]
    rows = []
    for f, v in ic.items():
        rows.append([f"{ui.esc(FZH.get(f, f))}<span class='kq-muted'> {f}</span>"] +
                    [f"<span style='color:{ui.color_of(v.get(y))}'>{v.get(y):+.3f}</span>" if v.get(y) is not None else "–" for y in years])
    st.markdown(ui.section("連動因子能不能用來選股？（逐年 IC，每日自動更新）",
                           f"中線母池 {li.get('pool', '')}：因子分數與之後 20 日報酬的等級相關；正＝分數高的股票之後表現較好"), unsafe_allow_html=True)
    st.markdown(ui.table(["因子"] + [str(y) for y in years], rows, left=(0,)), unsafe_allow_html=True)
    st.markdown("<div class='kq-muted'>v2.1 預先宣告的檢驗（docs/v2.1_linkage_protocol.md）：同族群動能、落後補漲、族群上漲家數，"
                "放進模型或固定 15% 加入都讓中線開發期超額報酬下降 0.8～5 個百分點，<b>未通過</b>；對費半連動高的股票在 2025–2026 年明顯較強，"
                "但 2022 年相反——屬於「AI 多頭期」的現象，現在加入等於追逐最近的行情。系統每天更新這張表，若連續轉正會在下一版重新檢驗。</div>",
                unsafe_allow_html=True)


def page_world():
    header(world=False)
    snap = world_snapshot()
    ls = link_stats(BID)
    st.markdown(ui.section("國際盤", "美股收盤、美股期貨、日韓早盤、匯率｜每 60 秒更新"), unsafe_allow_html=True)
    if not snap:
        st.info("國際市場資料尚未建立：下一次每日更新會自動抓取，或目前網路無法連到 Yahoo Finance。")
        return
    tiles = []
    wi = world_intraday(WORLD_ORDER)
    for k in WORLD_ORDER:
        if k not in snap:
            continue
        d = dict(snap[k])
        di = wi.get(k)
        if di is not None:                                     # 有分時：價格、漲跌與圖的平盤一致
            d.update(price=di["last"], chg=di["last"] / di["prev"] - 1, time=IDY.status(di), live=False)
        val = ui.num(d["price"], 2 if d["price"] < 1000 else 0)
        sub = ui.esc(WORLD_SUB.get(k, "")) + f"<br>{'即時 ' if d.get('live') else ''}{ui.esc(d['time'])}"
        if k == "TSM" and "TWD" in snap and HAS_Q and "2330" in TIDX:
            px2330 = last_close(TIDX["2330"])[0]
            prem = d["price"] * snap["TWD"]["price"] / 5 / px2330 - 1 if ui.ok(px2330) else np.nan
            if ui.ok(prem):
                sub = f"ADR 溢價 {ui.pct(prem, 1)}（vs 2330 收盤）<br>{'即時 ' if d.get('live') else ''}{ui.esc(d['time'])}"
        name = {"NQF": "那斯達克期貨", "ESF": "標普期貨", "TSM": "台積電 ADR", "TWD": "美元／台幣"}.get(k, linkage_name(k))
        tiles.append(ui.tile(name, val, d["chg"], sub, world_tile_chart(di, d.get("spark"))))
    st.markdown("<div class='kq-tiles'>" + "".join(tiles) + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='kq-muted' style='margin:-6px 0 10px'>小圖＝最近一個交易時段的分時走勢，虛線＝平盤（前一日收盤），"
                "線在上方紅、下方綠；美股收盤後顯示前一晚的完整走勢（台北時間）。取不到分時時改顯示近 1 月收盤（虛線＝期初）。</div>",
                unsafe_allow_html=True)
    lvl, txt, risk = risk_light()
    stt = world_state_now()
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
        st.markdown(ui.panel(f"美股風險燈號（{world_label(stt)}）", body), unsafe_allow_html=True)
    with c2:
        sox = snap.get("SOX", {})
        lab, row, adv = overnight_read(sox.get("chg", np.nan))
        if adv:
            nq = snap.get("NQF")
            extra = f"；那斯達克期貨目前 {ui.chg(nq['chg'])}" if nq else ""
            body = (f"<div style='margin-bottom:6px'>費半 <b>{ui.chg(sox['chg'])}</b>（{ui.esc(us_session_label(sox))}）{extra}　"
                    f"<span class='kq-tag'>{ui.esc(lab)}</span></div><div class='kq-muted'>{row.get('stat', '')}</div>"
                    f"<div style='margin-top:8px'>{ui.esc(adv)}</div>")
            n225, ks = snap.get("N225"), snap.get("KS11")
            if n225 or ks:
                body += ("<div class='kq-muted' style='margin-top:8px'>日韓 8:00 開盤（比台股早 1 小時）："
                         + "、".join(f"{nm} {ui.chg(x['chg'])}" for nm, x in (("日經", n225), ("韓股", ks)) if x)
                         + "；台股與日韓同日報酬相關係數約 0.6–0.7，開盤前可當領先參考。</div>")
            st.markdown(ui.panel("開盤前判讀（依歷史統計）", body), unsafe_allow_html=True)
    theme_section()
    intraday_section()
    linkage_ic_section()
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
            link_oc_text(yl),
            "美股 23 小時交易：Nasdaq 與 NYSE Arca 預計 2026-12-06 起週日至週五 23×5 交易（SEC 已核准 Nasdaq 方案）。"
            "夜盤時段 美東 21:00–04:00 ＝ 台北 10:00–17:00（冬令）／09:00–16:00（夏令），會與台股交易時間重疊。"
            "影響：台積電 ADR、美股科技股將在台股盤中即時交易，「隔夜跳空」可能縮小、改為盤中同步波動；"
            "本系統每日收盤後選股、隔日開盤執行的設計不受影響，盤中可多參考本頁的期貨／ADR 報價。"]), unsafe_allow_html=True)
    lead_section(compact=True)
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
        ("盤後｜決定明日標的", "每個交易日 17:23 出初版（約 18:30 完成），21:40 補融資券出完整版（約 23:00）。", [
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
        ("持有｜何時賣出", "模型不設固定停利停損，讓獲利奔跑；大盤急跌時先看「盤勢儀表板」的急跌應變。", [
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
        "0050 不是必要：關閉「包含 0050」即為全股票配置（報酬較高、波動較大）。短線與中線名單重疊約 7～10 成，兩者擇一。",
        "名單中同一產業有 3 檔以上時（例如航運），想降低波動可只買其中 2 檔、其餘名額用下一名。"
        "v2.0：大盤修正 ≥10% 且「盤勢儀表板」的止穩訊號 ≤2 項時，0050（或其他大盤型 ETF）暫時多 20 個百分點；止穩後恢復。",
        "ETF 當核心長期持有、不用擇時；配息與不配息看含息「年化報酬」比較即可（見「精選ETF」）。"])),
        unsafe_allow_html=True)
    st.markdown(ui.panel("大盤急跌時（v2.2）", ui.bullets([
        "先看「盤勢儀表板」：上方急跌警報、燈號與今日分時；「急跌應變」分頁有歷史統計與分批計畫。",
        "歷史上急跌當天很少是最低點（費半大跌後只有 2/14 次），之後 20 日內平均還會再跌約 6%；但 60 日後多數較高 → 不殺低、不重壓、分三批。",
        "第 1 批：回落 ≥10% 且盤中止跌（加權從低點反彈、那指期翻紅等）；第 2 批：止穩訊號 ≥3；第 3 批：站回月線且止穩 ≥5；跌破前低再跌 5% 暫停。",
        "已持有的股票照續抱門檻與警戒價處理；長線模型在 B1／B2 階段會自動降低持股水位。"])), unsafe_allow_html=True)
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


def auto_panel():
    from kosto import tournament as T
    au = (RES or {}).get("auto") or {}
    st.markdown(ui.section("冠軍／挑戰者（每日自動比較）",
                           f"規則：開發期超額多 {au.get('rule', {}).get('min_gain', 0.02):.0%} 以上、資訊比率不降、2019–20 不退步、"
                           f"最大回撤不深於冠軍 {float(au.get('rule', {}).get('max_mdd_worse') or 0.02) * 100:.0f} 個百分點（v2.4），"
                           f"連續 {au.get('rule', {}).get('streak', 10)} 個交易日第一名"
                           + (f"→ 先前瞻觀察 {CFG0['research'].get('auto_forward_days', 60)} 個交易日、新資料上不輸冠軍才換上（v3.15 三級晉級）"
                              if int(CFG0["research"].get("auto_forward_days", 0) or 0) > 0 else "才換上")
                           + ("" if au.get("enabled", True) else "（目前設定為只比較、不自動換）")), unsafe_allow_html=True)
    if not au.get("table"):
        st.markdown("<div class='kq-muted'>下一次每日研究後開始記錄。</div>", unsafe_allow_html=True)
        return
    rows, mc = [], []
    for h in [x for x in HORIZON_KEYS if x in au["table"]]:
        champ = au.get("champions", {}).get(h) or T.base_params(CFG0, h)
        ld = (au.get("leader") or {}).get(h) or {}
        need = au.get("rule", {}).get("streak", 10)
        lead = (f"{ui.esc(T.describe(ld['params'], champ))}（超額 +{ld['d_excess']:.1%}，已連續 {ld['streak']}/{need} 天）"
                if ld else "沒有挑戰者明顯勝出")
        ob = (au.get("observe") or {}).get(h) or {}
        if ob:
            fd = ob.get("fwd_diff")
            lead += (f"<br><span class='kq-sub'>前瞻觀察中：{ui.esc(T.describe(ob['params'], champ))}（{ob.get('since', '')} 起 {ob.get('fwd_days', 0)} 天"
                     + (f"，{'領先' if fd >= 0 else '落後'} {fd:+.1%}" if fd is not None else "") + "）</span>")
        last = (au.get("last_switch") or {}).get(h)
        rows.append([f"<b>{HZ[h]}</b>", ui.esc(T.describe(champ)), lead, last or "–"])
        mc.append(ui.mcard(f"<span class='nm'>{HZ[h]}</span>", "", "", f"冠軍：{ui.esc(T.describe(champ))}<br>挑戰者：{lead}"))
    st.markdown(ui.resp(ui.table(["期間", "目前模型（冠軍）", "領先的挑戰者", "上次換模型"], rows, left=(0, 1, 2)), "".join(mc)),
                unsafe_allow_html=True)
    rb = {h: x for h, x in (au.get("risk_blocked") or {}).items() if x}
    if rb:
        st.markdown("<div class='kq-muted' style='margin-top:6px'>風險護欄擋下（報酬較高但回撤較深）：" + "；".join(
            f"{HZ.get(h, h)} " + "、".join(f"{ui.esc(T.describe(b['params'], (au.get('champions') or {}).get(h) or T.base_params(CFG0, h)))}"
                                         f"（超額 {b['d_excess']:+.1%}、回撤 {b['d_mdd']:+.1%}）" for b in x)
            for h, x in rb.items()) + "</div>", unsafe_allow_html=True)
    if au.get("log"):
        st.markdown("<div class='kq-muted' style='margin-top:6px'>換模型紀錄：" + "；".join(
            f"{x['date']} {HZ.get(x['h'], x['h'])}：{ui.esc(x['text'])}" for x in au["log"][-6:]) + "</div>", unsafe_allow_html=True)
    with st.expander("今天的完整比較表"):
        for h in [x for x in HORIZON_KEYS if x in au["table"]]:
            t = au["table"][h].sort_values("d_excess", ascending=False)
            st.markdown(f"**{HZ[h]}**")
            st.dataframe(t[["desc", "dev_excess", "d_excess", "dev_ir", "d_ir"] + [c for c in ("d_mdd", "pre_excess") if c in t] + ["qualified"]]
                         .rename(columns={"desc": "設定（相對冠軍）", "dev_excess": "開發期超額", "d_excess": "差異", "dev_ir": "資訊比率",
                                          "d_ir": "IR差", "d_mdd": "回撤差", "pre_excess": "2019-20超額", "qualified": "今日合格"}),
                         hide_index=True, width="stretch")


def forward_panel():
    from kosto import forward
    au = (RES or {}).get("auto") or {}
    log = au.get("picks_log") or []
    st.markdown(ui.section("前瞻紀錄（上線後的真實績效）", f"從 {au.get('live_since', '–')} 起每天保存實際公布的名單；"
                                                     "照隔日開盤買進、持有到期計算，不經回測挑選"), unsafe_allow_html=True)
    sm = forward.summary(B, log) if log else pd.DataFrame()
    if sm.empty:
        st.markdown("<div class='kq-muted'>剛開始累積；約 3 個月後有參考價值。</div>", unsafe_allow_html=True)
        return
    rows = [[HZ.get(h, h), f"{int(r['筆數'])}（到期 {int(r['已到期'])}）", ui.pct(r["平均報酬"], 2, color=True), ui.pct(r["0050同期"], 2),
             ui.pct(r["平均超額"], 2, color=True), f"{r['勝過0050比例']:.0%}"] for h, r in sm.iterrows()]
    st.markdown(ui.table(["期間", "名單數", "平均報酬", "0050 同期", "平均超額", "勝過 0050"], rows, left=(0,)), unsafe_allow_html=True)


def auto_research_tab():
    st.markdown(ui.panel("今日選股用的就是這裡的結果，不需要手動調整", ui.bullets([
        "每個交易日 17:23（初版）與 21:40（完整版），GitHub Actions 自動：①下載最新資料 ②在 30 個母池中<b>自動挑出最佳母池</b> ③<b>自動估計因子權重</b>"
        "（每天用全部資料重估）④<b>冠軍／挑戰者比較</b>：把目前模型和鄰近設定（因子數、假說比例、續抱倍數、持股檔數）在五種換股日上比較，"
        "挑戰者連續 10 個交易日明顯勝出就<b>自動換上</b> ⑤產生名單與完整報告。",
        "分數只用開發期（盲測期不參與挑選），並檢查 2019–20 這段沒用過的期間不可明顯退步，避免越調越貼歷史。",
        "新增「因子」（新的資料或算法）仍需改程式並做預先宣告的研究。v1.9 測了 11 個新因子，只有「營收年增相對同業」通過，"
        "以固定 15% 加入三個期間的分數；其餘照常計算、不進模型（docs/v1.9_protocol.md）。",
        "「自行試驗」分頁只給想自己測參數的人用，不會影響每日名單。"])), unsafe_allow_html=True)
    if not RES:
        st.info("尚無每日研究結果。")
        return
    pr = RES.get("protocol", {})
    st.markdown(f"<div class='kq-muted'>資料至 {pr.get('last_date')}｜樣本外起點 {pr.get('first_oos')}｜盲測起點 {pr.get('holdout_start')}｜"
                f"權重重估日：{'、'.join(pr.get('retrain_points', [])[-4:])}…</div>", unsafe_allow_html=True)
    auto_panel()
    forward_panel()
    st.markdown(ui.section("各期間的自動研究結果", "母池、模型、回測、目前權重"), unsafe_allow_html=True)
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


def snapshot_html() -> str:
    """v3.14 這份名單是哪一版：快照編號、雜湊鏈、資料包與研究結果是否一致。"""
    from kosto import snapshot as SN
    if not RES:
        return "<div class='kq-muted'>尚無每日研究結果。</div>"
    sn = RES.get("snapshot") or {}
    chain = ((RES.get("auto") or {}).get("snapshots")) or []
    ok, bad = SN.verify(chain)
    issues = SN.consistency(B.meta, RES)
    items = [("快照編號", f"<span class='kq-mono'>{ui.esc(sn.get('id', '–'))}</span>",
              f"資料日 {ui.esc(sn.get('data_asof', '–'))}｜模型 v{ui.esc(sn.get('model_version', '–'))}"),
             ("發布時間", ui.esc(str(sn.get("published_at", "–"))[:16].replace("T", " ")),
              f"同一天第 {chain[-1]['seq']} 版" if chain else "（v3.14 以前沒有快照）"),
             (ui.light("green" if ok and chain else "amber", "雜湊鏈"), f"{len(chain)} 筆",
              "完整（舊紀錄沒有被改動）" if ok and chain else (f"第 {bad} 筆斷開" if chain else "從 v3.14 開始累積")),
             (ui.light("green" if not issues else "amber", "資料包＝研究結果"), "一致" if not issues else "注意",
              "；".join(ui.esc(x) for x in issues) or f"設定雜湊 {ui.esc(sn.get('config_hash', '')[:8])}")]
    return ui.cards(items, raw=True)


def validate_html() -> str:
    """v3.14「🧪 模型驗證」（回應獨立同儕評審）：證據等級、三種基準、風險與成本、穩健性、消融、統計、可解釋性、評審對照。"""
    from kosto import review as rv
    f2 = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) else ui.esc(v)          # noqa: E731
    f1 = lambda v: f"{v:+.1f}" if isinstance(v, (int, float)) else ui.esc(v)          # noqa: E731
    h = f"<div class='kq-panel' style='margin-bottom:10px'>{ui.esc(rv.V314_NOTE)}</div>"
    h += ui.section("1. 證據等級", "回測數字不能都當成「樣本外」：只有上線後每天的前瞻紀錄是真正獨立的")
    h += ui.table(["等級", "期間", "能證明什麼"], [[f"<b>{ui.esc(a)}</b>", ui.esc(b), ui.esc(c)] for a, b, c in rv.V314_TIERS], left=(0, 1, 2))
    h += ui.section("2. 三種基準與歸因", "開發期、第一種換股日、年化 %；選股範圍＝同母池等權 − 0050，選股＝模型 − 同母池等權")
    h += ui.table(["期間／母池", "模型", "同母池等權", "0050", "選股範圍", "選股", "α（扣掉市場、規模、動能）", "動能係數", "規模係數", "最近 12 月 α t"],
                  [[f"<b>{ui.esc(n)}</b>", f"{m:.1f}", f"{e:.1f}", f"{b:.1f}", f1(sc), f1(se), f"{al:.1f}%（t {at:.2f}）", f"{bm:.2f}（t {tm:.1f}）",
                    f"{bs:.2f}", f"{ht:.2f}"] for n, m, e, b, sc, se, al, at, bm, tm, bs, ht in rv.V314_ATTR], left=(0,))
    h += ("<div class='kq-muted'>母池本身（中小型股等權）每年落後 0050；超額全部來自選股。選股有很強的動能與小型股傾向"
          "（動能係數約 1）。扣掉之後開發期 α 仍顯著，最近 12 個月期間太短、還不顯著。</div>")
    h += ui.section("3. 風險與成本壓力", "五種換股日平均、超額 %；v3.14 起回測一律「跌停賣不掉」")
    h += ui.table(["期間", "情境", "開發期超額", "開發期回撤", "最近 12 個月", "整段", "每年成本拖累"],
                  [[f"<b>{a}</b>", ui.esc(b), f"{c:.2f}", f"{d:.1f}", f2(e), f"{g:.2f}", f"{i:.1f}"] for a, b, c, d, e, g, i in rv.V314_STRESS],
                  left=(0, 1))
    h += "<div class='kq-muted'>短線換手最快、對成交成本最敏感：滑價 0.5% 時最近 12 個月轉為負。短線請用限價、避開流動性差的股票。</div>"
    h += ui.section("4. 穩健性", "營收相對同業的同業定義（預先宣告規則：中線開發期不少於現行 − 1pp 就改用前視風險較低的版本）")
    h += ui.table(["", "同業定義", "中線開發期", "IR", "回撤", "最近 12 個月", "整段", "短線 開發／整段", "長線 開發／整段", "IC 開發／最近 12 月", "判定"],
                  [[f"<b>{k}</b>", ui.esc(n), f"{a:.2f}", f"{b:.3f}", f"{c:.1f}", f"{d:.2f}", f"{e:.2f}", ui.esc(g), ui.esc(i), ui.esc(j), ui.esc(m)]
                   for k, n, a, b, c, d, e, g, i, j, m in rv.V314_B1], left=(0, 1, 10))
    h += "<div class='kq-muted' style='margin:8px 0 2px'><b>月營收可用日延後</b>（第一種換股日、新模型，超額 %）</div>"
    h += ui.table(["情境", "短線開發期", "中線開發期", "中線最近 12 個月", "中線整段", "長線開發期"],
                  [[ui.esc(n), f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{d:.2f}", f"{e:.2f}"] for n, a, b, c, d, e in rv.V314_B2], left=(0,))
    h += "<div class='kq-muted' style='margin:8px 0 2px'><b>五種換股日的範圍</b>（新模型，超額 %）</div>"
    h += ui.table(["期間／母池", "開發期 平均", "最差", "最好", "最近 12 個月 平均", "最差"],
                  [[f"<b>{ui.esc(n)}</b>", f"{a:.1f}", f"{b:.1f}", f"{c:.1f}", f"{d:.1f}", f"{e:.1f}"] for n, (a, b, c), (d, e) in rv.V314_PHASE], left=(0,))
    h += ui.section("5. 結構消融與集中度", "五種換股日平均、超額 %：拿掉一部分後還剩多少")
    h += ui.table(["版本", "短線開發期", "中線開發期", "中線回撤", "中線最近 12 個月", "中線整段", "長線開發期"],
                  [[ui.esc(n), f"{a:.2f}", f"{b:.2f}", f"{c:.1f}", f2(d), f"{e:.2f}", f"{g:.2f}"] for n, a, b, c, d, e, g in rv.V314_ABL], left=(0,))
    mx = rv.V314_MIX
    h += (f"<div class='kq-muted'>固定假說與稀疏模型各自都差很多，合在一起才是現在的水準。0050＋中線各半（開發期）：年化 {mx['mix'][0]:.1f}%、"
          f"回撤 {mx['mix'][1]:.1f}%、夏普 {mx['mix'][2]:.2f}（模型 {mx['model'][0]:.1f}%／{mx['model'][1]:.1f}%／{mx['model'][2]:.2f}；"
          f"0050 {mx['0050'][0]:.1f}%／{mx['0050'][1]:.1f}%／{mx['0050'][2]:.2f}）→ 怕回撤的人，各半的回撤接近 0050、夏普接近模型。</div>")
    h += ui.section("6. 統計", "Newey–West t、20 日區塊自助法 90% 區間（年化超額 %）、Deflated Sharpe（以 1,800 次試驗計）")
    h += ui.table(["母池", "開發期 t", "開發期 90% 區間", "整段 t", "整段 90% 區間", "最近 12 個月 90% 區間", "開發期為正機率", "Deflated Sharpe"],
                  [[f"<b>{ui.esc(n)}</b>", f"{a:.2f}", ui.esc(b), f"{c:.2f}", ui.esc(d), ui.esc(e), f"{g:.0%}",
                    f"<b style='color:{'#15803d' if i >= 0.95 else '#b45309'}'>{i:.2f}</b>"] for n, a, b, c, d, e, g, i in rv.V314_STATS], left=(0,))
    h += "<div class='kq-muted'>Deflated Sharpe ≥ 0.95 才算統計上確定；目前 0.68～0.88：有優勢的機會較大，但還不能排除運氣。</div>"
    cr = rv.V314_CORR
    h += ui.section("7. 可解釋性", f"中線 BLEND500：權重 > 1% 的 {cr['n']} 個因子佔 {cr['share']:.0%}；有效因子數 {cr['eff']:.1f}")
    h += ui.table(["因子 A", "因子 B", "相關"], [[ui.esc(a), ui.esc(b), f"{c:.2f}"] for a, b, c in cr["high"]], left=(0, 1))
    h += ui.table(["家族", "IC 開發期", "扣掉動能後", "IC 最近 12 個月", "扣掉動能後"],
                  [[ui.esc(n), f"{a:.3f}", f"{b:.3f}", f"{c:.3f}", f"{d:.3f}"] for n, a, b, c, d in cr["inc"]], left=(0,))
    h += ui.section("8. 評審對照表", "獨立同儕評審（2026-11-19）每一項的處理；預先宣告：docs/v3.14_protocol.md")
    h += ui.table(["評審項目", "優先", "這次怎麼做", "結果"], [[f"<b>{ui.esc(a)}</b>", ui.esc(b), ui.esc(c), ui.esc(d)] for a, b, c, d in rv.V314_REVIEW],
                  left=(0, 2, 3))
    n_ideas = len(rv.IDEAS)
    n_ok = sum(1 for x in rv.IDEAS if x[5].startswith("✅"))
    h += (f"<div class='kq-muted'>試驗紀錄：歷次想法 {n_ideas} 組，其中 {n_ok} 組改變了模型（完整清單在「盤勢儀表板 → 模型體檢」）；"
          "若把每個想法的參數組合都算進去，約 1,800 次，所以統計檢定用 1,800 次試驗校正。</div>")
    return h


def v315_html() -> str:
    """v3.15 綜合第二、三份獨立評審：兩份意見對照、10 個事先固定的候選、0050 核心＋衛星、PBO。"""
    from kosto import review as rv
    f2 = lambda v: "0" if abs(v) < 0.005 else f"{v:+.2f}"          # noqa: E731
    f1 = lambda v: "0" if abs(v) < 0.05 else f"{v:+.1f}"           # noqa: E731
    h = ui.section("v3.15 綜合第二、三份評審", "每個建議寫成一個事先固定的候選，同一套資料與門檻；預先宣告：docs/v3.15_protocol.md")
    h += f"<div class='kq-panel' style='margin-bottom:10px'>{ui.esc(rv.V315_NOTE)}</div>"
    h += ui.table(["議題", "評審二（KOSTO Next）", "評審三（架構重構藍圖）", "這次怎麼處理"],
                  [[f"<b>{ui.esc(a)}</b>", ui.esc(b), ui.esc(c), ui.esc(d)] for a, b, c, d in rv.V315_COMPARE], left=(0, 1, 2, 3))
    b = rv.V315_BASE
    h += (f"<div class='kq-muted' style='margin:10px 0 2px'><b>候選與基準的差</b>（五種換股日平均，百分點；基準 v3.14 開發期超額 短 {b['dev'][0]:.1f}／中 {b['dev'][1]:.1f}／"
          f"長 {b['dev'][2]:.1f}；回撤正＝較淺）</div>")
    rows = []
    for k, n, src, kind, dv, ir, md, oo, hd, cons, q, vd in rv.V315_CAND:
        ok = vd.startswith("門檻全過")
        rows.append([f"<b>{k}</b>", ui.esc(n) + (ui.evidence("observe") if ok else ""), ui.esc(src), ui.esc(kind),
                     "／".join(f2(x) for x in dv), f"{ir:+.3f}" if ir else "–", "／".join(f1(x) for x in md), "／".join(f1(x) for x in oo),
                     f1(hd) if hd else "–", ui.esc(cons), f"{q:.2f}", f"<b>{ui.esc(vd)}</b>" if ok else ui.esc(vd)])
    h += ui.table(["", "候選", "來源", "類型", "開發期 短／中／長", "中線 IR", "回撤 短／中／長", "整段 短／中／長", "中線最近 12 月", "一致性", "q", "判定"],
                  rows, left=(0, 1, 2, 11))
    h += "<div class='kq-muted' style='margin:10px 0 2px'><b>兩種組合模式：0050 核心＋模型衛星</b>（第一種換股日；年化／最大回撤／夏普）</div>"
    pc = lambda t3: f"{t3[0]:.1f}%／{t3[1]:.1f}%" + (f"／{t3[2]:.2f}" if len(t3) > 2 else "")      # noqa: E731
    h += ui.table(["0050 比例", "中線 開發期", "中線 最近 12 個月", "長線 開發期", "長線 最近 12 個月"],
                  [[f"<b>{c:.0%}</b>", pc(a), pc(b_), pc(d), pc(e)] for c, a, b_, d, e in rv.V315_CORE], left=(0,))
    stt = rv.V315_STATS
    h += (f"<div class='kq-muted'>0050 七成＋模型三成的回撤比只買 0050 還淺、年化多約 12 個百分點：怕回撤時調整組合比例，比改選股模型有效（報酬型候選都沒有過關）。"
          f"短中線名單平均重疊 {stt['overlap']:.0%}（兩者擇一操作）。回測過度擬合機率 PBO {stt['pbo']:.2f}（{stt['pbo_cfg']} 個變體、CSCV 16 段）。</div>")
    return h


def v316_html() -> str:
    """v3.16 盤前日報的歷史驗證與前瞻命中率。"""
    from kosto import review as rv
    h = ui.section("盤前日報的歷史驗證（v3.16）", "2019～2023 估計、2024 年後檢驗；預先宣告：docs/v3.16_protocol.md")
    h += ui.table(["", "規則", "檢驗期結果", "門檻", "證據"], [[f"<b>{a}</b>", ui.esc(b), ui.esc(c), ui.esc(d), ui.evidence(e)] for a, b, c, d, e in rv.V316_RULES],
                  left=(0, 1, 2, 3))
    h += ui.table(["預估開盤", "天數", "實際開盤平均", "開盤→收盤（大型股）"],
                  [[ui.esc(a), str(b), f"{c:+.2f}%", f"{d:+.2f}%"] for a, b, c, d in rv.V316_BUCKETS], left=(0,))
    _, hist = morning_files()
    if hist:
        from kosto import morning as M
        sc = morning_score(BID, M.dumps(hist))
        if sc.get("n"):
            lt = "日報前瞻：{} 天已有實際開盤，方向命中 {}、平均誤差 {:.2f}%".format(sc["n"], "–" if sc.get("hit") is None else f"{sc['hit']:.0%}", sc["mae"] * 100)
            h += f"<div>{ui.light('green' if (sc.get('hit') or 0) >= 0.65 else 'amber', lt)}</div>"
    h += f"<div class='kq-muted'>{ui.esc(rv.V316_NOTE)}</div>"
    return h


def diagnostics_html() -> str:
    """v3.15 更新後自動異常診斷（第二份評審 9C；kosto/diagnose.py）：只警告、不改名單。"""
    dg = (RES or {}).get("diagnostics")
    h = ui.section("更新後異常診斷", "名單突變、缺值進榜、同產業過半、前瞻長期落後、資料未完整、母池更換；只警告，不改名單")
    if dg is None:
        return h + "<div class='kq-muted'>從 v3.15 的每日研究開始產生。</div>"
    if not dg:
        return h + f"<div>{ui.light('green', '今天沒有發現異常')}</div>"
    return h + "".join(f"<div style='margin:2px 0'>{ui.light('amber' if x['level'] == 'warn' else 'green', x['text'])}</div>" for x in dg)


def replay_section():
    """v3.15 每日歷史名單重現（第二份評審 9B）：當天實際公布的名單，不是用現在的模型回頭重算。"""
    au = (RES or {}).get("auto") or {}
    log = au.get("picks_log") or []
    st.markdown(ui.section("每日歷史名單重現", "選日期，看當天實際公布的名單（前瞻紀錄＋快照；不是用現在的模型回頭重算）"), unsafe_allow_html=True)
    dates = sorted({x["date"] for x in log}, reverse=True)
    if not dates:
        st.markdown("<div class='kq-muted'>還沒有紀錄。</div>", unsafe_allow_html=True)
        return
    d = st.selectbox("公布日期", dates, key="kq_replay")
    rows = [[f"<b>{HZ.get(x['h'], x['h'])}</b>", " ".join(f"{t} {ui.esc(NAMES[TIDX[t]]) if t in TIDX else ''}" for t in x["picks"])]
            for x in log if x["date"] == d]
    snaps = [x for x in (au.get("snapshots") or []) if x.get("data_asof") == d]
    st.markdown(ui.table(["期間", "當天公布的名單"], rows, left=(0, 1))
                + ("<div class='kq-muted'>快照：" + "；".join(f"{ui.esc(x['id'])} 第 {x['seq']} 版（{ui.esc(str(x.get('published_at', ''))[:16].replace('T', ' '))}）"
                                                         for x in snaps) + "</div>" if snaps else
                   "<div class='kq-muted'>這一天沒有快照（v3.14 以前只保存名單）。</div>"), unsafe_allow_html=True)


def page_validate():
    header()
    st.markdown(ui.section("🧪 模型驗證", "這份名單是哪一版、模型到底有沒有用、哪些證據還不夠（v3.14～v3.15 回應三份獨立評審）"), unsafe_allow_html=True)
    st.markdown(snapshot_html(), unsafe_allow_html=True)
    st.markdown(f"<div class='kq-muted' style='margin:4px 0'>證據徽章：{ui.evidence('tested')} 經過歷史回測與門檻的核心訊號（選股排名）；"
                f"{ui.evidence('ref')} 輔助資訊（價位尺、價格適合度、環境卡、標籤、持股建議）；{ui.evidence('observe')} 影子模型每天追蹤、尚未進模型</div>",
                unsafe_allow_html=True)
    st.markdown(diagnostics_html(), unsafe_allow_html=True)
    forward_panel()
    replay_section()
    st.markdown(v316_html(), unsafe_allow_html=True)
    st.markdown(v315_html(), unsafe_allow_html=True)
    st.markdown(validate_html(), unsafe_allow_html=True)


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


# ================================================================== v3.0 產業鏈地圖（上下游、細族群、AI 伺服器拆解、題材、消息衝擊評估）
@st.cache_data(show_spinner=False)
def chain_metrics(bid) -> pd.DataFrame:
    from kosto import chainmap as cm
    return cm.metrics(B)


@st.cache_data(show_spinner=False)
def chain_groups(bid) -> pd.DataFrame:
    from kosto import chainmap as cm
    return cm.group_stats(chain_metrics(bid))


@st.cache_data(show_spinner="計算歷史同類事件…")
def event_summary_cached(bid, key: str) -> tuple[pd.DataFrame, list]:
    from kosto import impact as I
    ev = I.event_returns(B, I.EVENT_SETS[key][1])
    return I.event_summary(ev), sorted(ev["event"].unique().tolist()) if len(ev) else []


@st.cache_data(show_spinner=False)
def since_cached(bid, date: str) -> pd.Series:
    from kosto import chainmap as cm
    return cm.since_event(B, date)


@st.cache_data(show_spinner=False)
def camp_cached(bid):
    """陣營輪動：近 120 日路徑、歷史競爭者利多日的反應（只在該陣營的代表股＋共用）。"""
    from kosto import impact as I
    paths = I.camp_paths(B, 120)
    ev = I.event_returns(B, I.EVENT_SETS["rival"][1], I.camp_index(B))
    sm = I.event_summary(ev)
    return paths, sm, sorted(ev["event"].unique().tolist()) if len(ev) else [], I.us_moves(B)


def camp_section(camps, key: str, title: str = "陣營輪動：輝達失分，誰補位？"):
    """v3.2 AI 晶片陣營輪動：輝達／AMD／ASIC 誰受傷、誰受惠＋近 120 日相對表現＋美國指標股。"""
    from kosto import chainmap as cm
    paths, sm, evs, us = camp_cached(BID)
    cell = lambda t: stock_cell(t) if t in TIDX else ui.esc(t)          # noqa: E731
    rows, mrows = cm.camp_rows(camps, sm, paths, cell)
    st.markdown(ui.section(title, "台灣 AI 供應商多數同時做輝達、AMD、ASIC 三個陣營；"
                           "只在單一陣營的公司才會明顯分化（代表股＝只在該陣營）"), unsafe_allow_html=True)
    st.markdown(ui.resp(ui.table(["陣營", "判斷", "多快反映", "原因", f"歷史競爭者利多日 當天／5 日（{len(evs)} 次）", "近 20 日相對大盤",
                                  "只在這個陣營的代表股"], rows, left=(0, 1, 3, 6)), "".join(mrows)), unsafe_allow_html=True)
    c1, c2 = st.columns([1.6, 1])
    if len(paths):
        c1.plotly_chart(cm.camp_figure(paths, dark=bool(DARK)), width="stretch", theme=None, key=f"{key}_camp",
                        config={"displaylogo": False, "displayModeBar": False})
    if len(us):
        urow = [[f"<b>{ui.esc(k)}</b>", ui.pct(r["r1"], 1, color=True), ui.pct(r["r5"], 1, color=True), ui.pct(r["r20"], 1, color=True)]
                for k, r in us.iterrows()]
        c2.markdown(ui.panel("美國指標股（前一晚、5 日、20 日）", ui.table(["", "前一晚", "5 日", "20 日"], urow, left=(0,))),
                    unsafe_allow_html=True)
    else:
        c2.markdown(ui.panel("美國指標股", "<div class='kq-muted'>輝達、AMD、博通、Alphabet 的報價會在每日更新的國際同業資料中提供。</div>"),
                    unsafe_allow_html=True)
    st.caption("歷史競爭者利多日：" + ("、".join(evs) if evs else "–") + "。只有「明確搶輝達份額」的消息（2025-11-26 Meta 傳採用 Google TPU）"
               "出現分化（ASIC 陣營 5 日 +4.5%、輝達陣營 +1.5%）；「AI 餅變大」的消息（OpenAI 與 AMD／博通合作）三個陣營一起漲。"
               "模型的動能與營收因子會在 1～2 季內跟著陣營輪動換股（中線名單 2025Q4 偏輝達、2026 上半年轉向 ASIC）。")


@st.cache_data(show_spinner=False)
def betas_cached(bid) -> pd.Series:
    from kosto import impact as I
    return I.betas(B)


@st.cache_data(show_spinner=False)
def heat_frame(bid, top: int, elec_only: bool) -> pd.DataFrame:
    from kosto import chainmap as cm
    return cm.treemap_frame(B, chain_metrics(bid), CH, top=top, elec_only=elec_only)


def heatmap_section(key: str, height: int | None = None, default_top: int = 300):
    """v3.1 台股熱力圖：面積＝近 20 日均成交金額、顏色＝漲跌（紅漲綠跌，越深越多）；電子依產業鏈層級與細族群分組。"""
    from kosto import chainmap as cm
    c1, c2, c3 = st.columns([1.3, 1.2, 1])
    per = c1.segmented_control("期間", list(cm.PERIODS), format_func=lambda k: cm.PERIODS[k][0], default="r1", key=f"{key}_per") or "r1"
    scope = c2.segmented_control("範圍", ["all", "elec"], format_func=lambda k: {"all": "全市場", "elec": "只看電子"}[k], default="all",
                                 key=f"{key}_scope") or "all"
    top = c3.segmented_control("檔數", [150, 300, 500], format_func=lambda k: f"前 {k}", default=default_top, key=f"{key}_top") or default_top
    df = heat_frame(BID, int(top), scope == "elec")
    if df is None or not len(df):
        st.info("熱力圖需要看盤資料包（下一次每日研究後提供）。")
        return
    live_n = 0
    if per == "r1" and market_phase() in ("盤前試撮", "盤中", "盤後"):     # v3.3：交易日改用證交所即時漲跌（研究資料還沒更新時）
        try:
            q, _ = quotes_for(df["t"].tolist())
            if len(q) and "chg" in q:
                q = q[q["date"].astype(str) > RDAY.strftime("%Y%m%d")] if "date" in q else q
                ch = q["chg"].dropna()
                if len(ch):
                    df = df.copy()
                    m = df["t"].isin(ch.index)
                    df.loc[m, "r1"] = df.loc[m, "t"].map(ch).astype(float)
                    live_n = int(m.sum())
        except Exception:  # noqa: BLE001
            live_n = 0
    fig = cm.treemap_figure(df, per, dark=bool(DARK), height=height or (560 if IS_MOBILE else 660))
    st.plotly_chart(fig, width="stretch", theme=None, key=f"{key}_fig", config={"displaylogo": False, "displayModeBar": False})
    up = df[df[per] > 0]["amt"].sum() / max(df["amt"].sum(), 1e-9)
    g = df.groupby("group").apply(lambda x: np.average(x[per].fillna(0), weights=x["amt"])).sort_values()
    st.caption((f"今日漲跌：{live_n} 檔用證交所即時報價（15 秒更新），其餘為 {RDAY:%m/%d} 收盤。" if live_n else "")
               + f"成交金額前 {int(top)} 名（{'電子' if scope == 'elec' else '全市場'}）；{cm.PERIODS[per][0]}上漲的成交額占 {up:.0%}。"
               f"最強族群：{'、'.join(f'{k} {v * 100:+.1f}%' for k, v in g.tail(3)[::-1].items())}；"
               f"最弱：{'、'.join(f'{k} {v * 100:+.1f}%' for k, v in g.head(3).items())}。點方塊可放大、點上方路徑可返回。資料到 {B.sig_dates[-1]:%m/%d}。")


def picks_all() -> set:
    return {p["ticker"] for h in HZ for p in latest(h).get("picks", [])}


def rank_mid_map() -> dict:
    try:
        rk = ranks(BID, "mid")
        return {t: int(r) for t, r in rk["rank"].dropna().items()}
    except Exception:  # noqa: BLE001
        return {}


def chain_position_html(t: str) -> str:
    """個股頁：這檔在產業鏈的位置（細族群、上下游、同業、題材、目前重大消息的曝險）。"""
    from kosto import impact as I
    from kosto import supply as S
    gs = S.groups_of(t)
    if not gs:
        g = S.primary_group(t, CH.get(t))
        return (f"<div class='kq-muted'>這檔不在代表個股名單；依產業鏈標籤歸為「{ui.esc(S.group_name(g))}」。"
                "完整上下游請看「產業鏈地圖」。</div>")
    lines = []
    for g in gs:
        d = S.SUBGROUPS[g]
        role = next(r for x, r in d["members"] if x == t)
        peers_ = [f"{x} {NAMES[TIDX[x]]}" for x, _ in d["members"] if x != t and x in TIDX][:6]
        lines.append(f"<a href='./chain?g={g}' target='_self'><b>{ui.esc(d['name'])}</b></a>（{ui.esc(S.LAYER_NAME[d['layer']])}）：{ui.esc(role)}"
                     f"<br><span class='kq-muted'>上游：{ui.esc('、'.join(S.group_name(x) for x in S.suppliers(g)) or '原料、國外設備')}｜"
                     f"下游：{ui.esc('、'.join(S.group_name(x) for x in S.customers(g)) or '國外終端客戶')}｜同業：{ui.esc('、'.join(peers_))}</span>")
    th = [x for x in S.themes_of(t) if "陣營" not in x]
    if th:
        lines.append("題材：" + "、".join(ui.tag(x) for x in th))
    cp = S.camps_of(t)
    if cp:
        lines.append("AI 晶片陣營：" + "、".join(ui.tag(S.CAMPS[c]["name"].split("（")[0]) for c in cp)
                     + ("（同時供應多個陣營，輝達與競爭者此消彼長時影響較小）" if len(cp) > 1 else "（只在這個陣營，陣營輪動時波動較大）"))
    ex = []
    for sc in I.SCENARIOS:
        dep = [x for x in sc["impacts"] if x[0] in gs and x[1] != 0]
        if dep:
            d0 = min(dep, key=lambda x: x[1]) if any(x[1] < 0 for x in dep) else max(dep, key=lambda x: x[1])
            ex.append(f"{ui.esc(sc['name'].split('（')[0])}：<b style='color:{ui.DOWN if d0[1] < 0 else ui.UP}'>"
                      f"{'受傷' if d0[1] < 0 else '受惠'}{I.depth_label(d0[1])}</b>")
    if ex:
        lines.append("重大消息曝險：" + "；".join(ex))
    return ui.bullets(lines)


def page_chain():
    from kosto import chainmap as cm
    from kosto import impact as I
    from kosto import supply as S
    header()
    st.markdown(ui.section("產業鏈地圖", "台股電子業從材料到整櫃伺服器的上下游、細族群與代表個股；AI 與新題材；重大消息會傷到誰"),
                unsafe_allow_html=True)
    met = chain_metrics(BID)
    gs = chain_groups(BID)
    mk = cm.market_ref(met)
    lead = cm.lead_names(B, met)
    picks, rmid = picks_all(), rank_mid_map()
    cell = lambda t: stock_cell(t) if t in TIDX else ui.esc(t)          # noqa: E731
    tabs = st.tabs(["🗺️ 全景圖・細族群", "🔥 熱力圖", "🧩 AI 伺服器拆解", "🚀 題材地圖", "⚡ 消息衝擊評估", "📏 族群敏感度"])
    with tabs[1]:
        heatmap_section("cm_heat", default_top=150 if IS_MOBILE else 300)
    tabs = [tabs[0], *tabs[2:]]
    with tabs[0]:
        q = st.query_params.get("g", "")
        sel = q if q in S.SUBGROUPS else None
        st.markdown(cm.overview_html(gs, sel), unsafe_allow_html=True)
        opts = list(S.SUBGROUPS)
        g = st.selectbox("細族群拆解（也可以直接點上方方塊）", opts, index=opts.index(sel) if sel else opts.index("server"),
                         format_func=lambda x: f"{S.LAYER_NAME[S.SUBGROUPS[x]['layer']]}｜{S.SUBGROUPS[x]['name']}", key="cm_sel")
        if g != sel:
            st.query_params["g"] = g
        st.markdown(ui.panel(f"拆解：{S.SUBGROUPS[g]['name']}", cm.group_detail(g, met, rmid, picks, cell)), unsafe_allow_html=True)
        st.caption("代表個股依主要營收來源人工整理，同一家公司可能出現在多個族群（例：台達電＝電源＋散熱）；名單不是推薦，也不代表與某客戶有公開供應關係。"
                   f"資料到 {B.sig_dates[-1]:%m/%d}；「相對大盤」＝減成交金額前 800 名平均。")
    with tabs[1]:
        mob = "".join(ui.mcard(f"<b>{ui.esc(nm)}</b>", ui.pct(gs.loc[g, "r20"], 1, color=True), ui.esc(plain),
                               f"{ui.esc('、'.join(lead.get(g, [])[:4]))}｜<a href='./chain?g={g}' target='_self'>{ui.esc(S.group_name(g))}</a>")
                      for nm, plain, g in cm.CHIP_PARTS[::-1] + cm.RACK_PARTS[::-1])
        st.markdown(cm.CSS + ui.resp(cm.rack_svg(lead), "<div class='kq-muted' style='font-size:12.5px;margin-bottom:6px'>由晶片（最內層）到整櫃（最外層）："
                                     "</div>" + mob), unsafe_allow_html=True)
        st.markdown(ui.panel("怎麼看這張圖", ui.bullets([
            "<b>由內到外</b>：晶片（台積電做）→ 封裝成一顆（CoWoS）→ 裝在載板與主機板 → 加上散熱、電源 → 放進托盤 → 18 盤組成一整櫃 → 交給雲端巨頭。",
            "<b>越靠近晶片，越難替代、毛利越高</b>（台積電、CoWoS、ABF 載板）；<b>越靠近整櫃，量越大、毛利越低</b>（組裝廠），但最先感受到出貨時程變化。",
            "<b>AI 伺服器跟一般伺服器的差別</b>：晶片更燙（要液冷）、更耗電（電源與 BBU 升級）、更重（滑軌與機殼升級）、傳輸更快（高速板材、光通訊）"
            "——這些「升級」就是台廠受惠最多的地方。"])), unsafe_allow_html=True)
        rows = []
        for nm, plain, g in cm.RACK_PARTS + cm.CHIP_PARTS:
            r = gs.loc[g]
            rows.append([f"<b>{ui.esc(nm)}</b>", ui.esc(plain), f"<a href='./chain?g={g}' target='_self'>{ui.esc(S.group_name(g))}</a>",
                         ui.esc("、".join(lead.get(g, [])[:5])), ui.pct(r["r20"], 1, color=True), ui.pct(r["x20"], 1, color=True)])
        st.markdown(ui.resp(ui.table(["零件", "白話", "細族群", "代表公司", "近 20 日", "相對大盤"], rows, left=(0, 1, 2, 3)), ""),
                    unsafe_allow_html=True)
    with tabs[2]:
        cards = cm.theme_cards(met, cell, mk)
        c1, c2 = st.columns(2)
        for k, h in enumerate(cards):
            (c1 if k % 2 == 0 else c2).markdown(h, unsafe_allow_html=True)
        from kosto import impact as I2
        camp_section(I2.SCEN_BY_ID["nvidia_ceo"]["camps"], "cm_theme", "AI 晶片陣營輪動（判斷欄＝輝達執行長請辭情境）")
        st.caption("階段：成熟＝營收已明顯貢獻；成長＝營收開始貢獻；題材＝主要看預期（股價波動大、容易「利多出盡」）。"
                   "新題材／轉型：CPO 矽光子、800V 高壓直流、FOPLP／CoPoS 面板級封裝、玻璃基板、人形機器人。")
    with tabs[3]:
        names = [sc["name"] for sc in I.SCENARIOS]
        k = st.selectbox("選一個重大消息", range(len(names)), format_func=lambda i: names[i], key="cm_sc")
        sc = I.SCENARIOS[k]
        hist, evs = (event_summary_cached(BID, sc["analog"]) if sc.get("analog") else (None, []))
        since = since_cached(BID, sc["date"])
        a1, a2 = st.columns([1.25, 1])
        a1.markdown(ui.panel("發生什麼事", ui.bullets([ui.esc(x) for x in sc["facts"]])
                             + "<div class='kq-muted' style='font-size:12px'>來源：" + "、".join(
                                 f"<a href='{u}' target='_blank' rel='noopener'>{ui.esc(n)}</a>" for n, u in sc.get("sources", [])) + "</div>"),
                    unsafe_allow_html=True)
        a2.markdown(ui.panel("影響怎麼傳下去（由快到慢）", cm.CSS + cm.steps_html(sc["chain"])), unsafe_allow_html=True)
        rows, mrows = cm.impact_rows(sc, hist, since, met, lead)
        an = I.EVENT_SETS.get(sc.get("analog") or "", ("", [], ""))
        st.markdown(ui.section("各族群受影響深淺", "基本面判斷（人工）＋" + (f"歷史同類事件（{ui.esc(an[0])}，{len(evs)} 次）" if evs else
                                                                         "歷史同類事件：無（看下方族群路徑）") + "＋事件後至今的實際表現（相對大盤）"),
                    unsafe_allow_html=True)
        st.markdown(ui.resp(ui.table(["細族群", "基本面判斷", "多快反映", "原因", "歷史同類 5 日／20 日", "事件後至今", "代表公司"], rows,
                                     left=(0, 1, 3, 6)), "".join(mrows)), unsafe_allow_html=True)
        st.caption(("歷史同類事件：" + "、".join(evs) + "。" if evs else "")
                   + ("「事件後至今」：資料庫還沒有事件日之後的資料（下一次每日更新後自動出現）。" if not len(since) else
                      f"「事件後至今」：{sc['date']} 前一日收盤到 {B.sig_dates[-1]:%m/%d}。") + " " + ui.esc(sc.get("note", "")))
        if sc.get("camps"):
            camp_section(sc["camps"], f"cmsc_{sc['id']}")
        if sc["id"] == "rates":
            ch, _ = event_summary_cached(BID, "cut")
            hk, _ = event_summary_cached(BID, "hike")
            rr = [[ui.esc(S.group_name(g)), ui.pct(hk.loc[g, "d5"], 1, color=True) if g in hk.index else "–",
                   ui.pct(ch.loc[g, "d5"], 1, color=True) if g in ch.index else "–"] for g in hk.index]
            with st.expander("升息日 vs 降息日：各族群之後 5 日相對大盤（2019 年以來）"):
                st.markdown(ui.table(["細族群", "升息後 5 日", "降息後 5 日"], rr, left=(0,)), unsafe_allow_html=True)
                st.caption("單一次升降息決議的影響都很小（多在 ±1% 內）；真正的影響是整段升息循環對高本益比族群的評價壓力（2022 年）。")
        if sc.get("theme"):
            js = I.theme_index(B, sc["theme"])
            p = I.path(B, js, sc["date"], 60, 90)
            if len(p) > 10:
                import plotly.graph_objects as go
                from kosto.charts import THEME
                BG, GRID, TXT, MUTED = THEME[bool(DARK)]
                fig = go.Figure(go.Scatter(x=p.index, y=p.values * 100, mode="lines", line=dict(color=ui.UP, width=2)))
                fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
                fig.update_layout(height=260, margin=dict(l=8, r=40, t=10, b=40), paper_bgcolor=BG, plot_bgcolor=BG,
                                  font=dict(color=TXT, family="Inter, Noto Sans TC, sans-serif"),
                                  xaxis=dict(title="事件前後交易日", gridcolor=GRID), yaxis=dict(title="累積超額 %", gridcolor=GRID, side="right"))
                st.markdown(ui.section(f"{S.THEME_BY_ID[sc['theme']]['name']}族群：事件前後相對大盤的累積表現",
                                       f"代表股等權，事件前一日＝0；{len(js)} 檔"), unsafe_allow_html=True)
                st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "displayModeBar": False})
                pre = (1 / (1 + p.get(-60, p.iloc[0])) - 1) if p.index.min() <= -20 else np.nan
                post = {d_: p.get(d_) for d_ in (5, 20, 60) if d_ in p.index}
                st.caption(f"事件前 60 日到事件前一日：相對大盤 {ui.pct(pre, 1)}；事件後 " + "、".join(f"{d_} 日 {ui.pct(v_, 1)}" for d_, v_ in post.items())
                           + "。題材股常見「上市前漲、上市後利多出盡」。")
        ex = []
        cdep = {c: d for c, d, *_ in sc.get("camps", [])}
        for h in HZ:
            for pk in latest(h).get("picks", []):
                t = pk["ticker"]
                hit = [x for x in sc["impacts"] if x[0] in S.groups_of(t) and x[1] != 0]
                cps = S.camps_of(t) if cdep else []
                if hit or cps:
                    d0 = min(hit, key=lambda x: x[1]) if hit else None
                    ex.append((h, t, d0, cps))
        if ex:
            cname = {c: S.CAMPS[c]["name"].split("（")[0] for c in S.CAMPS}
            rows = [[HZ[h], cell(t), ui.esc(S.group_name(d0[0])) if d0 else "–", cm.depth_bar(d0[1]) if d0 else "–"]
                    + ([("、".join(cname[c] for c in cps) + ("（多陣營，一進一出）" if len(cps) > 1 else "")) if cps else "–"] if cdep else [])
                    for h, t, d0, cps in ex]
            st.markdown(ui.section("模型名單在這個消息的曝險"), unsafe_allow_html=True)
            st.markdown(ui.table(["期間", "股票", "所屬細族群", "影響"] + (["AI 晶片陣營"] if cdep else []), rows, left=(0, 1, 2, 4)),
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='kq-muted'>模型名單沒有直接受這個消息影響的代表股。</div>", unsafe_allow_html=True)
        st.caption("基本面判斷是依營收曝險與傳導順序的人工整理（v3.0 研究），不進選股模型；歷史同類事件樣本少（3～12 次），只看方向與量級。")
    with tabs[4]:
        bs = betas_cached(BID)
        if len(bs):
            bs = bs.dropna().sort_values()
            rows = [[ui.esc(S.group_name(g)), f"{v:.2f}", f"<span class='cm-bar' style='width:{max(4, v * 160):.0f}px;background:var(--accent2)'></span>"]
                    for g, v in bs.items()]
            st.markdown(ui.section("對費半的敏感度（β）", "2023 年以來：前一晚費半漲跌 1%，隔天這個族群平均漲跌幾 %"), unsafe_allow_html=True)
            st.markdown(cm.CSS + ui.table(["細族群", "β", ""], rows, left=(0, 2)), unsafe_allow_html=True)
            st.caption("β 越高，美股半導體大跌時越容易被拖累（也越容易跟著反彈）。載板、先進封裝、ASIC 最敏感；通路、車用、PC 最不敏感。")
        sets = [k for k in I.EVENT_SETS]
        kk = st.selectbox("歷史事件組", sets, format_func=lambda x: I.EVENT_SETS[x][0], key="cm_es")
        sm, evs = event_summary_cached(BID, kk)
        if len(sm):
            rows = [[ui.esc(S.group_name(g)), ui.pct(r["d0"], 1, color=True), ui.pct(r["d5"], 1, color=True), ui.pct(r["d20"], 1, color=True),
                     f"{r['win5']:.0%}" if np.isfinite(r["win5"]) else "–"] for g, r in sm.iterrows()]
            st.markdown(ui.table(["細族群", "當天", "5 日", "20 日", "5 日勝過大盤比例"], rows, left=(0,)), unsafe_allow_html=True)
            st.caption(f"{I.EVENT_SETS[kk][0]}：{'、'.join(evs)}（{I.EVENT_SETS[kk][2]}）。相對成交金額前 800 名等權；依 5 日由弱到強排序。")


# ================================================================== navigation
# ================================================================== v3.4 美股選股（與台股模型完全分開）
from kosto.us import data as UD  # noqa: E402
from kosto.us import engine as UE  # noqa: E402
from kosto.us import model as UM  # noqa: E402
from kosto.us import names as UN  # noqa: E402
from kosto.us import view as UV  # noqa: E402

US_ASSET = "us_bundle.pkl.gz"
US_LOCAL = Path("artifacts/us_bundle.pkl.gz")


@st.cache_data(ttl=180, show_spinner=False)
def us_stamp(repo, token):
    info = release_info(repo, RESEARCH_TAG, token)
    return (info.get("assets", {}).get(US_ASSET) or {}).get("updated_at", "")


@st.cache_resource(show_spinner="下載美股資料包…", max_entries=1)
def us_load_github(repo, token, stamp):
    return UE.loads(fetch_asset(repo, RESEARCH_TAG, US_ASSET, token))


@st.cache_resource(show_spinner=False, max_entries=1)
def us_load_local(mtime):
    return UE.loads(US_LOCAL.read_bytes())


@st.cache_resource(show_spinner="建立美股資料（抓標普 500＋熱門股約 580 檔日線，約 1～2 分鐘，每天只需一次）…", max_entries=1)
def us_build_here(day: str):
    return UE.build(log=None)


def us_expected() -> str:
    """最近一個應該已經收盤的美股交易日（只看平日；美國假日當天會顯示「落後一天」屬正常）。"""
    et = UD.market_clock()["et"]
    d = et.tz_localize(None).normalize()
    if et.weekday() >= 5 or et.hour < 16:
        d = d - pd.offsets.BDay(1)
    return str(d.date())


def us_source() -> tuple[dict | None, str, str]:
    B, src, err = None, "", ""
    if REPO:
        try:
            stamp = us_stamp(REPO, TOKEN)
            if stamp:
                B, src = us_load_github(REPO, TOKEN, stamp), "GitHub 每日美股資料包"
        except Exception as e:  # noqa: BLE001
            err = f"雲端美股資料包讀取失敗：{str(e)[:80]}"
    if B is None and US_LOCAL.is_file():
        B, src = us_load_local(US_LOCAL.stat().st_mtime), "本機美股資料包"
    exp = us_expected()
    want = st.session_state.get("us_build") or B is None
    if want and not os.getenv("KOSTO_US_NOBUILD"):
        try:
            B2 = us_build_here(exp)
            if B is None or B2["asof"] >= B["asof"]:
                B, src = B2, "App 自行建立（Yahoo Finance）"
        except Exception as e:  # noqa: BLE001
            err = f"App 自行建立美股資料失敗：{str(e)[:100]}"
    return B, src, err


def us_slot(fast: bool) -> str:
    now = pd.Timestamp.now(tz="Asia/Taipei")
    return f"{now:%Y%m%d%H}{now.minute // 2}" if fast else f"{now:%Y%m%d%H}{now.minute // 30}"


@st.cache_data(ttl=900, show_spinner=False, max_entries=48)
def us_quotes(syms: tuple, slot: str):
    try:
        q, bars = UD.quotes(list(syms))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), {}
    return (q.set_index("sym") if len(q) else q), bars


@st.cache_data(ttl=1800, show_spinner=False, max_entries=96)
def us_daily(sym: str, slot: str) -> pd.DataFrame:
    try:
        return UD.daily(sym, "2y")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def us_live_now(clk) -> bool:
    return clk["phase"] in ("pre", "regular", "post", "overnight")


def us_strip(clk, iq) -> str:
    ph = UD.PHASE_ZH[clk["phase"]]
    col = {"pre": "#a16207", "regular": "#b4232a", "post": "#1e3a8a", "closed": "#334155", "overnight": "#6d28d9"}.get(clk["phase"], "#334155")
    night = (f"｜夜盤 {clk['night_tpe']}～{clk['pre_tpe']}" if clk.get("overnight_live")
             else f"｜那斯達克夜盤 12/7 起（台北 {clk.get('night_tpe', '10:00')}～{clk['pre_tpe']}）") if clk.get("night_tpe") else ""
    html = [f"<div class='kq-bar'><span class='kq-pill' style='background:{col};color:#fff'>美股{ph} 台北 {clk['tpe']:%H:%M}</span>",
            f"<span class='kq-item kq-hide-m'>盤前 {clk['pre_tpe']}｜開盤 {clk['open_tpe']}｜收盤 {clk['close_tpe']}{night}"
            f"（{'夏令' if clk['dst'] else '冬令'}）</span>"]
    for s, lab in (("ES=F", "標普期"), ("NQ=F", "那指期"), ("^GSPC", "標普"), ("^IXIC", "那指"), ("^SOX", "費半"), ("^VIX", "VIX")):
        if len(iq) and s in iq.index:
            r = iq.loc[s]
            html.append(f"<span class='kq-item{' kq-hide-m' if s in ('^IXIC', '^VIX') else ''}'>{lab}<b>"
                        f"{ui.num(r.get('price'), 1) if s == '^VIX' else ui.chg(r.get('chg'))}</b></span>")
    html.append("</div>")
    return "".join(html)


def us_validation_banner(R: dict) -> str:
    return ("<div class='kq-panel' style='border-left:4px solid #e9a23b'><b>⚠ 美股個股模型沒有通過驗證，名單只供參考；美股先看「💰 ETF 推薦」。</b>"
            "<div class='kq-muted' style='margin-top:4px'>用沒有倖存者偏差的標普 500 時點成分股（含下市公司）、2004–2021 共 18 年，"
            "測了三輪：4 種母池 × 9 種模型，再加大盤濾網、熊市濾網、巨型股趨勢 4 種——<b>沒有一種長期贏 SPY 3 個百分點以上</b>。"
            "動能類 2022 年後每年多 10～15 個百分點，但 2004–2021 每年少 3～10 個，另列「動能觀察名單」。"
            "建議<b>核心放 VOO／IVV，名單只當少量衛星</b>。詳見「驗證與說明」。</div></div>")


def us_pre_dict(bars: pd.DataFrame, base: float, clk) -> dict | None:
    """盤前分時（台北時間）→ intraday_figure 格式，平盤＝昨收。"""
    if bars is None or not len(bars) or not ui.ok(base):
        return None
    pre = bars[bars["session"] == "pre"]
    if len(pre) < 2:
        return None
    t = pd.DatetimeIndex(pre["t"]).tz_convert("Asia/Taipei").tz_localize(None)
    v = pre["price"].to_numpy(float)
    d0 = t[0].normalize()
    start = pd.Timestamp(f"{d0.date()} {clk['pre_tpe']}")
    end = pd.Timestamp(f"{d0.date()} {clk['open_tpe']}")
    return {"t": t, "v": v, "vol": pre["volume"].to_numpy(float), "prev": float(base), "last": float(v[-1]),
            "high": float(v.max()), "low": float(v.min()), "open": float(v[0]), "start": start, "end": max(end, t[-1]),
            "date": d0.date(), "time": t[-1], "tz": "Asia/Taipei", "live_point": False}


def us_bundle_daily(B, t: str) -> pd.DataFrame:
    """資料包內的日線（還原價換回實際成交價）。"""
    raw, c = B["RAW"][t].astype(float), B["C"][t].astype(float)
    ratio = (raw / c).where(c > 0)
    df = pd.DataFrame({"open": B["O"][t] * ratio, "high": B["H"][t] * ratio, "low": B["L"][t] * ratio, "close": raw,
                       "adj": c, "volume": B["V"][t].astype(float)})
    df.index = pd.DatetimeIndex(df.index)
    return df.dropna(subset=["close"])


def us_name_options(B) -> list[str]:
    info = B["info"]
    return list(dict.fromkeys(list(B["tickers"]) + list(B.get("etfs", [])) + list(info.index)))


def us_fmt(B, t: str) -> str:
    nm = UV.label(B, t)
    return f"{t}　{nm}" if nm != t else t


def us_list_section(B, h: str, q, live=None, mode="pre"):
    if h not in B["picks"]:
        return
    title = UV.list_title(B, h)
    note = ("名單依 " + B["asof"] + " 收盤計算" if not live else "以目前價格即時重算（沒有報價的股票視同平盤）")
    st.markdown(ui.section(title, note), unsafe_allow_html=True)
    st.markdown(UV.pick_table(B, h, q, live, mode), unsafe_allow_html=True)


def us_pre_tab(B, clk):
    ph = clk["phase"]
    cands = UE.candidates(B, 30)
    q, bars = us_quotes(tuple(cands), us_slot(ph in ("pre", "regular")))
    if ph == "pre":
        st.caption(f"盤前報價（Yahoo Finance，可能延遲數分鐘）｜正式開盤 台北 {clk['open_tpe']}。名單依 {B['asof']} 收盤，開盤後再看「即時精選」。")
    elif ph == "regular":
        st.caption("美股已開盤：盤前數字為今天開盤前最後一筆；名單請看「即時精選」。")
    else:
        st.caption(f"目前不是盤前時段（盤前 台北 {clk['pre_tpe']}～{clk['open_tpe']}）；顯示最近收盤與盤後報價。")
    st.markdown(ui.panel("盤前／盤後異動（候選股）", UV.movers(B, q)), unsafe_allow_html=True)
    for h in UV.H_SHOW:
        us_list_section(B, h, q)
    us_watch_section(B, q)
    st.caption("盤前提示：大型股很少盤前跳空 4% 以上，歷史樣本太少、沒有一致結論（v3.5 G1）——大漲別追價、大跌先看消息。")
    us_browse_section(B, q, "pre")


def us_watch_section(B, q, live=None):
    if "watch" not in B["picks"]:
        return
    with st.expander("🚀 動能觀察名單（2022 年後大幅領先、但 2004–2021 長期輸 SPY——不是參考名單）", expanded=False):
        us_list_section(B, "watch", q, live)
        st.caption("風險調整動能＝一年漲幅（扣最近一個月）÷ 波動，標普動能指數（SPMO）的選股方法。"
                   "回測：2022–2024 每年多 SPY 約 11 個百分點、2025 後約 20 個；但 2004–2021 每年少約 6 個、最大回撤 −62%（SPY −55%）。"
                   "動能有效的年代可以很強，失效時（2009、2016 年）跌得比大盤兇。")


def us_browse_section(B, q, key: str):
    st.markdown(ui.section("其他母池排行", "那斯達克 100、費半等目前成分股，用同一套模型排名（瀏覽用，不是驗證過的名單）"),
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1.6])
    sk = c1.segmented_control("範圍", list(UV.SET_ZH), default="ndx", format_func=UV.SET_ZH.get, key=f"us_set_{key}") or "ndx"
    mo = c2.segmented_control("排名方式", ["BAL_REV", "SPMO", "MINVOL", "QTREND"], default="BAL_REV",
                              format_func={"BAL_REV": "平衡動能＋反轉", "SPMO": "風險調整動能", "MINVOL": "最小波動",
                                           "QTREND": "品質趨勢"}.get, key=f"us_mod_{key}") or "BAL_REV"
    st.markdown(UV.set_table(B, sk, mo, q), unsafe_allow_html=True)


def us_live_tab(B, clk):
    ph = clk["phase"]
    auto = st.toggle("盤中每 90 秒自動更新", value=ph == "regular", key="us_auto")

    @st.fragment(run_every=90 if auto and ph == "regular" else None)
    def board():
        cands = UE.candidates(B, 30)
        q, bars = us_quotes(tuple(cands), us_slot(ph in ("pre", "regular")))
        live = None
        if ph == "regular" and len(q):
            prices = {t: float(r["price"]) for t, r in q.iterrows() if ui.ok(r.get("price"))}
            live = UE.live_picks(B, prices)
            st.caption(f"即時重算：{len(prices)} 檔有報價（{pd.Timestamp.now(tz='Asia/Taipei'):%H:%M}）。"
                       "短線（最小波動）與動能觀察名單看的是較長期的資料，盤中漲跌影響不大；中長線會隨今天漲跌小幅變動。")
        else:
            st.info(f"美股盤中（台北 {clk['open_tpe']}～{clk['close_tpe']}）才會即時重算；現在顯示 {B['asof']} 收盤名單與最新報價。")
        if len(q):
            up = q[q["chg"].notna()].sort_values("chg")
            if len(up) >= 4 and ph in ("regular", "post"):
                hot = "".join(UV.chip(B, t, f"{r['chg'] * 100:+.1f}%") for t, r in up.tail(5).iloc[::-1].iterrows())
                cold = "".join(UV.chip(B, t, f"{r['chg'] * 100:+.1f}%") for t, r in up.head(5).iterrows())
                st.markdown(ui.panel("盤中強弱（候選股）", f"<div class='kq-muted'>最強</div><div class='kq-chips'>{hot}</div>"
                                     f"<div class='kq-muted' style='margin-top:6px'>最弱</div><div class='kq-chips'>{cold}</div>"),
                            unsafe_allow_html=True)
        for h in UV.H_SHOW:
            us_list_section(B, h, q, live)
            if live and h in live and (live[h]["new"] or live[h]["out"]):
                st.caption(f"與收盤名單相比：新進 {'、'.join(live[h]['new']) or '無'}；掉出 {'、'.join(live[h]['out']) or '無'}"
                           "（收盤前 30 分鐘的即時名單＝研究中的「當天收盤買進」版本）。")
        us_watch_section(B, q, live)
    board()
    us_browse_section(B, us_quotes(tuple(UE.candidates(B, 30)), us_slot(ph in ("pre", "regular")))[0], "live")


def us_stock_tab(B, clk):
    opts = us_name_options(B)
    q0 = (st.query_params.get("u", "") or "").upper()
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    sel = c1.selectbox("美股代號或公司名（可直接打字搜尋，例如 NVDA、輝達、Apple）", opts,
                       index=opts.index(q0) if q0 in opts else None, format_func=lambda t: us_fmt(B, t), key="us_sel",
                       placeholder="例：NVDA 或 輝達")
    other = c2.text_input("其他代號（不在清單內）", value="" if (not q0 or q0 in opts) else q0, key="us_other",
                          placeholder="例：BRK-B、SPY")
    t = (other.strip().upper() or sel or "").replace(".", "-")
    if not t:
        st.markdown(ui.panel("使用方式", ui.bullets([
            "輸入美股代號（AAPL）或中文名（蘋果），顯示即時報價、盤前盤後、分時走勢、日 K 與技術分析。",
            "在模型母池內的股票會顯示 6 個候選模型的排名與各因子位置；和台股有關的，會列出台灣供應鏈。",
            "報價來源 Yahoo Finance（美股多為即時或延遲數分鐘）。"])), unsafe_allow_html=True)
        return
    if q0 != t:
        st.query_params["u"] = t
    q, bars = us_quotes((t,), us_slot(us_live_now(clk)))
    snap = q.loc[t] if len(q) and t in q.index else None
    nm = UV.label(B, t)
    desc = UN.zh_desc(t)
    info = B["info"].loc[t] if t in B["info"].index else None
    sub = " ｜ ".join(x for x in [desc, (info["sector"] + ("／" + info["sub"] if info["sub"] else "")) if info is not None else "",
                                   info["name_en"] if info is not None and info["name_en"] else ""] if x)
    if snap is not None:
        ph = UD.PHASE_ZH.get(snap["phase"], "")
        ext = (f"　<span class='kq-muted'>{snap['ext_label']}</span> <b>{UV.money(snap['ext'])}</b> {ui.chg(snap['ext_chg'])}"
               if ui.ok(snap.get("ext")) else "")
        tm = snap["time"].strftime("%m/%d %H:%M") if pd.notna(snap.get("time")) else ""
        st.markdown(f"<div class='kq-panel'><div style='font-size:20px'><b>{ui.esc(t)}</b> {ui.esc(nm)}　"
                    f"<b style='font-size:24px'>{UV.money(snap['price'])}</b> {ui.chg(snap['chg'])}{ext}</div>"
                    f"<div class='kq-muted'>{ui.esc(sub)}｜{ph}｜台北 {tm}｜美元</div></div>", unsafe_allow_html=True)
    else:
        st.warning(f"抓不到 {t} 的即時報價（代號錯誤或 Yahoo 暫時無法連線）。")
    # 分時
    pre_d = us_pre_dict(bars.get(t), snap["base"], clk) if snap is not None and snap["phase"] == "pre" else None
    reg = intraday_raw((t,), us_slot(us_live_now(clk)), "2m").get(t)
    a, b = st.columns(2) if (pre_d is not None and reg is not None and not IS_MOBILE) else (st.container(), st.container())
    if pre_d is not None:
        a.markdown(ui.section("盤前分時", "平盤＝昨收"), unsafe_allow_html=True)
        with a:
            intraday_chart(pre_d, key=f"us_pre_{t}", height=260, volume=False)
    if reg is not None and len(reg["v"]) >= 2:
        b.markdown(ui.section("最近一個交易時段分時", f"{reg['date']}（美東）"), unsafe_allow_html=True)
        b.markdown(intraday_stats(reg), unsafe_allow_html=True)
        with b:
            intraday_chart(reg, key=f"us_reg_{t}", height=300)
    # 日 K（Yahoo 抓不到時改用資料包內最近 300 日）
    df = us_daily(t, us_slot(False)[:10])
    if (df is None or df.empty) and t in B["C"].columns:
        df = us_bundle_daily(B, t)
        st.caption("日線暫時抓不到，改用美股資料包內的資料（最近約 300 個交易日）。")
    if df is None or df.empty:
        st.info("抓不到日線資料。")
        return
    ind = UV.indicators(df)
    spy = us_daily("SPY", us_slot(False)[:10])
    if (spy is None or spy.empty) and "SPY" in B["C"].columns:
        spy = us_bundle_daily(B, "SPY")
    g1, g2 = st.columns([1.15, 1]) if not IS_MOBILE else (st.container(), st.container())
    g1.markdown(ui.panel("技術面摘要", ui.kv(UV.analysis_rows(ind, spy["adj"] if len(spy) else None))), unsafe_allow_html=True)
    g2.markdown(ui.panel("模型評分（今日母池）", UV.model_rows(B, t)), unsafe_allow_html=True)
    tw = UV.tw_link_html(t, {c: NAMES[j] for c, j in TIDX.items()})
    if tw:
        st.markdown(ui.panel("和台股的關係（市場常見關聯）", tw), unsafe_allow_html=True)
    rng_opts = {"3個月": 63, "6個月": 126, "1年": 252, "2年": 504}
    r1, r2 = st.columns([1, 2], vertical_alignment="bottom")
    rng = r1.selectbox("區間", list(rng_opts), index=1, key="us_rng")
    pn = r2.multiselect("副圖", ["volume", "kd", "macd"], default=["volume"] if IS_MOBILE else ["volume", "macd"],
                        format_func={"volume": "成交量", "kd": "KD", "macd": "MACD"}.get, key="us_pn")
    from kosto.charts import kline
    fig = kline(ind, dark=DARK, mas=(20, 50, 200), boll=True, panels=pn, bars=rng_opts[rng],
                height=(300 + 95 * len(pn)) if IS_MOBILE else None, vol_label="成交量(股)")
    if IS_MOBILE:
        fig.update_layout(dragmode=False)
    st.plotly_chart(fig, width="stretch", theme=None, key=f"us_k_{t}",
                    config={"displaylogo": False, "scrollZoom": not IS_MOBILE, "displayModeBar": not IS_MOBILE})
    st.caption("均線：20 日（藍）、50 日（綠）、200 日（紫）；價格為還原前的實際成交價。"
               f"[Yahoo Finance](https://finance.yahoo.com/quote/{t})｜"
               f"[Google 新聞](https://news.google.com/search?q={t}%20stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant)｜"
               f"[SEC 財報](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker={t})")
    news = news_headlines(f"{t} {nm if nm != t else ''} 美股".strip(), n=6, days=3)
    st.markdown(ui.panel(f"最新新聞｜{t}", headlines_html(news)), unsafe_allow_html=True)


def us_market_tab(B, clk):
    iq, _ = us_quotes(tuple(UN.INDEXES), us_slot(us_live_now(clk)))
    tiles = []
    for s, lab in UN.INDEXES.items():
        if len(iq) and s in iq.index:
            r = iq.loc[s]
            tiles.append(ui.tile(lab, ui.num(r["price"], 2), r.get("chg"), UD.PHASE_ZH.get(r.get("phase"), "")))
    if tiles:
        st.markdown(f"<div class='kq-tiles'>{''.join(tiles)}</div>", unsafe_allow_html=True)
    else:
        st.caption("指數報價暫時抓不到（Yahoo Finance 無法連線）。")
    br = UV.breadth(B)
    if br:
        st.markdown(ui.cards([("母池檔數", f"{br['n']}", f"資料至 {B['asof']}"),
                              ("站上 50 日線", f"{br['above50']:.0%}", "短期多空"),
                              ("站上 200 日線", f"{br['above200']:.0%}", "長期多空"),
                              ("接近一年新高（5% 內）", f"{br['near_high']:.0%}", "強勢股比例"),
                              ("最近一日上漲家數", f"{br['up1']:.0%}", "")]), unsafe_allow_html=True)
    per = st.segmented_control("熱力圖期間", ["r1", "r5", "r20", "r60"], default="r1",
                               format_func={"r1": "最近一日", "r5": "5 日", "r20": "20 日", "r60": "60 日"}.get, key="us_heat") or "r1"
    live = None
    if per == "r1" and clk["phase"] == "regular":
        cq, _ = us_quotes(tuple(UE.candidates(B, 30)), us_slot(True))
        if len(cq):
            live = cq["chg"]
    st.plotly_chart(UV.treemap(B, per, DARK, 460 if IS_MOBILE else 600, live), width="stretch", theme=None, key="us_tree",
                    config={"displaylogo": False, "displayModeBar": False})
    st.caption("面積＝近 60 日平均成交金額（代替市值），顏色＝漲跌（紅漲綠跌）。盤中「最近一日」只有候選股換成即時漲跌，其他為前一日收盤。")
    st.markdown(ui.section("類股 ETF", "標普 11 大類股"), unsafe_allow_html=True)
    st.markdown(UV.sector_table(B), unsafe_allow_html=True)


def us_etf_tab(B, clk):
    """v3.6 美股 ETF 推薦順序（比照台股「精選 ETF」）：配置範例 → 推薦順序 → 各角色排行。"""
    R = UE.research()
    st.markdown(ui.section("美股 ETF 推薦順序", "先依角色排（核心 → 成長 → 避險 → 衛星），角色內依 2004 年起的風險調整報酬（Sharpe）；"
                           "長期數字含息，目前趨勢用最新收盤"), unsafe_allow_html=True)
    st.markdown("<div class='kq-panel' style='border-left:4px solid var(--accent)'><b>為什麼美股先看 ETF？</b>"
                "<div class='kq-muted' style='margin-top:4px'>三輪研究、100 多種選股組合，2004–2021 年都沒有長期贏過 S&P 500（SPY）；"
                "反而是「大盤 ETF＋一點債券或黃金」的組合，報酬接近、跌幅少很多。台灣投資人的台股部位已經重壓半導體，"
                "美股用大盤 ETF 分散最有效（S&P 500 與 0050 的週報酬相關只有 0.50）。</div></div>", unsafe_allow_html=True)
    prof = st.segmented_control("你的風險屬性", list(UN.MIX), default="穩健型", key="us_etf_prof") or "穩健型"
    st.markdown(UV.mix_html(R, prof), unsafe_allow_html=True)
    st.markdown(ui.section("推薦順序：第一次買美股，照這個順序", "點代號可看即時報價與 K 線"), unsafe_allow_html=True)
    st.markdown(UV.etf_starter_html(R, B), unsafe_allow_html=True)
    tabs = st.tabs([f"{['🏛️', '🚀', '🛡️', '🛰️'][k]} {UN.ROLE_ZH[r].split('（')[0]}" for k, r in enumerate(UN.ROLE_ORDER)] + ["⚖️ 配置比較"])
    for k, r in enumerate(UN.ROLE_ORDER):
        with tabs[k]:
            st.markdown(f"<div class='kq-muted' style='margin-bottom:6px'><b>{ui.esc(UN.ROLE_ZH[r])}</b>：{ui.esc(UN.ROLE_WHY[r])}</div>",
                        unsafe_allow_html=True)
            st.markdown(UV.etf_role_table(R, B, r), unsafe_allow_html=True)
    with tabs[-1]:
        st.markdown(UV.mix_table(R), unsafe_allow_html=True)
        st.caption("加債券、黃金的組合年化少 1～3 個百分點，但最大回撤從 −55% 降到 −19～−36%、Sharpe 反而最高；"
                   "全押 QQQ 報酬最高，但 2008 年跌 42%、2022 年跌 33%。配置每年調回一次即可，不必擇時。")
    st.caption(f"長期統計 {R.get('etf_asof', '')} 為止（Johnbrick123/sp500-data，含息）；費用率為 2025 年公開資料（約略，以發行商公告為準）；"
               "與 0050 相關＝2019 年起週報酬（美元計價，未換匯）。台股掛牌的同類 ETF 用台幣買、不用複委託，但費用較高、有追蹤誤差。"
               "以上為研究整理，不構成投資建議。")


def us_doc_tab(B, clk):
    R = UE.research()
    st.markdown(ui.section("第二輪研究：標普 500 時點母池（含下市公司）× 9 個模型，對 SPY 的年化超額",
                           "docs/v3.5_protocol.md｜⭐＝參考名單｜開發期 2004–2021"), unsafe_allow_html=True)
    hz = st.segmented_control("期間", ["short", "mid", "long"], default="long",
                              format_func={"short": "短線（5 日）", "mid": "中線（20 日）", "long": "長線（60 日）"}.get,
                              key="us_doc_h") or "long"
    st.markdown(UV.validation_html(R, hz), unsafe_allow_html=True)
    st.caption("驗收規則（先寫好再跑，和台股一樣以「直接買大盤 ETF」為基準）：開發期與驗證期都要每年多 SPY 3 個百分點以上、資訊比率 ≥ 0.4、"
               "回撤不比 SPY 深 10 個百分點、半年段落勝率 ≥ 50%、成本加倍仍為正。每次 10 檔、名次加權、看到名單後隔日開盤買、單邊成本 0.1%。"
               "108 組沒有一組通過；同母池等權平均對 SPY 開發期 +0.4 個百分點（確認回測正確）。")
    st.markdown(ui.section("第三輪：針對弱點再試 4 種（v3.6）", "docs/v3.6_protocol.md｜大盤濾網、熊市濾網、巨型股趨勢"), unsafe_allow_html=True)
    st.markdown(UV.round3_html(R), unsafe_allow_html=True)
    st.caption("仍然沒有通過。最接近的是「巨型股趨勢」（標普前 20 大中走多頭的才持有，其餘抱 SPY）：2014–2021 每年多約 5 個百分點、"
               "2022–24 多 3～8 個，但 2004–2013 每年少約 2 個、2025 年後也落後，長期只比 SPY 多 0.5～0.7。大盤濾網減少了 2009 年的動能崩盤，"
               "但 2014–2021 動能本身就輸 SPY。")
    st.markdown(ui.section("ETF 策略：想簡單操作的人"), unsafe_allow_html=True)
    st.markdown(UV.etf_html(R), unsafe_allow_html=True)
    st.caption("沒有 ETF 策略通過。200 日線趨勢策略把 2008 年的最大回撤從 −55% 降到 −17%，但 2022 年後常常來回被巴；"
               "一直抱 QQQ 在 2004–2021 每年多 SPY 4.6 個百分點，但那是科技股的黃金年代，2022 年後只多 0.6。")
    with st.expander("動能為什麼不穩？持股檔數的影響"):
        rows = [[x["model"], f"{x['k']} 檔", ui.pct(x["dev"], color=True), ui.pct(x["dev2"], color=True), ui.pct(x["val"], color=True)]
                for x in R.get("concentration", [])]
        st.markdown(ui.table(["模型（長線、標普 500）", "持股", "開發期 2004–21", "其中 2014–21", "驗證期 2022–24"], rows), unsafe_allow_html=True)
        st.caption("持股越分散，落後越少，但仍然輸 SPY；動能的超額集中在 2022 年後的 AI 行情。")
    with st.expander("盤前跳空研究（G1）"):
        st.markdown(UV.gap_html(R), unsafe_allow_html=True)
        st.caption("大型股很少盤前跳空 4% 以上，樣本太少，沒有一致結論 → 只提醒，不給方向。")
    with st.expander("第一輪研究（v3.4，舊資料）與因子 IC"):
        st.markdown(UV.round1_html(R), unsafe_allow_html=True)
        st.markdown(UV.ic_html(R.get("round1") or {}), unsafe_allow_html=True)
        st.caption("第一輪資料只有現在還在掛牌的股票（倖存者偏差）、缺 NYSE 大型股，而且是和「等權平均」比；第二輪資料完整、改和 SPY 比，結論更可靠。")
    st.markdown(ui.section("美股經驗 → 台股"), unsafe_allow_html=True)
    st.markdown(ui.bullets([
        "<b>風險調整動能</b>（一年漲幅扣最近一個月 ÷ 波動，標普動能指數的方法）：台股開發期（2021–2025）中線每年多 4.7 個百分點、短線 +4.1、"
        "長線 +2.0，過了門檻；但最近 12 個月 25 種組合<b>全部大幅落後</b>（中線 −27.5），2021–2026 整段平均反而 −0.6 → <b>不採用</b>，台股模型維持 v3.4。",
        "<b>避開暴漲暴跌的投機股</b>：美股最一致的訊號，在台股卻相反（中線 −16.5 個百分點）——台股的強勢飆股之後常續強，不採用。",
        "<b>縮小母池</b>：台股研究每天都在 5 種母池規則 × 6 種大小裡挑，本來就是「精選母池」；美股縮到前 100／200 大比全部好一點，但仍輸 SPY。",
        "<b>測試程式更正</b>：附加新因子的測試把「缺值」（上市未滿一年的新股）排在最前面；更正後重跑 v3.1 殘差動能，結論不變（仍未通過）。"]),
        unsafe_allow_html=True)
    st.markdown(ui.section("模型怎麼選股（白話）"), unsafe_allow_html=True)
    st.markdown(ui.bullets([
        f"<b>{ui.esc(UM.MODEL_ZH[m])}</b>：" + "、".join(f"{UM.FACTOR_ZH[k]} {w:.0%}" for k, w in UM.MODELS[m].items())
        for m in UM.MODELS]), unsafe_allow_html=True)
    st.markdown(ui.section("母池與名單"), unsafe_allow_html=True)
    st.markdown(ui.bullets([
        "參考名單：短線＝標普 500 成交金額前 100 大裡波動最低的 10 檔；中長線＝前 200 大裡「平衡動能＋短期反轉」前 10 檔（各期間開發期第一名）。"
        "股價 ≥ 5 美元、上市滿一年；已入選的股票排名掉到 25 名以外才換掉（中線 20 名）；名次加權（第 1 名 1.5 倍 → 第 10 名 0.5 倍）。",
        "那斯達克 100、費半：沒有可靠的歷史成分股資料，不能做無偏差的回測 → 只提供「目前成分股」排行瀏覽。",
        "每天美股收盤後（台北清晨）由 GitHub 自動更新；資料來源 Yahoo Finance。"]), unsafe_allow_html=True)
    tr = UE.track(B, "long")
    if len(tr):
        rows = [[r["date"], r["model"], ui.pct(r["ret"], color=True), ui.pct(r.get("spy"), color=True),
                 ui.pct(r["ret"] - r.get("spy", np.nan), color=True), f"{r['days']}"] for _, r in tr.iloc[::-1].iterrows()]
        st.markdown(ui.section("名單追蹤（中長線，每天的名單到今天）"), unsafe_allow_html=True)
        st.markdown(ui.table(["名單日", "模型", "名單報酬", "SPY", "差距", "經過交易日"], rows), unsafe_allow_html=True)
    st.markdown(ui.section("台灣投資人買美股：基本知識"), unsafe_allow_html=True)
    st.markdown(UV.basics_html(clk), unsafe_allow_html=True)


def page_us():
    header(world=False)
    clk = UD.market_clock()
    B, src, err = us_source()
    iq, _ = us_quotes(tuple(UN.INDEXES), us_slot(us_live_now(clk)))
    st.markdown(us_strip(clk, iq), unsafe_allow_html=True)
    if B is None:
        st.error(err or "還沒有美股資料。")
        st.markdown(ui.panel("怎麼開始", ui.bullets([
            "把更新包上傳到 GitHub，並新增 .github/workflows/us.yml（內容同 us_workflow_COPY.txt），Actions 會在美股收盤後自動產生美股資料包。",
            "也可以按下面的按鈕，讓 App 直接從 Yahoo Finance 建立（約 1～2 分鐘）。"])), unsafe_allow_html=True)
        if st.button("在 App 直接建立美股資料", key="us_build_btn"):
            st.session_state["us_build"] = True
            st.rerun()
        return
    exp = us_expected()
    st.markdown(ui.section("美股選股", f"名單資料至 {B['asof']}（美東收盤）｜{src}｜建立 {B.get('built', '')}"),
                unsafe_allow_html=True)
    if err:
        st.caption(f"⚠ {err}")
    if B["asof"] < exp:
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.warning(f"美股資料停在 {B['asof']}，最近一個收盤日應為 {exp}（美國假日時屬正常）。")
        if c2.button("在 App 直接更新", key="us_rebuild"):
            st.session_state["us_build"] = True
            st.rerun()
    st.markdown(us_validation_banner(UE.research()), unsafe_allow_html=True)
    names = {"etf": "💰 ETF 推薦（先看這裡）", "pre": "🌅 盤前精選", "live": "⚡ 即時精選", "stock": "🔍 個股查詢", "mkt": "🗺️ 美股大盤",
             "doc": "📘 驗證與說明"}
    order = ["etf"] + (["live", "pre"] if clk["phase"] == "regular" else ["pre", "live"]) + ["stock", "mkt", "doc"]
    if st.query_params.get("u"):
        order = ["stock"] + [k for k in order if k != "stock"]
    tabs = dict(zip(order, st.tabs([names[k] for k in order])))
    with tabs["etf"]:
        us_etf_tab(B, clk)
    with tabs["pre"]:
        us_pre_tab(B, clk)
    with tabs["live"]:
        us_live_tab(B, clk)
    with tabs["stock"]:
        us_stock_tab(B, clk)
    with tabs["mkt"]:
        us_market_tab(B, clk)
    with tabs["doc"]:
        us_doc_tab(B, clk)
    st.caption("美股報價：Yahoo Finance（多為即時或延遲數分鐘）。名單為研究參考，不構成投資建議。")


# ================================================================== v3.6 總覽（第一次使用的人從這裡開始）
def us_bundle_quick():
    """總覽頁用：只讀現成的美股資料包（GitHub／本機），不在這裡花 1～2 分鐘自己建。"""
    try:
        if REPO:
            stamp = us_stamp(REPO, TOKEN)
            if stamp:
                return us_load_github(REPO, TOKEN, stamp)
        if US_LOCAL.is_file():
            return us_load_local(US_LOCAL.stat().st_mtime)
    except Exception:  # noqa: BLE001
        return None
    return None


def hcard(n: int, title: str, body: str, go: str = "") -> str:
    return (f"<div class='kq-hcard'><div class='t'><span class='n'>{n}</span>{ui.esc(title)}</div><div class='b'>{body}</div>"
            + (f"<div class='go'>{go}</div>" if go else "") + "</div>")


def page_home():
    from kosto import market as mklib
    header()
    st.markdown(ui.section("總覽：第一次使用，照 ①→④ 的順序看", "這個 App 在做什麼、今天該怎麼做；每張卡片都可以點進去看細節"),
                unsafe_allow_html=True)
    prof = st.segmented_control("你的風險屬性（決定 ETF 和個股的比例）", list(mklib.PROFILES), default="穩健型", key="home_prof") or "穩健型"
    _, _, sto = insight_frame(BID)
    al = mklib.allocation(prof, sto or {})
    # ① 市場
    tone = {"green": "#16a35f", "amber": "#e9a23b", "red": "#f0464b"}.get((sto or {}).get("tone"), "#94a3b8")
    lvl, rtxt, risks = risk_light()
    b1 = (f"<b style='color:{tone};font-size:15px'>台股：{ui.esc((sto or {}).get('view', '資料不足'))}</b>"
          + (f"<br><span class='kq-muted'>加權距半年高點 {ui.pct(sto.get('dd'), 1)}｜止穩訊號 {sto.get('score', '–')}/8｜"
             f"市場階段 {ui.esc(STAGE_ZH.get(latest('mid').get('market_stage'), '').split(' ')[0] or '–')}</span>" if sto else "")
          + f"<br>{ui.light(lvl, '美股：' + rtxt)}" + (f"<span class='kq-muted'>（{ui.esc('、'.join(risks))}）</span>" if risks else ""))
    try:                                                              # v3.13 環境判定卡（一行摘要）
        ec = env_cards(BID)
        if ec:
            b1 += ("<br><span class='kq-muted'>環境判定卡（短線／長線）：" + "；".join(
                f"{ui.esc(nm)} {c['short_zh']}／{c['long_zh']}" for nm, c in ec) + "</span>")
    except Exception:  # noqa: BLE001
        pass
    b1 += morning_home_line()                                          # v3.16 盤前日報一行摘要
    c1 = hcard(1, "今天市場：能不能進場？", b1, "<a href='./insight' target='_self'>→ 盤勢儀表板（盤前日報、燈號、急跌應變、科斯托蘭尼階段）</a>")
    # ② 資金
    mx = (UE.research().get("mix") or {}).get(prof, {})
    us_mix = "、".join(f"{UV.etf_label(t)} {x:.0%}" for t, x in (mx.get("w") or {}).items())
    b2 = (f"<b>台股</b>：ETF（0050 等大盤型）<b>{al['etf']:.0%}</b>＋模型個股 <b>{al['stock']:.0%}</b>"
          + (f"<br><span class='kq-muted'>{ui.esc(al['why'])}</span>" if al.get("tilt") else "")
          + f"<br><b>美股</b>：{ui.esc(us_mix)}<br><span class='kq-muted'>美股個股模型沒有通過驗證 → 以 ETF 為主；"
          "台股模型驗證過，可以配個股。</span>")
    c2 = hcard(2, f"資金怎麼分（{prof}）", b2,
               "<a href='./alloc' target='_self'>→ 資金配置</a>　<a href='./etf' target='_self'>→ 台股精選 ETF</a>")
    # ③ 台股
    h = "mid"
    lt = latest(h)
    its = pick_items(h, lt)[:8]
    chips = "".join(ui.schip(it["t"], NAMES[it["j"]], f"{it['w'] * 100:.0f}%") for it in its)
    ph = phase_stats(h)
    val = (f"<span class='kq-muted'>驗證：2021 起回測年化 {ui.pct(ph['樣本外年化_平均'], 1)}（0050 {ui.pct(ph['0050樣本外'], 1)}）、"
           f"近一年 {ui.pct(ph['近一年_平均'], 1)}（0050 {ui.pct(ph['0050近一年'], 1)}）</span>" if ph else
           "<span class='kq-muted'>驗證成績見「今日選股」上方。</span>")
    b3 = (f"中線名單（{lt.get('date', '')} 收盤）依名次給資金：<div class='kq-chips' style='margin:6px 0'>{chips}</div>{val}"
          "<br><span class='kq-muted'>下一個交易日開盤照「買進／賣出／續抱」執行；短線、長線名單在今日選股切換。</span>")
    c3 = hcard(3, "台股買什麼：模型個股（衛星）", b3, "<a href='./picks' target='_self'>→ 今日選股（明天要做的事）</a>　"
                                                    "<a href='./live' target='_self'>→ 盤中監控</a>")
    # ④ 美股
    R = UE.research()
    rk = UV.etf_rank(R)
    BU = us_bundle_quick()
    order = [(1, rk["core"][0], "核心"), (2, rk["growth"][0], "成長"), (3, rk["hedge"][0], "避險"), (3, rk["hedge"][1], "避險"),
             (4, rk["sat"][0], "衛星")] if rk.get("core") else []
    lines = []
    for no, t, role in order:
        i = UN.ETF_INFO[t]
        tr = UV.etf_trend(BU, t) if BU else {}
        lines.append(f"<div class='kq-kv'><span>{'①②③④'[no - 1]} <a href='./us?u={t}' target='_self'><b>{ui.esc(UV.etf_label(t))}</b></a> "
                     f"{ui.esc(i['zh'])}<small class='kq-muted'>（{role}）</small></span><span class='kq-mono'>"
                     + (f"{'站上' if tr['above200'] else '跌破'} 200 日線" if tr else f"費用 {i['er']:.2f}%") + "</span></div>")
    b4 = "".join(lines) + ("<span class='kq-muted'>個股名單（盤前／即時精選）只供參考：三輪研究都沒有長期贏 SPY。</span>")
    c4 = hcard(4, "美股買什麼：ETF 推薦順序", b4, "<a href='./us' target='_self'>→ 美股選股（ETF 推薦、盤前／即時、個股查詢）</a>")
    st.markdown(f"<div class='kq-home'>{c1}{c2}{c3}{c4}</div>", unsafe_allow_html=True)
    st.markdown(ui.section("投資建議的脈絡（為什麼是這個順序）"), unsafe_allow_html=True)
    st.markdown(ui.bullets([
        "<b>先看市場、再決定買多少</b>：修正期且止穩訊號少時，ETF 比例自動提高 20 個百分點（回測 2021–2026 年化多約 1.5～2 個百分點、回撤略小）。",
        "<b>ETF 是核心</b>：不用選股、長期持有；台股用 0050 等大盤型，美股用 VOO／IVV。",
        "<b>台股個股是衛星</b>：模型 2021 年起的回測與最近一年都贏 0050，但個股波動大、要照名單換股；份量依風險屬性。"
        "三份獨立評審（v3.14～v3.15）後的誠實結論：超額主要來自動能與小型股傾向、扣掉後仍有優勢，但考慮做過的試驗次數還不能算統計上確定；"
        "怕回撤可用 0050 七成＋模型三成（歷史回撤比只買 0050 淺），"
        "真正的檢驗是之後的前瞻紀錄（<a href='./validate' target='_self'>→ 模型驗證</a>）。",
        "<b>美股個股只當參考</b>：用沒有偏差的 18 年資料測了三輪，都沒有長期贏 SPY，所以美股以 ETF 為主。",
        "<b>何時賣</b>：照「今日選股 → 持股追蹤」的賣出訊號；跌破警戒價時重新檢視理由（不自動停損，回測顯示機械式停損會拖累報酬）。"]),
        unsafe_allow_html=True)
    st.markdown(ui.section("每一頁在做什麼"), unsafe_allow_html=True)
    rows = [["📋 今日選股", "台股明天開盤要買／賣／續抱什麼（短、中、長線）", "每個交易日晚上（17:23 初版、21:40 完整版）"],
            ["⏱️ 盤中監控", "台股即時價、能不能進場、價位提醒", "台股 9:00–13:30"],
            ["🎛️ 盤勢儀表板", "大盤燈號、急跌應變、止穩檢查、科斯托蘭尼階段、模型體檢", "開盤前、大盤急跌時"],
            ["🗺️ 產業鏈地圖", "電子產業上下游、題材、消息衝擊評估", "想了解產業或消息影響時"],
            ["🔍 個股診斷", "任一台股的模型評分、K 線、買進區與警戒價", "想查某一檔時"],
            ["🗽 美股選股", "美股 ETF 推薦順序、盤前／即時名單、美股個股查詢", "晚上（美股盤前 17:00 起）"],
            ["更多 → 精選ETF／資金配置", "台股 ETF 排行、核心＋衛星的比例", "決定資金分配時"],
            ["更多 → 模型驗證", "這份名單是哪一版、證據等級、三種基準、成本壓力、統計檢定（v3.14）", "想知道模型可不可信時"],
            ["更多 → 操作指南／研究室", "使用說明、研究方法與完整驗證", "想深入了解時"]]
    st.markdown(ui.table(["頁面", "用途", "什麼時候看"], rows, left=(0, 1, 2)), unsafe_allow_html=True)
    st.caption("僅供研究參考，非投資建議；系統不會自動下單。")


pages = {"": [st.Page(page_home, title="總覽", icon="🏠", url_path="home", default=True),
              st.Page(page_picks, title="今日選股", icon="📋", url_path="picks"),
              st.Page(page_live, title="盤中監控", icon="⏱️", url_path="live"),
              st.Page(page_insight, title="盤勢儀表板", icon="🎛️", url_path="insight"),
              st.Page(page_chain, title="產業鏈地圖", icon="🗺️", url_path="chain"),
              st.Page(page_stock, title="個股診斷", icon="🔍", url_path="stock"),
              st.Page(page_us, title="美股選股", icon="🗽", url_path="us")],
         "更多": [st.Page(page_market, title="大盤", icon="📈", url_path="market"),
                  st.Page(page_world, title="國際盤", icon="🌏", url_path="world"),
                  st.Page(page_etf, title="精選ETF", icon="💰", url_path="etf"),
                  st.Page(page_alloc, title="資金配置", icon="⚖️", url_path="alloc"),
                  st.Page(page_guide, title="操作指南", icon="📖", url_path="guide"),
                  st.Page(page_validate, title="模型驗證", icon="🧪", url_path="validate"),
                  st.Page(page_lab, title="研究室", icon="🔬", url_path="lab")]}
_nav = st.navigation(pages, position="top")
CUR_PAGE = getattr(_nav, "url_path", "") or "home"
_nav.run()
