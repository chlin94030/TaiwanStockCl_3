"""v2.5 消息面：急漲急跌事件監控、即時消息解析、更新狀態診斷、輝達事件研究常數。"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kosto import freshness as F
from kosto import news
from kosto import review as rv


# ------------------------------------------------------------------ 即時消息解析（不連網）
RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>
<item><title>輝達執行長請辭 台股供應鏈開低 - 經濟日報</title><link>https://example.com/a</link>
<pubDate>Thu, 12 Nov 2026 23:10:00 GMT</pubDate><source url="https://money.udn.com">經濟日報</source></item>
<item><title>台股收盤跌 1,165 點 &amp; 量增 - 鉅亨網</title><link>https://example.com/b</link>
<pubDate>Thu, 12 Nov 2026 06:00:00 GMT</pubDate><source url="https://news.cnyes.com">鉅亨網</source></item>
<item><title>沒有來源與時間</title><link>https://example.com/c</link></item>
</channel></rss>"""


def test_parse_rss_strips_source_and_sorts_newest_first():
    xs = news.parse_rss(RSS, n=5)
    assert [x["link"] for x in xs][:2] == ["https://example.com/a", "https://example.com/b"]
    assert xs[0]["title"] == "輝達執行長請辭 台股供應鏈開低" and xs[0]["source"] == "經濟日報"
    assert xs[1]["title"] == "台股收盤跌 1,165 點 & 量增"
    assert xs[0]["time"].tz is not None and xs[0]["time"].strftime("%m/%d %H:%M") == "11/13 07:10"   # 台北時間
    assert xs[2]["time"] is None
    assert news.parse_rss("<not xml", 5) == [] and len(news.parse_rss(RSS, n=1)) == 1


def test_parse_material_roc_dates_and_spaced_keys():
    rows = [{"出表日期 ": "1151112", "發言日期": "1151112", "發言時間": "143005", "公司代號": "2330 ", "公司名稱": "台積電",
             "主旨 ": "代子公司公告取得設備", "說明": "1.事實發生日:115/11/12"},
            {"公司代號": "", "主旨": "沒有代號 → 略過"}, "不是 dict"]
    df = news.parse_material(rows)
    assert len(df) == 1 and df.loc[0, "ticker"] == "2330" and df.loc[0, "date"] == "2026-11-12" and df.loc[0, "time"] == "14:30"
    assert df.loc[0, "title"] == "代子公司公告取得設備"
    assert list(news.parse_material([]).columns) == ["date", "time", "ticker", "name", "title", "detail"]
    en = news.parse_material([{"Date": "1151112", "SecuritiesCompanyCode": "6488", "CompanyName": "環球晶", "Subject": "公告董事會決議",
                               "SpeechTime": "1615"}])
    assert (en.loc[0, "ticker"], en.loc[0, "date"], en.loc[0, "time"], en.loc[0, "title"]) == ("6488", "2026-11-12", "16:15", "公告董事會決議")
    assert news._roc_date("115/11/12") == "2026-11-12" and news._roc_date("20261112") == "2026-11-12"


def test_news_failures_return_empty(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("blocked")
    monkeypatch.setattr(news.requests, "get", boom)
    assert news.google_news("台股") == []
    assert news.material_today().empty
    assert all(len(s) == 4 and s[2].startswith("https://") for s in news.SOURCES)


# ------------------------------------------------------------------ 急漲急跌事件監控
def _bundle():
    """40 日、5 檔：0 急漲＋法人買＋營收高；1 急跌；2 漲 8% 但量不夠；3 急漲＋融資追＋法人賣；4 平穩。"""
    D, n = 40, 5
    days = pd.bdate_range("2026-09-01", periods=D)
    C = np.full((D, n), 100.0)
    V = np.full((D, n), 1000.0)
    FI, TR, MB = np.zeros((D, n)), np.zeros((D, n)), np.full((D, n), 5000.0)
    C[35:, 0] = 110.0; V[35, 0] = 5000; FI[35, 0] = 400; TR[35, 0] = 200          # +10%、5 倍量、法人買 12%
    C[37:, 1] = 91.0; V[37, 1] = 4000                                             # −9%、4 倍量
    C[38:, 2] = 108.0; V[38, 2] = 1500                                            # +8% 但量只有 1.5 倍
    C[39:, 3] = 109.0; V[39, 3] = 3500; FI[39, 3] = -300; MB[39:, 3] = 5300       # 融資 +300 張（8.6%）、法人賣 8.6%
    C[36:, 0] = 115.0                                                              # 事件後續漲
    RF = np.vstack([C[:1], C[:-1]])
    raw = np.full((n, 2), np.nan)
    raw[0, 1] = 0.35
    a = {"Q_C": C, "Q_V": V, "Q_RF": RF, "Q_FI": FI, "Q_TR": TR, "Q_MB": MB, "LATEST_RAW": raw}
    return SimpleNamespace(a=a, meta={"q_days": list(days)}, factors=["t_x", "f_rev_yoy3"])


def test_shock_events_detects_and_groups():
    df = rv.shock_events(_bundle(), days=10, top=800)
    got = {int(r.j): r for r in df.itertuples()}
    assert set(got) == {0, 1, 3}                                                  # 2 量不夠 → 不算
    e0 = got[0]
    assert e0.up and e0.ret == pytest.approx(0.10) and e0.volx == pytest.approx(5.0)
    assert e0.inst == pytest.approx(0.12) and e0.group == "法人買＋營收 > 20%"
    assert e0.since == pytest.approx(115 / 110 - 1) and e0.days_ago == 4
    assert e0.hist_x60 == pytest.approx(dict((x[0], x[5]) for x in rv.SHOCK_UP)["法人買＋營收 > 20%"])
    assert not got[1].up and got[1].ret == pytest.approx(-0.09)
    assert got[3].group == "融資追＋法人賣" and got[3].days_ago == 0
    assert list(df["date"]) == sorted(df["date"], reverse=True)                  # 新到舊
    assert rv.shock_events(SimpleNamespace(a={}, meta={}, factors=[])).empty


def test_shock_research_constants():
    for tbl in (rv.SHOCK_UP, rv.SHOCK_DOWN):
        assert all(len(x) == 8 and x[1] > 0 and 0 < x[6] < 1 for x in tbl)
        assert tbl[0][0] == "全部" and tbl[0][1] == max(x[1] for x in tbl)
    up = {x[0]: x for x in rv.SHOCK_UP}
    assert up["法人買（外資＋投信 > 當日量 5%）"][5] > up["全部"][5] > up["法人賣（< −5%）"][5]                        # 法人買延續、法人賣反轉
    assert up["法人買＋營收 > 20%"][5] == max(x[5] for x in rv.SHOCK_UP)
    assert rv.SHOCK_PEAK["day0"] > rv.SHOCK_PEAK["day1"] > 0
    assert len(rv.V25_TESTS) == 2 and all(t[-1].startswith("未通過") and len(t) == 8 for t in rv.V25_TESTS)
    for (thr, n), rows in rv.NV_CHAIN_SHOCK.items():
        assert n > 0 and [r[0] for r in rows] == ["輝達供應鏈", "其他電子", "非電子"] and all(len(r) == 8 for r in rows)
        assert rows[0][1] < 0 and rows[0][5] > 0                                  # 開低、60 日後反彈
    assert not any(x[2] == "使用者提問" and x[5].startswith("✅") for x in rv.IDEAS)


def test_nv_chain_membership():
    labels = {"2330": {"group": "電子上游", "segment": "晶圓代工"}, "3017": {"group": "電子中游", "segment": "散熱"},
              "6213": {"group": "電子中游", "segment": "銅箔基板（CCL）"}, "2603": {"group": "傳產", "segment": "貨櫃航運"}}
    s = rv.nv_chain(labels)
    assert {"2330", "3017", "6213"} <= s and "2603" not in s
    assert "2382" not in s                                                        # 不在 labels → 不列入


# ------------------------------------------------------------------ 更新狀態：說清楚落後幾天、前面失敗幾次
def _run(status, concl, created, event="schedule", updated=None):
    return {"status": status, "conclusion": concl, "created": created, "updated": updated or created, "event": event, "url": "u"}


def test_status_running_after_failures_shows_lag_and_streak():
    now = pd.Timestamp("2026-11-12 13:08", tz="Asia/Taipei")
    hist = [_run("in_progress", None, "2026-11-12T04:40:00Z", "push"),
            _run("completed", "failure", "2026-11-11T22:23:00Z"), _run("completed", "failure", "2026-11-11T13:40:00Z"),
            _run("completed", "success", "2026-11-10T13:40:00Z")]
    lvl, txt, url = F.status(now, pd.Timestamp("2026-11-10"), hist[0], hist)
    assert lvl == "red" and "上傳程式觸發，12:40 開始" in txt and "研究資料只到 11/10（應到 11/11）" in txt
    assert "連續 2 次更新失敗" in txt and url == "u"
    lvl, txt, _ = F.status(now, pd.Timestamp("2026-11-11"), hist[0], hist)        # 資料已是最新 → 只是執行中
    assert lvl == "amber" and "只到" not in txt
    assert F.fail_streak(hist) == 2 and F.fail_streak([]) == 0


def test_run_rows():
    rr = F.run_rows([_run("completed", "failure", "2026-11-11T13:40:00Z", updated="2026-11-11T14:25:00Z"),
                     _run("queued", None, "2026-11-12T04:40:00Z", "workflow_dispatch")])
    assert rr[0] == {"start": "11/11 21:40", "event": "排程", "result": "失敗", "minutes": 45, "url": "u"}
    assert rr[1]["result"] == "執行中" and rr[1]["minutes"] is None and rr[1]["event"].startswith("手動")


def test_docs_and_app_wired_v25():
    root = Path(__file__).resolve().parents[1]
    proto = (root / "docs" / "v2.5_protocol.md").read_text(encoding="utf-8")
    assert "## 結果" in proto and "主模型不調整" in proto
    app = (root / "app.py").read_text(encoding="utf-8")
    for s in ("def shock_section", "def nv_event_html", "def headlines_html", "def material_html", "def sources_html",
              "rv.V25_TESTS", "freshness.run_rows", "news_headlines(f\"{NAMES[j]} {t}\""):
        assert s in app
