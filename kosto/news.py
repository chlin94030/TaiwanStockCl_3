"""v2.5 即時消息：可靠來源連結、證交所每日重大訊息（OpenAPI）、Google 新聞 RSS 標題。

只用於 App 顯示，不參與選股。所有抓取都有逾時與失敗退回（抓不到就不顯示），不影響其他頁面。
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
TZ = "Asia/Taipei"

# 消息來源（官方優先；媒體只當線索，重大訊息以公開資訊觀測站為準）
SOURCES = [
    ("官方", "公開資訊觀測站｜即時重大訊息", "https://mops.twse.com.tw/mops/web/t05sr01_1", "上市櫃公司依法公告（打入供應鏈、合約、專利、人事、財測）"),
    ("官方", "證交所｜基本市況報導（盤中即時）", "https://mis.twse.com.tw/stock/index?lang=zhHant", "加權指數與個股 5 秒報價"),
    ("官方", "期交所｜台指期夜盤行情", "https://mis.taifex.com.tw/futures/", "美股交易時段的台股方向（15:00～隔日 05:00）"),
    ("官方", "證交所｜處置與注意股票", "https://www.twse.com.tw/zh/announcement/punish.html", "炒作過熱、分盤交易的股票"),
    ("媒體", "鉅亨網｜台股即時新聞", "https://news.cnyes.com/news/cat/tw_stock", "台股盤中快訊"),
    ("媒體", "經濟日報｜即時", "https://money.udn.com/money/breaknews/1001/0/1", "產業與個股新聞"),
    ("媒體", "工商時報｜證券", "https://www.ctee.com.tw/stock", "法人與產業動態"),
    ("媒體", "Yahoo 股市｜台股新聞", "https://tw.stock.yahoo.com/news", "整合多家媒體"),
    ("國際", "Reuters｜Technology", "https://www.reuters.com/technology/", "美股科技、半導體重大事件"),
    ("國際", "CNBC｜Markets", "https://www.cnbc.com/markets/", "美股盤前盤中快訊"),
    ("國際", "NVIDIA Newsroom", "https://nvidianews.nvidia.com/", "輝達官方公告（人事、財報日期）"),
    ("國際", "SEC EDGAR｜NVDA 申報（8-K）", "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=8-K", "美國上市公司重大事件申報"),
]

TWSE_MATERIAL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"      # 上市公司每日重大訊息
TPEX_MATERIAL = "https://www.tpex.org.tw/openapi/v1/mopsfe_t187ap04_O"     # 上櫃公司每日重大訊息


def _roc_date(x) -> str:
    """民國 1151112 / 115/11/12 → 2026-11-12；看不懂就原樣回傳。"""
    s = re.sub(r"\D", "", str(x or ""))
    if len(s) == 7:
        return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:7]}"
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return str(x or "")


def _pick(r: dict, *names) -> str:
    """依序找欄位（中文或英文；名稱可能帶空白或只是部分相同，例如「主旨 」）。"""
    for n in names:
        if r.get(n) not in (None, ""):
            return str(r[n]).strip()
    for n in names:
        for k, v in r.items():
            if n.lower() in k.lower() and v not in (None, ""):
                return str(v).strip()
    return ""


def parse_material(rows: list) -> pd.DataFrame:
    """OpenAPI 重大訊息 JSON（證交所中文欄位、櫃買中心可能是英文欄位）→ date、time、ticker、name、title、detail。"""
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        r = {str(k).strip(): v for k, v in r.items()}
        t = _pick(r, "公司代號", "SecuritiesCompanyCode", "CompanyCode", "Code")
        if not t:
            continue
        tm = re.sub(r"\D", "", _pick(r, "發言時間", "SpeechTime", "Time"))
        out.append({"date": _roc_date(_pick(r, "發言日期", "SpeechDate", "出表日期", "Date")),
                    "time": f"{tm[:2]}:{tm[2:4]}" if len(tm) >= 4 else "",
                    "ticker": t, "name": _pick(r, "公司名稱", "CompanyName", "CompanyAbbreviation", "Name"),
                    "title": _pick(r, "主旨", "Subject", "Title"), "detail": _pick(r, "說明", "Description", "Content")[:600]})
    return pd.DataFrame(out, columns=["date", "time", "ticker", "name", "title", "detail"])


def material_today(timeout: float = 6) -> pd.DataFrame:
    """上市＋上櫃當日重大訊息（證交所／櫃買中心 OpenAPI，盤後與盤中都會更新）；抓不到回傳空表。"""
    frames = []
    for url in (TWSE_MATERIAL, TPEX_MATERIAL):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            frames.append(parse_material(r.json()))
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(frames, ignore_index=True) if frames else parse_material([])


def parse_rss(text: str, n: int = 8) -> list[dict]:
    """Google 新聞 RSS → [{title, link, source, time}]（時間換成台北）。"""
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    out = []
    for it in root.iter("item"):
        title = html.unescape((it.findtext("title") or "").strip())
        src_el = it.find("source")
        src = (src_el.text or "").strip() if src_el is not None else ""
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        try:
            tm = pd.Timestamp(it.findtext("pubDate") or "")
            tm = None if pd.isna(tm) else (tm.tz_localize("UTC") if tm.tzinfo is None else tm).tz_convert(TZ)
        except Exception:  # noqa: BLE001
            tm = None
        out.append({"title": title, "link": (it.findtext("link") or "").strip(), "source": src, "time": tm})
    out.sort(key=lambda x: x["time"] or pd.Timestamp("1970-01-01", tz=TZ), reverse=True)
    return out[:n]


def google_news(query: str, n: int = 8, days: int = 3, timeout: float = 5) -> list[dict]:
    url = (f"https://news.google.com/rss/search?q={quote(query)}+when:{days}d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        return parse_rss(r.text, n)
    except Exception:  # noqa: BLE001
        return []
