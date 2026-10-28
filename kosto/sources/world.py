"""國際市場日收盤（Yahoo Finance chart API，免金鑰）：費城半導體、那斯達克、標普、VIX、日經、韓國、美元兌台幣、台積電 ADR。

研究用時間對齊（見 panel.align_world）：美股與匯率只用「美國日期 ≤ T−1」的收盤；日韓用 T 日收盤。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
SYMBOLS = {"SOX": "^SOX", "IXIC": "^IXIC", "GSPC": "^GSPC", "VIX": "^VIX", "N225": "^N225", "KS11": "^KS11",
           "TWD": "TWD=X", "TSM": "TSM", "NQF": "NQ=F", "ESF": "ES=F"}
NAME_ZH = {"SOX": "費城半導體", "IXIC": "那斯達克", "GSPC": "標普500", "VIX": "VIX恐慌指數", "N225": "日經225",
           "KS11": "韓國KOSPI", "TWD": "美元/台幣", "TSM": "台積電ADR", "NQF": "那斯達克期貨", "ESF": "標普期貨"}
US = {"SOX", "IXIC", "GSPC", "VIX", "TSM", "NQF", "ESF", "TWD"}       # 收盤晚於台股研究時間 → 落後一日使用
ASIA = {"N225", "KS11"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def parse_chart(payload: dict, key: str) -> pd.DataFrame:
    res = (payload.get("chart") or {}).get("result") or []
    if not res:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    r = res[0]
    ts = r.get("timestamp") or []
    off = int((r.get("meta") or {}).get("gmtoffset") or 0)
    close = ((r.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    rows = []
    for t, c in zip(ts, close):
        if c is None or not np.isfinite(c) or c <= 0:
            continue
        rows.append({"date": pd.Timestamp(int(t) + off, unit="s").normalize(), "ticker": key, "close": float(c)})
    f = pd.DataFrame(rows)
    return f.drop_duplicates(["date", "ticker"], keep="last") if len(f) else pd.DataFrame(columns=["date", "ticker", "close"])


def fetch(keys=None, start="2016-11-01", end=None, session=None, pause=1.0, log=print) -> pd.DataFrame:
    s = session or requests.Session()
    s.headers.update(UA)
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int((pd.Timestamp(end) if end else pd.Timestamp.now() + pd.Timedelta(days=2)).timestamp())
    out, dead = [], 0
    for k in keys or ["SOX", "IXIC", "GSPC", "VIX", "N225", "KS11", "TWD", "TSM"]:
        sym = SYMBOLS[k]
        for attempt in range(3):
            try:
                r = s.get(YAHOO + requests.utils.quote(sym, safe=""), timeout=20,
                          params={"period1": p1, "period2": p2, "interval": "1d", "includePrePost": "false"})
                if r.status_code == 429:
                    raise requests.RequestException("429")
                r.raise_for_status()
                out.append(parse_chart(r.json(), k))
                break
            except requests.ConnectionError as e:          # 連不上主機（防火牆/離線）→ 不必每檔重試
                dead += 1
                log(f"  world {k} 連線失敗（{str(e)[:60]}）")
                break
            except (requests.RequestException, ValueError) as e:
                log(f"  world {k} 失敗（{str(e)[:60]}），{3 * (attempt + 1)} 秒後重試")
                time.sleep(3 * (attempt + 1))
        if dead >= 2 and not out:
            log("  world 主機無法連線，略過國際市場資料（不影響研究）")
            break
        time.sleep(pause)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "ticker", "close"])


def _quote_one(k: str, timeout: float) -> dict | None:
    r = requests.get(YAHOO + requests.utils.quote(SYMBOLS[k], safe=""), params={"range": "1mo", "interval": "1d"},
                     headers=UA, timeout=timeout)
    r.raise_for_status()
    m = r.json()["chart"]["result"][0]
    meta = m.get("meta") or {}
    ts = m.get("timestamp") or []
    cl = (m.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
    pairs = [(t, c) for t, c in zip(ts, cl) if c is not None and np.isfinite(c)]
    if not pairs:
        return None
    px = meta.get("regularMarketPrice") or pairs[-1][1]
    prev = pairs[-2][1] if len(pairs) >= 2 else meta.get("chartPreviousClose")
    t = int(meta.get("regularMarketTime") or pairs[-1][0])
    return {"key": k, "name": NAME_ZH[k], "price": float(px), "prev": float(prev) if prev else np.nan,
            "chg": float(px) / float(prev) - 1 if prev else np.nan,
            "time": pd.Timestamp(t, unit="s", tz="UTC").tz_convert("Asia/Taipei"),
            "bar_date": pd.Timestamp(pairs[-1][0] + int(meta.get("gmtoffset") or 0), unit="s").normalize(),
            "spark": [float(c) for _, c in pairs[-22:]]}


def quote(keys=None, timeout: float = 6, workers: int = 6) -> pd.DataFrame:
    """App 用：最新價、與前一交易日相比漲跌、近一個月收盤（迷你走勢）；失敗的商品略過，全失敗回空表。"""
    from concurrent.futures import ThreadPoolExecutor
    keys = keys or ["SOX", "IXIC", "NQF", "ESF", "TSM", "VIX", "N225", "KS11", "TWD"]

    def safe(k):
        try:
            return _quote_one(k, timeout)
        except Exception:  # noqa: BLE001
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [x for x in ex.map(safe, keys) if x]
    return pd.DataFrame(rows)
