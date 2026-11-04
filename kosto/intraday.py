"""v2.1.1 當日分時走勢：Yahoo Finance chart API（免金鑰，1～2 分鐘一筆）＋台股盤中最新價（證交所 MIS，約 5 秒延遲）。

平盤（昨收）的取法：Yahoo 的 previousClose／chartPreviousClose 不可靠——2026-11-04 實測 ^TWII 給 44,920.1，
實際昨收是 43,307.54（＝收盤 42,948.91 − 今日漲跌 −358.63）。因此依序使用：
  1. fulldayPrice − fulldayChange（Yahoo 自己算好的今日漲跌，^TWII 也正確）
  2. regularMarketPrice ÷ (1 + regularMarketChangePercent)
  3. previousClose、chartPreviousClose（最後才用）
台股盤中再以證交所 MIS 的「昨收」覆蓋（官方數字），並把 MIS 最新成交價接在走勢最後（Yahoo 台股可能延遲）。

只用於顯示，不影響選股模型。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
TZ = "Asia/Taipei"
TAIEX = "^TWII"                      # Yahoo 沒有櫃買指數的分時（^TWOII 不存在）→ 櫃買只顯示 MIS 即時數字


def yahoo_symbol(ticker: str, board: str) -> str:
    b = str(board).upper()
    return f"{ticker}.TWO" if b.startswith("TP") or b == "OTC" else f"{ticker}.TW"


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _periods(meta: dict) -> list[tuple[int, int]]:
    tp = meta.get("tradingPeriods")
    if isinstance(tp, dict):                       # 有盤前盤後時是 {"pre":…, "regular":…, "post":…}
        tp = tp.get("regular") or []
    out = []
    for g in tp or []:
        for p in (g if isinstance(g, list) else [g]):
            if isinstance(p, dict) and "start" in p and "end" in p:
                out.append((int(p["start"]), int(p["end"])))
    if not out:
        r = ((meta.get("currentTradingPeriod") or {}).get("regular") or {})
        if "start" in r and "end" in r:
            out.append((int(r["start"]), int(r["end"])))
    return sorted(set(out))


def prev_close(meta: dict) -> float:
    fp, fc = _f(meta.get("fulldayPrice")), _f(meta.get("fulldayChange"))
    px, pct = _f(meta.get("regularMarketPrice")), _f(meta.get("regularMarketChangePercent"))
    for v in (fp - fc if np.isfinite(fp) and np.isfinite(fc) else np.nan,
              px - fc if np.isfinite(px) and np.isfinite(fc) and not np.isfinite(fp) else np.nan,
              px / (1 + pct / 100) if np.isfinite(px) and np.isfinite(pct) and pct > -100 else np.nan,
              _f(meta.get("previousClose")), _f(meta.get("chartPreviousClose"))):
        if np.isfinite(v) and v > 0:
            return float(v)
    return np.nan


def _local(ts) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(ts, dtype="int64"), unit="s", utc=True).tz_convert(TZ).tz_localize(None)


def parse_chart(payload: dict, sym: str = "") -> dict | None:
    """Yahoo chart JSON → 最近一個交易時段的分時資料（時間一律換成台北時間、不帶時區）。"""
    res = ((payload or {}).get("chart") or {}).get("result") or []
    if not res:
        return None
    m = res[0] or {}
    meta = m.get("meta") or {}
    ts = m.get("timestamp") or []
    q = (((m.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    n = len(ts)

    def col(name):
        a = list(q.get(name) or [])
        return (a + [None] * n)[:n]
    cl, op, vo = col("close"), col("open"), col("volume")
    rows = [(int(t), _f(c), _f(o), _f(v)) for t, c, o, v in zip(ts, cl, op, vo) if np.isfinite(_f(c))]
    if not rows:
        return None
    per = _periods(meta)
    last_t = rows[-1][0]
    start, end = rows[0][0], last_t
    for s, e in per:                                # 取包含最後一筆資料的交易時段
        if s - 600 <= last_t <= e + 600:
            start, end = s, e
    rows = [r for r in rows if start - 600 <= r[0] <= end + 600] or rows
    t = np.array([r[0] for r in rows])
    v = np.array([r[1] for r in rows], float)
    vol = np.array([r[3] for r in rows], float)
    prev = prev_close(meta)
    if not np.isfinite(prev) or abs(v[0] / prev - 1) > 0.35:          # 平盤明顯不合理 → 退回第一筆開盤
        prev = float(rows[0][2]) if np.isfinite(rows[0][2]) else float(v[0])
    hi, lo = _f(meta.get("regularMarketDayHigh")), _f(meta.get("regularMarketDayLow"))
    rmt = int(meta.get("regularMarketTime") or last_t)
    same = start - 600 <= rmt <= end + 4 * 3600                       # meta 的高低是不是這個時段的
    hi = float(max(hi, v.max())) if same and np.isfinite(hi) and abs(hi / v.max() - 1) < 0.2 else float(v.max())
    lo = float(min(lo, v.min())) if same and np.isfinite(lo) and abs(lo / v.min() - 1) < 0.2 else float(v.min())
    px = _f(meta.get("regularMarketPrice"))
    last = float(px) if same and np.isfinite(px) and abs(px / v[-1] - 1) < 0.2 else float(v[-1])
    opn = rows[0][2] if np.isfinite(rows[0][2]) else rows[0][1]
    tz = meta.get("exchangeTimezoneName") or TZ
    return {"sym": sym or meta.get("symbol", ""), "t": _local(t), "v": v, "vol": vol, "prev": float(prev),
            "last": last, "high": hi, "low": lo, "open": float(opn),
            "start": _local([start])[0], "end": _local([end])[0],
            "date": pd.Timestamp(start, unit="s", tz="UTC").tz_convert(tz).date(),
            "time": _local([min(rmt, end) if same else last_t])[0], "tz": tz, "live_point": False}


def fetch(sym: str, interval: str = "2m", timeout: float = 4, session=None) -> dict | None:
    s = session or requests.Session()

    def get(rng):
        r = s.get(YAHOO + requests.utils.quote(sym, safe=""), headers=UA, timeout=timeout,
                  params={"range": rng, "interval": interval, "includePrePost": "false"})
        r.raise_for_status()
        return parse_chart(r.json(), sym)
    d = get("1d")
    if d is None or len(d["v"]) < 3:                # 開盤前：1d 可能還沒有今天的資料 → 取最近 5 天的最後一個時段
        d5 = get("5d")
        d = d5 if d5 is not None else d
    return d


def fetch_many(syms, interval: str = "2m", timeout: float = 4, workers: int = 8) -> dict:
    syms = [s for s in dict.fromkeys(syms) if s]
    if not syms:
        return {}
    sess = requests.Session()

    def safe(s):
        try:
            return s, fetch(s, interval, timeout, sess)
        except Exception:  # noqa: BLE001
            return s, None
    with ThreadPoolExecutor(max_workers=min(workers, len(syms))) as ex:
        return {s: d for s, d in ex.map(safe, syms) if d is not None}


TW_OPEN, TW_CLOSE = (9, 0), (13, 30)


def tw_session(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = pd.Timestamp(day).normalize()
    return d + pd.Timedelta(hours=TW_OPEN[0], minutes=TW_OPEN[1]), d + pd.Timedelta(hours=TW_CLOSE[0], minutes=TW_CLOSE[1])


def merge_live(d: dict | None, q: dict | None, now: pd.Timestamp | None = None) -> dict | None:
    """台股：用證交所 MIS 的即時價補上最新一點、以官方昨收為平盤。
    q：quotes.parse_mis 的一列（price、prev、open、high、low、date=YYYYMMDD、time=HH:MM:SS）。"""
    if not q or not np.isfinite(_f(q.get("price"))):
        return d
    now = now if now is not None else pd.Timestamp.now(tz=TZ).tz_localize(None)
    try:
        day = pd.Timestamp(str(q.get("date") or now.date())).date()
        hh = str(q.get("time") or "")
        qt = pd.Timestamp(f"{day} {hh}") if len(hh) >= 5 else now
    except (ValueError, TypeError):
        day, qt = now.date(), now
    s0, s1 = tw_session(day)
    qt = min(max(qt, s0), s1)
    px, prev = float(q["price"]), _f(q.get("prev"))
    if d is None or d["date"] < day:                 # Yahoo 還沒有今天的分時 → 只用開盤與最新價兩點
        o = _f(q.get("open"))
        pts = [(s0, o)] if np.isfinite(o) and qt > s0 else []
        pts.append((qt, px))
        d = {"sym": "", "t": pd.DatetimeIndex([p[0] for p in pts]), "v": np.array([p[1] for p in pts], float),
             "vol": np.full(len(pts), np.nan), "prev": prev if np.isfinite(prev) else px, "last": px,
             "high": px, "low": px, "open": o if np.isfinite(o) else px, "start": s0, "end": s1, "date": day,
             "time": qt, "tz": TZ, "live_point": True, "sparse": True}
    elif d["date"] == day:
        d = dict(d)
        if qt > d["t"][-1]:
            d["t"] = d["t"].append(pd.DatetimeIndex([qt]))
            d["v"] = np.append(d["v"], px)
            d["vol"] = np.append(d["vol"], np.nan)
            d["live_point"] = True
        d["last"], d["time"] = px, max(qt, d["time"])
    else:
        return d
    if np.isfinite(prev) and prev > 0:
        d["prev"] = prev
    for k, f in (("high", max), ("low", min)):
        x = _f(q.get(k))
        d[k] = f(d[k], x, px) if np.isfinite(x) else f(d[k], px)
    return d


def frac(d: dict) -> np.ndarray:
    """每一點在交易時段中的位置（0＝開盤、1＝收盤）；盤中尚未走完的時段會留白。"""
    span = (d["end"] - d["start"]).total_seconds()
    if span <= 0:
        return np.linspace(0, 1, len(d["v"]))
    return np.clip(np.asarray((d["t"] - d["start"]).total_seconds(), float) / span, 0, 1)


def thin(x: np.ndarray, y: np.ndarray, n: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """點太多時抽樣（保留最高、最低與最後一點），給迷你圖用。"""
    if len(y) <= n:
        return x, y
    idx = set(np.linspace(0, len(y) - 1, n).round().astype(int).tolist()) | {int(np.argmax(y)), int(np.argmin(y)), len(y) - 1}
    idx = np.array(sorted(idx))
    return x[idx], y[idx]


def status(d: dict, now: pd.Timestamp | None = None) -> str:
    """'盤中 10:32'／'11/04 收盤'。"""
    now = now if now is not None else pd.Timestamp.now(tz=TZ).tz_localize(None)
    if d["start"] <= now <= d["end"] + pd.Timedelta(minutes=5):
        return f"盤中 {d['time']:%H:%M}"
    return f"{pd.Timestamp(d['date']):%m/%d} 收盤"
