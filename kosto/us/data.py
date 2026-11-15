"""v3.4 美股資料：Yahoo Finance chart API（免金鑰）。

* daily()：日線（含還原收盤；開高低依還原比例調整），每日名單與 K 線用。
* intraday()：當日分時（含盤前、盤後；1～5 分鐘一筆），盤前／即時名單與分時圖用。
* snapshot()：一檔股票目前在哪個時段、最新價、相對昨收、盤前漲跌、盤後漲跌。
* panel()：多檔日線 → 「日期 × 股票」寬表（模型輸入）。

Yahoo 美股報價大多為即時（交易所 Nasdaq 最後成交）或延遲數分鐘；只用於顯示與名單，不下單。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

HOSTS = ["https://query1.finance.yahoo.com/v8/finance/chart/", "https://query2.finance.yahoo.com/v8/finance/chart/"]
if os.getenv("KOSTO_YAHOO_HOST"):                     # 測試用：指向本機模擬伺服器
    HOSTS = [os.environ["KOSTO_YAHOO_HOST"]] * 2
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
ET = "America/New_York"
TPE = "Asia/Taipei"


def ysym(t: str) -> str:
    """BRK.B → BRK-B（Yahoo 寫法）。"""
    return str(t).strip().upper().replace(".", "-")


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _get(sym: str, params: dict, session=None, timeout: float = 15, tries: int = 3) -> dict:
    s = session or requests
    err = None
    for k in range(tries):
        try:
            r = s.get(HOSTS[k % 2] + requests.utils.quote(ysym(sym), safe="^=-"), params=params, headers=UA, timeout=timeout)
            if r.status_code == 404:
                return {}
            if r.status_code == 429:
                raise requests.RequestException("429 太頻繁")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            err = e
            time.sleep(0.8 * (k + 1))
    raise RuntimeError(f"{sym}: {err}")


# ------------------------------------------------------------------ 日線
def parse_daily(payload: dict) -> pd.DataFrame:
    """chart JSON → 日線 DataFrame（index＝美東日期；欄位 open/high/low/close/adj/volume，adj＝還原收盤）。"""
    res = ((payload or {}).get("chart") or {}).get("result") or []
    if not res:
        return pd.DataFrame(columns=["open", "high", "low", "close", "adj", "volume"])
    r = res[0]
    ts = r.get("timestamp") or []
    meta = r.get("meta") or {}
    off = int(meta.get("gmtoffset") or 0)
    ind = r.get("indicators") or {}
    q = (ind.get("quote") or [{}])[0]
    adj = ((ind.get("adjclose") or [{}])[0] or {}).get("adjclose")
    n = len(ts)

    def col(k, src=q):
        v = (src or {}).get(k) or [None] * n
        return [_f(x) for x in v[:n]] + [np.nan] * max(0, n - len(v))
    f = pd.DataFrame({"open": col("open"), "high": col("high"), "low": col("low"), "close": col("close"),
                      "volume": col("volume")},
                     index=pd.to_datetime(np.asarray(ts, dtype="int64") + off, unit="s").normalize())
    f["adj"] = [_f(x) for x in adj[:n]] if adj and len(adj) >= n else f["close"]
    f = f[f["close"] > 0]
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f[["open", "high", "low", "close", "adj", "volume"]]


def daily(sym: str, rng: str = "2y", session=None) -> pd.DataFrame:
    return parse_daily(_get(sym, {"range": rng, "interval": "1d", "includeAdjustedClose": "true",
                                  "events": "div,split"}, session))


def panel(symbols, rng: str = "2y", workers: int = 6, pause: float = 0.05, log=print) -> dict[str, pd.DataFrame]:
    """多檔日線 → 寬表 C（還原收盤）、O／H／L（依還原比例調整）、V（股數）、RAW（未還原收盤，顯示用）。"""
    S = requests.Session()
    out, fail = {}, []

    def one(t):
        time.sleep(pause)
        try:
            return t, daily(t, rng, S)
        except Exception as e:  # noqa: BLE001
            return t, str(e)
    with ThreadPoolExecutor(workers) as ex:
        for t, f in ex.map(one, list(dict.fromkeys(symbols))):
            if isinstance(f, pd.DataFrame) and len(f) > 0:
                out[t] = f
            else:
                fail.append(t)
    if log:
        log(f"  美股日線：成功 {len(out)} 檔、失敗 {len(fail)} 檔{('（' + '、'.join(fail[:10]) + '…）') if fail else ''}")
    if not out:
        return {}
    idx = sorted(set().union(*[set(f.index) for f in out.values()]))
    cols = list(out)

    def wide(k):
        return pd.DataFrame({t: out[t][k] for t in cols}).reindex(idx)
    raw = wide("close")
    C = wide("adj")
    ratio = (C / raw).where(raw > 0)
    return {"C": C, "O": wide("open") * ratio, "H": wide("high") * ratio, "L": wide("low") * ratio, "V": wide("volume"),
            "RAW": raw, "fail": fail}


# ------------------------------------------------------------------ 分時與時段
def _window(meta: dict, key: str) -> tuple[int, int] | None:
    p = ((meta.get("currentTradingPeriod") or {}).get(key) or {})
    if "start" in p and "end" in p:
        return int(p["start"]), int(p["end"])
    return None


def parse_intraday(payload: dict) -> tuple[pd.DataFrame, dict]:
    """chart JSON（range=1d、includePrePost）→ (分時 DataFrame[t(UTC), price, volume, session], meta)。"""
    res = ((payload or {}).get("chart") or {}).get("result") or []
    if not res:
        return pd.DataFrame(columns=["t", "price", "volume", "session"]), {}
    r = res[0]
    meta = r.get("meta") or {}
    ts = r.get("timestamp") or []
    q = ((r.get("indicators") or {}).get("quote") or [{}])[0]
    cl = q.get("close") or [None] * len(ts)
    vo = q.get("volume") or [None] * len(ts)
    f = pd.DataFrame({"ts": ts, "price": [_f(x) for x in cl[:len(ts)]], "volume": [_f(x) for x in vo[:len(ts)]]})
    f = f[f["price"] > 0]
    reg = _window(meta, "regular")
    if reg:
        f["session"] = np.where(f["ts"] < reg[0], "pre", np.where(f["ts"] >= reg[1], "post", "regular"))
    else:
        f["session"] = "regular"
    f["t"] = pd.to_datetime(f["ts"], unit="s", utc=True)
    return f[["t", "price", "volume", "session"]].reset_index(drop=True), meta


def intraday(sym: str, interval: str = "2m", session=None) -> tuple[pd.DataFrame, dict]:
    return parse_intraday(_get(sym, {"range": "1d", "interval": interval, "includePrePost": "true"}, session))


def phase(meta: dict, now: pd.Timestamp | None = None) -> str:
    """pre＝盤前、regular＝盤中、post＝盤後、closed＝休市（夜間／假日）。用 Yahoo 給的當日時段，夏令時間自動處理。"""
    now = int((now or pd.Timestamp.now(tz="UTC")).timestamp())
    pre, reg, post = _window(meta, "pre"), _window(meta, "regular"), _window(meta, "post")
    if reg and reg[0] <= now < reg[1]:
        return "regular"
    if pre and pre[0] <= now < pre[1]:
        return "pre"
    if post and post[0] <= now < post[1]:
        return "post"
    return "closed"


def snapshot(sym: str, bars: pd.DataFrame, meta: dict, now: pd.Timestamp | None = None) -> dict:
    """一檔股票目前的報價摘要。
    盤前：base＝昨收（regularMarketPrice 仍是昨天收盤），ext＝盤前最後一筆 → 盤前漲跌。
    盤中：price＝最新價，prev＝昨收。盤後：price＝今日收盤，ext＝盤後最後一筆。"""
    ph = phase(meta, now)
    reg_px = _f(meta.get("regularMarketPrice"))
    reg = _window(meta, "regular")
    has_today_regular = bool(len(bars)) and (bars["session"] == "regular").any()
    from ..intraday import prev_close
    prevc = prev_close(meta)                      # fulldayPrice − fulldayChange 等（Yahoo 的 previousClose 有時不可靠）
    if not np.isfinite(prevc) or (np.isfinite(reg_px) and abs(reg_px / prevc - 1) > 0.6):
        prevc = _f(meta.get("chartPreviousClose"))
    out = {"sym": sym, "phase": ph, "price": reg_px, "prev": np.nan, "chg": np.nan, "ext": np.nan, "ext_chg": np.nan,
           "ext_label": "", "time": pd.Timestamp(int(meta.get("regularMarketTime") or 0), unit="s", tz="UTC").tz_convert(TPE)
           if meta.get("regularMarketTime") else pd.NaT, "currency": meta.get("currency", "USD"),
           "name": meta.get("shortName") or meta.get("longName") or "", "day_high": _f(meta.get("regularMarketDayHigh")),
           "day_low": _f(meta.get("regularMarketDayLow")), "volume": _f(meta.get("regularMarketVolume")),
           "high52": _f(meta.get("fiftyTwoWeekHigh")), "low52": _f(meta.get("fiftyTwoWeekLow"))}
    if ph == "pre" or (not has_today_regular and ph != "regular"):
        # 今天還沒開盤：regularMarketPrice＝最近一次收盤
        out["prev"] = prevc
        out["chg"] = reg_px / prevc - 1 if np.isfinite(prevc) and prevc > 0 else np.nan
        pre = bars[bars["session"] == "pre"]
        if len(pre) and np.isfinite(reg_px) and reg_px > 0:
            out["ext"] = float(pre["price"].iloc[-1])
            out["ext_chg"] = out["ext"] / reg_px - 1
            out["ext_label"] = "盤前"
            out["time"] = pre["t"].iloc[-1].tz_convert(TPE)
        out["base"] = reg_px
        return out
    out["prev"] = prevc
    out["chg"] = reg_px / prevc - 1 if np.isfinite(prevc) and prevc > 0 else np.nan
    out["base"] = reg_px
    if ph in ("post", "closed"):
        post = bars[bars["session"] == "post"]
        if len(post) and np.isfinite(reg_px) and reg_px > 0:
            out["ext"] = float(post["price"].iloc[-1])
            out["ext_chg"] = out["ext"] / reg_px - 1
            out["ext_label"] = "盤後"
    if ph == "regular" and len(bars) and reg:
        last = bars[bars["session"] == "regular"]
        if len(last):
            out["time"] = last["t"].iloc[-1].tz_convert(TPE)
    return out


def quotes(symbols, interval: str = "5m", workers: int = 8, timeout: float = 8) -> tuple[pd.DataFrame, dict]:
    """多檔即時摘要（App 用；失敗的略過）→ (摘要表, {sym: 分時})。"""
    S = requests.Session()

    def one(t):
        try:
            bars, meta = parse_intraday(_get(t, {"range": "1d", "interval": interval, "includePrePost": "true"}, S,
                                             timeout=timeout, tries=2))
            if not meta:
                return None
            return snapshot(t, bars, meta), bars
        except Exception:  # noqa: BLE001
            return None
    rows, bars = [], {}
    with ThreadPoolExecutor(workers) as ex:
        for t, x in zip(symbols, ex.map(one, list(symbols))):
            if x:
                rows.append(x[0])
                bars[t] = x[1]
    return pd.DataFrame(rows), bars


def market_clock(now: pd.Timestamp | None = None) -> dict:
    """不靠網路的美股時鐘（夏令自動）：今天各時段的台北時間、目前時段（假日無法判斷，只看平日）。"""
    now = (now or pd.Timestamp.now(tz="UTC")).tz_convert(ET)
    d = now.normalize()
    t = {k: d + pd.Timedelta(hours=h, minutes=m) for k, (h, m) in
         {"pre": (4, 0), "open": (9, 30), "close": (16, 0), "post_end": (20, 0)}.items()}
    wk = now.weekday() < 5
    if not wk:
        ph = "closed"
    elif t["pre"] <= now < t["open"]:
        ph = "pre"
    elif t["open"] <= now < t["close"]:
        ph = "regular"
    elif t["close"] <= now < t["post_end"]:
        ph = "post"
    else:
        ph = "closed"
    tp = {k: v.tz_convert(TPE) for k, v in t.items()}
    dst = bool(now.dst() and now.dst().total_seconds() > 0)
    return {"phase": ph, "dst": dst, "et": now, "tpe": now.tz_convert(TPE),
            "pre_tpe": tp["pre"].strftime("%H:%M"), "open_tpe": tp["open"].strftime("%H:%M"),
            "close_tpe": tp["close"].strftime("%H:%M"), "post_end_tpe": tp["post_end"].strftime("%H:%M")}


PHASE_ZH = {"pre": "盤前", "regular": "盤中", "post": "盤後", "closed": "休市"}
