"""盤中即時報價：臺灣證券交易所「基本市況報導」(mis.twse.com.tw，上市＋上櫃，約 5 秒延遲，免費)。

取不到（非交易時間、網路限制）時回傳空表，App 會改用最近收盤價。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

MIS = "https://mis.twse.com.tw/stock"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Referer": "https://mis.twse.com.tw/stock/index.jsp", "Accept-Language": "zh-TW,zh;q=0.9"}
INDEX_CH = {"加權指數": "tse_t00.tw", "櫃買指數": "otc_o00.tw"}


def _num(x):
    try:
        v = float(str(x).split("_")[0])
    except (TypeError, ValueError):
        return np.nan
    return v if v > 0 else np.nan


def _levels(x) -> list[float]:
    """五檔價格 '0.0000_132.5000_132.0000_' → [0.0, 132.5, 132.0]；0＝市價委託（漲跌停鎖住時常排在最前面）。"""
    out = []
    for s_ in str(x or "").split("_"):
        try:
            v = float(s_)
        except ValueError:
            continue
        if np.isfinite(v) and v >= 0:
            out.append(v)
    return out


def parse_mis(payload: dict) -> pd.DataFrame:
    rows = []
    for m in payload.get("msgArray", []) or []:
        last = _num(m.get("z"))
        bl, al = _levels(m.get("b")), _levels(m.get("a"))
        bid = next((v for v in bl if v > 0), np.nan)
        ask = next((v for v in al if v > 0), np.nan)
        up, dn = _num(m.get("u")), _num(m.get("w"))
        src = "trade"
        # v2.2.1 這一瞬間沒有成交（z 為 "-"）：不用買賣「中間值」（31.83、103.8 這種不能下單的價格），
        # 依序改用 pz（最近成交／試撮價）→ 漲跌停鎖住 → 最佳委買價 → 最佳委賣價。
        # v2.3.1 漲跌停鎖住時委買（委賣）第一檔常是「市價」（0），舊版因此取不到價格、整檔改顯示前一日收盤 → 現在跳過 0、
        # 並在「買方在漲停價（或市價）、沒有賣方」時直接以漲停價為現價（跌停反之）。
        if not np.isfinite(last):
            lock_up = np.isfinite(up) and not np.isfinite(ask) and (bool(bl) and bl[0] == 0 or (np.isfinite(bid) and bid >= up - 1e-9))
            lock_dn = np.isfinite(dn) and not np.isfinite(bid) and (bool(al) and al[0] == 0 or (np.isfinite(ask) and ask <= dn + 1e-9))
            tz = _num((m.get("trade") or {}).get("z")) if isinstance(m.get("trade"), dict) else np.nan   # 最近一筆成交（2026 新欄位）
            for nm, v in (("pz", _num(m.get("pz"))), ("trade", tz), ("limit_up", up if lock_up else np.nan), ("limit_down", dn if lock_dn else np.nan),
                          ("bid", bid), ("ask", ask)):
                if np.isfinite(v):
                    last, src = v, nm
                    break
            else:
                src = ""
        prev = _num(m.get("y"))
        lock = ("up" if np.isfinite(up) and np.isfinite(last) and last >= up - 1e-9 else
                "down" if np.isfinite(dn) and np.isfinite(last) and last <= dn + 1e-9 else "")
        rows.append({"ticker": m.get("c", ""), "name": m.get("n", ""), "price": last, "src": src, "prev": prev,
                     "open": _num(m.get("o")), "high": _num(m.get("h")), "low": _num(m.get("l")),
                     "volume": _num(m.get("v")), "bid": bid, "ask": ask, "limit_up": up, "limit_down": dn, "lock": lock,
                     "time": m.get("t") or m.get("%", ""), "date": m.get("d", ""),
                     "chg": (last / prev - 1) if np.isfinite(last) and np.isfinite(prev) and prev > 0 else np.nan})
    df = pd.DataFrame(rows)
    return df.drop_duplicates("ticker", keep="last") if len(df) else df


def channel(ticker: str, board: str) -> str:
    return f"{'otc' if str(board).upper().startswith('TP') or str(board).upper() == 'OTC' else 'tse'}_{ticker}.tw"


class Quotes:
    def __init__(self, timeout: int = 6, cooldown: int = 300):
        self.s = requests.Session()
        self.s.headers.update(UA)
        self.timeout, self.cooldown = timeout, cooldown
        self._warm = 0.0
        self._fail = 0.0                 # 上次失敗時間：失敗後 cooldown 秒內不再嘗試（避免每次重新整理都卡住）
        self.error = ""

    def _warmup(self):
        if time.time() - self._warm > 600:
            try:
                self.s.get(f"{MIS}/index.jsp", timeout=self.timeout)
            except requests.RequestException:
                pass
            self._warm = time.time()

    def fetch(self, channels: list[str]) -> pd.DataFrame:
        """channels: ['tse_2330.tw', 'otc_6488.tw', ...]（每次最多 50 檔，自動分批）。"""
        if not channels:
            return pd.DataFrame()
        if time.time() - self._fail < self.cooldown:
            return pd.DataFrame()
        self.error = ""
        self._warmup()
        out = []
        for i in range(0, len(channels), 50):
            part = channels[i:i + 50]
            try:
                r = self.s.get(f"{MIS}/api/getStockInfo.jsp",
                               params={"ex_ch": "|".join(part), "json": "1", "delay": "0", "_": int(time.time() * 1000)},
                               timeout=self.timeout)
                r.raise_for_status()
                out.append(parse_mis(r.json()))
            except (requests.RequestException, ValueError) as e:
                self.error = f"即時報價暫時無法取得（{type(e).__name__}），以最近收盤價顯示"
                self._fail = time.time()
                break
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def market_phase(now: pd.Timestamp | None = None) -> str:
    """台北時間：盤前 / 盤中 / 盤後 / 休市（週末）。"""
    now = now or pd.Timestamp.now(tz="Asia/Taipei")
    if now.weekday() >= 5:
        return "休市"
    hm = now.hour * 60 + now.minute
    if hm < 8 * 60 + 30:
        return "盤前"
    if hm <= 13 * 60 + 30:
        return "盤中" if hm >= 9 * 60 else "盤前試撮"
    return "盤後"
