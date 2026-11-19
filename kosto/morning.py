"""v3.16 每日台股盤前日報（docs/v3.16_protocol.md）：台北 05:40～06:40 自動產出，07:00 前可看。

段落：美股收盤總評／夜盤趨勢／權值股動態／重大消息／影響台股關聯性／今日操作建議（保守、積極）。
* fit(B)：每日研究時估計（研究資料到前一個交易日）——M1 大盤開盤跳空、M2 台積電開盤、M3 開高不追、M4 大跌不恐慌的歷史驗證，
  與權值股的研究資料背景（昨收、外資）。結果存進 results.pkl 的 res["morning_model"]。
* fetch_live()：盤前抓 Yahoo 日線／分時、期交所夜盤（連不到就略過）、Google 新聞 RSS。
* build()：組成日報 dict（JSON 可存）；每段標證據等級：tested＝通過預先宣告的歷史檢驗，ref＝僅供參考。
不改選股模型、不下單；僅供研究參考，非投資建議。
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

TZ = "Asia/Taipei"
YAHOO = os.getenv("KOSTO_YAHOO_HOST") or "https://query1.finance.yahoo.com/v8/finance/chart/"   # 環境變數只供測試
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
SPLIT = "2024-01-01"            # 驗證切點：之前估計、之後檢驗（docs/v3.16_protocol.md）
VERSION = 1

US_IDX = [("DJI", "^DJI", "道瓊"), ("GSPC", "^GSPC", "標普500"), ("IXIC", "^IXIC", "那斯達克"), ("SOX", "^SOX", "費城半導體"),
          ("VIX", "^VIX", "VIX 恐慌指數")]
MACRO = [("TNX", "^TNX", "美10年債殖利率"), ("DXY", "DX-Y.NYB", "美元指數"), ("TWD", "TWD=X", "美元/台幣"), ("CL", "CL=F", "西德州原油"),
         ("GC", "GC=F", "黃金"), ("N225", "^N225", "日經225"), ("KS11", "^KS11", "韓國KOSPI")]
FUT = [("NQF", "NQ=F", "那指期"), ("ESF", "ES=F", "標普期"), ("YMF", "YM=F", "道瓊期")]
ADR = {"TSM": ("TSM", "台積電ADR", "2330", 5), "UMC": ("UMC", "聯電ADR", "2303", 5), "ASX": ("ASX", "日月光ADR", "3711", 2),
       "CHT": ("CHT", "中華電ADR", "2412", 10)}
PEERS = {"NVDA": "輝達", "AAPL": "蘋果", "AVGO": "博通", "AMD": "超微", "MU": "美光", "MSFT": "微軟", "GOOGL": "Alphabet", "AMZN": "亞馬遜",
         "META": "Meta", "TSLA": "特斯拉", "QCOM": "高通", "ARM": "安謀", "DELL": "戴爾", "SMCI": "美超微", "VRT": "Vertiv", "ANET": "Arista",
         "ASML": "艾司摩爾", "INTC": "英特爾"}
HEAVY = [("2330", "台積電", ["TSM", "NVDA", "AAPL"]), ("2317", "鴻海", ["AAPL", "NVDA", "DELL"]), ("2454", "聯發科", ["QCOM", "ARM"]),
         ("2308", "台達電", ["VRT", "NVDA"]), ("2382", "廣達", ["DELL", "SMCI", "NVDA"]), ("3711", "日月光投控", ["ASX", "NVDA"]),
         ("2303", "聯電", ["UMC", "INTC"]), ("2412", "中華電", ["CHT"]), ("2891", "中信金", []), ("2881", "富邦金", [])]
THEMES = [("AI 晶片與伺服器", ["NVDA", "AMD", "AVGO", "SMCI", "DELL", "VRT"], "廣達、緯創、緯穎、台達電、奇鋐"),
          ("記憶體", ["MU"], "南亞科、華邦電、旺宏、群聯"),
          ("半導體設備", ["ASML", "AMAT", "LRCX"], "弘塑、辛耘、家登、京鼎"),
          ("蘋果供應鏈", ["AAPL"], "鴻海、大立光、玉晶光、台郡"),
          ("網通與光通訊", ["ANET", "AVGO"], "智邦、聯亞、華星光、上詮"),
          ("手機晶片", ["QCOM", "ARM"], "聯發科、瑞昱"),
          ("電動車", ["TSLA"], "和大、貿聯、台達電")]
NEWS_Q = ["美股 收盤", "台股 盤前", "台積電", "聯準會 OR Fed", "輝達"]
TAGS = [("AI／半導體", r"AI|輝達|NVIDIA|半導體|晶片|台積|晶圓|伺服器|記憶體|HBM"), ("利率／央行", r"Fed|聯準會|升息|降息|利率|殖利率|通膨|CPI|PCE|就業|非農"),
        ("關稅／貿易", r"關稅|貿易|出口管制|禁令|制裁|川普|談判"), ("地緣政治", r"戰爭|衝突|飛彈|台海|中東|俄|烏|以色列|伊朗"),
        ("財報／展望", r"財報|營收|獲利|EPS|財測|展望|法說")]


def _f(x) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def _clean(o):
    """轉成 JSON 可存的型別（NaN→None、numpy→python）。"""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


# ================================================================== 歷史驗證（每日研究時估計）
def _daily_table(B) -> pd.DataFrame:
    """每日：市值前 50 名市值加權開盤跳空／開盤→收盤（加權指數代理）、前 300 名等權開盤→收盤、台積電開盤跳空。"""
    days = pd.DatetimeIndex(B.days)
    R, OC = B.a["R"].astype(float), B.a["OC"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        GAP = np.where(np.isfinite(OC), (1 + R) / (1 + OC) - 1, np.nan)
    MC = B.a["MCAP"].astype(float)
    sp = np.asarray(B.sig_pos)
    sod = np.searchsorted(sp, np.arange(len(days)), side="right") - 1
    j2330 = list(B.tickers).index("2330") if "2330" in B.tickers else None
    rows = []
    for d in range(1, len(days)):
        s = sod[d - 1]
        if s < 0:
            continue
        mc = np.nan_to_num(MC[s])
        ok = np.isfinite(GAP[d]) & np.isfinite(OC[d]) & (mc > 0)
        if ok.sum() < 100:
            continue
        order = np.argsort(-np.where(ok, mc, -1))
        top, top300 = order[:50], order[:300]
        w = mc[top] / mc[top].sum()
        rows.append({"date": days[d], "gap50": float(np.nansum(w * GAP[d, top])), "oc50": float(np.nansum(w * OC[d, top])),
                     "oc300": float(np.nanmean(OC[d, top300])),
                     "g2330": float(GAP[d, j2330]) if j2330 is not None else np.nan})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _overnight(B) -> pd.DataFrame:
    keys = B.meta.get("world_keys") or []
    if "WORLD" not in B.a or not {"SOX", "IXIC", "TSM", "TWD", "VIX"} <= set(keys):
        return pd.DataFrame()
    W = pd.DataFrame(B.a["WORLD"].astype(float), index=pd.DatetimeIndex(B.days), columns=keys)
    r = (W / W.shift(1) - 1).where(lambda x: x.abs() < 0.25)
    return pd.DataFrame({"sox": r["SOX"], "ixic": r["IXIC"], "tsm_twd": (1 + r["TSM"]) * (1 + r["TWD"]) - 1,
                         "dvix": W["VIX"] - W["VIX"].shift(1)})


X1 = ["sox", "ixic", "tsm_twd", "dvix"]


def _ols(df, y, xs):
    A = np.c_[np.ones(len(df)), df[xs].to_numpy(float)]
    return np.linalg.lstsq(A, df[y].to_numpy(float), rcond=None)[0]


def _pred(df, b, xs):
    return b[0] + df[xs].to_numpy(float) @ b[1:]


def _stats(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    c = float(np.corrcoef(y, p)[0, 1]) if len(y) > 10 else np.nan
    m = np.abs(p) >= 0.003
    return {"corr": c, "hit": float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.sum() else np.nan, "n_hit": int(m.sum()), "n": int(len(y)),
            "mae": float(np.mean(np.abs(y - p))), "mae0": float(np.mean(np.abs(y)))}


def _cond(df, m) -> dict:
    x = df[m]
    return {"n": int(len(x)), "oc300": float(x["oc300"].mean()) if len(x) else np.nan, "neg": float((x["oc300"] < 0).mean()) if len(x) else np.nan,
            "gap50": float(x["gap50"].mean()) if len(x) else np.nan}


def fit(B, split: str = SPLIT) -> dict:
    """M1～M4（docs/v3.16_protocol.md 第二節）＋權值股背景。失敗回傳含 error 的 dict（不影響每日研究）。"""
    try:
        T, X = _daily_table(B), _overnight(B)
        if T.empty or X.empty:
            return {"error": "研究資料缺少國際市場或個股開盤資料"}
        Z = T.join(X, how="inner").dropna(subset=["gap50"] + X1)
        Z = Z[(Z[["sox", "ixic", "tsm_twd"]].abs() > 0).any(axis=1)]
        tr, te = Z[Z.index < split], Z[Z.index >= split]
        out = {"version": VERSION, "asof": str(Z.index[-1].date()), "period": [str(Z.index[0].date()), str(Z.index[-1].date())], "split": split}
        b_tr = _ols(tr, "gap50", X1)
        p_te, p_tr = _pred(te, b_tr, X1), _pred(tr, b_tr, X1)
        b_all = _ols(Z, "gap50", X1)
        out["m1"] = {"coef": dict(zip(["const"] + X1, map(float, b_all))), "rmse": float((Z["gap50"] - _pred(Z, b_all, X1)).std()),
                     "train": _stats(tr["gap50"], p_tr), "test": _stats(te["gap50"], p_te)}
        t1 = out["m1"]["test"]
        out["m1"]["tested"] = bool(t1["corr"] >= 0.4 and t1["hit"] >= 0.65)
        z2 = Z.dropna(subset=["g2330"])
        tr2, te2 = z2[z2.index < split], z2[z2.index >= split]
        b2 = _ols(tr2, "g2330", ["tsm_twd"])
        b2a = _ols(z2, "g2330", ["tsm_twd"])
        out["m2"] = {"coef": {"const": float(b2a[0]), "tsm_twd": float(b2a[1])}, "rmse": float((z2["g2330"] - _pred(z2, b2a, ["tsm_twd"])).std()),
                     "test": _stats(te2["g2330"], _pred(te2, b2, ["tsm_twd"]))}
        out["m2"]["tested"] = bool(out["m2"]["test"]["corr"] >= 0.4)
        te = te.assign(p=p_te)
        out["m3"] = _cond(te, te["p"] >= 0.01)
        out["m3"]["tested"] = bool(out["m3"]["n"] >= 20 and out["m3"]["oc300"] <= 0)
        out["m4"] = _cond(te, te["p"] <= -0.01)
        out["m4"]["tested"] = bool(out["m4"]["n"] >= 20 and out["m4"]["oc300"] >= 0)
        edges = [-1, -0.02, -0.01, -0.003, 0.003, 0.01, 0.02, 1]
        labs = ["≤−2%", "−2～−1%", "−1～−0.3%", "±0.3%", "+0.3～1%", "+1～2%", "≥+2%"]
        te["bk"] = pd.cut(te["p"], edges, labels=labs)
        tb = te.groupby("bk", observed=True).agg(n=("p", "size"), pred=("p", "mean"), gap=("gap50", "mean"), oc=("oc300", "mean"))
        out["buckets"] = [{"bucket": str(k), **{c: float(v) for c, v in r.items()}} for k, r in tb.iterrows()]
        out["edges"], out["labels"] = edges, labs
        out["heavy"] = heavy_context(B)
        return _clean(out)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def heavy_context(B) -> dict:
    """權值股的研究資料背景：最近收盤、漲跌、外資買賣超（張；今日／近 5 日）、近 20 日漲跌。"""
    out = {}
    if "Q_C" not in B.a or not B.meta.get("q_days"):
        return out
    qd = B.meta["q_days"]
    for t, nm, _ in HEAVY:
        if t not in B.tickers:
            continue
        j = list(B.tickers).index(t)
        c = pd.Series(B.a["Q_C"][:, j].astype(float), index=qd).dropna()
        if len(c) < 2:
            continue
        fi = pd.Series(B.a["Q_FI"][:, j].astype(float), index=qd).fillna(0) if "Q_FI" in B.a else pd.Series(dtype=float)
        out[t] = {"name": nm, "date": c.index[-1], "close": float(c.iloc[-1]), "chg": float(c.iloc[-1] / c.iloc[-2] - 1),
                  "r20": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else np.nan,
                  "fi1": float(fi.iloc[-1]) if len(fi) else np.nan, "fi5": float(fi.tail(5).sum()) if len(fi) else np.nan}
    return out


def realized(B, dates) -> dict:
    """日報預估的事後檢驗：指定日期的實際大盤（市值前 50 名市值加權）開盤跳空。"""
    T = _daily_table(B)
    if T.empty:
        return {}
    return {str(d.date()): float(T.at[d, "gap50"]) for d in pd.DatetimeIndex(dates) if d in T.index}


def score(hist: list, B) -> dict:
    """前瞻命中率：日報當天的預估方向 vs 實際開盤（|預估| ≥ 0.3% 的日子）。"""
    hs = [h for h in (hist or []) if h.get("pred_gap") is not None]
    if not hs:
        return {"n": 0}
    rz = realized(B, [h["date"] for h in hs])
    rows = [(h["pred_gap"], rz[h["date"]]) for h in hs if h["date"] in rz]
    if not rows:
        return {"n": 0, "pending": len(hs)}
    p, y = np.array(rows).T
    m = np.abs(p) >= 0.003
    return {"n": int(len(rows)), "n_dir": int(m.sum()), "hit": float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.sum() else np.nan,
            "mae": float(np.mean(np.abs(y - p))), "pending": int(len(hs) - len(rows)),
            "rows": [{"date": h["date"], "pred": h["pred_gap"], "actual": rz.get(h["date"])} for h in hs[-20:]]}


# ================================================================== 盤前即時資料
def _chart(sym: str, rng: str = "1mo", interval: str = "1d", timeout: float = 8) -> dict | None:
    import requests
    r = requests.get(YAHOO + requests.utils.quote(sym, safe=""), params={"range": rng, "interval": interval}, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()["chart"]["result"][0]


def daily_quote(sym: str) -> dict | None:
    """最近一根日線收盤與前一根相比；bar_date＝交易所當地日期。"""
    try:
        m = _chart(sym)
    except Exception:  # noqa: BLE001
        return None
    meta = m.get("meta") or {}
    ts = m.get("timestamp") or []
    cl = ((m.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    pairs = [(t, c) for t, c in zip(ts, cl) if c is not None and np.isfinite(c)]
    if len(pairs) < 2:
        return None
    off = int(meta.get("gmtoffset") or 0)
    px = float(meta.get("regularMarketPrice") or pairs[-1][1])
    prev = float(pairs[-2][1])
    if abs(px - pairs[-1][1]) > 1e-9 and meta.get("regularMarketTime") and int(meta["regularMarketTime"]) > pairs[-1][0] + 6 * 3600:
        prev = float(pairs[-1][1])                                    # 盤中報價比最後一根日線新
    t = int(meta.get("regularMarketTime") or pairs[-1][0])
    return {"sym": sym, "price": px, "prev": prev, "chg": px / prev - 1 if prev else np.nan,
            "time": pd.Timestamp(t, unit="s", tz="UTC").tz_convert(TZ).isoformat(),
            "bar_date": str(pd.Timestamp(pairs[-1][0] + off, unit="s").date()), "spark": [float(c) for _, c in pairs[-22:]]}


def since_tw_close(sym: str, now: pd.Timestamp) -> dict | None:
    """台股收盤（前一個台股交易日 13:30）之後的漲跌（期貨 24 小時交易 → 夜盤方向的代理）。"""
    try:
        m = _chart(sym, "5d", "15m")
    except Exception:  # noqa: BLE001
        return None
    ts = m.get("timestamp") or []
    cl = ((m.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    s = pd.Series(cl, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(TZ), dtype=float).dropna()
    if s.empty:
        return None
    d = now.normalize()
    if now.hour * 60 + now.minute < 13 * 60 + 30:                    # 台股收盤前 → 用前一個台股交易日的收盤
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    t0 = d + pd.Timedelta(hours=13, minutes=30)
    base = s[s.index <= t0]
    if base.empty:
        return None
    last = s.iloc[-1]
    return {"sym": sym, "base_time": base.index[-1].isoformat(), "base": float(base.iloc[-1]), "last": float(last),
            "last_time": s.index[-1].isoformat(), "chg": float(last / base.iloc[-1] - 1)}


def taifex_night(timeout: float = 6) -> dict | None:
    """台指期夜盤（期交所 MIS 盤後交易時段，15:00～05:00）；欄位名稱若改版或連不到 → None（日報改用那指期估算）。"""
    import requests
    try:
        r = requests.post("https://mis.taifex.com.tw/futures/api/getQuoteList",
                          json={"MarketType": "1", "SymbolType": "F", "KindID": "1", "CID": "TXF", "ExpireMonth": "", "RowSize": "全部",
                                "PageNo": "", "SortColumn": "", "AscDesc": "A"}, headers=UA, timeout=timeout)
        r.raise_for_status()
        rows = ((r.json() or {}).get("RtData") or {}).get("QuoteList") or []
    except Exception:  # noqa: BLE001
        return None
    for q in rows:
        last, ref = _f(q.get("CLastPrice")), _f(q.get("CRefPrice"))
        if np.isfinite(last) and last > 0 and np.isfinite(ref) and ref > 0:
            return {"name": str(q.get("DispCName") or "台指期"), "last": last, "ref": ref, "chg": last / ref - 1,
                    "time": str(q.get("CTime") or ""), "src": "期交所 MIS（盤後交易時段）"}
    return None


def news(n: int = 10) -> list[dict]:
    from .news import google_news
    seen, out = set(), []
    for q in NEWS_Q:
        for x in google_news(q, n=6, days=1):
            k = re.sub(r"\W", "", x["title"])[:30]
            if k in seen:
                continue
            seen.add(k)
            tags = [t for t, pat in TAGS if re.search(pat, x["title"], re.I)]
            out.append({"title": x["title"], "link": x["link"], "source": x["source"],
                        "time": x["time"].isoformat() if x.get("time") is not None else None, "tags": tags})
    out.sort(key=lambda x: x["time"] or "", reverse=True)
    return out[:n]


def fetch_live(now: pd.Timestamp | None = None, with_news: bool = True) -> dict:
    now = now or pd.Timestamp.now(tz=TZ)
    syms = {k: s for k, s, _ in US_IDX + MACRO + FUT} | {k: v[0] for k, v in ADR.items()} | {k: k for k in PEERS}
    with ThreadPoolExecutor(max_workers=8) as ex:
        q = dict(zip(syms, ex.map(daily_quote, syms.values())))
        nq, es = ex.submit(since_tw_close, "NQ=F", now), ex.submit(since_tw_close, "ES=F", now)
        tx = ex.submit(taifex_night)
        nw = ex.submit(news) if with_news else None
        live = {"quotes": {k: v for k, v in q.items() if v}, "night": {"NQF": nq.result(), "ESF": es.result(), "TX": tx.result()},
                "news": nw.result() if nw else [], "fetched_at": now.isoformat()}
    return live


# ================================================================== 日報
def report_date(now: pd.Timestamp) -> pd.Timestamp:
    """日報對應的台股交易日：13:30 前＝今天（週末順延到週一），13:30 後＝下一個平日。"""
    d = now.tz_convert(TZ).normalize().tz_localize(None) if now.tzinfo else now.normalize()
    if now.hour * 60 + now.minute >= 13 * 60 + 30:
        d += pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def _pct(x, d=1) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def us_summary(q: dict) -> str:
    idx = [(k, nm) for k, _, nm in US_IDX if k != "VIX" and k in q]
    if not idx:
        return "美股報價暫時取不到。"
    up = sum(1 for k, _ in idx if q[k]["chg"] > 0)
    tone = "全面上漲" if up == len(idx) else "全面下跌" if up == 0 else "漲跌互見"
    parts = "、".join(f"{nm} {_pct(q[k]['chg'], 2)}" for k, nm in idx)
    s = f"美股{tone}：{parts}。"
    if "SOX" in q and "IXIC" in q:
        rel = q["SOX"]["chg"] - q["IXIC"]["chg"]
        if abs(rel) >= 0.01:
            s += f"費半{'強於' if rel > 0 else '弱於'}那指 {abs(rel) * 100:.1f} 個百分點，{'半導體領漲' if rel > 0 else '半導體偏弱'}。"
    if "VIX" in q:
        v = q["VIX"]["price"]
        s += f"VIX {v:.1f}（{'恐慌偏高' if v >= 25 else '偏高' if v >= 20 else '平穩'}）。"
    if "TNX" in q:
        bp = (q["TNX"]["price"] - q["TNX"]["prev"]) * 10      # ^TNX 報價單位＝殖利率 × 10
        s += f"10 年債殖利率 {q['TNX']['price'] / 10:.2f}%（{bp:+.0f}bp）。" if abs(q["TNX"]["price"]) > 20 else \
            f"10 年債殖利率 {q['TNX']['price']:.2f}%（{(q['TNX']['price'] - q['TNX']['prev']) * 100:+.0f}bp）。"
    return s


def predict(model: dict, q: dict) -> dict:
    m1 = (model or {}).get("m1") or {}
    if not m1 or not all(k in q for k in ("SOX", "IXIC", "TSM", "TWD", "VIX")):
        return {}
    x = {"sox": q["SOX"]["chg"], "ixic": q["IXIC"]["chg"], "tsm_twd": (1 + q["TSM"]["chg"]) * (1 + q["TWD"]["chg"]) - 1,
         "dvix": q["VIX"]["price"] - q["VIX"]["prev"]}
    c = m1["coef"]
    g = c["const"] + sum(c[k] * x[k] for k in X1)
    out = {"gap": g, "lo": g - m1["rmse"], "hi": g + m1["rmse"], "x": x, "tested": m1.get("tested", False)}
    lab = None
    for lo, hi, lb in zip(model.get("edges", [])[:-1], model.get("edges", [])[1:], model.get("labels", [])):
        if lo < g <= hi:
            lab = lb
    out["bucket"] = lab
    bk = next((b for b in model.get("buckets") or [] if b["bucket"] == lab), None)
    if bk:
        out["hist"] = bk
    m2 = model.get("m2") or {}
    if m2:
        out["g2330"] = m2["coef"]["const"] + m2["coef"]["tsm_twd"] * x["tsm_twd"]
        out["g2330_tested"] = m2.get("tested", False)
    return out


def heavy_rows(model: dict, q: dict) -> list[dict]:
    ctx = (model or {}).get("heavy") or {}
    usd = q.get("TWD", {}).get("price")
    rows = []
    for t, nm, peers in HEAVY:
        c = ctx.get(t) or {}
        row = {"ticker": t, "name": nm, "close": c.get("close"), "chg": c.get("chg"), "fi1": c.get("fi1"), "fi5": c.get("fi5"),
               "date": c.get("date"), "peers": [{"sym": p, "name": PEERS.get(p, ADR.get(p, ("", p))[1]), "chg": q[p]["chg"]} for p in peers if p in q]}
        for k, (sym, zh, tw, ratio) in ADR.items():
            if tw == t and k in q and usd and c.get("close"):
                imp = q[k]["price"] * usd / ratio
                row["adr"] = {"name": zh, "chg": q[k]["chg"], "implied": imp, "premium": imp / c["close"] - 1}
        rows.append(row)
    return rows


def theme_rows(q: dict, thr: float = 0.015) -> list[dict]:
    out = []
    for nm, syms, tw in THEMES:
        xs = [(s, q[s]["chg"]) for s in syms if s in q and np.isfinite(q[s]["chg"])]
        if not xs:
            continue
        avg = float(np.mean([c for _, c in xs]))
        if abs(avg) >= thr:
            out.append({"theme": nm, "avg": avg, "us": [{"sym": s, "name": PEERS.get(s, s), "chg": c} for s, c in xs], "tw": tw})
    return sorted(out, key=lambda x: -abs(x["avg"]))[:4]


def night_text(night: dict, q: dict | None = None) -> tuple[str, str]:
    tx, nq, es = night.get("TX"), night.get("NQF"), night.get("ESF")
    q = q or {}
    parts = []
    if tx:
        parts.append(f"台指期夜盤 {tx['last']:,.0f}（{_pct(tx['chg'], 2)}，{tx.get('time', '')}）")
    if nq:
        parts.append(f"台股收盤後那指期 {_pct(nq['chg'], 2)}")
    elif "NQF" in q:                                                   # 分時取不到 → 用日線（相對前一日結算）
        parts.append(f"那指期 {_pct(q['NQF']['chg'], 2)}（相對前一日）")
        nq = {"chg": q["NQF"]["chg"]}
    if es:
        parts.append(f"標普期 {_pct(es['chg'], 2)}")
    elif "ESF" in q:
        parts.append(f"標普期 {_pct(q['ESF']['chg'], 2)}（相對前一日）")
    if not parts:
        return "夜盤與期貨報價暫時取不到。", "ref"
    ref = tx["chg"] if tx else (nq or es or {}).get("chg")
    trend = "" if ref is None or not np.isfinite(ref) else ("夜盤偏多" if ref >= 0.005 else "夜盤偏空" if ref <= -0.005 else "夜盤持平")
    return "；".join(parts) + (f" → {trend}" if trend else "") + "。", "ref"


def advice(model: dict, pred: dict, q: dict, res: dict | None) -> dict:
    """保守／積極各 3～5 點；每點附證據等級（tested／ref）。只用 v3.16 第二節通過的規則、模型名單與 v3.15 D1。"""
    g = pred.get("gap") if pred else None
    m3, m4 = (model or {}).get("m3") or {}, (model or {}).get("m4") or {}
    lt = (((res or {}).get("horizons") or {}).get("mid") or {}).get("latest") or {}
    picks = lt.get("picks") or []
    new = [p for p in picks if not p.get("kept")]
    top3 = "、".join(f"{p['ticker']} {p.get('name', '')}" for p in picks[:3])
    news_ = "、".join(f"{p['ticker']} {p.get('name', '')}" for p in new[:4])
    vix = q.get("VIX", {}).get("price")
    cons, aggr = [], []
    cons.append(("配置：大盤型 ETF（0050）約七成＋模型名單約三成；歷史回撤（−30.7%）比只買 0050（−34.0%）淺、年化多約 12 個百分點（v3.15）", "tested"))
    aggr.append(("配置：照中線名單（名次加權，第 1 名資金最多）全部執行；短線名單用限價、避開冷門股（v3.14：短線最怕成交成本）", "tested"))
    if g is not None and np.isfinite(g):
        if g >= 0.01:
            t = m3.get("tested")
            s = f"預估開高 {g * 100:+.1f}%：不追開盤價，新買進掛在買進區或等開盤 30 分鐘後"
            if m3.get("n"):
                s += f"（2024 年後 {m3['n']} 次預估開高 ≥1%，大型股開盤→收盤平均 {m3['oc300'] * 100:+.2f}%、{m3['neg']:.0%} 收在開盤價之下）"
            cons.append((s, "tested" if t else "ref"))
            aggr.append(("開高不追：名單新進股掛買進區，高於追價上限不買；持股可趁開高調節已達賣出條件的部位", "tested" if t else "ref"))
        elif g <= -0.01:
            t = m4.get("tested")
            s = f"預估開低 {g * 100:+.1f}%：不要在開盤恐慌賣出"
            if m4.get("n"):
                s += f"（2024 年後 {m4['n']} 次預估開低 ≤−1%，大型股開盤→收盤平均 {m4['oc300'] * 100:+.2f}%）"
            cons.append((s, "tested" if t else "ref"))
            aggr.append(("開低承接：名單股回到買進區可分 2～3 批承接（歷史上開低後盤中平均回升）", "tested" if t else "ref"))
        else:
            cons.append((f"預估開盤 {g * 100:+.1f}%（平盤附近）：照計畫執行，不必搶開盤", "tested" if pred.get("tested") else "ref"))
            aggr.append(("開盤平穩：名單新進股可在開盤後於買進區分批進場", "ref"))
    if top3:
        cons.append((f"新買進只做中線名單前 3 名（{top3}），分 2～3 批、每批間隔至少半天", "ref"))
    if news_:
        aggr.append((f"今日中線新進：{news_}（模型排名已回測；價位尺、價格適合度僅供參考）", "tested"))
    if vix is not None and np.isfinite(vix) and vix >= 25:
        cons.append((f"VIX {vix:.1f} 偏高：個股部位先降到原計畫的一半，等 VIX 回到 20 以下", "ref"))
        aggr.append((f"VIX {vix:.1f} 偏高：照名單執行但不加碼、不用融資", "ref"))
    cons.append(("跌破警戒價時重新檢視理由（不自動停損；回測顯示機械式停損拖累報酬）", "tested"))
    return {"conservative": [{"text": a, "ev": b} for a, b in cons[:5]], "aggressive": [{"text": a, "ev": b} for a, b in aggr[:5]]}


def headline(pred: dict, q: dict, night: dict) -> list[str]:
    out = []
    if pred:
        out.append(f"預估大盤開盤 {pred['gap'] * 100:+.1f}%（區間 {pred['lo'] * 100:+.1f}%～{pred['hi'] * 100:+.1f}%）"
                   + (f"；台積電 {pred['g2330'] * 100:+.1f}%" if pred.get("g2330") is not None else ""))
    if "SOX" in q:
        out.append(f"費半 {_pct(q['SOX']['chg'], 2)}、那指 {_pct(q.get('IXIC', {}).get('chg'), 2)}、台積電 ADR {_pct(q.get('TSM', {}).get('chg'), 2)}")
    tx = night.get("TX") or night.get("NQF")
    if tx:
        out.append(("台指期夜盤 " if night.get("TX") else "台股收盤後那指期 ") + _pct(tx["chg"], 2))
    return out


def build(model: dict | None, live: dict, res: dict | None = None, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz=TZ)
    q = live.get("quotes") or {}
    night = live.get("night") or {}
    pred = predict(model or {}, q)
    nt, nev = night_text(night, q)
    sec_ok = {"us": "SOX" in q and "GSPC" in q, "night": bool(night.get("TX") or night.get("NQF") or "NQF" in q), "heavy": bool((model or {}).get("heavy")),
              "news": bool(live.get("news")), "link": bool(pred)}
    rep = {"version": VERSION, "date": str(report_date(now).date()), "generated_at": now.isoformat(), "fetched_at": live.get("fetched_at"),
           "us_date": (q.get("GSPC") or q.get("IXIC") or {}).get("bar_date"), "model_asof": (model or {}).get("asof"),
           "status": "complete" if all(sec_ok.values()) else "partial", "sections_ok": sec_ok,
           "headline": headline(pred, q, night),
           "us": {"summary": us_summary(q), "rows": [{"key": k, "name": nm, **{c: q[k].get(c) for c in ("price", "chg", "bar_date")}}
                                                     for k, _, nm in US_IDX + MACRO if k in q], "ev": "ref"},
           "night": {"text": nt, "tx": night.get("TX"), "nq": night.get("NQF"), "es": night.get("ESF"), "ev": nev,
                     "nq_day": {"chg": q["NQF"]["chg"]} if "NQF" in q else None},
           "heavy": {"rows": heavy_rows(model or {}, q), "ev": "ref"},
           "news": {"items": live.get("news") or [], "ev": "ref"},
           "link": {"pred": pred, "themes": theme_rows(q), "m1": (model or {}).get("m1"), "m2": (model or {}).get("m2"),
                    "m3": (model or {}).get("m3"), "m4": (model or {}).get("m4"), "buckets": (model or {}).get("buckets"),
                    "ev": "tested" if pred.get("tested") else "ref"},
           "advice": advice(model or {}, pred, q, res),
           "disclaimer": "盤前日報由程式依公開資料自動產生；預估與建議只用經過歷史檢驗的規則，仍可能失準。僅供研究參考，非投資建議。"}
    return _clean(rep)


def append_hist(hist: list | None, rep: dict, keep: int = 400) -> list:
    """日報歷史：每個日報日期保留最後一次產出的預估（之後用實際開盤檢驗）。"""
    hist = [h for h in (hist or []) if h.get("date") != rep.get("date")]
    pred = (rep.get("link") or {}).get("pred") or {}
    hist.append({"date": rep["date"], "generated_at": rep["generated_at"], "pred_gap": pred.get("gap"), "pred_2330": pred.get("g2330"),
                 "us_date": rep.get("us_date"), "status": rep.get("status")})
    return hist[-keep:]


def dumps(o) -> str:
    return json.dumps(_clean(o), ensure_ascii=False, indent=1)
