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


def fit(B, split: str = SPLIT, res: dict | None = None) -> dict:
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
        out.update(crash_context(B, res))
        out["link"] = stock_betas(B, res)                                # v3.19 個股對費半的隔夜 β（族群／個股開盤預估）
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


# ================================================================== v3.19 台美連動預測（docs/v3.19_protocol.md 第二節）
LINK_ETF = ["SPY", "XLK", "XLE", "XLF", "XLI", "XLB", "XLY", "XLV", "XLP", "XLU"]
LINK_Q = {"SOX": "SOX", "IXIC": "IXIC", "TSM": "TSM", **{k: k for k in LINK_ETF}}      # 研究訊號 → 盤前報價鍵
_LINK = {}


def link_model() -> dict:
    """2019–2023 估計、2024 後檢驗的族群開盤模型（凍結係數；kosto/data/linkage_v319.json）。"""
    if "m" not in _LINK:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parent / "data" / "linkage_v319.json"
        _LINK["m"] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return _LINK["m"]


def stock_betas(B, res: dict | None = None, top: int = 300, days: int = 250) -> dict:
    """每檔個股開盤跳空對前一晚費半的 β（近 250 日）＋所屬族群；成交金額前 300 名＋名單股＋權值股。"""
    try:
        keys = list(B.meta.get("world_keys") or [])
        if "SOX" not in keys:
            return {}
        Rr, OC = np.asarray(B.a["R"][-days:], float), np.asarray(B.a["OC"][-days:], float)
        gap = np.where(np.isfinite(OC), (1 + Rr) / (1 + OC) - 1, np.nan)
        gap = np.where(np.abs(gap) < 0.105, gap, np.nan)
        sox = pd.Series(np.asarray(B.a["WORLD"], float)[:, keys.index("SOX")]).ffill()
        sx = (sox / sox.shift(1) - 1).to_numpy()[-days:]
        tick = list(B.tickers)
        cap = np.nan_to_num(np.asarray(B.a["CAP"][-1], float))
        want = set(np.array(tick)[np.argsort(-cap)[:top]])
        want |= {t for t, _, _ in HEAVY}
        for hb in ((res or {}).get("horizons") or {}).values():
            want |= {p["ticker"] for p in ((hb.get("latest") or {}).get("picks") or [])}
        groups = list(B.meta.get("groups") or [""] * len(tick))
        out, gsum = {}, {}
        ok_x = np.isfinite(sx) & (np.abs(sx) > 0)
        for j, t in enumerate(tick):
            y = gap[:, j]
            m = ok_x & np.isfinite(y)
            if m.sum() < 120:
                continue
            xv, yv = sx[m], y[m]
            b = float(((xv - xv.mean()) * (yv - yv.mean())).mean() / xv.var())
            gsum.setdefault(groups[j], []).append(b)
            if t in want:
                out[t] = {"beta": round(b, 4), "group": groups[j]}
        return {"stocks": out, "gbeta": {g: round(float(np.mean(v)), 4) for g, v in gsum.items() if len(v) >= 3}}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def link_predict(model: dict, q: dict, tickers: list | None = None) -> dict:
    """前一晚美股 → 大盤、各族群、個股的開盤預估。通過 v3.19 門檻的族群（脊迴歸）與個股相對開盤標 tested，
    其餘族群用「大盤預估 × 族群 β」（ref，附檢驗期相關）。"""
    lm = link_model()
    if not lm:
        return {}
    x = {k: (q.get(v) or {}).get("chg") for k, v in LINK_Q.items()}
    if any(x.get(k) is None or not np.isfinite(x[k]) for k in ("SOX", "IXIC", "TSM")):
        return {}
    mk = lm["market"]
    mkt = mk["const"] + sum(mk[k] * x[k] for k in ("SOX", "IXIC", "TSM"))
    full = all(x.get(k) is not None and np.isfinite(x[k]) for k in lm["signals"])
    groups = {}
    ridge = {}
    for g, m in lm["groups"].items():
        if not m.get("n_test") or m.get("corr") is None or not np.isfinite(m["corr"]):
            continue                                                    # 檢驗期沒有資料的族群不預估
        idx = m["idx_const"] + m["idx_slope"] * mkt
        ridge[g] = m["const"] + sum(m["coef"][k] * x[k] for k in lm["signals"]) if full else idx     # L2 檢驗時用的族群預估
        if m.get("pass") and full:
            p_, ev = ridge[g], "tested"
        else:
            p_, ev = idx, "ref"
        groups[g] = {"pred": float(p_), "ev": ev, "corr": m["corr"] if ev == "tested" else m["corr_idx"], "hit": m.get("hit")}
    lk = (model or {}).get("link") or {}
    st_, gb = lk.get("stocks") or {}, lk.get("gbeta") or {}
    stocks = {}
    for t in tickers or []:
        s_ = st_.get(t)
        if not s_:
            continue
        g = s_.get("group")
        pg = ridge[g] if g in ridge else mkt
        p_ = pg + (s_["beta"] - gb.get(g, s_["beta"])) * x["SOX"]
        stocks[t] = {"pred": float(p_), "rel": float(p_ - mkt), "group": g, "beta": s_["beta"]}
    return {"market": float(mkt), "groups": groups, "stocks": stocks, "full": full, "l2": lm.get("L2", {}).get("test", {}),
            "l3": lm.get("L3", [])}


def taiex_trend() -> dict | None:
    """v3.17：加權指數近一年日線（Yahoo ^TWII）→ 距一年高點、近 20 日；研究資料落後時用來判斷今天的市況。"""
    try:
        m = _chart("^TWII", "1y")
        ts = m.get("timestamp") or []
        cl = ((m.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        off = int((m.get("meta") or {}).get("gmtoffset") or 28800)
        s = pd.Series([c for c in cl], index=[pd.Timestamp(t + off, unit="s").normalize() for t in ts], dtype=float).dropna()
        if len(s) < 30:
            return None
        return {"date": str(s.index[-1].date()), "close": float(s.iloc[-1]), "high250": float(s.tail(250).max()),
                "dd250": float(s.iloc[-1] / s.tail(250).max() - 1), "dd120": float(s.iloc[-1] / s.tail(120).max() - 1),
                "r20": float(s.iloc[-1] / s.iloc[-21] - 1)}
    except Exception:  # noqa: BLE001
        return None


def live_state(model: dict | None, live: dict) -> dict:
    """研究資料的市況（model["state"]）若比今天的加權指數舊，用加權指數更新距高點與急跌狀態（只會更嚴重、不會更樂觀）。"""
    st = dict((model or {}).get("state") or {})
    tw = (live or {}).get("taiex") or {}
    if not tw or tw.get("date", "") <= st.get("date", ""):
        return st
    st["dd250"] = min(float(st.get("dd250", 0) or 0), tw["dd250"])
    st["dd120"] = min(float(st.get("dd120", 0) or 0), tw["dd120"])
    st["r20_live"] = tw["r20"]
    st["panic"] = bool(st.get("panic") or tw["r20"] <= -0.08 or tw["dd120"] <= -0.12)
    st["taiex"], st["taiex_date"], st["taiex_high"] = tw["close"], tw["date"], tw["high250"]
    return st


def fetch_live(now: pd.Timestamp | None = None, with_news: bool = True) -> dict:
    now = now or pd.Timestamp.now(tz=TZ)
    syms = {k: s for k, s, _ in US_IDX + MACRO + FUT} | {k: v[0] for k, v in ADR.items()} | {k: k for k in PEERS} | {k: k for k in LINK_ETF}
    with ThreadPoolExecutor(max_workers=8) as ex:
        q = dict(zip(syms, ex.map(daily_quote, syms.values())))
        nq, es = ex.submit(since_tw_close, "NQ=F", now), ex.submit(since_tw_close, "ES=F", now)
        tx = ex.submit(taifex_night)
        tw = ex.submit(taiex_trend)
        nw = ex.submit(news) if with_news else None
        live = {"quotes": {k: v for k, v in q.items() if v}, "night": {"NQF": nq.result(), "ESF": es.result(), "TX": tx.result()},
                "taiex": tw.result(), "news": nw.result() if nw else [], "fetched_at": now.isoformat()}
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


def crash_context(B, res: dict | None = None) -> dict:
    """v3.17：市況（急跌狀態、去槓桿進度、跌停家數）＋名單與權值股的急跌風險分級、近 5 日跌停次數。"""
    from . import crash as C
    out = {"state": C.market_state(B)}
    risk = C.crash_risk(B)
    tick = {t for t, _, _ in HEAVY}
    for hb in ((res or {}).get("horizons") or {}).values():
        tick |= {p["ticker"] for p in ((hb.get("latest") or {}).get("picks") or [])}
    out["risk"] = {t: {**risk[t], "ld5": C.recent_limit_downs(B, t)} for t in sorted(tick) if t in risk}
    out["weather"] = C.model_weather(B)                                # v3.18 模型天氣
    return out


def live_weather(weather: dict, q: dict, st: dict | None = None) -> dict:
    """v3.18：用今晨的 VIX、費半與今天的加權距高點更新模型天氣的狀態（歷史統計不變）。"""
    from .review import V318_REGIME
    if not weather or not weather.get("rows"):
        return weather or {}
    stat = {(d, s): (m, n, t, sh, lg, wk) for d, s, m, n, t, sh, lg, wk in V318_REGIME}
    rows = [dict(r) for r in weather["rows"]]
    vix = (q.get("VIX") or {}).get("price")
    sox = (q.get("SOX") or {}).get("price")
    w = dict(weather)
    for r in rows:
        new = None
        if r["dim"] == "美股恐慌" and vix is not None and np.isfinite(vix):
            new = "VIX ≥ 30" if vix >= 30 else "VIX 20～30" if vix >= 20 else "VIX < 20"
            w["vix"] = vix
        if r["dim"] == "美股半導體" and sox is not None and np.isfinite(sox) and weather.get("sox_ma50"):
            ma = (weather["sox_ma50"] * 49 + sox) / 50
            new = "費半在 50 日線之上" if sox > ma else "費半在 50 日線之下"
            w["sox_vs_ma50"] = sox / ma - 1
        if r["dim"] == "台股趨勢" and st and st.get("taiex_date") and (st.get("dd250") or 0) <= -0.15:
            new = "0050 在 200 日線之下"
        if r["dim"] == "台股市況" and st and C_regime(st) == "crash":
            new = "急跌"
        if new and new != r["state"]:
            m, n, t, sh, lg, wk = stat.get((r["dim"], new), (None, None, None, None, None, False))
            r.update({"state": new, "mid": m, "days": n, "t": t, "short": sh, "long": lg, "weak": bool(wk), "live": True})
    w["rows"] = rows
    w["weak"] = sum(r["weak"] for r in rows)
    if w.get("vix") is not None and w.get("sox_vs_ma50") is not None:
        w["us_off"] = bool(w["sox_vs_ma50"] < 0 and w["vix"] >= 25)
    return w


def C_regime(st: dict) -> str:
    from . import crash as C
    return C.regime(st)[0]


def news_themes(items: list) -> list[dict]:
    """新聞依主題歸納：每個主題的則數與代表標題（最新一則）。"""
    groups = {}
    for x in items or []:
        for t in (x.get("tags") or ["其他"]):
            groups.setdefault(t, []).append(x)
    out = [{"tag": t, "n": len(v), "top": v[0]["title"], "link": v[0].get("link"), "source": v[0].get("source")} for t, v in groups.items()]
    return sorted(out, key=lambda r: (r["tag"] == "其他", -r["n"]))


def _p(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def analysis(model: dict, pred: dict, q: dict, night: dict, res: dict | None, news_items: list) -> dict:
    """v3.17 盤前評析：把數字與新聞整理成「今天代表什麼、該做什麼、要看什麼」。規則產生，每段標證據等級。"""
    from . import crash as C
    from .review import V317_MODEL, V317_TIERS, V317_TWO
    st = (model or {}).get("state") or {}
    code, zh = C.regime(st) if st else ("normal", "正常")
    g = pred.get("gap") if pred else None
    vix = (q.get("VIX") or {}).get("price")
    paras = []
    # ① 市況定位
    if st:
        if st.get("taiex_date"):                                       # 研究資料落後：用今天的加權指數
            t = (f"加權指數 {st['taiex_date']} 收 {st['taiex']:,.0f}，距一年高點 {_p(st.get('dd250'))}、近 20 日 {_p(st.get('r20_live'))}，"
                 f"屬於「{zh}」" + (f"（研究資料到 {st['date']}，融資與跌停家數是那天的數字）。" if st.get("date") else "。"))
        else:
            t = (f"研究資料 {st.get('date', '')}：0050 距一年高點 {_p(st.get('dd250'))}、近 20 日 {_p(st.get('r20'))}，屬於「{zh}」。")
        if st.get("margin_chg120") is not None:
            t += f"融資餘額較 120 日高點 {_p(st['margin_chg120'], 0)}"
            if code == "crash":
                t += "（歷史上跌到 −28% 左右的 3 次崩跌，見底時融資減 48～52%；2024 年的快速崩跌減 25% 就見底）"
            t += "。"
        if st.get("limit_down5"):
            t += f"近 5 日跌停家數 {'、'.join(str(x) for x in st['limit_down5'])}。"
        if st.get("foreign_streak"):
            fs = st["foreign_streak"]
            t += f"外資連 {abs(fs)} 日{'買' if fs > 0 else '賣'}超，近 5 日 {st.get('foreign5', 0):+,.0f} 億元。"
        if code == "crash":
            an = C.analogs(st.get("dd250", 0))
            if an:
                t += "歷史上 0050 也跌到這個深度的：" + "；".join(
                    f"{a['peak'][:4]} 年（最深 {a['dd']:.0f}%，見底後 60 日 {a['r60']:+.0f}%）" if a.get("r60") is not None else f"{a['peak'][:4]} 年（最深 {a['dd']:.0f}%）"
                    for a in an) + "。每一次都在 1～15 個月內回到前高，但不保證這次一樣。"
        paras.append({"title": "市況定位", "text": t, "ev": "ref"})
    # ② 今晨美股的意義
    if pred:
        x = pred.get("x") or {}
        hb = pred.get("hist") or {}
        t = (f"前一夜費半 {_p(x.get('sox'), 2)}、那指 {_p(x.get('ixic'), 2)}、台積電 ADR（台幣計）{_p(x.get('tsm_twd'), 2)} → "
             f"預估大盤開盤 {_p(g)}（區間 {_p(pred.get('lo'))}～{_p(pred.get('hi'))}），台積電 {_p(pred.get('g2330'))}。")
        if hb:
            t += f"同一級預估在 2024 年後實際開盤平均 {_p(hb.get('gap'), 2)}、開盤後到收盤 {_p(hb.get('oc'), 2)}。"
        if g is not None and g >= 0.01:
            t += "開高的日子盤中通常回落，不追開盤價。"
        elif g is not None and g <= -0.01:
            t += "開低的日子盤中通常回升一些，開盤恐慌賣出最吃虧。"
        tx = night.get("TX") or night.get("NQF")
        if tx and tx.get("chg") is not None:
            t += f"{'台指期夜盤' if night.get('TX') else '台股收盤後那指期'} {_p(tx['chg'], 2)}。"
        paras.append({"title": "今晨美股對台股的意義", "text": t, "ev": "tested" if pred.get("tested") else "ref"})
    # ③ 消息面
    th = news_themes(news_items)
    if th or vix is not None:
        t = ""
        if th:
            t += "消息集中在" + "、".join(f"{r['tag']}（{r['n']} 則）" for r in th[:3] if r["tag"] != "其他") + "。" if any(r["tag"] != "其他" for r in th[:3]) else ""
            tops = list(dict.fromkeys(r["top"] for r in th))[:2]
            t += "代表標題：" + "；".join(f"「{x_}」" for x_ in tops) + "。"
        if vix is not None and np.isfinite(vix):
            t += f"VIX {vix:.1f}（{'恐慌' if vix >= 30 else '偏高' if vix >= 20 else '平穩'}）"
            if vix >= 35:
                t += "：歷史上 VIX 衝上 35 之後「回到 30 以下」才是較可靠的止穩訊號。"
            else:
                t += "。"
        if "TNX" in q:
            tn = q["TNX"]["price"]
            t += f"美 10 年債殖利率 {tn / 10 if tn > 20 else tn:.2f}%。"
        paras.append({"title": "消息與風險溫度", "text": t, "ev": "ref"})
    # ④ 名單怎麼看
    risk = (model or {}).get("risk") or {}
    lt = (((res or {}).get("horizons") or {}).get("mid") or {}).get("latest") or {}
    picks = lt.get("picks") or []
    flags = []
    for pk in picks:
        r = risk.get(pk["ticker"]) or {}
        if r.get("tier") == "高" or (r.get("ld5") or 0) >= 2:
            flags.append({"ticker": pk["ticker"], "name": pk.get("name", ""), "tier": r.get("tier"), "ld5": r.get("ld5", 0),
                          "kept": bool(pk.get("kept"))})
    t = ""
    if code == "crash":
        t += (f"模型是動能型選股：歷史 5 次崩跌中平均跌幅和 0050 一樣（{V317_MODEL['avg_crash'][0]:.0f}%），但最近兩次跌得比 0050 多，"
              f"低點後 60 日平均 {V317_MODEL['avg_reb60'][0]:+.0f}%、0050 {V317_MODEL['avg_reb60'][1]:+.0f}%：剛開始反彈時，0050 通常比名單股反彈得快。")
    hi = [f for f in flags if f["tier"] == "高"]
    if hi:
        th_ = V317_TIERS[0]
        t += ("名單中急跌風險「高」（融資使用率與波動都在前段）：" + "、".join(f"{f['ticker']} {f['name']}" for f in hi)
              + f"；歷史崩跌中這一級平均跌 {abs(th_[1]):.0f}%、{th_[3]:.0f}% 出現 2 次以上跌停 → 急跌期間不新買、已持有不加碼。")
    ld = [f for f in flags if f["ld5"] >= 2]
    if ld:
        t += ("近 5 日已有 2 根以上跌停：" + "、".join(f"{f['ticker']} {f['name']}（{f['ld5']} 根）" for f in ld)
              + f"。歷史上急跌中連續 2 根跌停後，隔天再跌停的機率 {V317_TWO['next_ld']:.0f}%；但 20 日內平均反彈 {V317_TWO['a20']:.0f}%"
              f"（比 0050 多 {V317_TWO['r20']:.1f} 個百分點），60 日後有 {100 - V317_TWO['p_r60_pos']:.0f}% 落後 0050 → 不在跌停價排隊殺出，"
              "等打開跌停後的反彈再決定減碼或換成 0050；不要攤平。")
    wx = live_weather((model or {}).get("weather") or {}, q, st)
    weak = [r for r in wx.get("rows") or [] if r.get("weak") and r.get("mid") is not None]
    if weak:
        t += (f"模型天氣：目前 {len(weak)} 個面向處於模型弱勢狀態（" + "、".join(f"{r['state']} {r['mid']:+.0f}%" for r in weak)
              + "，歷史上中線名單在這些狀態年化落後 0050）；這時動能因子幾乎失效（VIX≥30 時 IC 轉負），營收成長、短期反轉較有效。"
              "要不要把新資金偏向 0050 由你決定：v1.7 檢驗過依美股狀態把個股減半，回撤少 6pp、但報酬少 2.3pp。")
        if wx.get("us_off"):
            t += "費半跌破 50 日線且 VIX≥25（美股風險關閉）：觀察名單 W7「動能減半」正用新資料前瞻追蹤。"
    if t:
        paras.append({"title": "模型名單怎麼看", "text": t, "ev": "ref"})
    # ⑤ 資金與部位
    if code in ("crash", "correction"):
        t = ("資金：歷史 6 次崩跌中，0050 從跌 15% 開始買，第一筆之後平均再跌 9%（最多 21%，2022 年），一年後全部為正。"
             "剩下的資金分 3～4 份，用時間（每 20 個交易日一份）或反轉訊號分批，不要一天用完；先補 0050 等大盤型 ETF，再補名單股；"
             "絕對不用融資（斷頭會在最低點被迫賣出）。")
        paras.append({"title": "資金與部位", "text": t, "ev": "ref"})
    # 今日一句話
    if code == "crash":
        if g is not None and g >= 0.01:
            one = "急跌中的反彈日：開高不追，持股可趁反彈調節急跌風險高、連續跌停過的股票；現金照分批計畫。"
        elif g is not None and g <= -0.01:
            one = "急跌延續、今晨美股偏空：不搶開盤、不在跌停價殺出、不用融資；手上現金今天最多動用一份。"
        else:
            one = "急跌中的平盤日：先看跌停家數是否明顯減少；現金照分批計畫，名單股只買急跌風險低、中的。"
    elif code == "correction":
        one = "修正期：照名單執行但不追高，現金分批、優先 0050；看融資與外資是否止血。"
    else:
        one = "市況正常：照模型名單與計畫執行，開盤不追價。" if g is None or g < 0.01 else "市況正常、預估開高：不追開盤價，掛買進區。"
    return {"regime": [code, zh], "oneliner": one, "paragraphs": paras, "watch": C.reversal_checklist(st, vix) if st else [],
            "flags": flags, "themes": th, "weather": wx}


def advice(model: dict, pred: dict, q: dict, res: dict | None) -> dict:
    """保守／積極各 3～5 點；每點附證據等級（tested／ref）。只用 v3.16 第二節通過的規則、模型名單與 v3.15 D1。"""
    g = pred.get("gap") if pred else None
    m3, m4 = (model or {}).get("m3") or {}, (model or {}).get("m4") or {}
    lt = (((res or {}).get("horizons") or {}).get("mid") or {}).get("latest") or {}
    picks = lt.get("picks") or []
    from . import crash as C
    st0 = (model or {}).get("state") or {}
    crash_now = bool(st0) and C.regime(st0)[0] == "crash"
    rk = (model or {}).get("risk") or {}
    risky = {p["ticker"] for p in picks if (rk.get(p["ticker"]) or {}).get("tier") == "高" or ((rk.get(p["ticker"]) or {}).get("ld5") or 0) >= 2}
    usable = [p for p in picks if not (crash_now and p["ticker"] in risky)]     # v3.17 急跌時排除高風險、連續跌停的名單股
    new = [p for p in usable if not p.get("kept")]
    top3 = "、".join(f"{p['ticker']} {p.get('name', '')}" for p in usable[:3])
    news_ = "、".join(f"{p['ticker']} {p.get('name', '')}" for p in new[:4])
    vix = q.get("VIX", {}).get("price")
    cons, aggr = [], []
    cons.append(("配置：大盤型 ETF（0050）約七成＋模型名單約三成；歷史回撤（−30.7%）比只買 0050（−34.0%）淺、年化多約 12 個百分點（v3.15）", "tested"))
    aggr.append((("配置：照中線名單（名次加權）執行，但跳過下一點列出的高風險股；短線名單用限價" if crash_now else
                   "配置：照中線名單（名次加權，第 1 名資金最多）全部執行；短線名單用限價、避開冷門股（v3.14：短線最怕成交成本）"), "tested"))
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
    if vix is not None and np.isfinite(vix) and vix >= 25 and crash_now:   # 急跌中：不在低點砍倉，只放慢新買
        cons.append((f"VIX {vix:.1f} 偏高：新買進的個股只用原計畫一半金額；等 VIX 衝上 35 後回到 30 以下（歷史上最可靠的止穩訊號）再補足", "ref"))
        aggr.append((f"VIX {vix:.1f} 偏高：照名單執行但不加碼、不用融資", "ref"))
    elif vix is not None and np.isfinite(vix) and vix >= 25:
        cons.append((f"VIX {vix:.1f} 偏高：個股部位先降到原計畫的一半，等 VIX 回到 20 以下", "ref"))
        aggr.append((f"VIX {vix:.1f} 偏高：照名單執行但不加碼、不用融資", "ref"))
    if crash_now:                                                      # v3.17 急跌時的紀律（歷史統計，非選股規則）
        cons.insert(1, ("急跌期間：不用融資；剩下的現金分 3～4 份，每 20 個交易日或出現反轉訊號才動用一份，優先 0050", "ref"))
        aggr.insert(1, ("急跌期間名單股只新買急跌風險低、中、近 5 日沒有連續跌停的" + (f"（暫不新買：{'、'.join(sorted(risky))}）" if risky else "")
                        + "；連續跌停的持股不在跌停價殺出，等反彈再處理", "ref"))
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
    if model:
        model = {**model, "state": live_state(model, live)}             # v3.17 研究資料落後時，市況用今天的加權指數
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
           "analysis": analysis(model or {}, pred, q, night, res, live.get("news") or []),
           "disclaimer": "盤前日報由程式依公開資料自動產生；預估與建議只用經過歷史檢驗的規則，仍可能失準。僅供研究參考，非投資建議。"}
    rep["groups"] = link_section(model or {}, q, res)                   # v3.19 族群與個股開盤預估
    ns = news_shock_text(q, res)                                         # v3.21 重大消息（輝達單晚大漲大跌）
    if ns:
        rep["analysis"].setdefault("paragraphs", []).insert(0, {"title": "重大消息", "text": ns, "ev": "ref"})
        rep["analysis"]["news_shock"] = True
    gt = rep["groups"].get("text")
    if gt:
        for p_ in rep["analysis"].get("paragraphs") or []:
            if p_["title"] == "今晨美股對台股的意義":
                p_["text"] += gt
                break
        else:
            rep["analysis"].setdefault("paragraphs", []).insert(1, {"title": "今晨美股對台股的意義", "text": gt.lstrip(), "ev": "tested"})
    return _clean(rep)


def news_shock_text(q: dict, res: dict | None, thr: float = 0.05) -> str:
    """v3.21：輝達前一晚漲跌 ≥ 5% 時，說明這類消息歷史上在台股怎麼走（每日研究的消息日研究）與該怎麼做。"""
    nv = (q.get("NVDA") or {}).get("chg")
    if nv is None or not np.isfinite(nv) or abs(nv) < thr:
        return ""
    ix = (q.get("IXIC") or {}).get("chg")
    sh = (res or {}).get("shock") or {}
    b = (sh.get("linkage") or {}).get("b_nasdaq")
    t = f"輝達前一晚 {nv * 100:+.1f}%" + (f"（那指 {ix * 100:+.1f}%，扣掉大盤後輝達特有約 {(nv - (b or 1.5) * ix) * 100:+.1f}%）" if ix is not None and np.isfinite(ix) else "") + "。"
    side = "利多（輝達特有上漲）" if nv > 0 else "利空（輝達特有下跌）"
    sm = next((r for r in ((sh.get("study") or {}).get("summary") or []) if r.get("side") == side), None)
    if sm:
        t += (f"歷史上同方向的輝達特有消息日 {sm['n']} 次：跟輝達連動高的股票比連動低的，當天多反應 {sm['d0'] * 100:+.1f}%，"
              f"之後 5 日 {sm['d5'] * 100:+.1f}%、20 日 {sm['d20'] * 100:+.1f}%（正＝延續、負＝回吐）。")
    gs = [g["group"] for g in ((sh.get("linkage") or {}).get("groups") or []) if str(g["group"]).startswith("電子")][:4]
    if gs:
        t += "連動最高的族群：" + "、".join(gs) + "。"
    t += ("消息大多在開盤一次反映（v3.19：族群跳空後當天與 5 日內幾乎不延續）→ 不追漲停開盤、也不在跌停價殺出；"
          "中長線名單不因單一消息改變，名單股照原本的買進區與續抱規則。")
    return t


def link_section(model: dict, q: dict, res: dict | None) -> dict:
    """v3.19：族群開盤預估（前 5 高／前 5 低）、名單股與權值股的預估開盤（相對大盤）、開盤後延續的歷史。"""
    names = {t: nm for t, nm, _ in HEAVY}
    order = []
    for h in ("mid", "short", "long"):
        for p_ in ((((res or {}).get("horizons") or {}).get(h) or {}).get("latest") or {}).get("picks") or []:
            names.setdefault(p_["ticker"], p_.get("name", ""))
            if p_["ticker"] not in order:
                order.append(p_["ticker"])
    order += [t for t, _, _ in HEAVY if t not in order]
    lp = link_predict(model, q, order)
    if not lp:
        return {}
    gs = sorted(({"group": g, **v} for g, v in lp["groups"].items()), key=lambda r: -r["pred"])
    pk = {p_["ticker"] for hb in ((res or {}).get("horizons") or {}).values() for p_ in ((hb.get("latest") or {}).get("picks") or [])}
    st_ = [{"ticker": t, "name": names.get(t, ""), **lp["stocks"][t], "pick": t in pk} for t in order if t in lp["stocks"]]
    up = [r for r in gs if r["pred"] >= 0.003][:3]
    dn = [r for r in gs[::-1] if r["pred"] <= -0.003][:3]
    txt = ""
    if up or dn:
        txt = "族群（成分股平均，口徑和 0050 不同）：" + ("；".join(x for x in (
            ("預估開高較多 " + "、".join(f"{r['group']}{'' if r['ev'] == 'tested' else '*'} {r['pred'] * 100:+.1f}%" for r in up)) if up else "",
            ("開低較多 " + "、".join(f"{r['group']}{'' if r['ev'] == 'tested' else '*'} {r['pred'] * 100:+.1f}%" for r in dn)) if dn else "") if x))
        txt += "（* 為全市場預估×族群 β，未單獨通過檢驗）。歷史上族群跳空在開盤一次反映：預估開高 ≥1% 的族群開盤後到收盤平均只比大盤少 0.1%、5 日內幾乎沒有延續；預估開低 ≤−1% 的族群盤中反而平均強 0.3%。"
    return {"market": lp["market"], "groups": gs, "stocks": st_, "l2": lp.get("l2"), "l3": lp.get("l3"), "full": lp["full"],
            "text": (" " + txt) if txt else "", "ev_stock": "tested"}


def append_hist(hist: list | None, rep: dict, keep: int = 400) -> list:
    """日報歷史：每個日報日期保留最後一次產出的預估（之後用實際開盤檢驗）。"""
    hist = [h for h in (hist or []) if h.get("date") != rep.get("date")]
    pred = (rep.get("link") or {}).get("pred") or {}
    hist.append({"date": rep["date"], "generated_at": rep["generated_at"], "pred_gap": pred.get("gap"), "pred_2330": pred.get("g2330"),
                 "us_date": rep.get("us_date"), "status": rep.get("status")})
    return hist[-keep:]


def dumps(o) -> str:
    return json.dumps(_clean(o), ensure_ascii=False, indent=1)
