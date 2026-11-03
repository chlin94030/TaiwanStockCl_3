"""v2.0 精選 ETF：上市／上櫃 ETF 的含息報酬、年化報酬、殖利率、配息頻率、波動與回撤。

報酬計算與個股相同：日報酬 = 收盤 / 參考價 − 1；除息日的參考價用交易所公告的除息參考價（等於配息再投入），
分割、反分割由交易所的漲跌價差（相對參考價）自動處理。這裡的結果只用在 ETF 頁，不進選股模型。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

WIN = {"1月": 21, "3月": 63, "6月": 126, "1年": 252, "3年": 756, "5年": 1260}

OVERSEAS = re.compile(r"美國|美股|標普|S&P|SP500|那斯達克|NASDAQ|費城|費半|道瓊|日本|日經|東證|中國|滬深|上証|上證|深證|恒生|香港|印度|"
                      r"越南|歐洲|德國|全球|國際|亞洲|新興|韓國|泰國|印尼|東協|北美|FANG|科技巨頭|元宇宙|電動車|未來車|網路|資安|"
                      r"基因|免疫|生技|醫療|太空|衛星|5G|航運龍頭|美元|美債|公司債|投等|非投等|新興債|金融債|電信債|銀行債|納斯達克|ARK", re.I)


def classify(code: str, name: str) -> str:
    n = name or ""
    if code.endswith("L") or code.endswith("R") or re.search(r"正2|正二|反1|反一|槓桿|反向|2倍|-1倍", n):
        return "槓桿反向"
    if code.endswith("B") or re.search(r"債", n):
        return "債券"
    if code.endswith("U") or re.search(r"期貨|原油|黃金|白銀|黃豆|農產|美元指數|VIX", n):
        return "期貨商品"
    if code.endswith("A") or n.startswith("主動"):
        return "主動式"
    if OVERSEAS.search(n) and not re.search(r"台灣|臺灣|台股|臺股", n):
        return "海外股票"
    return "國內股票"


def returns(px: pd.DataFrame, div: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """px: date ticker close change event amount → (日報酬 wide, 收盤 wide)。"""
    if px is None or px.empty:
        return pd.DataFrame(), pd.DataFrame()
    px = px.drop_duplicates(["date", "ticker"], keep="last")
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    chg = px.pivot(index="date", columns="ticker", values="change").reindex_like(close)
    ref = close - chg
    if div is not None and not div.empty:
        d = div.drop_duplicates(["date", "ticker"], keep="last")
        exref = d.pivot(index="date", columns="ticker", values="ref").reindex(index=close.index, columns=close.columns)
        ref = ref.where(exref.isna(), exref)
    prev = close.ffill().shift(1)
    ref = ref.where(ref.notna() | close.isna(), prev)                 # 缺漲跌價差（X）→ 前一日收盤
    ratio = ref / prev
    ref = ref.where(((ratio > 0.02) & (ratio < 50)) | prev.isna())
    r = (close / ref - 1)
    r = r.where(r.abs() < 0.6)                                         # 異常值（資料錯誤）視為缺
    return r.astype(float), close.astype(float)


def _ann(tot: float, days: int) -> float:
    if not np.isfinite(tot) or days <= 0 or tot <= -1:
        return np.nan
    return (1 + tot) ** (252.0 / days) - 1


def metrics(r: pd.DataFrame, close: pd.DataFrame, px: pd.DataFrame, div: pd.DataFrame | None, asof=None) -> pd.DataFrame:
    """每檔 ETF 一列：各期間含息報酬（3、5 年為年化）、成立以來年化、波動、回撤、殖利率、配息頻率、成交金額。"""
    if r.empty:
        return pd.DataFrame()
    asof = pd.Timestamp(asof) if asof is not None else close.index[-1]
    r, close = r.loc[:asof], close.loc[:asof]
    names = px.drop_duplicates("ticker", keep="last").set_index("ticker")
    amt = px.pivot(index="date", columns="ticker", values="amount").reindex_like(close)
    tr = (1 + r.fillna(0)).cumprod()
    rows = []
    year0 = pd.Timestamp(asof.year, 1, 1)
    dv = div.copy() if div is not None and not div.empty else pd.DataFrame(columns=["date", "ticker", "before", "ref"])
    if len(dv):
        dv["cash"] = dv["before"] - dv["ref"]
    for t in close.columns:
        c = close[t]
        live = c.dropna()
        if len(live) < 15 or live.index[-1] < asof - pd.Timedelta(days=10):
            continue                                                   # 已下市或停止交易
        first = live.index[0]
        n = int(c.loc[first:].notna().sum())
        idx = tr[t].loc[first:]
        row = {"ticker": t, "name": str(names["name"].get(t, "")), "board": str(names["board"].get(t, "")),
               "close": float(live.iloc[-1]), "days": n, "since": str(first.date())}
        row["type"] = classify(t, row["name"])
        for k, w in WIN.items():
            if n > w:
                tot = float(idx.iloc[-1] / idx.iloc[-1 - w] - 1)
                row[k] = _ann(tot, w) if w >= 756 else tot
            else:
                row[k] = np.nan
        y = idx[idx.index < year0]
        row["今年"] = float(idx.iloc[-1] / y.iloc[-1] - 1) if len(y) else np.nan
        row["成立年化"] = _ann(float(idx.iloc[-1] / idx.iloc[0] - 1), n) if n >= 252 else np.nan
        rr = r[t].loc[first:].dropna()
        row["波動"] = float(rr.tail(252).std() * np.sqrt(252)) if len(rr) > 60 else np.nan
        seg = idx.tail(756)
        row["最大回撤"] = float((seg / seg.cummax() - 1).min()) if len(seg) > 20 else np.nan
        row["近期回落"] = float(idx.iloc[-1] / idx.tail(252).max() - 1)
        if "0050" in r and t != "0050":
            a, b = r[t].tail(252), r["0050"].tail(252)
            ok = a.notna() & b.notna()
            with np.errstate(invalid="ignore", divide="ignore"):
                row["相關0050"] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 60 else np.nan
        else:
            row["相關0050"] = 1.0 if t == "0050" else np.nan
        row["成交額"] = float(amt[t].tail(60).mean() / 1e8) if t in amt else np.nan   # 億元／日
        d = dv[dv["ticker"] == t] if len(dv) else dv
        d12 = d[(d["date"] > asof - pd.Timedelta(days=365)) & (d["date"] <= asof)] if len(d) else d
        row["配息次數"] = int(len(d12))
        row["殖利率"] = float(d12["cash"].sum() / live.iloc[-1]) if len(d12) and live.iloc[-1] > 0 else 0.0
        row["配息"] = bool(len(d[d["date"] > asof - pd.Timedelta(days=550)]) > 0) if len(d) else False
        row["頻率"] = freq_label(len(d12), row["配息"], n)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["大盤型"] = (out["type"] == "國內股票") & (out["相關0050"] >= 0.95)       # 和 0050 幾乎同步 → 適合當核心
    out["長期年化"] = out["3年"].where(out["3年"].notna(), out["1年"])
    out["長期依據"] = np.where(out["3年"].notna(), "近3年", np.where(out["1年"].notna(), "近1年", "未滿1年"))
    return out.sort_values("長期年化", ascending=False, na_position="last").reset_index(drop=True)


def freq_label(n12: int, pays: bool, days: int) -> str:
    if n12 >= 10:
        return "月配"
    if n12 >= 3:
        return "季配"
    if n12 == 2:
        return "半年配"
    if n12 == 1:
        return "年配"
    if pays:
        return "配息"
    return "不配息" if days >= 280 else "未滿1年"


def picks(m: pd.DataFrame, kind: str, n: int = 10, min_amt: float = 0.1, min_days: int = 252) -> pd.DataFrame:
    """精選清單：kind = 大盤型／配息／不配息／債券／槓桿反向；排除成交額太小、上市未滿 1 年；依長期年化（近3年，不足以近1年）排序。"""
    if m is None or m.empty:
        return pd.DataFrame()
    x = m[(m["days"] >= min_days) & (m["成交額"].fillna(0) >= min_amt)]
    if kind == "大盤型":
        x = x[x["大盤型"].fillna(False).astype(bool)]
    elif kind == "債券":
        x = x[x["type"] == "債券"]
    elif kind == "槓桿反向":
        x = x[x["type"] == "槓桿反向"]
    else:
        x = x[~x["type"].isin(["債券", "槓桿反向", "期貨商品"])]
        x = x[x["配息"]] if kind == "配息" else x[~x["配息"]]
    return x.sort_values("長期年化", ascending=False).head(n)


def pack(px: pd.DataFrame, div: pd.DataFrame | None, keep_days: int = 1300) -> tuple[dict, dict]:
    """給 bundle：ETF_TR（含息淨值，近 keep_days 日）＋ meta（代號、名稱、指標表、配息紀錄）。"""
    r, close = returns(px, div)
    if r.empty:
        return {}, {}
    m = metrics(r, close, px, div)
    if m.empty:
        return {}, {}
    keep = m["ticker"].tolist()
    tr = (1 + r[keep].fillna(0)).cumprod().where(close[keep].ffill().notna())
    tr = tr.iloc[-keep_days:]
    tr = tr / tr.bfill().iloc[0]
    meta = {"etf_tickers": keep, "etf_days": [str(d.date()) for d in tr.index],
            "etf_table": m.replace({np.nan: None}).to_dict("records"), "etf_asof": str(close.index[-1].date())}
    if div is not None and not div.empty:
        d = div[div["ticker"].isin(keep) & (div["date"] >= close.index[-1] - pd.Timedelta(days=800))].copy()
        d["cash"] = (d["before"] - d["ref"]).round(4)
        meta["etf_divs"] = [[str(x.date()), t, float(c)] for x, t, c in zip(d["date"], d["ticker"], d["cash"])]
    return {"ETF_TR": tr.to_numpy(np.float32)}, meta
