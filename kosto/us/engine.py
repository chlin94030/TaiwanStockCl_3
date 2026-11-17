"""v3.4／v3.5 美股每日資料包：母體（標普 500＋那斯達克 100＋費半＋熱門股）→ Yahoo 日線 → 因子 → 參考名單；以及盤中「即時重算」。

v3.5：參考名單改用第二輪研究（docs/v3.5_protocol.md，標普 500 時點母池、無倖存者偏差、基準 SPY）各期間開發期第一名：
短線＝標普 500 成交金額前 100 大裡的「最小波動」、中長線＝前 200 大裡的「平衡動能＋短期反轉」（皆未通過驗收，只供參考）；
另列「動能觀察名單」（風險調整動能，前 200 大）：2022 年後大幅領先、2004–2021 輸 SPY。

資料包（dict，gzip pickle）供 App 使用；每日由 GitHub Actions（us.yml）在美股收盤後產生，
App 找不到或太舊時也可以自己建（約 1～2 分鐘）。
"""
from __future__ import annotations

import gzip
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import model as M
from . import names as NM

VERSION = 2
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
REFERENCE = {"short": ("MINVOL", 100), "mid": ("BAL_REV", 200), "long": ("BAL_REV", 200)}   # (模型, 標普 500 成交金額前 N 大)
WATCH = ("SPMO", 200)                                                     # 動能觀察名單（不是參考名單）
UNI_ZH = {100: "標普 500 前 100 大", 200: "標普 500 前 200 大", 300: "標普 500 前 300 大", 0: "標普 500"}
TAIL = 300                                                              # 保留最近 300 個交易日（即時重算與走勢用）


def research() -> dict:
    p = DATA_DIR / "us_research.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def universe_table(fetch: bool = True, log=print) -> pd.DataFrame:
    """母體：標普 500（每日從公開資料集更新，失敗時用內建快照）＋ EXTRAS 熱門股。"""
    sp = None
    if fetch:
        try:
            import requests
            r = requests.get(SP500_URL, timeout=20)
            r.raise_for_status()
            sp = pd.read_csv(io.StringIO(r.text))
        except Exception as e:  # noqa: BLE001
            if log:
                log(f"  標普 500 名單更新失敗（{str(e)[:60]}），改用內建快照")
    if sp is None:
        sp = pd.read_csv(DATA_DIR / "us_sp500.csv")
    sp = sp.rename(columns={"Symbol": "sym", "Security": "name_en", "GICS Sector": "sector_en", "GICS Sub-Industry": "sub_en"})
    sp["sym"] = sp["sym"].map(D.ysym)
    sp["sp500"] = True
    ex = [t for t in NM.EXTRAS if t not in set(sp["sym"])]
    add = pd.DataFrame({"sym": ex, "name_en": "", "sector_en": "", "sub_en": "", "sp500": False})
    u = pd.concat([sp[["sym", "name_en", "sector_en", "sub_en", "sp500"]], add], ignore_index=True)
    u = u.drop_duplicates("sym")
    u["ndx"] = u["sym"].isin(NM.NDX_NOW)
    u["sox"] = u["sym"].isin(NM.SOX_NOW)
    u["cyber"] = u["sym"].isin(NM.CYBER_NOW)
    u["power"] = u["sym"].isin(NM.POWER_NOW)
    u["sector"] = [NM.sector_zh(s) if s else NM.EXTRA_SECTOR.get(t, "其他") for t, s in zip(u["sym"], u["sector_en"])]
    u["sub"] = [NM.subind_zh(s) for s in u["sub_en"]]
    u["zh"] = u["sym"].map(NM.zh_name)
    u["desc"] = u["sym"].map(NM.zh_desc)
    return u.set_index("sym")


def _drop_partial(P: dict, now=None) -> dict:
    """盤前／盤中建資料包時，Yahoo 日線最後一根是今天還沒收完的 → 去掉。"""
    clk = D.market_clock(now)
    if clk["phase"] in ("pre", "regular") and len(P["C"]):
        today = clk["et"].tz_localize(None).normalize()
        if P["C"].index[-1] >= today:
            P = {k: (v.iloc[:-1] if isinstance(v, pd.DataFrame) else v) for k, v in P.items()}
    return P


def _last_full_row(C: pd.DataFrame) -> int:
    """最後一個「大部分股票都有收盤價」的交易日（避免某天只有少數 ADR 有資料）。"""
    cnt = C.notna().sum(axis=1).to_numpy()
    ok = np.where(cnt >= 0.8 * cnt.max())[0]
    return int(ok[-1]) if len(ok) else len(C) - 1


def ref(B: dict, h: str) -> tuple[str, int]:
    """參考名單的（模型, 前 N 大）；相容 v3.4 資料包（只有模型名稱）。"""
    x = B.get("reference", {}).get(h)
    return (x, 0) if isinstance(x, str) else (x[0], int(x[1] or 0))


def _masks(Cs: pd.DataFrame, DV: pd.DataFrame, info: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sp = info["sp500"].reindex(Cs.columns).fillna(False).astype(bool)
    mem = pd.DataFrame(np.broadcast_to(sp.to_numpy(), Cs.shape), index=Cs.index, columns=Cs.columns)
    allm = pd.DataFrame(True, index=Cs.index, columns=Cs.columns)
    out = {"all": M.sp_universe(Cs, DV, allm), "sp500": M.sp_universe(Cs, DV, mem)}
    for n in (100, 200, 300):
        out[str(n)] = M.sp_universe(Cs, DV, mem, top=n)
    return out


def _mask_key(top: int) -> str:
    return str(top) if top else "sp500"


def build(P: dict | None = None, info: pd.DataFrame | None = None, prev: dict | None = None, log=print,
          fetch_list: bool = True, now=None) -> dict:
    """產生資料包。P＝data.panel() 的結果（沒給就自己抓）；prev＝前一天的資料包（續抱用）。"""
    info = info if info is not None else universe_table(fetch_list, log)
    if P is None:
        P = D.panel(list(info.index) + NM.ETFS, "2y", log=log)
    if not P:
        raise RuntimeError("美股日線全部抓取失敗（Yahoo 可能暫時無法連線）")
    P = _drop_partial(P, now)
    last = _last_full_row(P["C"])
    C, O, V, RAW = (P[k].iloc[:last + 1] for k in ("C", "O", "V", "RAW"))
    H, L = P["H"].iloc[:last + 1], P["L"].iloc[:last + 1]
    etf = [t for t in C.columns if t in NM.ETFS]
    stocks = [t for t in C.columns if t not in NM.ETFS]
    Cs, Vs = C[stocks], V[stocks]
    DV = Cs * Vs
    MK = _masks(Cs, DV, info)
    F = M.factors(Cs, Vs)
    day = Cs.index[-1]
    um = {k: v.iloc[-1] for k, v in MK.items()}
    u_all = um["all"]
    fr = pd.DataFrame({k: M.pct_rank(F[k].iloc[[-1]], MK["all"].iloc[[-1]]).iloc[0] for k in F})
    fv = pd.DataFrame({k: F[k].iloc[-1] for k in F})
    scores = pd.DataFrame({m: M.score({k: F[k].iloc[[-1]] for k in w}, MK["all"].iloc[[-1]], w).iloc[0] for m, w in M.MODELS.items()})
    picks = {}
    plan = dict(REFERENCE)
    plan["watch"] = WATCH
    for h, (mname, top) in plan.items():
        mk = MK[_mask_key(top)].iloc[[-1]]
        sc = M.score({k: F[k].iloc[[-1]] for k in M.MODELS[mname]}, mk, M.MODELS[mname]).iloc[0]
        held = (prev or {}).get("picks", {}).get(h, {}) if prev else {}
        held = held.get("names", []) if held.get("model") == mname and int(held.get("top", -1)) == top else []
        hyst = M.HORIZONS.get(h, M.HORIZONS["mid"])["hyst"]
        names = M.pick(sc, M.TOP_K, held, hyst)
        picks[h] = {"model": mname, "top": top, "names": names, "weights": M.rank_weights(len(names)).round(4).tolist(),
                    "date": str(day.date()), "rank": {t: int(r) for t, r in sc.rank(ascending=False).dropna().items() if r <= 60}}
    r = lambda n: Cs.iloc[-1] / Cs.iloc[-1 - n] - 1 if len(Cs) > n else pd.Series(np.nan, Cs.columns)  # noqa: E731
    vol60 = (Cs / Cs.shift(1) - 1).iloc[-60:].std() * np.sqrt(252)
    summ = pd.DataFrame({"close": RAW[stocks].iloc[-1], "r1": r(1), "r5": r(5), "r20": r(20), "r60": r(60), "r120": r(120),
                         "r252": r(252), "hi52": Cs.iloc[-1] / Cs.iloc[-252:].max(), "ma50": Cs.iloc[-1] / Cs.iloc[-50:].mean() - 1,
                         "ma200": Cs.iloc[-1] / Cs.iloc[-200:].mean() - 1, "vol60": vol60,
                         "adv60": DV.iloc[-60:].mean(), "in_u": u_all})
    summ = summ.join(info[[c for c in ["sector", "sub", "zh", "desc", "name_en", "sp500", "ndx", "sox", "cyber", "power"] if c in info]], how="left")
    log_hist = list((prev or {}).get("picks_log", []))
    for h, p in picks.items():
        if not any(x["date"] == p["date"] and x["h"] == h for x in log_hist):
            log_hist.append({"date": p["date"], "h": h, "model": p["model"], "top": p["top"], "names": p["names"],
                             "px": [float(RAW.at[day, t]) if t in RAW else np.nan for t in p["names"]]})
    log_hist = log_hist[-600:]
    tail = slice(max(0, len(C) - TAIL), len(C))
    B = {"version": VERSION, "asof": str(day.date()), "built": pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
         "tickers": stocks, "etfs": etf, "info": info, "summary": summ, "factor_rank": fr, "factor_value": fv,
         "scores": scores, "universe": u_all, "umask": um, "picks": picks, "picks_log": log_hist,
         "reference": {h: list(v) for h, v in REFERENCE.items()}, "watch": list(WATCH),
         "dates": [str(d.date()) for d in C.index[tail]],
         "C": C.iloc[tail].astype("float32"), "O": O.iloc[tail].astype("float32"), "H": H.iloc[tail].astype("float32"),
         "L": L.iloc[tail].astype("float32"), "V": V.iloc[tail].astype("float32"), "RAW": RAW.iloc[tail].astype("float32"),
         "fail": P.get("fail", [])}
    return B


def dumps(B: dict) -> bytes:
    bio = io.BytesIO()
    with gzip.GzipFile(fileobj=bio, mode="wb", compresslevel=6) as f:
        pickle.dump(B, f, protocol=4)
    return bio.getvalue()


def loads(b: bytes) -> dict:
    with gzip.GzipFile(fileobj=io.BytesIO(b), mode="rb") as f:
        return pickle.load(f)


def mask(B: dict, key: str) -> pd.Series:
    """資料包內某個母池（all／sp500／100／200／300）或成分股集合（ndx／sox／cyber）的遮罩。"""
    tk = B["tickers"]
    um = B.get("umask") or {}
    base = um.get("all", B["universe"]).reindex(tk).fillna(False).astype(bool)
    if key in um:
        return um[key].reindex(tk).fillna(False).astype(bool)
    if key in ("ndx", "sox", "cyber", "power"):
        if key in B["info"]:
            flag = B["info"][key].reindex(tk).fillna(False).astype(bool)
        else:   # 舊資料包沒有這個欄位 → 直接用名單
            lst = {"ndx": NM.NDX_NOW, "sox": NM.SOX_NOW, "cyber": NM.CYBER_NOW, "power": NM.POWER_NOW}[key]
            flag = pd.Series([t in lst for t in tk], index=tk)
        return base & flag
    return base


def live_scores(B: dict, prices: dict[str, float], key: str = "all") -> pd.DataFrame:
    """盤中即時重算：把「目前價格」當作今天收盤，重新計算因子與各模型分數（母池＝key）。
    prices＝{代號: 最新成交價（未還原）}；沒有報價的股票沿用昨收（＝今天不漲不跌）。"""
    C = B["C"][B["tickers"]].astype("float64")
    RAW = B["RAW"][B["tickers"]].astype("float64")
    ratio = (C.iloc[-1] / RAW.iloc[-1]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    last = C.iloc[-1].copy()
    for t, p in prices.items():
        if t in last.index and np.isfinite(p) and p > 0:
            last[t] = p * ratio[t]
    nxt = C.index[-1] + pd.Timedelta(days=1)
    C2 = pd.concat([C, pd.DataFrame([last.values], index=[nxt], columns=C.columns)])
    need = sorted({k for w in M.MODELS.values() for k in w})
    F = M.factors(C2, None, need)
    U1 = pd.DataFrame([mask(B, key).reindex(C.columns).values], index=[nxt], columns=C.columns)
    out = {m: M.score({k: F[k].iloc[[-1]] for k in w}, U1, w).iloc[0] for m, w in M.MODELS.items()}
    df = pd.DataFrame(out)
    df["live_chg"] = last / C.iloc[-1] - 1
    return df


def live_picks(B: dict, prices: dict[str, float]) -> dict:
    """即時名單：各參考名單（＋動能觀察名單）用自己的母池與模型重算；目前名單內的股票仍在前 2.5×10 名就保留。"""
    out = {}
    plan = {h: ref(B, h) for h in B["reference"]}
    if B.get("watch"):
        plan["watch"] = (B["watch"][0], int(B["watch"][1]))
    cache = {}
    for h, (m, top) in plan.items():
        key = _mask_key(top) if B.get("umask") else "all"
        if key not in cache:
            cache[key] = live_scores(B, prices, key)
        S = cache[key]
        held = B["picks"][h]["names"]
        names = M.pick(S[m], M.TOP_K, held, M.HORIZONS.get(h, M.HORIZONS["mid"])["hyst"])
        rk = S[m].rank(ascending=False)
        out[h] = {"names": names, "rank": {t: int(rk[t]) for t in names if np.isfinite(rk.get(t, np.nan))},
                  "new": [t for t in names if t not in held], "out": [t for t in held if t not in names]}
    out["scores"] = cache.get("all") if "all" in cache else next(iter(cache.values()))
    return out


def set_rank(B: dict, key: str, model: str, n: int = 15) -> pd.DataFrame:
    """瀏覽用排行：在某個成分股集合（sp500／ndx／sox／all）內，用某個模型排名（母池內百分位）。"""
    mk = mask(B, key)
    fv = B["factor_value"].reindex(B["tickers"])
    w = M.MODELS[model]
    tot = sum(w.values())
    sc = sum(fv[k].where(mk).rank(pct=True).fillna(0.5).where(mk) * (v / tot) for k, v in w.items())
    sc = sc.dropna().sort_values(ascending=False)
    return pd.DataFrame({"score": sc.head(n), "rank": np.arange(1, min(n, len(sc)) + 1)})


def candidates(B: dict, n: int = 30) -> list[str]:
    """盤前／盤中要抓即時報價的股票：各參考名單與動能觀察名單＋各自前 n 名（約 60～100 檔）。"""
    out = []
    for h, p in B["picks"].items():
        out += p["names"]
        out += [t for t, r in sorted((p.get("rank") or {}).items(), key=lambda x: x[1])[:n]]
    if not any(p.get("rank") for p in B["picks"].values()):        # v3.4 資料包
        s = B["scores"]
        for h in B["reference"]:
            m = ref(B, h)[0]
            out += list(s[m].where(B["universe"].reindex(s.index).fillna(False).astype(bool)).dropna()
                        .sort_values(ascending=False).head(n).index)
    return list(dict.fromkeys(out))


def track(B: dict, h: str, n: int = 12) -> pd.DataFrame:
    """名單追蹤：過去每次名單（依名次加權）之後到最新收盤的報酬，對同期 SPY 與母池等權平均。"""
    C = B["C"]
    rows = []
    logs = [x for x in B.get("picks_log", []) if x["h"] == h][-n:]
    spy = C["SPY"].astype(float) if "SPY" in C.columns else None
    for x in logs:
        d = pd.Timestamp(x["date"])
        if d not in C.index:
            continue
        names = [t for t in x["names"] if t in C.columns]
        if not names:
            continue
        w = M.rank_weights(len(names))
        r = (C[names].iloc[-1] / C.loc[d, names] - 1).to_numpy()
        u = [t for t, ok in mask(B, _mask_key(int(x.get("top", 0) or 0))).items() if ok and t in C.columns]
        ru = (C[u].iloc[-1] / C.loc[d, u] - 1).mean() if u else np.nan
        rs = float(spy.iloc[-1] / spy.loc[d] - 1) if spy is not None else np.nan
        rows.append({"date": x["date"], "model": x["model"], "n": len(names), "ret": float(np.nansum(w * np.nan_to_num(r))),
                     "bench": float(ru), "spy": rs, "days": int((C.index > d).sum())})
    return pd.DataFrame(rows)
