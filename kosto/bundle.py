"""Research bundle: everything the optimiser/backtester needs, precomputed once (GitHub Actions)
and loaded by Streamlit for seconds-level re-tuning.

Arrays (S = signal dates, D = trading days from first signal, N = tickers ever in a pool, F = factors)
  U      uint16 [S,N,F]  universe percentile rank of each factor among eligible stocks (0 = missing)
  FWD    float32[3,S,N]  forward return open(t+1) -> open(t+H+1) for H = 5/20/60 (NaN = not matured)
  R      float32[D,N]    daily total return (close/ref - 1)
  OC     float32[D,N]    open -> close return (NaN = no open)
  OK     bool   [D,N]    buyable at the open (traded, not limit-up open)
  RANKS  int16  [P,S,N]  pool rank per rule (1 = best, 0 = not eligible)
  STAGE  int8   [S,N]    stock egg stage (0 unknown, 1..6 = A1..B3)
  DTOP   bool   [S,N]    M double-top flag ; WSUP bool [S,N] W support / stabilisation flag
  TRADE  bool   [S,N]    traded at the signal close ; CAP float32[S,N] 20-day average traded value
  BENCH  float32[D]      0050 daily return ; MKT float32[S,7] market stage scores + code
  WORLD  float32[D,K]    v1.7 國際市場收盤（美股/匯率＝美國日期 ≤ T−1，日韓＝T 日；meta.world_keys）
  VOL60  float32[S,N]    60 日報酬波動（波動倒數加權用） ; meta.groups = 產業鏈族群
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import HORIZON_KEYS, STAGE_CODES, STAGES
from .factors import REGISTRY, chain_groups, compute, rsi14
from .kostolany import market_regime, mw_patterns, stage_codes
from .panel import Panel
from .optimize import avg_rank_cols
from .universe import RULES, pool_inputs, pool_ranks


@dataclass
class Bundle:
    meta: dict
    a: dict = field(default_factory=dict)

    # convenience
    @property
    def sig_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.meta["sig_dates"])

    @property
    def days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.meta["days"])

    @property
    def tickers(self) -> list:
        return self.meta["tickers"]

    @property
    def factors(self) -> list:
        return self.meta["factors"]

    @property
    def sig_pos(self) -> np.ndarray:
        return np.asarray(self.meta["sig_pos"], int)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, __meta__=np.frombuffer(json.dumps(self.meta, ensure_ascii=False).encode(), np.uint8),
                            **self.a)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)

    @staticmethod
    def load(path) -> "Bundle":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["__meta__"]).decode())
            a = {k: z[k] for k in z.files if k != "__meta__"}
        return Bundle(meta, a)


def signal_positions(dates: pd.DatetimeIndex, start, step: int) -> np.ndarray:
    last = len(dates) - 1
    first = int(dates.searchsorted(pd.Timestamp(start)))
    pos = np.arange(last, first - 1, -step)[::-1]
    return pos[pos >= 0]


def _universe_rank(x: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Percentile rank among eligible & finite values → uint16 1..65535 (0 = missing)."""
    out = np.zeros(x.shape, np.uint16)
    for i in range(x.shape[0]):
        m = elig[i] & np.isfinite(x[i])
        n = int(m.sum())
        if n < 5:
            continue
        r = avg_rank_cols(x[i, m][:, None].astype(float))[:, 0]
        out[i, m] = (1 + np.round(r * 65534 / max(n - 1, 1))).astype(np.uint16)
    return out


def forward_returns(R: np.ndarray, OC: np.ndarray, sig_rel: np.ndarray, H: int) -> np.ndarray:
    """open(t+1) → open(t+H+1) total return for each signal (rows) and ticker (cols)."""
    D, N = R.shape
    L = np.vstack([np.zeros((1, N)), np.cumsum(np.log1p(np.nan_to_num(R, nan=0.0).astype(float)), axis=0)])
    out = np.full((len(sig_rel), N), np.nan, np.float32)
    for k, t in enumerate(sig_rel):
        if t + H >= D:
            continue
        e = t + 1
        oc_in = OC[e].astype(float)
        if t + H + 1 < D:
            x = t + H + 1
            oc_out = np.where(np.isfinite(OC[x]), OC[x], np.nan).astype(float)
            grow = np.exp(L[x + 1] - L[e + 1])               # R[e+1..x]
            val = (1 + oc_in) * grow / np.where(np.isfinite(oc_out), 1 + oc_out, 1 + R[x])
            # not traded at exit open → hold to previous close (gap 0)
            val = np.where(np.isfinite(oc_out), val, (1 + oc_in) * np.exp(L[x] - L[e + 1]))
        else:  # exit open not observed yet → not matured
            continue
        out[k] = (val - 1).astype(np.float32)
    return out


MKT_COLS = ["open", "high", "low", "close", "amount", "foreign", "trust", "dealer", "mbal", "sbal", "up", "down",
            "mflow", "sflow",   # mbal/sbal＝餘額張數×收盤價（市值估算）；mflow/sflow＝當日張數增減×收盤價（不受股價漲跌影響）
            # v2.0 市場解讀：融資張數、站上 20／60 日線比例、20 日新高／新低家數、大盤本益比／殖利率、國際市場
            "mlots", "ab20", "ab60", "hi20", "lo20", "pe_mkt", "dy_mkt", "sox", "vix", "nasdaq", "tsm", "twd",
            # v2.0 上市官方合計（億元／張；回補完成前為空）
            "o_foreign", "o_trust", "o_dealer", "o_margin", "o_mlots"]


def market_pack(P: Panel, qdays, series: pd.DataFrame | None = None) -> tuple[np.ndarray, list]:
    """加權指數開高低收（官方；缺時只有收盤）、成交金額（億，官方；缺時用個股加總）、
    三大法人買賣超與融資券餘額（億元，個股加總估算）、上漲／下跌家數。"""
    out = pd.DataFrame(index=qdays, columns=MKT_COLS, dtype=float)
    close = P.m["close"].astype(float)
    if P.index_daily is not None and len(P.index_daily):
        ix = P.index_daily.reindex(qdays)
        for c in ("open", "high", "low", "close"):
            if c in ix:
                out[c] = ix[c]
        if "amount" in ix:
            out["amount"] = ix["amount"] / 1e8
    if P.taiex is not None:
        out["close"] = out["close"].fillna(P.taiex.reindex(qdays))
    stock_amt = (P.m["amount"].astype(float).sum(axis=1, min_count=1) / 1e8).reindex(qdays)
    out["amount"] = out["amount"].fillna(stock_amt)
    px = close.ffill()
    for c in ("foreign", "trust", "dealer"):
        if c in P.m:
            out[c] = ((P.m[c].astype(float) * px).sum(axis=1, min_count=1) / 1e8).reindex(qdays)
    for c, fl in (("mbal", "mflow"), ("sbal", "sflow")):
        if c in P.m:
            lots = P.m[c].astype(float)
            out[c] = ((lots * 1000 * px).sum(axis=1, min_count=1) / 1e8).reindex(qdays)
            out[fl] = ((lots.diff() * 1000 * px).sum(axis=1, min_count=1) / 1e8).reindex(qdays)
    r = P.m["ret"].astype(float).where(P.m["tradable"])
    out["up"] = (r > 0).sum(axis=1).reindex(qdays)
    out["down"] = (r < 0).sum(axis=1).reindex(qdays)
    ms = series if series is not None else _market_series(P)
    for c in ("mlots", "ab20", "ab60", "hi20", "lo20", "pe_mkt", "dy_mkt", "sox", "vix", "nasdaq", "tsm", "twd"):
        if c in ms:
            out[c] = ms[c].reindex(qdays)
    mo = getattr(P, "mkt_off", None)
    if mo is not None and len(mo):
        mo = mo.reindex(qdays)
        for c, src, sc in (("o_foreign", "foreign", 1e-8), ("o_trust", "trust", 1e-8), ("o_dealer", "dealer", 1e-8),
                           ("o_margin", "margin_amt", 1e-8), ("o_mlots", "margin_lots", 1.0)):
            if src in mo:
                out[c] = mo[src].astype(float) * sc
    return out.to_numpy(np.float32), list(MKT_COLS)


def _market_series(P):
    from .market import market_series
    return market_series(P)


FIN_COLS = ["revenue", "gross", "op", "net", "eps"]


def fin_pack(fin: pd.DataFrame | None, tickers, n: int = 8) -> tuple[np.ndarray | None, list]:
    """v2.1 每季財報 → 單季數字（累計值相減），最近 n 季：[n, 檔數, 5]（營收、毛利、營業利益、淨利 元；EPS 元）。"""
    if fin is None or fin.empty:
        return None, []
    f = fin.drop_duplicates(["ticker", "yq"], keep="last").copy()
    f = f[f["ticker"].isin(set(tickers))]
    if f.empty:
        return None, []
    qs = sorted(f["yq"].unique())
    # 至少 5% 公司已申報的季別（期限前先申報的公司也能看到最新一季）
    cnt = f.groupby("yq")["ticker"].nunique()
    qs = [q for q in qs if cnt.get(q, 0) >= 0.05 * cnt.max()]
    out = np.full((n, len(tickers), len(FIN_COLS)), np.nan, np.float32)
    pos = {t: j for j, t in enumerate(tickers)}
    cum = {c: f.pivot(index="ticker", columns="yq", values=c) for c in FIN_COLS if c in f}
    keep = qs[-n:]
    for k, q in enumerate(keep):
        y, s_ = divmod(int(q), 10)
        prev = y * 10 + s_ - 1 if s_ > 1 else None
        for ci, c in enumerate(FIN_COLS):
            if c not in cum or q not in cum[c]:
                continue
            v = cum[c][q]
            if prev is not None:
                pv = cum[c][prev] if prev in cum[c] else pd.Series(np.nan, index=v.index)
                v = v - pv
            idx = [pos[t] for t in v.index if t in pos]
            out[k + (n - len(keep)), idx, ci] = v.reindex([t for t in v.index if t in pos]).to_numpy(np.float32)
    labels = [None] * (n - len(keep)) + [int(q) for q in keep]
    return out, labels


def data_dates(P: Panel) -> dict:
    """v2.0 各資料集最新日期（App 顯示「分析依據」與過期警示）。"""
    def last_valid(df):
        try:
            v = df.notna().any(axis=1) if isinstance(df, pd.DataFrame) else df.notna()
            return str(v[v].index[-1].date()) if v.any() else None
        except Exception:  # noqa: BLE001
            return None
    out = {"price": str(P.dates[-1].date())}
    for k in ("institutional", "margin", "valuation"):
        out[k] = P.quality.get(f"{k}_last")
    if P.rev_m is not None and len(P.rev_m):
        ok = P.rev_m.notna().sum(axis=1)
        ym = int(ok[ok > 0.5 * ok.max()].index[-1])
        out["revenue"] = f"{ym // 100}-{ym % 100:02d}"
    out["world"] = P.quality.get("world_last")
    if P.index_daily is not None and len(P.index_daily):
        out["index"] = str(P.index_daily.index[-1].date())
    if P.etf_px is not None and len(P.etf_px):
        out["etf"] = str(pd.Timestamp(P.etf_px["date"].max()).date())
    if getattr(P, "fin", None) is not None and len(P.fin):
        q = int(P.fin["yq"].max())
        out["fin"] = f"{q // 10}Q{q % 10}"
    if getattr(P, "mkt_off", None) is not None and len(P.mkt_off):
        out["mkt"] = str(P.mkt_off.index[-1].date())
    return out


def build_bundle(P: Panel, cfg: dict, log=print) -> Bundle:
    t0 = time.time()
    r = cfg["research"]
    pos = signal_positions(P.dates, r["start"], r["signal_step"])
    sig_dates = P.dates[pos]
    log(f"訊號日 {len(pos)} 個：{sig_dates[0].date()} → {sig_dates[-1].date()}")
    vals, ctx = compute(P, sample_idx=pos)
    log(f"因子計算 {len(vals)} 個 ({time.time()-t0:.0f}s)")
    inp = pool_inputs(ctx, pos, cfg)
    ranks = pool_ranks(inp, sig_dates)
    maxsize = max(cfg["pools"]["sizes"])
    ever = np.zeros(len(P.tickers), bool)
    for rk in ranks.values():
        ever |= ((rk >= 1) & (rk <= maxsize)).any(axis=0)
    cols = np.where(ever)[0]
    log(f"曾進入任一母池的股票 {len(cols)} 檔")
    elig = inp["elig"]
    fnames = list(vals.keys())
    U = np.zeros((len(pos), len(cols), len(fnames)), np.uint16)
    for j, f in enumerate(fnames):
        U[:, :, j] = _universe_rank(vals[f], elig)[:, cols]

    d0 = pos[0]
    days = P.dates[d0:]
    R = P["ret"].to_numpy(dtype=np.float32)[d0:, cols]
    OC = P["oc"].to_numpy(dtype=np.float32)[d0:, cols]
    gap = P["gap"].to_numpy(dtype=np.float32)[d0:, cols]
    trad = P["tradable"].to_numpy()[d0:, cols]
    OK = trad & np.isfinite(OC) & ~(gap >= 0.095)
    sig_rel = pos - d0
    FWD = np.stack([forward_returns(R, OC, sig_rel, cfg["horizons"][h]["days"]) for h in HORIZON_KEYS])

    stg = stage_codes(ctx).to_numpy()[pos][:, cols]
    mw = mw_patterns(ctx)
    dtop = mw["double_top"].to_numpy(dtype=bool)[pos][:, cols]
    adj = ctx.adj.to_numpy(dtype=float)
    ma5 = ctx.ma(5).to_numpy(dtype=float)
    mom5 = ctx.mom(5).to_numpy(dtype=float)
    stab = (adj > ma5) & (mom5 > 0)
    wsup = (mw["w_support"].to_numpy(dtype=bool) | stab)[pos][:, cols]
    mk = market_regime(ctx)
    mkt = mk[list(STAGES)].to_numpy(dtype=np.float32)[pos]
    mcode = np.array([STAGE_CODES.get(s, 0) for s in mk["stage"].to_numpy()[pos]], np.float32)
    bench = P.bench.reindex(days).fillna(0).to_numpy(np.float32) if P.bench is not None else np.zeros(len(days), np.float32)

    info = P.info.reindex(P.tickers[cols]).astype(object)
    meta = {
        "version": __version__, "built": pd.Timestamp.now(tz="Asia/Taipei").isoformat(timespec="seconds"),
        "synthetic": bool(P.synthetic), "sig_dates": [str(d.date()) for d in sig_dates],
        "days": [str(d.date()) for d in days], "sig_pos": sig_rel.tolist(), "tickers": list(P.tickers[cols]),
        "names": info["name"].fillna("").tolist(), "sectors": info["sector"].fillna("NA").tolist(),
        "boards": info["board"].fillna("").tolist(),
        "factors": fnames, "families": [REGISTRY[f].family for f in fnames],
        "directions": [REGISTRY[f].direction for f in fnames], "horizons": [list(REGISTRY[f].horizons) for f in fnames],
        "factor_zh": [REGISTRY[f].zh for f in fnames], "rules": list(RULES),
        "horizon_days": {h: cfg["horizons"][h]["days"] for h in HORIZON_KEYS},
        "data_quality": P.quality, "config": cfg,
        "coverage": {f: float((U[:, :, j] > 0).mean()) for j, f in enumerate(fnames)},
    }
    a = {"U": U, "FWD": FWD, "R": R, "OC": OC, "OK": OK,
         "RANKS": np.stack([ranks[rl][:, cols] for rl in RULES]).astype(np.int16),
         "STAGE": stg.astype(np.int8), "DTOP": dtop, "WSUP": wsup,
         "TRADE": trad[sig_rel], "CAP": inp["amt20"][:, cols].astype(np.float32),
         "CLOSE": inp["close"][:, cols].astype(np.float32),
         "MCAP": inp["mcap"][:, cols].astype(np.float32),
         # raw overheat indicators at each signal date (v1.3 不追高閘門: RSI14 and 20-day bias)
         "RSI14": rsi14(ctx).to_numpy(dtype=np.float32)[pos][:, cols],
         "BIAS20": (ctx.adj / ctx.ma(20) - 1).to_numpy(dtype=np.float32)[pos][:, cols],
         "BENCH": bench, "MKT": np.column_stack([mkt, mcode]).astype(np.float32),
         "MKT_DAILY": mk[list(STAGES) + ["confidence", "index", "drawdown", "turnover_ratio", "breadth"]]
         .reindex(days).to_numpy(np.float32),
         "MKT_DAILY_CODE": np.array([STAGE_CODES.get(s, 0) for s in mk["stage"].reindex(days).fillna("UNKNOWN")], np.int8)}
    # ---- v1.7 國際市場（已對齊到台股 T 日收盤後可知）、60 日波動、產業鏈族群
    W = P.world if P.world is not None else pd.DataFrame(index=P.dates)
    meta["world_keys"] = [str(k) for k in W.columns]
    a["WORLD"] = W.reindex(days).to_numpy(np.float32) if W.shape[1] else np.zeros((len(days), 0), np.float32)
    a["VOL60"] = (P["ret"].astype(float).where(P["tradable"]).rolling(60, min_periods=40).std()
                  .to_numpy(dtype=np.float32)[pos][:, cols])
    meta["groups"] = ctx.get("chain_groups", lambda: chain_groups(P)).reindex(P.tickers[cols]).fillna("其他").tolist()
    # ---- 看盤資料包（App 的 K 線／籌碼圖用）：最近 chart_days 個交易日的原始價量與籌碼，單位：價(元)、量與籌碼(張)
    nq = int((cfg.get("app") or {}).get("chart_days", 500))
    q0 = max(0, len(P.dates) - nq)
    qdays = P.dates[q0:]

    def qa(name, scale=1.0):
        if name not in P.m:
            return np.full((len(qdays), len(cols)), np.nan, np.float32)
        return (P.m[name].to_numpy(dtype=np.float32)[q0:, cols] * scale).astype(np.float32)
    for key, name, sc in (("Q_O", "open", 1), ("Q_H", "high", 1), ("Q_L", "low", 1), ("Q_C", "close", 1),
                          ("Q_V", "volume", 1e-3), ("Q_FI", "foreign", 1e-3), ("Q_TR", "trust", 1e-3),
                          ("Q_DL", "dealer", 1e-3), ("Q_MB", "mbal", 1), ("Q_SB", "sbal", 1), ("Q_ML", "mlimit", 1),
                          ("Q_FR", "foreign_ratio", 1), ("Q_H400", "big400", 1), ("Q_H1000", "big1000", 1),
                          ("Q_HR", "retail50", 1), ("Q_HN", "holders", 1),
                          ("Q_RF", "ref", 1)):                     # v2.3.2 參考價（除權息日≠前一日收盤）→ 漲跌幅與官方一致
        a[key] = qa(name, sc)
    for key, name in (("Q_PE", "pe"), ("Q_PB", "pb"), ("Q_DY", "dy")):      # v1.8 估值歷史（基本面快照）
        a[key] = qa(name)
    if P.rev_m is not None and len(P.rev_m):                                  # v1.8 近 25 個月月營收（元）
        rm = P.rev_m.iloc[-25:]
        a["REV_M"] = rm.reindex(columns=P.tickers[cols]).to_numpy(np.float32)
        meta["rev_ym"] = [int(x) for x in rm.index]
    # ---- v2.0 市場解讀（止穩檢查表、修正事件統計、反彈候選）與 ETF；都不進選股模型
    from . import etf as etflib
    from . import market as mklib
    ms = mklib.market_series(P)
    a["Q_MKT"], meta["q_mkt_cols"] = market_pack(P, qdays, ms)             # v1.9 大盤 K 線＋市場籌碼（億元）
    try:
        meta["mkt_stats"] = mklib.history_stats(ms)
        meta["crash_stats"] = mklib.crash_stats(ms["close"], ms.get("sox"))          # v2.2 急跌後的歷史基準
        rr = vals.get("f_rev_rel")
        meta["rebound"] = mklib.rebound_candidates(P, rr[-1] if rr is not None else None)
        meta["rebound_asof"] = str(P.dates[-1].date())
    except Exception as e:  # noqa: BLE001 - optional analysis
        log(f"市場解讀統計失敗（不影響選股）：{e}")
    try:
        ea, em = etflib.pack(P.etf_px, P.etf_div) if P.etf_px is not None else ({}, {})
        a.update(ea)
        meta.update(em)
        if em:
            log(f"ETF {len(em.get('etf_tickers', []))} 檔，資料至 {em.get('etf_asof')}")
    except Exception as e:  # noqa: BLE001
        log(f"ETF 整理失敗（不影響選股）：{e}")
    try:                                                                   # v2.1 每季財報（同族群比較、基本面短評）
        fq, fl = fin_pack(getattr(P, "fin", None), list(P.tickers[cols]))
        if fq is not None:
            a["FIN_Q"], meta["fin_q"], meta["fin_cols"] = fq, fl, FIN_COLS
    except Exception as e:  # noqa: BLE001
        log(f"財報整理失敗（不影響選股）：{e}")
    meta["data_dates"] = data_dates(P)
    a["Q_TAIEX"] = (P.taiex.reindex(qdays).to_numpy(np.float32) if P.taiex is not None
                    else np.full(len(qdays), np.nan, np.float32))
    meta["q_days"] = [str(d.date()) for d in qdays]
    # raw values of factors at the latest signal date (for explaining today's picks)
    a["LATEST_RAW"] = np.stack([vals[f][-1, cols] for f in fnames], axis=1).astype(np.float32)
    log(f"Bundle 完成 ({time.time()-t0:.0f}s)：S={len(pos)} D={len(days)} N={len(cols)} F={len(fnames)}")
    return Bundle(meta, a)
