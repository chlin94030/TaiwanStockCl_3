"""Factor registry: 技術面 T / 籌碼面 C / 基本面 F / 科斯托蘭尼 K.

Every factor is computed from information available at the close of date t (no look-ahead):
rolling windows end at t, revenue is mapped by its legal filing deadline, and nothing uses
returns after t. `direction` is the economic prior (+1 higher is better, 0 = let data decide);
the optimiser never flips a prior sign — if training data disagree, the factor weight goes to 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .panel import Panel, revenue_daily

F32 = np.float32


@dataclass(frozen=True)
class Factor:
    name: str
    family: str          # T C F K
    direction: int       # +1 / 0
    horizons: tuple      # subset of ("short","mid","long")
    zh: str
    func: Callable


REGISTRY: dict[str, Factor] = {}
ALL_H = ("short", "mid", "long")


def factor(name, family, direction, horizons, zh):
    def deco(fn):
        REGISTRY[name] = Factor(name, family, direction, tuple(horizons), zh, fn)
        return fn
    return deco


# ------------------------------------------------------------------ cached intermediates
class Ctx:
    """Lazy cache of shared intermediate matrices (float64 for numerical safety)."""

    def __init__(self, P: Panel):
        self.P = P
        self._c = {}

    def get(self, key, fn):
        if key not in self._c:
            v = fn()
            if isinstance(v, pd.DataFrame) and len(v.dtypes) and (v.dtypes == np.float64).all():
                v = v.astype(np.float32)  # halves peak memory on the full market (≈1,900 × 2,400)
            self._c[key] = v
        return self._c[key]

    @property
    def adj(self):
        return self.get("adj", lambda: self.P["adj"].astype(float).ffill())

    @property
    def scale(self):  # adjusted / raw price ratio (for adjusting high/low)
        return self.get("scale", lambda: self.adj / self.P["close"].astype(float).ffill())

    @property
    def hi(self):
        return self.get("hi", lambda: (self.P["high"].astype(float) * self.scale).fillna(self.adj))

    @property
    def lo(self):
        return self.get("lo", lambda: (self.P["low"].astype(float) * self.scale).fillna(self.adj))

    @property
    def vol(self):
        return self.get("vol", lambda: self.P["volume"].astype(float).where(self.P["tradable"]))

    def ma(self, n):
        return self.get(f"ma{n}", lambda: self.adj.rolling(n, min_periods=max(2, int(n * 0.8))).mean())

    def vma(self, n):
        return self.get(f"vma{n}", lambda: self.vol.rolling(n, min_periods=max(2, int(n * 0.6))).mean())

    def amt(self, n):
        return self.get(f"amt{n}", lambda: self.P["amount"].astype(float).where(self.P["tradable"])
                        .rolling(n, min_periods=max(2, int(n * 0.6))).mean())

    def mom(self, n, skip=0):
        return self.get(f"mom{n}_{skip}", lambda: self.adj.shift(skip) / self.adj.shift(n) - 1)

    def rsum(self, name, n):
        return self.get(f"sum_{name}{n}", lambda: self.P[name].astype(float).rolling(n, min_periods=max(1, n // 2)).sum())

    @property
    def mbal(self):
        return self.get("mbal", lambda: self.P["mbal"].astype(float))

    @property
    def shares(self):
        return self.get("shares", lambda: self.P["shares"].astype(float).where(self.P["shares"] > 0))


def ema(df, span):
    return df.ewm(span=span, adjust=False, min_periods=span).mean()


# ================================================================== 技術面 Technical
@factor("t_mom20", "T", 1, ("mid", "long"), "20日動能")
def _t_mom20(c): return c.mom(20)


@factor("t_mom60", "T", 1, ("mid", "long"), "60日動能")
def _t_mom60(c): return c.mom(60)


@factor("t_mom120x20", "T", 1, ("mid", "long"), "中期動能(120日扣近20日)")
def _t_mom120(c): return c.mom(120, 20)


@factor("t_rs120", "T", 1, ("mid", "long"), "相對0050強度(120日)")
def _t_rs120(c): return c.mom(120)   # cross-sectional rank = rank of (stock − 0050) 120-day return (V23.1 rs_0050_120)


@factor("t_rev5", "T", 1, ("short",), "短期反轉(近5日跌多)")
def _t_rev5(c): return -c.mom(5)


@factor("t_ma_align", "T", 1, ALL_H, "均線多頭排列")
def _t_ma_align(c):
    a = c.adj
    s = ((a > c.ma(5)).astype(float) + (c.ma(5) > c.ma(10)).astype(float) + (c.ma(10) > c.ma(20)).astype(float)
         + (c.ma(20) > c.ma(60)).astype(float)) / 4
    return s + 5 * (c.ma(20) / c.ma(20).shift(5) - 1)


@factor("t_above_ma60", "T", 1, ("mid", "long"), "站上季線幅度")
def _t_ma60(c): return c.adj / c.ma(60) - 1


@factor("t_high20", "T", 1, ("short", "mid"), "接近20日新高")
def _t_high20(c): return c.adj / c.hi.rolling(20, min_periods=15).max()


@factor("t_high252", "T", 1, ("long",), "接近一年新高")
def _t_high252(c): return c.adj / c.adj.rolling(252, min_periods=120).max()


@factor("t_kd", "T", 1, ("short",), "KD(9) K-D")
def _t_kd(c):
    lo9, hi9 = c.lo.rolling(9, min_periods=9).min(), c.hi.rolling(9, min_periods=9).max()
    rsv = ((c.adj - lo9) / (hi9 - lo9).where(hi9 > lo9)).fillna(0.5)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return (k - d).where(c.adj.notna())


@factor("t_macd", "T", 1, ("short", "mid"), "MACD柱狀體")
def _t_macd(c):
    m = ema(c.adj, 12) - ema(c.adj, 26)
    return (m - ema(m, 9)) / c.adj


@factor("t_rsi_health", "T", 1, ("short", "mid"), "RSI健康區(接近60)")
def _t_rsi(c):
    d = c.adj.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 * up / (up + dn).where(up + dn > 0)
    return -(rsi - 60).abs()


@factor("t_lowvol", "T", 1, ("mid", "long"), "低波動")
def _t_lowvol(c): return -c.P["ret"].astype(float).where(c.P["tradable"]).rolling(60, min_periods=40).std()


@factor("t_lowidio", "T", 1, ("mid", "long"), "低特異波動(60日，扣除大盤)")
def _t_lowidio(c):
    """V23.1 residual_lowvol60: volatility of the part of daily returns not explained by the equal-weight market."""
    ret = c.P["ret"].astype(float).where(c.P["tradable"])
    mkt = ret.mean(axis=1)
    mm, mv = mkt.rolling(60, min_periods=40).mean(), mkt.rolling(60, min_periods=40).var()
    beta = (ret.mul(mkt, axis=0).rolling(60, min_periods=40).mean() - ret.rolling(60, min_periods=40).mean().mul(mm, axis=0)).div(mv, axis=0)
    return -ret.sub(beta.mul(mkt, axis=0)).rolling(60, min_periods=40).std()


@factor("t_vol_surge", "T", 0, ("short",), "量能放大(5/60日量比)")
def _t_vsurge(c): return np.log(c.vma(5) / c.vma(60))


# ================================================================== 籌碼面 Chips
def _per_volume(c, col, n):
    return c.rsum(col, n) / (c.vma(20) * n)


@factor("c_foreign5", "C", 1, ("short", "mid"), "外資5日買超(占量)")
def _c_f5(c): return _per_volume(c, "foreign", 5)


@factor("c_foreign20", "C", 1, ALL_H, "外資20日買超(占量)")
def _c_f20(c): return _per_volume(c, "foreign", 20)


@factor("c_trust5", "C", 1, ("short", "mid"), "投信5日買超(占量)")
def _c_t5(c): return _per_volume(c, "trust", 5)


@factor("c_trust20", "C", 1, ALL_H, "投信20日買超(占量)")
def _c_t20(c): return _per_volume(c, "trust", 20)


@factor("c_dealer5", "C", 0, ("short",), "自營商5日買賣超")
def _c_d5(c): return _per_volume(c, "dealer", 5)


@factor("c_inst60", "C", 1, ("mid", "long"), "三大法人60日累計買超(占股本)")
def _c_i60(c): return c.rsum("total", 60) / c.shares


@factor("c_inst_streak", "C", 1, ("short", "mid"), "外資/投信連續買超天數")
def _c_streak(c):
    x = (c.P["foreign"].astype(float) + c.P["trust"].astype(float)).to_numpy()
    out = np.zeros_like(x)
    run = np.zeros(x.shape[1])
    for t in range(x.shape[0]):
        v = x[t]
        run = np.where(v > 0, np.maximum(run, 0) + 1, np.where(v < 0, np.minimum(run, 0) - 1, run))
        out[t] = run
    return pd.DataFrame(np.clip(out, -10, 10), index=c.P.dates, columns=c.P.tickers).where(c.adj.notna())


@factor("c_margin_chg20", "C", 1, ("short", "mid"), "融資20日減少(占量)")
def _c_mchg(c):
    m = c.mbal
    return -((m - m.shift(20)) * 1000 / (c.vma(20) * 20))


@factor("c_short_ratio", "C", 0, ("short", "mid"), "券資比")
def _c_sr(c): return c.P["sbal"].astype(float) / c.mbal.where(c.mbal > 0)


@factor("c_foreign_hold", "C", 1, ("mid", "long"), "外資持股比率60日變化")
def _c_fh(c):
    if "foreign_ratio" not in c.P.m:
        return pd.DataFrame(np.nan, index=c.P.dates, columns=c.P.tickers)
    fr = c.P["foreign_ratio"].astype(float)
    return fr - fr.shift(60)


@factor("c_inst2y", "C", 1, ("long",), "近2年法人累計買超(占市值)")
def _c_i2y(c):
    val = (c.P["total"].astype(float) * c.P["close"].astype(float).ffill()).rolling(500, min_periods=120).sum()
    return val / c.P["mcap"].astype(float).rolling(250, min_periods=60).mean()


# ================================================================== 基本面 Fundamental
def _rev(c, fn, key):
    return c.get(key, lambda: revenue_daily(c.P, fn).astype(float))


def _yoy(r, n=1):
    s = r.rolling(n, min_periods=n).sum()
    return (s / s.shift(12) - 1).where(s.shift(12) > 0).clip(-1, 5)


@factor("f_rev_yoy", "F", 1, ("mid", "long"), "月營收年增率")
def _f_yoy(c): return _rev(c, lambda r: _yoy(r, 1), "rev_yoy1")


@factor("f_rev_yoy3", "F", 1, ("mid", "long"), "近3月營收年增率")
def _f_yoy3(c): return _rev(c, lambda r: _yoy(r, 3), "rev_yoy3")


@factor("f_rev_accel", "F", 1, ("mid", "long"), "營收成長加速")
def _f_acc(c): return _rev(c, lambda r: _yoy(r, 3) - _yoy(r, 12), "rev_acc")


@factor("f_rev_streak", "F", 1, ("mid", "long"), "月營收連續年增月數")
def _f_streak(c):
    def fn(r):
        y = _yoy(r, 1).to_numpy(dtype=float)
        out = np.full_like(y, np.nan)
        run = np.zeros(y.shape[1])
        for i in range(len(y)):
            v = y[i]
            ok = np.isfinite(v)
            run = np.where(ok, np.where(v > 0, run + 1, 0), 0)
            out[i] = np.where(ok, np.minimum(run, 12), np.nan)
        return pd.DataFrame(out, index=r.index, columns=r.columns)
    return _rev(c, fn, "rev_streak")


@factor("f_rev_high", "F", 1, ("mid", "long"), "營收創新高程度(3月/24月高)")
def _f_high(c):
    return _rev(c, lambda r: r.rolling(3, min_periods=3).sum() / r.rolling(3, min_periods=3).sum()
                .rolling(24, min_periods=12).max(), "rev_high")


def _valuation_present(c):
    return c.P["pb"].notna()


@factor("f_ep", "F", 1, ("mid", "long"), "盈餘殖利率(1/本益比)")
def _f_ep(c): return (1 / c.P["pe"].astype(float)).where(c.P["pe"].notna(), -0.05).where(_valuation_present(c))


@factor("f_bp", "F", 1, ("long",), "淨值股價比(1/PB)")
def _f_bp(c): return 1 / c.P["pb"].astype(float).where(c.P["pb"] > 0)


@factor("f_dy", "F", 1, ("long",), "現金殖利率")
def _f_dy(c): return c.P["dy"].astype(float)


@factor("f_roe", "F", 1, ("mid", "long"), "ROE(=PB/PE)")
def _f_roe(c):
    pe, pb = c.P["pe"].astype(float), c.P["pb"].astype(float)
    return (pb / pe).where(pe.notna(), -0.05).where(_valuation_present(c)).clip(-0.5, 1.0)


def eps_ttm(c):
    """Trailing-4Q EPS implied by the exchange's daily P/E (close / PE). PE is blank when EPS ≤ 0 → NaN.
    The exchange updates PE after each quarterly report, so this is point-in-time without a separate
    financial-statement feed (v1.3, idea taken from Alpha Radar V23.1's eps_growth_yoy)."""
    return c.get("eps_ttm", lambda: (c.P["close"].astype(float) / c.P["pe"].astype(float)).where(c.P["pe"] > 0))


@factor("f_eps_yoy", "F", 1, ("mid", "long"), "EPS年增(由本益比推算)")
def _f_eps_yoy(c):
    e = eps_ttm(c)
    prev = e.shift(250)
    return (e / prev - 1).where((prev > 0) & (e > 0)).clip(-1, 5)


def rsi14(c):
    d = c.adj.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return c.get("rsi14", lambda: 100 * up / (up + dn).where(up + dn > 0))


@factor("f_size", "F", 0, ("mid", "long"), "小市值(−log市值)")
def _f_size(c): return -np.log(c.P["mcap"].astype(float).where(c.P["mcap"] > 0))


# ================================================================== v1.7 國際連動／產業鏈輪動（研究候選；未通過驗收者由 sparse_exclude 排除）
def _wser(c, key):
    W = c.P.world
    if W is None or key not in W or not np.isfinite(W[key].to_numpy(dtype=float)).any():
        return None
    return W[key].astype(float)


def _nan_like(c):
    return pd.DataFrame(np.nan, index=c.P.dates, columns=c.P.tickers)


def _beta_to(c, key, n=120):
    """個股日報酬對「同一交易日已發生」的國際市場報酬的滾動 β（美股＝前一夜，日韓＝同日）。"""
    def calc():
        w = _wser(c, key)
        if w is None:
            return None
        x = w.pct_change(fill_method=None).where(lambda v: v.abs() < 0.25)
        ret = c.P["ret"].astype(float).where(c.P["tradable"])
        m = ret.notna() & x.notna().to_numpy()[:, None]
        r0, x0 = ret.where(m), m.mul(x, axis=0).where(m)
        mp = max(40, int(n * 0.6))
        ex, er = x0.rolling(n, min_periods=mp).mean(), r0.rolling(n, min_periods=mp).mean()
        cov = (r0 * x0).rolling(n, min_periods=mp).mean() - ex * er
        var = (x0 * x0).rolling(n, min_periods=mp).mean() - ex * ex
        return cov / var.where(var > 1e-10)
    return c.get(f"beta_{key}{n}", calc)


@factor("g_us_beta", "T", 0, ALL_H, "對美股費半連動(β)")
def _g_us_beta(c):
    b = _beta_to(c, "SOX")
    return _nan_like(c) if b is None else b


@factor("g_us_lead", "T", 1, ALL_H, "美股費半趨勢×連動(β×費半20日)")
def _g_us_lead(c):
    b, w = _beta_to(c, "SOX"), _wser(c, "SOX")
    return _nan_like(c) if b is None else b.mul(w / w.shift(20) - 1, axis=0)


@factor("g_asia_lead", "T", 1, ALL_H, "韓股趨勢×連動(β×KOSPI 20日)")
def _g_asia_lead(c):
    b, w = _beta_to(c, "KS11"), _wser(c, "KS11")
    return _nan_like(c) if b is None else b.mul(w / w.shift(20) - 1, axis=0)


@factor("g_fx_flow", "T", 1, ALL_H, "台幣升值×外資持股(熱錢流向)")
def _g_fx_flow(c):
    w = _wser(c, "TWD")
    if w is None or "foreign_ratio" not in c.P.m:
        return _nan_like(c)
    return c.P["foreign_ratio"].astype(float).mul(-(w / w.shift(20) - 1), axis=0)


def chain_groups(P: Panel) -> pd.Series:
    """每檔股票的產業鏈族群（電子上游-半導體…；同 App 標籤）。"""
    from .chain import label
    sec = P.info["sector"] if P.info is not None and "sector" in P.info else pd.Series("NA", index=P.tickers)
    return pd.Series([label(t, s)["group"] for t, s in zip(P.tickers, sec.reindex(P.tickers).fillna("NA"))], index=P.tickers)


@factor("t_ind_mom", "T", 1, ("mid", "long"), "族群動能(同產業鏈其他股60日)")
def _t_ind_mom(c):
    g = c.get("chain_groups", lambda: chain_groups(c.P))
    m = c.mom(60).astype(float)
    ok = m.notna()
    out = _nan_like(c)
    for grp, cols in g.groupby(g).groups.items():
        if len(cols) < 3:
            continue
        sub, k = m[cols], ok[cols]
        tot, n = sub.sum(axis=1, min_count=1), k.sum(axis=1)
        ex = (tot.to_numpy()[:, None] - sub.fillna(0).to_numpy()) / (n.to_numpy()[:, None] - k.to_numpy())
        ex[(n.to_numpy()[:, None] - k.to_numpy()) < 2] = np.nan
        out[cols] = ex
    return out.where(c.adj.notna())


# ================================================================== v1.9 候選因子（預先宣告研究；未通過者由 inactive_factors 排除）
def _ret(c):
    return c.get("ret_tr", lambda: c.P["ret"].astype(float).where(c.P["tradable"]))


def _group_rel(c, x: pd.DataFrame) -> pd.DataFrame:
    """x 減去同產業鏈族群（≥5 檔）的當日中位數。"""
    g = c.get("chain_groups", lambda: chain_groups(c.P))
    out = x.copy()
    for grp, cols in g.groupby(g).groups.items():
        if len(cols) < 5:
            out[cols] = np.nan
            continue
        sub = x[cols]
        out[cols] = sub.sub(sub.median(axis=1), axis=0)
    return out


@factor("t_max20", "T", 1, ("short", "mid"), "近20日無暴漲(MAX效應)")
def _t_max20(c): return -_ret(c).rolling(20, min_periods=15).max()


@factor("t_skew60", "T", 1, ("mid", "long"), "報酬偏態低(非彩券股)")
def _t_skew60(c): return -_ret(c).rolling(60, min_periods=40).skew()


@factor("t_limitup20", "T", 0, ("short", "mid"), "近20日漲停次數(注意力)")
def _t_lu20(c): return (_ret(c) >= 0.095).astype(float).where(c.P["tradable"]).rolling(20, min_periods=15).sum()


@factor("t_high60", "T", 1, ("short", "mid"), "接近60日新高")
def _t_high60(c): return c.adj / c.hi.rolling(60, min_periods=45).max()


@factor("t_bbw", "T", 0, ("short", "mid"), "布林帶寬收斂(相對120日)")
def _t_bbw(c):
    sd = c.adj.rolling(20, min_periods=15).std()
    bw = (4 * sd / c.ma(20)).where(c.ma(20) > 0)
    return -(bw / bw.rolling(120, min_periods=80).median())


@factor("f_rev_surprise", "F", 1, ("mid", "long"), "營收驚喜(年增率相對近12月常態)")
def _f_rev_surprise(c):
    def fn(r):
        y = _yoy(r, 1)
        m, s_ = y.shift(1).rolling(12, min_periods=8).mean(), y.shift(1).rolling(12, min_periods=8).std()
        return ((y - m) / s_.where(s_ > 0.02)).clip(-5, 5)
    return _rev(c, fn, "rev_surprise")


@factor("f_rev_rel", "F", 1, ("mid", "long"), "營收年增率相對同業")
def _f_rev_rel(c): return _group_rel(c, _rev(c, lambda r: _yoy(r, 3), "rev_yoy3"))


@factor("c_trust60", "C", 1, ("mid", "long"), "投信60日累計買超(占股本)")
def _c_trust60(c): return c.rsum("trust", 60) / c.shares


@factor("c_fr_high", "C", 1, ("mid", "long"), "外資持股比率接近半年高點")
def _c_fr_high(c):
    if "foreign_ratio" not in c.P.m:
        return _nan_like(c)
    fr = c.P["foreign_ratio"].astype(float)
    return fr - fr.rolling(120, min_periods=60).max()


@factor("f_pe_self", "F", 1, ("mid", "long"), "本益比低於自身兩年水準")
def _f_pe_self(c):
    pe = c.P["pe"].astype(float)
    return -(pe / pe.rolling(500, min_periods=250).median() - 1)


@factor("f_dy_rel", "F", 1, ("long",), "殖利率相對同業")
def _f_dy_rel(c): return _group_rel(c, c.P["dy"].astype(float))


V19_CANDIDATES = ["t_max20", "t_skew60", "t_limitup20", "t_high60", "t_bbw", "f_rev_surprise", "f_rev_rel", "c_trust60",
                  "c_fr_high", "f_pe_self", "f_dy_rel"]


def compute(P: Panel, names=None, sample_idx=None, log=None) -> tuple[dict, Ctx]:
    """Compute factors; returns {name: ndarray[n_sample, n_tickers] float32} and the context cache."""
    from . import kostolany  # noqa: F401  (import registers the K factors)
    ctx = Ctx(P)
    out = {}
    for name, f in REGISTRY.items():
        if names and name not in names:
            continue
        v = f.func(ctx)
        v = v.replace([np.inf, -np.inf], np.nan) if isinstance(v, pd.DataFrame) else v
        arr = v.to_numpy(dtype=float) if isinstance(v, pd.DataFrame) else np.asarray(v, float)
        if sample_idx is not None:
            arr = arr[sample_idx]
        out[name] = arr.astype(F32)
        if log:
            log(f"  factor {name}: coverage {np.isfinite(arr).mean():.0%}")
    return out, ctx
