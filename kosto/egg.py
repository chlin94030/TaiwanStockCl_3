"""v3.3 科斯托蘭尼：雞蛋圖、目前階段、各階段的歷史表現、貨幣＋心理溫度計、對消息的反應、原著原則對照（docs/v3.3_kostolany_review.md）。

只做顯示與描述性統計，不改選股模型。原著頁面：「股票市場的永恆週期循環」（A1～B3、成交量與持股人數）、
「超買還是超賣」（長期上漲後對好消息無反應＝超買；長期下跌後對壞消息無反應＝超賣）、固執與猶豫的投資者。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STAGES

# 原著的定義（成交量、持股人數）與該做的事
STAGE_INFO = {
    "A1": dict(name="修正階段（上漲起點）", book="成交量小、持股人數少", act="買入", color="#3b82f6"),
    "A2": dict(name="相隨階段（上漲）", book="成交量和持股人數增加", act="等待、保留股票", color="#22c55e"),
    "A3": dict(name="過熱階段（上漲末段）", book="成交量異常活躍、持股人數多；X 點最高", act="賣出", color="#f59e0b"),
    "B1": dict(name="修正階段（下跌起點）", book="成交量小、持股人數開始下降", act="賣出", color="#ef4444"),
    "B2": dict(name="相隨階段（下跌）", book="成交量增加、持股人數繼續降低", act="等待、保留現金", color="#a855f7"),
    "B3": dict(name="過熱階段（下跌末段）", book="成交量很大、持股人數很低；Y 點最低", act="買入", color="#06b6d4"),
}

# 原著原則 → 模型做法 → 歷史驗證 → 狀態（v1.1～v3.3 的結論）
PRINCIPLES = [
    ("雞蛋理論：A1、B3 買，A3、B1 賣（大盤）", "大盤階段：成交量比、融資水位（代替持股人數）、廣度、位置",
     "A1 之後 60 日 +6.8%（上漲機率 83%）、B3 恐慌出清 +14.7%（93%）最好；B1 −2.0%（35%）最差", "✅ 符合（v3.3 修正 B3 判定）"),
    ("A3 過熱要賣、B2 要持現金等待", "長線 A3 60%、B1 50%、B2 40% 持股，其餘放 0050",
     "B2 之後 60 日 +1.3%（上漲機率 47%）＝等待合理；但台股 A3 之後仍 +6.6%；短中線照書減碼年化少 4～7 個百分點", "⚠ B2 符合、A3 不符；只用在長線"),
    ("B3 是最低點（Y 點）", "v3.3 修正：距高點 ≥15% 且近 20 日 ≤ −8% 或廣度 ≤ 25% → B3",
     "原分類把 2020-03、2022-10、2025-04 都判成 B2；修正後長線持股水位測試 −0.9 個百分點（B2 減碼的資金本來就放 0050）",
     "✅ 顯示已修正；持股水位不變"),
    ("個股雞蛋階段：A3、B1、B2 不買", "個股階段分數（量、融資、位置）",
     "加回階段閘門：開發期年化少 23～50 個百分點（台股動能延續）", "❌ 關閉，改顯示在個股頁"),
    ("固執的投資者 vs 猶豫的投資者（股票已付清）", "法人買超−融資增加、融資占股本",
     "「已付清」因子 15%：回撤少 12～16 個百分點，但超額少 10～19 個百分點（偏防禦）", "❌ 未通過（v3.3 K1）"),
    ("長期下跌後利空不跌＝超賣、進場", "大盤：距高點 ≥10% 且費半大跌當天台股沒跌",
     "之後 60 日平均 +11.4%（一般下跌期 +7.7%），22 次", "✅ 符合，做成燈號"),
    ("長期上漲後利多不漲＝超買、退場", "大盤：一年區間上 80% 且費半大漲當天台股沒漲",
     "之後 60 日平均 +6.7%（一般高檔期 +4.9%），48 次——台股沒有出現書中的現象", "⚠ 不符，只顯示不當訊號"),
    ("耐心、不追高、不被小消息左右", "續抱倍數（排名掉出 2～3 倍才賣）、不追高閘門、不讀新聞",
     "移除續抱：短線開發期年化少 14 個百分點、周轉率增加；中長線影響在 ±1 個百分點內", "✅ 採用"),
    ("不設機械式停損", "只顯示警戒價、不自動賣出", "加回停損：開發期年化少 7～14 個百分點", "✅ 採用"),
    ("只看 6～12 個月的長期線圖、鋸齒形上升", "120 日動能、一年新高、M 頭／W 底", "動能是三個期間最穩定的因子", "✅ 採用"),
    ("貨幣＋心理＝趨勢（利率、資金）", "美債殖利率、美元／台幣、融資、成交量、VIX", "研究資料庫沒有完整利率歷史，無法回測", "ℹ 顯示（溫度計）"),
    ("持股人數", "集保股權分散（每週，持續累積中）", "沒有足夠歷史可回測；以融資餘額代替", "ℹ 顯示"),
    ("三分之二逆勢、三分之一順勢", "模型以順勢（動能＋營收）為主", "2019–2026 台股順勢明顯勝過逆勢；逆勢做成「反彈候選」名單", "⚠ 與原著比例不同（資料決定）"),
]


B3_DD, B3_R20, B3_BREADTH = 0.15, 0.08, 0.25


def _days(B) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(B.meta["days"])


def market_stage(B) -> pd.DataFrame:
    """每日大盤階段分數（A1～B3）、階段代碼、信心、指數、回落、成交量比、廣度。"""
    cols = list(STAGES) + ["confidence", "index", "drawdown", "turnover_ratio", "breadth"]
    if "MKT_DAILY" not in B.a:
        return pd.DataFrame()
    md = pd.DataFrame(B.a["MKT_DAILY"].astype(float), index=_days(B), columns=cols[:B.a["MKT_DAILY"].shape[1]])
    code = B.a.get("MKT_DAILY_CODE")
    md["code"] = code.astype(int) if code is not None else 0
    md["stage_model"] = [STAGES[c - 1] if 1 <= c <= 6 else "UNKNOWN" for c in md["code"]]
    # v3.3 依原著修正 B3（恐慌出清、Y 點）：原分類把 2020-03、2022-10、2025-04 的重挫都判成 B2，從未出現 B3。
    # 修正：距一年高點 ≥ 15%，且近 20 日 ≤ −8% 或站上季線的股票 ≤ 25% → B3（只用於顯示；模型持股水位仍用原分類）
    r20 = md["index"] / md["index"].shift(20) - 1 if "index" in md else pd.Series(np.nan, index=md.index)
    b3 = (md.get("drawdown", np.nan) <= -B3_DD) & ((r20 <= -B3_R20) | (md.get("breadth", np.nan) <= B3_BREADTH))
    md["b3_fix"] = b3.fillna(False)
    md["stage"] = np.where(md["b3_fix"], "B3", md["stage_model"])
    return md


def mkt300(B) -> pd.Series:
    """每日：前一個換股日市值前 300 名等權報酬（大盤代理；與 v2.3 相同）。"""
    days = _days(B)
    R = B.a["R"].astype(float)
    if "RANKS" in B.a and "MCAP" in (B.meta.get("rules") or []):
        rk = B.a["RANKS"][B.meta["rules"].index("MCAP")]
        sp = np.asarray(B.sig_pos)
        sig_of = np.searchsorted(sp, np.arange(len(days)), side="right") - 1
        out = np.zeros(len(days))
        for d in range(1, len(days)):
            s = max(sig_of[d - 1], 0)
            m = (rk[s] >= 1) & (rk[s] <= 300)
            out[d] = np.nanmean(R[d, m]) if m.any() else np.nan
        return pd.Series(np.nan_to_num(out), index=days)
    with np.errstate(invalid="ignore"):
        return pd.Series(np.nan_to_num(np.nanmean(R, axis=1)), index=days)


def stage_forward(B, since: str = "2019-07-01") -> pd.DataFrame:
    """每個大盤階段之後 20／60 日的大盤報酬與上漲機率。"""
    md = market_stage(B)
    if md.empty:
        return pd.DataFrame()
    lv = (1 + mkt300(B)).cumprod()
    f20, f60 = lv.shift(-20) / lv - 1, lv.shift(-60) / lv - 1
    base = f60.notna() & (md.index >= pd.Timestamp(since))
    rows = []
    for s in list(STAGES) + ["全部"]:
        m = base & ((md["stage"] == s) if s != "全部" else True)
        rows.append({"stage": s, "n": int(m.sum()), "f20": float(f20[m].mean()) if m.any() else np.nan,
                     "f60": float(f60[m].mean()) if m.any() else np.nan, "w60": float((f60[m] > 0).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows).set_index("stage")


def news_reaction(B, window: int = 20, thr: float = 0.02) -> dict:
    """大盤對消息的反應（原著 132 題）：利多日＝前一晚費半 ≥ +2%，利空日＝ ≤ −2%；看台股當天有沒有跟著動。"""
    keys = list(B.meta.get("world_keys") or [])
    if "SOX" not in keys or "WORLD" not in B.a:
        return {}
    days = _days(B)
    sox = pd.Series(B.a["WORLD"][:, keys.index("SOX")].astype(float), index=days).ffill().pct_change()
    m = mkt300(B)
    md = market_stage(B)
    idx = md["index"] if "index" in md else (1 + m).cumprod()
    mx, mn = idx.rolling(250, min_periods=120).max(), idx.rolling(250, min_periods=120).min()
    pos, dd = (idx - mn) / (mx - mn), idx / mx - 1
    good_nr = (sox >= thr) & (m <= 0)
    bad_nr = (sox <= -thr) & (m >= 0)
    tail = slice(len(days) - window, len(days))
    gd = [str(d.date()) for d in days[tail][good_nr.iloc[tail].to_numpy()]]
    bd = [str(d.date()) for d in days[tail][bad_nr.iloc[tail].to_numpy()]]
    return {"good_days": int((sox.iloc[tail] >= thr).sum()), "bad_days": int((sox.iloc[tail] <= -thr).sum()),
            "good_no_react": gd, "bad_no_react": bd, "pos": float(pos.iloc[-1]) if np.isfinite(pos.iloc[-1]) else np.nan,
            "dd": float(dd.iloc[-1]) if np.isfinite(dd.iloc[-1]) else np.nan,
            "oversold_signal": bool(bd) and float(dd.iloc[-1]) <= -0.10,
            "overbought_watch": bool(gd) and float(pos.iloc[-1]) >= 0.8}


def money_psych(B) -> list[tuple[str, str, float, str, str]]:
    """貨幣＋心理溫度計：(類別, 名稱, 數值, 說明, 傾向 +1 偏多／−1 偏空／0)。只用資料包裡已有的資料。"""
    out = []
    keys = list(B.meta.get("world_keys") or [])
    W = pd.DataFrame(B.a["WORLD"].astype(float), index=_days(B), columns=keys).ffill() if "WORLD" in B.a and keys else pd.DataFrame()

    def chg(k, n=60):
        x = W[k].dropna() if k in W else pd.Series(dtype=float)
        return (float(x.iloc[-1]), float(x.iloc[-1] - x.iloc[-1 - n])) if len(x) > n else (np.nan, np.nan)
    for k, nm in (("TNX", "美國 10 年期公債殖利率"), ("IRX", "美國 3 個月國庫券利率")):
        lv, d = chg(k)
        if np.isfinite(lv):
            out.append(("貨幣", nm, lv, f"{lv:.2f}%，近 60 日 {d:+.2f} 個百分點", -1 if d > 0.25 else (1 if d < -0.25 else 0)))
    if "TWD" in W:
        x = W["TWD"].dropna()
        if len(x) > 60:
            c = float(x.iloc[-1] / x.iloc[-61] - 1)
            trend = "大致持平" if abs(c) < 0.005 else (f"台幣{'貶' if c > 0 else '升'}值，外資資金{'流出' if c > 0 else '流入'}傾向")
            out.append(("貨幣", "美元／台幣", float(x.iloc[-1]), f"{x.iloc[-1]:.2f}，近 60 日 {c * 100:+.1f}%（{trend}）",
                        -1 if c > 0.02 else (1 if c < -0.02 else 0)))
    cols = list(B.meta.get("q_mkt_cols") or [])
    Q = pd.DataFrame(B.a["Q_MKT"].astype(float), index=pd.DatetimeIndex(B.meta["q_days"]), columns=cols) if "Q_MKT" in B.a and cols else pd.DataFrame()
    if "dy_mkt" in Q and "TNX" in W:
        dy = Q["dy_mkt"].dropna()
        if len(dy):
            spread = float(dy.iloc[-1]) - float(W["TNX"].dropna().iloc[-1])
            out.append(("貨幣", "股息殖利率 − 美債殖利率", spread, f"{dy.iloc[-1]:.2f}% − {W['TNX'].dropna().iloc[-1]:.2f}% = {spread:+.2f} 個百分點（越高股票越有吸引力）",
                        1 if spread > 0 else -1))
    if "mbal" in Q:
        mb = Q["mbal"].dropna()
        if len(mb) > 250:
            z = float((mb.iloc[-1] - mb.tail(250).mean()) / mb.tail(250).std())
            c60 = float(mb.iloc[-1] / mb.iloc[-61] - 1)
            out.append(("心理", "融資餘額（猶豫的投資者）", z, f"一年位階 z = {z:+.1f}，近 60 日 {c60 * 100:+.1f}%（越高越過熱）",
                        -1 if z > 1.0 else (1 if z < -1.0 else 0)))
    md = market_stage(B)
    if len(md):
        vr = float(md["turnover_ratio"].iloc[-1])
        out.append(("心理", "成交量（近 20 日／一年平均）", vr, f"{vr:.2f} 倍（> 1.3 熱絡、< 0.8 冷清）", -1 if vr > 1.3 else (1 if vr < 0.8 else 0)))
        br = float(md["breadth"].iloc[-1])
        out.append(("心理", "廣度（站上季線的股票）", br, f"{br:.0%}（> 70% 樂觀、< 30% 悲觀）", -1 if br > 0.7 else (1 if br < 0.3 else 0)))
    if "VIX" in W:
        v = float(W["VIX"].dropna().iloc[-1])
        out.append(("心理", "VIX 恐慌指數", v, f"{v:.1f}（> 30 恐慌、< 14 安逸）", 1 if v > 30 else (-1 if v < 14 else 0)))
    if "Q_HN" in B.a:
        H = pd.DataFrame(B.a["Q_HN"].astype(float), index=pd.DatetimeIndex(B.meta["q_days"])).dropna(how="all")
        if len(H) > 25:
            tot = H.sum(axis=1, min_count=100).dropna()
            if len(tot) > 25:
                c = float(tot.iloc[-1] / tot.iloc[-21] - 1)
                out.append(("心理", "持股人數（集保，全部個股加總）", c, f"近 4 週 {c * 100:+.2f}%（增加＝股票流向更多散戶）", -1 if c > 0.01 else (1 if c < -0.01 else 0)))
    return out


def egg_svg(stage: str, scores: dict | None = None, width: int = 560) -> str:
    """原著的雞蛋圖：左邊上漲（A1→A3）、右邊下跌（B1→B3），目前階段發光標示。"""
    pos = {"A1": (90, 230), "A2": (60, 140), "A3": (120, 52), "B1": (440, 52), "B2": (500, 140), "B3": (470, 230)}
    s = [f"<svg viewBox='0 0 {width} 290' width='100%' style='max-width:{width}px' role='img' aria-label='科斯托蘭尼雞蛋圖'>",
         "<ellipse cx='280' cy='145' rx='230' ry='112' fill='none' stroke='var(--line)' stroke-width='3'/>",
         "<path d='M 120 248 Q 30 145 140 40' fill='none' stroke='var(--muted)' stroke-width='2' marker-end='url(#eg)'/>",
         "<path d='M 420 40 Q 530 145 440 248' fill='none' stroke='var(--muted)' stroke-width='2' marker-end='url(#eg)'/>",
         "<defs><marker id='eg' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
         "<path d='M0,0 L10,5 L0,10 z' fill='var(--muted)'/></marker></defs>",
         "<text x='280' y='26' text-anchor='middle' fill='var(--muted)' font-size='12'>X 點（最高）</text>",
         "<text x='280' y='278' text-anchor='middle' fill='var(--muted)' font-size='12'>Y 點（最低）</text>",
         "<text x='150' y='150' text-anchor='middle' fill='var(--faint)' font-size='13'>上漲</text>",
         "<text x='410' y='150' text-anchor='middle' fill='var(--faint)' font-size='13'>下跌</text>",
         "<text x='280' y='106' text-anchor='middle' fill='var(--muted)' font-size='12'>A3、B1 賣出</text>",
         "<text x='280' y='132' text-anchor='middle' fill='var(--muted)' font-size='12'>A2 等待（保留股票）</text>",
         "<text x='280' y='156' text-anchor='middle' fill='var(--muted)' font-size='12'>B2 等待（保留現金）</text>",
         "<text x='280' y='182' text-anchor='middle' fill='var(--muted)' font-size='12'>A1、B3 買入</text>"]
    for k, (x, y) in pos.items():
        info = STAGE_INFO[k]
        on = k == stage
        sc = (scores or {}).get(k)
        r = 30 if on else 22
        s.append(f"<circle cx='{x}' cy='{y}' r='{r}' fill='{info['color'] if on else 'var(--panel2)'}' stroke='{info['color']}' stroke-width='{3 if on else 1.5}'"
                 + (" style='filter:drop-shadow(0 0 8px " + info["color"] + ")'" if on else "") + "/>")
        s.append(f"<text x='{x}' y='{y + 5}' text-anchor='middle' fill='{'#0b1020' if on else 'var(--strong)'}' font-size='{15 if on else 13}' font-weight='700'>{k}</text>")
        if sc is not None and np.isfinite(sc):
            s.append(f"<text x='{x}' y='{y + (r + 14)}' text-anchor='middle' fill='var(--muted)' font-size='10.5'>{sc:.2f}</text>")
    s.append("</svg>")
    return "".join(s)
