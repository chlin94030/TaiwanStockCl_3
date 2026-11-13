"""v3.0 消息衝擊評估：重大消息 → 哪些細族群受影響、多深、多快；並用歷史同類事件量化（docs/v3.0_protocol.md B）。

兩種證據並列：
1. 基本面判斷（人工整理，依營收曝險與傳導順序）：depth −3～+3、speed（立即／1～2 季／較慢）、白話理由；
2. 歷史同類事件：細族群代表個股等權報酬減全市場（成交金額前 800 名）等權，事件日、5 日、20 日（自前一日收盤起算）。
歷史事件日＝美國消息之後台股的第一個交易日；資料庫沒有的日期自動略過，之後的新事件（例 2026-11-05）資料進來就自動納入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kosto import supply

# ------------------------------------------------------------------ 歷史事件組（台股反應日）
EVENT_SETS = {
    "ai_down": ("AI 資本支出疑慮", ["2024-09-04", "2025-02-03", "2025-02-24", "2025-03-27", "2026-11-05"],
                "輝達單日 −9.5%、DeepSeek、微軟傳取消資料中心租約（兩次）、美國期中選舉後資料中心暫停疑慮"),
    "ai_up": ("AI 資本支出利多", ["2023-05-25", "2024-02-22", "2025-01-22", "2025-09-23"],
              "輝達財測大幅上修、輝達財報、星際之門 Stargate、OpenAI 與輝達千億合作"),
    "hike": ("聯準會升息", ["2022-03-17", "2022-05-05", "2022-06-16", "2022-07-28", "2022-09-22", "2022-11-03", "2022-12-15",
                         "2023-02-02", "2023-03-23", "2023-05-04", "2023-07-27", "2026-09-17"], "2022–23 年 11 次、2026-09 一次"),
    "cut": ("聯準會降息", ["2019-08-01", "2019-09-19", "2019-10-31", "2020-03-04", "2020-03-16", "2024-09-19", "2024-11-08",
                        "2024-12-19", "2025-09-18", "2025-10-30", "2025-12-11"], "2019、2020 疫情緊急降息、2024–25"),
    "election": ("美國大選／期中選舉", ["2020-11-04", "2022-11-09", "2024-11-06", "2026-11-04"], "開票當天（台股盤中）"),
    "apple": ("蘋果秋季發表會", ["2019-09-11", "2020-10-14", "2021-09-15", "2022-09-08", "2023-09-13", "2024-09-10", "2025-09-10",
                           "2026-09-10"], "2026-09-10 為 iPhone Duo 發表"),
    "rival": ("競爭者利多（輝達相對利空）", ["2025-10-07", "2025-10-14", "2025-11-26", "2026-11-10", "2026-11-13"],
              "AMD 與 OpenAI 合作（AMD +24%）、博通與 OpenAI 合作、Meta 傳採用 Google TPU（輝達 −2.6%）；2026-11 黃仁勳風波日資料進來後自動納入"),
}

# ------------------------------------------------------------------ 情境：基本面判斷（depth：− 受傷、＋ 受惠；3 最深）
SCENARIOS = [
    dict(id="dc_slowdown", name="美國期中選舉民主黨大勝・德州資料中心暫停", date="2026-11-04", analog="ai_down",
         facts=["11/03 期中選舉民主黨拿下參眾兩院；德州州長由民主黨 Hinojosa 當選，主張 1/19 就任後暫停所有資料中心審批，直到州議會立法。",
                "德州約占美國 2030 年前新增資料中心容量的四分之一；ERCOT 8 月起已暫停 75MW 以上新資料中心併網、9 月停發州許可（共和黨州長任內）。",
                "美股 11/04：費半 −3.1%、Vistra −13.8%、GE Vernova −9.3%；11/05 盤中費半一度 −11.6%。台股 11/05 盤中一度跌逾千點，PCB／載板多檔跌停。",
                "分析師多數看法：是「變慢、變貴、搬到別州」，不是「取消」；已進入施工的約 22GW 不受影響，風險主要在 2028 年以後的案子。"],
         chain=["電網與電力設備（重電）：最直接，但訂單已排到 2029～2030，傷的是股價評價",
                "機房施工、上架（散熱、電源、機殼）：最先感受到延遲",
                "伺服器組裝：出貨要等機房完工 → 營收遞延",
                "PCB、載板、光通訊：高評價、高波動，股價跌最兇",
                "晶片（台積電、CoWoS）：產能仍供不應求、訂單有緩衝 → 傳導最慢"],
         impacts=[("heavy", -3, "立即", "最直接連結資料中心建置；股價已從高點跌 3～5 成，訂單能見度到 2029 年"),
                  ("thermal", -3, "1～2 季", "最貼近機房施工與上架時程"), ("power", -3, "1～2 季", "機櫃電源、BBU 跟著機房進度"),
                  ("chassis", -2, "1～2 季", "滑軌、機殼跟著整櫃出貨"), ("optical", -3, "立即", "高評價題材股，資本支出疑慮時跌最多"),
                  ("ccl", -3, "立即", "11/05 多檔跌停；高速板材需求跟著伺服器"), ("pcb", -2, "1～2 季", "伺服器板占比高者受影響"),
                  ("abf", -2, "1～2 季", "AI 晶片載板"), ("server", -2, "1～2 季", "營收遞延，但毛利低、已有訂單"),
                  ("asic", -2, "立即", "雲端客製晶片評價高"), ("bmc", -2, "立即", "伺服器出貨量指標股"),
                  ("conn", -2, "1～2 季", "伺服器高速線材"), ("memory", -1, "立即", "實際需求有時間差；過去 AI 疑慮事件沒有明顯跟跌，但今年已成 AI 題材、波動變大"),
                  ("probe", -2, "較慢", "AI 晶片測試"), ("equip", -1, "較慢", "看台積電資本支出，短期不變"),
                  ("adv_pkg", -1, "較慢", "CoWoS 供不應求"), ("foundry", -1, "較慢", "11/05 台積電只跌約 0.4%"),
                  ("passive", -1, "1～2 季", "用量大但分散"), ("pc", 0, "—", "消費性產品不受影響"), ("drv", 0, "—", "非資料中心"),
                  ("lens", 0, "—", "手機鏡頭不受影響")],
         note="資金常轉向非電子（金融、航運、電信）避險；台股 11/05 航空、金融逆勢上漲。",
         sources=[("Bloomberg", "https://www.bloomberg.com/news/articles/2026-11-04/ai-data-center-boom-is-set-for-upheaval-after-us-election-rout"),
                  ("鉅亨網 11/05 開盤", "https://news.cnyes.com/news/id/6647548"),
                  ("Infrastructure Brief", "https://infrastructurebrief.com/texas-votes-to-rein-in-data-centres-now-what/")]),
    dict(id="rates", name="升息循環（2026-09 升息、12 月可能再升，10 年期殖利率約 5%）", date="2026-10-28", analog="hike",
         facts=["聯準會 2026-09-16 升息一碼（2023 年以來首次），10/28 維持 3.75%～4.00%（兩位理事主張再升）。",
                "期貨市場估 12 月再升息機率約 78%；美國 10 年期公債殖利率約 5.0%。"],
         chain=["利率上升 → 未來獲利折現值下降：高本益比、看遠期題材的股票受傷最多（IP、ASIC、光通訊、新題材）",
                "美元走強 → 外資匯出、台幣貶值（出口電子廠帳上有匯兌收益，部分抵銷）",
                "銀行、壽險利差擴大 → 金融股相對受惠",
                "降息時方向相反：成長股、高本益比族群先漲"],
         impacts=[("ip", -2, "立即", "本益比最高的族群之一"), ("asic", -2, "立即", "評價看遠期"), ("optical", -2, "立即", "題材股評價"),
                  ("bmc", -2, "立即", "高價股"), ("equip", -1, "立即", "成長股"), ("memory", -1, "1～2 季", "景氣循環＋美元"),
                  ("foundry", -1, "立即", "權值股，外資進出"), ("server", -1, "立即", "毛利低、評價較低，受傷較小"),
                  ("pc", -1, "1～2 季", "消費需求受利率影響"), ("heavy", -1, "立即", "高評價"), ("dist", 0, "—", "低本益比")],
         note="金融股（銀行、壽險）通常相對受惠；可在「歷史同類事件」看 2022 年 11 次升息後各族群的平均反應。",
         sources=[("Stock Analysis：10 月 FOMC", "https://stockanalysis.com/news/fed-holds-rates-steady-oct-2026/"),
                  ("Axios：9 月升息", "https://www.axios.com/2026/09/16/fed-rates-warsh-trump")]),
    dict(id="spacex", name="SpaceX 上市（2026-06-12，Nasdaq：SPCX）", date="2026-06-12", analog=None, theme="leo",
         facts=["2026-06-12 上市，承銷價 135 美元、市值約 1.77 兆美元；首日收 161 美元（+19%），之後一個多月跌破承銷價、較高點回落 4～5 成。",
                "Starlink 占 SpaceX 2025 年營收約 6 成；台廠有公開供應紀錄的只有少數（昇達科、華通、啟碁、敬鵬）。"],
         chain=["上市本身不改變訂單；募到的錢用在衛星發射與星艦 → 未來幾年衛星與地面設備訂單",
                "股價面：上市前題材發酵、上市後「利多出盡」常見（見下方低軌衛星族群路徑）",
                "真正受惠看營收：有供應紀錄、衛星營收占比高的公司"],
         impacts=[("rf", 3, "1～2 季", "衛星波導、濾波器（昇達科有供應紀錄）"), ("pcb", 1, "較慢", "華通、敬鵬有衛星板供應紀錄；占營收比重不高"),
                  ("network", 1, "較慢", "啟碁地面設備"), ("compound", 1, "較慢", "砷化鎵射頻")],
         note="題材股波動大：族群路徑顯示上市前後的漲跌，買在題材高點的風險高。",
         sources=[("Wikipedia：SpaceX IPO", "https://en.wikipedia.org/wiki/Initial_public_offering_of_SpaceX"),
                  ("數位時代：台灣低軌衛星供應鏈", "https://www.bnext.com.tw/article/91220/spacex-taiwan-leo-satellite-supply-chain")]),
    dict(id="iphone_duo", name="Apple 摺疊機 iPhone Duo（2026-09-09 發表）", date="2026-09-10", analog="apple", theme="apple",
         facts=["2026-09-09 發表，售價 1,999 美元，護照大小、展開 7.6 吋，A20 處理器與自研 C2 數據機。",
                "IDC 預期蘋果能賣完所有產能、2027 年底拿下摺疊機約 4 成市占。"],
         chain=["摺疊機新增或升級的零件：鉸鏈（單價最高的新零件）、軟板、超薄玻璃與面板、散熱（均熱片）",
                "組裝（鴻海、和碩）與晶片（台積電）量大但占比小",
                "股價面：發表會當天常「利多出盡」，要看出貨量與拉貨月份（9～12 月）"],
         impacts=[("hinge", 3, "1～2 季", "新日興、兆利、富世達：鉸鏈單價高"), ("pcb", 1, "1～2 季", "臻鼎、台郡軟板"),
                  ("lens", 1, "1～2 季", "大立光、玉晶光"), ("thermal", 1, "1～2 季", "奇鋐均熱片"), ("pc", 0, "—", "組裝廠占比小"),
                  ("foundry", 0, "—", "台積電占比小")],
         note="蘋果發表會當天常見「利多出盡」；請看下方歷史發表會後的族群反應。",
         sources=[("財報狗：iPhone Duo 13 檔台廠", "https://statementdog.com/news/17652")]),
    dict(id="nvidia_ceo", name="輝達執行長黃仁勳請辭（2026-11-12）", date="2026-11-13", analog="rival",
         facts=["黃仁勳辭去總裁、執行長與董事，財務長 Colette Kress 暫代；公司表示營運、客戶承諾與產品路線圖不受影響。",
                "美股 11/12：輝達 −4.3%（收 166.19 美元），AMD +5.9%、博通 +5.5%（風波期間 AMD 單日一度 +13%、博通 +8.8%）；"
                "費半 −0.5%、台積電 ADR −1.9%；台指期夜盤 +548 點（43,676）。",
                "第三季財報照原訂台灣 11/18 清晨 5:20 公布。"],
         chain=["治理事件，不是 AI 需求消失：雲端巨頭的資本支出還在，只是可能從輝達「分一些」給 AMD 與自研晶片（ASIC／TPU）",
                "台灣大部分 AI 供應商同時做三個陣營（台積電、日月光、欣興、台光電、金像電、緯創、緯穎、英業達）→ 一進一出，影響小",
                "只做輝達整櫃的環節（鴻海、廣達的 GB 機櫃、水冷板、電源櫃、滑軌）相對受傷",
                "AMD 與 ASIC 專屬的環節（世芯、創意、聯發科 TPU、京元電與測試介面、智邦；力成、南電、嘉澤、技嘉）相對受惠",
                "歷史經驗：只有「明確搶輝達份額」的消息（Meta 傳採用 TPU）才出現分化；「整個 AI 餅變大」的消息（OpenAI 與 AMD／博通合作）三個陣營一起漲"],
         impacts=[("server", -2, "立即", "鴻海、廣達以輝達整櫃為主"), ("thermal", -2, "立即", "水冷板以輝達 GB 機櫃為主"),
                  ("power", -1, "1～2 季", "電源櫃、800V 與輝達合作最深"), ("chassis", -1, "1～2 季", "機櫃滑軌"),
                  ("abf", -1, "立即", "輝達、AMD 都用，一進一出"), ("ccl", -1, "立即", "高速板材三個陣營都用"),
                  ("foundry", 0, "—", "三個陣營的晶片都在台積電生產"), ("adv_pkg", 0, "—", "CoWoS 產能仍供不應求，只是換客戶"),
                  ("asic", 2, "1～2 季", "世芯、創意：雲端自研晶片設計服務"), ("soc", 1, "1～2 季", "聯發科：Google TPU 推論晶片"),
                  ("probe", 1, "1～2 季", "ASIC／TPU 測試時間長，探針卡、測試座受惠")],
         camps=[("nv", -2, "立即", "輝達整櫃專屬環節：份額可能被分走、短線情緒最差"),
                ("amd", 1, "1～2 季", "AMD 份額有機會增加；但 AMD 本身產能與軟體生態仍在追趕"),
                ("asic", 2, "1～2 季", "雲端巨頭加速自研晶片的理由更充分（Google TPU、AWS Trainium、Meta）")],
         note="陣營名單是市場常見分類，不代表公開供應關係；同一家公司常同時供應多個陣營。也請看「盤勢儀表板 → 觀察重點」的輝達事件歷史參考。",
         sources=[("NVIDIA 公告", "https://nvidianews.nvidia.com/news/nvidia-announces-ceo-transition"),
                  ("Yahoo Finance：AMD、博通大漲", "https://finance.yahoo.com/markets/article/amd-broadcom-stocks-jump-as-nvidia-ceo-jensen-huang-resigns-143725561.html"),
                  ("聯合報：台指期夜盤 +548 點", "https://udn.com/news/story/6811/9875869"),
                  ("my-finance：AMD 台灣供應鏈", "https://www.my-finance.com.tw/tw/News_Detail/2336"),
                  ("經濟日報：TPU 供應鏈", "https://money.udn.com/money/amp/story/5607/9337921")]),
]
SCEN_BY_ID = {s["id"]: s for s in SCENARIOS}


# ------------------------------------------------------------------ 計算
def _market(B) -> np.ndarray:
    """全市場等權日報酬（可交易、近 20 日成交金額前 800 名；沒有成交金額時用全部可交易股票）。"""
    R = B.a["R"].astype(float)
    ok = B.a["OK"] if "OK" in B.a else np.isfinite(R)
    Rm = np.where(ok & np.isfinite(R), R, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(Rm, axis=1)


def group_index(B, gids=None) -> dict[str, list[int]]:
    tix = {t: j for j, t in enumerate(B.tickers)}
    out = {}
    for g in gids or supply.SUBGROUPS:
        js = [tix[t] for t, _ in supply.SUBGROUPS[g]["members"] if t in tix] if g in supply.SUBGROUPS else []
        if js:
            out[g] = js
    return out


def theme_index(B, tid: str) -> list[int]:
    tix = {t: j for j, t in enumerate(B.tickers)}
    return [tix[t] for t, _ in supply.theme_members(supply.THEME_BY_ID[tid]) if t in tix]


def _cum(R: np.ndarray, js, d0: int, d1: int) -> float:
    """代表個股等權：每檔 d0..d1 的累積報酬取平均。"""
    seg = R[d0:d1 + 1][:, js]
    with np.errstate(invalid="ignore"):
        c = np.nanprod(1 + np.nan_to_num(seg, nan=0.0), axis=0) - 1
    return float(np.nanmean(c)) if len(c) else np.nan


def event_returns(B, dates, groups: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """每個事件 × 細族群：當天、5 日、20 日超額（減全市場等權）。日期不是交易日 → 用之後第一個交易日；超出資料範圍 → 略過。"""
    groups = groups or group_index(B)
    days = pd.DatetimeIndex(B.meta["days"]) if "days" in B.meta else pd.DatetimeIndex(B.days)
    R = B.a["R"].astype(float)
    m = _market(B)
    mR = m[:, None]
    rows = []
    for ds in dates:
        d = pd.Timestamp(ds)
        i = int(days.searchsorted(d))
        if i >= len(days) or i == 0 or (days[i] - d).days > 4:
            continue
        for g, js in groups.items():
            r = {"event": str(d.date()), "day": days[i], "group": g}
            for h, lab in ((1, "d0"), (5, "d5"), (20, "d20")):
                e = i + h - 1
                if e >= len(days):
                    r[lab] = np.nan
                    continue
                r[lab] = _cum(R, js, i, e) - _cum(mR, [0], i, e)
            rows.append(r)
    return pd.DataFrame(rows, columns=["event", "day", "group", "d0", "d5", "d20"])


def event_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """細族群平均（次數、平均、勝率），依 5 日超額由低到高排序。"""
    if ev is None or not len(ev):
        return pd.DataFrame(columns=["n", "d0", "d5", "d20", "win5"])
    g = ev.groupby("group")
    out = g[["d0", "d5", "d20"]].mean()
    out["n"] = g["d0"].count()
    out["win5"] = g["d5"].apply(lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else np.nan)
    return out.sort_values("d5")


def path(B, js, date: str, before: int = 60, after: int = 60) -> pd.Series:
    """事件前 before 日到後 after 日的累積超額（事件前一日＝0）。"""
    days = pd.DatetimeIndex(B.meta["days"]) if "days" in B.meta else pd.DatetimeIndex(B.days)
    i = int(days.searchsorted(pd.Timestamp(date)))
    if i >= len(days) or not js:
        return pd.Series(dtype=float)
    R = B.a["R"].astype(float)
    m = _market(B)
    a, b = max(0, i - before), min(len(days) - 1, i + after)
    with np.errstate(invalid="ignore"):
        g = np.nanmean(np.where(np.isfinite(R[a:b + 1][:, js]), R[a:b + 1][:, js], np.nan), axis=1)
    x = np.nan_to_num(g - m[a:b + 1])
    c = np.cumprod(1 + x)
    base = c[i - a - 1] if i - a - 1 >= 0 else 1.0
    return pd.Series(c / base - 1, index=np.arange(a - i, b - i + 1))


def betas(B, groups: dict[str, list[int]] | None = None, key: str = "SOX", since: str = "2023-01-01") -> pd.Series:
    """細族群日報酬（等權）對前一晚美國指標（WORLD[key]）報酬的敏感度 β。"""
    groups = groups or group_index(B)
    keys = list(B.meta.get("world_keys") or [])
    if key not in keys or "WORLD" not in B.a:
        return pd.Series(dtype=float)
    days = pd.DatetimeIndex(B.meta["days"])
    W = B.a["WORLD"][:, keys.index(key)].astype(float)            # 已對齊到台股日：當天開盤前最近一次美國收盤
    x = pd.Series(W, index=days).pct_change()
    R = B.a["R"].astype(float)
    sel = days >= pd.Timestamp(since)
    out = {}
    for g, js in groups.items():
        with np.errstate(invalid="ignore"):
            y = pd.Series(np.nanmean(np.where(np.isfinite(R[:, js]), R[:, js], np.nan), axis=1), index=days)
        d = pd.concat([x, y], axis=1)[sel].dropna()
        d = d[d.iloc[:, 0].abs() > 0]
        out[g] = float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / d.iloc[:, 0].var()) if len(d) > 40 else np.nan
    return pd.Series(out)


def depth_label(x: int) -> str:
    return {3: "深", 2: "中", 1: "淺", 0: "幾乎無", -1: "淺", -2: "中", -3: "深"}.get(int(x), "")


# ------------------------------------------------------------------ v3.2 AI 晶片陣營輪動
def camp_index(B, exclusive: bool = True) -> dict[str, list[int]]:
    """陣營 → 代表股欄位；exclusive＝只算只在這個陣營的公司；另加 shared（同時供應兩個以上陣營）。"""
    tix = {t: j for j, t in enumerate(B.tickers)}
    out = {c: [tix[t] for t, _ in supply.camp_members(c, exclusive) if t in tix] for c in supply.CAMPS}
    out["shared"] = [tix[t] for t in supply.shared_members() if t in tix]
    return {k: v for k, v in out.items() if v}


def camp_paths(B, days: int = 120) -> pd.DataFrame:
    """近 days 個交易日，各陣營（只在該陣營的代表股）等權相對全市場的累積超額（起點＝0）。"""
    dd = pd.DatetimeIndex(B.meta["days"])
    R = B.a["R"].astype(float)
    m = _market(B)
    a = max(0, len(dd) - days)
    out = {}
    for c, js in camp_index(B).items():
        with np.errstate(invalid="ignore"):
            g = np.nanmean(np.where(np.isfinite(R[a:, js]), R[a:, js], np.nan), axis=1)
        out[c] = np.cumprod(1 + np.nan_to_num(g - m[a:])) - 1
    return pd.DataFrame(out, index=dd[a:])


def us_moves(B, keys=("NVDA", "AMD", "AVGO", "GOOGL", "SOX")) -> pd.DataFrame:
    """美國指標股前一晚、5 日、20 日漲跌（WORLD 有資料時；每日更新的國際同業）。"""
    wk = list(B.meta.get("world_keys") or [])
    if "WORLD" not in B.a or not wk:
        return pd.DataFrame()
    W = pd.DataFrame(B.a["WORLD"].astype(float), index=pd.DatetimeIndex(B.meta["days"]), columns=wk).ffill()
    rows = []
    for k in keys:
        if k not in W or W[k].notna().sum() < 25:
            continue
        x = W[k].dropna()
        rows.append({"key": k, "r1": x.iloc[-1] / x.iloc[-2] - 1, "r5": x.iloc[-1] / x.iloc[-6] - 1, "r20": x.iloc[-1] / x.iloc[-21] - 1})
    return pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()
