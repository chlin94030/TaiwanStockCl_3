"""v2.1 國際同業與台股連動主題：每個主題列出國際指標股（Yahoo 代號）與台股對應的細產業關鍵字。

用途（都只做資訊顯示，不進選股模型；理由見 docs/v2.1_linkage_protocol.md）：
  * 國際盤「台股連動精選」：依主題分組的即時報價＋與台股族群的歷史連動。
  * 個股快照「國際同業」：依個股細產業找對應的國際同業，顯示近 1／5／20 日漲跌與連動係數。
"""
from __future__ import annotations

import re

# key: (Yahoo 代號, 中文名, 地區)；地區 us／eu 的收盤在台股收盤之後 → 研究時落後一日使用；asia 同日
GLOBAL = {
    "NVDA": ("NVDA", "輝達", "us"), "AMD": ("AMD", "超微", "us"), "AVGO": ("AVGO", "博通", "us"), "MU": ("MU", "美光", "us"),
    "ASML": ("ASML", "艾司摩爾", "us"), "AMAT": ("AMAT", "應用材料", "us"), "LRCX": ("LRCX", "科林研發", "us"),
    "KLAC": ("KLAC", "科磊", "us"), "ARM": ("ARM", "安謀", "us"), "INTC": ("INTC", "英特爾", "us"), "QCOM": ("QCOM", "高通", "us"),
    "AAPL": ("AAPL", "蘋果", "us"), "MSFT": ("MSFT", "微軟", "us"), "GOOGL": ("GOOGL", "Alphabet", "us"), "AMZN": ("AMZN", "亞馬遜", "us"),
    "META": ("META", "Meta", "us"), "TSLA": ("TSLA", "特斯拉", "us"), "SMCI": ("SMCI", "美超微", "us"), "DELL": ("DELL", "戴爾", "us"),
    "VRT": ("VRT", "Vertiv（電源散熱）", "us"), "ANET": ("ANET", "Arista（網通）", "us"), "COHR": ("COHR", "Coherent（光通訊）", "us"),
    "HYNIX": ("000660.KS", "SK海力士", "asia"), "SAMSUNG": ("005930.KS", "三星電子", "asia"),
    "TEL": ("8035.T", "東京威力科創", "asia"), "ADVANTEST": ("6857.T", "愛德萬測試", "asia"),
    "ZIM": ("ZIM", "以星航運", "us"), "MAERSK": ("MAERSK-B.CO", "快桅（Maersk）", "eu"), "HLAG": ("HLAG.DE", "赫伯羅特", "eu"),
    "BDRY": ("BDRY", "乾散貨運價ETF（波羅的海指數代理）", "us"),
}

# 主題：名稱、國際指標、台股細產業／題材關鍵字（比對產業鏈標籤的族群、區段、題材）
THEMES = [
    ("記憶體", ["HYNIX", "SAMSUNG", "MU"], r"記憶體|DRAM|Flash|NAND|NOR"),
    ("AI 晶片與伺服器", ["NVDA", "AMD", "AVGO", "SMCI", "DELL", "VRT"],
     r"伺服器|散熱|液冷|滑軌|機殼|BMC|ASIC|主機板|UPS|不斷電|AI晶片|電源供應"),
    ("晶圓代工與設備", ["TSM", "ASML", "AMAT", "LRCX", "KLAC", "TEL", "ADVANTEST"],
     r"晶圓代工|半導體設備|設備|探針|測試|封測|封裝|廠務|無塵室|量測|矽晶圓|特用化學|特殊氣體"),
    ("網通與光通訊", ["ANET", "COHR", "AVGO"], r"光通訊|網通|交換器|高速傳輸|光纖|CPO|矽光"),
    ("PCB 與載板", ["NVDA", "AVGO", "AMD"], r"PCB|CCL|銅箔|載板|ABF|鑽針"),
    ("蘋果與手機", ["AAPL", "QCOM", "ARM"], r"光學鏡頭|手機|觸控|指紋|CIS|藍牙|IC設計（手機"),
    ("雲端巨頭（AI 資本支出）", ["MSFT", "GOOGL", "AMZN", "META"], r"雲端"),
    ("電動車", ["TSLA"], r"電動車|車用|充電"),
    ("貨櫃航運", ["ZIM", "MAERSK", "HLAG"], r"貨櫃|航運|海運"),
    ("散裝航運", ["BDRY"], r"散裝|散貨"),
]
EXTRA_KEYS = {"TSM"}          # 已在原本的國際市場資料中（台積電 ADR）


def all_keys() -> list[str]:
    return list(GLOBAL)


def region(key: str) -> str:
    return GLOBAL.get(key, ("", "", "us"))[2]


def themes_for(label: dict) -> list[tuple[str, list[str]]]:
    """個股產業鏈標籤（group／segment／label／themes）→ 對應的主題與國際同業。"""
    text = " ".join([str(label.get("group", "")), str(label.get("segment", "")), str(label.get("label", ""))]
                    + [str(x) for x in (label.get("themes") or [])])
    out = []
    for name, keys, pat in THEMES:
        if re.search(pat, text, re.I):
            out.append((name, keys))
    return out


def peers_for(label: dict, n: int = 6) -> list[str]:
    seen = []
    for _, keys in themes_for(label):
        for k in keys:
            if k not in seen:
                seen.append(k)
    return seen[:n]


def tw_members(theme_pat: str, labels: dict) -> list[str]:
    """主題關鍵字 → 台股代號（labels：ticker → 產業鏈標籤）。"""
    out = []
    for t, lab in labels.items():
        text = " ".join([str(lab.get("group", "")), str(lab.get("segment", "")), str(lab.get("label", ""))]
                        + [str(x) for x in (lab.get("themes") or [])])
        if re.search(theme_pat, text, re.I):
            out.append(t)
    return out


def name_of(key: str) -> str:
    if key == "TSM":
        return "台積電ADR"
    return GLOBAL.get(key, (key, key, ""))[1]
