"""v3.4 美股中文對照：產業（GICS）中文、常見公司中文名與一句話說明、台灣供應鏈關聯、選股母體以外的熱門股。

中文名與說明是給「不熟美股」的人快速認識公司用，不影響選股模型。
"""
from __future__ import annotations

SECTOR_ZH = {
    "Information Technology": "資訊科技", "Communication Services": "通訊服務", "Consumer Discretionary": "非必需消費",
    "Consumer Staples": "必需消費", "Health Care": "醫療保健", "Financials": "金融", "Industrials": "工業",
    "Energy": "能源", "Materials": "原物料", "Utilities": "公用事業", "Real Estate": "不動產", "ETF": "ETF",
}

SUBIND_ZH = {
    "Advertising": "廣告", "Aerospace & Defense": "航太國防", "Agricultural & Farm Machinery": "農業機械",
    "Agricultural Products & Services": "農產品", "Air Freight & Logistics": "航空貨運物流", "Apparel Retail": "服飾零售",
    "Apparel, Accessories & Luxury Goods": "服飾精品", "Application Software": "應用軟體",
    "Asset Management & Custody Banks": "資產管理", "Automobile Manufacturers": "汽車製造", "Automotive Parts & Equipment": "汽車零件",
    "Automotive Retail": "汽車零售", "Biotechnology": "生技", "Broadcasting": "廣播電視", "Broadline Retail": "綜合零售（電商）",
    "Building Products": "建材", "Cable & Satellite": "有線電視衛星", "Cargo Ground Transportation": "陸運貨運", "Casinos & Gaming": "賭場",
    "Commodity Chemicals": "基礎化工", "Communications Equipment": "通訊設備", "Computer & Electronics Retail": "3C 零售",
    "Construction & Engineering": "營建工程", "Construction Machinery & Heavy Transportation Equipment": "工程機械重車",
    "Construction Materials": "建築材料", "Consumer Electronics": "消費電子", "Consumer Finance": "消費金融",
    "Consumer Staples Merchandise Retail": "量販零售", "Copper": "銅礦", "Data Center REITs": "資料中心 REIT",
    "Data Processing & Outsourced Services": "資料處理外包", "Distillers & Vintners": "酒類", "Distributors": "批發通路",
    "Diversified Banks": "大型銀行", "Diversified Support Services": "綜合支援服務", "Electric Utilities": "電力公用",
    "Electrical Components & Equipment": "電機設備", "Electronic Components": "電子零組件",
    "Electronic Equipment & Instruments": "電子儀器", "Electronic Manufacturing Services": "電子代工",
    "Environmental & Facilities Services": "環保設施服務", "Fertilizers & Agricultural Chemicals": "肥料農化",
    "Financial Exchanges & Data": "交易所與金融資訊", "Food Distributors": "食品批發", "Food Retail": "食品零售", "Footwear": "鞋類",
    "Gas Utilities": "天然氣公用", "Gold": "黃金礦業", "Health Care Distributors": "醫藥通路", "Health Care Equipment": "醫療器材",
    "Health Care Facilities": "醫院", "Health Care REITs": "醫療 REIT", "Health Care Services": "醫療服務",
    "Health Care Supplies": "醫療耗材", "Health Care Technology": "醫療科技", "Heavy Electrical Equipment": "重電設備",
    "Home Improvement Retail": "居家修繕零售", "Homebuilding": "建商", "Homefurnishing Retail": "家具零售",
    "Hotel & Resort REITs": "飯店 REIT", "Hotels, Resorts & Cruise Lines": "飯店郵輪", "Household Products": "家用品",
    "Human Resource & Employment Services": "人力資源", "IT Consulting & Other Services": "IT 顧問",
    "Independent Power Producers & Energy Traders": "獨立發電", "Industrial Conglomerates": "工業集團",
    "Industrial Gases": "工業氣體", "Industrial Machinery & Supplies & Components": "工業機械", "Industrial REITs": "倉儲物流 REIT",
    "Insurance Brokers": "保險經紀", "Integrated Oil & Gas": "綜合石油", "Integrated Telecommunication Services": "電信",
    "Interactive Home Entertainment": "電玩遊戲", "Interactive Media & Services": "網路平台", "Internet Services & Infrastructure": "網路基礎服務",
    "Investment Banking & Brokerage": "投資銀行券商", "Leisure Products": "休閒用品", "Life & Health Insurance": "壽險",
    "Life Sciences Tools & Services": "生命科學儀器", "Managed Health Care": "健康保險", "Metal, Glass & Plastic Containers": "容器包材",
    "Movies & Entertainment": "影視娛樂", "Multi-Family Residential REITs": "公寓 REIT", "Multi-Sector Holdings": "控股公司",
    "Multi-Utilities": "綜合公用", "Multi-line Insurance": "綜合保險", "Office REITs": "辦公室 REIT",
    "Oil & Gas Equipment & Services": "油服", "Oil & Gas Exploration & Production": "石油探勘",
    "Oil & Gas Refining & Marketing": "煉油", "Oil & Gas Storage & Transportation": "油氣管線", "Other Specialized REITs": "特殊 REIT",
    "Other Specialty Retail": "專賣零售", "Packaged Foods & Meats": "包裝食品", "Paper & Plastic Packaging Products & Materials": "紙塑包材",
    "Passenger Airlines": "航空", "Passenger Ground Transportation": "叫車運輸", "Personal Care Products": "個人保養",
    "Pharmaceuticals": "製藥", "Property & Casualty Insurance": "產險", "Publishing": "出版", "Rail Transportation": "鐵路",
    "Real Estate Services": "房地產服務", "Regional Banks": "區域銀行", "Reinsurance": "再保險", "Research & Consulting Services": "研究顧問",
    "Restaurants": "餐飲", "Retail REITs": "零售 REIT", "Self-Storage REITs": "迷你倉 REIT",
    "Semiconductor Materials & Equipment": "半導體設備材料", "Semiconductors": "半導體", "Single-Family Residential REITs": "獨棟住宅 REIT",
    "Soft Drinks & Non-alcoholic Beverages": "飲料", "Specialized Consumer Services": "消費服務", "Specialty Chemicals": "特用化學",
    "Steel": "鋼鐵", "Systems Software": "系統軟體", "Technology Distributors": "電子通路",
    "Technology Hardware, Storage & Peripherals": "電腦硬體儲存", "Telecom Tower REITs": "電塔 REIT", "Timber REITs": "林地 REIT",
    "Tobacco": "菸草", "Trading Companies & Distributors": "貿易商", "Transaction & Payment Processing Services": "支付",
    "Water Utilities": "水務", "Wireless Telecommunication Services": "行動電信",
}

# 代號 → (中文名, 一句話說明)
ZH: dict[str, tuple[str, str]] = {
    # ---- 科技巨頭
    "AAPL": ("蘋果", "iPhone、Mac、服務訂閱"), "MSFT": ("微軟", "Windows、Office、Azure 雲端、OpenAI 合作"),
    "NVDA": ("輝達", "AI GPU 龍頭、整櫃 AI 伺服器"), "GOOGL": ("Alphabet（Google）A", "搜尋、YouTube、Google Cloud、TPU"),
    "GOOG": ("Alphabet（Google）C", "同 GOOGL，無投票權股"), "AMZN": ("亞馬遜", "電商、AWS 雲端、自研 AI 晶片"),
    "META": ("Meta", "Facebook、Instagram、AI 資料中心"), "TSLA": ("特斯拉", "電動車、儲能、自駕、機器人"),
    "AVGO": ("博通", "客製 AI 晶片（ASIC）、網通晶片、VMware"), "ORCL": ("甲骨文", "資料庫、雲端、AI 資料中心"),
    "NFLX": ("Netflix", "串流影音"), "CRM": ("Salesforce", "企業客戶管理軟體"), "ADBE": ("Adobe", "Photoshop、設計軟體"),
    "IBM": ("IBM", "企業 IT、混合雲、量子電腦"), "CSCO": ("思科", "網通設備"), "INTU": ("Intuit", "報稅與記帳軟體"),
    "NOW": ("ServiceNow", "企業流程雲端軟體"), "PLTR": ("Palantir", "政府與企業 AI 資料分析"), "UBER": ("Uber", "叫車與外送平台"),
    "SHOP": ("Shopify", "電商開店平台"), "SPOT": ("Spotify", "音樂串流"), "APP": ("AppLovin", "手機廣告 AI 平台"),
    # ---- 半導體與 AI 硬體
    "AMD": ("超微 AMD", "CPU、AI GPU（MI 系列）"), "INTC": ("英特爾", "CPU、晶圓代工"), "QCOM": ("高通", "手機晶片、AI PC"),
    "TXN": ("德州儀器", "類比晶片"), "MU": ("美光", "DRAM、HBM 記憶體"), "AMAT": ("應用材料", "半導體設備"),
    "LRCX": ("科林研發", "蝕刻設備"), "KLAC": ("科磊", "檢測量測設備"), "ADI": ("亞德諾", "類比晶片"),
    "MRVL": ("邁威爾", "客製 AI 晶片、光通訊晶片"), "ARM": ("安謀", "晶片架構授權"), "ASML": ("艾司摩爾", "EUV 微影設備（荷蘭）"),
    "TSM": ("台積電 ADR", "晶圓代工龍頭（美股掛牌）"), "SMCI": ("美超微", "AI 伺服器"), "DELL": ("戴爾", "伺服器、PC"),
    "HPE": ("慧與", "伺服器、網通"), "ANET": ("Arista", "資料中心交換器"), "VRT": ("維諦技術", "資料中心電源與散熱"),
    "COHR": ("Coherent", "光通訊模組、雷射"), "MPWR": ("芯源", "電源管理晶片"), "NXPI": ("恩智浦", "車用晶片"),
    "ON": ("安森美", "車用功率半導體"), "MCHP": ("微芯", "微控制器"), "SNPS": ("新思", "晶片設計軟體 EDA"),
    "CDNS": ("益華", "晶片設計軟體 EDA"), "WDC": ("威騰", "硬碟"), "STX": ("希捷", "硬碟"), "SNDK": ("SanDisk", "快閃記憶體"),
    "TER": ("泰瑞達", "半導體測試設備"), "GLW": ("康寧", "玻璃、光纖"), "APH": ("安費諾", "連接器"),
    "TEL": ("泰科電子", "連接器、感測器"), "JBL": ("捷普", "電子代工"), "CIEN": ("Ciena", "光通訊網路設備"),
    "ALAB": ("Astera Labs", "AI 伺服器傳輸晶片"), "CRDO": ("Credo", "高速傳輸線與晶片"), "LITE": ("Lumentum", "光通訊雷射元件"),
    "FN": ("Fabrinet", "光模組代工"), "CLS": ("Celestica", "AI 伺服器、交換器代工"), "AAOI": ("應用光電", "光收發模組"),
    "ONTO": ("Onto", "半導體檢測設備"), "ENTG": ("Entegris", "半導體材料"), "AMKR": ("艾克爾", "封裝測試"),
    "GFS": ("格羅方德", "晶圓代工"), "UMC": ("聯電 ADR", "晶圓代工（美股掛牌）"), "ASX": ("日月光 ADR", "封裝測試（美股掛牌）"),
    "HIMX": ("奇景光電", "顯示驅動 IC（台灣公司）"), "SIMO": ("慧榮", "記憶體控制晶片（台灣公司）"), "IMOS": ("南茂 ADR", "記憶體封測"),
    "CRWV": ("CoreWeave", "AI 雲端算力出租"), "NBIS": ("Nebius", "AI 雲端算力"), "IONQ": ("IonQ", "量子電腦"),
    "RGTI": ("Rigetti", "量子電腦"), "QBTS": ("D-Wave", "量子電腦"),
    # ---- 軟體、網路、資安
    "CRWD": ("CrowdStrike", "資安"), "PANW": ("Palo Alto", "資安"), "FTNT": ("Fortinet", "資安防火牆"),
    "ZS": ("Zscaler", "雲端資安"), "NET": ("Cloudflare", "網路加速與資安"), "DDOG": ("Datadog", "雲端監控軟體"),
    "SNOW": ("Snowflake", "雲端資料倉儲"), "MDB": ("MongoDB", "資料庫"), "WDAY": ("Workday", "人資財務軟體"),
    "TEAM": ("Atlassian", "協作軟體"), "ADSK": ("Autodesk", "設計軟體"), "TTD": ("The Trade Desk", "程式化廣告"),
    "DASH": ("DoorDash", "外送平台"), "ABNB": ("Airbnb", "民宿平台"), "BKNG": ("Booking", "訂房平台"),
    "RBLX": ("Roblox", "遊戲平台"), "U": ("Unity", "遊戲引擎"), "PINS": ("Pinterest", "圖片社群"), "SNAP": ("Snap", "社群"),
    "MELI": ("MercadoLibre", "拉美電商與金融"), "SE": ("Sea", "東南亞電商（蝦皮）與遊戲"), "PDD": ("拼多多", "中國電商、Temu"),
    "BABA": ("阿里巴巴", "中國電商與雲端"), "JD": ("京東", "中國電商"), "BIDU": ("百度", "中國搜尋與 AI"),
    "CPNG": ("Coupang", "韓國電商"), "GRAB": ("Grab", "東南亞叫車外送"), "NU": ("Nu", "巴西數位銀行"),
    "DUOL": ("Duolingo", "語言學習 App"), "HOOD": ("Robinhood", "網路券商"), "COIN": ("Coinbase", "加密貨幣交易所"),
    "MSTR": ("Strategy", "持有大量比特幣的公司"), "SOFI": ("SoFi", "數位銀行"), "AFRM": ("Affirm", "先買後付"),
    "PYPL": ("PayPal", "線上支付"), "XYZ": ("Block", "支付（Square、Cash App）"), "TOST": ("Toast", "餐廳 POS"),
    # ---- 金融
    "BRK-B": ("波克夏 B", "巴菲特的控股公司"), "JPM": ("摩根大通", "美國最大銀行"), "V": ("Visa", "信用卡網路"),
    "MA": ("萬事達卡", "信用卡網路"), "BAC": ("美國銀行", "大型銀行"), "WFC": ("富國銀行", "大型銀行"), "C": ("花旗", "大型銀行"),
    "GS": ("高盛", "投資銀行"), "MS": ("摩根士丹利", "投資銀行、財富管理"), "SCHW": ("嘉信理財", "券商"),
    "BLK": ("貝萊德", "全球最大資產管理（iShares ETF）"), "AXP": ("美國運通", "信用卡"), "SPGI": ("標普全球", "信評與指數"),
    "BX": ("黑石", "私募基金"), "KKR": ("KKR", "私募基金"), "APO": ("Apollo", "私募基金"), "IBKR": ("盈透", "網路券商"),
    "CME": ("芝商所", "期貨交易所"), "ICE": ("洲際交易所", "交易所（紐約證交所母公司）"), "PGR": ("前進保險", "汽車保險"),
    "COF": ("第一資本", "信用卡銀行"),
    # ---- 醫療
    "LLY": ("禮來", "減重藥、糖尿病藥"), "NVO": ("諾和諾德", "減重藥（丹麥）"), "UNH": ("聯合健康", "健康保險"),
    "JNJ": ("嬌生", "製藥、醫材"), "ABBV": ("艾伯維", "製藥"), "MRK": ("默克", "製藥（癌症藥）"), "PFE": ("輝瑞", "製藥"),
    "AMGN": ("安進", "生技"), "GILD": ("吉利德", "生技"), "VRTX": ("福泰", "生技"), "REGN": ("再生元", "生技"),
    "ISRG": ("直覺手術", "達文西手術機器人"), "TMO": ("賽默飛", "生命科學儀器"), "ABT": ("亞培", "醫材、營養品"),
    "DHR": ("丹納赫", "生命科學"), "BSX": ("波士頓科學", "醫材"), "SYK": ("史賽克", "骨科醫材"), "AZN": ("阿斯特捷利康", "製藥（英國）"),
    "HIMS": ("Hims & Hers", "線上醫療與減重藥"), "TEM": ("Tempus AI", "AI 醫療數據"),
    # ---- 消費
    "WMT": ("沃爾瑪", "量販零售"), "COST": ("好市多", "會員制量販"), "HD": ("家得寶", "居家修繕零售"), "MCD": ("麥當勞", "速食"),
    "KO": ("可口可樂", "飲料"), "PEP": ("百事", "飲料零食"), "PG": ("寶僑", "家用品"), "NKE": ("Nike", "運動鞋服"),
    "SBUX": ("星巴克", "咖啡"), "CMG": ("Chipotle", "墨西哥速食"), "LOW": ("勞氏", "居家修繕零售"), "TGT": ("Target", "零售"),
    "TJX": ("TJX", "折扣服飾零售"), "LULU": ("Lululemon", "運動服飾"), "DIS": ("迪士尼", "影視、樂園、串流"),
    "CVNA": ("Carvana", "線上二手車"), "RCL": ("皇家加勒比", "郵輪"), "MAR": ("萬豪", "飯店"), "PM": ("菲利普莫里斯", "菸草"),
    "CELH": ("Celsius", "機能飲料"), "CAVA": ("CAVA", "地中海速食"), "DKNG": ("DraftKings", "運動博彩"),
    "RIVN": ("Rivian", "電動車"), "GM": ("通用汽車", "汽車"), "F": ("福特", "汽車"), "ROKU": ("Roku", "串流電視平台"),
    # ---- 工業、能源、電力
    "GE": ("GE 航太", "飛機引擎"), "GEV": ("GE Vernova", "燃氣渦輪、電網設備"), "CAT": ("開拓重工", "工程機械、發電機組"),
    "BA": ("波音", "飛機"), "RTX": ("雷神技術", "航太國防"), "LMT": ("洛克希德馬丁", "國防"), "HON": ("漢威聯合", "工業自動化"),
    "ETN": ("伊頓", "電力設備（資料中心配電）"), "DE": ("迪爾", "農業機械"), "UNP": ("聯合太平洋", "鐵路"), "UPS": ("UPS", "物流"),
    "PWR": ("廣達服務", "電網工程（非台灣廣達）"), "EME": ("Emcor", "機電工程"), "AXON": ("Axon", "警用電擊槍與攝影機"),
    "CEG": ("星座能源", "核能發電"), "VST": ("Vistra", "發電（資料中心用電）"), "NRG": ("NRG", "發電"), "TLN": ("Talen", "核能發電"),
    "OKLO": ("Oklo", "小型核反應爐"), "SMR": ("NuScale", "小型模組化核電"), "CCJ": ("Cameco", "鈾礦"), "BE": ("Bloom Energy", "燃料電池發電"),
    "NEE": ("新紀元能源", "電力、再生能源"), "XOM": ("埃克森美孚", "石油"), "CVX": ("雪佛龍", "石油"), "LIN": ("林德", "工業氣體"),
    "FCX": ("自由港", "銅礦"), "NEM": ("紐蒙特", "金礦"), "RKLB": ("Rocket Lab", "小型火箭、衛星"), "ASTS": ("AST SpaceMobile", "衛星直連手機"),
    "SPCX": ("SpaceX", "火箭、Starlink 衛星網路"), "JOBY": ("Joby", "電動飛行計程車"), "ACHR": ("Archer", "電動飛行計程車"),
    "GRMN": ("Garmin", "GPS 穿戴裝置（台灣生產）"), "NVMI": ("Nova", "半導體量測設備（以色列）"), "MTSI": ("MACOM", "射頻與光通訊晶片"),
    "RMBS": ("Rambus", "記憶體介面晶片"), "QRVO": ("Qorvo", "射頻晶片"), "SWKS": ("思佳訊", "射頻晶片"), "TRI": ("湯森路透", "資訊服務"),
    "FER": ("Ferrovial", "基礎建設（西班牙）"), "CCEP": ("可口可樂歐洲太平洋", "可口可樂裝瓶商"), "PH": ("派克漢尼汾", "工業動力控制"), "NVT": ("nVent", "資料中心液冷與電氣"),
    "POWL": ("Powell", "電力配電設備"), "MOD": ("Modine", "散熱（資料中心冷卻）"), "TT": ("特靈", "空調（資料中心冷卻）"),
    # ---- 通訊
    "T": ("AT&T", "電信"), "VZ": ("威訊", "電信"), "TMUS": ("T-Mobile", "電信"), "CMCSA": ("康卡斯特", "有線電視、NBC"),
    "WBD": ("華納兄弟探索", "影視"), "EA": ("藝電", "電玩"), "TTWO": ("Take-Two", "電玩（GTA）"),
    # ---- ETF（參考用，不進選股）
    "SPY": ("標普 500 ETF", "美國大型股 500 檔"), "VOO": ("Vanguard 標普 500", "同 SPY，費用較低"),
    "QQQ": ("那斯達克 100 ETF", "科技成長股 100 檔"), "DIA": ("道瓊 ETF", "30 檔藍籌"), "IWM": ("羅素 2000 ETF", "美國小型股"),
    "SMH": ("VanEck 半導體 ETF", "半導體 25 檔（含台積電 ADR）"), "SOXX": ("iShares 半導體 ETF", "半導體 30 檔"),
    "VTI": ("Vanguard 全市場", "美國全部上市股"), "XLK": ("科技類股 ETF", "標普科技類股"), "XLF": ("金融類股 ETF", "標普金融類股"),
    "XLE": ("能源類股 ETF", "標普能源類股"), "XLV": ("醫療類股 ETF", "標普醫療類股"), "XLY": ("非必需消費 ETF", "標普非必需消費"),
    "XLP": ("必需消費 ETF", "標普必需消費"), "XLI": ("工業類股 ETF", "標普工業類股"), "XLU": ("公用事業 ETF", "標普公用事業"),
    "XLB": ("原物料 ETF", "標普原物料"), "XLRE": ("不動產 ETF", "標普不動產"), "XLC": ("通訊服務 ETF", "標普通訊服務"),
    "TLT": ("20 年美債 ETF", "長天期美國公債"), "GLD": ("黃金 ETF", "黃金"),
}

# 標普 500 以外、但台灣投資人常關注或成交熱絡的股票（加入每日母體；仍需通過股價與成交金額門檻）
EXTRAS = ["ASML", "ARM", "TSM", "SHOP", "SPOT", "MELI", "SE", "PDD", "BABA", "JD", "BIDU", "CPNG", "GRAB", "NU", "AZN", "NVO",
          "MRVL", "MSTR", "SNOW", "NET", "ZS", "MDB", "TEAM", "RBLX", "U", "PINS", "SNAP", "SOFI", "AFRM", "DUOL", "TOST", "CAVA",
          "HIMS", "TEM", "CELH", "ROKU", "RIVN", "DKNG", "ALAB", "CRDO", "LITE", "FN", "CLS", "AAOI", "ONTO", "ENTG", "AMKR", "GFS",
          "UMC", "ASX", "HIMX", "SIMO", "IMOS", "CRWV", "NBIS", "IONQ", "RGTI", "QBTS", "OKLO", "SMR", "CCJ", "BE", "TLN", "RKLB",
          "ASTS", "SPCX", "JOBY", "ACHR", "NVT", "POWL", "MOD", "TEL",
          # v3.5 那斯達克 100、費半中不在標普 500 的成分股
          "CCEP", "TRI", "FER", "NVMI", "MTSI", "RMBS", "QRVO", "SWKS"]

# v3.5 目前成分股（瀏覽篩選用；依 2026 年中公開的 ETF 持股整理，約略，以官方為準）。歷史成分股沒有可靠資料 → 不當模型母池
NDX_NOW = ["AAPL", "NVDA", "MSFT", "GOOGL", "GOOG", "AMZN", "MU", "AMD", "TSLA", "META", "WMT", "AVGO", "COST", "PLTR", "CSCO", "INTC",
           "AMAT", "LRCX", "TXN", "PANW", "NFLX", "AMGN", "LIN", "CRWD", "GILD", "KLAC", "PEP", "ADI", "TMUS", "QCOM", "SHOP", "ISRG",
           "MRVL", "ASML", "VRTX", "SNDK", "APP", "ADP", "BKNG", "SBUX", "STX", "MAR", "WDC", "CEG", "ADBE", "REGN", "CSX", "FTNT",
           "INTU", "MNST", "CTAS", "CMCSA", "CDNS", "SNPS", "MDLZ", "ARM", "ABNB", "ORLY", "MELI", "ROST", "HON", "DASH", "AEP", "MPWR",
           "NXPI", "FAST", "DDOG", "IDXX", "ADSK", "PCAR", "CCEP", "PDD", "PYPL", "FANG", "XEL", "BKR", "KDP", "ROP", "EXC", "LITE",
           "TTWO", "TRI", "PAYX", "WDAY", "TER", "GEHC", "ALAB", "MSTR", "ODFL", "RKLB", "MCHP", "DXCM", "CPRT", "KHC", "AXON", "NBIS",
           "CRWV", "WBD", "EA", "FER"]
SOX_NOW = ["NVDA", "AVGO", "MU", "AMD", "TXN", "ADI", "TSM", "ASML", "AMAT", "KLAC", "MRVL", "LRCX", "QCOM", "MPWR", "NXPI", "INTC",
           "COHR", "TER", "ALAB", "MCHP", "ON", "ARM", "CRDO", "GFS", "ENTG", "MTSI", "SWKS", "NVMI", "RMBS", "QRVO"]
SETS = {"sp500": "標普 500", "ndx": "那斯達克 100", "sox": "費城半導體 30", "all": "全部（含熱門股、ADR）"}
ETFS = ["SPY", "VOO", "IVV", "VTI", "DIA", "RSP", "QQQ", "QQQM", "XLK", "IVW", "IWF", "VUG", "SOXX", "SMH", "IJH", "IWD", "IVE",
        "DVY", "SCHD", "IWM", "VNQ", "EEM", "EFA", "VWO", "VEU", "GLD", "IAU", "IEF", "TLT", "AGG", "BND", "TIP",
        "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
SECTOR_ETF = {"XLK": "資訊科技", "XLC": "通訊服務", "XLY": "非必需消費", "XLP": "必需消費", "XLV": "醫療保健", "XLF": "金融",
              "XLI": "工業", "XLE": "能源", "XLB": "原物料", "XLU": "公用事業", "XLRE": "不動產"}
INDEXES = {"^GSPC": "標普 500", "^IXIC": "那斯達克", "^DJI": "道瓊", "^SOX": "費城半導體", "^RUT": "羅素 2000", "^VIX": "VIX 恐慌指數",
           "^TNX": "10 年期美債殖利率", "NQ=F": "那斯達克期貨", "ES=F": "標普期貨"}

# 不在標普 500 名單裡的 EXTRAS：產業別（GICS 中文大類）
EXTRA_SECTOR = {
    "ASML": "資訊科技", "ARM": "資訊科技", "TSM": "資訊科技", "SHOP": "資訊科技", "SPOT": "通訊服務", "MELI": "非必需消費",
    "SE": "非必需消費", "PDD": "非必需消費", "BABA": "非必需消費", "JD": "非必需消費", "BIDU": "通訊服務", "CPNG": "非必需消費",
    "GRAB": "工業", "NU": "金融", "AZN": "醫療保健", "NVO": "醫療保健", "MRVL": "資訊科技", "MSTR": "資訊科技", "SNOW": "資訊科技",
    "NET": "資訊科技", "ZS": "資訊科技", "MDB": "資訊科技", "TEAM": "資訊科技", "RBLX": "通訊服務", "U": "資訊科技",
    "PINS": "通訊服務", "SNAP": "通訊服務", "SOFI": "金融", "AFRM": "金融", "DUOL": "非必需消費", "TOST": "金融", "CAVA": "非必需消費",
    "HIMS": "醫療保健", "TEM": "醫療保健", "CELH": "必需消費", "ROKU": "通訊服務", "RIVN": "非必需消費", "DKNG": "非必需消費",
    "ALAB": "資訊科技", "CRDO": "資訊科技", "LITE": "資訊科技", "FN": "資訊科技", "CLS": "資訊科技", "AAOI": "資訊科技",
    "ONTO": "資訊科技", "ENTG": "資訊科技", "AMKR": "資訊科技", "GFS": "資訊科技", "UMC": "資訊科技", "ASX": "資訊科技",
    "HIMX": "資訊科技", "SIMO": "資訊科技", "IMOS": "資訊科技", "CRWV": "資訊科技", "NBIS": "資訊科技", "IONQ": "資訊科技",
    "RGTI": "資訊科技", "QBTS": "資訊科技", "OKLO": "公用事業", "SMR": "工業", "CCJ": "能源", "BE": "工業", "TLN": "公用事業",
    "RKLB": "工業", "ASTS": "通訊服務", "SPCX": "工業", "JOBY": "工業", "ACHR": "工業", "NVT": "工業", "POWL": "工業",
    "MOD": "非必需消費", "TEL": "資訊科技", "CCEP": "必需消費", "TRI": "工業", "FER": "工業", "NVMI": "資訊科技", "MTSI": "資訊科技",
    "RMBS": "資訊科技", "QRVO": "資訊科技", "SWKS": "資訊科技",
}

# 美股 → 台灣供應鏈（題材／陣營）；由 supply.THEMES 的 us 欄位與 CAMPS 自動整理，這裡補常見的直接客戶關係
TW_LINK_EXTRA = {                 # 只列市場普遍認定、公開報導較多的關係（不是完整供應鏈）
    "AAPL": "蘋果供應鏈：鴻海（組裝）、台積電（晶片）、大立光（鏡頭）、可成（機殼）、臻鼎-KY、台郡（軟板）",
    "TSLA": "常被歸類的特斯拉概念股：和大、貿聯-KY、胡連、乙盛-KY",
    "AMZN": "AWS 自研 AI 晶片 Trainium：世芯-KY（設計服務）",
    "MSFT": "微軟資料中心伺服器：緯穎、鴻海",
    "META": "Meta 資料中心伺服器：緯穎、廣達",
    "DELL": "戴爾伺服器與 PC：鴻海、緯創",
    "SMCI": "美超微：勤誠（伺服器機殼）",
    "HPE": "慧與伺服器：英業達、鴻海",
    "MU": "美光：台灣有大型 DRAM 廠（台中、桃園）",
    "TSM": "台積電本尊（2330）",
    "UMC": "聯電本尊（2303）", "ASX": "日月光投控本尊（3711）",
    "SPCX": "SpaceX：昇達科、華通、啟碁、敬鵬（有公開供應紀錄）",
}


def zh_name(t: str) -> str:
    return ZH.get(t, ("", ""))[0]


def zh_desc(t: str) -> str:
    return ZH.get(t, ("", ""))[1]


def sector_zh(sector: str) -> str:
    return SECTOR_ZH.get(sector, sector or "—")


def subind_zh(sub: str) -> str:
    return SUBIND_ZH.get(sub, sub or "")


def tw_links(t: str) -> list[str]:
    """這檔美股和台股有什麼關係（題材／AI 陣營／直接供應鏈）。"""
    from .. import supply as S
    out = []
    if t in TW_LINK_EXTRA:
        out.append(TW_LINK_EXTRA[t])
    for c in S.CAMPS.values():
        if t in str(c.get("us", "")).replace("、", ",").split(","):
            mem = "、".join(x for x, _ in c["members"][:6])
            out.append(f"{c['name']}：{mem}")
    for th in S.THEMES:
        us = [x.strip() for x in str(th.get("us", "")).split("、")]
        if t in us and not str(th.get("id", "")).startswith("camp_"):
            out.append(f"台股題材「{th['name']}」")
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ v3.6 美股 ETF 推薦（docs/v3.6_protocol.md 第二節）
ROLE_ZH = {"core": "核心（長期持有的主力）", "growth": "成長（多一點報酬、多一點波動）", "hedge": "避險（股市大跌時的緩衝）",
           "sat": "衛星（小量、看好再加）"}
ROLE_ORDER = ["core", "growth", "hedge", "sat"]
# 代號 → 中文名、角色、追蹤、費用率（%，2025 年公開資料，約略，以發行商公告為準）、同類替代、台股掛牌同類（台幣、免複委託）
ETF_INFO = {
    "IVV": dict(zh="iShares 標普 500", role="core", idx="S&P 500", er=0.03, alt="VOO 0.03%、SPY 0.0945%（同一指數）", tw="00646 元大S&P500"),
    "VTI": dict(zh="Vanguard 美國全市場", role="core", idx="CRSP 美國全市場（約 3,600 檔）", er=0.03, alt="ITOT", tw=""),
    "DIA": dict(zh="SPDR 道瓊工業", role="core", idx="道瓊 30 檔（股價加權）", er=0.16, alt="", tw="00668 國泰美國道瓊"),
    "RSP": dict(zh="Invesco 標普 500 等權", role="core", idx="S&P 500 等權", er=0.20, alt="", tw=""),
    "QQQ": dict(zh="Invesco 那斯達克 100", role="growth", idx="Nasdaq-100", er=0.20, alt="QQQM 0.15%（同一指數、費用較低）", tw="00662 富邦NASDAQ"),
    "XLK": dict(zh="SPDR 科技類股", role="growth", idx="S&P 500 資訊科技", er=0.08, alt="VGT", tw=""),
    "IVW": dict(zh="iShares 標普 500 成長", role="growth", idx="S&P 500 Growth", er=0.18, alt="VUG 0.04%（CRSP 成長）", tw=""),
    "IWF": dict(zh="iShares 羅素 1000 成長", role="growth", idx="Russell 1000 Growth", er=0.19, alt="", tw=""),
    "GLD": dict(zh="SPDR 黃金", role="hedge", idx="實體黃金", er=0.40, alt="IAU 0.25%、GLDM 0.10%（同為實體黃金）", tw="00635U 元大S&P黃金（期貨型）"),
    "IEF": dict(zh="iShares 7–10 年美債", role="hedge", idx="美國公債 7–10 年", er=0.15, alt="", tw=""),
    "TLT": dict(zh="iShares 20 年以上美債", role="hedge", idx="美國公債 20 年以上", er=0.15, alt="", tw="00679B 元大美債20年"),
    "SOXX": dict(zh="iShares 半導體", role="sat", idx="NYSE 半導體（30 檔）", er=0.35, alt="SMH 0.35%（25 檔、台積電 ADR 比重高）", tw="00830 國泰費城半導體"),
    "IJH": dict(zh="iShares 標普中型 400", role="sat", idx="S&P MidCap 400", er=0.05, alt="", tw=""),
    "IWD": dict(zh="iShares 羅素 1000 價值", role="sat", idx="Russell 1000 Value", er=0.19, alt="VTV 0.04%", tw=""),
    "IVE": dict(zh="iShares 標普 500 價值", role="sat", idx="S&P 500 Value", er=0.18, alt="", tw=""),
    "DVY": dict(zh="iShares 精選高股息", role="sat", idx="道瓊美國精選股息", er=0.38, alt="SCHD 0.06%（不同指數：道瓊美國股息 100）", tw=""),
    "IWM": dict(zh="iShares 羅素 2000 小型股", role="sat", idx="Russell 2000", er=0.19, alt="", tw=""),
    "VNQ": dict(zh="Vanguard 不動產 REIT", role="sat", idx="MSCI 美國 REIT", er=0.13, alt="", tw=""),
    "EEM": dict(zh="iShares 新興市場", role="sat", idx="MSCI 新興市場", er=0.70, alt="VWO 0.07%", tw=""),
    "EFA": dict(zh="iShares 歐澳遠東已開發", role="sat", idx="MSCI EAFE", er=0.32, alt="VEA", tw=""),
    "VWO": dict(zh="Vanguard 新興市場", role="sat", idx="FTSE 新興市場", er=0.07, alt="", tw=""),
    "VEU": dict(zh="Vanguard 美國以外全世界", role="sat", idx="FTSE 全世界（不含美國）", er=0.04, alt="VXUS", tw=""),
}
ROLE_WHY = {
    "core": "三輪研究、100 多種選股組合，18 年都沒有長期贏過 S&P 500 → 美股的主力放大盤 ETF 最實在。",
    "growth": "長期比大盤多賺，但跌起來也更兇（2022 年那斯達克 100 跌 33%）；適合能忍受波動、投資期間長的人。",
    "hedge": "和股票走勢相關很低：2008、2020 年股災時上漲或跌得少；代價是平常報酬較低，而且 2022 年升息時債券也大跌。",
    "sat": "看好某個領域再小量加碼；波動大或長期報酬沒有比大盤好，不適合當主力。",
}
MIX = {"保守型": {"IVV": 0.4, "IEF": 0.4, "GLD": 0.2}, "穩健型": {"IVV": 0.6, "QQQ": 0.1, "IEF": 0.2, "GLD": 0.1},
       "積極型": {"IVV": 0.5, "QQQ": 0.3, "SOXX": 0.1, "IEF": 0.1}}
