"""v2.0：ETF／大盤官方合計解析、ETF 含息報酬、止穩檢查表、反彈候選、資金配置。"""
import numpy as np
import pandas as pd

from tests.test_engine import world  # noqa: F401  (示範資料 fixture)


def test_v20_parsers():
    from kosto.sources.official import parse_twse_bfi82u, parse_twse_etf, parse_twse_margin_ms, parse_tpex_price, ETF_RE
    etf = {"stat": "OK", "tables": [{"title": "每日收盤行情(ETF)", "fields": [
        "證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差",
        "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比"], "data": [
        ["0050", "元大台灣50", "1,000", "10", "100,600", "101", "102", "100", "100.60", "<p style= color:green>-</p>", "3.35",
         "", "", "", "", "0.00"],
        ["00878", "國泰永續高股息", "2,000", "20", "67,740", "34", "34.2", "33.5", "33.87", "<p style= color:red>+</p>", "0.12",
         "", "", "", "", "0.00"],
        ["00679B", "元大美債20年", "1", "1", "26", "26", "26", "26", "26.54", "X", "0.00", "", "", "", "", "0.00"],
        ["2330", "台積電", "1", "1", "1", "1", "1", "1", "1", " ", "0.00", "", "", "", "", "0.00"]]}]}
    f = parse_twse_etf(etf, "2026-11-03")
    assert list(f["ticker"]) == ["0050", "00878", "00679B"]
    assert f.loc[0, "change"] == -3.35 and f.loc[1, "change"] == 0.12 and f.loc[2, "event"] == "X" and np.isnan(f.loc[2, "change"])
    assert ETF_RE.match("00631L") and ETF_RE.match("00400A") and not ETF_RE.match("2330")
    bfi = {"stat": "OK", "fields": ["單位名稱", "買進金額", "賣出金額", "買賣差額"], "data": [
        ["自營商(自行買賣)", "7,516,969,419", "16,221,368,737", "-8,704,399,318"],
        ["自營商(避險)", "43,662,243,819", "94,012,185,056", "-50,349,941,237"],
        ["投信", "28,084,455,473", "9,624,296,915", "18,460,158,558"],
        ["外資及陸資(不含外資自營商)", "353,895,158,410", "461,201,878,767", "-107,306,720,357"],
        ["外資自營商", "0", "0", "0"], ["合計", "433,158,827,121", "581,059,729,475", "-147,900,902,354"]]}
    b = parse_twse_bfi82u(bfi)
    assert round(b["foreign"] / 1e8) == -1073 and round(b["trust"] / 1e8) == 185 and round(b["dealer"] / 1e8) == -591
    ms = {"stat": "OK", "tables": [{"title": "信用交易統計", "fields": ["項目", "買進", "賣出", "現金(券)償還", "前日餘額", "今日餘額"],
                                    "data": [["融資(交易單位)", "377,983", "450,131", "16,103", "9,571,090", "9,482,839"],
                                             ["融券(交易單位)", "31,896", "103,607", "8,680", "194,930", "257,961"],
                                             ["融資金額(仟元)", "21,189,249", "41,240,308", "2,873,840", "622,699,365", "599,774,466"]]}, {}]}
    m = parse_twse_margin_ms(ms)
    assert m["margin_lots"] == 9482839 and round(m["margin_amt"] / 1e8) == 5998 and m["short_lots"] == 257961
    tp = {"tables": [{"fields": ["代號", "名稱", "收盤 ", "漲跌", "開盤 ", "最高 ", "最低", "成交股數  ", " 成交金額(元)"],
                      "data": [["00679B", "元大美債20年", "26.54", "+0.14", "26.49", "26.60", "26.46", "37,112,000", "984,797,430"],
                               ["6488", "環球晶", "500", "+1", "499", "501", "498", "1,000", "500,000"]]}]}
    t = parse_tpex_price(tp, "2026-11-03", keep=ETF_RE)
    assert list(t["ticker"]) == ["00679B"] and t.loc[0, "change"] == 0.14


def _etf_px():
    d = pd.bdate_range("2023-01-02", periods=800)
    rows = []
    # A：每日 +0.1%，第 300 天除息 1 元（參考價＝前收 − 1）
    pa = 50.0
    for i, x in enumerate(d):
        if i == 300:
            ref = pa - 1.0
            close = ref * 1.001
            rows.append({"date": x, "ticker": "0056", "board": "TWSE", "name": "元大高股息", "close": close, "change": close - ref,
                         "event": "", "amount": 5e8})
            pa = close
            continue
        close = pa * 1.001
        rows.append({"date": x, "ticker": "0056", "board": "TWSE", "name": "元大高股息", "close": close, "change": close - pa,
                     "event": "", "amount": 5e8})
        pa = close
    # B：第 400 天 1 拆 4（參考價＝前收 / 4，漲跌價差相對參考價）
    pb = 200.0
    for i, x in enumerate(d):
        ref = pb / 4 if i == 400 else pb
        close = ref * 1.0005
        rows.append({"date": x, "ticker": "0050", "board": "TWSE", "name": "元大台灣50", "close": close, "change": close - ref,
                     "event": "", "amount": 30e8})
        pb = close
    px = pd.DataFrame(rows)
    div = pd.DataFrame([{"date": d[300], "ticker": "0056", "board": "TWSE", "before": float(px[(px.ticker == "0056")].close.iloc[299]),
                         "ref": float(px[(px.ticker == "0056")].close.iloc[299]) - 1.0}])
    return px, div


def test_v20_etf_total_return_dividend_and_split():
    from kosto import etf as E
    px, div = _etf_px()
    r, close = E.returns(px, div)
    assert abs(r["0056"].iloc[300] - 0.001) < 1e-9                       # 除息日：含息報酬＝當天漲幅，不是 −2%
    assert abs(r["0050"].iloc[400] - 0.0005) < 1e-9                      # 分割日：不是 −75%
    m = E.metrics(r, close, px, div)
    a = m.set_index("ticker")
    assert abs(a.loc["0056", "1年"] - (1.001 ** 252 - 1)) < 1e-6
    assert abs(a.loc["0050", "3年"] - (1.0005 ** 252 - 1)) < 1e-6       # 3 年為年化
    assert a.loc["0056", "配息"] is False or a.loc["0056", "配息"] == False  # noqa: E712  除息在 500 天前 → 近 18 個月沒有配息
    assert a.loc["0050", "type"] == "國內股票" and E.classify("00646", "元大S&P500") == "海外股票"
    assert E.classify("00631L", "元大台灣50正2") == "槓桿反向" and E.classify("00679B", "元大美債20年") == "債券"
    assert E.classify("0055", "元大MSCI金融") == "國內股票" and E.classify("00400A", "主動國泰動能高息") == "主動式"
    arrays, meta = E.pack(px, div)
    assert arrays["ETF_TR"].shape[1] == 2 and meta["etf_tickers"] and abs(arrays["ETF_TR"][0]).max() > 0


def test_v20_checklist_allocation_and_bundle(world):
    from kosto import market as MK
    m = MK.market_series(world.P)
    c = MK.checklist(m)
    assert c["score"].between(0, 8).all() and len(c) == len(m)
    hs = MK.history_stats(m, c)
    assert set(hs["buckets"]) == {"dev", "val", "all"}
    st = MK.state(m, c)
    assert st["regime"] in ("多頭", "盤整", "拉回", "修正", "空頭／深度修正")
    a = MK.allocation("穩健型", {"corr": True, "score": 1})
    b = MK.allocation("穩健型", {"corr": True, "score": 4})
    assert abs(a["etf"] - 0.40) < 1e-9 and abs(b["etf"] - 0.20) < 1e-9 and abs(a["etf"] + a["stock"] - 1) < 1e-9
    from kosto.plan import PROFILES
    assert all(abs(MK.PROFILES[k] - PROFILES[k].get("0050", 0)) < 1e-9 for k in PROFILES)      # 兩頁一致
    w = MK.tilt_weights(PROFILES["保守型"], 0.2)
    assert abs(sum(w.values()) - 1) < 1e-9 and abs(w["0050"] - 0.7) < 1e-9 and abs(w["mid"] / w["long"] - 2 / 3) < 1e-9
    # 下跌 12% 的序列：修正期判定與事件
    tx = pd.Series(np.r_[np.linspace(100, 120, 100), np.linspace(120, 105, 30)], index=pd.bdate_range("2024-01-01", periods=130))
    ep = MK.episodes(tx)
    assert ep and ep[-1]["ongoing"] and ep[-1]["max_dd"] < -0.10
    B = world.B
    assert "mkt_stats" in B.meta and "data_dates" in B.meta and B.meta["data_dates"]["price"]
    cols = B.meta["q_mkt_cols"]
    assert "ab20" in cols and "mlots" in cols and B.a["Q_MKT"].shape[1] == len(cols)
    assert isinstance(B.meta.get("rebound", []), list)


def test_v21_mops_income_fin_pack_peers_and_status():
    from kosto.sources.official import parse_mops_income
    from kosto.bundle import fin_pack
    html = ("<table><tr><th>公司代號</th><th>公司名稱</th><th>營業收入</th><th>營業成本</th><th>營業毛利（毛損）</th>"
            "<th>營業毛利（毛損）淨額</th><th>營業利益（損失）</th><th>本期淨利（淨損）</th><th>淨利（淨損）歸屬於母公司業主</th>"
            "<th>基本每股盈餘（元）</th></tr>"
            "<tr><td>2344</td><td>華邦電</td><td>98,096,329</td><td>38,036,047</td><td>60,060,282</td><td>60,060,282</td>"
            "<td>41,529,830</td><td>34,316,577</td><td>34,431,342</td><td>7.65</td></tr></table>"
            "<table><tr><th>公司代號</th><th>公司名稱</th><th>利息淨收益</th><th>本期稅後淨利（淨損）</th><th>基本每股盈餘（元）</th></tr>"
            "<tr><td>2801</td><td>彰銀</td><td>15,221,502</td><td>11,314,353</td><td>0.98</td></tr></table>")
    f = parse_mops_income(html, 2026, 2).set_index("ticker")
    assert abs(f.loc["2344", "revenue"] - 98096329e3) < 1 and f.loc["2344", "eps"] == 7.65
    assert abs(f.loc["2344", "net"] - 34431342e3) < 1 and f.loc["2344", "yq"] == 20262   # 歸屬母公司優先
    assert pd.isna(f.loc["2801"].get("revenue")) and abs(f.loc["2801", "net"] - 11314353e3) < 1      # 銀行：無營收欄、淨利照抓
    fin = pd.DataFrame([{"ticker": "2344", "yq": 20261, "revenue": 40e9, "gross": 20e9, "op": 12e9, "net": 10e9, "eps": 2.5},
                        {"ticker": "2344", "yq": 20262, "revenue": 98e9, "gross": 60e9, "op": 41e9, "net": 34e9, "eps": 7.65},
                        {"ticker": "2408", "yq": 20262, "revenue": 50e9, "gross": 25e9, "op": 10e9, "net": 8e9, "eps": 2.0}])
    arr, lab = fin_pack(fin, ["2344", "2408"], n=4)
    assert lab[-2:] == [20261, 20262] and lab[:2] == [None, None]
    assert abs(arr[-1, 0, 0] - 58e9) < 1e3 and abs(arr[-1, 0, 4] - 5.15) < 1e-5        # 第二季單季＝累計 − 第一季
    assert np.isnan(arr[-1, 1, 0])                                                       # 2408 缺第一季 → 單季無法計算
    from kosto import peers as PE
    lab_mem = {"group": "電子中游-半導體", "segment": "記憶體（DRAM）", "label": "", "themes": []}
    assert PE.peers_for(lab_mem)[:3] == ["HYNIX", "SAMSUNG", "MU"]
    assert "ZIM" in PE.peers_for({"group": "傳產-交通運輸及航運", "segment": "貨櫃航運"})
    assert PE.peers_for({"group": "傳產-食品", "segment": "食品"}) == []
    from kosto.sources import world as W
    assert "MAERSK" in W.US and "HYNIX" in W.ASIA and W.SYMBOLS["HYNIX"] == "000660.KS"
    # 名單定版判斷：融資券還沒公布，但模型沒用到融資因子 → 定版
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("kcli", Path(__file__).parents[1] / "cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    B = type("B", (), {"meta": {"data_dates": {"price": "2026-11-04", "institutional": "2026-11-04", "margin": "2026-11-03",
                                                "valuation": "2026-11-04"}}})()
    res = {"horizons": {"mid": {"latest": {"weights": {"t_rs120": 0.3, "f_rev_rel": 0.15}}},
                        "long": {"latest": {"weights": {"t_rs120": 0.3}}}}}
    ds = cli.data_status(B, res)
    assert ds["final"] and not ds["complete"] and "定版" in ds["text"]
    res["horizons"]["mid"]["latest"]["weights"]["k_paid"] = 0.05
    ds = cli.data_status(B, res)
    assert not ds["final"] and ds["margin_factors"] == {"mid": ["k_paid"]}
    from kosto import tournament as T
    st = T.log_picks({}, {"protocol": {"last_date": "2026-11-04"}, "horizons": {"mid": {"days": 20, "latest": {"picks": [{"ticker": "2330"}]}}}})
    st = T.log_picks(st, {"protocol": {"last_date": "2026-11-04"}, "horizons": {"mid": {"days": 20, "latest": {"picks": [{"ticker": "2317"}]}}}})
    assert len(st["picks_log"]) == 1 and st["picks_log"][0]["picks"] == ["2317"]          # 同一天以最後一次（完整版）為準
