"""Kosto Lab command line (used by GitHub Actions and for local runs).

  python cli.py demo [--stocks 500] [--quick]            合成資料示範：建資料→bundle→研究→報告
  python cli.py update --group price [--budget 330]       官方/FinMind 資料增量更新（可續傳）
  python cli.py bundle                                    由資料庫建立研究 bundle（因子、母池、報酬）
  python cli.py research [--quick] [--compare old.zip]    完整研究 + 報告 ZIP
  python cli.py status                                    資料庫狀態
  python cli.py auto                                      本機一鍵：建資料(可續傳)→完成後自動研究+報告
  python cli.py progress [--write status/status.json]     資料庫完成度（GitHub 自動續跑用）
  python cli.py pack --group price --out x.tar.gz         打包某資料群組（Actions 用）
  python cli.py unpack x.tar.gz                           解開資料群組
  python cli.py us-build [--prev us_bundle.pkl.gz]        v3.4 美股：抓日線→因子→參考名單（輸出 artifacts/us_bundle.pkl.gz）
  python cli.py morning [--hist h.json] [--prev m.json]   v3.16 台股盤前日報（輸出 artifacts/morning.json、morning_hist.json）
"""
from __future__ import annotations

import argparse
import os
import warnings
import pickle
import sys
import time
from pathlib import Path

from kosto.config import HORIZON_KEYS, load_config

ART = Path("artifacts")
warnings.filterwarnings("ignore", category=RuntimeWarning)   # "mean of empty slice" etc. — harmless, keeps logs readable


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cmd_update(a, cfg):
    from kosto.build import build, progress
    token = os.getenv("FINMIND_TOKEN", "")
    budget = a.budget
    try:   # v3.11：資料庫已完整＝每日增量更新 → 時間上限縮短，避免資料源卡住（重試、配額等待）拖到整晚、讓研究排不上
        pg = float(progress(cfg)["groups"].get(a.group, 0.0))
        cap = float(cfg["data"].get("daily_budget_min", 80))
        if pg >= 0.97 and budget > cap:
            log(f"{a.group} 資料庫已完整（{pg:.1%}）→ 每日增量更新，時間上限 {cap:.0f} 分鐘")
            budget = cap
    except Exception as e:  # noqa: BLE001
        log(f"讀不到資料完成度（{e}），用原本的時間上限 {budget:.0f} 分鐘")
    rep = build(cfg, a.group, token, budget, log=log)
    log(f"完成：{rep}")


def cmd_bundle(a, cfg):
    from kosto.bundle import build_bundle
    from kosto.panel import build_panel
    from kosto.store import Store
    st = Store(cfg["data"]["dir"])
    log(st.summary().to_string())
    P = build_panel(st, start=cfg["data"]["start"], end=cfg["data"].get("end"))
    log(f"panel: {len(P.dates)} 日 × {len(P.tickers)} 檔；品質 {P.quality}")
    B = build_bundle(P, cfg, log=log)
    if not P.synthetic:                                   # v2.1 台股盤中 vs 那斯達克期貨（逐季）；連不到 Yahoo 就略過
        try:
            from kosto.sources import world as W
            B.meta["intraday_link"] = W.intraday_link()
            log(f"盤中連動：{len(B.meta['intraday_link'])} 季")
        except Exception as e:  # noqa: BLE001
            log(f"盤中連動略過：{str(e)[:80]}")
    ART.mkdir(exist_ok=True)
    B.save(ART / "bundle.npz")
    if a.compare_panel:
        from kosto.compare import Daily
        import numpy as np
        d = Daily.from_panel(P, start=cfg["research"]["start"])
        np.savez_compressed(ART / "daily_all.npz", R=d.R, OC=d.OC, OK=d.OK, bench=d.bench,
                            days=np.array([str(x.date()) for x in d.days]), tickers=np.array(d.tickers))
    log(f"bundle 已存：{ART/'bundle.npz'} ({(ART/'bundle.npz').stat().st_size/1e6:.1f} MB)")


def load_prev_auto() -> dict:
    """前一次每日研究的自動晉級狀態（冠軍、連續領先天數、前瞻紀錄）：本機 artifacts 或 GitHub release。"""
    p = ART / "results.pkl"
    try:
        if p.is_file():
            with open(p, "rb") as f:
                return (pickle.load(f).get("res") or {}).get("auto") or {}
    except Exception:  # noqa: BLE001
        pass
    repo = os.getenv("GITHUB_REPOSITORY")
    if repo:
        try:
            from kosto.remote import RESEARCH_TAG, fetch_asset, token_from_env
            d = pickle.loads(fetch_asset(repo, RESEARCH_TAG, "results.pkl", token_from_env()))
            return (d.get("res") or {}).get("auto") or {}
        except Exception as e:  # noqa: BLE001
            log(f"讀取前一次研究狀態失敗（第一次執行時正常）：{e}")
    return {}


def cmd_research(a, cfg):
    from kosto import tournament as T
    from kosto.bundle import Bundle
    from kosto.report import build_zip
    from kosto.research import Research
    B = Bundle.load(a.bundle)
    prev = {} if a.quick else load_prev_auto()
    if cfg["research"].get("auto_reset") and (prev.get("champions") or prev.get("observe")):
        log("research.auto_reset：回到設定檔模型（清除自動晉級的冠軍與前瞻觀察）")   # v3.15 一鍵回復
        prev = {**prev, "champions": {}, "observe": {}, "leader": {}}
    cfg_eff = T.apply(cfg, prev.get("champions"))
    if prev.get("champions"):
        log("使用自動晉級後的模型：" + "；".join(f"{h} {T.describe(p)}" for h, p in prev["champions"].items()))
    R = Research(B, cfg_eff, log=log)
    hz = a.horizons.split(",") if a.horizons else list(HORIZON_KEYS)
    res = R.run_all(hz, quick=a.quick, progress=log, prev_pools=prev.get("pools"))
    auto = dict(prev)
    if (not a.quick and cfg["research"].get("phase_check", True) and not B.meta.get("synthetic")
            and (Path(cfg["data"]["dir"]) / "raw" / "price").is_dir()):
        from kosto.phases import phase_check
        rows = []
        log("冠軍／挑戰者比較：目前資料包…")
        try:
            rows += T.evaluate(R, res, cfg_eff, 0, presample=True, observe=prev.get("observe"))
        except Exception as e:  # noqa: BLE001
            log(f"冠軍／挑戰者比較失敗：{e}")
        log("換股日敏感度＋冠軍／挑戰者：重建 4 個平移起點的研究資料包（約 10～20 分鐘）…")
        res["phases"] = phase_check(cfg_eff, res, log=log, on_bundle=lambda Rk, k: rows.extend(T.evaluate(Rk, res, cfg_eff, k, observe=prev.get("observe"))))
        if rows:
            auto = T.update(prev, rows, res, cfg, cfg_eff)
            for h, ld in (auto.get("leader") or {}).items():
                if ld:
                    log(f"{h} 領先挑戰者：{T.describe(ld['params'], T.base_params(cfg_eff, h))}（+{ld['d_excess']:.1%}，連續 {ld['streak']} 天）")
            for x in auto.get("log", []):
                if x["date"] == res["protocol"]["last_date"]:
                    ev = x.get("event", "promote")
                    log((f"★ {x['h']} 自動換上新模型（明天起生效）：" if ev == "promote" else f"☆ {x['h']} ") + x["text"])
    res["auto"] = T.log_picks(auto, res)
    res["auto"]["pools"] = T.pool_state(prev.get("pools"), res)        # v2.4 母池沿用狀態（隔天讀回）
    res["data_status"] = data_status(B, res)
    try:                                                              # v3.14 每日快照＋雜湊鏈（只新增、不覆寫）
        from kosto import __version__, snapshot
        res["snapshot"] = snapshot.make(res, B.meta, cfg_eff, __version__)
        res["auto"]["snapshots"] = snapshot.append(prev.get("snapshots"), res["snapshot"])
        ok, bad = snapshot.verify(res["auto"]["snapshots"])
        log(f"快照 {res['snapshot']['id']}（第 {res['auto']['snapshots'][-1]['seq']} 版）" + ("" if ok else f"｜⚠ 雜湊鏈在第 {bad} 筆斷開"))
    except Exception as e:  # noqa: BLE001
        log(f"快照略過：{e}")
    try:                                                              # v3.15 更新後自動異常診斷（只警告，不改名單）
        from kosto import diagnose, forward
        res["diagnostics"] = diagnose.run(res, res["auto"].get("picks_log"), prev.get("pools"),
                                          forward.summary(B, res["auto"].get("picks_log") or []))
        for x in res["diagnostics"]:
            log(("⚠ " if x["level"] == "warn" else "") + "診斷：" + x["text"])
    except Exception as e:  # noqa: BLE001
        log(f"異常診斷略過：{e}")
    try:                                                              # v3.16 盤前日報的歷史驗證模型（隔天清晨 cli morning 使用）
        from kosto import morning
        res["morning_model"] = morning.fit(B)
        mm = res["morning_model"]
        if mm.get("error"):
            log(f"盤前日報模型略過：{mm['error']}")
        else:
            t1 = mm["m1"]["test"]
            log(f"盤前日報模型：大盤開盤預估（2024 年後）相關 {t1['corr']:.2f}、方向命中 {t1['hit']:.0%}（{t1['n_hit']} 天）")
    except Exception as e:  # noqa: BLE001
        log(f"盤前日報模型略過：{e}")
    try:
        res["linkage_ic"] = linkage_ic(R, res)                       # v2.1 連動因子逐年有效性（監控用，不進模型）
    except Exception as e:  # noqa: BLE001
        log(f"連動因子監控略過：{e}")
    try:
        from kosto import watch
        res["watch"] = watch.evaluate(R, res)                        # v3.12 觀察名單前瞻追蹤（影子模型；不影響選股）
        for x in res["watch"]["errors"]:
            log(f"觀察名單：{x}")
    except Exception as e:  # noqa: BLE001
        log(f"觀察名單前瞻追蹤略過：{e}")
    log("資料狀態：" + res["data_status"]["text"])
    comparison = None
    if a.compare:
        comparison = run_compare(a.compare, res, B, cfg_eff)
    ART.mkdir(exist_ok=True)
    with open(ART / "results.pkl", "wb") as f:
        pickle.dump({"res": res, "cfg": cfg_eff, "comparison": comparison}, f, protocol=4)
    (ART / "report.zip").write_bytes(build_zip(res, cfg_eff, comparison))
    log(f"報告：{ART/'report.zip'}  ({res['elapsed']:.0f}s)")


MARGIN_FACTORS = {"c_margin_chg20", "c_short_ratio", "k_hands", "k_paid", "k_calm", "k_egg"}   # 用到融資券資料的因子


def data_status(B, res) -> dict:
    """v2.1 今日名單是否已定版：融資券（約 21:00 公布）尚未更新時，若各期間模型都沒用到融資相關因子，名單不會再變。"""
    dd = B.meta.get("data_dates") or {}
    price = dd.get("price")
    late = {k: dd.get(k) for k in ("institutional", "margin", "valuation") if dd.get(k) and price and dd.get(k) < price}
    used = {}
    for h, hb in res.get("horizons", {}).items():
        w = (hb.get("latest") or {}).get("weights") or {}
        m = [f for f, x in w.items() if f in MARGIN_FACTORS and abs(x) > 1e-6]
        if m:
            used[h] = m
    complete = not late
    final = complete or (set(late) == {"margin"} and not used)
    if complete:
        text = "資料完整（收盤、法人、融資券、本益比都是同一天），名單已定版"
    elif final:
        text = f"融資券資料還沒公布（目前到 {late['margin']}），但模型沒有用到融資因子 → 名單已定版"
    else:
        miss = "、".join({"institutional": "三大法人", "margin": "融資券", "valuation": "本益比"}[k] + f"（到 {v}）" for k, v in late.items())
        text = f"初版：{miss} 還沒更新，晚上 21:40 更新後名單可能微調"
    return {"price": price, "late": late, "margin_factors": used, "complete": complete, "final": final, "text": text}


LINK_FACTORS = ["g_us_beta", "g_us_lead", "g_asia_lead", "t_ind_mom", "g_seg_mom20", "g_seg_lag5", "g_seg_breadth5"]


def linkage_ic(R, res) -> dict:
    """中線（每日研究選定的母池）：各連動因子與之後 20 日報酬的等級相關（IC），逐年平均。"""
    import numpy as np
    import pandas as pd
    hb = res["horizons"].get("mid")
    if hb is None:
        return {}
    rule, size = hb["chosen_pool"]["rule"], int(hb["chosen_pool"]["size"])
    ic = R.ic(rule, size, "mid")
    fac = list(R.B.factors)
    yrs = pd.Series(R.B.sig_dates.year)
    out = {}
    for f in LINK_FACTORS:
        if f in fac:
            s = pd.Series(ic[:, fac.index(f)]).groupby(yrs).mean()
            out[f] = {int(y): (float(v) if np.isfinite(v) else None) for y, v in s.items()}
    return {"pool": f"{rule}{size}", "ic": out}


def run_compare(paths, res, B, cfg):
    import numpy as np
    import pandas as pd
    from kosto.compare import Daily, head_to_head, parse_external
    frames = []
    for p in paths:
        frames.append(parse_external(p, Path(p).read_bytes()))
    ext = pd.concat(frames, ignore_index=True)
    daily = Daily.from_bundle(B)
    if (ART / "daily_all.npz").is_file():
        z = np.load(ART / "daily_all.npz")
        daily = Daily(pd.DatetimeIndex(z["days"]), list(z["tickers"]), z["R"], z["OC"], z["OK"], z["bench"])
    ours = {r["days"]: r["tranches"] for r in res["horizons"].values()}
    per, summ = head_to_head(daily, ext, ours, cfg["costs"])
    log(summ.to_string())
    return per, summ


def cmd_demo(a, cfg):
    from kosto.sources.synthetic import write_demo
    from kosto.store import Store
    d = Path(cfg["data"]["dir"])
    if a.fresh and d.exists():
        import shutil
        shutil.rmtree(d)
    st = Store(d)
    if not (d / "SYNTHETIC_DEMO_DATA").exists():
        log(f"產生合成資料 {a.stocks} 檔…")
        log(str(write_demo(st, a.stocks, a.seed)))
    a.compare_panel = False
    cmd_bundle(a, cfg)
    a.bundle = str(ART / "bundle.npz")
    a.compare = None
    cmd_research(a, cfg)


def cmd_summary(a, cfg):
    """Markdown summary (GitHub Actions job summary)."""
    with open(ART / "results.pkl", "rb") as f:
        res = pickle.load(f)["res"]
    lines = ["## Kosto Lab 研究摘要", "", f"資料至 {res['protocol']['last_date']}｜盲測起點 {res['protocol']['holdout_start']}", "",
             "| 期間 | 母池 | 模型 | 開發期CAGR | 0050 | 盲測CAGR | 0050 | 市場階段 | 今日推薦 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for h, r in res["horizons"].items():
        e, lt = r["eval"], r["latest"]
        f = lambda x: "–" if x is None or x != x else f"{x:+.1%}"  # noqa: E731
        picks = " ".join(f"{x['ticker']}{x['name']}" for x in lt["picks"][:10])
        lines.append(f"| {r['label']} | {r['chosen_pool']['rule']}{r['chosen_pool']['size']} | {r['chosen']['variant']} | "
                     f"{f(e.get('dev_cagr'))} | {f(e.get('dev_bench_cagr'))} | {f(e.get('hold_cagr'))} | "
                     f"{f(e.get('hold_bench_cagr'))} | {lt['market_stage']} | {picks} |")
    print("\n".join(lines))


def latest_dates(cfg) -> dict:
    """各資料集在本機資料庫的最新日期（只讀近 40 天，很快）。"""
    import pandas as pd
    from kosto.store import Store
    out = {}
    try:
        st = Store(cfg["data"]["dir"])
        since = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize() - pd.Timedelta(days=40)
        for ds in ("price", "institutional", "margin", "valuation"):
            try:
                f = st.read(ds, since, None, columns=["date"])
                out[ds] = str(pd.Timestamp(f["date"].max()).date()) if len(f) else None
            except Exception:  # noqa: BLE001
                out[ds] = None
    except Exception:  # noqa: BLE001
        pass
    return out


def write_status(cfg, path, extra=None):
    import json
    from kosto.build import progress
    st = progress(cfg)
    st.update(extra or {})
    st["latest"] = latest_dates(cfg)                       # v1.9.1 備援排程判斷「今天的資料（含融資融券）是否已更新」
    if os.getenv("RUN_STARTED"):
        st["run_started"] = os.getenv("RUN_STARTED")
    if (ART / "results.pkl").is_file():
        try:
            with open(ART / "results.pkl", "rb") as f:
                res = pickle.load(f)["res"]
            st["research"] = {"last_date": res["protocol"]["last_date"], "built": res["meta"].get("built"),
                              "synthetic": bool(res["meta"].get("synthetic"))}
        except Exception:  # noqa: BLE001
            pass
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    if "research" not in st and "research" in old:
        st["research"] = old["research"]
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st


def cmd_progress(a, cfg):
    from kosto.build import progress
    st = write_status(cfg, a.write, {"stage": a.stage} if a.stage else None) if a.write else progress(cfg)
    log(f"資料完成度 {st['percent']:.1%}：" + "、".join(f"{k} {v:.0%}" for k, v in st["groups"].items()))
    for k, v in st.get("blocked", {}).items():
        log(f"  ⚠ {k}: {v}")
    if a.github_output and os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"complete={'true' if st['complete'] else 'false'}\n")
    print("COMPLETE" if st["complete"] else "INCOMPLETE")


def cmd_auto(a, cfg):
    """Local one-click: build/continue the database, then research + report when complete."""
    from kosto.build import auto_build
    token = os.getenv("FINMIND_TOKEN", "")
    status = Path("status/status.json")
    write_status(cfg, status, {"stage": "downloading"})
    p = auto_build(cfg, token, a.budget, log=log)
    write_status(cfg, status, {"stage": "downloading" if not p["complete"] else "researching"})
    if not p["complete"] and not a.force_research:
        log(f"資料尚未完整（{p['percent']:.0%}）。再執行一次即可續傳。")
        return
    a.compare_panel = True
    cmd_bundle(a, cfg)
    a.bundle = str(ART / "bundle.npz")
    a.compare = [str(x) for x in Path("compare").glob("*") if x.suffix.lower() in (".csv", ".zip")] or None
    a.horizons = None
    cmd_research(a, cfg)
    write_status(cfg, status, {"stage": "done"})


def cmd_us_build(a, cfg):
    """v3.4 美股資料包（與台股流程完全分開）。"""
    from kosto.us import engine as UE
    prev = None
    if a.prev and Path(a.prev).is_file():
        try:
            prev = UE.loads(Path(a.prev).read_bytes())
            log(f"讀到前一份美股資料包（{prev.get('asof')}），名單續抱用")
        except Exception as e:  # noqa: BLE001
            log(f"前一份美股資料包無法讀取（{e}），不續抱")
    B = UE.build(prev=prev, log=log)
    ART.mkdir(exist_ok=True)
    out = ART / "us_bundle.pkl.gz"
    out.write_bytes(UE.dumps(B))
    log(f"美股資料包 {B['asof']}：{len(B['tickers'])} 檔、母池 {int(B['universe'].sum())} 檔、"
        f"失敗 {len(B['fail'])} 檔 → {out}（{out.stat().st_size / 1e6:.1f} MB）")
    for h, p in B["picks"].items():
        log(f"  {h}（{p['model']}）：{'、'.join(p['names'])}")
    if len(B["tickers"]) < 300:
        raise SystemExit("美股日線成功檔數太少（< 300），不發布，保留前一份")


def cmd_status(a, cfg):
    from kosto.store import Store
    print(Store(cfg["data"]["dir"]).summary().to_string())


def cmd_morning(a, cfg):
    """v3.16 每日台股盤前日報：讀前一晚的研究結果（含 morning_model）＋盤前即時行情與新聞 → artifacts/morning.json。"""
    import json

    import pandas as pd

    from kosto import morning as M
    now = pd.Timestamp.now(tz="Asia/Taipei")
    day = str(M.report_date(now).date())
    if a.prev and Path(a.prev).is_file() and not a.force:
        try:
            old = json.loads(Path(a.prev).read_text(encoding="utf-8"))
            if old.get("date") == day and old.get("status") == "complete":
                log(f"{day} 的盤前日報已完成（{old.get('generated_at', '')[:16]}），略過；要重做請加 --force")
                return
        except Exception:  # noqa: BLE001
            pass
    res = {}
    rp = Path(a.results)
    if rp.is_file():
        try:
            with open(rp, "rb") as f:
                res = pickle.load(f).get("res") or {}
        except Exception as e:  # noqa: BLE001
            log(f"讀取研究結果失敗（日報只整理行情與新聞）：{e}")
    model = res.get("morning_model") or {}
    if not model or model.get("error"):
        log("研究結果沒有盤前日報模型（v3.16 以前，或研究失敗）→ 只整理行情與新聞，不做開盤預估")
    live = M.fetch_live(now)
    rep = M.build(model, live, res, now)
    if not rep["us"]["rows"]:
        log("美股報價全部取不到 → 不發布（App 繼續顯示前一份）")
        return 2
    hist = []
    if a.hist and Path(a.hist).is_file():
        try:
            hist = json.loads(Path(a.hist).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            hist = []
    hist = M.append_hist(hist, rep)
    ART.mkdir(exist_ok=True)
    (ART / "morning.json").write_text(M.dumps(rep), encoding="utf-8")
    (ART / "morning_hist.json").write_text(M.dumps(hist), encoding="utf-8")
    for x in rep["headline"]:
        log("日報：" + x)
    miss = [k for k, v in rep["sections_ok"].items() if not v]
    log(f"{day} 盤前日報完成（{rep['status']}" + (f"；缺：{'、'.join(miss)}" if miss else "") + "）")


def main(argv=None):
    p = argparse.ArgumentParser(description="Kosto Lab 科斯托蘭尼量化選股實驗室")
    p.add_argument("--config", default=None, help="自訂 YAML（會覆蓋 config/default.yaml 的同名欄位）")
    p.add_argument("--data-dir", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update"); u.add_argument("--group", required=True); u.add_argument("--budget", type=float, default=330)
    b = sub.add_parser("bundle"); b.add_argument("--compare-panel", action="store_true")
    r = sub.add_parser("research")
    r.add_argument("--bundle", default=str(ART / "bundle.npz")); r.add_argument("--quick", action="store_true")
    r.add_argument("--horizons", default=None); r.add_argument("--compare", nargs="*", default=None)
    d = sub.add_parser("demo")
    d.add_argument("--stocks", type=int, default=500); d.add_argument("--seed", type=int, default=7)
    d.add_argument("--quick", action="store_true"); d.add_argument("--fresh", action="store_true")
    d.add_argument("--horizons", default=None)
    sub.add_parser("status")
    sub.add_parser("summary")
    pg = sub.add_parser("progress"); pg.add_argument("--write", default=None); pg.add_argument("--stage", default=None)
    pg.add_argument("--github-output", action="store_true")
    au = sub.add_parser("auto"); au.add_argument("--budget", type=float, default=600)
    au.add_argument("--quick", action="store_true"); au.add_argument("--force-research", action="store_true")
    pk = sub.add_parser("pack"); pk.add_argument("--group", required=True); pk.add_argument("--out", required=True)
    up = sub.add_parser("unpack"); up.add_argument("archive", nargs="+")
    ub = sub.add_parser("us-build"); ub.add_argument("--prev", default=None)
    mo = sub.add_parser("morning"); mo.add_argument("--results", default=str(ART / "results.pkl"))
    mo.add_argument("--hist", default=None); mo.add_argument("--prev", default=None); mo.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    over = {"data": {"dir": a.data_dir}} if a.data_dir else None
    if a.cmd == "demo" and not a.data_dir:
        over = {"data": {"dir": "data_demo"}}
    cfg = load_config(a.config, over)
    if a.cmd == "pack":
        from kosto.remote import pack
        print(pack(cfg["data"]["dir"], a.group, a.out))
        return
    if a.cmd == "unpack":
        from kosto.remote import unpack
        for x in a.archive:
            if Path(x).is_file():
                unpack(cfg["data"]["dir"], x)
                log(f"解開 {x}")
        return
    {"update": cmd_update, "bundle": cmd_bundle, "research": cmd_research, "demo": cmd_demo,
     "status": cmd_status, "summary": cmd_summary, "progress": cmd_progress, "auto": cmd_auto,
     "us-build": cmd_us_build, "morning": cmd_morning}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
