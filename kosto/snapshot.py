"""v3.14 每日研究快照（回應同儕評審 P0「不可覆寫的每日快照」）。

每次每日研究完成時產生一筆快照：快照編號（資料日＋設定雜湊）、模型版本、設定雜湊、資料日期、資料包建立時間、
各資料來源日期、三個期間的名單與雜湊。快照依序串成雜湊鏈（每筆記下前一筆的雜湊），同一天重跑（初版→完整版）
會新增一筆、不覆寫舊的；之後任何人改動舊紀錄，鏈就驗證不過。App 用它檢查「資料包」與「研究結果」是不是同一版。
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

KEEP = 600            # 雜湊鏈保留筆數（約 1 年多，每天 1～2 筆）


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def config_hash(cfg: dict) -> str:
    """選股相關設定的雜湊（不含資料路徑、排程等與結果無關的欄位）。"""
    keep = {k: cfg.get(k) for k in ("research", "horizons", "pools", "kostolany", "backtest") if k in cfg}
    return _h(keep)[:12]


def picks_of(res: dict) -> dict:
    out = {}
    for h, hb in (res.get("horizons") or {}).items():
        lt = hb.get("latest") or {}
        out[h] = {"date": lt.get("date"), "pool": lt.get("pool"),
                  "tickers": [p["ticker"] for p in lt.get("picks") or []],
                  "scores": [round(float(p.get("score") or 0), 4) for p in lt.get("picks") or []]}
    return out


def make(res: dict, bundle_meta: dict, cfg: dict, version: str, now: pd.Timestamp | None = None) -> dict:
    """本次研究的快照（不含鏈欄位）。"""
    now = now or pd.Timestamp.now(tz="Asia/Taipei")
    asof = (res.get("protocol") or {}).get("last_date") or (bundle_meta.get("sig_dates") or [""])[-1]
    ch = config_hash(cfg)
    picks = picks_of(res)
    snap = {"id": f"{str(asof).replace('-', '')}-{ch[:8]}", "model_version": version, "config_hash": ch,
            "data_asof": str(asof), "bundle_built": bundle_meta.get("built"), "bundle_version": bundle_meta.get("version"),
            "published_at": now.isoformat(timespec="seconds"), "sources": dict(bundle_meta.get("data_dates") or {}),
            "picks": picks, "picks_hash": _h(picks)[:16]}
    snap["hash"] = _h(snap)
    return snap


def append(chain: list | None, snap: dict, keep: int = KEEP) -> list:
    """把快照接到雜湊鏈後面（只新增、不覆寫）。每筆記下前一筆的鏈雜湊；同一天重跑另起一筆（seq 加一）。"""
    chain = list(chain or [])
    prev = chain[-1]["link"] if chain else "0" * 64
    seq = 1 + sum(1 for x in chain if x.get("data_asof") == snap["data_asof"])
    row = {k: snap[k] for k in ("id", "model_version", "config_hash", "data_asof", "bundle_built", "published_at",
                                "picks_hash", "hash")}
    row.update({"seq": seq, "prev": prev, "tickers": {h: p["tickers"] for h, p in snap["picks"].items()}})
    row["link"] = hashlib.sha256((prev + snap["hash"]).encode()).hexdigest()
    chain.append(row)
    if len(chain) > keep:              # 截掉最舊的：保留截斷點的 prev，驗證從那裡接續
        chain = chain[-keep:]
    return chain


def verify(chain: list | None) -> tuple[bool, int | None]:
    """(是否完整, 第一筆斷掉的位置)。只驗證鏈的連續性（每筆的 prev＝前一筆的 link，link＝sha256(prev＋hash)）。"""
    chain = chain or []
    for k, row in enumerate(chain):
        if k and row.get("prev") != chain[k - 1].get("link"):
            return False, k
        if hashlib.sha256((row.get("prev", "") + row.get("hash", "")).encode()).hexdigest() != row.get("link"):
            return False, k
    return True, None


def consistency(bundle_meta: dict, res: dict) -> list[str]:
    """App 載入時檢查資料包與研究結果是否同一版；回傳問題清單（空＝一致）。"""
    snap = res.get("snapshot") or {}
    issues = []
    if not snap:
        return ["研究結果沒有快照紀錄（v3.14 以前的版本）"]
    b_last = (bundle_meta.get("sig_dates") or [None])[-1]
    if b_last and snap.get("data_asof") and str(b_last) != str(snap["data_asof"]):
        issues.append(f"資料包到 {b_last}，研究結果到 {snap['data_asof']}")
    if bundle_meta.get("built") and snap.get("bundle_built") and bundle_meta["built"] != snap["bundle_built"]:
        issues.append("資料包已重建，研究結果還是用上一版資料包算的（下一次研究完成後會一致）")
    if snap.get("hash"):
        body = {k: v for k, v in snap.items() if k != "hash"}
        if _h(body) != snap["hash"]:
            issues.append("快照內容與雜湊不符（研究結果檔可能被改動）")
    if snap.get("picks") and _h(picks_of(res))[:16] != snap.get("picks_hash"):
        issues.append("名單與快照不符")
    return issues


def update_state(run: dict | None, last_ok: str | None, expected: str | None, partial: bool = False) -> tuple[str, str]:
    """v3.14 更新狀態（同儕評審 4.2）：已完成／執行中（保留上一版）／部分完成／延遲／失敗。
    執行中一律不給紅燈：上一版名單照常可用。回傳 (代碼, 中文)。"""
    running = bool(run) and run.get("status") in ("queued", "in_progress", "waiting", "requested", "pending")
    if running:
        return "running", "執行中（保留上一版）"
    if run and run.get("conclusion") == "failure":
        return "failed", "失敗"
    if expected and last_ok and str(last_ok) < str(expected):
        return "late", "延遲"
    if partial:
        return "partial", "部分完成"
    return "done", "已完成"
