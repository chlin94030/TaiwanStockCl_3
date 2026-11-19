"""資料新鮮度燈號（v2.1）：每日 17:23 初版（約 18:30 完成，融資券待補）、21:40 完整版（約 23:00 完成）；
GitHub 排程若延遲或被略過，另有 23:17、00:47、06:23 備援排程。"""
from __future__ import annotations

import pandas as pd

TZ = "Asia/Taipei"
RUNNING = ("queued", "in_progress", "waiting", "requested", "pending")
EVENT = {"schedule": "排程", "push": "上傳程式", "workflow_dispatch": "手動／自動續跑"}


def expected_data_day(now: pd.Timestamp, cutoff: int = 19 * 60) -> pd.Timestamp:
    """現在這個時間點，每日研究「應該」已經涵蓋到哪一個交易日：19:00 後＝今天（17:23 初版約 18:30 完成），之前＝前一個平日。"""
    d = pd.Timestamp(now.date())
    if now.hour * 60 + now.minute < cutoff:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _prev_weekday(d: pd.Timestamp) -> pd.Timestamp:
    d = d - pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _tpe(x) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(x)
        t = t.tz_localize("UTC") if t.tzinfo is None else t
        return t.tz_convert(TZ)
    except Exception:  # noqa: BLE001
        return None


def run_desc(run: dict | None) -> str:
    """「最近一次執行 11/02 21:52（排程）成功」之類的說明（放在燈號的滑鼠提示）。"""
    if not run:
        return ""
    t = _tpe(run.get("created"))
    res = {"success": "成功", "failure": "失敗", "cancelled": "已取消", "skipped": "略過"}.get(run.get("conclusion") or "",
                                                                                    "執行中" if run.get("status") in RUNNING else "")
    return f"最近一次執行 {t:%m/%d %H:%M}（{EVENT.get(run.get('event'), run.get('event') or '')}）{res}" if t is not None else ""


def run_rows(history: list | None) -> list[dict]:
    """v2.5 最近幾次自動更新：開始時間、觸發方式、結果、耗時、連結（讓使用者自己判斷是失敗、取消還是被延遲）。"""
    out = []
    for r in history or []:
        t0, t1 = _tpe(r.get("created")), _tpe(r.get("updated"))
        running = r.get("status") in RUNNING
        res = "執行中" if running else {"success": "成功", "failure": "失敗", "cancelled": "已取消", "skipped": "略過"}.get(
            r.get("conclusion") or "", r.get("conclusion") or "")
        mins = int((t1 - t0).total_seconds() // 60) if (t0 is not None and t1 is not None and not running) else None
        out.append({"start": f"{t0:%m/%d %H:%M}" if t0 is not None else "", "event": EVENT.get(r.get("event"), r.get("event") or ""),
                    "result": res, "minutes": mins, "url": r.get("url", "")})
    return out


def fail_streak(history: list | None) -> int:
    """最近連續失敗的次數（略過執行中的那一次）。"""
    n = 0
    for r in history or []:
        if r.get("status") in RUNNING:
            continue
        if r.get("conclusion") == "failure":
            n += 1
        elif r.get("conclusion") in ("success",):
            break
    return n


def status(now: pd.Timestamp, last: pd.Timestamp, run: dict | None, history: list | None = None) -> tuple[str, str, str]:
    """(燈號, 文字, 連結)。綠＝已完成（已是最新）；黃＝執行中（保留上一版）／延遲／可能休市；紅＝失敗或今晚的自動更新沒有啟動。
    run＝GitHub Actions 最近一次執行 {status, conclusion, created, event, url}（本機模式為 None）；
    history＝最近幾次執行（新到舊，v2.5：用來說明「資料落後幾天、前面失敗幾次」）。"""
    last = pd.Timestamp(last)
    url = (run or {}).get("url", "")
    exp = expected_data_day(now)
    behind = len(pd.bdate_range(last + pd.Timedelta(days=1), exp)) if last < exp else 0
    lag = f"｜研究資料只到 {last:%m/%d}（應到 {exp:%m/%d}）" if behind else ""
    nf = fail_streak(history)
    fails = f"｜之前連續 {nf} 次更新失敗" if nf else ""
    if run and run.get("status") in RUNNING:
        t = _tpe(run.get("created"))
        why = EVENT.get(run.get("event"), "")
        start = f"{why}觸發，{t:%H:%M} 開始，" if t is not None else ""
        # v3.14（同儕評審 4.2）：執行中一律黃燈＝「執行中（保留上一版）」；上一版名單照常可用，不再因先前失敗顯示紅燈
        return "amber", f"每日更新執行中，上一版名單照常可用（{start}約 30～60 分鐘）{lag}{fails}", url
    if run and run.get("conclusion") == "failure":
        return "red", f"最近一次更新失敗（點此看原因）{lag}{fails if nf > 1 else ''}", url
    hm = now.hour * 60 + now.minute
    if last >= exp:
        if now.weekday() < 5 and 13 * 60 + 30 <= hm < 19 * 60 and last < pd.Timestamp(now.date()):
            return "green", "已是最新｜今日初版 17:20 起更新", url
        return "green", "已是最新", url
    t = _tpe((run or {}).get("created"))
    due = (exp + pd.Timedelta(hours=17)).tz_localize(TZ)                  # 17:23 初版排程之後
    late_today = (pd.Timestamp(now.date()) == exp and 19 * 60 <= hm < 23 * 60 + 30 and last >= _prev_weekday(exp))
    if late_today and not (run and t is not None and t >= due and run.get("conclusion") == "success"):
        cancelled = bool(run and t is not None and t >= due and run.get("conclusion") == "cancelled")
        if hm >= 22 * 60 + 50:                     # v3.11：完整版約 23:00 完成；過了還沒有就明講並給「立即更新」
            return "red", ("今天 17:23、21:40 的自動更新都還沒完成（" + ("被取消" if cancelled else "GitHub 排程延遲或被略過")
                           + "）；23:17、00:47、06:23 的備援排程會再試，也可以按「立即更新」"), url
        if hm >= 21 * 60 + 40:
            return "amber", "今日初版沒有完成；21:40 完整版排隊或執行中（約 23:00 完成）", url
        return "amber", "今日初版尚未完成（GitHub 排程延遲），21:40 完整版會補上", url
    if run and t is not None and t >= due:
        if run.get("conclusion") == "success":            # 之後已成功跑過但沒有新資料 → 多半是休市（或資料源延遲）
            return "amber", f"已更新，沒有 {exp:%m/%d} 的資料（休市？）", url
        return "amber", f"{exp:%m/%d} 的更新被取消，備援排程會再試", url
    if run is not None:
        return "red", f"{exp:%m/%d} 的自動更新沒有啟動（GitHub 排程延遲），可按「立即更新」", url
    if len(pd.bdate_range(last + pd.Timedelta(days=1), exp)) <= 1:
        return "amber", f"資料到 {last:%m/%d}（中間若為假日屬正常）", url
    return "red", f"資料落後：應更新到 {exp:%m/%d}", url


STATE_ZH = {"green": "已完成", "amber": "執行中或延遲", "red": "失敗"}


def state_label(level: str, text: str) -> str:
    """v3.14 五種更新狀態（同儕評審 4.2）：已完成／執行中（保留上一版）／部分完成／延遲／失敗。"""
    if "執行中" in text:
        return "執行中（保留上一版）"
    if level == "green":
        return "部分完成" if "融資" in text and ("補上" in text or "待補" in text) else "已完成"
    if level == "red":
        return "失敗" if "失敗" in text else "延遲"
    return "延遲" if any(k in text for k in ("尚未", "落後", "沒有完成", "取消", "排隊")) else "已完成（可能休市）"
