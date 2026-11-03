"""資料新鮮度燈號（v1.9.1）：每日研究 21:40 開始、約 23:00 完成（判斷保留緩衝到 23:30）；
GitHub 排程若延遲或被略過，另有 23:17、00:47、06:23 備援排程。"""
from __future__ import annotations

import pandas as pd

TZ = "Asia/Taipei"
RUNNING = ("queued", "in_progress", "waiting", "requested", "pending")
EVENT = {"schedule": "排程", "push": "上傳程式", "workflow_dispatch": "手動"}


def expected_data_day(now: pd.Timestamp) -> pd.Timestamp:
    """現在這個時間點，每日研究「應該」已經涵蓋到哪一個交易日（不含國定假日判斷）。"""
    d = pd.Timestamp(now.date())
    if now.hour * 60 + now.minute < 23 * 60 + 30:
        d -= pd.Timedelta(days=1)
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


def status(now: pd.Timestamp, last: pd.Timestamp, run: dict | None) -> tuple[str, str, str]:
    """(燈號, 文字, 連結)。綠＝已是最新；黃＝更新中／可能休市；紅＝更新失敗或今晚的自動更新沒有啟動。
    run＝GitHub Actions 最近一次執行 {status, conclusion, created, event, url}（本機模式為 None）。"""
    last = pd.Timestamp(last)
    url = (run or {}).get("url", "")
    if run and run.get("status") in RUNNING:
        return "amber", "每日更新執行中（約 30～60 分鐘）", url
    if run and run.get("conclusion") == "failure":
        return "red", "最近一次更新失敗（點此看原因）", url
    exp = expected_data_day(now)
    hm = now.hour * 60 + now.minute
    if last >= exp:
        if now.weekday() < 5 and 13 * 60 + 30 <= hm < 23 * 60 + 30 and last < pd.Timestamp(now.date()):
            return "green", "已是最新｜今日資料 21:40 起更新", url
        return "green", "已是最新", url
    t = _tpe((run or {}).get("created"))
    due = (exp + pd.Timedelta(hours=21, minutes=30)).tz_localize(TZ)
    if run and t is not None and t >= due:
        if run.get("conclusion") == "success":            # 之後已成功跑過但沒有新資料 → 多半是休市（或資料源延遲）
            return "amber", f"已更新，沒有 {exp:%m/%d} 的資料（休市？）", url
        return "amber", f"{exp:%m/%d} 的更新被取消，備援排程會再試", url
    if run is not None:
        return "red", f"{exp:%m/%d} 的自動更新沒有啟動（GitHub 排程延遲），可按「立即更新」", url
    if len(pd.bdate_range(last + pd.Timedelta(days=1), exp)) <= 1:
        return "amber", f"資料到 {last:%m/%d}（中間若為假日屬正常）", url
    return "red", f"資料落後：應更新到 {exp:%m/%d}", url
