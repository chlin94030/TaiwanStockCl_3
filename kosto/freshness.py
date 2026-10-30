"""v1.9 資料新鮮度燈號：每日研究 21:40 開始、約 23:00 完成（判斷保留緩衝到 23:30）。"""
from __future__ import annotations

import pandas as pd

TZ = "Asia/Taipei"


def expected_data_day(now: pd.Timestamp) -> pd.Timestamp:
    """現在這個時間點，每日研究「應該」已經涵蓋到哪一個交易日（不含國定假日判斷）。"""
    d = pd.Timestamp(now.date())
    if now.hour * 60 + now.minute < 23 * 60 + 30:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def status(now: pd.Timestamp, last: pd.Timestamp, run: dict | None) -> tuple[str, str, str]:
    """(燈號, 文字, 連結)。綠＝已是最新；黃＝更新中／可能是假日；紅＝更新失敗或資料落後。
    run＝GitHub Actions 最近一次執行 {status, conclusion, created, url}（本機模式為 None）。"""
    last = pd.Timestamp(last)
    url = (run or {}).get("url", "")
    if run and run.get("status") in ("in_progress", "queued"):
        return "amber", "每日更新執行中（約 23:00 完成）", url
    if run and run.get("conclusion") == "failure":
        return "red", "最近一次更新失敗（點此看原因）", url
    exp = expected_data_day(now)
    ran_after = False
    if run and run.get("conclusion") == "success" and run.get("created"):
        try:
            t = pd.Timestamp(run["created"])
            t = t.tz_localize("UTC") if t.tzinfo is None else t
            ran_after = t.tz_convert(TZ) >= (exp + pd.Timedelta(hours=21, minutes=30)).tz_localize(TZ)
        except Exception:  # noqa: BLE001
            ran_after = False
    if last >= exp or ran_after:                       # ran_after：之後已成功跑過（中間是假日、沒有新資料）也算最新
        hm = now.hour * 60 + now.minute
        if now.weekday() < 5 and 13 * 60 + 30 <= hm < 23 * 60 + 30 and last < pd.Timestamp(now.date()):
            return "green", "已是最新｜今日資料 21:40 起更新", url
        return "green", "已是最新", url
    if run is None and len(pd.bdate_range(last + pd.Timedelta(days=1), exp)) <= 1:
        return "amber", f"資料到 {last:%m/%d}（中間若為假日屬正常）", url
    return "red", f"資料落後：應更新到 {exp:%m/%d}" + ("（點此看執行紀錄）" if url else ""), url
