"""備援排程用：今天（台北）該有的資料是否已經更新完成？只用 Python 標準函式庫（不需安裝套件）。

應涵蓋的交易日：台北 21:00 以後＝今天，之前＝前一個平日（週一～五；國定假日不另判斷）。
status/status.json（每次自動流程結束時寫入）中：
  research.last_date ≥ 應涵蓋日，且 latest.margin ≥ research.last_date（當天融資融券也有）→ skip=true。
任何讀不到、格式不對的情況一律 skip=false（寧可多跑一次）。
用法：python3 scripts/fresh_check.py status/status.json   → 寫 skip=true/false 到 $GITHUB_OUTPUT 並印出原因。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

TPE = dt.timezone(dt.timedelta(hours=8))


def expected_day(now: dt.datetime) -> dt.date:
    d = now.date() if now.hour * 60 + now.minute >= 21 * 60 else now.date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def decide(status: dict, now: dt.datetime) -> tuple[bool, str]:
    exp = expected_day(now).isoformat()
    last = str((status.get("research") or {}).get("last_date") or "")[:10]
    margin = str((status.get("latest") or {}).get("margin") or "")[:10]
    if not last:
        return False, "狀態檔沒有研究日期 → 執行"
    if last < exp:
        return False, f"研究資料到 {last}，應到 {exp} → 執行"
    if not margin or margin < last:
        return False, f"研究資料到 {last}，但融資融券只到 {margin or '–'} → 執行"
    return True, f"研究資料已到 {last}（含融資融券）→ 這次備援排程不需要執行"


def main(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        skip, why = decide(st, dt.datetime.now(TPE))
    except Exception as e:  # noqa: BLE001
        skip, why = False, f"讀不到狀態檔（{e}）→ 執行"
    print(why)
    out = os.getenv("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"skip={'true' if skip else 'false'}\n")
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(why + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "status/status.json")
