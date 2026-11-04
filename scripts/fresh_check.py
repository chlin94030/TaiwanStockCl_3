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


def expected_day(now: dt.datetime, after: int = 21 * 60) -> dt.date:
    d = now.date() if now.hour * 60 + now.minute >= after else now.date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def twse_holidays(year: int) -> set:
    """證交所休市日（抓不到就當作沒有假日；只影響「要不要多跑一次」）。"""
    try:
        import urllib.request
        url = f"https://www.twse.com.tw/rwd/zh/holidaySchedule/holidaySchedule?date={year}0101&response=json"
        with urllib.request.urlopen(url, timeout=15) as r:
            js = json.loads(r.read().decode("utf-8"))
        out = set()
        for row in js.get("data") or []:
            if "交易日" in str(row[1] if len(row) > 1 else ""):     # 「開始交易日」「最後交易日」是有交易的日子
                continue
            d = str(row[0]).replace("/", "-")
            if d[:3].isdigit() and len(d.split("-")[0]) == 3:            # 民國年
                y, m, dd = d.split("-")[:3]
                d = f"{int(y) + 1911}-{int(m):02d}-{int(dd):02d}"
            out.add(d[:10])
        return out
    except Exception:  # noqa: BLE001
        return set()


def _back_off_holiday(d: dt.date, hol: set) -> dt.date:
    while d.weekday() >= 5 or d.isoformat() in hol:
        d -= dt.timedelta(days=1)
    return d


def decide(status: dict, now: dt.datetime, hol: set | None = None) -> tuple[bool, str]:
    """收盤價／法人：16:30 後應有今天；融資券：21:00 後應有今天。兩者都已更新 → 這次排程不用跑。"""
    hol = hol or set()
    exp_px = _back_off_holiday(expected_day(now, 16 * 60 + 30), hol).isoformat()
    exp_mg = _back_off_holiday(expected_day(now, 21 * 60), hol).isoformat()
    last = str((status.get("research") or {}).get("last_date") or "")[:10]
    lat = status.get("latest") or {}
    margin = str(lat.get("margin") or "")[:10]
    inst = str(lat.get("institutional") or "")[:10]
    if not last:
        return False, "狀態檔沒有研究日期 → 執行"
    if last < exp_px:
        return False, f"研究資料到 {last}，應到 {exp_px} → 執行"
    if inst and inst < last:
        return False, f"研究資料到 {last}，但三大法人只到 {inst} → 執行"
    if not margin or margin < exp_mg:
        return False, f"研究資料到 {last}，融資融券只到 {margin or '–'}（應到 {exp_mg}）→ 執行"
    return True, f"研究資料已到 {last}（融資融券到 {margin}）→ 這次排程不需要執行"


def main(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
        now = dt.datetime.now(TPE)
        skip, why = decide(st, now, twse_holidays(now.year))
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
