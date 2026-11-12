# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Turn recorder statistics rows (hourly) into daily values in local time."""
import datetime as dt
from collections import defaultdict
from zoneinfo import ZoneInfo


def _local(ts: float, tz: ZoneInfo) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz)


def daily_from_sum(rows: list[dict], tz: ZoneInfo, cap: float = 50.0) -> dict[dt.date, float]:
    """Energy per local day from cumulative `sum` rows; rises above `cap` per hour (meter resets) are dropped."""
    out: dict[dt.date, float] = {}
    for a, b in zip(rows, rows[1:]):
        if a.get('sum') is None or b.get('sum') is None:
            continue
        kwh = b['sum'] - a['sum']
        if 0 <= kwh <= cap:
            k = _local(b['start'], tz).date()
            out[k] = out.get(k, 0.0) + kwh
    return out


def hourly_share(rows: list[dict], tz: ZoneInfo, today: dt.date, days: int = 14) -> list[float] | None:
    """Share of a day's energy booked in each local hour over the last `days` full days."""
    prof = [0.0] * 24
    since = today - dt.timedelta(days=days)
    for a, b in zip(rows, rows[1:]):
        if a.get('sum') is None or b.get('sum') is None:
            continue
        kwh = b['sum'] - a['sum']
        t = _local(b['start'], tz)
        if 0 <= kwh <= 50 and since <= t.date() < today:
            prof[t.hour] += kwh
    total = sum(prof)
    return [p / total for p in prof] if total > 0 else None


def daily_mean(rows: list[dict], tz: ZoneInfo, min_hours: int = 18) -> dict[dt.date, float]:
    """Mean of hourly `mean` rows per local day, for days with enough hours."""
    acc: dict[dt.date, list[float]] = defaultdict(list)
    for r in rows:
        if r.get('mean') is not None:
            acc[_local(r['start'], tz).date()].append(r['mean'])
    return {k: sum(v) / len(v) for k, v in acc.items() if len(v) >= min_hours}


def daily_running_max(rows: list[dict], tz: ZoneInfo) -> dict[dt.date, float]:
    """Day total of a sensor that counts up and resets at midnight (e.g. heating time today).

    The 00:00 hour still holds yesterday's total until the reset, so it is skipped.
    """
    out: dict[dt.date, float] = {}
    for r in rows:
        if r.get('max') is None:
            continue
        t = _local(r['start'], tz)
        if t.hour:
            out[t.date()] = max(out.get(t.date(), 0.0), r['max'])
    return out


def hour_means(hours: dict[float, float], tz: ZoneInfo, missing: int = 1) -> dict[dt.date, float]:
    """Mean per local day of hourly values (UTC hour start -> value), for days with at most `missing` hours absent."""
    acc: dict[dt.date, list[float]] = defaultdict(list)
    for t, v in hours.items():
        acc[_local(t, tz).date()].append(v)
    out = {}
    for d, v in acc.items():
        start = dt.datetime.combine(d, dt.time(), tz).timestamp()
        end = dt.datetime.combine(d + dt.timedelta(days=1), dt.time(), tz).timestamp()
        if len(v) >= round((end - start) / 3600) - missing:
            out[d] = sum(v) / len(v)
    return out


def local_offset(local: dict[dt.date, float], main: dict[dt.date, float], today: dt.date,
                 days: int = 30, min_days: int = 7) -> float | None:
    """Mean of local minus main daily temperature over the last `days` days before today; None with too few."""
    diffs = [local[k] - main[k] for k in local if k in main and today - dt.timedelta(days=days) <= k < today]
    return sum(diffs) / len(diffs) if len(diffs) >= min_days else None


def expected_total(forecast: float, used: float, share: list[float] | None, now: dt.datetime) -> float:
    """Gas used so far today plus the forecast's usual share for the rest of the day (the current hour pro rata)."""
    if not share:
        return max(forecast, used)
    done = sum(share[:now.hour]) + share[now.hour] * (now.minute + now.second / 60) / 60
    return used + forecast * max(0.0, 1.0 - done)
