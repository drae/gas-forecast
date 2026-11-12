# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Daily gas forecast: candidate models, rolling-origin scoring and model choice."""
import datetime as dt
import math
from dataclasses import dataclass

from .model import Day, Model

# The slope is the house's heat loss, which carries over between winters (winter 25/26: 4.8 kWh per degree-day,
# or 4.2 + 0.07 per degree-day² with the curve); the level (a, base) is learned afresh each season.
PRIOR = {'a': (1.5, 3.0), 'b': (4.8, 1.0)}
PRIOR_HEATING = {'a': (0.0, 1.0), 'b': (4.8, 1.0)}
PRIOR_CURVE = {'a': (1.5, 3.0), 'b': (4.2, 1.0), 'q': (0.07, 0.04)}
MEMORY_DAYS = 1.0  # time constant of the outdoor temperature's fading memory
LOOKBACK = 6       # days in the weighted temperature
BASELINE_DAYS = 14   # non-heating gas (hot water, hob) averaged over this many split days
FIRST_HEATING_KWH = 5.0  # the card rule: fit from the first day of the window at or above this
COLD_OFF_T2 = 12.0       # a cold day with no heating at all is an away day or missing data


def make(name: str) -> Model:
    if name == 'ols15':
        return Model(window=30)
    if name == 'change_point':
        return Model(fit_base=True, window=30)
    if name == 'bayes':
        return Model(fit_base=True, window=None, half_life=21, prior=PRIOR)
    if name == 'bayes_heating':
        return Model(fit_base=True, window=None, half_life=21, prior=PRIOR_HEATING)
    if name == 'bayes_curve':
        return Model(fit_base=True, window=None, half_life=21, prior=PRIOR_CURVE, curve=True)
    raise ValueError(name)


CANDIDATES = ('ols15', 'change_point', 'bayes', 'bayes_heating', 'bayes_curve')


@dataclass
class Rec:
    gas: float | None       # kWh
    t: float | None         # daily mean outdoor temperature, °C
    excluded: bool = False  # away or heating off: not fitted, not scored


def season_start(d: dt.date) -> dt.date:
    """Heating seasons start on 1 September."""
    return dt.date(d.year if d.month >= 9 else d.year - 1, 9, 1)


def build(gas: dict, temp: dict, heat_hours: dict | None, away: dict | None,
          away_minutes: int = 360) -> dict[dt.date, Rec]:
    recs = {}
    for k in sorted(set(gas) | set(temp)):
        r = Rec(gas.get(k), temp.get(k))
        prev = temp.get(k - dt.timedelta(days=1))
        if away and away.get(k, 0) >= away_minutes:
            r.excluded = True
        elif heat_hours is not None and k in heat_hours and heat_hours[k] == 0 and r.t is not None \
                and prev is not None and (r.t + prev) / 2 < COLD_OFF_T2:
            r.excluded = True
        recs[k] = r
    return recs


def weighted(temps: list[float | None]) -> float | None:
    """Outdoor temperature with a fading memory, from daily means newest first (today, yesterday, ...).

    Weights fall by e per MEMORY_DAYS. Today and yesterday are needed; older missing days are skipped.
    """
    if len(temps) < 2 or temps[0] is None or temps[1] is None:
        return None
    pairs = [(math.exp(-i / MEMORY_DAYS), t) for i, t in enumerate(temps[:LOOKBACK]) if t is not None]
    return sum(w * t for w, t in pairs) / sum(w for w, _ in pairs)


def temp_of(recs: dict, k: dt.date, temps: dict | None = None) -> float | None:
    """Weighted outdoor temperature for day k; `temps` overrides recorded ones (forecasts)."""
    def t(d):
        if temps and d in temps:
            return temps[d]
        r = recs.get(d)
        return r.t if r else None
    return weighted([t(k - dt.timedelta(days=i)) for i in range(LOOKBACK)])


def training(recs: dict, name: str, day: dt.date) -> list[Day]:
    """Season days before `day`, aged relative to it."""
    out = []
    for k in sorted(recs):
        if not season_start(day) <= k < day:
            continue
        r, temp = recs[k], temp_of(recs, k)
        if temp is None:
            continue
        out.append(Day(gas=None if r.excluded else r.gas, temp=temp, age=(day - k).days))
    if name == 'ols15':
        out = [d for d in out if d.age <= 30]
        hot = [d.age for d in out if d.gas is not None and d.gas >= FIRST_HEATING_KWH]
        if hot:
            out = [d for d in out if d.age <= max(hot)]
    return out


def reliable(split_day: dict | None) -> bool:
    return bool(split_day) and split_day.get('reliable', False)


def heating_recs(recs: dict, splits: dict) -> dict[dt.date, Rec]:
    """Heating-only gas on days with a reliable split; other days are left out."""
    return {k: Rec(splits[k]['heating'] if reliable(splits.get(k)) else None, r.t, r.excluded)
            for k, r in recs.items()}


def baseline(splits: dict, day: dt.date) -> float | None:
    days = [splits[k]['gas'] - splits[k]['heating'] for k in sorted(splits, reverse=True)
            if k < day and reliable(splits[k])][:BASELINE_DAYS]
    return sum(days) / len(days) if len(days) >= 3 else None


def fitted(recs: dict, name: str, day: dt.date, splits: dict | None = None) -> Model | None:
    m = make(name)
    if name == 'bayes_heating':
        base = baseline(splits or {}, day)
        if base is None or not m.fit(training(heating_recs(recs, splits), name, day)):
            return None
        m.offset = base
        return m
    return m if m.fit(training(recs, name, day)) else None


def proxy_forecasts(recs: dict, name: str, start: dt.date, end: dt.date,
                    splits: dict | None = None) -> dict[dt.date, float]:
    """Rolling origin: fit on days before D, forecast D with D's observed temperature."""
    out = {}
    k = start
    while k <= end:
        r, temp = recs.get(k), temp_of(recs, k)
        if r and temp is not None:
            m = fitted(recs, name, k, splits)
            if m:
                out[k] = m.predict(temp)
        k += dt.timedelta(days=1)
    return out


def errors(recs: dict, issued: dict[dt.date, float], today: dt.date, days: int) -> list[float]:
    """Actual minus forecast for scored days in the last `days` before today."""
    out = []
    for k, f in issued.items():
        r = recs.get(k)
        if r and not r.excluded and r.gas is not None and today - dt.timedelta(days=days) <= k < today:
            out.append(r.gas - f)
    return out


def mae(errs: list[float]) -> float | None:
    return sum(abs(e) for e in errs) / len(errs) if errs else None


def choose(scores: dict[str, dict], current: str | None, min_days: int = 14, margin: float = 0.05) -> str:
    """Lowest 28-day MAE wins; the current model is kept unless beaten by `margin`."""
    ok = {n: s['mae28'] for n, s in scores.items() if s['mae28'] is not None and s['days28'] >= min_days}
    if not ok:
        return current or 'bayes'
    best = min(ok, key=ok.get)
    if current in ok and ok[best] > ok[current] * (1 - margin):
        return current
    return best
