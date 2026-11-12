# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core import data, gas
from core.model import Day, Model, _wls

TZ = ZoneInfo('Europe/London')


def hours(start: dt.datetime, n: int):
    """UTC hour starts from a local start time."""
    t0 = start.astimezone(dt.timezone.utc)
    return [(t0 + dt.timedelta(hours=i)).timestamp() for i in range(n)]


def test_daily_from_sum_across_clock_change():
    # 25 Oct 2026 has 25 local hours; 1 kWh booked every hour
    ts = hours(dt.datetime(2026, 10, 24, 23, tzinfo=TZ), 27)
    rows = [{'start': t, 'sum': float(i)} for i, t in enumerate(ts)]
    out = data.daily_from_sum(rows, TZ)
    assert out[dt.date(2026, 10, 25)] == 25.0


def test_daily_from_sum_drops_resets():
    ts = hours(dt.datetime(2026, 10, 1, 0, tzinfo=TZ), 4)
    rows = [{'start': ts[0], 'sum': 10}, {'start': ts[1], 'sum': 11}, {'start': ts[2], 'sum': 0}, {'start': ts[3], 'sum': 2}]
    assert data.daily_from_sum(rows, TZ) == {dt.date(2026, 10, 1): 3.0}


def test_running_max_skips_midnight_carry_over():
    ts = hours(dt.datetime(2026, 10, 12, 0, tzinfo=TZ), 24)
    rows = [{'start': ts[0], 'max': 0.39}] + [{'start': t, 'max': 0.29 if i > 7 else 0.0} for i, t in enumerate(ts[1:], 1)]
    assert data.daily_running_max(rows, TZ) == {dt.date(2026, 10, 12): 0.29}


def test_hourly_share_sums_to_one():
    ts = hours(dt.datetime(2026, 10, 10, 0, tzinfo=TZ), 24 * 3)
    s, rows = 0.0, []
    for t in ts:
        h = dt.datetime.fromtimestamp(t, TZ).hour
        s += 3.0 if h == 7 else 1.0 if h == 18 else 0.0
        rows.append({'start': t, 'sum': s})
    share = data.hourly_share(rows, TZ, dt.date(2026, 10, 13))
    assert abs(sum(share) - 1) < 1e-9
    assert share[7] == 0.75 and share[18] == 0.25


def test_model_recovers_line_and_base():
    rng = np.random.default_rng(0)
    temps = rng.uniform(2, 18, 60)
    days = [Day(gas=1.0 + 3.0 * max(0, 14.5 - t), temp=float(t)) for t in temps]
    m = Model(fit_base=True, window=None)
    assert m.fit(days)
    assert m.base == 14.5 and abs(m.a - 1) < 1e-6 and abs(m.b - 3) < 1e-6


def test_prior_holds_early_season():
    # Two warm days alone can't pin a slope: the prior keeps it near last winter's 4.8
    days = [Day(gas=1.0, temp=16.0, age=1), Day(gas=1.2, temp=15.5, age=2)]
    m = Model(window=None, prior=gas.PRIOR, min_days=1)
    assert m.fit(days)
    assert 3.8 < m.b < 5.8


def test_weighted_temperature():
    assert gas.weighted([10.0, None]) is None
    assert gas.weighted([10.0, 10.0, None, 10.0]) == 10.0
    w = gas.weighted([4.0, 8.0])
    assert 4.0 < w < 6.0  # today counts more than yesterday


def test_curve_model_recovers_curvature():
    rng = np.random.default_rng(2)
    temps = rng.uniform(-2, 14, 80)
    days = [Day(gas=2.0 + 4.0 * max(0, 13.5 - t) + 0.1 * max(0, 13.5 - t) ** 2, temp=float(t)) for t in temps]
    m = Model(fit_base=True, window=None, curve=True)
    assert m.fit(days)
    assert m.base == 13.5 and abs(m.b - 4.0) < 1e-6 and abs(m.q - 0.1) < 1e-6
    assert m.predict(3.5) == pytest.approx(2.0 + 40.0 + 10.0)


def test_cold_day_without_heating_is_excluded():
    k = dt.date(2026, 10, 8)
    recs = gas.build({k: 0.4}, {k: 10.0, k - dt.timedelta(days=1): 11.0}, {k: 0.0}, None)
    assert recs[k].excluded


def test_ols15_window_starts_at_first_heating_day():
    day = dt.date(2026, 10, 24)
    temps, gases = {}, {}
    for i in range(1, 32):
        k = day - dt.timedelta(days=i)
        temps[k] = 10.0
        gases[k] = 8.0 if i <= 14 else 0.5
    recs = gas.build(gases, temps, None, None)
    tr = gas.training(recs, 'ols15', day)
    assert max(d.age for d in tr) == 14


def test_choose_keeps_current_unless_clearly_beaten():
    s = lambda m: {'mae28': m, 'days28': 20}
    assert gas.choose({'ols15': s(2.0), 'bayes': s(1.95)}, 'ols15') == 'ols15'
    assert gas.choose({'ols15': s(2.0), 'bayes': s(1.8)}, 'ols15') == 'bayes'
    assert gas.choose({'ols15': {'mae28': 1.0, 'days28': 3}}, None) == 'bayes'


def test_hour_means_need_the_whole_day():
    # 25 Oct 2026 has 25 local hours: 24 are enough, 23 are not
    ts = hours(dt.datetime(2026, 10, 25, 0, tzinfo=TZ), 25)
    full = data.hour_means({t: float(i) for i, t in enumerate(ts)}, TZ)
    assert full == {dt.date(2026, 10, 25): 12.0}
    assert data.hour_means({t: 1.0 for t in ts[:24]}, TZ) == {dt.date(2026, 10, 25): 1.0}
    assert data.hour_means({t: 1.0 for t in ts[:23]}, TZ) == {}


def test_base_is_chosen_with_the_prior_penalty():
    rng = np.random.default_rng(1)
    temps = rng.uniform(4, 16, 12)
    days = [Day(gas=float(1.0 + 3.0 * max(0, 14 - t) + rng.normal(0, 1.5)), temp=float(t)) for t in temps]
    prior = {'a': (1.5, 0.2), 'b': (6.0, 0.2)}
    m = Model(fit_base=True, window=None, prior=prior)
    assert m.fit(days)

    def objective(base):
        X = np.column_stack([np.ones(len(days)), np.maximum(0.0, base - np.array([d.temp for d in days]))])
        return _wls(X, np.array([d.gas for d in days]), np.ones(len(days)), [1.5, 6.0], [0.2, 0.2])[1]

    assert objective(m.base) == min(objective(b) for b in np.arange(10.0, 18.01, 0.5))
