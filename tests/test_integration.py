# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""End-to-end: config flow, setup on synthetic statistics, sensors and away tracking."""
import datetime as dt
import math
from unittest.mock import patch

import pytest
from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.gas_forecast.const import DOMAIN
from custom_components.gas_forecast.core import gas

CONF = {
    'gas_energy': 'sensor.gas',
    'outdoor_temperature': 'sensor.outdoor',
    'heating_time': 'sensor.heating_time',
    'climate': 'climate.thermostat',
    'weather': 'weather.home',
}
A, B, BASE = 1.0, 3.0, 15.0


def temp_of(day: dt.date) -> float:
    return 10 + 4 * math.sin(day.toordinal() / 5)


def model_temp(day: dt.date) -> float:
    return gas.weighted([temp_of(day - dt.timedelta(days=i)) for i in range(gas.LOOKBACK)])


def synthetic(hass, start, end, ids, period, units, types):
    """Hourly rows from `start` to now: gas booked at 07:00, temperature constant per day."""
    tz = dt_util.get_default_time_zone()
    out = {i: [] for i in ids}
    t = start.astimezone(dt.UTC).replace(minute=0, second=0, microsecond=0)
    now = dt_util.utcnow()
    total, heat = 0.0, 0.0
    while t < now:
        local = t.astimezone(tz)
        d = local.date()
        if local.hour == 7:
            total += A + B * max(0.0, BASE - model_temp(d))
        heat = 0.0 if local.hour == 1 else heat + (0.1 if local.hour == 7 else 0.0)
        ts = t.timestamp()
        out['sensor.gas'].append({'start': ts, 'sum': total})
        out['sensor.outdoor'].append({'start': ts, 'mean': temp_of(d)})
        out['sensor.heating_time'].append({'start': ts, 'max': heat})
        t += dt.timedelta(hours=1)
    return out


@pytest.fixture(autouse=True)
def auto_enable(recorder_mock, enable_custom_integrations):
    yield


async def _weather(hass: HomeAssistant, hourly: bool = True, skew: float = 0.0):
    """Daily forecasts with high and low `skew` above the day's mean; 24 hourly ones from the current hour."""
    async def handler(call):
        today = dt_util.now().date()
        if call.data['type'] == 'hourly':
            if not hourly:
                raise ValueError('no hourly forecast')
            t0 = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
            hours = [t0 + dt.timedelta(hours=i) for i in range(24)]
            return {'weather.home': {'forecast': [
                {'datetime': t.isoformat(), 'temperature': temp_of(dt_util.as_local(t).date())} for t in hours]}}
        return {'weather.home': {'forecast': [
            {'datetime': dt_util.start_of_local_day(today + dt.timedelta(days=i)).isoformat(),
             'temperature': temp_of(today + dt.timedelta(days=i)) + 3 + skew,
             'templow': temp_of(today + dt.timedelta(days=i)) - 3 + skew}
            for i in range(10)]}}
    hass.services.async_register('weather', 'get_forecasts', handler, supports_response=SupportsResponse.ONLY)


async def test_config_flow(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={'source': SOURCE_USER})
    assert result['type'] is FlowResultType.FORM
    with patch('custom_components.gas_forecast.async_setup_entry', return_value=True), \
            patch('custom_components.gas_forecast.async_unload_entry', return_value=True):
        result = await hass.config_entries.flow.async_configure(result['flow_id'], CONF)
        await hass.async_block_till_done()
        assert result['type'] is FlowResultType.CREATE_ENTRY
        assert result['data'] == CONF
        await hass.config_entries.async_remove(result['result'].entry_id)


async def test_setup_forecasts_and_scores(hass: HomeAssistant):
    await hass.config.async_update(time_zone='Europe/London')
    await _weather(hass)
    hass.states.async_set('climate.thermostat', 'heat')
    entry = MockConfigEntry(domain=DOMAIN, data=CONF)
    entry.add_to_hass(hass)
    with patch('custom_components.gas_forecast.coordinator.statistics_during_period', side_effect=synthetic):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    today = dt_util.now().date()
    t = model_temp(today)
    st = hass.states.get('sensor.gas_forecast_today')
    assert st is not None, [s.entity_id for s in hass.states.async_all()]
    # the forecast uses (hi + lo) / 2 = the true mean, so every model should land on the line
    assert float(st.state) == pytest.approx(A + B * max(0.0, BASE - t), abs=0.3)
    assert len(st.attributes['forecast']) >= 10
    assert st.attributes['hourly_share'][7] == pytest.approx(1.0)
    assert st.attributes['model'] in ('ols15', 'change_point', 'bayes', 'bayes_curve')

    err = hass.states.get('sensor.gas_forecast_error')
    assert float(err.state) < 0.5
    assert set(err.attributes['models']) == {'ols15', 'change_point', 'bayes', 'bayes_heating', 'bayes_curve'}

    # Away tracking: off then back on accumulates minutes for today
    coordinator = entry.runtime_data
    hass.states.async_set('climate.thermostat', 'off')
    await hass.async_block_till_done()
    coordinator._off_since -= dt.timedelta(minutes=30)
    hass.states.async_set('climate.thermostat', 'heat')
    await hass.async_block_till_done()
    # summed: just after midnight the 30 minutes are split across two days
    assert sum(coordinator.saved['away'].values()) == pytest.approx(30, abs=1)

    assert await hass.config_entries.async_unload(entry.entry_id)


def _shifted_evening():
    """The real 24 Oct evening moved to end an hour before now, in whole half-hours."""
    from fixtures_evenings import BATHROOM_24, CLIMATE_24, GAS_24, KITCHEN_24, POWER_24, ts
    off = (dt_util.utcnow().timestamp() - 3600 - ts('2026-10-25T00:30')) // 1800 * 1800
    sh = lambda rows: [(t + off, *rest) for t, *rest in rows]
    return {'start': ts('2026-10-24T16:30') + off, 'gas': sh(GAS_24), 'shower': [sh(BATHROOM_24)],
            'hob': [sh(KITCHEN_24)], 'climate': sh(CLIMATE_24), 'power': sh(POWER_24)}


async def test_gas_split_on_a_real_evening(hass: HomeAssistant):
    await hass.config.async_update(time_zone='Europe/London')
    await _weather(hass)
    hass.states.async_set('climate.thermostat', 'auto')
    conf = {**CONF, 'shower_humidity': ['sensor.bathroom_humidity'], 'hob_humidity': ['sensor.kitchen_humidity'],
            'electricity_power': 'sensor.power'}
    entry = MockConfigEntry(domain=DOMAIN, data=conf)
    entry.add_to_hass(hass)
    added = []
    with patch('custom_components.gas_forecast.coordinator.statistics_during_period', side_effect=synthetic), \
            patch('custom_components.gas_forecast.coordinator.GasForecastCoordinator._history',
                  side_effect=lambda start: _shifted_evening()), \
            patch('custom_components.gas_forecast.coordinator.async_add_external_statistics',
                  side_effect=lambda hass, meta, rows: added.append((meta, list(rows)))):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    for use in ('heating', 'hot_water', 'hob', 'boost'):
        st = hass.states.get(f'sensor.gas_forecast_{use}_today')
        assert st is not None and st.attributes['device_class'] == 'energy'
        # the integration writes the statistics: the recorder must not compile its own from the state
        assert 'state_class' not in st.attributes and 'last_reset' not in st.attributes
        assert st.attributes['statistic_id'].startswith('gas_forecast:')
    stats = {meta['statistic_id']: rows for meta, rows in added}
    assert stats['gas_forecast:heating'][-1]['sum'] == pytest.approx(5.492 + 5.136, abs=1e-3)
    assert stats['gas_forecast:hot_water'][-1]['sum'] == pytest.approx(4.019, abs=1e-3)
    assert stats['gas_forecast:boost'][-1]['sum'] == pytest.approx(5.136, abs=1e-3)
    assert stats['gas_forecast:hob'][-1]['sum'] == 0
    heat = stats['gas_forecast:heating']
    # whole hours, the last one ended
    assert all(r['start'].minute == 0 for r in heat) and heat[-1]['start'] <= dt_util.utcnow() - dt.timedelta(hours=1)
    meta = next(m for m, _ in added if m['statistic_id'] == 'gas_forecast:heating')
    assert meta['has_sum'] and meta['source'] == DOMAIN and meta['unit_of_measurement'] == 'kWh'
    sums = entry.runtime_data.saved['sums']
    assert sums[max(sums)]['heating'] == pytest.approx(5.492 + 5.136, abs=1e-3)
    days = entry.runtime_data.saved['split'].values()
    assert sum(d['hot_water'] for d in days) == pytest.approx(4.019, abs=1e-3)
    assert sum(d['heating'] for d in days) == pytest.approx(5.492 + 5.136, abs=1e-3)
    assert sum(d['hob'] for d in days) == 0
    assert sum(d['boost_heating'] for d in days) == pytest.approx(5.136, abs=1e-3)
    split_days = hass.states.get('sensor.gas_forecast_heating_today').attributes['days']
    assert sum(v['hot_water'] for v in split_days.values()) == pytest.approx(4.019, abs=1e-3)
    assert sum(v['boost'] for v in split_days.values()) == pytest.approx(5.136, abs=1e-3)
    kw = hass.states.get('sensor.gas_forecast_heating_today').attributes['kw']
    assert 14 < kw['dhw'] < 15 and 17 < kw['heat'] < 18
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_no_forecast_issued_without_the_weather(hass: HomeAssistant):
    await hass.config.async_update(time_zone='Europe/London')
    entry = MockConfigEntry(domain=DOMAIN, data=CONF)
    entry.add_to_hass(hass)
    with patch('custom_components.gas_forecast.coordinator.statistics_during_period', side_effect=synthetic):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    coordinator = entry.runtime_data
    today = dt_util.now().date().isoformat()
    # the forecast falls back to recent temperatures, but nothing is issued for scoring
    assert hass.states.get('sensor.gas_forecast_today').state not in ('unknown', 'unavailable')
    assert all(today not in issued for issued in coordinator.saved['issued'].values())
    assert coordinator._retry_unsub is not None
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert coordinator._retry_unsub is None


async def test_issued_forecast_is_the_hourly_mean(hass: HomeAssistant):
    # the daily forecast's midpoint is 2 °C warmer than the day's mean; only the hourly mean is scored
    await hass.config.async_update(time_zone='Europe/London')
    await _weather(hass, skew=2.0)
    entry = MockConfigEntry(domain=DOMAIN, data=CONF)
    entry.add_to_hass(hass)
    with patch('custom_components.gas_forecast.coordinator.statistics_during_period', side_effect=synthetic):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    c = entry.runtime_data
    today = dt_util.now().date()
    t = model_temp(today)
    m = c.saved['active']
    assert c.saved['issued'][m][today.isoformat()] == pytest.approx(A + B * max(0.0, BASE - t), abs=0.3)
    st = hass.states.get('sensor.gas_forecast_today')
    assert float(st.state) == pytest.approx(A + B * max(0.0, BASE - t), abs=0.3)
    assert c._retry_unsub is None
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_no_forecast_issued_without_the_hourly_forecast(hass: HomeAssistant):
    await hass.config.async_update(time_zone='Europe/London')
    await _weather(hass, hourly=False)
    entry = MockConfigEntry(domain=DOMAIN, data=CONF)
    entry.add_to_hass(hass)
    with patch('custom_components.gas_forecast.coordinator.statistics_during_period', side_effect=synthetic):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    c = entry.runtime_data
    today = dt_util.now().date().isoformat()
    assert all(today not in issued for issued in c.saved['issued'].values())
    assert hass.states.get('sensor.gas_forecast_today').state not in ('unknown', 'unavailable')
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_history_rebuilds_from_the_last_stored_day(hass: HomeAssistant):
    from custom_components.gas_forecast.coordinator import GasForecastCoordinator
    entry = MockConfigEntry(domain=DOMAIN, data=CONF)
    entry.add_to_hass(hass)
    c = GasForecastCoordinator(hass, entry)
    today = dt.date(2026, 10, 29)
    assert c._history_from(today) == dt.date(2026, 10, 20)
    c.saved['split'] = {'2026-10-28': {}, '2026-10-29': {}}
    # no statistics yet: the whole history once, to start them
    assert c._history_from(today) == dt.date(2026, 10, 20)
    c.saved['sums'] = {'2026-10-28': {}}
    assert c._history_from(today) == dt.date(2026, 10, 27)
    c.saved['split'] = {'2026-10-23': {}, '2026-10-24': {}}
    assert c._history_from(today) == dt.date(2026, 10, 24)
    c.saved['split'] = {'2026-09-01': {}}
    assert c._history_from(today) == dt.date(2026, 10, 20)


async def test_options_flow_adds_split_sensors(hass: HomeAssistant):
    entry = MockConfigEntry(domain=DOMAIN, data=CONF)
    entry.add_to_hass(hass)
    with patch('custom_components.gas_forecast.async_setup_entry', return_value=True), \
            patch('custom_components.gas_forecast.async_unload_entry', return_value=True):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        assert result['type'] is FlowResultType.FORM
        result = await hass.config_entries.options.async_configure(
            result['flow_id'], {**CONF, 'shower_humidity': ['sensor.bathroom_humidity']})
        assert result['type'] is FlowResultType.CREATE_ENTRY
        assert entry.options['shower_humidity'] == ['sensor.bathroom_humidity']
        await hass.config_entries.async_remove(entry.entry_id)


async def test_split_statistics_reach_the_recorder(hass: HomeAssistant):
    from homeassistant.components.recorder.statistics import statistics_during_period as real_stats
    from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

    await hass.config.async_update(time_zone='Europe/London')
    await _weather(hass)
    hass.states.async_set('climate.thermostat', 'auto')
    conf = {**CONF, 'shower_humidity': ['sensor.bathroom_humidity'], 'hob_humidity': ['sensor.kitchen_humidity'],
            'electricity_power': 'sensor.power'}
    entry = MockConfigEntry(domain=DOMAIN, data=conf)
    entry.add_to_hass(hass)
    with patch('custom_components.gas_forecast.coordinator.statistics_during_period', side_effect=synthetic), \
            patch('custom_components.gas_forecast.coordinator.GasForecastCoordinator._history',
                  side_effect=lambda start: _shifted_evening()):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        await async_wait_recording_done(hass)

    ids = {'gas_forecast:heating', 'gas_forecast:hot_water'}
    rows = await hass.async_add_executor_job(real_stats, hass, dt_util.utcnow() - dt.timedelta(days=1), None, ids,
                                             'hour', None, {'sum'})
    assert rows['gas_forecast:heating'][-1]['sum'] == pytest.approx(5.492 + 5.136, abs=1e-3)
    assert rows['gas_forecast:hot_water'][-1]['sum'] == pytest.approx(4.019, abs=1e-3)
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_statistics_continue_across_rebuild_windows(hass: HomeAssistant):
    """A later, shorter rebuild continues the sums from the last day before it and rewrites its hours unchanged."""
    from fixtures_evenings import BATHROOM_24, CLIMATE_24, GAS_24, KITCHEN_24, POWER_24, TZ, ts

    from custom_components.gas_forecast.coordinator import GasForecastCoordinator
    entry = MockConfigEntry(domain=DOMAIN, data={**CONF, 'shower_humidity': ['sensor.b'], 'hob_humidity': ['sensor.k']})
    entry.add_to_hass(hass)
    c = GasForecastCoordinator(hass, entry)

    def hist(start):
        gas = [g for g in GAS_24 if g[0] >= start] or GAS_24[-1:]
        before = [g for g in GAS_24 if g[0] < start]
        if before:
            gas = [(start, before[-1][1]), *gas]
        return {'start': start, 'gas': gas, 'shower': [BATHROOM_24], 'hob': [KITCHEN_24],
                'climate': CLIMATE_24, 'power': POWER_24}

    def rebuild(start, now):
        _, rows = c._split(c.saved, hist(start), {}, TZ, dt.date(2026, 10, 25), dt.datetime.fromtimestamp(now, TZ))
        return {use: dict(r) for use, r in rows.items()}

    full = rebuild(ts('2026-10-24T00:00'), ts('2026-10-25T02:00'))
    later = rebuild(ts('2026-10-25T00:00'), ts('2026-10-25T02:00'))
    # the later window holds only 25 Oct, and its hours match the full rebuild's: no step at the join
    assert min(later['heating']) == ts('2026-10-25T00:00')
    for use in later:
        for h, s in later[use].items():
            assert s == pytest.approx(full[use][h])
    assert c.saved['sums']['2026-10-24']['heating'] == pytest.approx(5.492 + 5.136, abs=1e-3)


async def test_history_reads_meter_reports_and_sensor_gaps(hass: HomeAssistant):
    """The recorder query keeps the meter's last unchanged report and marks humidity gaps as None."""
    from homeassistant.components.recorder import get_instance
    from pytest_homeassistant_custom_component.components.recorder.common import async_wait_recording_done

    from custom_components.gas_forecast.coordinator import GasForecastCoordinator
    entry = MockConfigEntry(domain=DOMAIN, data={**CONF, 'hob_humidity': ['sensor.k']})
    entry.add_to_hass(hass)
    start = dt_util.utcnow() - dt.timedelta(minutes=1)
    hass.states.async_set('sensor.gas', '1.0')
    hass.states.async_set('sensor.k', '50')
    hass.states.async_set('climate.thermostat', 'auto', {'hvac_action': 'idle'})
    await async_wait_recording_done(hass)
    hass.states.async_set('sensor.gas', '1.0')  # reported, unchanged
    hass.states.async_set('sensor.k', 'unavailable')
    await async_wait_recording_done(hass)
    reported = hass.states.get('sensor.gas').last_reported.timestamp()
    hass.states.async_set('sensor.gas', '1.5')
    hass.states.async_set('sensor.k', '52')
    await async_wait_recording_done(hass)

    hist = await get_instance(hass).async_add_executor_job(GasForecastCoordinator(hass, entry)._history, start)
    (t0, v0, seen0), (_, v1, _) = hist['gas']
    assert (v0, v1) == (1.0, 1.5) and seen0 == pytest.approx(reported) and seen0 > t0
    assert [v for _, v in hist['hob'][0]] == [50.0, None, 52.0]
    assert hist['shower'] == []
