# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Hourly refresh: read statistics and the weather forecast, refit, score and forecast."""
import copy
import datetime as dt
import logging
from zoneinfo import ZoneInfo

from homeassistant.components.recorder import get_instance, history
from homeassistant.components.recorder.models import StatisticData, StatisticMeanType, StatisticMetaData
from homeassistant.components.recorder.statistics import async_add_external_statistics, statistics_during_period
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later, async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter

from .const import (CONF_CLIMATE, CONF_GAS, CONF_HEATING_TIME, CONF_HOB_HUMIDITY, CONF_OUTDOOR,
                    CONF_OUTDOOR_FALLBACK, CONF_POWER, CONF_SHOWER_HUMIDITY, CONF_WEATHER, DOMAIN,
                    EVIDENCE_DAYS, HISTORY_DAYS, KEEP_AWAY_DAYS, KEEP_ISSUED_DAYS, KEEP_SPLIT_DAYS, SPLIT_ATTR_DAYS,
                    STAT_IDS, STORE_VERSION)
from .core import data, gas, split

_LOGGER = logging.getLogger(__name__)
FORECAST_DAYS = 10
WORK = ('issued', 'split', 'evidence', 'sums', 'active')  # saved state the refit changes, on a copy in the executor


class GasForecastCoordinator(DataUpdateCoordinator[dict]):
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, _LOGGER, name=DOMAIN, update_interval=dt.timedelta(hours=1), config_entry=entry)
        self.conf = {**entry.data, **entry.options}
        self._store = Store(hass, STORE_VERSION, f'{DOMAIN}.{entry.entry_id}')
        # issued: model -> ISO date -> kWh forecast made on that day; away: ISO date -> minutes with heating off;
        # split: ISO date -> gas by use; evidence: ISO date -> clean kWh/hours per use;
        # sums: ISO date -> each use's statistics sum at the day's last hour
        self.saved = {'issued': {}, 'away': {}, 'active': None, 'split': {}, 'evidence': {}, 'sums': {}}
        self._proxy: dict[str, dict[dt.date, float]] = {}
        self._proxy_day: dt.date | None = None
        self._off_since: dt.datetime | None = None
        self._unsub = None
        self._retry_day: dt.date | None = None
        self._last_fc: tuple[dict[dt.date, float], set[dt.date]] = ({}, set())
        self._retry_unsub = None

    async def async_load(self) -> None:
        if stored := await self._store.async_load():
            self.saved.update(stored)
        if climate := self.conf.get(CONF_CLIMATE):
            st = self.hass.states.get(climate)
            if st and st.state == 'off':
                self._off_since = dt_util.now()
            self._unsub = async_track_state_change_event(self.hass, climate, self._climate_changed)

    async def async_shutdown_tracking(self) -> None:
        self._flush_off(dt_util.now())
        if self._unsub:
            self._unsub()
        if self._retry_unsub:
            self._retry_unsub()
            self._retry_unsub = None
        await self._store.async_save(self.saved)

    @callback
    def _climate_changed(self, event: Event[EventStateChangedData]) -> None:
        now = dt_util.now()
        old, new = event.data['old_state'], event.data['new_state']
        if old and old.state == 'off':
            self._flush_off(now)
        self._off_since = now if new and new.state == 'off' else None

    def _flush_off(self, now: dt.datetime) -> None:
        """Add minutes with the heating off to each local day they fall in."""
        t = self._off_since
        if t is None:
            return
        while t < now:
            nxt = min(now, dt_util.start_of_local_day(t.date() + dt.timedelta(days=1)))
            k = t.date().isoformat()
            self.saved['away'][k] = self.saved['away'].get(k, 0) + (nxt - t).total_seconds() / 60
            t = nxt
        self._off_since = now

    async def _retry(self, _now) -> None:
        self._retry_unsub = None
        await self.async_request_refresh()

    async def _forecasts(self, kind: str) -> list[dict]:
        entity = self.conf[CONF_WEATHER]
        resp = await self.hass.services.async_call(
            'weather', 'get_forecasts', {'entity_id': entity, 'type': kind}, blocking=True, return_response=True)
        return [f for f in resp.get(entity, {}).get('forecast', []) if f.get('temperature') is not None]

    async def _forecast_temps(self, tz: ZoneInfo, observed: dict[float, float]) -> tuple[dict[dt.date, float], set[dt.date]]:
        """Daily mean temperatures ahead, and the days measured like the recorded ones.

        A day the observed and hourly forecast hours cover gets their mean, as recorded days do; other days get the
        daily forecast's (high + low) / 2, which is only shown, never scored.
        """
        try:
            daily = await self._forecasts('daily')
        except Exception as err:  # weather integration down: keep the last forecast
            _LOGGER.warning('weather forecast unavailable: %s', err)
            return self._last_fc
        try:
            hourly = await self._forecasts('hourly')
        except Exception as err:
            _LOGGER.warning('hourly weather forecast unavailable: %s', err)
            hourly = []
        out = {}
        for f in daily:
            k = dt_util.parse_datetime(f['datetime']).astimezone(tz).date()
            lo = f.get('templow')
            out[k] = (f['temperature'] + lo) / 2 if lo is not None else f['temperature']
        hours = dict(observed)
        ahead = set()
        for f in hourly:
            t = dt_util.parse_datetime(f['datetime']).timestamp()
            hours.setdefault(t, f['temperature'])
            ahead.add(dt.datetime.fromtimestamp(t, tz).date())
        exact = {k: v for k, v in data.hour_means(hours, tz).items() if k in ahead}
        out.update(exact)
        self._last_fc = (out, set(exact))
        return self._last_fc

    async def _async_update_data(self) -> dict:
        tz = ZoneInfo(self.hass.config.time_zone)
        now = dt_util.now()
        today = now.date()
        self._flush_off(now)
        start = dt_util.start_of_local_day(gas.season_start(today) - dt.timedelta(days=2))
        ids = {self.conf[CONF_GAS], self.conf[CONF_OUTDOOR]}
        for key in (CONF_OUTDOOR_FALLBACK, CONF_HEATING_TIME):
            if self.conf.get(key):
                ids.add(self.conf[key])
        stats = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period, self.hass, start, None, ids, 'hour', None, {'sum', 'mean', 'max'})
        fc, exact = await self._forecast_temps(tz, self._observed_hours(stats, today))
        if today not in exact and self._retry_unsub is None and self._retry_day != today:
            # the weather entity is often not ready yet just after Home Assistant starts; once a day, then hourly
            self._retry_day = today
            self._retry_unsub = async_call_later(self.hass, 300, self._retry)
        self._prune(today)
        away = {dt.date.fromisoformat(k): v for k, v in self.saved['away'].items()}
        hist = None
        if self._split_enabled():
            hist = await get_instance(self.hass).async_add_executor_job(
                self._history, dt_util.start_of_local_day(self._history_from(today)))
        work = copy.deepcopy({k: self.saved[k] for k in WORK})
        result, rows = await self.hass.async_add_executor_job(
            self._compute, work, stats, fc, exact, away, hist, tz, today, now)
        self.saved.update(work)
        self._add_statistics(rows)
        await self._store.async_save(self.saved)
        return result

    def _observed_hours(self, stats: dict, today: dt.date) -> dict[float, float]:
        """Today's recorded hourly outdoor means, the fallback sensor filling hours the main one lacks."""
        since = dt_util.start_of_local_day(today).timestamp()
        out: dict[float, float] = {}
        for key in (CONF_OUTDOOR, CONF_OUTDOOR_FALLBACK):
            for r in stats.get(self.conf.get(key), []) if self.conf.get(key) else []:
                if r['start'] >= since and r.get('mean') is not None:
                    out.setdefault(r['start'], r['mean'])
        return out

    def _add_statistics(self, rows: dict[str, list[tuple[int, float]]] | None) -> None:
        """Write the split's hourly sums; rebuilt hours replace what was there."""
        for use, series in (rows or {}).items():
            if not series:
                continue
            meta = StatisticMetaData(mean_type=StatisticMeanType.NONE, has_sum=True, name=STAT_IDS[use][1],
                                     source=DOMAIN, statistic_id=f'{DOMAIN}:{STAT_IDS[use][0]}',
                                     unit_class=EnergyConverter.UNIT_CLASS, unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR)
            async_add_external_statistics(self.hass, meta, [StatisticData(start=dt_util.utc_from_timestamp(h), sum=s)
                                                            for h, s in series])

    def _split_enabled(self) -> bool:
        return bool(self.conf.get(CONF_CLIMATE) and (self.conf.get(CONF_SHOWER_HUMIDITY) or self.conf.get(CONF_HOB_HUMIDITY)))

    def _history_from(self, today: dt.date) -> dt.date:
        """First day to rebuild: the whole history once (and when the statistics start), then from the last stored day."""
        if not self.saved['split'] or not self.saved['sums']:
            return today - dt.timedelta(days=HISTORY_DAYS - 1)
        last = dt.date.fromisoformat(max(self.saved['split']))
        back = min(max((today - last).days, 2), HISTORY_DAYS - 1)
        return today - dt.timedelta(days=back)

    def _history(self, start: dt.datetime) -> dict:
        """State history for the gas split (recorder executor)."""
        def numeric(states, gaps=False):
            out = []
            for st in states:
                try:
                    out.append((st.last_updated.timestamp(), float(st.state)))
                except (TypeError, ValueError):
                    if gaps:
                        out.append((st.last_updated.timestamp(), None))
            return out

        def readings(states):
            """Meter readings with when each was last reported unchanged."""
            out = []
            for st in states:
                try:
                    out.append((st.last_updated.timestamp(), float(st.state), st.last_reported.timestamp()))
                except (TypeError, ValueError):
                    continue
            return out

        shower = list(self.conf.get(CONF_SHOWER_HUMIDITY) or [])
        hob = list(self.conf.get(CONF_HOB_HUMIDITY) or [])
        num = history.get_significant_states(self.hass, start, None, [*shower, *hob], include_start_time_state=True,
                                             significant_changes_only=False, no_attributes=True) if shower or hob else {}
        # only this query carries last_reported
        meter = history.state_changes_during_period(self.hass, start, None, self.conf[CONF_GAS], no_attributes=True,
                                                    include_start_time_state=True)
        clim = history.get_significant_states(self.hass, start, None, [self.conf[CONF_CLIMATE]],
                                              include_start_time_state=True, significant_changes_only=False)
        climate = [(st.last_updated.timestamp(), st.attributes.get('hvac_action'), st.attributes.get('preset_mode'))
                   for st in clim.get(self.conf[CONF_CLIMATE], []) if st.state not in ('unavailable', 'unknown')]
        power = None
        if self.conf.get(CONF_POWER):
            rows = statistics_during_period(self.hass, start, None, {self.conf[CONF_POWER]}, '5minute', None,
                                            {'mean', 'max'})
            power = [(r['start'], r['mean'], r.get('max')) for r in rows.get(self.conf[CONF_POWER], [])
                     if r.get('mean') is not None]
        return {
            'start': start.timestamp(),
            'gas': readings(meter.get(self.conf[CONF_GAS], [])),
            'shower': [numeric(num.get(e, []), gaps=True) for e in shower],
            'hob': [numeric(num.get(e, []), gaps=True) for e in hob],
            'climate': climate,
            'power': power,
        }

    def _split(self, saved: dict, hist: dict, lts_gas: dict, tz: ZoneInfo, today: dt.date,
               now: dt.datetime) -> tuple[dict, dict]:
        """Recompute the split for the fetched window and store its days; the hourly sums for statistics."""
        evs = split.events(hist['shower'], hist['hob'], hist['power'], hist['climate'])
        first = dt.datetime.fromtimestamp(hist['start'], tz).date()
        old = [v for d, v in saved['evidence'].items()
               if d < first.isoformat() and d >= (today - dt.timedelta(days=EVIDENCE_DAYS)).isoformat()]
        kw = split.kw_from(old)
        slots = split.build_slots(hist['start'], now.timestamp(), hist['gas'], hist['climate'], evs, kw)
        for d in [d for d in saved['evidence'] if d >= first.isoformat()]:
            del saved['evidence'][d]
        for d, ev in split.evidence(slots, evs, tz).items():
            saved['evidence'][d.isoformat()] = ev
        kw = split.kw_from([v for d, v in saved['evidence'].items()
                            if d >= (today - dt.timedelta(days=EVIDENCE_DAYS)).isoformat()])
        for d, v in split.daily(slots, kw, tz).items():
            lts = lts_gas.get(d)
            v['reliable'] = (d < today and not v['lumped'] and v['slots'] >= 46
                             and lts is not None and abs(v['gas'] - lts) <= 0.5)
            saved['split'][d.isoformat()] = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}
        today_evs = [{'kind': e.kind, 'start': dt.datetime.fromtimestamp(e.start, tz).isoformat(timespec='minutes'),
                      'minutes': round((e.end - e.start) / 60, 1)}
                     for e in evs if dt.datetime.fromtimestamp(e.start, tz).date() == today
                     and (not e.inferred or split.claimed(slots, e))]
        # statistics continue from the sums at the end of the last day before the window
        before = [d for d in saved['sums'] if d < first.isoformat()]
        base = saved['sums'][max(before)] if before else {}
        rows, ends = split.cumulative(split.hourly(slots, now.timestamp()), base, tz)
        for d in [d for d in saved['sums'] if d >= first.isoformat()]:
            del saved['sums'][d]
        for d, s in ends.items():
            saved['sums'][d.isoformat()] = s
        return {'kw': {k: round(v, 2) for k, v in kw.items()}, 'events': today_evs}, rows

    def _compute(self, saved: dict, stats: dict, fc: dict, exact: set, away: dict, hist: dict | None, tz: ZoneInfo,
                 today: dt.date, now: dt.datetime) -> tuple[dict, dict | None]:
        """The refit, in the executor: changes only `saved`, a copy of the stored state."""
        rows = lambda key: stats.get(self.conf.get(key), []) if self.conf.get(key) else []
        gas_rows = rows(CONF_GAS)
        temp = data.daily_mean(rows(CONF_OUTDOOR), tz)
        for k, v in data.daily_mean(rows(CONF_OUTDOOR_FALLBACK), tz).items():
            temp.setdefault(k, v)
        heat = data.daily_running_max(rows(CONF_HEATING_TIME), tz) if self.conf.get(CONF_HEATING_TIME) else None
        lts_gas = data.daily_from_sum(gas_rows, tz)
        recs = gas.build(lts_gas, temp, heat, away)
        split_info, rows = self._split(saved, hist, lts_gas, tz, today, now) if hist else (None, None)
        splits = {dt.date.fromisoformat(d): v for d, v in saved['split'].items()}

        # Rolling-origin scores on observed weather until the model's own issued forecasts take over
        if self._proxy_day != today:
            first = gas.season_start(today)
            self._proxy = {n: gas.proxy_forecasts(recs, n, first, today - dt.timedelta(days=1), splits)
                           for n in gas.CANDIDATES}
            self._proxy_day = today

        # Forecast temperatures: the weather forecast, then the last week's mean beyond it
        recent = [temp[k] for k in temp if today - dt.timedelta(days=7) <= k < today]
        fallback = sum(recent) / len(recent) if recent else None
        month_end = (today.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
        horizon = max(today + dt.timedelta(days=FORECAST_DAYS - 1), month_end)
        temps, k = {}, today
        while k <= horizon:
            if k in fc or fallback is not None:
                temps[k] = fc.get(k, fallback)
            k += dt.timedelta(days=1)

        scores, models = {}, {}
        for n in gas.CANDIDATES:
            m = gas.fitted(recs, n, today, splits)
            models[n] = m
            issued = saved['issued'].setdefault(n, {})
            temp = gas.temp_of(recs, today, temps)
            # only a day measured like the recorded ones is issued: it is scored against them
            if m and temp is not None and today in exact and today.isoformat() not in issued:
                issued[today.isoformat()] = round(m.predict(temp), 3)
            merged = dict(self._proxy.get(n, {}))
            merged.update({dt.date.fromisoformat(d): v for d, v in issued.items()})
            e14, e28 = gas.errors(recs, merged, today, 14), gas.errors(recs, merged, today, 28)
            scores[n] = {'mae14': gas.mae(e14), 'mae28': gas.mae(e28), 'days14': len(e14), 'days28': len(e28)}
        active = gas.choose(scores, saved['active'])
        if active != saved['active']:
            _LOGGER.info('gas model: %s -> %s', saved['active'], active)
            saved['active'] = active

        m = models.get(active)
        forecast = []
        for k in sorted(temps):
            temp = gas.temp_of(recs, k, temps)
            if m and temp is not None:
                forecast.append({'date': k.isoformat(), 'kwh': round(m.predict(temp), 2),
                                 'weather': 'forecast' if k in fc else 'recent'})
        return {
            'today': forecast[0]['kwh'] if forecast and forecast[0]['date'] == today.isoformat() else None,
            'forecast': forecast,
            'hourly_share': data.hourly_share(gas_rows, tz, today),
            'model': active,
            'coefficients': {'a': round(m.a, 3), 'b': round(m.b, 3), 'q': round(m.q, 4), 'base': m.base,
                             'days': m.days, 'offset': round(m.offset, 3)} if m else None,
            'scores': {n: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()} for n, s in scores.items()},
            'split_today': saved['split'].get(today.isoformat()),
            'split_yesterday': saved['split'].get((today - dt.timedelta(days=1)).isoformat()),
            'split_days': self._split_days(saved, today),
            'split_info': split_info,
            'updated': now.isoformat(),
        }, rows

    def _split_days(self, saved: dict, today: dt.date) -> dict:
        """The last SPLIT_ATTR_DAYS days of the split, rounded, for charts that cover more than the statistics do."""
        old = (today - dt.timedelta(days=SPLIT_ATTR_DAYS - 1)).isoformat()
        return {d: {'heating': round(v['heating'], 3), 'hot_water': round(v['hot_water'], 3),
                    'taps': round(v.get('taps', 0.0), 3), 'hob': round(v['hob'], 3),
                    'boost': round(v.get('boost_heating', 0.0), 3), 'gas': round(v['gas'], 3), 'reliable': v.get('reliable')}
                for d, v in sorted(saved['split'].items()) if d >= old}

    def _prune(self, today: dt.date) -> None:
        old = (today - dt.timedelta(days=KEEP_ISSUED_DAYS)).isoformat()
        for issued in self.saved['issued'].values():
            for d in [d for d in issued if d < old]:
                del issued[d]
        old = (today - dt.timedelta(days=KEEP_AWAY_DAYS)).isoformat()
        for d in [d for d in self.saved['away'] if d < old]:
            del self.saved['away'][d]
        old = (today - dt.timedelta(days=KEEP_SPLIT_DAYS)).isoformat()
        for d in [d for d in self.saved['split'] if d < old]:
            del self.saved['split'][d]
        old = (today - dt.timedelta(days=EVIDENCE_DAYS + HISTORY_DAYS)).isoformat()
        for d in [d for d in self.saved['evidence'] if d < old]:
            del self.saved['evidence'][d]
        # the statistics only ever continue from the last day before a rebuild
        for d in sorted(self.saved['sums'])[:-HISTORY_DAYS * 2]:
            del self.saved['sums'][d]
