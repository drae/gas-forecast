# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Sensors: today's gas forecast (with the daily forecast list), the expected total today and the forecast error."""
import datetime as dt

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.const import UnitOfEnergy
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_interval
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from . import GasForecastConfigEntry
from .const import CONF_GAS, DOMAIN, STAT_IDS
from .coordinator import GasForecastCoordinator
from .core import data


async def async_setup_entry(hass: HomeAssistant, entry: GasForecastConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    c = entry.runtime_data
    entities = [GasForecastSensor(c), GasExpectedTodaySensor(c), GasForecastErrorSensor(c)]
    if c._split_enabled():
        entities += [GasSplitSensor(c, use) for use in ('heating', 'hot_water', 'hob', 'boost')]
    async_add_entities(entities)


class _Base(CoordinatorEntity[GasForecastCoordinator], SensorEntity):
    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: GasForecastCoordinator, key: str) -> None:
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_translation_key = key
        self._attr_unique_id = f'{entry.entry_id}_{key}'
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, entry.entry_id)}, name='Gas forecast',
                                            entry_type=DeviceEntryType.SERVICE)


class GasForecastSensor(_Base):
    """Whole-day gas forecast for today; the daily list runs to the month end."""
    _attr_icon = 'mdi:fire'
    _unrecorded_attributes = frozenset({'forecast', 'hourly_share', 'coefficients'})

    def __init__(self, coordinator: GasForecastCoordinator) -> None:
        super().__init__(coordinator, 'today')

    @property
    def native_value(self):
        return self.coordinator.data.get('today')

    @property
    def extra_state_attributes(self):
        d = self.coordinator.data
        return {k: d.get(k) for k in ('model', 'coefficients', 'local_offset', 'forecast', 'hourly_share', 'updated')}


class GasExpectedTodaySensor(_Base):
    """Gas used so far today plus the forecast's usual share for the rest of the day.

    Follows the meter between the hourly refits, and every few minutes as the rest of the day shrinks.
    """
    _attr_icon = 'mdi:fire-circle'

    def __init__(self, coordinator: GasForecastCoordinator) -> None:
        super().__init__(coordinator, 'expected_today')
        self._meter = coordinator.conf[CONF_GAS]

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(async_track_state_change_event(self.hass, self._meter, self._changed))
        self.async_on_remove(async_track_time_interval(self.hass, self._changed, dt.timedelta(minutes=5)))

    @callback
    def _changed(self, _event: Event[EventStateChangedData] | dt.datetime) -> None:
        self.async_write_ha_state()

    def _used(self) -> float | None:
        d = self.coordinator.data
        st = self.hass.states.get(self._meter)
        if d.get('date') != dt_util.now().date().isoformat() or d.get('midnight_reading') is None or st is None:
            return None
        try:
            return max(0.0, float(st.state) - d['midnight_reading'])
        except ValueError:
            return None

    @property
    def native_value(self):
        d = self.coordinator.data
        used = self._used()
        if d.get('today') is None or used is None:
            return None
        return round(data.expected_total(d['today'], used, d.get('hourly_share'), dt_util.now()), 2)

    @property
    def extra_state_attributes(self):
        used = self._used()
        return {'used': round(used, 3) if used is not None else None, 'forecast': self.coordinator.data.get('today')}


class GasForecastErrorSensor(_Base):
    """Mean absolute error of the active model over the last 28 scored days."""
    _attr_icon = 'mdi:target-variant'
    _unrecorded_attributes = frozenset({'models'})

    def __init__(self, coordinator: GasForecastCoordinator) -> None:
        super().__init__(coordinator, 'error')

    @property
    def native_value(self):
        d = self.coordinator.data
        return (d.get('scores', {}).get(d.get('model')) or {}).get('mae28')

    @property
    def extra_state_attributes(self):
        d = self.coordinator.data
        return {'model': d.get('model'), 'models': d.get('scores')}


class GasSplitSensor(_Base):
    """Gas so far today for one use, from the 30-minute meter slots; boost is the part of heating with boost on.

    No state class: late meter readings change earlier hours, so the integration writes the hourly statistics itself.
    """
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = None
    _unrecorded_attributes = frozenset({'yesterday', 'events', 'kw', 'days'})
    ICONS = {'heating': 'mdi:radiator', 'hot_water': 'mdi:shower-head', 'hob': 'mdi:pot-steam', 'boost': 'mdi:fire-alert'}
    FIELDS = {'boost': 'boost_heating'}

    def __init__(self, coordinator: GasForecastCoordinator, use: str) -> None:
        super().__init__(coordinator, f'{use}_today')
        self._use = use
        self._field = self.FIELDS.get(use, use)
        self._attr_icon = self.ICONS[use]

    @property
    def native_value(self):
        day = self.coordinator.data.get('split_today')
        return round(day.get(self._field, 0.0), 2) if day else None

    @property
    def extra_state_attributes(self):
        d = self.coordinator.data
        y = d.get('split_yesterday') or {}
        attrs = {'yesterday': round(y[self._field], 2) if self._field in y else None,
                 'yesterday_reliable': y.get('reliable'), 'statistic_id': f'{DOMAIN}:{STAT_IDS[self._field][0]}'}
        info = d.get('split_info') or {}
        if self._use == 'heating':
            today = d.get('split_today') or {}
            attrs['boost'] = round(today.get('boost_heating', 0.0), 2)
            attrs['kw'] = info.get('kw')
            attrs['days'] = d.get('split_days')
        elif self._use == 'hot_water':
            today = d.get('split_today') or {}
            attrs['shower'] = round(today.get('shower', 0.0), 2)
            attrs['taps'] = round(today.get('taps', 0.0), 2)
            attrs['events'] = [e for e in info.get('events', []) if e['kind'] in ('shower', 'tap')]
        elif self._use == 'hob':
            attrs['events'] = [e for e in info.get('events', []) if e['kind'] == 'hob']
        return attrs
