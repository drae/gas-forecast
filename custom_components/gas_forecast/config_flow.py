# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Config flow: pick the gas meter, outdoor temperature and weather entities."""
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (CONF_CLIMATE, CONF_GAS, CONF_HEATING_TIME, CONF_HOB_HUMIDITY, CONF_LOCAL, CONF_OUTDOOR,
                    CONF_OUTDOOR_FALLBACK, CONF_POWER, CONF_SHOWER_HUMIDITY, CONF_WEATHER, DOMAIN)

HUMIDITY = selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor', device_class='humidity', multiple=True))
SPLIT = {
    vol.Optional(CONF_SHOWER_HUMIDITY): HUMIDITY,
    vol.Optional(CONF_HOB_HUMIDITY): HUMIDITY,
    vol.Optional(CONF_POWER): selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor', device_class='power')),
}

SCHEMA = vol.Schema({
    vol.Required(CONF_GAS): selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor', device_class='energy')),
    vol.Required(CONF_OUTDOOR): selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor', device_class='temperature')),
    vol.Optional(CONF_OUTDOOR_FALLBACK): selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor', device_class='temperature')),
    vol.Optional(CONF_LOCAL): selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor', device_class='temperature')),
    vol.Optional(CONF_HEATING_TIME): selector.EntitySelector(selector.EntitySelectorConfig(domain='sensor')),
    vol.Optional(CONF_CLIMATE): selector.EntitySelector(selector.EntitySelectorConfig(domain='climate')),
    vol.Required(CONF_WEATHER): selector.EntitySelector(selector.EntitySelectorConfig(domain='weather')),
    **SPLIT,
})


class GasForecastConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(title='Gas forecast', data=user_input)
        return self.async_show_form(step_id='user', data_schema=SCHEMA)

    @staticmethod
    @callback
    def async_get_options_flow(entry: ConfigEntry) -> OptionsFlow:
        return GasForecastOptionsFlow()


class GasForecastOptionsFlow(OptionsFlow):
    """Every sensor can be changed later; the integration reloads with the new set."""

    async def async_step_init(self, user_input=None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(step_id='init', data_schema=self.add_suggested_values_to_schema(SCHEMA, current))
