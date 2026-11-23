"""Config flow: ask for the server URL, check that /metrics is Basalt, SGLang or vLLM."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_URL
from homeassistant.core import callback

from .const import BACKEND_NAMES, CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL, DOMAIN, MAX_SCAN_INTERVAL, MIN_SCAN_INTERVAL
from .coordinator import LlmServerConfigEntry, normalize_url, probe, server_id


class LlmServerConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            base = normalize_url(user_input[CONF_URL])
            try:
                backend = await probe(self.hass, base)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                errors["base"] = "cannot_connect"
            else:
                if backend is None:
                    errors["base"] = "unknown_backend"
                else:
                    await self.async_set_unique_id(server_id(base))
                    self._abort_if_unique_id_configured(updates={CONF_URL: base})
                    return self.async_create_entry(title=f"LLM server {server_id(base)}", data={CONF_URL: base},
                                                   description_placeholders={"backend": BACKEND_NAMES[backend]})
        return self.async_show_form(step_id="user", data_schema=vol.Schema({vol.Required(CONF_URL): str}), errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: LlmServerConfigEntry) -> OptionsFlow:
        return LlmServerOptionsFlow()


class LlmServerOptionsFlow(OptionsFlow):
    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        current = self.config_entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
        schema = vol.Schema({vol.Required(CONF_SCAN_INTERVAL, default=current):
                             vol.All(vol.Coerce(int), vol.Range(min=MIN_SCAN_INTERVAL, max=MAX_SCAN_INTERVAL))})
        return self.async_show_form(step_id="init", data_schema=schema)
