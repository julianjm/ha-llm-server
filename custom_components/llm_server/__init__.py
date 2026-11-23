"""LLM Server: monitor a local Basalt, SGLang or vLLM server."""
from __future__ import annotations

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr

from .const import BACKEND_NAMES, DOMAIN
from .coordinator import LlmServerConfigEntry, LlmServerCoordinator

PLATFORMS = [Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: LlmServerConfigEntry) -> bool:
    coordinator = LlmServerCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator

    @callback
    def sync_device() -> None:
        """Keep the device's engine, model and version in step with what the server is running now."""
        data = coordinator.data or {}
        device = dr.async_get(hass).async_get_device_by_identifier((DOMAIN, entry.unique_id), entry.entry_id)
        if device is None or not data:
            return
        manufacturer = BACKEND_NAMES.get(data.get("backend"), data.get("backend"))
        model, version = data.get("model"), data.get("version")
        if (device.manufacturer, device.model, device.sw_version) != (manufacturer, model, version):
            dr.async_get(hass).async_update_device(device.id, manufacturer=manufacturer, model=model, sw_version=version)

    entry.async_on_unload(coordinator.async_add_listener(sync_device))
    entry.async_on_unload(entry.add_update_listener(_reload))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    sync_device()
    return True


async def _reload(hass: HomeAssistant, entry: LlmServerConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LlmServerConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
