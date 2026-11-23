"""Diagnostics: the last raw scrape and the values derived from it."""
from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from .coordinator import LlmServerConfigEntry


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: LlmServerConfigEntry) -> dict[str, Any]:
    coordinator = entry.runtime_data
    raw = dict(coordinator.last_raw)
    if isinstance(raw.get("metrics"), str):  # Prometheus text can be long: keep the metric lines only
        raw["metrics"] = [line for line in raw["metrics"].splitlines() if line and not line.startswith("#")]
    return {"url": coordinator.base, "options": dict(entry.options), "values": coordinator.data, "raw": raw}
