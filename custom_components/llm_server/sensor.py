"""Sensors of the LLM server, plus a GPU energy counter for the Energy dashboard and lifetime totals that survive
server restarts."""
from __future__ import annotations

from dataclasses import dataclass

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfEnergy,
    UnitOfInformation,
    UnitOfPower,
    UnitOfTemperature,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import BACKEND_NAMES, DOMAIN
from .coordinator import LlmServerConfigEntry, LlmServerCoordinator

TOK_S = "tok/s"
TOKENS = "tokens"
MEASURE = SensorStateClass.MEASUREMENT


@dataclass(frozen=True, kw_only=True)
class LlmSensorDescription(SensorEntityDescription):
    """`key` is the field in the coordinator data."""


SENSORS: tuple[LlmSensorDescription, ...] = (
    LlmSensorDescription(key="state", translation_key="state"),
    LlmSensorDescription(key="decode_speed", translation_key="decode_speed", native_unit_of_measurement=TOK_S,
                         state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="prefill_speed", translation_key="prefill_speed", native_unit_of_measurement=TOK_S,
                         state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="active_requests", translation_key="active_requests", state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="queued_requests", translation_key="queued_requests", state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="prefix_hit", translation_key="prefix_hit", native_unit_of_measurement=PERCENTAGE,
                         state_class=MEASURE, suggested_display_precision=1),
    LlmSensorDescription(key="prefix_hit_lifetime", translation_key="prefix_hit_lifetime", native_unit_of_measurement=PERCENTAGE,
                         state_class=MEASURE, suggested_display_precision=1),
    LlmSensorDescription(key="ttft", translation_key="ttft", device_class=SensorDeviceClass.DURATION,
                         native_unit_of_measurement=UnitOfTime.SECONDS, state_class=MEASURE, suggested_display_precision=2),
    LlmSensorDescription(key="kv_usage", translation_key="kv_usage", native_unit_of_measurement=PERCENTAGE,
                         state_class=MEASURE, suggested_display_precision=1),
    LlmSensorDescription(key="requests_total", translation_key="requests_total", state_class=SensorStateClass.TOTAL_INCREASING,
                         suggested_display_precision=0),
    LlmSensorDescription(key="output_tokens_total", translation_key="output_tokens_total", native_unit_of_measurement=TOKENS,
                         state_class=SensorStateClass.TOTAL_INCREASING, suggested_display_precision=0),
    LlmSensorDescription(key="prompt_tokens_total", translation_key="prompt_tokens_total", native_unit_of_measurement=TOKENS,
                         state_class=SensorStateClass.TOTAL_INCREASING, suggested_display_precision=0, entity_registry_enabled_default=False),
    LlmSensorDescription(key="cached_tokens_total", translation_key="cached_tokens_total", native_unit_of_measurement=TOKENS,
                         state_class=SensorStateClass.TOTAL_INCREASING, suggested_display_precision=0, entity_registry_enabled_default=False),
    LlmSensorDescription(key="gpu_power", translation_key="gpu_power", device_class=SensorDeviceClass.POWER,
                         native_unit_of_measurement=UnitOfPower.WATT, state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="gpu_utilization", translation_key="gpu_utilization", native_unit_of_measurement=PERCENTAGE,
                         state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="gpu_temperature", translation_key="gpu_temperature", device_class=SensorDeviceClass.TEMPERATURE,
                         native_unit_of_measurement=UnitOfTemperature.CELSIUS, state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="vram_used", translation_key="vram_used", device_class=SensorDeviceClass.DATA_SIZE,
                         native_unit_of_measurement=UnitOfInformation.BYTES, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES,
                         state_class=MEASURE, suggested_display_precision=1),
    LlmSensorDescription(key="vram_usage", translation_key="vram_usage", native_unit_of_measurement=PERCENTAGE,
                         state_class=MEASURE, suggested_display_precision=0),
    LlmSensorDescription(key="ram_used", translation_key="ram_used", device_class=SensorDeviceClass.DATA_SIZE,
                         native_unit_of_measurement=UnitOfInformation.BYTES, suggested_unit_of_measurement=UnitOfInformation.GIBIBYTES,
                         state_class=MEASURE, suggested_display_precision=1, entity_category=EntityCategory.DIAGNOSTIC),
    LlmSensorDescription(key="cpu_usage", translation_key="cpu_usage", native_unit_of_measurement=PERCENTAGE,
                         state_class=MEASURE, suggested_display_precision=0, entity_category=EntityCategory.DIAGNOSTIC),
    LlmSensorDescription(key="backend", translation_key="backend", entity_category=EntityCategory.DIAGNOSTIC),
    LlmSensorDescription(key="model", translation_key="model", entity_category=EntityCategory.DIAGNOSTIC),
)


async def async_setup_entry(hass: HomeAssistant, entry: LlmServerConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    coordinator = entry.runtime_data
    entities: list[SensorEntity] = [LlmSensor(coordinator, d) for d in SENSORS]
    entities.append(GpuEnergySensor(coordinator))
    entities += [LifetimeSensor(coordinator, *spec) for spec in LIFETIME]
    async_add_entities(entities)


class LlmEntity(CoordinatorEntity[LlmServerCoordinator]):
    _attr_has_entity_name = True

    def __init__(self, coordinator: LlmServerCoordinator, key: str) -> None:
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self._attr_unique_id = f"{entry.unique_id}_{key}"
        data = coordinator.data or {}
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.unique_id)}, name=entry.title, configuration_url=coordinator.base,
            manufacturer=BACKEND_NAMES.get(data.get("backend"), data.get("backend")), model=data.get("model"), sw_version=data.get("version"))


class LlmSensor(LlmEntity, SensorEntity):
    entity_description: LlmSensorDescription

    def __init__(self, coordinator: LlmServerCoordinator, description: LlmSensorDescription) -> None:
        super().__init__(coordinator, description.key)
        self.entity_description = description

    @property
    def native_value(self):
        value = (self.coordinator.data or {}).get(self.entity_description.key)
        if self.entity_description.key == "backend" and value:
            return BACKEND_NAMES.get(value, value)
        return value

    @property
    def available(self) -> bool:
        # Fields the current engine does not report (e.g. GPU power on SGLang) show as unavailable, not as 0
        return super().available and (self.coordinator.data or {}).get(self.entity_description.key) is not None


def new_samples(prev: list[float] | None, cur: list[float] | None, dt: float) -> list[float] | None:
    """The 1 Hz samples in `cur` that were not in `prev`, or None if the histories cannot be lined up.

    The histories carry no timestamps. The shift is the one near `dt` seconds under which the overlap matches; with
    a flat signal several match and the closest to `dt` wins, and if none does `dt` itself is used.
    """
    if not prev or not cur or len(prev) != len(cur) or dt > len(cur) + 1:
        return None
    n, want = len(cur), round(dt)
    shifts = [k for k in range(max(0, want - 2), min(n, want + 2) + 1) if prev[k:] == cur[:n - k]]
    k = min(shifts, key=lambda k: abs(k - dt)) if shifts else min(want, n)
    return cur[n - k:] if k else []


class GpuEnergySensor(LlmEntity, RestoreSensor):
    """GPU energy in kWh, integrated from the power readings and kept across restarts.

    Basalt keeps the last minute of power at about 1 Hz, so each scrape adds the mean of the samples taken since the
    previous one (found by lining up the two histories) over the time elapsed. Engines that only give the current power are integrated with the
    trapezoidal rule between scrapes. Two scrapes further apart than MAX_GAP (server down, engine without power data,
    HA stopped) are not bridged: the energy used in between is unknown, so it is left out rather than guessed.
    """

    _attr_translation_key = "gpu_energy"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 3

    def __init__(self, coordinator: LlmServerCoordinator) -> None:
        super().__init__(coordinator, "gpu_energy")
        self._energy = 0.0
        self._last: tuple[float, float] | None = None  # (time, watts) of the previous reading
        self._history: list[float] | None = None  # 1 Hz power history of the previous reading
        self._max_gap = max(60.0, 3 * coordinator.update_interval.total_seconds())

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_sensor_data()
        if last is not None and last.native_value is not None:
            try:
                self._energy = float(last.native_value)
            except (TypeError, ValueError):
                pass
        self._accumulate()

    def _accumulate(self) -> None:
        data = self.coordinator.data or {}
        now, watts, history = data.get("time"), data.get("gpu_power"), data.get("gpu_power_history")
        if now is None or watts is None:
            self._last = self._history = None
            return
        if self._last is not None:
            t0, w0 = self._last
            if now <= t0:
                return  # same reading delivered again (e.g. after a failed update)
            if now - t0 <= self._max_gap:
                new = new_samples(self._history, history, now - t0)
                if new:  # mean of the samples over the real elapsed time: they come a bit slower than 1 Hz
                    self._energy += sum(new) / len(new) * (now - t0) / 3_600_000  # W·s -> kWh
                else:
                    self._energy += (w0 + watts) / 2 * (now - t0) / 3_600_000
        self._last, self._history = (now, watts), history

    @callback
    def _handle_coordinator_update(self) -> None:
        if self.coordinator.last_update_success:
            self._accumulate()
        super()._handle_coordinator_update()

    @property
    def native_value(self) -> float:
        return round(self._energy, 6)

    @property
    def available(self) -> bool:
        return True  # the counter is valid even while the server is unreachable


# (key of the server counter, unit, enabled by default)
LIFETIME = (
    ("requests_total", None, True),
    ("output_tokens_total", TOKENS, True),
    ("prompt_tokens_total", TOKENS, True),
    ("cached_tokens_total", TOKENS, True),
)


class LifetimeSensor(LlmEntity, RestoreSensor):
    """A server counter summed across server restarts and kept across HA restarts.

    Each scrape adds how far the server counter moved since the previous one. A restart is spotted when the server's
    start time (`counters_since`) changes, the engine changes, or the counter goes back; the new run's count is then
    added from 0. Tokens counted between the last scrape and a crash are lost.
    """

    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: LlmServerCoordinator, source: str, unit: str | None, enabled: bool) -> None:
        super().__init__(coordinator, f"{source}_lifetime")
        self._source = source
        self._attr_translation_key = f"{source}_lifetime"
        self._attr_native_unit_of_measurement = unit
        self._attr_entity_registry_enabled_default = enabled
        self._total = 0.0
        self._seed = True  # brand new sensor: start from what the running server has counted so far
        self._last: float | None = None  # server counter at the previous scrape
        self._since = self._backend = None  # server start time and engine of that reading

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        data = await self.async_get_last_sensor_data()
        if data is not None and data.native_value is not None:
            try:
                self._total, self._seed = float(data.native_value), False
            except (TypeError, ValueError):
                pass
            last = await self.async_get_last_state()
            attrs = last.attributes if last is not None else {}
            try:
                self._last = float(attrs["server_count"])
                self._since, self._backend = attrs.get("server_since"), attrs.get("server_backend")
            except (TypeError, ValueError, KeyError):
                self._last = None  # no reference: the next reading only becomes one
        self._accumulate()

    def _accumulate(self) -> None:
        data = self.coordinator.data or {}
        value, since, backend = data.get(self._source), data.get("counters_since"), data.get("backend")
        if value is None:
            return
        if self._seed:
            self._total, self._seed = value, False
        elif self._last is not None:
            restarted = (value < self._last or backend != self._backend
                         or (since is not None and self._since is not None and since != self._since))
            self._total += value if restarted else value - self._last
        self._last, self._since, self._backend = value, since, backend

    @callback
    def _handle_coordinator_update(self) -> None:
        if self.coordinator.last_update_success:
            self._accumulate()
        super()._handle_coordinator_update()

    @property
    def native_value(self) -> float:
        return self._total

    @property
    def extra_state_attributes(self) -> dict:
        return {"server_count": self._last, "server_since": self._since, "server_backend": self._backend}

    @property
    def available(self) -> bool:
        return True  # the total is valid even while the server is unreachable
