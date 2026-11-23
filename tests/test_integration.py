"""Config flow, setup and sensors against a mocked Basalt server."""
from datetime import timedelta
from unittest.mock import patch

from homeassistant import config_entries
from homeassistant.const import CONF_URL
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import device_registry as dr
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_fire_time_changed

from custom_components.llm_server.const import DOMAIN

from .conftest import URL, load


async def test_config_flow(hass, basalt):
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
    assert result["type"] is FlowResultType.FORM
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {CONF_URL: "llm.test:3030/metrics/"})
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_URL: URL}
    assert result["result"].unique_id == "llm.test:3030"


async def test_config_flow_unknown_backend(hass, aioclient_mock):
    aioclient_mock.get(URL + "/metrics", text="something else")
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {CONF_URL: URL})
    assert result["errors"] == {"base": "unknown_backend"}


async def setup(hass):
    entry = MockConfigEntry(domain=DOMAIN, unique_id="llm.test:3030", data={CONF_URL: URL}, title="LLM server llm.test:3030")
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


async def test_sensors(hass, basalt):
    entry = await setup(hass)
    state = lambda e: hass.states.get(f"sensor.llm_server_llm_test_3030_{e}")
    assert float(state("decode_speed").state) == 250
    assert float(state("gpu_power").state) == 250
    assert state("gpu_power").attributes["device_class"] == "power"
    assert state("gpu_temperature").attributes["unit_of_measurement"] == "°C"
    assert state("vram_used").attributes["unit_of_measurement"] == "GiB"
    assert state("state").state == "decode"
    assert state("engine").state == "Basalt"
    energy = state("gpu_energy")
    assert energy.attributes["device_class"] == "energy"
    assert energy.attributes["state_class"] == "total_increasing"
    assert energy.attributes["unit_of_measurement"] == "kWh"
    device = dr.async_get(hass).async_get_device_by_identifier((DOMAIN, "llm.test:3030"), entry.entry_id)
    assert (device.manufacturer, device.model) == ("Basalt", "test-model")


async def test_energy_integrates_power(hass, basalt):
    metrics = load("basalt_metrics.json")
    metrics["history"]["gpus"][0]["power"] = [250.0] * 60  # flat 250 W for the last minute
    basalt.clear_requests()
    basalt.get(URL + "/metrics", json=metrics)
    basalt.get(URL + "/slots", json=load("basalt_slots.json"))
    await setup(hass)
    eid = "sensor.llm_server_llm_test_3030_gpu_energy"
    assert float(hass.states.get(eid).state) == 0
    t0 = dt_util.utcnow()
    clock = [1000.0]
    with patch("custom_components.llm_server.coordinator.time.time", side_effect=lambda: clock[0]):
        # reset the reference reading to the patched clock, then 36 s at 250 W = 2.5 Wh
        await hass.config_entries.async_reload(hass.config_entries.async_entries(DOMAIN)[0].entry_id)
        await hass.async_block_till_done()
        clock[0] += 36
        async_fire_time_changed(hass, t0 + timedelta(seconds=11))
        await hass.async_block_till_done()
    assert abs(float(hass.states.get(eid).state) - 0.0025) < 1e-6


async def test_unavailable_server(hass, basalt):
    await setup(hass)
    basalt.clear_requests()
    basalt.get(URL + "/metrics", status=500)
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=11))
    await hass.async_block_till_done()
    assert hass.states.get("sensor.llm_server_llm_test_3030_gpu_power").state == "unavailable"
    assert hass.states.get("sensor.llm_server_llm_test_3030_gpu_energy").state != "unavailable"


async def test_lifetime_survives_server_restart(hass, basalt):
    await setup(hass)
    eid = "sensor.llm_server_llm_test_3030_output_tokens_lifetime"
    assert float(hass.states.get(eid).state) == 2189123  # seeded from the running server

    def serve(since, written):
        metrics = load("basalt_metrics.json")
        metrics["totals"].update(since=since, written=written)
        basalt.clear_requests()
        basalt.get(URL + "/metrics", json=metrics)
        basalt.get(URL + "/slots", json=load("basalt_slots.json"))

    now = dt_util.utcnow()
    for i, (since, written) in enumerate(((1795133880, 2189223), (1795200000, 50), (1795200000, 80)), 1):
        serve(since, written)  # +100, then a restart (new since) counting 50, then +30
        async_fire_time_changed(hass, now + timedelta(seconds=11 * i))
        await hass.async_block_till_done()
    assert float(hass.states.get(eid).state) == 2189123 + 100 + 50 + 30
    assert float(hass.states.get("sensor.llm_server_llm_test_3030_output_tokens").state) == 80


async def test_lifetime_restored_after_ha_restart(hass, basalt):
    from homeassistant.core import State
    from pytest_homeassistant_custom_component.common import mock_restore_cache_with_extra_data

    eid = "sensor.llm_server_llm_test_3030_output_tokens_lifetime"
    attrs = {"server_count": 5000, "server_since": 1795000000, "server_backend": "basalt"}
    stored = {"native_value": 9000, "native_unit_of_measurement": "tokens"}
    mock_restore_cache_with_extra_data(hass, ((State(eid, "9000", attrs), stored),))
    await setup(hass)
    # The server restarted while HA was down (new since): its whole count is new
    assert float(hass.states.get(eid).state) == 9000 + 2189123


def test_new_samples_lines_up_histories():
    from custom_components.llm_server.sensor import new_samples

    prev = [float(i) for i in range(60)]
    cur = prev[9:] + [100.0 + i for i in range(9)]  # 9 new samples although the scrape was 10 s apart
    assert new_samples(prev, cur, 10.0) == [100.0 + i for i in range(9)]
    assert new_samples(None, cur, 10.0) is None
    assert new_samples(prev, cur, 90.0) is None  # longer than the history: fall back to the trapezoid
    assert new_samples([5.0] * 60, [5.0] * 60, 10.0) == [5.0] * 10  # flat signal: the shift nearest dt
