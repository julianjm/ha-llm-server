# LLM Server for Home Assistant

Monitor a local LLM inference server — Basalt, SGLang or vLLM — from Home Assistant. The engine is detected
from what its `/metrics` endpoint returns, so switching engines on the same port needs no reconfiguration.

Nothing runs on the server side: Home Assistant polls `/metrics` (and `/slots` on Basalt) every few seconds.

## Sensors

| Sensor | Unit | Engines |
|---|---|---|
| State | idle / decode / prefill… | all |
| Decode speed, Prefill speed | tok/s | all |
| Active / queued requests | | all |
| Prefix cache hit (last interval, lifetime) | % | all |
| Time to first token (mean of requests finished in the last interval) | s | all |
| KV cache usage | % | all |
| Requests, output tokens, prompt tokens\*, cached prompt tokens\* | total | all |
| GPU power, utilization, temperature, VRAM used / usage | W, %, °C, GiB | Basalt |
| **GPU energy** | kWh | Basalt |
| Host RAM used, host CPU usage, engine, model (diagnostic) | | Basalt (engine/model: all) |

\* disabled by default. Sensors an engine does not report show as *unavailable* rather than 0.
The device's manufacturer, model and version follow the engine currently running.

## Energy dashboard

**GPU energy** is a `total_increasing` kWh sensor, integrated from the GPU power readings (trapezoidal rule) and kept
across Home Assistant restarts. Add it under *Settings → Dashboards → Energy → Individual devices*, with
**GPU power** as its power sensor. If the server is plugged into a metering smart plug, set that plug as the
GPU's upstream device so it is not counted twice.

Gaps longer than 60 s (or 3 scan intervals) — server down, an engine without power data, Home Assistant stopped —
are not bridged: that energy is left out rather than guessed. A shorter scan interval gives a more accurate total.

## Installation

**HACS**: add this repository as a custom repository (type *Integration*), install *LLM Server*, restart Home Assistant.

**Manual**: copy `custom_components/llm_server` into your Home Assistant `config/custom_components/` and restart.

Then *Settings → Devices & services → Add integration → LLM Server* and enter the server's base URL
(e.g. `http://192.168.1.10:3030`). The scan interval (default 10 s) is in the integration's options.

Moving from a manual install to HACS: delete the copied folder and install from HACS without removing the
integration entry; entities and their history are kept.

## Development

```
pip install pytest-homeassistant-custom-component
pytest
```
