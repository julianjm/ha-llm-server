"""Polls the server's /metrics (and /slots on Basalt) and turns it into sensor values."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import timedelta
from urllib.parse import urlsplit

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .backends import Tracker, detect
from .const import CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL, DOMAIN

_LOGGER = logging.getLogger(__name__)
TIMEOUT = 8


def normalize_url(url: str) -> str:
    """Base URL of the server: scheme added if missing, no trailing /metrics or slash."""
    url = url.strip()
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    if url.endswith("/metrics"):
        url = url[: -len("/metrics")]
    return url.rstrip("/")


def server_id(base: str) -> str:
    """Stable id (host:port) used for unique ids, so entities survive removing and re-adding the integration."""
    return urlsplit(base).netloc.lower()


async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as resp:
        resp.raise_for_status()
        return await resp.text()


async def probe(hass: HomeAssistant, base: str) -> str | None:
    """Backend behind `base`, or None if /metrics is not one we know. Raises on connection errors."""
    return detect(await fetch(async_get_clientsession(hass), base + "/metrics"))


type LlmServerConfigEntry = ConfigEntry[LlmServerCoordinator]


class LlmServerCoordinator(DataUpdateCoordinator[dict]):
    config_entry: LlmServerConfigEntry

    def __init__(self, hass: HomeAssistant, entry: LlmServerConfigEntry) -> None:
        interval = entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
        super().__init__(hass, _LOGGER, config_entry=entry, name=DOMAIN, update_interval=timedelta(seconds=interval))
        self.base = entry.data[CONF_URL]
        self.session = async_get_clientsession(hass)
        self.tracker = Tracker()
        self.last_raw: dict = {}  # for diagnostics

    async def _async_update_data(self) -> dict:
        try:
            raw = await fetch(self.session, self.base + "/metrics")
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise UpdateFailed(f"{self.base}/metrics: {err or type(err).__name__}") from err
        backend = detect(raw)
        if backend is None:
            raise UpdateFailed(f"{self.base}/metrics is not Basalt, SGLang or vLLM")
        metrics, slots = raw, None
        if backend == "basalt":
            metrics = json.loads(raw)
            try:  # only live prefill and KV usage depend on it
                slots = json.loads(await fetch(self.session, self.base + "/slots"))
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
                _LOGGER.debug("%s/slots: %s", self.base, err)
        self.last_raw = {"backend": backend, "metrics": metrics, "slots": slots}
        return self.tracker.update(backend, metrics, slots, time.time())
