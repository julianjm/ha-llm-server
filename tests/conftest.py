"""Shared fixtures."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
URL = "http://llm.test:3030"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    yield


def load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def basalt(aioclient_mock):
    aioclient_mock.get(URL + "/metrics", json=load("basalt_metrics.json"))
    aioclient_mock.get(URL + "/slots", json=load("basalt_slots.json"))
    return aioclient_mock
