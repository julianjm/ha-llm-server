"""Constants for the LLM Server integration."""
DOMAIN = "llm_server"
CONF_SCAN_INTERVAL = "scan_interval"
DEFAULT_SCAN_INTERVAL = 10  # seconds
MIN_SCAN_INTERVAL, MAX_SCAN_INTERVAL = 2, 300
BACKEND_NAMES = {"basalt": "Basalt", "sglang": "SGLang", "vllm": "vLLM"}
