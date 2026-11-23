"""Backend detection and metric parsing for Basalt, SGLang and vLLM.

Pure Python (no Home Assistant imports) so it can be tested on its own. A Tracker turns successive scrapes into
sensor values: rates and per-interval figures need the previous scrape.
"""
from __future__ import annotations

import json
import re

PROM_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)")
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
SPLIT_LABELS = ("mode", "category", "pool")  # series kept apart as "name|value" besides the sum

BACKENDS = ("basalt", "sglang", "vllm")


def detect(raw: str) -> str | None:
    """Name the server from the shape of its /metrics."""
    text = raw.lstrip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return None
        return "basalt" if isinstance(data, dict) and "engine" in data and "live" in data else None
    if "sglang:" in raw:
        return "sglang"
    if "vllm:" in raw:
        return "vllm"
    return None


def num(v) -> float | None:
    """A number from a JSON field that may be missing, null or a string."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def parse_prom(raw: str, prefix: str) -> tuple[dict[str, float], dict[str, str]]:
    """Sum every series of each metric (dropping the prefix); histograms keep only _sum and _count.

    Returns (values, labels) where labels holds model_name and the like, taken from the first series that has them.
    """
    values: dict[str, float] = {}
    labels: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.startswith(prefix):
            continue
        m = PROM_RE.match(line)
        if not m:
            continue
        name = m.group(1)[len(prefix):]
        if name.endswith("_bucket"):
            continue
        v = num(m.group(4))
        if v is None:
            continue
        values[name] = values.get(name, 0.0) + v
        for key, lv in LABEL_RE.findall(m.group(3) or ""):
            if key in SPLIT_LABELS:
                split = f"{name}|{lv}"
                values[split] = values.get(split, 0.0) + v
            elif key in ("model_name", "version") and key not in labels:
                labels[key] = lv
    return values, labels


def process_start(raw: str) -> float | None:
    """process_start_time_seconds from the Prometheus process collector, which marks a server restart."""
    m = re.search(r"^process_start_time_seconds(?:\{[^}]*\})?\s+(\S+)", raw, re.M)
    return num(m.group(1)) if m else None


def ratio(a, b, scale=100.0):
    return scale * a / b if a is not None and b else None


class Tracker:
    """Keeps the previous scrape to turn counters into rates and per-interval averages."""

    def __init__(self) -> None:
        self.prev_time: float | None = None
        self.prev_backend: str | None = None
        self.prev: dict[str, float] = {}
        self.sticky: dict[str, float] = {}  # per-interval stats keep their last value while nothing happens
        self.read_done: dict = {}
        self.read_total = 0.0
        self.read_time, self.read_until = 0.0, 0.0  # seconds spent reading prompts; end of the time already counted

    # ---- helpers on the counters of this scrape vs the previous one
    def _delta(self, cur: dict, key: str) -> float | None:
        if key not in cur or key not in self.prev or cur[key] is None or self.prev[key] is None:
            return None
        d = cur[key] - self.prev[key]
        return d if d >= 0 else None  # a counter going back means the server restarted

    def _rate(self, cur: dict, key: str, dt: float | None) -> float | None:
        d = self._delta(cur, key)
        return d / dt if d is not None and dt else None

    def _read_speed(self, cur: dict) -> float | None:
        read, busy = self._delta(cur, "read"), self._delta(cur, "read_time")
        if read is None or busy is None:
            return None
        return read / busy if busy > 0 else 0.0

    def _per_interval(self, name: str, num_d: float | None, den_d: float | None, scale=1.0) -> float | None:
        if num_d is not None and den_d:
            self.sticky[name] = scale * num_d / den_d
        return self.sticky.get(name)

    def update(self, backend: str, metrics, slots, now: float) -> dict:
        if backend != self.prev_backend:  # switched engine: nothing to diff against
            self.prev, self.prev_time, self.sticky, self.read_done, self.read_until = {}, None, {}, {}, 0.0
        dt = now - self.prev_time if self.prev_time is not None and now > self.prev_time else None
        if backend == "basalt":
            values, counters = self._basalt(metrics, slots, now, dt)
        elif backend == "sglang":
            values, counters = self._sglang(metrics, dt)
        else:
            values, counters = self._vllm(metrics, dt)
        values["backend"] = backend
        values["time"] = now
        self.prev, self.prev_time, self.prev_backend = counters, now, backend
        return values

    # ---- Basalt: JSON /metrics plus /slots
    def _basalt(self, m: dict, slots, now: float, dt):
        eng, live = m.get("engine") or {}, m.get("live") or {}
        totals = {k: num(v) for k, v in (m.get("totals") or {}).items()}
        hw = m.get("hardware") or {}
        gpus = [g for g in (hw.get("gpus") or []) if isinstance(g, dict)]
        si = live.get("slots") or {}
        slots = slots if isinstance(slots, list) else None

        # Live prefill: tokens newly read by busy slots. /slots never fills `reused` while a request runs, and
        # prefill_done jumps straight to the cached length when a parked conversation is restored, so an advance
        # faster than any real read (1.5x the server's best prefill, or 30k tok/s if unknown) is taken as cache and
        # not counted. The speed is measured from the request's previous advance plus one scrape of slack, since a read
        # lands a chunk at a time (up to 32k tokens). Keyed by prompt size so a new request in the same slot is not
        # diffed against the previous one. The prefill speed divides by the time spent reading (chunk to chunk), not
        # the whole scrape interval, or idle time around a prompt would dilute it.
        if slots is not None:
            best = num(((m.get("best") or {}).get("prefill") or {}).get("tok_s"))
            cap, step_dt = (1.5 * best if best else 30000.0), dt or 10.0
            done, start = {}, None
            for sl in slots:
                if isinstance(sl, dict) and sl.get("is_processing") and num(sl.get("prefill_done")) is not None:
                    key = (sl.get("id"), sl.get("prompt_tokens"))
                    cur = num(sl.get("prefill_done"))
                    # A request first seen at 0 starts now; one already advanced started at most a scrape ago
                    last, since = self.read_done.get(key, (0.0, now - step_dt if cur > 0 else now))
                    step = cur - last
                    if 0 < step <= cap * (now - since + step_dt):
                        self.read_total += step
                        start = since if start is None else min(start, since)
                    done[key] = (cur, now if step > 0 else since)
            self.read_done = done
            if start is not None:
                self.read_time += now - max(start, self.read_until)
                self.read_until = now
        counters = {"requests": totals.get("requests"), "prompt": totals.get("prompt"), "reused": totals.get("reused"),
                    "written": totals.get("written"), "read": self.read_total if slots is not None else None,
                    "read_time": self.read_time if slots is not None else None}

        # TTFT: mean over the requests that finished since the previous scrape
        since = self.prev.get("last_request_time")
        reqs = [r for r in (m.get("requests") or []) if isinstance(r, dict) and num(r.get("time")) is not None]
        newest = max((num(r.get("time")) for r in reqs), default=since)
        counters["last_request_time"] = newest
        if since is not None:
            ttfts = [num(r.get("ttft_s")) for r in reqs if num(r.get("time")) > since and num(r.get("ttft_s")) is not None]
            if ttfts:
                self.sticky["ttft"] = sum(ttfts) / len(ttfts)
        elif reqs and num(reqs[0].get("ttft_s")) is not None:
            self.sticky.setdefault("ttft", num(reqs[0].get("ttft_s")))

        kv_used = sum(num(sl.get("ctx_used")) or 0 for sl in slots or [] if isinstance(sl, dict))
        kv_cap = num(eng.get("elastic_kv_ceiling")) or sum(num(sl.get("n_ctx")) or 0 for sl in slots or [] if isinstance(sl, dict))
        powers = [num(g.get("power")) for g in gpus if num(g.get("power")) is not None]
        # Last minute of power at 1 Hz per GPU (oldest first), summed across GPUs; lets the energy counter integrate
        # every second instead of one reading per scrape
        hist = [[num(w) for w in g.get("power") or []] for g in ((m.get("history") or {}).get("gpus") or []) if isinstance(g, dict)]
        hist = [h for h in hist if h and None not in h]
        power_hist = [sum(ws) for ws in zip(*hist)] if hist and len({len(h) for h in hist}) == 1 else None
        utils = [num(g.get("util")) for g in gpus if num(g.get("util")) is not None]
        temps = [num(g.get("temp")) for g in gpus if num(g.get("temp")) is not None]
        mem_used = [num(g.get("mem_used")) for g in gpus if num(g.get("mem_used")) is not None]
        mem_total = [num(g.get("mem_total")) for g in gpus if num(g.get("mem_total")) is not None]
        state = str(live.get("state") or "unknown")
        values = {
            "counters_since": totals.get("since"),
            "model": eng.get("model"), "version": eng.get("version"), "state": state,
            "decode_speed": num(live.get("tok_s")) or 0.0,
            "prefill_speed": self._read_speed(counters),
            "active_requests": num(si.get("active")), "max_requests": num(si.get("total")), "queued_requests": num(live.get("queued")),
            "requests_total": totals.get("requests"), "prompt_tokens_total": totals.get("prompt"),
            "cached_tokens_total": totals.get("reused"), "output_tokens_total": totals.get("written"),
            "prefix_hit": self._per_interval("hit", self._delta(counters, "reused"), self._delta(counters, "prompt"), 100.0),
            "prefix_hit_lifetime": ratio(totals.get("reused"), totals.get("prompt")),
            "ttft": self.sticky.get("ttft"),
            "kv_usage": ratio(kv_used, kv_cap) if slots is not None else None,
            "gpu_power": sum(powers) if powers else None, "gpu_power_history": power_hist,
            "gpu_utilization": sum(utils) / len(utils) if utils else None,
            "gpu_temperature": max(temps) if temps else None,
            "vram_used": sum(mem_used) if mem_used else None,
            "vram_usage": ratio(sum(mem_used), sum(mem_total)) if mem_used and mem_total else None,
            "ram_used": num(hw.get("ram_used")), "cpu_usage": num(hw.get("cpu")),
        }
        return values, counters

    # ---- SGLang: Prometheus text
    def _sglang(self, raw: str, dt):
        s, labels = parse_prom(raw, "sglang:")
        counters = {k: s.get(k) for k in ("num_requests_total", "prompt_tokens_total", "generation_tokens_total", "cached_tokens_total",
                                          "realtime_tokens_total|decode", "realtime_tokens_total|prefill_compute",
                                          "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count")}
        decode = self._rate(counters, "realtime_tokens_total|decode", dt)
        if decode is None:
            decode = s.get("gen_throughput")
        prefill = self._rate(counters, "realtime_tokens_total|prefill_compute", dt)
        if prefill is None:
            p, c = self._rate(counters, "prompt_tokens_total", dt), self._rate(counters, "cached_tokens_total", dt)
            prefill = max(0.0, p - (c or 0)) if p is not None else None
        running, queued = s.get("num_running_reqs"), s.get("num_queue_reqs")
        kv = 100 * s["token_usage"] if "token_usage" in s else ratio(s.get("kv_used_tokens"), s.get("max_total_num_tokens"))
        values = {
            "counters_since": process_start(raw),
            "model": labels.get("model_name"), "version": None,
            "state": "busy" if running else "queued" if queued else "idle",
            "decode_speed": decode, "prefill_speed": prefill,
            "active_requests": running, "max_requests": s.get("max_running_requests"), "queued_requests": queued,
            "requests_total": s.get("num_requests_total"), "prompt_tokens_total": s.get("prompt_tokens_total"),
            "cached_tokens_total": s.get("cached_tokens_total"), "output_tokens_total": s.get("generation_tokens_total"),
            "prefix_hit": self._per_interval("hit", self._delta(counters, "cached_tokens_total"), self._delta(counters, "prompt_tokens_total"), 100.0),
            "prefix_hit_lifetime": ratio(s.get("cached_tokens_total"), s.get("prompt_tokens_total")),
            "ttft": self._per_interval("ttft", self._delta(counters, "time_to_first_token_seconds_sum"),
                                       self._delta(counters, "time_to_first_token_seconds_count")),
            "kv_usage": kv,
        }
        return values, counters

    # ---- vLLM: Prometheus text
    def _vllm(self, raw: str, dt):
        s, labels = parse_prom(raw, "vllm:")
        counters = {k: s.get(k) for k in ("request_success_total", "prompt_tokens_total", "generation_tokens_total",
                                          "prefix_cache_hits_total", "prefix_cache_queries_total",
                                          "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count")}
        prompt, hits = self._rate(counters, "prompt_tokens_total", dt), self._rate(counters, "prefix_cache_hits_total", dt)
        running, queued = s.get("num_requests_running"), s.get("num_requests_waiting")
        kv = s.get("kv_cache_usage_perc", s.get("gpu_cache_usage_perc"))
        values = {
            "counters_since": process_start(raw),
            "model": labels.get("model_name"), "version": None,
            "state": "busy" if running else "queued" if queued else "idle",
            "decode_speed": self._rate(counters, "generation_tokens_total", dt),
            "prefill_speed": max(0.0, prompt - (hits or 0)) if prompt is not None else None,
            "active_requests": running, "max_requests": None, "queued_requests": queued,
            "requests_total": s.get("request_success_total"), "prompt_tokens_total": s.get("prompt_tokens_total"),
            "cached_tokens_total": s.get("prefix_cache_hits_total"), "output_tokens_total": s.get("generation_tokens_total"),
            "prefix_hit": self._per_interval("hit", self._delta(counters, "prefix_cache_hits_total"),
                                             self._delta(counters, "prefix_cache_queries_total"), 100.0),
            "prefix_hit_lifetime": ratio(s.get("prefix_cache_hits_total"), s.get("prefix_cache_queries_total")),
            "ttft": self._per_interval("ttft", self._delta(counters, "time_to_first_token_seconds_sum"),
                                       self._delta(counters, "time_to_first_token_seconds_count")),
            "kv_usage": 100 * kv if kv is not None else None,
        }
        return values, counters
