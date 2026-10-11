"""
Deterministic incident analysis (D7.3): parse logs, cluster error signatures, build a timeline,
and correlate the incident with recent deploys - BEFORE (and independently of) any model call.

Why deterministic first? It is cheap, reproducible, and it becomes the ground truth we use to
check the triage agent's claims. An LLM should accelerate the investigation, not be the only witness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

LINE_RE = re.compile(r"^(\S+)\s+(DEBUG|INFO|WARN|ERROR)\s+(\S+)\s+(.*)$")
KV_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')
# Per-occurrence values that would split one problem into many clusters.
VOLATILE = {
    "request_id", "latency_ms", "retry_after", "attempt", "delay_ms", "input_tokens", "output_tokens",
    "cache_read_input_tokens", "cache_creation_input_tokens", "error_rate", "p95_ms", "threshold", "threshold_ms",
}


def epoch(ts: str) -> int:
    return int(datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


@dataclass
class LogRecord:
    ts: str
    epoch: int
    time: str          # HH:MM:SS
    level: str
    component: str
    event: str
    fields: dict[str, str]
    raw: str


def parse_log(text: str) -> tuple[list[LogRecord], int]:
    records: list[LogRecord] = []
    unparsed = 0
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            unparsed += 1
            continue
        ts, level, component, rest = m.groups()
        first_kv = KV_RE.search(rest)
        event = rest[: first_kv.start()].strip() if first_kv else rest.strip()
        fields = {k: v.strip('"') for k, v in KV_RE.findall(rest)}
        records.append(LogRecord(ts, epoch(ts), ts[11:19], level, component, event, fields, line))
    return records, unparsed


def signature(rec: LogRecord) -> str:
    stable = [f"{k}={v}" for k, v in rec.fields.items() if k not in VOLATILE]
    return " ".join([rec.component, rec.event, *stable])


@dataclass
class Cluster:
    signature: str
    level: str
    count: int = 0
    first: str = ""
    last: str = ""


def cluster_errors(records: list[LogRecord]) -> list[Cluster]:
    clusters: dict[str, Cluster] = {}
    for r in records:
        if r.level not in ("WARN", "ERROR"):
            continue
        sig = signature(r)
        c = clusters.setdefault(sig, Cluster(sig, r.level, first=r.time))
        c.count += 1
        c.last = r.time
    return sorted(clusters.values(), key=lambda c: (-c.count, c.first))


def bar(value: float, max_value: float, width: int) -> str:
    n = int(value / max_value * width + 0.5) if max_value > 0 else 0
    return "#" * n + " " * (width - n)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class Facts:
    deploy_version: str = ""
    deploy_finished: str = ""
    first_429: str = ""
    first_529: str = ""
    seconds_deploy_to_first_429: int = 0
    retries_ignoring_retry_after: int = 0
    max_retry_after_s: int = 0
    cache_ratio_before: float = 0.0
    cache_ratio_after: float = 0.0
    uncached_input_per_call_before: float = 0.0
    uncached_input_per_call_after: float = 0.0
    error_rate_before: float = 0.0
    error_rate_after: float = 0.0
    p95_before_ms: float = 0.0
    p95_after_ms: float = 0.0
    itpm_peak: int = 0
    itpm_limit: int = 0
    in_flight_before: float = 0.0
    in_flight_after: float = 0.0


def correlate(records: list[LogRecord], metrics: dict[str, Any], deploys: list[dict[str, Any]]) -> Facts:
    f = Facts()
    errors_429 = [r for r in records if r.fields.get("status") == "429" and r.component == "claude-client"]
    errors_529 = [r for r in records if r.fields.get("status") == "529" and r.component == "claude-client"]
    first_err_epoch = min([r.epoch for r in errors_429 + errors_529], default=0)
    # Suspect deploy = the most recent deploy that finished before the first error.
    candidates = [d for d in deploys if epoch(d["finished"]) <= first_err_epoch]
    suspect = max(candidates, key=lambda d: epoch(d["finished"])) if candidates else None
    if suspect:
        f.deploy_version, f.deploy_finished = suspect["version"], suspect["finished"][11:19]
        if errors_429:
            f.seconds_deploy_to_first_429 = errors_429[0].epoch - epoch(suspect["finished"])
    f.first_429 = errors_429[0].time if errors_429 else ""
    f.first_529 = errors_529[0].time if errors_529 else ""

    # Retry storm: a retry scheduled sooner than the retry-after the API just asked for.
    last_retry_after = 0
    for r in records:
        if r.component != "claude-client":
            continue
        if r.fields.get("status") == "429":
            last_retry_after = int(r.fields.get("retry_after", "0"))
            f.max_retry_after_s = max(f.max_retry_after_s, last_retry_after)
        elif r.event == "retrying" and r.fields.get("reason") == "rate_limit_error":
            if int(r.fields.get("delay_ms", "0")) < last_retry_after * 1000:
                f.retries_ignoring_retry_after += 1

    cut = epoch(suspect["finished"]) if suspect else 0
    responses = [r for r in records if r.component == "claude-client" and r.event == "response"]

    def ratio(rs: list[LogRecord]) -> float:
        read = sum(int(r.fields.get("cache_read_input_tokens", "0")) for r in rs)
        uncached = sum(int(r.fields.get("input_tokens", "0")) for r in rs)
        return read / (read + uncached) if read + uncached else 0.0

    before = [r for r in responses if r.epoch < cut]
    after = [r for r in responses if r.epoch >= cut]
    f.cache_ratio_before, f.cache_ratio_after = ratio(before), ratio(after)
    f.uncached_input_per_call_before = mean([float(r.fields["input_tokens"]) for r in before])
    f.uncached_input_per_call_after = mean([float(r.fields["input_tokens"]) for r in after])

    pts = metrics["points"]
    pre = [p for p in pts if p["minute"] < "13:59"]
    post = [p for p in pts if p["minute"] >= "14:00"]
    f.error_rate_before = mean([p["error_rate"] for p in pre])
    f.error_rate_after = mean([p["error_rate"] for p in post])
    f.p95_before_ms = mean([p["p95_latency_ms"] for p in pre])
    f.p95_after_ms = mean([p["p95_latency_ms"] for p in post])
    f.itpm_peak = max(p["input_tokens_per_min"] for p in pts)
    f.itpm_limit = metrics["limits"]["claude_opus_input_tokens_per_min"]
    f.in_flight_before = mean([p["in_flight"] for p in pre])
    f.in_flight_after = mean([p["in_flight"] for p in post])
    return f
