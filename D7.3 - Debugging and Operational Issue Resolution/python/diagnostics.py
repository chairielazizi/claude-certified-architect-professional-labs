"""
Operational debugging checklist as code (D7.3).

* lint_request()     - catch the 400s before they happen (removed params on current models, prefill,
                       forced tool_choice, broken tool_use/tool_result pairing, bad model ids)
* diagnose_response() - interpret stop_reason (max_tokens truncation, refusal, unfinished tool loop)
* cache_check()      - flag cache_read_input_tokens == 0 on repeated calls (silent cache invalidators)
"""

from __future__ import annotations

import re
from typing import Any

from claude_client import MODEL_FABLE, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET  # main.py puts shared/python on sys.path

KNOWN_MODELS = {MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU}
NO_SAMPLING_PARAMS = {MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU}   # temperature/top_p/top_k -> 400
NO_FORCED_TOOL_CHOICE = {MODEL_FABLE, MODEL_OPUS, MODEL_SONNET}             # tool_choice any/tool -> 400
NO_THINKING_DISABLED = {MODEL_FABLE, MODEL_OPUS, MODEL_SONNET}              # thinking.type=disabled -> 400


def _blocks(content: Any) -> list[dict[str, Any]]:
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def lint_request(params: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return (code, problem, fix) tuples for request shapes the API would reject."""
    issues: list[tuple[str, str, str]] = []
    model = str(params.get("model", ""))
    messages: list[dict[str, Any]] = params.get("messages") or []

    if model not in KNOWN_MODELS:
        hint = re.sub(r"(\d)\.(\d)", r"\1-\2", model)
        issues.append(("R009", f'unknown model id "{model}" -> 404 not_found_error', f'use an exact id, e.g. "{hint}"'))
    for p in ("temperature", "top_p", "top_k"):
        if p in params and model in NO_SAMPLING_PARAMS:
            issues.append(("R001", f"{p} is not accepted on {model} -> 400", "delete it; steer with the prompt and output_config.effort"))
    thinking = params.get("thinking") or {}
    if thinking.get("type") == "enabled" or "budget_tokens" in thinking:
        issues.append(("R002", "thinking budget_tokens is removed on current models -> 400",
                       'use thinking={"type": "adaptive"} and output_config={"effort": ...}'))
    if thinking.get("type") == "disabled" and model in NO_THINKING_DISABLED:
        fix = 'Sonnet 5.5: use {"type": "between_tools"} at effort <= high' if model == MODEL_SONNET else "omit thinking and lower output_config.effort"
        issues.append(("R003", f"thinking disabled is not supported on {model} -> 400", fix))
    if (params.get("tool_choice") or {}).get("type") in ("any", "tool") and model in NO_FORCED_TOOL_CHOICE:
        issues.append(("R005", f"forced tool_choice is not supported on {model} -> 400",
                       "use auto + an instruction (strict: true tools) or structured outputs"))
    if not isinstance(params.get("max_tokens"), int) or params["max_tokens"] <= 0:
        issues.append(("R008", "max_tokens missing or not a positive integer -> 400", "set max_tokens explicitly"))
    if messages and messages[0].get("role") != "user":
        issues.append(("R006", "first message must have role user -> 400", "start the conversation with a user turn"))
    if messages and messages[-1].get("role") == "assistant":
        issues.append(("R004", "conversation ends with an assistant turn (prefill) -> 400 on current models",
                       "end with a user turn; use structured outputs to control the format"))

    # tool_use / tool_result pairing: every tool_use id must be answered in the NEXT user message,
    # tool_result blocks first, and every tool_result must reference a tool_use from the turn before.
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        ids = [b["id"] for b in _blocks(m.get("content")) if b.get("type") == "tool_use"]
        if not ids:
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        blocks = _blocks(nxt.get("content")) if nxt and nxt.get("role") == "user" else []
        results = [b.get("tool_use_id") for b in blocks if b.get("type") == "tool_result"]
        for tid in ids:
            if tid not in results:
                issues.append(("R007", f"tool_use {tid} has no matching tool_result in the next user message -> 400",
                               "send one tool_result per tool_use id, in the very next user turn"))
        for rid in results:
            if rid not in ids:
                issues.append(("R007", f"tool_result references unknown tool_use_id {rid} -> 400", "copy the id from the tool_use block"))
        first_result = next((k for k, b in enumerate(blocks) if b.get("type") == "tool_result"), None)
        if first_result is not None and any(b.get("type") != "tool_result" for b in blocks[:first_result]):
            issues.append(("R007", "text placed before tool_result blocks in the user message -> 400", "put tool_result blocks first, text after"))
    return issues


def diagnose_response(msg: Any) -> str:
    reason = getattr(msg, "stop_reason", None)
    if reason == "max_tokens":
        return "TRUNCATED: hit max_tokens - raise max_tokens or stream; never parse or act on partial output"
    if reason == "refusal":
        return "REFUSAL: inspect stop_details.category; rephrase or route - do not blind-retry"
    if reason == "tool_use":
        return "TOOL_USE: the loop is not finished - run the tools and send tool_result blocks"
    if reason == "pause_turn":
        return "PAUSE_TURN: server tool paused - resend the conversation to continue"
    if reason == "end_turn":
        has_text = any(getattr(b, "type", "") == "text" and getattr(b, "text", "") for b in msg.content)
        return "OK: end_turn" if has_text else "SUSPICIOUS: end_turn with no text"
    return f"UNEXPECTED stop_reason={reason}"


def cache_check(usages: list[Any]) -> str:
    reads = [getattr(u, "cache_read_input_tokens", 0) or 0 for u in usages]
    writes = [getattr(u, "cache_creation_input_tokens", 0) or 0 for u in usages]
    if len(usages) > 1 and all(r == 0 for r in reads):
        if all(w > 0 for w in writes):
            return ("FAIL: every call WRITES the cache and none READS it - the prefix changes per request "
                    "(timestamp/UUID in system prompt, unsorted JSON, tool list or model changes). You pay 1.25x for nothing.")
        return "FAIL: no cache activity - prefix below the minimum cacheable length or no cache_control breakpoint"
    return f"OK: cache reads on {sum(1 for r in reads if r > 0)}/{len(usages)} calls"


CHECKLIST: list[tuple[str, str, str]] = [
    ("Output cut off / invalid JSON", "stop_reason == max_tokens?", "raise max_tokens, stream, or split the task"),
    ("Agent loop stops early", "stop_reason tool_use / pause_turn handled?", "execute tools, append tool_result, continue"),
    ("400 tool_result errors", "ids paired, results first, next user turn?", "fix the history builder (append-only)"),
    ("400 after model upgrade", "temperature/top_p/top_k, budget_tokens, prefill, forced tool_choice?", "remove; use adaptive thinking + effort"),
    ("Costs jumped, cache_read=0", "volatile data early in prompt? tools/model changed?", "freeze the prefix, move volatile data last"),
    ("429 rate_limit_error", "which limit (RPM/ITPM/OTPM)? retry-after honored?", "throttle, honor retry-after, restore caching"),
    ("529 overloaded_error", "status page? our retries amplifying?", "jittered backoff, circuit breaker, fallback model"),
    ("404 not_found_error", "exact model id (claude-opus-5-5)? enabled for org?", "fix the id; GET /v1/models to list"),
    ("Can't reproduce a failure", "request_id logged? ANTHROPIC_LOG=debug?", "log request_id on every call and error"),
]
