"""
D6.4 - Architecture Documentation and Implementation Guidance

Docs-as-code for the Claims Correspondence Copilot:

  data/architecture_spec.json (machine-readable source of truth: components, models,
  prompts + versions, tools + auth scopes, data stores, integrations, guardrails, SLOs,
  alerts, request flow, rollout, operations)
     -> doc linter (rules DOC-001..013) gates publication; the draft spec FAILS
     -> a reviewed patch (data/spec_fixes.json) is applied; the spec re-lints clean
     -> Claude drafts the narrative parts (overview, container responsibilities, key
        decisions, runbook diagnosis/remediation) as STRUCTURED OUTPUT
     -> code validates Claude's coverage (every component, every alert) and renders
        deterministic templates:
          00 lint report, 01 C4 context/container description, 02 Mermaid diagrams
          (flowcharts + sequence), 03 implementation guide, 04 operational runbook,
          05 handoff checklist (+ the fixed spec)
     -> a final coverage check proves the docs are complete

Principle: the spec is the truth, code renders it, Claude writes prose - and nothing
Claude writes can drop a component or an alert without the build noticing.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_FABLE,
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    MockRequest,
    banner,
    get_client,
    section,
    usage_line,
)

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"
DOCS = OUT / "docs"
KNOWN_MODELS = [MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU]


# ---------------------------------------------------------------------------
# Structured outputs: Claude writes prose INTO slots that code owns.
# ---------------------------------------------------------------------------
class ContainerResponsibility(BaseModel):
    component_id: str
    responsibility: str


class KeyDecision(BaseModel):
    title: str
    rationale: str
    adr: Optional[str]


class NarrativeDraft(BaseModel):
    system_overview: str
    container_responsibilities: list[ContainerResponsibility]
    key_decisions: list[KeyDecision]
    operational_notes: list[str]


class RunbookEntry(BaseModel):
    alert_id: str
    likely_causes: list[str]
    diagnosis_steps: list[str]
    remediation_steps: list[str]
    escalation: str


class RunbookDraft(BaseModel):
    entries: list[RunbookEntry]


NARRATIVE_SYSTEM = """You are a solution architect writing architecture documentation from a machine-readable spec.
Write a concise system overview, one responsibility statement per component id in the spec, the key design
decisions (cite ADR ids only if the spec or context names them), and operational notes. Do not invent components,
systems or numbers that are not in the spec."""

RUNBOOK_SYSTEM = """You are an SRE writing an operational runbook from an architecture spec.
For EVERY alert id in the spec, give likely causes, ordered diagnosis steps, remediation steps and an escalation path.
Prefer safe, reversible remediations (rollback, pin, fallback mode); never suggest disabling a guardrail."""


def responder(req: MockRequest) -> str:
    mock = DATA / "mock_responses"
    if "architecture documentation" in req.system_text:
        return (mock / "narrative.json").read_text(encoding="utf-8")
    if "operational runbook" in req.system_text:
        return (mock / "runbook.json").read_text(encoding="utf-8")
    return "{}"


# ---------------------------------------------------------------------------
# Doc linter: completeness rules an architecture review board would enforce
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    rule: str
    severity: str  # ERROR | WARN
    ref: str
    message: str


RULES = {
    "DOC-001": "Every component has an accountable owner",
    "DOC-002": "Every dependency resolves to a known component, system or data store",
    "DOC-003": "Every tool declares a least-privilege auth scope",
    "DOC-004": "Write tools either write to a pending state or require human approval",
    "DOC-005": "Every SLO is backed by an existing alert",
    "DOC-006": "Every alert points at an existing component",
    "DOC-007": "Every prompt has a semantic version and an eval suite",
    "DOC-008": "PII data stores define retention, encryption and residency",
    "DOC-009": "Models use current model ids, are used by a component and have a fallback",
    "DOC-010": "Every agent has an output guardrail on itself or its caller",
    "DOC-011": "Every alert has a runbook entry",
    "DOC-012": "Secrets are sourced from the vault",
    "DOC-013": "Every request-flow participant exists in the spec",
}


def known_ids(spec: dict[str, Any]) -> set[str]:
    return ({c["id"] for c in spec["components"]} | {e["id"] for e in spec["external_systems"]}
            | {d["id"] for d in spec["data_stores"]})


def lint(spec: dict[str, Any]) -> list[Finding]:
    f: list[Finding] = []
    ids = known_ids(spec)
    comp_ids = {c["id"] for c in spec["components"]}
    for c in spec["components"]:
        if not c.get("owner"):
            f.append(Finding("DOC-001", "ERROR", c["id"], "component has no owner"))
        for dep in c.get("depends_on", []):
            if dep not in ids:
                f.append(Finding("DOC-002", "ERROR", c["id"], f"depends on unknown '{dep}'"))
    for t in spec["tools"]:
        if not t.get("auth_scope"):
            f.append(Finding("DOC-003", "ERROR", t["name"], "tool has no auth scope"))
        if t["side_effect"] == "write" and t.get("write_mode") != "pending" and not t.get("requires_approval"):
            f.append(Finding("DOC-004", "ERROR", t["name"], f"'{t.get('write_mode', 'final')}' write without human approval"))
    alert_ids = {a["id"] for a in spec["alerts"]}
    for s in spec["slos"]:
        if not s.get("alert"):
            f.append(Finding("DOC-005", "ERROR", s["id"], "SLO has no alert"))
        elif s["alert"] not in alert_ids:
            f.append(Finding("DOC-005", "ERROR", s["id"], f"alert '{s['alert']}' does not exist"))
    for a in spec["alerts"]:
        if a["component"] not in comp_ids:
            f.append(Finding("DOC-006", "ERROR", a["id"], f"alert component '{a['component']}' does not exist"))
    for p in spec["prompts"]:
        if not re.fullmatch(r"\d+\.\d+\.\d+", p.get("version") or ""):
            f.append(Finding("DOC-007", "ERROR", p["id"], "prompt version is not semver"))
        if not p.get("eval_suite"):
            f.append(Finding("DOC-007", "ERROR", p["id"], "prompt has no eval suite"))
    for d in spec["data_stores"]:
        if d.get("contains_pii"):
            missing = [k for k in ("retention", "encryption", "residency") if not d.get(k)]
            if missing:
                f.append(Finding("DOC-008", "ERROR", d["id"], f"PII store missing {', '.join(missing)}"))
    used_models = {m for c in spec["components"] for m in c.get("models", [])}
    for m in spec["models"]:
        if m["model"] not in KNOWN_MODELS:
            f.append(Finding("DOC-009", "WARN", m["id"], f"model id '{m['model']}' is not a current model"))
        if m["id"] not in used_models:
            f.append(Finding("DOC-009", "WARN", m["id"], "model is not used by any component"))
        if not m.get("fallback"):
            f.append(Finding("DOC-009", "WARN", m["id"], "model has no fallback"))
    output_guarded = {g["component"] for g in spec["guardrails"] if g["type"] == "output"}
    for c in spec["components"]:
        if c["type"] == "agent":
            callers = {x["id"] for x in spec["components"] if c["id"] in x.get("depends_on", [])}
            if c["id"] not in output_guarded and not (callers & output_guarded):
                f.append(Finding("DOC-010", "WARN", c["id"], "agent output is not guarded"))
    for cfg in spec["config"]:
        if cfg["secret"] and not str(cfg["source"]).startswith("Vault:"):
            f.append(Finding("DOC-012", "ERROR", cfg["key"], "secret not sourced from the vault"))
    participants = ids | {p["id"] for p in spec["people"]}
    for step in spec["request_flow"]:
        for end in (step["from"], step["to"]):
            if end not in participants:
                f.append(Finding("DOC-013", "ERROR", end, "request-flow participant not in spec"))
    return f


# ---------------------------------------------------------------------------
# Spec patching: tiny path language  collection[key=value].field
# ---------------------------------------------------------------------------
PATH_RE = re.compile(r"^(\w+)(?:\[(\w+)=([^\]]+)\])?(?:\.(\w+))?$")


def apply_fix(spec: dict[str, Any], op: dict[str, Any]) -> None:
    m = PATH_RE.match(op["path"])
    if not m:
        raise ValueError(f"bad path {op['path']}")
    coll, key, val, fld = m.groups()
    items: list[dict[str, Any]] = spec[coll]
    if op["op"] == "append":
        items.append(op["value"])
        return
    idx = next((i for i, it in enumerate(items) if str(it.get(key)) == val), None)
    if idx is None:
        raise ValueError(f"no item {key}={val} in {coll}")
    if op["op"] == "remove":
        items.pop(idx)
    elif op["op"] == "set":
        items[idx][fld] = op["value"]
    else:
        raise ValueError(f"unknown op {op['op']}")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def name_of(spec: dict[str, Any]) -> dict[str, str]:
    names = {p["id"]: p["name"] for p in spec["people"]}
    names.update({c["id"]: c["name"] for c in spec["components"]})
    names.update({e["id"]: e["name"] for e in spec["external_systems"]})
    names.update({d["id"]: d["name"] for d in spec["data_stores"]})
    return names


def node(id_: str, label: str, kind: str) -> str:
    if kind == "agent":
        return f'{id_}{{{{"{label}"}}}}'
    if kind == "job":
        return f'{id_}[/"{label}"/]'
    if kind == "datastore":
        return f'{id_}[("{label}")]'
    if kind == "person":
        return f'{id_}(["{label}"])'
    return f'{id_}["{label}"]'


def render_diagrams(spec: dict[str, Any]) -> str:
    names = name_of(spec)
    comp_ids = {c["id"] for c in spec["components"]}
    used_ext = [e for e in spec["external_systems"]
                if any(e["id"] in c.get("depends_on", []) for c in spec["components"])]
    L: list[str] = []
    a = L.append
    a(f"# Diagrams - {spec['system']['name']} v{spec['system']['version']}")
    a("")
    a("Generated from `architecture_spec.json`; paste into any Mermaid renderer (GitHub renders these natively).")
    a("")
    a("## C4 level 1 - system context")
    a("")
    a("```mermaid")
    a("flowchart LR")
    for p in spec["people"]:
        a(f"    {node(p['id'], p['name'], 'person')}")
    a(f'    {spec["system"]["id"]}["{spec["system"]["name"]}<br/>v{spec["system"]["version"]}"]')
    for e in spec["external_systems"]:
        a(f'    {e["id"]}["{e["name"]}<br/>{e["description"]}"]')
    for p in spec["people"]:
        a(f"    {p['id']} -->|uses| {spec['system']['id']}")
    for e in used_ext:
        a(f"    {spec['system']['id']} --> {e['id']}")
    unused = [e for e in spec["external_systems"] if e not in used_ext]
    for e in unused:
        a(f"    {spec['system']['id']} -.->|not called - by design| {e['id']}")
    a("    classDef ext fill:#eeeeee,stroke:#999999")
    a("    class " + ",".join(e["id"] for e in spec["external_systems"]) + " ext")
    a("```")
    a("")
    a("## C4 level 2 - containers")
    a("")
    a("```mermaid")
    a("flowchart LR")
    a('    subgraph people["People"]')
    for p in spec["people"]:
        a(f"        {node(p['id'], p['name'], 'person')}")
    a("    end")
    a(f'    subgraph system["{spec["system"]["name"]}"]')
    for c in spec["components"]:
        a(f"        {node(c['id'], c['name'] + '<br/>' + c['tech'], c['type'])}")
    a("    end")
    a('    subgraph stores["Data stores"]')
    for d in spec["data_stores"]:
        a(f"        {node(d['id'], d['name'] + '<br/>' + d['kind'], 'datastore')}")
    a("    end")
    a('    subgraph external["External systems"]')
    for e in used_ext:
        a(f'        {e["id"]}["{e["name"]}"]')
    a("    end")
    for p in spec["people"]:
        for u in p.get("uses", []):
            a(f"    {p['id']} --> {u}")
    for c in spec["components"]:
        for dep in c.get("depends_on", []):
            a(f"    {c['id']} --> {dep}")
    a("```")
    a("")
    a("## Sequence - one drafting request through the agent")
    a("")
    a("```mermaid")
    a("sequenceDiagram")
    seen: list[str] = []
    for step in spec["request_flow"]:
        for end in (step["from"], step["to"]):
            if end not in seen:
                seen.append(end)
    people_ids = {p["id"] for p in spec["people"]}
    for pid in seen:
        kind = "actor" if pid in people_ids else "participant"
        a(f"    {kind} {pid} as {names[pid]}")
    for step in spec["request_flow"]:
        arrow = "-->>" if step.get("reply") else "->>"
        msg = step["message"].replace(";", ",").replace("#", "no.")
        a(f"    {step['from']}{arrow}{step['to']}: {msg}")
    a("```")
    a("")
    a(f"_Components in the container view: {len(comp_ids)}; request-flow steps: {len(spec['request_flow'])}._")
    a("")
    return "\n".join(L)


def render_overview(spec: dict[str, Any], nar: NarrativeDraft) -> str:
    names = name_of(spec)
    resp = {r.component_id: r.responsibility for r in nar.container_responsibilities}
    L: list[str] = []
    a = L.append
    s = spec["system"]
    a(f"# {s['name']} - architecture overview (v{s['version']}, {s['status']})")
    a("")
    a(f"**Owner:** {s['owner']}  ")
    a(f"**Summary:** {s['summary']}")
    a("")
    a("## System overview")
    a("")
    a(nar.system_overview)
    a("")
    a("## C4 level 1 - context")
    a("")
    a("| Element | Type | Description | Owner |")
    a("|---|---|---|---|")
    for p in spec["people"]:
        a(f"| {p['name']} | Person | {p['description']} | - |")
    a(f"| **{s['name']}** | Software system | {s['summary']} | {s['owner']} |")
    for e in spec["external_systems"]:
        a(f"| {e['name']} | External system | {e['description']} | {e['owner']} |")
    a("")
    a("## C4 level 2 - containers")
    a("")
    a("| Container | Type | Technology | Responsibility | Owner | Depends on |")
    a("|---|---|---|---|---|---|")
    for c in spec["components"]:
        deps = ", ".join(names.get(d, d) for d in c.get("depends_on", []))
        a(f"| {c['name']} (`{c['id']}`) | {c['type']} | {c['tech']} | {resp.get(c['id'], '**MISSING**')} | {c['owner']} | {deps} |")
    a("")
    a("## Data stores")
    a("")
    a("| Store | Kind | Classification | PII | Retention | Encryption | Residency | Owner |")
    a("|---|---|---|---|---|---|---|---|")
    for d in spec["data_stores"]:
        a(f"| {d['name']} | {d['kind']} | {d['classification']} | {'yes' if d['contains_pii'] else 'no'} | {d['retention']} | "
          f"{d['encryption']} | {d['residency']} | {d['owner']} |")
    a("")
    a("## Key design decisions")
    a("")
    for k in nar.key_decisions:
        a(f"- **{k.title}**{f' ({k.adr})' if k.adr else ''}: {k.rationale}")
    a("")
    a("## Operational notes")
    a("")
    for n in nar.operational_notes:
        a(f"- {n}")
    a("")
    a("_Prose drafted by Claude from the spec (structured output); tables rendered from `architecture_spec.json`._")
    a("")
    return "\n".join(L)


API_CONVENTIONS = [
    "Model ids come from configuration (`DRAFT_MODEL`), never hard-coded; no date suffixes on current aliases.",
    "Thinking: `{\"type\": \"adaptive\"}`; control depth with `output_config.effort` per route (drafting `low`, judging `medium`).",
    "No `temperature`/`top_p`/`top_k` and no assistant prefill on current models - use structured outputs (`output_config.format`) for machine-readable results.",
    "Tools: `tool_choice` auto with `strict: true` schemas; read-only tools for the agent, writes only through the API layer.",
    "Stream user-facing drafts (time to first token matters more than total latency for adjusters).",
    "Prompt caching: keep the system prompt and policy prefix byte-stable (no timestamps or request ids); claim context goes after the cache breakpoint.",
    "Bulk and evaluation work uses the Message Batches API (asynchronous, lower cost).",
    "Log the response `request-id`, model, prompt version and token usage with every draft (audit and cost attribution).",
]

ERROR_HANDLING = [
    ("429 rate_limit_error", "HTTP 429", "SDK retries with exponential backoff (max_retries); shed batch load first; alert on sustained 429s"),
    ("529 overloaded_error / 5xx", "HTTP 529 / 500-504", "Retry with jitter; after repeated failures switch to template fallback mode"),
    ("400 invalid_request_error", "HTTP 400", "Do not retry; log request id; a spike after a deploy usually means an unsupported parameter"),
    ("stop_reason = max_tokens", "Response field", "Never show a truncated draft; raise the per-letter-type max_tokens or ask for a shorter letter"),
    ("stop_reason = refusal", "Response field", "Show 'manual drafting needed', log the category; do not auto-retry with a reworded prompt"),
    ("Structured output does not parse", "Parse error", "Treat as a failed attempt; retry once, then fall back"),
    ("Guardrail block (G2)", "Guardrail service", "Do not display the draft; route to manual drafting with the reason"),
    ("Field validator mismatch (G3)", "Drafting API", "Block the draft and highlight the mismatched fields against ClaimCore"),
    ("ClaimCore timeout / 5xx", "Circuit breaker", "Open the breaker, switch to template mode, page via AL-6"),
]


def render_impl_guide(spec: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    s = spec["system"]
    a(f"# Implementation guide - {s['name']} v{s['version']}")
    a("")
    a("## 1. Environments")
    a("")
    a("| Environment | Purpose | Data | Model access |")
    a("|---|---|---|---|")
    for e in spec["environments"]:
        a(f"| {e['name']} | {e['purpose']} | {e['data']} | {e['model_access']} |")
    a("")
    a("## 2. Configuration and secrets")
    a("")
    a("| Key | Secret | Source | Default |")
    a("|---|---|---|---|")
    for c in spec["config"]:
        a(f"| `{c['key']}` | {'yes' if c['secret'] else 'no'} | {c['source']} | {c['default'] if c['default'] is not None else '-'} |")
    a("")
    a("## 3. Models and Claude API conventions")
    a("")
    a("| Id | Model | Purpose | Effort | Thinking | max_tokens | Fallback |")
    a("|---|---|---|---|---|---:|---|")
    for m in spec["models"]:
        a(f"| {m['id']} | `{m['model']}` | {m['purpose']} | {m['effort']} | {m['thinking']} | {m['max_tokens']} | {m['fallback']} |")
    a("")
    for c in API_CONVENTIONS:
        a(f"- {c}")
    a("")
    a("## 4. Tools and least privilege")
    a("")
    a("| Tool | Component | Side effect | Auth scope | Human approval / write mode |")
    a("|---|---|---|---|---|")
    for t in spec["tools"]:
        gate = "approval required" if t.get("requires_approval") else (f"writes {t['write_mode']} only" if t.get("write_mode") else "n/a (read)")
        a(f"| {t['name']} | {t['component']} | {t['side_effect']} | `{t['auth_scope']}` | {gate} |")
    a("")
    a("## 5. Prompt management")
    a("")
    a("| Prompt | Version | Owner | Component | Path | Eval suite |")
    a("|---|---|---|---|---|---|")
    for p in spec["prompts"]:
        a(f"| {p['id']} | {p['version']} | {p['owner']} | {p['component']} | `{p['path']}` | `{p['eval_suite']}` |")
    a("")
    for rule in [
        "Prompts live in the prompt registry (Git, signed tags) and are versioned with semver: major = behaviour change, minor = wording, patch = typo.",
        "Production pins a version (`PROMPT_VERSION_PIN`); no prompt change reaches production without passing its eval suite in staging.",
        "Every change runs shadow, then canary with latency, cost and quality gates (see rollout plan); rollback = re-pin the previous version.",
        "The prompt owner approves wording changes; the AI Delivery Team approves structural and model changes.",
        "Record the prompt version and model id with every draft in the audit store.",
    ]:
        a(f"- {rule}")
    a("")
    a("## 6. Integrations")
    a("")
    a("| System | Protocol | Auth | Direction | Limits |")
    a("|---|---|---|---|---|")
    names = name_of(spec)
    for i in spec["integrations"]:
        a(f"| {names.get(i['system'], i['system'])} | {i['protocol']} | {i['auth']} | {i['direction']} | {i['limits']} |")
    a("")
    a("## 7. Guardrails")
    a("")
    a("| Id | Type | Control | Implemented in |")
    a("|---|---|---|---|")
    for g in spec["guardrails"]:
        a(f"| {g['id']} | {g['type']} | {g['description']} | {g['component']} |")
    a("")
    a("## 8. Error handling")
    a("")
    a("| Condition | Detection | Handling |")
    a("|---|---|---|")
    for cond, det, hand in ERROR_HANDLING:
        a(f"| {cond} | {det} | {hand} |")
    a("")
    a("## 9. Rollout plan")
    a("")
    a("| Phase | Traffic | Entry criteria | Exit criteria | Rollback trigger |")
    a("|---|---|---|---|---|")
    for r in spec["rollout"]:
        a(f"| {r['phase']} | {r['traffic']} | {r['entry']} | {r['exit']} | {r['rollback']} |")
    a("")
    a("## 10. Observability")
    a("")
    a("| SLO | Target | Alert |")
    a("|---|---|---|")
    for slo in spec["slos"]:
        a(f"| {slo['id']} {slo['name']} | {slo['target']} | {slo['alert']} |")
    a("")
    a("Dashboards: " + "; ".join(spec["operations"]["dashboards"]) + ".")
    a("")
    return "\n".join(L)


def render_runbook(spec: dict[str, Any], rb: RunbookDraft) -> str:
    comp = {c["id"]: c for c in spec["components"]}
    entry = {e.alert_id: e for e in rb.entries}
    slo_by_alert = {s["alert"]: s for s in spec["slos"] if s.get("alert")}
    L: list[str] = []
    a = L.append
    a(f"# Operational runbook - {spec['system']['name']}")
    a("")
    a(f"**On-call:** {spec['operations']['on_call']}")
    a("")
    a("Flow for every alert: acknowledge -> diagnose (steps below) -> remediate with the safest reversible action -> "
      "escalate if not mitigated in 30 minutes -> postmortem if the error budget impact is above 20%.")
    a("")
    a("| Alert | Severity | Component | Owner |")
    a("|---|---|---|---|")
    for al in spec["alerts"]:
        a(f"| {al['id']} {al['name']} | {al['severity'].upper()} | {al['component']} | {comp[al['component']]['owner']} |")
    a("")
    for al in spec["alerts"]:
        e = entry.get(al["id"])
        a(f"### {al['id']} - {al['name']} ({al['severity'].upper()})")
        a("")
        a(f"- **Condition:** {al['condition']}")
        a(f"- **Component / owner:** {al['component']} / {comp[al['component']]['owner']}")
        if al["id"] in slo_by_alert:
            s = slo_by_alert[al["id"]]
            a(f"- **Protects:** {s['id']} {s['name']} ({s['target']})")
        if e is None:
            a("- **MISSING RUNBOOK ENTRY**")
            a("")
            continue
        a("")
        a("**Likely causes**")
        a("")
        for x in e.likely_causes:
            a(f"- {x}")
        a("")
        a("**Diagnosis**")
        a("")
        for i, x in enumerate(e.diagnosis_steps, 1):
            a(f"{i}. {x}")
        a("")
        a("**Remediation**")
        a("")
        for i, x in enumerate(e.remediation_steps, 1):
            a(f"{i}. {x}")
        a("")
        a(f"**Escalation:** {e.escalation}")
        a("")
    a("_Diagnosis and remediation drafted by Claude from the spec; alert metadata, owners and SLO links rendered from the spec._")
    a("")
    return "\n".join(L)


def render_handoff(spec: dict[str, Any], checks: list[tuple[str, bool, str]]) -> str:
    L: list[str] = []
    a = L.append
    done = sum(1 for _, ok, _ in checks if ok)
    a(f"# Handoff checklist - {spec['system']['name']} v{spec['system']['version']}")
    a("")
    a(f"**Status:** {'READY' if done == len(checks) else 'NOT READY'} ({done}/{len(checks)} items complete)")
    a("")
    for label, ok, evidence in checks:
        a(f"- [{'x' if ok else ' '}] {label} - _{evidence}_")
    a("")
    a("Knowledge transfer: " + "; ".join(spec["operations"]["kt_sessions"]) + ".")
    a("")
    return "\n".join(L)


def render_lint_report(draft: list[Finding], fixed: list[Finding], fixes: dict[str, Any], runbook_missing: list[str]) -> str:
    L: list[str] = []
    a = L.append
    a("# Doc-lint report")
    a("")
    a("| Rule | Description |")
    a("|---|---|")
    for k, v in RULES.items():
        a(f"| {k} | {v} |")
    a("")
    a(f"## Draft spec: {sum(1 for x in draft if x.severity == 'ERROR')} errors, {sum(1 for x in draft if x.severity == 'WARN')} warnings - publication blocked")
    a("")
    a("| Rule | Severity | Ref | Finding |")
    a("|---|---|---|---|")
    for x in draft:
        a(f"| {x.rule} | {x.severity} | {x.ref} | {x.message} |")
    a("")
    a(f"## Fixes applied - {fixes['description']}")
    a("")
    for op in fixes["operations"]:
        a(f"- `{op['op']} {op['path']}` - {op['reason']}")
    a("")
    a(f"## Fixed spec: {sum(1 for x in fixed if x.severity == 'ERROR')} errors, {sum(1 for x in fixed if x.severity == 'WARN')} warnings")
    a("")
    a(f"Runbook coverage (DOC-011): {'complete' if not runbook_missing else 'missing ' + ', '.join(runbook_missing)}.")
    a("")
    return "\n".join(L)


def main() -> None:
    banner("D6.4 - Architecture Documentation and Implementation Guidance")
    spec = json.loads((DATA / "architecture_spec.json").read_text(encoding="utf-8"))
    fixes = json.loads((DATA / "spec_fixes.json").read_text(encoding="utf-8"))
    client = get_client(responder)

    # -- 1. Spec ----------------------------------------------------------------
    section("1. Machine-readable architecture spec (single source of truth)")
    s = spec["system"]
    print(f"  {s['name']} v{s['version']} ({s['status']}), owner {s['owner']}")
    for key in ("people", "external_systems", "components", "models", "prompts", "tools", "data_stores",
                "integrations", "guardrails", "slos", "alerts", "request_flow", "rollout"):
        print(f"    {key:<17} {len(spec[key]):>3}")

    # -- 2. Lint draft --------------------------------------------------------------
    section("2. Doc linter on the draft spec (gate before publishing)")
    draft_findings = lint(spec)
    for x in draft_findings:
        print(f"  {x.severity:<5} {x.rule} {x.ref:<17} {x.message}")
    errors = sum(1 for x in draft_findings if x.severity == "ERROR")
    print(f"  => {errors} errors: publication BLOCKED (docs are built like code - lint must pass)")

    # -- 3. Fix + re-lint -----------------------------------------------------------
    section("3. Apply the reviewed patch and re-lint")
    fixed = copy.deepcopy(spec)
    for op in fixes["operations"]:
        apply_fix(fixed, op)
        print(f"  {op['op']:<6} {op['path']}")
    fixed_findings = lint(fixed)
    for x in fixed_findings:
        print(f"  {x.severity:<5} {x.rule} {x.ref:<17} {x.message}")
    fixed_errors = sum(1 for x in fixed_findings if x.severity == "ERROR")
    print(f"  => {fixed_errors} errors, {len(fixed_findings) - fixed_errors} warnings: {'PASS' if fixed_errors == 0 else 'FAIL'}")
    if fixed_errors:
        raise SystemExit("spec still fails lint")

    # -- 4. Claude narrative ------------------------------------------------------
    section("4. Claude drafts narrative sections (structured output)")
    spec_json = json.dumps(fixed, indent=1)
    resp = client.messages.parse(
        model=MODEL_OPUS, max_tokens=4000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        system=NARRATIVE_SYSTEM,
        messages=[{"role": "user", "content": "Architecture spec (JSON):\n" + spec_json
                   + "\n\nRelated decision: ADR-0007 (long-context grounding with prompt caching)."}],
        output_format=NarrativeDraft,
    )
    nar = resp.parsed_output
    if nar is None:
        raise RuntimeError("narrative did not parse")
    comp_ids = [c["id"] for c in fixed["components"]]
    got = [r.component_id for r in nar.container_responsibilities]
    print(f"  narrative | {usage_line(resp)}")
    print(f"  coverage: {sum(1 for c in comp_ids if c in got)}/{len(comp_ids)} components described; "
          f"unknown ids: {[g for g in got if g not in comp_ids] or 'none'}; key decisions: {len(nar.key_decisions)}")

    # Spec-driven, well-bounded drafting: Sonnet is enough, and validation enforces completeness.
    resp = client.messages.parse(
        model=MODEL_SONNET, max_tokens=4000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        system=RUNBOOK_SYSTEM,
        messages=[{"role": "user", "content": "Architecture spec (JSON):\n" + spec_json}],
        output_format=RunbookDraft,
    )
    rb = resp.parsed_output
    if rb is None:
        raise RuntimeError("runbook did not parse")
    alert_ids = [al["id"] for al in fixed["alerts"]]
    rb_ids = [e.alert_id for e in rb.entries]
    runbook_missing = [x for x in alert_ids if x not in rb_ids]
    thin = [e.alert_id for e in rb.entries if len(e.diagnosis_steps) < 2 or not e.remediation_steps]
    print(f"  runbook   | {usage_line(resp)}")
    print(f"  DOC-011 runbook coverage: {len(alert_ids) - len(runbook_missing)}/{len(alert_ids)} alerts; "
          f"missing={runbook_missing or 'none'}; thin entries={thin or 'none'}")

    # -- 5. Generate docs -----------------------------------------------------------
    section("5. Render docs from templates (deterministic)")
    check_results = {r: not any(x.rule == r and x.severity == "ERROR" for x in fixed_findings) for r in RULES}
    check_results["DOC-011"] = not runbook_missing
    ops = fixed["operations"]
    checks = [
        ("Every component has an accountable owner", check_results["DOC-001"], "DOC-001"),
        ("All dependencies resolve (no phantom components)", check_results["DOC-002"], "DOC-002"),
        ("Every tool has a least-privilege auth scope", check_results["DOC-003"], "DOC-003"),
        ("No write tool bypasses human approval", check_results["DOC-004"], "DOC-004"),
        ("Every SLO is backed by an alert", check_results["DOC-005"], "DOC-005"),
        ("Every alert has a runbook entry", check_results["DOC-011"], "DOC-011"),
        ("Every prompt is versioned with an eval suite", check_results["DOC-007"], "DOC-007"),
        ("PII data stores define retention, encryption and residency", check_results["DOC-008"], "DOC-008"),
        ("Secrets are sourced from the vault", check_results["DOC-012"], "DOC-012"),
        ("On-call rotation defined", bool(ops["on_call"]), ops["on_call"]),
        ("Dashboards cover SLOs, cost and guardrails", len(ops["dashboards"]) >= 3, f"{len(ops['dashboards'])} dashboards"),
        ("Every rollout phase has exit and rollback criteria", all(r["exit"] and r["rollback"] for r in fixed["rollout"]),
         f"{len(fixed['rollout'])} phases"),
        ("Knowledge-transfer sessions completed", all("(done)" in k for k in ops["kt_sessions"]), f"{len(ops['kt_sessions'])} sessions"),
        ("Operations acceptance sign-off recorded", ops["acceptance_signoff"] is not None,
         ops["acceptance_signoff"] or "pending - Claims Platform lead"),
    ]
    docs = {
        "00-lint-report.md": render_lint_report(draft_findings, fixed_findings, fixes, runbook_missing),
        "01-architecture-overview.md": render_overview(fixed, nar),
        "02-diagrams.md": render_diagrams(fixed),
        "03-implementation-guide.md": render_impl_guide(fixed),
        "04-runbook.md": render_runbook(fixed, rb),
        "05-handoff-checklist.md": render_handoff(fixed, checks),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, content in docs.items():
        (DOCS / name).write_text(content, encoding="utf-8")
        print(f"  wrote labs_output/docs/{name:<29} {len(content.splitlines()):>4} lines")
    (DOCS / "architecture_spec.fixed.json").write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")
    print("  wrote labs_output/docs/architecture_spec.fixed.json")

    # -- 6. Coverage of the generated docs -------------------------------------------
    section("6. Coverage check on the generated docs")
    diag, guide, runbook = docs["02-diagrams.md"], docs["03-implementation-guide.md"], docs["04-runbook.md"]
    cov = [
        ("components in container diagram", sum(1 for c in comp_ids if f"{c}[" in diag or f"{c}{{{{" in diag), len(comp_ids)),
        ("alerts with a runbook section", sum(1 for x in alert_ids if f"### {x} " in runbook and "MISSING RUNBOOK" not in runbook), len(alert_ids)),
        ("prompts in the registry table", sum(1 for p in fixed["prompts"] if f"| {p['id']} |" in guide), len(fixed["prompts"])),
        ("tools with auth scope in the guide", sum(1 for t in fixed["tools"] if f"| {t['name']} |" in guide), len(fixed["tools"])),
        ("request-flow steps in sequence diagram", sum(1 for st in fixed["request_flow"] if st["message"] in diag), len(fixed["request_flow"])),
    ]
    for label, have, total in cov:
        print(f"  [{'OK ' if have == total else 'GAP'}] {have}/{total} {label}")
    done = sum(1 for _, ok, _ in checks if ok)
    print(f"  Handoff checklist: {done}/{len(checks)} complete -> {'READY' if done == len(checks) else 'NOT READY'}"
          f" (open: {', '.join(lbl for lbl, ok, _ in checks if not ok) or 'none'})")


if __name__ == "__main__":
    main()
