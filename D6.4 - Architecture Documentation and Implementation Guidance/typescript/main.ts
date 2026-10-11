/**
 * D6.4 - Architecture Documentation and Implementation Guidance
 *
 * Docs-as-code for the Claims Correspondence Copilot:
 *
 *   data/architecture_spec.json (machine-readable source of truth: components, models,
 *   prompts + versions, tools + auth scopes, data stores, integrations, guardrails, SLOs,
 *   alerts, request flow, rollout, operations)
 *     -> doc linter (rules DOC-001..013) gates publication; the draft spec FAILS
 *     -> a reviewed patch (data/spec_fixes.json) is applied; the spec re-lints clean
 *     -> Claude drafts the narrative parts (overview, container responsibilities, key
 *        decisions, runbook diagnosis/remediation) as STRUCTURED OUTPUT
 *     -> code validates Claude's coverage (every component, every alert) and renders
 *        deterministic templates:
 *          00 lint report, 01 C4 context/container description, 02 Mermaid diagrams
 *          (flowcharts + sequence), 03 implementation guide, 04 operational runbook,
 *          05 handoff checklist (+ the fixed spec)
 *     -> a final coverage check proves the docs are complete
 *
 * Principle: the spec is the truth, code renders it, Claude writes prose - and nothing
 * Claude writes can drop a component or an alert without the build noticing.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import {
  MODEL_FABLE, MODEL_HAIKU, MODEL_OPUS, MODEL_SONNET, banner, getClient, section, usageLine,
} from "../../shared/typescript/claudeClient.ts";
import type { MockRequest } from "../../shared/typescript/claudeClient.ts";

const LAB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA = path.join(LAB, "data");
const OUT = path.join(LAB, "labs_output");
const DOCS = path.join(OUT, "docs");
const KNOWN_MODELS = [MODEL_FABLE, MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU];

// ---------------------------------------------------------------------------
// Structured outputs: Claude writes prose INTO slots that code owns.
// ---------------------------------------------------------------------------
const NarrativeDraft = z.object({
  system_overview: z.string(),
  container_responsibilities: z.array(z.object({ component_id: z.string(), responsibility: z.string() })),
  key_decisions: z.array(z.object({ title: z.string(), rationale: z.string(), adr: z.string().nullable() })),
  operational_notes: z.array(z.string()),
});
type NarrativeDraft = z.infer<typeof NarrativeDraft>;

const RunbookDraft = z.object({
  entries: z.array(
    z.object({
      alert_id: z.string(),
      likely_causes: z.array(z.string()),
      diagnosis_steps: z.array(z.string()),
      remediation_steps: z.array(z.string()),
      escalation: z.string(),
    }),
  ),
});
type RunbookDraft = z.infer<typeof RunbookDraft>;

const NARRATIVE_SYSTEM = `You are a solution architect writing architecture documentation from a machine-readable spec.
Write a concise system overview, one responsibility statement per component id in the spec, the key design
decisions (cite ADR ids only if the spec or context names them), and operational notes. Do not invent components,
systems or numbers that are not in the spec.`;

const RUNBOOK_SYSTEM = `You are an SRE writing an operational runbook from an architecture spec.
For EVERY alert id in the spec, give likely causes, ordered diagnosis steps, remediation steps and an escalation path.
Prefer safe, reversible remediations (rollback, pin, fallback mode); never suggest disabling a guardrail.`;

function responder(req: MockRequest): string {
  const mock = path.join(DATA, "mock_responses");
  if (req.systemText.includes("architecture documentation")) return fs.readFileSync(path.join(mock, "narrative.json"), "utf-8");
  if (req.systemText.includes("operational runbook")) return fs.readFileSync(path.join(mock, "runbook.json"), "utf-8");
  return "{}";
}

// ---------------------------------------------------------------------------
// Spec shape (the JSON file is the source of truth)
// ---------------------------------------------------------------------------
interface Component { id: string; name: string; type: string; tech: string; owner: string | null; depends_on: string[]; models?: string[] }
interface Tool { name: string; component: string; description: string; side_effect: string; write_mode?: string; auth_scope: string | null; requires_approval: boolean }
interface Spec {
  system: { id: string; name: string; version: string; status: string; owner: string; summary: string };
  people: { id: string; name: string; description: string; uses: string[] }[];
  external_systems: { id: string; name: string; description: string; owner: string }[];
  components: Component[];
  models: { id: string; model: string; purpose: string; effort: string; thinking: string; max_tokens: number; fallback: string | null }[];
  prompts: { id: string; version: string | null; owner: string; component: string; path: string; eval_suite: string | null }[];
  tools: Tool[];
  data_stores: {
    id: string; name: string; kind: string; classification: string; contains_pii: boolean;
    retention: string | null; encryption: string | null; residency: string | null; owner: string;
  }[];
  integrations: { system: string; protocol: string; auth: string; direction: string; limits: string }[];
  guardrails: { id: string; type: string; description: string; component: string }[];
  slos: { id: string; name: string; target: string; alert: string | null }[];
  alerts: { id: string; name: string; condition: string; severity: string; component: string }[];
  request_flow: { from: string; to: string; message: string; reply?: boolean }[];
  environments: { name: string; purpose: string; data: string; model_access: string }[];
  config: { key: string; secret: boolean; source: string; default: string | null }[];
  rollout: { phase: string; traffic: string; entry: string; exit: string; rollback: string }[];
  operations: { on_call: string; dashboards: string[]; acceptance_signoff: string | null; kt_sessions: string[] };
}
interface FixOp { op: string; path: string; value?: unknown; reason: string }
interface Fixes { description: string; operations: FixOp[] }

// ---------------------------------------------------------------------------
// Doc linter: completeness rules an architecture review board would enforce
// ---------------------------------------------------------------------------
interface Finding { rule: string; severity: string; ref: string; message: string }

const RULES: Record<string, string> = {
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
};

const knownIds = (spec: Spec): Set<string> =>
  new Set([...spec.components.map((c) => c.id), ...spec.external_systems.map((e) => e.id), ...spec.data_stores.map((d) => d.id)]);

function lint(spec: Spec): Finding[] {
  const f: Finding[] = [];
  const add = (rule: string, severity: string, ref: string, message: string): void => {
    f.push({ rule, severity, ref, message });
  };
  const ids = knownIds(spec);
  const compIds = new Set(spec.components.map((c) => c.id));
  for (const c of spec.components) {
    if (!c.owner) add("DOC-001", "ERROR", c.id, "component has no owner");
    for (const dep of c.depends_on ?? []) if (!ids.has(dep)) add("DOC-002", "ERROR", c.id, `depends on unknown '${dep}'`);
  }
  for (const t of spec.tools) {
    if (!t.auth_scope) add("DOC-003", "ERROR", t.name, "tool has no auth scope");
    if (t.side_effect === "write" && t.write_mode !== "pending" && !t.requires_approval) {
      add("DOC-004", "ERROR", t.name, `'${t.write_mode ?? "final"}' write without human approval`);
    }
  }
  const alertIds = new Set(spec.alerts.map((a) => a.id));
  for (const s of spec.slos) {
    if (!s.alert) add("DOC-005", "ERROR", s.id, "SLO has no alert");
    else if (!alertIds.has(s.alert)) add("DOC-005", "ERROR", s.id, `alert '${s.alert}' does not exist`);
  }
  for (const a of spec.alerts) if (!compIds.has(a.component)) add("DOC-006", "ERROR", a.id, `alert component '${a.component}' does not exist`);
  for (const p of spec.prompts) {
    if (!/^\d+\.\d+\.\d+$/.test(p.version ?? "")) add("DOC-007", "ERROR", p.id, "prompt version is not semver");
    if (!p.eval_suite) add("DOC-007", "ERROR", p.id, "prompt has no eval suite");
  }
  for (const d of spec.data_stores) {
    if (d.contains_pii) {
      const missing = (["retention", "encryption", "residency"] as const).filter((k) => !d[k]);
      if (missing.length) add("DOC-008", "ERROR", d.id, `PII store missing ${missing.join(", ")}`);
    }
  }
  const usedModels = new Set(spec.components.flatMap((c) => c.models ?? []));
  for (const m of spec.models) {
    if (!KNOWN_MODELS.includes(m.model)) add("DOC-009", "WARN", m.id, `model id '${m.model}' is not a current model`);
    if (!usedModels.has(m.id)) add("DOC-009", "WARN", m.id, "model is not used by any component");
    if (!m.fallback) add("DOC-009", "WARN", m.id, "model has no fallback");
  }
  const outputGuarded = new Set(spec.guardrails.filter((g) => g.type === "output").map((g) => g.component));
  for (const c of spec.components) {
    if (c.type === "agent") {
      const callers = spec.components.filter((x) => (x.depends_on ?? []).includes(c.id)).map((x) => x.id);
      if (!outputGuarded.has(c.id) && !callers.some((x) => outputGuarded.has(x))) add("DOC-010", "WARN", c.id, "agent output is not guarded");
    }
  }
  for (const cfg of spec.config) if (cfg.secret && !String(cfg.source).startsWith("Vault:")) add("DOC-012", "ERROR", cfg.key, "secret not sourced from the vault");
  const participants = new Set([...ids, ...spec.people.map((p) => p.id)]);
  for (const step of spec.request_flow) {
    for (const end of [step.from, step.to]) if (!participants.has(end)) add("DOC-013", "ERROR", end, "request-flow participant not in spec");
  }
  return f;
}

// ---------------------------------------------------------------------------
// Spec patching: tiny path language  collection[key=value].field
// ---------------------------------------------------------------------------
const PATH_RE = /^(\w+)(?:\[(\w+)=([^\]]+)\])?(?:\.(\w+))?$/;

function applyFix(spec: Spec, op: FixOp): void {
  const m = PATH_RE.exec(op.path);
  if (!m) throw new Error(`bad path ${op.path}`);
  const [, coll, key, val, fld] = m;
  const items = (spec as unknown as Record<string, Record<string, unknown>[]>)[coll!]!;
  if (op.op === "append") {
    items.push(op.value as Record<string, unknown>);
    return;
  }
  const idx = items.findIndex((it) => String(it[key!]) === val);
  if (idx < 0) throw new Error(`no item ${key}=${val} in ${coll}`);
  if (op.op === "remove") items.splice(idx, 1);
  else if (op.op === "set") items[idx]![fld!] = op.value;
  else throw new Error(`unknown op ${op.op}`);
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------
function nameOf(spec: Spec): Map<string, string> {
  return new Map([
    ...spec.people.map((p): [string, string] => [p.id, p.name]),
    ...spec.components.map((c): [string, string] => [c.id, c.name]),
    ...spec.external_systems.map((e): [string, string] => [e.id, e.name]),
    ...spec.data_stores.map((d): [string, string] => [d.id, d.name]),
  ]);
}

function node(id: string, label: string, kind: string): string {
  if (kind === "agent") return `${id}{{"${label}"}}`;
  if (kind === "job") return `${id}[/"${label}"/]`;
  if (kind === "datastore") return `${id}[("${label}")]`;
  if (kind === "person") return `${id}(["${label}"])`;
  return `${id}["${label}"]`;
}

function renderDiagrams(spec: Spec): string {
  const names = nameOf(spec);
  const usedExt = spec.external_systems.filter((e) => spec.components.some((c) => (c.depends_on ?? []).includes(e.id)));
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  const sys = spec.system;
  a(`# Diagrams - ${sys.name} v${sys.version}`);
  a("");
  a("Generated from `architecture_spec.json`; paste into any Mermaid renderer (GitHub renders these natively).");
  a("");
  a("## C4 level 1 - system context");
  a("");
  a("```mermaid");
  a("flowchart LR");
  for (const p of spec.people) a(`    ${node(p.id, p.name, "person")}`);
  a(`    ${sys.id}["${sys.name}<br/>v${sys.version}"]`);
  for (const e of spec.external_systems) a(`    ${e.id}["${e.name}<br/>${e.description}"]`);
  for (const p of spec.people) a(`    ${p.id} -->|uses| ${sys.id}`);
  for (const e of usedExt) a(`    ${sys.id} --> ${e.id}`);
  for (const e of spec.external_systems.filter((x) => !usedExt.includes(x))) a(`    ${sys.id} -.->|not called - by design| ${e.id}`);
  a("    classDef ext fill:#eeeeee,stroke:#999999");
  a("    class " + spec.external_systems.map((e) => e.id).join(",") + " ext");
  a("```");
  a("");
  a("## C4 level 2 - containers");
  a("");
  a("```mermaid");
  a("flowchart LR");
  a('    subgraph people["People"]');
  for (const p of spec.people) a(`        ${node(p.id, p.name, "person")}`);
  a("    end");
  a(`    subgraph system["${sys.name}"]`);
  for (const c of spec.components) a(`        ${node(c.id, `${c.name}<br/>${c.tech}`, c.type)}`);
  a("    end");
  a('    subgraph stores["Data stores"]');
  for (const d of spec.data_stores) a(`        ${node(d.id, `${d.name}<br/>${d.kind}`, "datastore")}`);
  a("    end");
  a('    subgraph external["External systems"]');
  for (const e of usedExt) a(`        ${e.id}["${e.name}"]`);
  a("    end");
  for (const p of spec.people) for (const u of p.uses ?? []) a(`    ${p.id} --> ${u}`);
  for (const c of spec.components) for (const dep of c.depends_on ?? []) a(`    ${c.id} --> ${dep}`);
  a("```");
  a("");
  a("## Sequence - one drafting request through the agent");
  a("");
  a("```mermaid");
  a("sequenceDiagram");
  const seen: string[] = [];
  for (const step of spec.request_flow) for (const end of [step.from, step.to]) if (!seen.includes(end)) seen.push(end);
  const peopleIds = new Set(spec.people.map((p) => p.id));
  for (const pid of seen) a(`    ${peopleIds.has(pid) ? "actor" : "participant"} ${pid} as ${names.get(pid)}`);
  for (const step of spec.request_flow) {
    const arrow = step.reply ? "-->>" : "->>";
    const msg = step.message.replace(/;/g, ",").replace(/#/g, "no.");
    a(`    ${step.from}${arrow}${step.to}: ${msg}`);
  }
  a("```");
  a("");
  a(`_Components in the container view: ${spec.components.length}; request-flow steps: ${spec.request_flow.length}._`);
  a("");
  return L.join("\n");
}

function renderOverview(spec: Spec, nar: NarrativeDraft): string {
  const names = nameOf(spec);
  const resp = new Map(nar.container_responsibilities.map((r) => [r.component_id, r.responsibility]));
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  const s = spec.system;
  a(`# ${s.name} - architecture overview (v${s.version}, ${s.status})`);
  a("");
  a(`**Owner:** ${s.owner}  `);
  a(`**Summary:** ${s.summary}`);
  a("");
  a("## System overview");
  a("");
  a(nar.system_overview);
  a("");
  a("## C4 level 1 - context");
  a("");
  a("| Element | Type | Description | Owner |");
  a("|---|---|---|---|");
  for (const p of spec.people) a(`| ${p.name} | Person | ${p.description} | - |`);
  a(`| **${s.name}** | Software system | ${s.summary} | ${s.owner} |`);
  for (const e of spec.external_systems) a(`| ${e.name} | External system | ${e.description} | ${e.owner} |`);
  a("");
  a("## C4 level 2 - containers");
  a("");
  a("| Container | Type | Technology | Responsibility | Owner | Depends on |");
  a("|---|---|---|---|---|---|");
  for (const c of spec.components) {
    const deps = (c.depends_on ?? []).map((d) => names.get(d) ?? d).join(", ");
    a(`| ${c.name} (\`${c.id}\`) | ${c.type} | ${c.tech} | ${resp.get(c.id) ?? "**MISSING**"} | ${c.owner} | ${deps} |`);
  }
  a("");
  a("## Data stores");
  a("");
  a("| Store | Kind | Classification | PII | Retention | Encryption | Residency | Owner |");
  a("|---|---|---|---|---|---|---|---|");
  for (const d of spec.data_stores) {
    a(`| ${d.name} | ${d.kind} | ${d.classification} | ${d.contains_pii ? "yes" : "no"} | ${d.retention} | ${d.encryption} | ${d.residency} | ${d.owner} |`);
  }
  a("");
  a("## Key design decisions");
  a("");
  for (const k of nar.key_decisions) a(`- **${k.title}**${k.adr ? ` (${k.adr})` : ""}: ${k.rationale}`);
  a("");
  a("## Operational notes");
  a("");
  for (const n of nar.operational_notes) a(`- ${n}`);
  a("");
  a("_Prose drafted by Claude from the spec (structured output); tables rendered from `architecture_spec.json`._");
  a("");
  return L.join("\n");
}

const API_CONVENTIONS = [
  "Model ids come from configuration (`DRAFT_MODEL`), never hard-coded; no date suffixes on current aliases.",
  'Thinking: `{"type": "adaptive"}`; control depth with `output_config.effort` per route (drafting `low`, judging `medium`).',
  "No `temperature`/`top_p`/`top_k` and no assistant prefill on current models - use structured outputs (`output_config.format`) for machine-readable results.",
  "Tools: `tool_choice` auto with `strict: true` schemas; read-only tools for the agent, writes only through the API layer.",
  "Stream user-facing drafts (time to first token matters more than total latency for adjusters).",
  "Prompt caching: keep the system prompt and policy prefix byte-stable (no timestamps or request ids); claim context goes after the cache breakpoint.",
  "Bulk and evaluation work uses the Message Batches API (asynchronous, lower cost).",
  "Log the response `request-id`, model, prompt version and token usage with every draft (audit and cost attribution).",
];

const ERROR_HANDLING: [string, string, string][] = [
  ["429 rate_limit_error", "HTTP 429", "SDK retries with exponential backoff (max_retries); shed batch load first; alert on sustained 429s"],
  ["529 overloaded_error / 5xx", "HTTP 529 / 500-504", "Retry with jitter; after repeated failures switch to template fallback mode"],
  ["400 invalid_request_error", "HTTP 400", "Do not retry; log request id; a spike after a deploy usually means an unsupported parameter"],
  ["stop_reason = max_tokens", "Response field", "Never show a truncated draft; raise the per-letter-type max_tokens or ask for a shorter letter"],
  ["stop_reason = refusal", "Response field", "Show 'manual drafting needed', log the category; do not auto-retry with a reworded prompt"],
  ["Structured output does not parse", "Parse error", "Treat as a failed attempt; retry once, then fall back"],
  ["Guardrail block (G2)", "Guardrail service", "Do not display the draft; route to manual drafting with the reason"],
  ["Field validator mismatch (G3)", "Drafting API", "Block the draft and highlight the mismatched fields against ClaimCore"],
  ["ClaimCore timeout / 5xx", "Circuit breaker", "Open the breaker, switch to template mode, page via AL-6"],
];

function renderImplGuide(spec: Spec): string {
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  const s = spec.system;
  a(`# Implementation guide - ${s.name} v${s.version}`);
  a("");
  a("## 1. Environments");
  a("");
  a("| Environment | Purpose | Data | Model access |");
  a("|---|---|---|---|");
  for (const e of spec.environments) a(`| ${e.name} | ${e.purpose} | ${e.data} | ${e.model_access} |`);
  a("");
  a("## 2. Configuration and secrets");
  a("");
  a("| Key | Secret | Source | Default |");
  a("|---|---|---|---|");
  for (const c of spec.config) a(`| \`${c.key}\` | ${c.secret ? "yes" : "no"} | ${c.source} | ${c.default !== null ? c.default : "-"} |`);
  a("");
  a("## 3. Models and Claude API conventions");
  a("");
  a("| Id | Model | Purpose | Effort | Thinking | max_tokens | Fallback |");
  a("|---|---|---|---|---|---:|---|");
  for (const m of spec.models) a(`| ${m.id} | \`${m.model}\` | ${m.purpose} | ${m.effort} | ${m.thinking} | ${m.max_tokens} | ${m.fallback} |`);
  a("");
  for (const c of API_CONVENTIONS) a(`- ${c}`);
  a("");
  a("## 4. Tools and least privilege");
  a("");
  a("| Tool | Component | Side effect | Auth scope | Human approval / write mode |");
  a("|---|---|---|---|---|");
  for (const t of spec.tools) {
    const gate = t.requires_approval ? "approval required" : t.write_mode ? `writes ${t.write_mode} only` : "n/a (read)";
    a(`| ${t.name} | ${t.component} | ${t.side_effect} | \`${t.auth_scope}\` | ${gate} |`);
  }
  a("");
  a("## 5. Prompt management");
  a("");
  a("| Prompt | Version | Owner | Component | Path | Eval suite |");
  a("|---|---|---|---|---|---|");
  for (const p of spec.prompts) a(`| ${p.id} | ${p.version} | ${p.owner} | ${p.component} | \`${p.path}\` | \`${p.eval_suite}\` |`);
  a("");
  for (const rule of [
    "Prompts live in the prompt registry (Git, signed tags) and are versioned with semver: major = behaviour change, minor = wording, patch = typo.",
    "Production pins a version (`PROMPT_VERSION_PIN`); no prompt change reaches production without passing its eval suite in staging.",
    "Every change runs shadow, then canary with latency, cost and quality gates (see rollout plan); rollback = re-pin the previous version.",
    "The prompt owner approves wording changes; the AI Delivery Team approves structural and model changes.",
    "Record the prompt version and model id with every draft in the audit store.",
  ]) a(`- ${rule}`);
  a("");
  a("## 6. Integrations");
  a("");
  a("| System | Protocol | Auth | Direction | Limits |");
  a("|---|---|---|---|---|");
  const names = nameOf(spec);
  for (const i of spec.integrations) a(`| ${names.get(i.system) ?? i.system} | ${i.protocol} | ${i.auth} | ${i.direction} | ${i.limits} |`);
  a("");
  a("## 7. Guardrails");
  a("");
  a("| Id | Type | Control | Implemented in |");
  a("|---|---|---|---|");
  for (const g of spec.guardrails) a(`| ${g.id} | ${g.type} | ${g.description} | ${g.component} |`);
  a("");
  a("## 8. Error handling");
  a("");
  a("| Condition | Detection | Handling |");
  a("|---|---|---|");
  for (const [cond, det, hand] of ERROR_HANDLING) a(`| ${cond} | ${det} | ${hand} |`);
  a("");
  a("## 9. Rollout plan");
  a("");
  a("| Phase | Traffic | Entry criteria | Exit criteria | Rollback trigger |");
  a("|---|---|---|---|---|");
  for (const r of spec.rollout) a(`| ${r.phase} | ${r.traffic} | ${r.entry} | ${r.exit} | ${r.rollback} |`);
  a("");
  a("## 10. Observability");
  a("");
  a("| SLO | Target | Alert |");
  a("|---|---|---|");
  for (const slo of spec.slos) a(`| ${slo.id} ${slo.name} | ${slo.target} | ${slo.alert} |`);
  a("");
  a("Dashboards: " + spec.operations.dashboards.join("; ") + ".");
  a("");
  return L.join("\n");
}

function renderRunbook(spec: Spec, rb: RunbookDraft): string {
  const comp = new Map(spec.components.map((c) => [c.id, c]));
  const entry = new Map(rb.entries.map((e) => [e.alert_id, e]));
  const sloByAlert = new Map(spec.slos.filter((s) => s.alert).map((s) => [s.alert!, s]));
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  a(`# Operational runbook - ${spec.system.name}`);
  a("");
  a(`**On-call:** ${spec.operations.on_call}`);
  a("");
  a(
    "Flow for every alert: acknowledge -> diagnose (steps below) -> remediate with the safest reversible action -> " +
      "escalate if not mitigated in 30 minutes -> postmortem if the error budget impact is above 20%.",
  );
  a("");
  a("| Alert | Severity | Component | Owner |");
  a("|---|---|---|---|");
  for (const al of spec.alerts) a(`| ${al.id} ${al.name} | ${al.severity.toUpperCase()} | ${al.component} | ${comp.get(al.component)!.owner} |`);
  a("");
  for (const al of spec.alerts) {
    const e = entry.get(al.id);
    a(`### ${al.id} - ${al.name} (${al.severity.toUpperCase()})`);
    a("");
    a(`- **Condition:** ${al.condition}`);
    a(`- **Component / owner:** ${al.component} / ${comp.get(al.component)!.owner}`);
    const s = sloByAlert.get(al.id);
    if (s) a(`- **Protects:** ${s.id} ${s.name} (${s.target})`);
    if (!e) {
      a("- **MISSING RUNBOOK ENTRY**");
      a("");
      continue;
    }
    a("");
    a("**Likely causes**");
    a("");
    for (const x of e.likely_causes) a(`- ${x}`);
    a("");
    a("**Diagnosis**");
    a("");
    e.diagnosis_steps.forEach((x, i) => a(`${i + 1}. ${x}`));
    a("");
    a("**Remediation**");
    a("");
    e.remediation_steps.forEach((x, i) => a(`${i + 1}. ${x}`));
    a("");
    a(`**Escalation:** ${e.escalation}`);
    a("");
  }
  a("_Diagnosis and remediation drafted by Claude from the spec; alert metadata, owners and SLO links rendered from the spec._");
  a("");
  return L.join("\n");
}

type Check = [string, boolean, string];

function renderHandoff(spec: Spec, checks: Check[]): string {
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  const done = checks.filter(([, ok]) => ok).length;
  a(`# Handoff checklist - ${spec.system.name} v${spec.system.version}`);
  a("");
  a(`**Status:** ${done === checks.length ? "READY" : "NOT READY"} (${done}/${checks.length} items complete)`);
  a("");
  for (const [label, ok, evidence] of checks) a(`- [${ok ? "x" : " "}] ${label} - _${evidence}_`);
  a("");
  a("Knowledge transfer: " + spec.operations.kt_sessions.join("; ") + ".");
  a("");
  return L.join("\n");
}

function renderLintReport(draft: Finding[], fixed: Finding[], fixes: Fixes, runbookMissing: string[]): string {
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  const count = (fs_: Finding[], sev: string): number => fs_.filter((x) => x.severity === sev).length;
  a("# Doc-lint report");
  a("");
  a("| Rule | Description |");
  a("|---|---|");
  for (const [k, v] of Object.entries(RULES)) a(`| ${k} | ${v} |`);
  a("");
  a(`## Draft spec: ${count(draft, "ERROR")} errors, ${count(draft, "WARN")} warnings - publication blocked`);
  a("");
  a("| Rule | Severity | Ref | Finding |");
  a("|---|---|---|---|");
  for (const x of draft) a(`| ${x.rule} | ${x.severity} | ${x.ref} | ${x.message} |`);
  a("");
  a(`## Fixes applied - ${fixes.description}`);
  a("");
  for (const op of fixes.operations) a(`- \`${op.op} ${op.path}\` - ${op.reason}`);
  a("");
  a(`## Fixed spec: ${count(fixed, "ERROR")} errors, ${count(fixed, "WARN")} warnings`);
  a("");
  a(`Runbook coverage (DOC-011): ${runbookMissing.length ? "missing " + runbookMissing.join(", ") : "complete"}.`);
  a("");
  return L.join("\n");
}

const lineCount = (s: string): number => s.split("\n").length - (s.endsWith("\n") ? 1 : 0);
const listOrNone = (xs: string[]): string => (xs.length ? `[${xs.map((x) => `'${x}'`).join(", ")}]` : "none");

async function main(): Promise<void> {
  banner("D6.4 - Architecture Documentation and Implementation Guidance");
  const spec: Spec = JSON.parse(fs.readFileSync(path.join(DATA, "architecture_spec.json"), "utf-8"));
  const fixes: Fixes = JSON.parse(fs.readFileSync(path.join(DATA, "spec_fixes.json"), "utf-8"));
  const client = getClient(responder);

  // -- 1. Spec ----------------------------------------------------------------
  section("1. Machine-readable architecture spec (single source of truth)");
  const s = spec.system;
  console.log(`  ${s.name} v${s.version} (${s.status}), owner ${s.owner}`);
  const keys = [
    "people", "external_systems", "components", "models", "prompts", "tools", "data_stores",
    "integrations", "guardrails", "slos", "alerts", "request_flow", "rollout",
  ] as const;
  for (const key of keys) console.log(`    ${key.padEnd(17)} ${String(spec[key].length).padStart(3)}`);

  // -- 2. Lint draft --------------------------------------------------------------
  section("2. Doc linter on the draft spec (gate before publishing)");
  const draftFindings = lint(spec);
  for (const x of draftFindings) console.log(`  ${x.severity.padEnd(5)} ${x.rule} ${x.ref.padEnd(17)} ${x.message}`);
  const errors = draftFindings.filter((x) => x.severity === "ERROR").length;
  console.log(`  => ${errors} errors: publication BLOCKED (docs are built like code - lint must pass)`);

  // -- 3. Fix + re-lint -----------------------------------------------------------
  section("3. Apply the reviewed patch and re-lint");
  const fixed: Spec = structuredClone(spec);
  for (const op of fixes.operations) {
    applyFix(fixed, op);
    console.log(`  ${op.op.padEnd(6)} ${op.path}`);
  }
  const fixedFindings = lint(fixed);
  for (const x of fixedFindings) console.log(`  ${x.severity.padEnd(5)} ${x.rule} ${x.ref.padEnd(17)} ${x.message}`);
  const fixedErrors = fixedFindings.filter((x) => x.severity === "ERROR").length;
  console.log(`  => ${fixedErrors} errors, ${fixedFindings.length - fixedErrors} warnings: ${fixedErrors === 0 ? "PASS" : "FAIL"}`);
  if (fixedErrors) throw new Error("spec still fails lint");

  // -- 4. Claude narrative ------------------------------------------------------
  section("4. Claude drafts narrative sections (structured output)");
  const specJson = JSON.stringify(fixed, null, 1);
  const nresp = await client.messages.parse({
    model: MODEL_OPUS,
    max_tokens: 4000,
    thinking: { type: "adaptive" },
    output_config: { effort: "medium", format: zodOutputFormat(NarrativeDraft) },
    system: NARRATIVE_SYSTEM,
    messages: [
      {
        role: "user",
        content: "Architecture spec (JSON):\n" + specJson + "\n\nRelated decision: ADR-0007 (long-context grounding with prompt caching).",
      },
    ],
  });
  const nar = nresp.parsed_output;
  if (!nar) throw new Error("narrative did not parse");
  const compIds = fixed.components.map((c) => c.id);
  const got = nar.container_responsibilities.map((r) => r.component_id);
  console.log(`  narrative | ${usageLine(nresp)}`);
  console.log(
    `  coverage: ${compIds.filter((c) => got.includes(c)).length}/${compIds.length} components described; ` +
      `unknown ids: ${listOrNone(got.filter((x) => !compIds.includes(x)))}; key decisions: ${nar.key_decisions.length}`,
  );

  // Spec-driven, well-bounded drafting: Sonnet is enough, and validation enforces completeness.
  const rresp = await client.messages.parse({
    model: MODEL_SONNET,
    max_tokens: 4000,
    thinking: { type: "adaptive" },
    output_config: { effort: "medium", format: zodOutputFormat(RunbookDraft) },
    system: RUNBOOK_SYSTEM,
    messages: [{ role: "user", content: "Architecture spec (JSON):\n" + specJson }],
  });
  const rb = rresp.parsed_output;
  if (!rb) throw new Error("runbook did not parse");
  const alertIds = fixed.alerts.map((al) => al.id);
  const rbIds = rb.entries.map((e) => e.alert_id);
  const runbookMissing = alertIds.filter((x) => !rbIds.includes(x));
  const thin = rb.entries.filter((e) => e.diagnosis_steps.length < 2 || !e.remediation_steps.length).map((e) => e.alert_id);
  console.log(`  runbook   | ${usageLine(rresp)}`);
  console.log(
    `  DOC-011 runbook coverage: ${alertIds.length - runbookMissing.length}/${alertIds.length} alerts; ` +
      `missing=${listOrNone(runbookMissing)}; thin entries=${listOrNone(thin)}`,
  );

  // -- 5. Generate docs -----------------------------------------------------------
  section("5. Render docs from templates (deterministic)");
  const checkResults: Record<string, boolean> = {};
  for (const r of Object.keys(RULES)) checkResults[r] = !fixedFindings.some((x) => x.rule === r && x.severity === "ERROR");
  checkResults["DOC-011"] = runbookMissing.length === 0;
  const ops = fixed.operations;
  const checks: Check[] = [
    ["Every component has an accountable owner", checkResults["DOC-001"]!, "DOC-001"],
    ["All dependencies resolve (no phantom components)", checkResults["DOC-002"]!, "DOC-002"],
    ["Every tool has a least-privilege auth scope", checkResults["DOC-003"]!, "DOC-003"],
    ["No write tool bypasses human approval", checkResults["DOC-004"]!, "DOC-004"],
    ["Every SLO is backed by an alert", checkResults["DOC-005"]!, "DOC-005"],
    ["Every alert has a runbook entry", checkResults["DOC-011"]!, "DOC-011"],
    ["Every prompt is versioned with an eval suite", checkResults["DOC-007"]!, "DOC-007"],
    ["PII data stores define retention, encryption and residency", checkResults["DOC-008"]!, "DOC-008"],
    ["Secrets are sourced from the vault", checkResults["DOC-012"]!, "DOC-012"],
    ["On-call rotation defined", Boolean(ops.on_call), ops.on_call],
    ["Dashboards cover SLOs, cost and guardrails", ops.dashboards.length >= 3, `${ops.dashboards.length} dashboards`],
    ["Every rollout phase has exit and rollback criteria", fixed.rollout.every((r) => Boolean(r.exit && r.rollback)), `${fixed.rollout.length} phases`],
    ["Knowledge-transfer sessions completed", ops.kt_sessions.every((k) => k.includes("(done)")), `${ops.kt_sessions.length} sessions`],
    ["Operations acceptance sign-off recorded", ops.acceptance_signoff !== null, ops.acceptance_signoff ?? "pending - Claims Platform lead"],
  ];
  const docs: [string, string][] = [
    ["00-lint-report.md", renderLintReport(draftFindings, fixedFindings, fixes, runbookMissing)],
    ["01-architecture-overview.md", renderOverview(fixed, nar)],
    ["02-diagrams.md", renderDiagrams(fixed)],
    ["03-implementation-guide.md", renderImplGuide(fixed)],
    ["04-runbook.md", renderRunbook(fixed, rb)],
    ["05-handoff-checklist.md", renderHandoff(fixed, checks)],
  ];
  fs.mkdirSync(DOCS, { recursive: true });
  for (const [name, content] of docs) {
    fs.writeFileSync(path.join(DOCS, name), content, "utf-8");
    console.log(`  wrote labs_output/docs/${name.padEnd(29)} ${String(lineCount(content)).padStart(4)} lines`);
  }
  fs.writeFileSync(path.join(DOCS, "architecture_spec.fixed.json"), JSON.stringify(fixed, null, 2) + "\n", "utf-8");
  console.log("  wrote labs_output/docs/architecture_spec.fixed.json");

  // -- 6. Coverage of the generated docs -------------------------------------------
  section("6. Coverage check on the generated docs");
  const doc = new Map(docs);
  const diag = doc.get("02-diagrams.md")!;
  const guide = doc.get("03-implementation-guide.md")!;
  const runbook = doc.get("04-runbook.md")!;
  const cov: [string, number, number][] = [
    ["components in container diagram", compIds.filter((c) => diag.includes(`${c}[`) || diag.includes(`${c}{{`)).length, compIds.length],
    ["alerts with a runbook section", alertIds.filter((x) => runbook.includes(`### ${x} `) && !runbook.includes("MISSING RUNBOOK")).length, alertIds.length],
    ["prompts in the registry table", fixed.prompts.filter((p) => guide.includes(`| ${p.id} |`)).length, fixed.prompts.length],
    ["tools with auth scope in the guide", fixed.tools.filter((t) => guide.includes(`| ${t.name} |`)).length, fixed.tools.length],
    ["request-flow steps in sequence diagram", fixed.request_flow.filter((st) => diag.includes(st.message)).length, fixed.request_flow.length],
  ];
  for (const [label, have, total] of cov) console.log(`  [${have === total ? "OK " : "GAP"}] ${have}/${total} ${label}`);
  const done = checks.filter(([, ok]) => ok).length;
  const open = checks.filter(([, ok]) => !ok).map(([lbl]) => lbl);
  console.log(
    `  Handoff checklist: ${done}/${checks.length} complete -> ${done === checks.length ? "READY" : "NOT READY"}` +
      ` (open: ${open.length ? open.join(", ") : "none"})`,
  );
}

await main();
