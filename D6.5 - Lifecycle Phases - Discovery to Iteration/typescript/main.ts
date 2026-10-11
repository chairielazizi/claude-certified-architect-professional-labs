/**
 * D6.5 - Lifecycle Phases - Discovery to Iteration
 *
 * A lifecycle state machine for a Claude solution:
 *
 *     discovery -> design -> build -> evaluation -> handoff -> monitoring -> iteration -> design ...
 *                                         ^   |
 *                                         +---+ (evaluation can send work back to build)
 *
 * Each phase has exit criteria evaluated against an artifact registry, so a gate review is
 * computed (PASSED, or BLOCKED with reasons) rather than argued. The simulated project
 * (data/project_events.json) walks through every phase and shows:
 *
 *   - a FAILED handoff gate (incomplete handoff package) that passes after fixes
 *   - monitoring signals (quality drift, cost creep, latency, model release, deprecation)
 *   - an iteration loop triggered by a model deprecation: impact assessment, API
 *     compatibility scan against the target models (breaking changes from the migration
 *     guide), config fixes, and a regression eval of old vs new model through the shared
 *     Claude client - round 1 finds a regression, a prompt patch fixes it, round 2 passes
 *   - the timeline, gate log and phase durations (labs_output/lifecycle_timeline.md)
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { PRICING, banner, getClient, section, textOf } from "../../shared/typescript/claudeClient.ts";
import type { MockRequest } from "../../shared/typescript/claudeClient.ts";

const LAB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA = path.join(LAB, "data");
const OUT = path.join(LAB, "labs_output");

const load = <T>(name: string): T => JSON.parse(fs.readFileSync(path.join(DATA, name), "utf-8")) as T;

const MOCK_OUTPUTS = load<Record<string, string>>("mock_responses/regression_outputs.json");

/** Offline stand-in: canned letters per (model, prompt version, case). */
function responder(req: MockRequest): string {
  const pv = /Prompt version: (\d+\.\d+\.\d+)/.exec(req.systemText);
  const kase = /Case: (C-\d+)/.exec(req.lastUserText);
  if (pv && kase) {
    for (const key of [`${req.model}|${pv[1]}|${kase[1]}`, `${req.model}|*|${kase[1]}`]) {
      if (key in MOCK_OUTPUTS) return MOCK_OUTPUTS[key]!;
    }
  }
  return "Dear Customer,\n\n(no canned reply)\n";
}

// ---------------------------------------------------------------------------
// Small helpers (mirrored in Python)
// ---------------------------------------------------------------------------
type Json = Record<string, unknown>;
const MISSING = Symbol("missing");

function getPath(d: Json, dotted: string): unknown {
  let cur: unknown = d;
  for (const part of dotted.split(".")) {
    if (cur === null || typeof cur !== "object" || Array.isArray(cur) || !(part in (cur as Json))) return MISSING;
    cur = (cur as Json)[part];
  }
  return cur;
}

const g = (v: number): string => String(Number(v.toPrecision(6)));
const commas = (s: string): string => s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
function fixedC(x: number, d: number): string {
  const [i, f] = x.toFixed(d).split(".");
  return commas(i!) + (f !== undefined ? "." + f : "");
}
const w2 = (n: number): string => String(n).padStart(2, "0");

function fmtVal(metric: string, v: unknown): string {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "boolean") return v ? "yes" : "no";
  const n = v as number;
  if (metric.endsWith("coverage") || metric.endsWith("rate")) return `${(n * 100).toFixed(0)}%`;
  if (metric === "cost_per_task") return `$${n.toFixed(3)}`;
  return Number.isInteger(n) ? String(n) : g(n);
}

/** Same text as Python's json.dumps (", " and ": " separators). */
function pyJson(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(pyJson).join(", ")}]`;
  if (v !== null && typeof v === "object") {
    return `{${Object.entries(v as Json).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(", ")}}`;
  }
  return JSON.stringify(v);
}

const wordsOf = (text: string): number => (text.match(/\S+/g) ?? []).length;
const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// ---------------------------------------------------------------------------
// Data shapes
// ---------------------------------------------------------------------------
interface Criterion { id: string; text: string; artifact: string; check: string; metric?: string; value?: number }
interface Phase { id: string; name: string; goal: string; reviewers: string[]; artifacts: string[]; exit: Criterion[] }
interface Lifecycle { project: string; start_date: string; note: string; transitions: Record<string, string[]>; phases: Phase[] }
interface Event { week: number; type: string; [k: string]: unknown }
interface Package {
  runbook: { alerts: string[]; covered: string[] };
  eval: { suite: string; cases: number; baseline_recorded: boolean };
  dashboards: string[];
  owners: Record<string, string | null>;
  prompt_registry: { pinned: boolean; prompts: Record<string, string> };
  model_inventory: { route: string; model: string; deprecation_watch: boolean }[];
  ops_signoff: boolean;
}
interface PackageFile {
  v1: Package;
  fix: { runbook_covered_add: string[]; dashboards_add: string[]; owners: Record<string, string>; deprecation_watch: boolean; ops_signoff: boolean };
  delta: { model_inventory: Package["model_inventory"]; prompt_registry: Record<string, string> };
}
interface Monitoring {
  signals: { id: string; name: string; rule: string; action: string }[];
  thresholds: { quality_drop: number; quality_watch: number; cost_trigger: number; cost_watch: number; p95_max_s: number; deprecation_horizon_days: number };
  weeks: { week: number; quality: number; cost_per_task: number; p95_s: number; note?: string }[];
  external_events: { week: number; type: string; models?: string[]; successor_of?: Record<string, string>; model?: string; retirement_date?: string }[];
}
interface Rule { id: string; targets: string[]; when: { path: string; op: string; value?: unknown }; severity: string; message: string; fix: Json | null }
interface Route { route: string; params: Json; monthly_calls: number; tokens: { input: number; cache_read: number; cache_write: number; output: number } }
interface ModelConfig {
  routes: Route[];
  legacy_pricing: Record<string, unknown>;
  migration_targets: Record<string, string>;
  tokenizer_factor: Record<string, number>;
  rules: Rule[];
}
interface Case { id: string; type: string; input: string; must_include: string[]; must_not_include: string[]; max_words: number }
interface Suite {
  prompts: Record<string, string>;
  cases: Case[];
  recorded_baseline: { model: string; prompt: string; recorded_week: number; results: Record<string, boolean> };
}

interface Artifact { id: string; status: string; metrics: Json }
interface Row { week: number; date: string; phase: string; kind: string; text: string }
type CheckResult = [string, boolean, string];
interface GateRecord { week: number; from: string; to: string; passed: boolean; allowed: boolean; results: CheckResult[]; loop: number }
interface Segment { phase: string; start: number; end: number; loop: number }
interface Finding { route: string; rule: string; severity: string; message: string; fix: Json | null; target: string }
interface CaseResult { ok: boolean; fails: string[]; words: number; outTokens: number; cost: number }
interface EvalRound { round: number; prompt: string; model: string; cand: Record<string, CaseResult>; regressions: string[]; improved: string[] }
interface MonRow { week: number; quality: number; roll: number; q: string; cost: number; ratio: number; c: string; p95: number; l: string; note: string; ext: string[] }
interface Impact { rows: [string, string, string, number, number][]; before: number; after: number; blocking: number; tune: number }

// ---------------------------------------------------------------------------
// Lifecycle engine: phases, gates, artifact registry, timeline
// ---------------------------------------------------------------------------
class Project {
  lc: Lifecycle;
  phases: Map<string, Phase>;
  startUtc: number;
  phase = "";
  loop = 1;
  artifacts = new Map<string, Artifact>();
  rows: Row[] = [];
  gates: GateRecord[] = [];
  segments: Segment[] = [];

  constructor(lifecycle: Lifecycle) {
    this.lc = lifecycle;
    this.phases = new Map(lifecycle.phases.map((p) => [p.id, p]));
    const [y, m, d] = lifecycle.start_date.split("-").map(Number);
    this.startUtc = Date.UTC(y!, m! - 1, d!);
  }

  dateOf(week: number): string {
    return new Date(this.startUtc + 7 * (week - 1) * 86_400_000).toISOString().slice(0, 10);
  }

  log(week: number, kind: string, text: string): void {
    this.rows.push({ week, date: this.dateOf(week), phase: this.phase, kind, text });
    console.log(`  W${w2(week)} ${this.dateOf(week)} [${this.phase.padEnd(10)}] ${kind.padEnd(6)} ${text}`);
  }

  enter(phase: string, week: number, first = false): void {
    if (phase === "design" && this.phase === "iteration") this.loop += 1; // iteration re-enters design: a new loop
    if (this.segments.length) this.segments[this.segments.length - 1]!.end = week;
    this.phase = phase;
    this.segments.push({ phase, start: first ? week : week + 1, end: week, loop: this.loop });
  }

  upsert(artId: string, status: string, metrics: Json = {}): void {
    const art = this.artifacts.get(artId) ?? { id: artId, status, metrics: {} };
    art.status = status;
    Object.assign(art.metrics, metrics);
    this.artifacts.set(artId, art);
  }

  check(c: Criterion): CheckResult {
    const art = this.artifacts.get(c.artifact);
    if (!art) return [c.id, false, `${c.artifact} missing`];
    if (c.check === "exists") return [c.id, true, `${c.artifact} present`];
    if (c.check === "approved") return [c.id, art.status === "approved", `${c.artifact} ${art.status}`];
    const metric = c.metric!;
    const val = art.metrics[metric];
    if (c.check === "gte" || c.check === "lte") {
      const n = val as number | undefined;
      const ok = n !== undefined && n !== null && (c.check === "gte" ? n >= c.value! : n <= c.value!);
      const op = c.check === "gte" ? ">=" : "<=";
      return [c.id, ok, `${metric} ${fmtVal(metric, val)} (need ${op} ${fmtVal(metric, c.value)})`];
    }
    return [c.id, val === true, `${metric} = ${fmtVal(metric, val)}`];
  }

  gate(week: number, to: string): boolean {
    const cur = this.phases.get(this.phase)!;
    const allowed = (this.lc.transitions[this.phase] ?? []).includes(to);
    const results = cur.exit.map((c) => this.check(c));
    const passed = allowed && results.every(([, ok]) => ok);
    const nOk = results.filter(([, ok]) => ok).length;
    this.gates.push({ week, from: this.phase, to, passed, allowed, results, loop: this.loop });
    if (passed) {
      this.log(week, "GATE", `${cur.name} -> ${this.phases.get(to)!.name}: PASSED (${nOk}/${results.length} exit criteria)`);
      this.enter(to, week);
    } else {
      const reasons = results.filter(([, ok]) => !ok).map(([cid, , detail]) => `${cid} ${detail}`);
      if (!allowed) reasons.unshift(`transition ${this.phase} -> ${to} not allowed`);
      const toName = this.phases.get(to)?.name ?? to;
      this.log(week, "GATE", `${cur.name} -> ${toName}: BLOCKED (${nOk}/${results.length}) - ` + reasons.join("; "));
    }
    return passed;
  }
}

// ---------------------------------------------------------------------------
// Model-migration tooling: compatibility scan + fixes + cost impact
// ---------------------------------------------------------------------------
/** mode 'migration': check legacy routes against their TARGET model's rules.
 *  mode 'current'  : check each route against the rules of the model it runs on. */
function scan(config: ModelConfig, mode: string): Finding[] {
  const findings: Finding[] = [];
  for (const r of config.routes) {
    const p = r.params;
    const model = p.model as string;
    let target: string;
    let migrating: boolean;
    if (mode === "migration") {
      if (!(model in config.migration_targets)) continue;
      target = config.migration_targets[model]!;
      migrating = true;
    } else {
      target = model;
      migrating = false;
    }
    for (const rule of config.rules) {
      if (!rule.targets.includes("*") && !rule.targets.includes(target)) continue;
      const w = rule.when;
      let hit: boolean;
      if (w.op === "migrating") hit = migrating;
      else {
        const val = getPath(p, w.path);
        hit =
          val !== MISSING &&
          ((w.op === "eq" && val === w.value) || (w.op === "in" && (w.value as unknown[]).includes(val)) || w.op === "present");
      }
      if (hit) findings.push({ route: r.route, rule: rule.id, severity: rule.severity, message: rule.message, fix: rule.fix, target });
    }
  }
  return findings;
}

function applyFixes(config: ModelConfig, findings: Finding[]): void {
  const routes = new Map(config.routes.map((r) => [r.route, r]));
  for (const f of findings) {
    if (f.severity !== "BLOCKS" || !f.fix) continue;
    const p = routes.get(f.route)!.params;
    if (f.fix.set_model) p.model = f.target;
    for (const key of (f.fix.unset as string[] | undefined) ?? []) delete p[key];
    for (const [key, val] of Object.entries((f.fix.set as Json | undefined) ?? {})) p[key] = structuredClone(val);
  }
}

interface Price { input: number; output: number; cacheWrite: number; cacheRead: number }
function priceOf(model: string, config: ModelConfig): Price {
  if (PRICING[model]) return PRICING[model]!;
  const lp = config.legacy_pricing[model] as { input: number; output: number; cache_write: number; cache_read: number };
  return { input: lp.input, output: lp.output, cacheWrite: lp.cache_write, cacheRead: lp.cache_read };
}

function monthlyCost(route: Route, model: string, config: ModelConfig): number {
  const p = priceOf(model, config);
  const t = route.tokens;
  const factor = config.tokenizer_factor[model] ?? 1.0;
  const perCall = ((t.input * p.input + t.cache_read * p.cacheRead + t.cache_write * p.cacheWrite + t.output * p.output) * factor) / 1_000_000;
  return perCall * route.monthly_calls;
}

// ---------------------------------------------------------------------------
// Regression eval through the shared Claude client
// ---------------------------------------------------------------------------
function grade(text: string, c: Case): [boolean, string[], number] {
  const low = text.toLowerCase();
  const n = wordsOf(text);
  const fails = c.must_include.filter((t) => !low.includes(t.toLowerCase())).map((t) => `missing '${t}'`);
  fails.push(
    ...c.must_not_include
      .filter((t) => new RegExp(`(?<![a-z])${escapeRe(t.toLowerCase())}(?![a-z])`).test(low))
      .map((t) => `contains '${t}'`),
  );
  if (n > c.max_words) fails.push(`${n} words > ${c.max_words}`);
  return [fails.length === 0, fails, n];
}

async function runSuite(
  client: Anthropic, suite: Suite, model: string, prompt: string, migrated: boolean, config: ModelConfig,
): Promise<Record<string, CaseResult>> {
  const results: Record<string, CaseResult> = {};
  for (const c of suite.cases) {
    const base: Anthropic.MessageCreateParamsNonStreaming = {
      model,
      max_tokens: 800,
      system: suite.prompts[prompt]!,
      messages: [{ role: "user", content: `Case: ${c.id}\n${c.input}` }],
    };
    // The migrated route config: adaptive thinking, effort low.
    const params: Anthropic.MessageCreateParamsNonStreaming = migrated
      ? { ...base, thinking: { type: "adaptive" }, output_config: { effort: "low" } }
      : base;
    const resp = await client.messages.create(params);
    const [ok, fails, n] = grade(textOf(resp), c);
    const p = priceOf(model, config);
    const cost = (resp.usage.input_tokens * p.input + resp.usage.output_tokens * p.output) / 1_000_000;
    results[c.id] = { ok, fails, words: n, outTokens: resp.usage.output_tokens, cost };
  }
  return results;
}

async function main(): Promise<void> {
  banner("D6.5 - Lifecycle Phases - Discovery to Iteration");
  const lifecycle = load<Lifecycle>("lifecycle.json");
  const events = load<{ events: Event[] }>("project_events.json").events;
  const pkgFile = load<PackageFile>("handoff_package.json");
  const monitoring = load<Monitoring>("monitoring.json");
  const config = load<ModelConfig>("model_config.json");
  const suite = load<Suite>("regression_suite.json");
  const client = getClient(responder);

  section("1. Lifecycle model: phases, exit criteria, transitions");
  for (const p of lifecycle.phases) {
    const nxt = lifecycle.transitions[p.id]!.join(", ");
    console.log(`  ${p.name.padEnd(18)} -> ${nxt.padEnd(18)} exit: ` + p.exit.map((c) => `${c.id} ${c.text}`).join("; "));
  }
  console.log("  Gates are computed from the artifact registry: evidence, not opinion.");

  section("2. Simulated project timeline");
  const proj = new Project(lifecycle);
  const pkg: Package = structuredClone(pkgFile.v1);
  const pkgHistory: [string, Json][] = [];
  const monRows: MonRow[] = [];
  let baseline: { quality: number; cost: number } | null = null;
  let ticketReasons: string[] = [];
  const qualityHist: number[] = [];
  let impact: Impact | null = null;
  let scanBefore: Finding[] = [];
  let scanAfter: Finding[] = [];
  const configBefore: Route[] = structuredClone(config.routes);
  const evalRounds: EvalRound[] = [];
  let frozenBaseline: Record<string, CaseResult> | null = null;
  let baselineSource = "";
  const th = monitoring.thresholds;

  const packageMetrics = (p: Package): Json => {
    const { alerts, covered } = p.runbook;
    return {
      runbook_coverage: alerts.filter((a) => covered.includes(a)).length / alerts.length,
      baseline_recorded: p.eval.baseline_recorded,
      dashboards: p.dashboards.length,
      owners_missing: Object.values(p.owners).filter((v) => !v).length,
      prompt_registry_pinned: p.prompt_registry.pinned,
      model_inventory_ok: p.model_inventory.every((m) => m.deprecation_watch),
      ops_signoff: p.ops_signoff,
    };
  };

  for (const ev of events) {
    const wk = ev.week;
    const note = ev.note as string | undefined;
    switch (ev.type) {
      case "start":
        proj.enter(ev.phase as string, wk, true);
        proj.log(wk, "START", note!);
        break;
      case "artifact":
        proj.upsert(ev.id as string, ev.status as string, (ev.metrics as Json | undefined) ?? {});
        proj.log(wk, "ART", `${ev.id} (${ev.status}): ${note}`);
        break;
      case "gate":
        proj.gate(wk, ev.to as string);
        break;
      case "handoff_package": {
        const met = packageMetrics(pkg);
        pkgHistory.push(["v1 (submitted)", met]);
        proj.upsert("handoff_package", "submitted", met);
        const missing = pkg.runbook.alerts.filter((a) => !pkg.runbook.covered.includes(a));
        proj.log(
          wk, "ART",
          `handoff_package v1: ${note} (runbook missing ${missing.join(", ")}; ${met.dashboards} dashboards; ${met.owners_missing} owner missing)`,
        );
        break;
      }
      case "handoff_fix": {
        const fx = pkgFile.fix;
        pkg.runbook.covered.push(...fx.runbook_covered_add);
        pkg.dashboards.push(...fx.dashboards_add);
        Object.assign(pkg.owners, fx.owners);
        for (const m of pkg.model_inventory) m.deprecation_watch = fx.deprecation_watch;
        pkg.ops_signoff = fx.ops_signoff;
        const met = packageMetrics(pkg);
        pkgHistory.push(["v2 (after fixes)", met]);
        proj.upsert("handoff_package", "approved", met);
        proj.log(wk, "ART", `handoff_package v2: ${note}`);
        break;
      }
      case "handoff_delta": {
        const dl = pkgFile.delta;
        pkg.model_inventory = structuredClone(dl.model_inventory);
        Object.assign(pkg.prompt_registry.prompts, dl.prompt_registry);
        const met = packageMetrics(pkg);
        pkgHistory.push(["v3 (migration delta)", met]);
        proj.upsert("handoff_package", "approved", met);
        proj.log(wk, "ART", `handoff_package v3: ${note} (inventory: ` + pkg.model_inventory.map((m) => `${m.route}=${m.model}`).join(", ") + ")");
        break;
      }
      case "monitoring_week": {
        if (!baseline) {
          const er = proj.artifacts.get("eval_report")!.metrics;
          baseline = { quality: er.pass_rate as number, cost: er.cost_per_task as number };
          proj.log(wk, "NOTE", `Monitoring baseline from evaluation: quality ${baseline.quality.toFixed(3)}, cost/task $${baseline.cost.toFixed(3)}`);
        }
        const row = monitoring.weeks.find((x) => x.week === wk)!;
        qualityHist.push(row.quality);
        const last2 = qualityHist.slice(-2);
        const roll = last2.reduce((a, b) => a + b, 0) / last2.length;
        const qStat = roll < baseline.quality - th.quality_drop ? "TRIGGER" : row.quality < baseline.quality - th.quality_watch ? "WATCH" : "OK";
        const ratio = row.cost_per_task / baseline.cost;
        const cStat = ratio > th.cost_trigger ? "TRIGGER" : ratio > th.cost_watch ? "WATCH" : "OK";
        const lStat = row.p95_s > th.p95_max_s ? "INCIDENT" : "OK";
        const ext: string[] = [];
        const triggers: string[] = [];
        if (qStat === "TRIGGER") triggers.push(`SIG-Q quality drift (2-week mean ${roll.toFixed(3)})`);
        if (cStat === "TRIGGER") triggers.push(`SIG-C cost x${ratio.toFixed(2)}`);
        const inUse = config.routes.map((r) => r.params.model as string);
        for (const e of monitoring.external_events) {
          if (e.week !== wk) continue;
          if (e.type === "deprecation_notice" && inUse.includes(e.model!)) {
            const [y, m, d] = e.retirement_date!.split("-").map(Number);
            const [y0, m0, d0] = proj.dateOf(wk).split("-").map(Number);
            const days = Math.round((Date.UTC(y!, m! - 1, d!) - Date.UTC(y0!, m0! - 1, d0!)) / 86_400_000);
            const status = days <= th.deprecation_horizon_days ? "TRIGGER" : "WATCH";
            ext.push(`SIG-D ${e.model} retires ${e.retirement_date} (${days} days) [${status}]`);
            if (status === "TRIGGER") triggers.push(`SIG-D ${e.model} retires ${e.retirement_date} (${days} days): mandatory migration`);
          } else if (e.type === "model_release") {
            const succ = Object.entries(e.successor_of!).filter(([, old]) => inUse.includes(old)).map(([nw, old]) => `${nw} (successor of ${old})`);
            if (succ.length) {
              ext.push(`SIG-R released: ${succ.join(", ")} [EVALUATE]`);
              triggers.push("SIG-R successor models released: evaluate upgrade (optional)");
            }
          }
        }
        monRows.push({
          week: wk, quality: row.quality, roll, q: qStat, cost: row.cost_per_task, ratio, c: cStat, p95: row.p95_s, l: lStat,
          note: row.note ?? "", ext,
        });
        proj.log(
          wk, "SIGNAL",
          `quality ${row.quality.toFixed(3)} [${qStat}] cost $${row.cost_per_task.toFixed(3)} x${ratio.toFixed(2)} [${cStat}] ` +
            `p95 ${row.p95_s.toFixed(1)}s [${lStat}]` + (row.note ? ` - ${row.note}` : ""),
        );
        for (const x of ext) proj.log(wk, "SIGNAL", x);
        const mandatory = triggers.filter((t) => !t.includes("SIG-R"));
        if (mandatory.length && !proj.artifacts.has("iteration_ticket")) {
          if (cStat === "WATCH") triggers.push(`SIG-C cost watch x${ratio.toFixed(2)} (fold into the same iteration)`);
          ticketReasons = triggers;
          proj.upsert("iteration_ticket", "open", { reasons: triggers.length });
          proj.log(wk, "EVENT", "Iteration ticket ITR-001 raised: " + triggers.join(" | "));
        }
        break;
      }
      case "impact_assessment": {
        const pre = scan(config, "migration");
        const rows: Impact["rows"] = config.routes.map((r) => {
          const old = r.params.model as string;
          const nw = config.migration_targets[old] ?? old;
          return [r.route, old, nw, monthlyCost(r, old, config), monthlyCost(r, nw, config)];
        });
        const before = rows.reduce((a, x) => a + x[3], 0);
        const after = rows.reduce((a, x) => a + x[4], 0);
        impact = {
          rows, before, after,
          blocking: pre.filter((f) => f.severity === "BLOCKS").length, tune: pre.filter((f) => f.severity === "TUNE").length,
        };
        proj.upsert("impact_assessment", "approved", { blocking: impact.blocking });
        proj.log(
          wk, "ART",
          `impact_assessment: ${impact.blocking} breaking changes, ${impact.tune} tuning tasks; monthly model cost $${fixedC(before, 2)} -> $${fixedC(after, 2)}`,
        );
        for (const [route, old, nw, cb, ca] of rows) {
          console.log(`           ${route.padEnd(15)} ${old.padEnd(17)} -> ${nw.padEnd(18)} $${fixedC(cb, 2).padStart(8)} -> $${fixedC(ca, 2).padStart(8)}/month`);
        }
        break;
      }
      case "compat_scan": {
        const findings = scan(config, ev.mode as string);
        const blocking = findings.filter((f) => f.severity === "BLOCKS").length;
        const tune = findings.filter((f) => f.severity === "TUNE").length;
        if (ev.mode === "migration") scanBefore = findings;
        proj.upsert("compat_scan", "approved", { blocking, tune });
        proj.log(wk, "SCAN", `${note}: ${blocking} blocking, ${tune} tuning`);
        for (const f of findings) console.log(`           ${f.severity.padEnd(6)} ${f.route.padEnd(15)} ${f.rule.padEnd(8)} ${f.message}`);
        break;
      }
      case "apply_migration_fixes": {
        applyFixes(config, scan(config, "migration"));
        scanAfter = scan(config, "current");
        const blocking = scanAfter.filter((f) => f.severity === "BLOCKS").length;
        const tune = scanAfter.filter((f) => f.severity === "TUNE").length;
        proj.upsert("compat_scan", "approved", { blocking, tune });
        proj.log(wk, "SCAN", `${note}; re-scan: ${blocking} blocking, ${tune} tuning task(s) left`);
        configBefore.forEach((oldR, i) => {
          console.log(`           ${oldR.route}: ${pyJson(oldR.params)}`);
          console.log(`           ${" ".repeat(oldR.route.length)}  -> ${pyJson(config.routes[i]!.params)}`);
        });
        break;
      }
      case "regression_eval": {
        const route = config.routes.find((r) => r.route === "draft_letter")!;
        const newModel = route.params.model as string;
        const legacyModel = suite.recorded_baseline.model;
        if (frozenBaseline === null) {
          // Freeze the reference ONCE: later rounds compare against the same baseline outputs.
          try {
            frozenBaseline = await runSuite(client, suite, legacyModel, suite.recorded_baseline.prompt, false, config);
            baselineSource = `${legacyModel} @ prompt ${suite.recorded_baseline.prompt} (re-run)`;
          } catch (err) {
            if (!(err instanceof Anthropic.APIError)) throw err; // e.g. the legacy model is already retired (404)
            const rec = suite.recorded_baseline;
            frozenBaseline = Object.fromEntries(
              Object.entries(rec.results).map(([cid, ok]) => [cid, { ok, fails: ok ? [] : ["recorded failure"], words: 0, outTokens: 0, cost: 0 }]),
            );
            baselineSource = `recorded W${rec.recorded_week} results (${err.constructor.name} calling ${legacyModel})`;
          }
        }
        const fb = frozenBaseline;
        const cand = await runSuite(client, suite, newModel, ev.prompt as string, true, config);
        const ids = Object.keys(cand);
        const regressions = ids.filter((c) => fb[c]!.ok && !cand[c]!.ok);
        const improved = ids.filter((c) => !fb[c]!.ok && cand[c]!.ok);
        const nPass = Object.values(cand).filter((c) => c.ok).length;
        const bPass = Object.values(fb).filter((c) => c.ok).length;
        evalRounds.push({ round: ev.round as number, prompt: ev.prompt as string, model: newModel, cand, regressions, improved });
        proj.upsert("eval_report", "approved", { pass_rate: nPass / ids.length, regressions: regressions.length });
        proj.log(
          wk, "EVAL",
          `round ${ev.round}: ${newModel} @ prompt ${ev.prompt} ${nPass}/${ids.length} vs baseline ${bPass}/${ids.length}; ` +
            `regressions: ${regressions.join(", ") || "none"}; improved: ${improved.join(", ") || "none"}` + (note ? ` - ${note}` : ""),
        );
        console.log(`           baseline: ${baselineSource}`);
        console.log(`           ${"case".padEnd(6)}${"type".padEnd(22)}${"baseline".padEnd(10)}${"candidate".padEnd(11)}${"verdict".padEnd(12)}detail`);
        for (const c of suite.cases) {
          const b = fb[c.id]!;
          const r = cand[c.id]!;
          const verdict = regressions.includes(c.id) ? "REGRESSION" : improved.includes(c.id) ? "improved" : "same";
          console.log(
            `           ${c.id.padEnd(6)}${c.type.padEnd(22)}${(b.ok ? "PASS" : "FAIL").padEnd(10)}${(r.ok ? "PASS" : "FAIL").padEnd(11)}` +
              `${verdict.padEnd(12)}${(r.fails.length ? r.fails : b.fails).join("; ")}`,
          );
        }
        const vals = Object.values(cand);
        const avgOut = vals.reduce((a, c) => a + c.outTokens, 0) / vals.length;
        const avgCost = vals.reduce((a, c) => a + c.cost, 0) / vals.length;
        console.log(`           candidate avg output ${avgOut.toFixed(0)} tokens, $${avgCost.toFixed(5)}/case (offline token counts are estimates)`);
        break;
      }
      case "note":
        proj.log(wk, "NOTE", note!);
        break;
    }
  }

  const lastWeek = Math.max(...events.map((e) => e.week));
  proj.segments[proj.segments.length - 1]!.end = lastWeek;

  section("3. Phase timeline (Gantt) and gate summary");
  const weeks = Array.from({ length: lastWeek }, (_, i) => i + 1);
  console.log(`  ${"".padEnd(28)}` + weeks.map((w) => String(w % 10)).join(""));
  for (const s of proj.segments) {
    const label = proj.phases.get(s.phase)!.name + (s.loop > 1 ? ` (loop ${s.loop})` : "");
    console.log(`  ${label.padEnd(28)}` + weeks.map((w) => (s.start <= w && w <= s.end ? "#" : ".")).join("") + `  W${w2(s.start)}-W${w2(s.end)}`);
  }
  const passed = proj.gates.filter((g_) => g_.passed).length;
  console.log(
    `\n  Gates: ${proj.gates.length} reviewed, ${passed} passed, ${proj.gates.length - passed} blocked; ` +
      `iteration loops: ${proj.loop - 1}; current phase: ${proj.phases.get(proj.phase)!.name}`,
  );
  for (const gr of proj.gates) {
    if (!gr.passed) {
      console.log(`  Blocked gate W${w2(gr.week)} ${gr.from} -> ${gr.to}: ` + gr.results.filter(([, ok]) => !ok).map(([cid, , d]) => `${cid} ${d}`).join("; "));
    }
  }

  const report = renderReport({
    lifecycle, proj, pkgHistory, monRows, monitoring, ticketReasons, impact, scanBefore, configBefore, config, scanAfter,
    evalRounds, frozen: frozenBaseline ?? {}, baselineSource, suite, lastWeek,
  });
  fs.mkdirSync(OUT, { recursive: true });
  fs.writeFileSync(path.join(OUT, "lifecycle_timeline.md"), report, "utf-8");
  const lineCount = report.split("\n").length - (report.endsWith("\n") ? 1 : 0);
  console.log(`\n  wrote labs_output/lifecycle_timeline.md (${lineCount} lines); sample in examples/`);
}

interface ReportInput {
  lifecycle: Lifecycle; proj: Project; pkgHistory: [string, Json][]; monRows: MonRow[]; monitoring: Monitoring; ticketReasons: string[];
  impact: Impact | null; scanBefore: Finding[]; configBefore: Route[]; config: ModelConfig; scanAfter: Finding[]; evalRounds: EvalRound[];
  frozen: Record<string, CaseResult>; baselineSource: string; suite: Suite; lastWeek: number;
}

function renderReport(x: ReportInput): string {
  const { lifecycle, proj } = x;
  const L: string[] = [];
  const a = (s: string): void => {
    L.push(s);
  };
  a(`# Lifecycle timeline - ${lifecycle.project} (simulated)`);
  a("");
  a(`> ${lifecycle.note}`);
  a("");
  a("## 1. Phase model");
  a("");
  a("| Phase | Goal | Exit criteria (gate) | Reviewers | Next |");
  a("|---|---|---|---|---|");
  for (const p of lifecycle.phases) {
    a(`| ${p.name} | ${p.goal} | ` + p.exit.map((c) => `${c.id} ${c.text}`).join("<br>") + ` | ${p.reviewers.join(", ")} | ${lifecycle.transitions[p.id]!.join(", ")} |`);
  }
  a("");
  a("## 2. Timeline");
  a("");
  a("| Week | Date | Phase | Type | Event |");
  a("|---|---|---|---|---|");
  for (const r of proj.rows) a(`| W${w2(r.week)} | ${r.date} | ${r.phase} | ${r.kind} | ${r.text.replace(/\|/g, "/")} |`);
  a("");
  a("## 3. Gate reviews");
  a("");
  for (const gr of proj.gates) {
    a(
      `### W${w2(gr.week)} ${proj.phases.get(gr.from)!.name} -> ${proj.phases.get(gr.to)!.name}: ` +
        `${gr.passed ? "PASSED" : "BLOCKED"}${gr.loop > 1 ? ` (loop ${gr.loop})` : ""}`,
    );
    a("");
    a("| Criterion | Result | Evidence |");
    a("|---|---|---|");
    const crit = new Map(proj.phases.get(gr.from)!.exit.map((c) => [c.id, c.text]));
    for (const [cid, ok, detail] of gr.results) a(`| ${cid} ${crit.get(cid)} | ${ok ? "pass" : "**FAIL**"} | ${detail} |`);
    a("");
  }
  a("## 4. Handoff package validation");
  a("");
  const metrics = x.pkgHistory.length ? Object.keys(x.pkgHistory[0]![1]) : [];
  a("| Check | " + x.pkgHistory.map(([label]) => label).join(" | ") + " |");
  a("|---|" + "---|".repeat(x.pkgHistory.length));
  for (const mk of metrics) a(`| ${mk} | ` + x.pkgHistory.map(([, met]) => fmtVal(mk, met[mk])).join(" | ") + " |");
  a("");
  a("## 5. Monitoring signals");
  a("");
  a("| Week | Quality (2-wk mean) | Quality signal | Cost/task (x baseline) | Cost signal | p95 | Latency signal | Notes |");
  a("|---|---|---|---|---|---|---|---|");
  for (const m of x.monRows) {
    const notes = [...(m.note ? [m.note] : []), ...m.ext].join("; ");
    a(
      `| W${w2(m.week)} | ${m.quality.toFixed(3)} (${m.roll.toFixed(3)}) | ${m.q} | $${m.cost.toFixed(3)} (x${m.ratio.toFixed(2)}) | ${m.c} | ` +
        `${m.p95.toFixed(1)} s | ${m.l} | ${notes || "-"} |`,
    );
  }
  a("");
  a("Signal rules: " + x.monitoring.signals.map((s) => `${s.id} ${s.name}: ${s.rule} -> ${s.action}`).join("; ") + ".");
  a("");
  a("Iteration ticket ITR-001 reasons:");
  a("");
  for (const t of x.ticketReasons) a(`- ${t}`);
  a("");
  a("## 6. Iteration: model migration");
  a("");
  a("### Impact assessment");
  a("");
  a("| Route | Current model | Target model | Monthly cost now | Monthly cost after |");
  a("|---|---|---|---:|---:|");
  for (const [route, old, nw, cb, ca] of x.impact?.rows ?? []) a(`| ${route} | \`${old}\` | \`${nw}\` | $${fixedC(cb, 2)} | $${fixedC(ca, 2)} |`);
  if (x.impact) a(`| **Total** | | | **$${fixedC(x.impact.before, 2)}** | **$${fixedC(x.impact.after, 2)}** |`);
  a("");
  a("### API compatibility scan (against the target models)");
  a("");
  a("| Route | Rule | Severity | Finding |");
  a("|---|---|---|---|");
  for (const f of x.scanBefore) a(`| ${f.route} | ${f.rule} | ${f.severity} | ${f.message} |`);
  a("");
  a("Config before -> after the fixes:");
  a("");
  a("```");
  x.configBefore.forEach((oldR, i) => {
    a(`${oldR.route}:`);
    a(`  before: ${pyJson(oldR.params)}`);
    a(`  after:  ${pyJson(x.config.routes[i]!.params)}`);
  });
  a("```");
  a("");
  a("Remaining after the fixes: " + (x.scanAfter.map((f) => `${f.severity} ${f.route} ${f.rule}: ${f.message}`).join("; ") || "none") + ".");
  a("");
  a("### Regression eval (old vs new model)");
  a("");
  a(`Frozen baseline: ${x.baselineSource}. Programmatic checks per case (must include / must not include / word budget).`);
  a("");
  for (const rnd of x.evalRounds) {
    const vals = Object.values(rnd.cand);
    const nPass = vals.filter((c) => c.ok).length;
    a(
      `**Round ${rnd.round}** - \`${rnd.model}\` with prompt ${rnd.prompt}: ${nPass}/${vals.length} pass; ` +
        `regressions: ${rnd.regressions.join(", ") || "none"}; improved: ${rnd.improved.join(", ") || "none"}`,
    );
    a("");
    a("| Case | Type | Baseline | Candidate | Words | Verdict | Detail |");
    a("|---|---|---|---|---:|---|---|");
    for (const c of x.suite.cases) {
      const b = x.frozen[c.id]!;
      const r = rnd.cand[c.id]!;
      const verdict = rnd.regressions.includes(c.id) ? "REGRESSION" : rnd.improved.includes(c.id) ? "improved" : "same";
      a(
        `| ${c.id} | ${c.type} | ${b.ok ? "PASS" : "FAIL"} | ${r.ok ? "PASS" : "FAIL"} | ${r.words} | ` +
          `${verdict} | ${(r.fails.length ? r.fails : b.fails).join("; ") || "-"} |`,
      );
    }
    a("");
  }
  a(
    "Decision: switch only after a round with zero regressions against the frozen baseline; then canary, then remove " +
      "the legacy model ids. Fix regressions in the prompt or config - never by loosening the check.",
  );
  a("");
  a("## 7. Phase durations");
  a("");
  a("| Phase | Loop | Weeks | Duration |");
  a("|---|---:|---|---:|");
  for (const s of proj.segments) a(`| ${proj.phases.get(s.phase)!.name} | ${s.loop} | W${w2(s.start)}-W${w2(s.end)} | ${s.end - s.start + 1} |`);
  a("");
  a("```");
  const weeks = Array.from({ length: x.lastWeek }, (_, i) => i + 1);
  a("".padEnd(28) + weeks.map((w) => String(w % 10)).join(""));
  for (const s of proj.segments) {
    const label = proj.phases.get(s.phase)!.name + (s.loop > 1 ? ` (loop ${s.loop})` : "");
    a(label.padEnd(28) + weeks.map((w) => (s.start <= w && w <= s.end ? "#" : ".")).join(""));
  }
  a("```");
  a("");
  return L.join("\n");
}

await main();
