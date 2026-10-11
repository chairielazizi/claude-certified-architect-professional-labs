/**
 * D2.5 - Prompt Reuse: Caching, Modular Prompts, Skills
 * =====================================================
 *
 * Three reuse strategies for an insurance-claims assistant:
 *
 *   (b) Modular prompts - a versioned prompt library (persona / safety / policy /
 *       reference / format components, semver-pinned per use case), composed in a
 *       deterministic order with dependency checks and a fingerprint.
 *   (a) Prompt caching - exam sample question 2: the same ~7.5K-token system prompt
 *       (handbook + coverage schedule) on every request -> static content first,
 *       cache_control at the end of it. Measured cache writes vs reads and cost,
 *       TTL break-even, and two silent anti-patterns (timestamp in the system
 *       prompt; automatic caching on single-shot requests with a unique tail).
 *       Plus cross-use-case reuse with a breakpoint after the shared prefix.
 *   (c) Agent Skills - SKILL.md folders with YAML frontmatter, a local loader with
 *       validation and progressive disclosure (name+description -> SKILL.md ->
 *       referenced file) driven by real tool use, and the Claude API Skills
 *       request code (container.skills + code execution), live-only.
 */

import Anthropic, { toFile } from "@anthropic-ai/sdk";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  MODEL_OPUS,
  PRICING,
  type MockRequest,
  type ResponderResult,
  banner,
  costUsd,
  getClient,
  isLive,
  section,
  textBlock,
  textOf,
  toolUseBlock,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const LIBRARY_DIR = path.join(DATA, "prompt_library");
const SKILLS_DIR = path.join(DATA, "skills");
const readText = (p: string): string => readFileSync(p, "utf-8").replace(/\r\n/g, "\n");

interface TriageRequest { claim_id: string; text: string; offline_decision: Record<string, unknown> }
interface ClaimsFile {
  triage_requests: TriageRequest[];
  memo_requests: string[];
  memo_offline: string;
  skill_requests: { id: string; text: string }[];
}
interface ScheduleFile {
  version: string;
  products: { product: string; base_deductible: number; base_limit: number; property: boolean }[];
  regions: { code: string; name: string; multiplier_pct: number; cat_surge_pct: number; flood_excluded: boolean }[];
}

const USE_CASES = (JSON.parse(readText(path.join(DATA, "use_cases.json"))) as { use_cases: Record<string, string[]> }).use_cases;
const CLAIMS = JSON.parse(readText(path.join(DATA, "claims.json"))) as ClaimsFile;
const SCHEDULE = JSON.parse(readText(path.join(DATA, "coverage_schedule.json"))) as ScheduleFile;
const MODEL = MODEL_OPUS;
const ONE_HOUR_WRITE_MULT = 2.0; // 1h TTL writes cost 2x base input (5m writes: 1.25x, in PRICING)

type SystemBlocks = Anthropic.TextBlockParam[];

// ---------------------------------------------------------------------------
// Frontmatter (shared by prompt components and SKILL.md files)
// ---------------------------------------------------------------------------
/** Minimal YAML-frontmatter parser for flat `key: value` headers (no external deps). */
function parseFrontmatter(text: string): [Record<string, string>, string] {
  if (!text.startsWith("---\n")) throw new Error("missing frontmatter");
  const end = text.indexOf("\n---\n", 4);
  if (end < 0) throw new Error("unterminated frontmatter");
  const meta: Record<string, string> = {};
  for (const line of text.slice(4, end).split("\n")) {
    if (!line.trim()) continue;
    const idx = line.indexOf(":");
    const key = (idx >= 0 ? line.slice(0, idx) : line).trim();
    const value = (idx >= 0 ? line.slice(idx + 1) : "").trim().replace(/^"+|"+$/g, "");
    meta[key] = value;
  }
  return [meta, text.slice(end + 5).trim()];
}

// ---------------------------------------------------------------------------
// (b) Modular prompt library
// ---------------------------------------------------------------------------
const LAYER_ORDER = ["safety", "policy", "reference", "persona", "format"]; // most shared/stable first

class CompositionError extends Error {}

interface Component {
  id: string;
  version: string;
  layer: string;
  requires: string[];
  summary: string;
  body: string;
  source: string;
}
const refOf = (c: Component): string => `${c.id}@${c.version}`;
const renderComponent = (c: Component): string => `<component id="${c.id}" version="${c.version}">\n${c.body}\n</component>`;

const vtuple = (v: string): number[] => v.split(".").map(Number);
function vcompare(a: string, b: string): number {
  const x = vtuple(a);
  const y = vtuple(b);
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const d = (x[i] ?? 0) - (y[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
}

function matches(version: string, constraint: string): boolean {
  if (constraint === "*") return true;
  if (constraint.startsWith("^")) return version.split(".")[0] === constraint.slice(1);
  return version === constraint;
}

const round5 = (x: number): number => Math.floor((x + 2) / 5) * 5;
const money = (x: number): string => x.toLocaleString("en-US");

/** Reference tables rendered from structured data must be deterministic: same data -> same bytes. */
function renderSchedule(data: ScheduleFile): string {
  const lines = [
    `# Coverage schedule ${data.version} (per-claim deductible and limit by rating region and product)`,
    "| region | product | deductible | limit | CAT surge | flood |",
    "|---|---|---|---|---|---|",
  ];
  for (const r of data.regions) {
    for (const p of data.products) {
      const ded = round5(Math.floor((p.base_deductible * r.multiplier_pct) / 100));
      const lim = Math.floor(Math.floor((p.base_limit * r.multiplier_pct) / 100) / 1000) * 1000;
      const surge = r.cat_surge_pct;
      const flood = !p.property ? "n/a" : r.flood_excluded ? "excluded" : "covered";
      lines.push(
        `| ${r.code} ${r.name} | ${p.product} | $${money(ded)} | $${money(lim)} | ` +
          `x${Math.floor(surge / 100)}.${String(surge % 100).padStart(2, "0")} | ${flood} |`,
      );
    }
  }
  return lines.join("\n");
}

class PromptLibrary {
  readonly components: Component[];
  constructor(components: Component[]) {
    this.components = components;
  }
  static load(folder: string, generated: Component[]): PromptLibrary {
    const comps: Component[] = [];
    for (const name of readdirSync(folder).filter((f) => f.endsWith(".md")).sort()) {
      // sorted: directory order is not guaranteed
      const [meta, body] = parseFrontmatter(readText(path.join(folder, name)));
      const requires = (meta.requires ?? "").split(",").map((r) => r.trim()).filter(Boolean);
      comps.push({ id: meta.id!, version: meta.version!, layer: meta.layer!, requires, summary: meta.summary!, body, source: name });
    }
    return new PromptLibrary([...comps, ...generated]);
  }
  resolve(ref: string): Component {
    const at = ref.indexOf("@");
    const cid = at >= 0 ? ref.slice(0, at) : ref;
    const constraint = at >= 0 ? ref.slice(at + 1) : "*";
    const candidates = this.components.filter((c) => c.id === cid && matches(c.version, constraint || "*"));
    if (!candidates.length) throw new CompositionError(`no version of ${cid} matches '${constraint}'`);
    return candidates.reduce((a, b) => (vcompare(b.version, a.version) > 0 ? b : a));
  }
}

type Breakpoint = "end" | "split" | "none";

class Composition {
  readonly components: Component[];
  constructor(components: Component[]) {
    this.components = components;
  }
  get text(): string {
    return this.components.map(renderComponent).join("\n\n");
  }
  get fingerprint(): string {
    return createHash("sha256").update(this.text, "utf8").digest("hex").slice(0, 12);
  }
  /**
   * System blocks. split = number of leading (shared) components in block 1.
   * breakpoint: 'end' (cache everything), 'split' (cache only the shared prefix), 'none'.
   */
  blocks(split: number | null = null, breakpoint: Breakpoint = "end"): SystemBlocks {
    const n = split === null ? this.components.length : split;
    const parts = [this.components.slice(0, n), this.components.slice(n)];
    const blocks: SystemBlocks = [];
    parts.forEach((group, i) => {
      if (!group.length) return;
      const block: Anthropic.TextBlockParam = { type: "text", text: group.map(renderComponent).join("\n\n") };
      if ((breakpoint === "split" && i === 0) || (breakpoint === "end" && i === parts.length - 1) || (breakpoint === "end" && i === 0 && !parts[1]!.length)) {
        block.cache_control = { type: "ephemeral" };
      }
      blocks.push(block);
    });
    return blocks;
  }
}

function compose(library: PromptLibrary, refs: string[], canonical = true): Composition {
  let comps = refs.map((r) => library.resolve(r));
  const ids = comps.map((c) => c.id);
  const dupes = [...new Set(ids.filter((id) => ids.indexOf(id) !== ids.lastIndexOf(id)))].sort();
  if (dupes.length) throw new CompositionError(`duplicate components: ${dupes.join(", ")}`);
  for (const c of comps) {
    for (const req of c.requires) {
      if (!ids.includes(req)) throw new CompositionError(`${refOf(c)} requires ${req}, which the manifest does not include`);
    }
  }
  if (canonical) {
    // Deterministic order = stable cache key, whatever order the manifest lists things in.
    comps = [...comps].sort((a, b) => LAYER_ORDER.indexOf(a.layer) - LAYER_ORDER.indexOf(b.layer) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }
  return new Composition(comps);
}

function commonPrefix(a: Composition, b: Composition): number {
  let n = 0;
  while (n < Math.min(a.components.length, b.components.length) && refOf(a.components[n]!) === refOf(b.components[n]!)) n++;
  return n;
}

// ---------------------------------------------------------------------------
// (c) Agent Skills: loader, validation, progressive disclosure
// ---------------------------------------------------------------------------
class SkillError extends Error {}

interface Skill {
  name: string;
  description: string;
  folder: string;
  body: string;
}
const skillMd = (s: Skill): string => readText(path.join(s.folder, "SKILL.md"));

const NAME_RE = /^[a-z0-9-]{1,64}$/;

function validateSkillMeta(meta: Record<string, string>, folderName: string): string[] {
  const errors: string[] = [];
  const name = meta.name ?? "";
  const desc = meta.description ?? "";
  if (!NAME_RE.test(name)) errors.push("name must be 1-64 chars of lowercase letters, digits, hyphens");
  if (name.includes("anthropic") || name.includes("claude")) errors.push("name must not contain reserved words 'anthropic' or 'claude'");
  if (name && name !== folderName) errors.push(`name '${name}' must match folder '${folderName}'`);
  if (!desc) errors.push("description is required (it is how the model decides to load the skill)");
  if (desc.length > 1024) errors.push("description must be <= 1024 chars");
  if (/<[^>]+>/.test(desc)) errors.push("description must not contain XML tags");
  return errors;
}

function loadSkills(root: string): Skill[] {
  const skills: Skill[] = [];
  for (const name of readdirSync(root).filter((f) => statSync(path.join(root, f)).isDirectory()).sort()) {
    const folder = path.join(root, name);
    const [meta, body] = parseFrontmatter(readText(path.join(folder, "SKILL.md")));
    const errors = validateSkillMeta(meta, name);
    if (errors.length) throw new SkillError(`${name}: ` + errors.join("; "));
    skills.push({ name: meta.name!, description: meta.description!, folder, body });
  }
  return skills;
}

/** Level 1: only name + description sit in context (~100 tokens per skill). */
function skillsIndex(skills: Skill[]): string {
  const lines = ["<available_skills>", ...skills.map((s) => `<skill name="${s.name}">${s.description}</skill>`), "</available_skills>"];
  lines.push(
    "When a request matches a skill, call load_skill with its name and follow the loaded instructions. " +
      "Load referenced files with read_skill_file only when a step needs them. Otherwise answer directly.",
  );
  return lines.join("\n");
}

/** Level 3 loader with a path-traversal guard: model-supplied paths are untrusted. */
function readSkillFile(skills: Record<string, Skill>, skill: string, relPath: string): string {
  const s = skills[skill];
  if (!s) throw new SkillError(`unknown skill '${skill}'`);
  const base = path.resolve(s.folder);
  const target = path.resolve(base, relPath);
  const rel = path.relative(base, target);
  if (rel === ".." || rel.startsWith(".." + path.sep) || path.isAbsolute(rel)) throw new SkillError(`path '${relPath}' escapes the skill directory`);
  if (!existsSync(target) || !statSync(target).isFile()) throw new SkillError(`no such file '${relPath}' in skill '${skill}'`);
  return readText(target);
}

function filesUnder(dir: string): string[] {
  return (readdirSync(dir, { recursive: true }) as string[])
    .map((rel) => path.join(dir, rel))
    .filter((p) => statSync(p).isFile())
    .sort();
}

function skillTools(names: string[]): Anthropic.Tool[] {
  // Tools render first in the prompt: keep them deterministic (sorted) so they stay cacheable.
  const enumNames = [...names].sort();
  const tools: Anthropic.Tool[] = [
    {
      name: "load_skill",
      strict: true,
      description: "Load the full SKILL.md instructions of one available skill before handling a matching request.",
      input_schema: { type: "object", properties: { name: { type: "string", enum: enumNames } }, required: ["name"], additionalProperties: false },
    },
    {
      name: "read_skill_file",
      strict: true,
      description: "Read a supporting file that a loaded skill references, e.g. references/red_flags.md.",
      input_schema: {
        type: "object",
        properties: { skill: { type: "string", enum: enumNames }, path: { type: "string" } },
        required: ["skill", "path"],
        additionalProperties: false,
      },
    },
  ];
  return tools.sort((a, b) => (a.name < b.name ? -1 : 1));
}

async function runSkillAgent(
  client: Anthropic,
  system: SystemBlocks,
  tools: Anthropic.Tool[],
  skills: Record<string, Skill>,
  userText: string,
  maxTurns = 6,
): Promise<[string, string[]]> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: userText }];
  const trace: string[] = [];
  for (let turn = 1; turn <= maxTurns; turn++) {
    const msg = await client.messages.create({ model: MODEL, max_tokens: 2048, system, tools, messages, output_config: { effort: "medium" } });
    const u = msg.usage;
    const inTok = u.input_tokens + (u.cache_read_input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0);
    if (msg.stop_reason !== "tool_use") {
      trace.push(`turn ${turn}: ${String(msg.stop_reason).padEnd(9)} final answer                                   in=${inTok} tok`);
      return [textOf(msg), trace];
    }
    messages.push({ role: "assistant", content: msg.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const block of msg.content) {
      if (block.type !== "tool_use") continue;
      const input = block.input as Record<string, string>;
      const args = Object.entries(input).map(([k, v]) => `${k}=${v}`).join(", ");
      try {
        let content: string;
        if (block.name === "load_skill") {
          const s = skills[input.name!];
          if (!s) throw new SkillError(`unknown skill '${input.name}'`);
          content = skillMd(s); // level 2
        } else {
          content = readSkillFile(skills, input.skill!, input.path!); // level 3
        }
        results.push({ type: "tool_result", tool_use_id: block.id, content });
      } catch (err) {
        if (!(err instanceof SkillError)) throw err;
        results.push({ type: "tool_result", tool_use_id: block.id, content: err.message, is_error: true });
      }
      trace.push(`turn ${turn}: tool_use  ${block.name}(${args})`.padEnd(66) + ` in=${inTok} tok`);
    }
    messages.push({ role: "user", content: results }); // all results in ONE user message
  }
  return ["(stopped: max turns)", trace];
}

// ---------------------------------------------------------------------------
// Offline responder
// ---------------------------------------------------------------------------
const FRAUD_REPORT =
  "Score: 55/100 -> refer to SIU\n" +
  "Flags: RF-01 (20), RF-02 (20), RF-04 (15)\n" +
  "Information needed: camera purchase date (RF-03), police report reference and filing time (RF-08), " +
  "prior claims history (RF-07)\n" +
  "Handling note: refer under H9.2 and H3.1 (loss in the first 14 days); request proof of ownership " +
  "neutrally and do not mention fraud to the claimant.";

function responder(req: MockRequest): ResponderResult {
  const user = req.lastUserText;
  if (req.toolNames.includes("load_skill")) {
    const results = req.toolResults;
    if (!results.length) {
      if (user.includes("suspicious") || user.includes("fraud")) {
        return [textBlock("This matches the fraud-screening skill; loading it."), toolUseBlock("load_skill", { name: "claims-fraud-screening" })];
      }
      return "Motor claims must be notified within 7 days of the loss (handbook H2.1).";
    }
    const last = String(results[results.length - 1]!.content ?? "");
    if (last.includes("## Procedure")) return [toolUseBlock("read_skill_file", { skill: "claims-fraud-screening", path: "references/red_flags.md" })];
    if (last.includes("RED FLAG CATALOGUE")) return FRAUD_REPORT;
    return "I could not load the skill; proceeding with the standard handbook checks.";
  }
  if (req.systemText.includes("persona.underwriting_analyst")) return CLAIMS.memo_offline;
  for (const c of CLAIMS.triage_requests) if (user.includes(c.claim_id)) return JSON.stringify(c.offline_decision);
  return "{}";
}

// ---------------------------------------------------------------------------
// Live-only: Agent Skills on the Claude API
// ---------------------------------------------------------------------------
const SKILLS_API_SNIPPET = `\
const skillDir = "data/skills/claims-fraud-screening";
const files = await Promise.all(filesUnder(skillDir).map((p) =>
  toFile(readFileSync(p), "claims-fraud-screening/" + path.relative(skillDir, p).split(path.sep).join("/"), { type: "text/markdown" })));
const skill = await client.skills.create({ files, display_name: "Claims fraud screening" }); // Skills API (no beta header)

const resp = await client.beta.messages.create({
  model: "claude-opus-5-5", max_tokens: 16000,
  betas: ["code-execution-2025-08-25"],
  container: { skills: [
    { type: "custom", skill_id: skill.id, version: "latest" },
    { type: "anthropic", skill_id: "xlsx", version: "latest" },   // Anthropic-managed skill
  ] },
  tools: [{ type: "code_execution_20260521", name: "code_execution" }], // skills run in the container
  messages: [{ role: "user", content: "Screen claim CLM-7731 for fraud and export the flags to a spreadsheet." }],
});
// generated files come back as file ids -> client.beta.files.download(fileId)`;

async function runLiveSkills(client: Anthropic, fraudText: string): Promise<void> {
  if (process.env.LABS_RUN_SKILLS !== "1") {
    console.log("  Live: listing skills (read-only). Set LABS_RUN_SKILLS=1 to upload the skill and run a container");
    console.log("  (creates an org resource and bills code-execution time).");
    let i = 0;
    for await (const s of client.skills.list()) {
      if (i++ >= 5) break;
      console.log(`    ${s.id}  ${s.display_name}  source=${s.source}`);
    }
    return;
  }
  const skillDir = path.join(SKILLS_DIR, "claims-fraud-screening");
  const files = await Promise.all(
    filesUnder(skillDir).map((p) =>
      toFile(readFileSync(p), "claims-fraud-screening/" + path.relative(skillDir, p).split(path.sep).join("/"), { type: "text/markdown" }),
    ),
  );
  const skill = await client.skills.create({ files, display_name: "Claims fraud screening" });
  const resp = await client.beta.messages.create({
    model: MODEL, max_tokens: 16000, betas: ["code-execution-2025-08-25"],
    container: { skills: [{ type: "custom", skill_id: skill.id, version: "latest" }, { type: "anthropic", skill_id: "xlsx", version: "latest" }] },
    tools: [{ type: "code_execution_20260521", name: "code_execution" }],
    messages: [{ role: "user", content: fraudText + " Export the flags to a spreadsheet." }],
  });
  const text = resp.content.filter((b): b is Anthropic.Beta.BetaTextBlock => b.type === "text").map((b) => b.text).join("");
  console.log(`  uploaded ${skill.id}; container run stop_reason=${resp.stop_reason}`);
  console.log("  " + text.slice(0, 400).replace(/\n/g, "\n  "));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function table(headers: string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i]!.length)));
  console.log("  " + headers.map((h, i) => h.padEnd(widths[i]!)).join("  "));
  console.log("  " + widths.map((w) => "-".repeat(w)).join("  "));
  for (const r of rows) console.log("  " + r.map((c, i) => c.padEnd(widths[i]!)).join("  "));
}
const num = (x: number, d = 0): string => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x: number): string => `${Math.round(x * 100)}%`;
const fnol = (c: TriageRequest): string => `<fnol claim_id="${c.claim_id}">\n${c.text}\n</fnol>`;
function usageCells(m: Anthropic.Message): string[] {
  const u = m.usage;
  return [String(u.input_tokens), String(u.cache_creation_input_tokens ?? 0), String(u.cache_read_input_tokens ?? 0), `$${costUsd(MODEL, u).toFixed(5)}`];
}

banner("D2.5 - Prompt Reuse - Caching, Modular Prompts, Skills");
const client = getClient(responder);
const schedule: Component = {
  id: "reference.coverage_schedule", version: SCHEDULE.version, layer: "reference", requires: [],
  summary: "Generated from coverage_schedule.json", body: renderSchedule(SCHEDULE), source: "generated",
};
const library = PromptLibrary.load(LIBRARY_DIR, [schedule]);

// ------------------------------------------------------------------ 1
section("1. Prompt library: reusable, versioned components");
table(
  ["component", "layer", "requires", "~tokens", "source"],
  library.components.map((c) => [refOf(c), c.layer, c.requires.join(",") || "-", String(Math.floor(c.body.length / 4)), c.source]),
);

// ------------------------------------------------------------------ 2
section("2. Compose per use case: resolve semver, check deps, canonical order");
const triage = compose(library, USE_CASES.claims_triage!);
const memo = compose(library, USE_CASES.underwriting_memo!);
for (const [name, comp] of [["claims_triage", triage], ["underwriting_memo", memo]] as const) {
  const tok = (await client.messages.countTokens({ model: MODEL, system: comp.blocks(null, "none"), messages: [{ role: "user", content: "x" }] })).input_tokens;
  console.log(`  ${name.padEnd(18)} fingerprint=${comp.fingerprint}  ~${num(tok)} tok`);
  console.log("    order: " + comp.components.map(refOf).join(" > "));
}
const sharedN = commonPrefix(triage, memo);
const sharedText = triage.components.slice(0, sharedN).map(renderComponent).join("\n\n");
console.log(`  shared leading components: ${sharedN} (~${num(Math.floor(sharedText.length / 4))} tok) -> one cache entry can serve both`);
{
  const naive = compose(library, USE_CASES.claims_triage!, false);
  const shuffled = compose(library, [...USE_CASES.claims_triage!].reverse());
  const pinned = compose(library, USE_CASES.claims_triage!.map((r) => r.replace("safety.core@^2", "safety.core@2.2.0")));
  console.log(`  manifest order, no canonical sort : ${naive.fingerprint}  ${naive.fingerprint !== triage.fingerprint ? "(differs -> cache miss)" : ""}`);
  console.log(`  reversed manifest, canonical sort : ${shuffled.fingerprint}  ${shuffled.fingerprint === triage.fingerprint ? "(identical -> cache hit)" : "(differs!)"}`);
  console.log(`  pin safety.core@2.2.0             : ${pinned.fingerprint}  (version change = deliberate cache reset)`);
  try {
    compose(library, ["safety.pii@^1", "persona.claims_assistant@^1"]);
  } catch (err) {
    if (!(err instanceof CompositionError)) throw err;
    console.log(`  dependency check                  : CompositionError -> ${err.message}`);
  }
}

// ------------------------------------------------------------------ 3
section("3. Exam Q2: same large system prompt every request -> cache it");
const claims = CLAIMS.triage_requests;
let prefixTokens = 0;
{
  const rows: string[][] = [];
  let uncachedCost = 0;
  let cachedCost = 0;
  for (const c of claims) {
    const mUn = await client.messages.create({ model: MODEL, max_tokens: 512, system: triage.blocks(null, "none"), messages: [{ role: "user", content: fnol(c) }], output_config: { effort: "low" } });
    const mCa = await client.messages.create({ model: MODEL, max_tokens: 512, system: triage.blocks(null, "end"), messages: [{ role: "user", content: fnol(c) }], output_config: { effort: "low" } });
    uncachedCost += costUsd(MODEL, mUn.usage);
    cachedCost += costUsd(MODEL, mCa.usage);
    const decision = String((JSON.parse(textOf(mCa)) as { decision?: string }).decision ?? "?");
    rows.push([c.claim_id, decision, ...usageCells(mCa), String(mUn.usage.input_tokens), `$${costUsd(MODEL, mUn.usage).toFixed(5)}`]);
  }
  table(["claim", "decision", "in", "cache_write", "cache_read", "cost (cached)", "in (no cache)", "cost (no cache)"], rows);
  prefixTokens = parseInt(rows[0]![3]!, 10);
  console.log(
    `  5 requests: $${cachedCost.toFixed(4)} cached vs $${uncachedCost.toFixed(4)} uncached -> ${pct(1 - cachedCost / uncachedCost)} saved ` +
      "(and cached prefixes cut time-to-first-token on real traffic).",
  );
  console.log("  Rule: static content (tools -> system -> stable docs) first, breakpoint at its end, dynamic content after.");
}

// ------------------------------------------------------------------ 4
section("4. TTL economics on Opus 5.5 (cost of the cached prefix only)");
{
  const p = PRICING[MODEL]!;
  const rows: string[][] = [];
  let even5m: number | null = null;
  let even1h: number | null = null;
  for (const n of [1, 2, 3, 5, 10, 100]) {
    const unc = (n * prefixTokens * p.input) / 1e6;
    const c5 = (prefixTokens * p.cacheWrite + (n - 1) * prefixTokens * p.cacheRead) / 1e6;
    const c1 = (prefixTokens * p.input * ONE_HOUR_WRITE_MULT + (n - 1) * prefixTokens * p.cacheRead) / 1e6;
    if (even5m === null && c5 < unc) even5m = n;
    if (even1h === null && c1 < unc) even1h = n;
    rows.push([String(n), `$${unc.toFixed(4)}`, `$${c5.toFixed(4)}`, `$${c1.toFixed(4)}`]);
  }
  table(["requests in TTL", "no cache", "5-min TTL (1.25x write)", "1-hour TTL (2x write)"], rows);
  console.log(`  Break-even: 5-min TTL from request ${even5m}, 1-hour TTL from request ${even1h}. Reads refresh the TTL;`);
  console.log("  use 1h only when requests sharing the prefix arrive 5-60 minutes apart.");
}

// ------------------------------------------------------------------ 5
section("5. Silent invalidator: a timestamp at the top of the system prompt");
{
  const t0 = 9 * 3600; // fake clock (seconds since midnight) - real code would call new Date()
  const two = (x: number): string => String(x).padStart(2, "0");
  const stamp = (i: number): string => {
    const s = t0 + 37 * i;
    return `2026-10-11T${two(Math.floor(s / 3600))}:${two(Math.floor((s % 3600) / 60))}:${two(s % 60)}Z`;
  };
  const rows: string[][] = [];
  for (const [i, c] of claims.slice(0, 3).entries()) {
    const badSystem: SystemBlocks = [{ type: "text", text: `Current UTC time: ${stamp(i)}` }, ...triage.blocks(null, "end")];
    const m = await client.messages.create({ model: MODEL, max_tokens: 512, system: badSystem, messages: [{ role: "user", content: fnol(c) }], output_config: { effort: "low" } });
    rows.push(["timestamp in system", c.claim_id, ...usageCells(m)]);
  }
  for (const [j, c] of claims.slice(3, 5).entries()) {
    const i = j + 3;
    const m = await client.messages.create({
      model: MODEL, max_tokens: 512, system: triage.blocks(null, "end"),
      messages: [{ role: "user", content: `<request_time>${stamp(i)}</request_time>\n${fnol(c)}` }], output_config: { effort: "low" },
    });
    rows.push(["timestamp in user turn", c.claim_id, ...usageCells(m)]);
  }
  table(["variant", "claim", "in", "cache_write", "cache_read", "cost"], rows);
  console.log("  Every request rewrote the cache (1.25x price) and never read it - no error, just a bigger bill.");
  console.log("  Same family: UUIDs/request ids, unsorted JSON, per-user names, conditional sections, varying tools.");
}

// ------------------------------------------------------------------ 6
section("6. Wrong tool: top-level automatic caching on single-shot requests");
{
  const rows: string[][] = [];
  for (const q of CLAIMS.memo_requests) {
    const m = await client.messages.create({
      model: MODEL, max_tokens: 512, system: memo.blocks(null, "none"),
      cache_control: { type: "ephemeral" }, // breakpoint lands on the unique question
      messages: [{ role: "user", content: q }], output_config: { effort: "low" },
    });
    rows.push([q.slice(0, 48) + "...", ...usageCells(m)]);
  }
  table(["memo request", "in", "cache_write", "cache_read", "cost"], rows);
  console.log("  Automatic caching places the breakpoint on the LAST block - here the unique question - so each");
  console.log("  request writes a new entry nobody reads. It shines for multi-turn chats (the breakpoint moves");
  console.log("  forward with the conversation); for shared-prefix + unique-suffix, mark the shared prefix explicitly.");
}

// ------------------------------------------------------------------ 7
section("7. Cross-use-case reuse: breakpoint after the shared components");
{
  const m1 = await client.messages.create({ model: MODEL, max_tokens: 512, system: triage.blocks(sharedN, "split"), messages: [{ role: "user", content: fnol(claims[0]!) }], output_config: { effort: "low" } });
  const m2 = await client.messages.create({ model: MODEL, max_tokens: 512, system: memo.blocks(sharedN, "split"), messages: [{ role: "user", content: CLAIMS.memo_requests[0]! }], output_config: { effort: "low" } });
  table(["use case", "in", "cache_write", "cache_read", "cost"], [["claims_triage", ...usageCells(m1)], ["underwriting_memo", ...usageCells(m2)]]);
  console.log("  The memo assistant read the prefix the triage assistant wrote: canonical ordering + shared components");
  console.log("  turn a prompt library into a cache-sharing strategy (up to 4 breakpoints per request).");
}

// ------------------------------------------------------------------ 8
section("8. Agent Skills: validation + progressive disclosure");
const skills = loadSkills(SKILLS_DIR);
const byName: Record<string, Skill> = Object.fromEntries(skills.map((s) => [s.name, s]));
const fraudText = CLAIMS.skill_requests[0]!.text;
{
  for (const s of skills) console.log(`  ${s.name.padEnd(24)} OK  description ${s.description.length} chars, SKILL.md ${num(skillMd(s).length)} chars`);
  const bad = validateSkillMeta({ name: "Claude-Helper", description: "" }, "claude-helper");
  console.log("  invalid example 'Claude-Helper': " + bad.join("; "));
  try {
    readSkillFile(byName, "claims-fraud-screening", "../../use_cases.json");
  } catch (err) {
    if (!(err instanceof SkillError)) throw err;
    console.log(`  path traversal attempt          : SkillError -> ${err.message}`);
  }
  const base = compose(library, USE_CASES.skills_demo!).text;
  const tools = skillTools(Object.keys(byName));
  const everything =
    base +
    "\n\n" +
    skills
      .map((s) => skillMd(s) + filesUnder(s.folder).filter((f) => f.endsWith(".md") && path.basename(f) !== "SKILL.md").map((f) => "\n\n" + readText(f)).join(""))
      .join("\n\n");
  const indexSystem: SystemBlocks = [
    { type: "text", text: base },
    { type: "text", text: skillsIndex(skills), cache_control: { type: "ephemeral" } },
  ];
  const userMsg: Anthropic.MessageParam[] = [{ role: "user", content: fraudText }];
  const tokAll = (await client.messages.countTokens({ model: MODEL, system: everything, messages: userMsg })).input_tokens;
  const tokIdx = (await client.messages.countTokens({ model: MODEL, system: indexSystem, tools, messages: userMsg })).input_tokens;
  const tokBase = (await client.messages.countTokens({ model: MODEL, system: base, messages: userMsg })).input_tokens;
  const tokBaseTools = (await client.messages.countTokens({ model: MODEL, system: base, tools, messages: userMsg })).input_tokens;
  console.log(`  all skills inlined in system : ${num(tokAll)} tok on EVERY request`);
  console.log(`  index only (+2 loader tools) : ${num(tokIdx)} tok -> ${pct(1 - tokIdx / tokAll)} smaller; full text loads on demand`);
  const n = skills.length;
  const inlinePer = (tokAll - tokBase) / n;
  const indexPer = (tokIdx - tokBaseTools) / n;
  console.log(
    `  per skill: ~${inlinePer.toFixed(0)} tok inlined vs ~${indexPer.toFixed(0)} tok as an index entry -> with 40 skills: ` +
      `~${num(40 * inlinePer)} vs ~${num(40 * indexPer + (tokBaseTools - tokBase))} tok per request`,
  );
  const [answer, trace] = await runSkillAgent(client, indexSystem, tools, byName, fraudText);
  console.log("  Request 1 (fraud check) - the model pulls in only what it needs:");
  for (const line of trace) console.log("    " + line);
  for (const line of answer.split("\n")) console.log("    | " + line);
  const [answer2, trace2] = await runSkillAgent(client, indexSystem, tools, byName, CLAIMS.skill_requests[1]!.text);
  console.log("  Request 2 (no matching skill) - nothing extra is loaded:");
  for (const line of trace2) console.log("    " + line);
  console.log("    | " + answer2);
}

// ------------------------------------------------------------------ 9
section("9. Agent Skills on the Claude API (live-only)");
for (const line of SKILLS_API_SNIPPET.split("\n")) console.log("    " + line);
if (isLive()) {
  await runLiveSkills(client, fraudText);
} else {
  console.log("  [live-only] The offline mock has no Skills API or code-execution container. The loader above");
  console.log("  mirrors what the platform does: name+description in context, SKILL.md and files on demand.");
}

section("Takeaways");
console.log("  * Cache the big static prefix (exam Q2): static first, breakpoint at its end, verify cache_read > 0.");
console.log("  * Hunt silent invalidators: timestamps, ids, unsorted JSON, varying tools; pick the right breakpoint.");
console.log("  * Modular prompts: versioned components + canonical order = reuse AND a stable, shareable cache key.");
console.log("  * Skills: package procedures as SKILL.md; only descriptions cost tokens until a task needs them.");
