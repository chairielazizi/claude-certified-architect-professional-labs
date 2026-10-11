/**
 * D7.1 - Configuring Claude Code for Teams (TypeScript twin of python/main.py)
 *
 * A "team config linter + permission simulator" for the example Claude Code setup in
 * ../example-project:
 *   1. inventory of the shared team setup (settings, hooks, CLAUDE.md, MCP, subagent, skill, command)
 *   2. lint the team config (expected: clean) and a deliberately risky config (expected: many findings)
 *   3. merge five settings layers (user < project < local < CLI < managed) with provenance
 *   4. evaluate sample tool requests: PreToolUse hook -> deny > ask > allow -> permission mode
 *   5. run the real hook scripts with Claude Code's stdin/exit-code contract
 *   6. bonus: Claude drafts a CLAUDE.md from a repo summary, then the linter checks the draft
 *
 * Nothing here touches your real Claude Code configuration; everything is read from this lab folder.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MODEL_OPUS, banner, getClient, section, textOf, usageLine } from "../../shared/typescript/claudeClient.ts";
import type { MockRequest } from "../../shared/typescript/claudeClient.ts";
import {
  SEVERITY_ORDER, evaluate, lintAgent, lintCommand, lintMcp, lintMemory, lintSettings, lintSkill, loadJson,
  mergeLayers, readText, ruleMatches, splitLines,
} from "./teamconfig.ts";
import type { Effective, Finding, Json } from "./teamconfig.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LAB = path.join(HERE, "..");
const PROJECT = path.join(LAB, "example-project");
const DATA = path.join(LAB, "data");
const OUT = path.join(LAB, "labs_output");
const REPO_ROOT = path.join(LAB, "..");

// ---------------------------------------------------------------------------
// Offline responder: canned CLAUDE.md draft (live mode calls the real model)
// ---------------------------------------------------------------------------
const CANNED_CLAUDE_MD = `# orders-platform

Order management monorepo: Python API in \`api/\`, TypeScript web client in \`web/\`.

## Commands
- Setup: \`make setup\`
- Lint: \`make lint\` (run before every commit)
- Tests: \`make test\`; web unit tests: \`npm run test:unit\` inside \`web/\`
- Migrations: \`make migrate\` (ask a human first); seed data: \`make seed-db\`

## Architecture
- \`api/\` service code, \`web/\` client, \`migrations/\` SQL, \`tests/\` test suites
- Module boundaries: @docs/architecture.md

## Conventions
- Conventional Commits for every commit message.
- Money is stored as integer cents.
- Never edit applied migrations; add a new migration instead.
- Follow best practices for security.

## Guard rails
- Never read \`.env\`, \`secrets/\`, or \`deploy/certs/\`.
`;

function responder(req: MockRequest): string {
  if (req.systemText.includes("CLAUDE.md")) return CANNED_CLAUDE_MD;
  return "ok";
}

// ---------------------------------------------------------------------------
// Printing helpers
// ---------------------------------------------------------------------------
function printFindings(findings: Finding[]): void {
  if (!findings.length) {
    console.log("  (no findings)");
    return;
  }
  const sorted = [...findings].sort(
    (a, b) => SEVERITY_ORDER[a.severity]! - SEVERITY_ORDER[b.severity]! || (a.code < b.code ? -1 : a.code > b.code ? 1 : 0),
  );
  for (const f of sorted) console.log(`  [${f.severity.padEnd(5)}] ${f.code}  ${f.target}: ${f.message}`);
}

function counts(findings: Finding[]): string {
  const c = (s: string) => findings.filter((f) => f.severity === s).length;
  return `ERROR=${c("ERROR")} WARN=${c("WARN")} INFO=${c("INFO")}`;
}

function short(value: unknown, width: number): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length <= width ? text : text.slice(0, width - 3) + "...";
}

// ---------------------------------------------------------------------------
// Hook runner: emulates how Claude Code invokes a command hook
// ---------------------------------------------------------------------------
const HOOK_SCRIPT_RE = /\$CLAUDE_PROJECT_DIR\/([^"'\s]+)/;

/**
 * Hooks are language-agnostic executables; the team's hooks happen to be Python. Find an
 * interpreter (repo .venv first, then PATH) so this TypeScript lab can execute them too.
 */
function findPython(): string | null {
  const candidates = [
    process.env.PYTHON,
    path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),
    path.join(REPO_ROOT, ".venv", "bin", "python"),
    "python3",
    "python",
  ].filter((c): c is string => Boolean(c));
  for (const c of candidates) {
    if (c.includes(path.sep) && !existsSync(c)) continue;
    const r = spawnSync(c, ["--version"], { encoding: "utf8" });
    if (r.status === 0) return c;
  }
  return null;
}
const PYTHON = findPython();

/** stdin = JSON payload; exit 0 = continue, 2 = block (stderr goes back to Claude), other = non-blocking error. */
function runHookScript(script: string, payload: string, projectDir: string): [number, string, string] {
  if (!PYTHON) return [0, "", "SKIPPED: no Python interpreter found to run the hook"];
  const r = spawnSync(PYTHON, [script], {
    input: payload,
    encoding: "utf8",
    env: { ...process.env, CLAUDE_PROJECT_DIR: projectDir },
    timeout: 30_000,
  });
  return [r.status ?? 1, (r.stdout ?? "").trim(), (r.stderr ?? "").trim()];
}

function matcherMatches(matcher: string, tool: string): boolean {
  return matcher === "" || matcher === "*" || new RegExp(`^(?:${matcher})$`).test(tool);
}

function runPreToolHooks(hooks: Record<string, Json[]>, tool: string, toolInput: Json): ["pass" | "block", string] {
  for (const entry of hooks.PreToolUse ?? []) {
    if (!matcherMatches(String(entry.matcher ?? ""), tool)) continue;
    for (const hook of (entry.hooks ?? []) as Json[]) {
      const m = HOOK_SCRIPT_RE.exec(String(hook.command ?? ""));
      if (!m) continue;
      const payload = JSON.stringify({
        session_id: "lab-session", cwd: PROJECT, hook_event_name: "PreToolUse", tool_name: tool, tool_input: toolInput,
      });
      const [code, , err] = runHookScript(path.join(PROJECT, m[1]!), payload, PROJECT);
      if (code === 2) return ["block", err];
    }
  }
  return ["pass", ""];
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------
function stepInventory(): void {
  section("1. The shared team setup (example-project/)");
  const items: [string, string, string][] = [
    [".claude/settings.json", "project (shared)", "permissions allow/ask/deny, env, hooks, MCP opt-in"],
    [".claude/hooks/protect_paths.py", "project (shared)", "PreToolUse guard: exit 2 blocks protected paths"],
    [".claude/hooks/format_changed_file.py", "project (shared)", "PostToolUse formatter for edited files"],
    ["CLAUDE.md", "project memory", "team commands, architecture, conventions"],
    ["web/CLAUDE.md", "nested memory", "loaded when Claude works inside web/"],
    ["docs/architecture.md", "memory import", "pulled in via @docs/architecture.md"],
    [".mcp.json", "project MCP", "servers with ${ENV_VAR} secrets (approved per user)"],
    [".claude/agents/code-reviewer.md", "subagent", "read-only reviewer (tools: Read, Grep, Glob)"],
    [".claude/skills/api-endpoint-checklist/SKILL.md", "skill", "loaded on demand by description"],
    [".claude/commands/pr-ready.md", "slash command", "/pr-ready [base-branch]"],
    [".gitignore", "repo hygiene", "keeps settings.local.json and .env out of git"],
  ];
  for (const [rel, scope, purpose] of items) {
    const status = existsSync(path.join(PROJECT, rel)) ? "present" : "MISSING";
    console.log(`  ${status.padEnd(8)} ${rel.padEnd(48)} ${scope.padEnd(17)} ${purpose}`);
  }
}

const mdFiles = (dir: string): string[] => (existsSync(dir) ? readdirSync(dir).filter((f) => f.endsWith(".md")).sort() : []);

function lintProjectDir(root: string): Finding[] {
  const findings: Finding[] = [];
  findings.push(...lintSettings(loadJson(path.join(root, ".claude", "settings.json")), ".claude/settings.json", root));
  findings.push(...lintMcp(loadJson(path.join(root, ".mcp.json")), ".mcp.json"));
  for (const agent of mdFiles(path.join(root, ".claude", "agents"))) {
    findings.push(...lintAgent(readText(path.join(root, ".claude", "agents", agent)), `.claude/agents/${agent}`));
  }
  const skillsDir = path.join(root, ".claude", "skills");
  for (const dir of existsSync(skillsDir) ? readdirSync(skillsDir).sort() : []) {
    const file = path.join(skillsDir, dir, "SKILL.md");
    if (existsSync(file)) findings.push(...lintSkill(readText(file), `.claude/skills/${dir}/SKILL.md`, dir));
  }
  for (const cmd of mdFiles(path.join(root, ".claude", "commands"))) {
    findings.push(...lintCommand(readText(path.join(root, ".claude", "commands", cmd)), `.claude/commands/${cmd}`));
  }
  findings.push(...lintMemory(readText(path.join(root, "CLAUDE.md")), "CLAUDE.md", root));
  findings.push(...lintMemory(readText(path.join(root, "web", "CLAUDE.md")), "web/CLAUDE.md", root));
  return findings;
}

function stepLint(): [Finding[], Finding[]] {
  section("2. Lint the team config (expected: clean)");
  const good = lintProjectDir(PROJECT);
  printFindings(good);
  console.log(`  summary: ${counts(good)}`);

  section("3. Lint a deliberately risky config (data/risky/)");
  const r = (f: string) => path.join(DATA, "risky", f);
  const bad: Finding[] = [
    ...lintSettings(loadJson(r("settings.risky.json")), "settings.risky.json"),
    ...lintMcp(loadJson(r("mcp.risky.json")), "mcp.risky.json"),
    ...lintAgent(readText(r("agent-reviewer.risky.md")), "agent-reviewer.risky.md"),
    ...lintSkill(readText(r("skill.risky.md")), "skill.risky.md"),
    ...lintCommand(readText(r("command.risky.md")), "command.risky.md"),
    ...lintMemory(readText(r("memory.risky.md")), "memory.risky.md", PROJECT),
  ];
  printFindings(bad);
  console.log(`  summary: ${counts(bad)}`);
  console.log("  -> Run this linter in CI on every PR that touches .claude/, .mcp.json or CLAUDE.md.");
  return [good, bad];
}

function loadLayers(): [string, Json][] {
  return [
    ["user", loadJson(path.join(DATA, "layers", "user.settings.json"))],
    ["project", loadJson(path.join(PROJECT, ".claude", "settings.json"))],
    ["local", loadJson(path.join(DATA, "layers", "local.settings.json"))],
    ["cli", loadJson(path.join(DATA, "layers", "cli-args.json")).settings],
    ["managed", loadJson(path.join(DATA, "layers", "managed.settings.json"))],
  ];
}

function stepPrecedence(): Effective {
  section("4. Settings precedence: merge five layers with provenance");
  console.log("  precedence (highest first): managed policy > command-line args > local project");
  console.log("  (.claude/settings.local.json) > shared project (.claude/settings.json) > user (~/.claude/settings.json)");
  console.log(`  CLI layer simulates: ${(loadJson(path.join(DATA, "layers", "cli-args.json")).argv as string[]).join(" ")}`);
  const layers = loadLayers();
  const eff = mergeLayers(layers);
  console.log("\n  effective scalar settings (winner <- overridden values):");
  for (const key of Object.keys(eff.values).sort()) {
    const hist = eff.history[key]!;
    const lost = hist.slice(0, -1).map(([layer, v]) => `${layer}=${short(v, 18)}`);
    const lostTxt = lost.length ? `   overrides: ${lost.join(", ")}` : "";
    console.log(`    ${key.padEnd(42)} = ${short(eff.values[key], 20).padEnd(20)} [${eff.source[key]}]${lostTxt}`);
  }
  console.log("\n  permission rules are COMBINED across layers (deny anywhere wins at evaluation):");
  for (const kind of ["allow", "ask", "deny"] as const) {
    const bySrc = new Map<string, number>();
    for (const r of eff.rules) if (r.kind === kind) bySrc.set(r.source, (bySrc.get(r.source) ?? 0) + 1);
    const total = [...bySrc.values()].reduce((a, b) => a + b, 0);
    const detail = [...bySrc.entries()].map(([src, n]) => `${src}=${n}`).join(", ");
    console.log(`    ${kind.padEnd(5)} ${String(total).padStart(2)} rules  (${detail})`);
  }
  console.log(`    hooks: ${Object.entries(eff.hooks).map(([ev, v]) => `${ev} x${v.length}`).join(", ")}`);

  // What-if: a developer tries to switch on bypass mode locally.
  const whatIf = layers.map(([n, s]): [string, Json] =>
    n !== "local" ? [n, s] : [n, { ...s, permissions: { ...s.permissions, defaultMode: "bypassPermissions" } }],
  );
  const effBypass = mergeLayers(whatIf);
  console.log("\n  what-if: local settings request defaultMode=bypassPermissions");
  for (const note of effBypass.notes) console.log(`    -> ${note}`);

  mkdirSync(OUT, { recursive: true });
  const dump = { values: eff.values, source: eff.source, rules: eff.rules.map((r) => ({ kind: r.kind, rule: r.rule, source: r.source })) };
  writeFileSync(path.join(OUT, "effective-settings.json"), JSON.stringify(dump, null, 2), "utf8");
  console.log("\n  wrote labs_output/effective-settings.json");
  return eff;
}

function describeInput(toolInput: Json): string {
  for (const key of ["command", "file_path", "url"]) if (key in toolInput) return String(toolInput[key]);
  return JSON.stringify(toolInput);
}

function stepSimulate(eff: Effective): [number, number] {
  section("5. Permission simulator: PreToolUse hook -> deny > ask > allow -> mode");
  const mode = String(eff.values["permissions.defaultMode"] ?? "default");
  console.log(`  effective defaultMode = ${mode}  (hooks run first; a hook exit 2 blocks even an allowed call)`);
  const requests = JSON.parse(readText(path.join(DATA, "tool_requests.json"))) as { tool: string; input: Json; expect: string }[];
  let ok = 0;
  requests.forEach((req, idx) => {
    const { tool, input: toolInput, expect } = req;
    console.log(`\n  #${String(idx + 1).padStart(2, "0")} ${tool.padEnd(34)} ${short(describeInput(toolInput), 56)}`);
    const [hookState, hookMsg] = runPreToolHooks(eff.hooks, tool, toolInput);
    let outcome: string;
    if (hookState === "block") {
      outcome = "blocked-by-hook";
      console.log(`      hook  : BLOCK (exit 2) -> ${short(hookMsg, 90)}`);
    } else {
      const d = evaluate(eff.rules, tool, toolInput, mode);
      outcome = d.decision;
      const via = d.rule ? `${d.rule} [${d.source}]` : d.reason;
      const sub = d.subCommand && d.subCommand !== toolInput.command ? ` on sub-command '${d.subCommand}'` : "";
      console.log(`      result: ${d.decision.toUpperCase().padEnd(5)} via ${via}${sub}`);
      const losers = eff.rules.filter((r) => r.kind !== d.decision && d.rule && ruleMatches(r.rule, tool, toolInput, d.subCommand));
      if (losers.length) console.log("      note  : overrides " + losers.map((r) => `${r.kind} ${r.rule} [${r.source}]`).join(", "));
    }
    const match = outcome === expect;
    if (match) ok++;
    console.log(`      expect: ${expect.padEnd(15)} ${match ? "OK" : "MISMATCH"}`);
  });
  console.log(`\n  simulator agreed with expectations on ${ok}/${requests.length} requests`);
  return [ok, requests.length];
}

function stepHooks(): void {
  section("6. Hooks up close: the stdin / exit-code contract");
  const guard = path.join(PROJECT, ".claude", "hooks", "protect_paths.py");
  const cases: [string, Json][] = [
    ["Edit", { file_path: "api/routes/orders.py" }],
    ["Write", { file_path: "secrets/stripe.key" }],
    ["Edit", { file_path: "../../outside-the-repo.txt" }],
    ["Edit", { file_path: "web/package-lock.json" }],
  ];
  const meaning = (code: number) => (code === 0 ? "continue" : code === 2 ? "BLOCK (stderr -> Claude)" : "non-blocking error");
  for (const [tool, toolInput] of cases) {
    const payload = JSON.stringify({ hook_event_name: "PreToolUse", tool_name: tool, tool_input: toolInput, cwd: PROJECT });
    const [code, , err] = runHookScript(guard, payload, PROJECT);
    console.log(`  PreToolUse ${tool.padEnd(5)} ${String(toolInput.file_path).padEnd(28)} exit=${code} ${meaning(code)}`);
    if (err) console.log(`      stderr: ${short(err, 96)}`);
  }
  const [code, , err] = runHookScript(guard, "this is not json", PROJECT);
  console.log(`  PreToolUse malformed payload                 exit=${code} ${meaning(code)} (${err})`);

  // PostToolUse formatter, run against a scratch file in labs_output (never the example project).
  const sandbox = path.join(OUT, "hook-sandbox");
  mkdirSync(sandbox, { recursive: true });
  const target = path.join(sandbox, "notes.md");
  writeFileSync(target, "# Release notes   \n\n- fixed refund rounding  \n- added audit log\t\n\n\n", "utf8");
  const before = readText(target);
  const payload = JSON.stringify({ hook_event_name: "PostToolUse", tool_name: "Write", tool_input: { file_path: "notes.md" }, cwd: sandbox });
  const [pcode, pout] = runHookScript(path.join(PROJECT, ".claude", "hooks", "format_changed_file.py"), payload, sandbox);
  const after = readText(target);
  const trailing = (t: string) => t.split("\n").filter((line) => line !== line.trimEnd()).length;
  console.log(`\n  PostToolUse Write notes.md -> exit=${pcode}, hook said: '${pout}'`);
  console.log(`      before: ${before.length} chars, ${trailing(before)} lines with trailing whitespace`);
  console.log(`      after : ${after.length} chars, ${trailing(after)} lines with trailing whitespace`);
}

async function stepGenerateClaudeMd(): Promise<Finding[]> {
  section("7. Bonus: Claude drafts a CLAUDE.md from a repo summary");
  const client = getClient(responder);
  const summary = readText(path.join(DATA, "repo_summary.json"));
  const system =
    "You write CLAUDE.md memory files for Claude Code. Rules: under 60 lines; sections Commands, " +
    "Architecture, Conventions, Guard rails; only facts present in the repo summary; never include " +
    "secrets; specific, checkable instructions in imperative voice.";
  const msg = await client.messages.create({
    model: MODEL_OPUS,
    max_tokens: 2000,
    thinking: { type: "adaptive" },
    output_config: { effort: "medium" },
    system,
    messages: [{ role: "user", content: `<repo_summary>\n${summary}\n</repo_summary>\nDraft the CLAUDE.md.` }],
  });
  const draft = textOf(msg);
  console.log(`  model=${msg.model} stop_reason=${msg.stop_reason} ${usageLine(msg)}`);
  if (msg.stop_reason === "max_tokens") console.log("  WARNING: draft truncated - raise max_tokens before using it");
  const lines = splitLines(draft);
  for (const line of lines.slice(0, 8)) console.log(`    | ${line}`);
  console.log(`    | ... (${lines.length} lines total)`);
  const findings = lintMemory(draft, "CLAUDE.generated.md", PROJECT);
  console.log("  linter on the generated draft (human review still required):");
  printFindings(findings);
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, "CLAUDE.generated.md"), draft, "utf8");
  console.log("  wrote labs_output/CLAUDE.generated.md");
  return findings;
}

banner("D7.1 - Configuring Claude Code for Teams");
stepInventory();
const [good, bad] = stepLint();
const eff = stepPrecedence();
const [ok, total] = stepSimulate(eff);
stepHooks();
const draftFindings = await stepGenerateClaudeMd();

section("8. Takeaways");
console.log(`  team config findings: ${counts(good)} | risky config: ${counts(bad)} | draft: ${counts(draftFindings)}`);
console.log(`  permission simulator: ${ok}/${total} decisions matched expectations`);
console.log("  - Commit .claude/settings.json, CLAUDE.md, .mcp.json; keep settings.local.json personal.");
console.log("  - Deny wins over ask and allow, in every layer; managed policy cannot be overridden.");
console.log("  - Hooks are deterministic guard rails (exit 2 blocks); prompts and memory are guidance.");
console.log("  - Secrets: deny reads of .env/secrets, and use ${ENV_VAR} expansion in .mcp.json.");
