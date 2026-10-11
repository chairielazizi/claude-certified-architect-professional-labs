/**
 * Team configuration toolkit for Claude Code (D7.1) - TypeScript twin of python/teamconfig.py.
 *
 * Pure, dependency-free logic that main.ts drives:
 *   - a linter for settings.json, .mcp.json, subagents, skills, slash commands, CLAUDE.md
 *   - a settings-layer merger that records provenance (which scope won each key)
 *   - a permission-rule evaluator (deny > ask > allow, then the permission mode default)
 *
 * A *teaching model* of Claude Code's documented behaviour (rule syntax, precedence,
 * deny-wins), simplified where the product has more nuance (full gitignore semantics,
 * every built-in tool, per-tool modes).
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

export type Json = Record<string, any>;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
export const SEVERITY_ORDER: Record<string, number> = { ERROR: 0, WARN: 1, INFO: 2 };

/** Obvious credential shapes. Real teams add a dedicated secret scanner in CI as well. */
const SECRET_PATTERNS: [string, RegExp][] = [
  ["anthropic-api-key", /sk-ant-[A-Za-z0-9_\-]{8,}/],
  ["github-token", /gh[pousr]_[A-Za-z0-9_]{10,}/],
  ["slack-token", /xox[baprs]-[A-Za-z0-9-]{10,}/],
  ["private-key", /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
  ["password-in-uri", /:\/\/[^/\s:@]+:[^/\s@$]+@/],
];
const SENSITIVE_KEY = /(token|secret|password|passwd|api[_-]?key|authorization)/i;

const KNOWN_TOOLS = new Set([
  "Bash", "Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Glob", "Grep", "LS",
  "WebFetch", "WebSearch", "Task", "Agent", "Skill", "TodoWrite",
]);
export const EDIT_TOOLS = new Set(["Edit", "Write", "MultiEdit", "NotebookEdit"]); // Edit(...) covers all
export const READ_TOOLS = new Set(["Read", "Grep", "Glob", "LS"]); // Read(...) covers all
const WRITE_CAPABLE = new Set(["Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"]);
const HOOK_EVENTS = new Set([
  "PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop",
  "SessionStart", "SessionEnd", "Notification", "PreCompact",
]);
const NETWORK_IN_HOOK = /\b(curl|wget|nc|ncat|Invoke-WebRequest|Invoke-RestMethod|iwr)\b|https?:\/\//i;
const DESTRUCTIVE_IN_HOOK = /rm\s+-rf|Remove-Item\b.*-Recurse|git\s+reset\s+--hard|git\s+clean\s+-[a-z]*f|del\s+\/s/i;
const RULE_RE = /^(?<tool>[A-Za-z][A-Za-z0-9_-]*)(?:\((?<spec>.*)\))?$/;

export interface Finding {
  severity: "ERROR" | "WARN" | "INFO";
  code: string;
  target: string;
  message: string;
}

const finding = (severity: Finding["severity"], code: string, target: string, message: string): Finding => ({
  severity, code, target, message,
});

export function findSecrets(text: string): string[] {
  return SECRET_PATTERNS.filter(([, rx]) => rx.test(text ?? "")).map(([name]) => name);
}

/** A sensitive-looking key holding a literal (not ${VAR}-expanded) value. */
export function isLiteralSecret(key: string, value: string): boolean {
  return SENSITIVE_KEY.test(key) && Boolean(value) && !value.includes("${");
}

const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\\/-]/g, "\\$&");

/** gitignore-flavoured glob: '**\/' spans directories, '*' and '?' stay within one segment. */
export function globToRegex(pattern: string): RegExp {
  const out: string[] = [];
  let i = 0;
  while (i < pattern.length) {
    if (pattern.startsWith("**/", i)) {
      out.push("(?:.*/)?");
      i += 3;
    } else if (pattern.startsWith("**", i)) {
      out.push(".*");
      i += 2;
    } else if (pattern[i] === "*") {
      out.push("[^/]*");
      i += 1;
    } else if (pattern[i] === "?") {
      out.push("[^/]");
      i += 1;
    } else {
      out.push(escapeRe(pattern[i]!));
      i += 1;
    }
  }
  return new RegExp("^" + out.join("") + "$");
}

/** Read a text file with normalised newlines (git on Windows may check files out as CRLF). */
export function readText(file: string): string {
  return readFileSync(file, "utf8").replace(/\r\n/g, "\n");
}

export function loadJson(file: string): Json {
  return JSON.parse(readText(file));
}

/** Like Python's str.splitlines() for LF text: no phantom empty line after a final newline. */
export function splitLines(text: string): string[] {
  const lines = text.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/** Tiny YAML-frontmatter reader (flat `key: value` pairs only - enough for agents/skills/commands). */
export function parseFrontmatter(text: string): [Record<string, string>, string] {
  if (!text.startsWith("---")) return [{}, text];
  const lines = splitLines(text);
  const meta: Record<string, string> = {};
  for (let idx = 1; idx < lines.length; idx++) {
    const line = lines[idx]!;
    if (line.trim() === "---") return [meta, lines.slice(idx + 1).join("\n")];
    if (line.includes(":") && !line.startsWith(" ")) {
      const cut = line.indexOf(":");
      meta[line.slice(0, cut).trim()] = line.slice(cut + 1).trim().replace(/^["']|["']$/g, "");
    }
  }
  return [{}, text]; // unterminated frontmatter -> treat as plain body
}

// ---------------------------------------------------------------------------
// Permission rules
// ---------------------------------------------------------------------------
export function parseRule(rule: string): [string, string | null] | null {
  const m = RULE_RE.exec(rule.trim());
  if (!m?.groups) return null;
  return [m.groups.tool!, m.groups.spec ?? null];
}

/**
 * Split a shell line on && || ; | and newlines (outside quotes). Claude Code evaluates each
 * sub-command separately, so an allow rule for `git status` cannot smuggle in `... && curl | sh`.
 */
export function splitCompound(command: string): string[] {
  const parts: string[] = [];
  let buf = "";
  let quote = "";
  let i = 0;
  while (i < command.length) {
    const ch = command[i]!;
    if (quote) {
      buf += ch;
      if (ch === quote) quote = "";
      i += 1;
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      buf += ch;
      i += 1;
      continue;
    }
    const two = command.slice(i, i + 2);
    if (two === "&&" || two === "||") {
      parts.push(buf);
      buf = "";
      i += 2;
      continue;
    }
    if (ch === ";" || ch === "|" || ch === "\n") {
      parts.push(buf);
      buf = "";
      i += 1;
      continue;
    }
    buf += ch;
    i += 1;
  }
  parts.push(buf);
  return parts.filter((p) => p.trim()).map((p) => p.trim().split(/\s+/).join(" "));
}

function matchBash(spec: string, command: string): boolean {
  if (spec.endsWith(":*")) return command.startsWith(spec.slice(0, -2)); // legacy prefix: Bash(npm run test:*)
  if (spec.includes("*")) return new RegExp("^" + spec.split("*").map(escapeRe).join(".*") + "$").test(command);
  return command === spec; // exact: Bash(make lint)
}

function matchPath(spec: string, filePath: string): boolean {
  let p = filePath.replace(/\\/g, "/");
  while (p.startsWith("./")) p = p.slice(2);
  if (spec.startsWith("~/") || spec.startsWith("//")) return globToRegex(spec).test(p);
  let anchored: string;
  if (spec.startsWith("./")) anchored = spec.slice(2);
  else if (spec.startsWith("/")) anchored = spec.slice(1); // relative to the settings file's project root
  else if (!spec.includes("/")) anchored = "**/" + spec; // bare name -> any depth (gitignore style)
  else anchored = spec;
  return globToRegex(anchored).test(p);
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return "";
  }
}

function matchDomain(spec: string, url: string): boolean {
  if (!spec.startsWith("domain:")) return false;
  const host = hostOf(url);
  const domain = spec.slice("domain:".length).toLowerCase();
  return host === domain || host.endsWith("." + domain);
}

export function ruleMatches(rule: string, tool: string, toolInput: Json, subCommand: string | null = null): boolean {
  const parsed = parseRule(rule);
  if (!parsed) return false; // invalid rules are ignored by the product - the linter flags them
  const [rtool, spec] = parsed;
  if (rtool.startsWith("mcp__")) {
    if (spec !== null) return false;
    if (rtool.endsWith("__*")) return tool.startsWith(rtool.slice(0, -1));
    return tool === rtool || tool.startsWith(rtool + "__");
  }
  const applies = rtool === "Edit" ? EDIT_TOOLS.has(tool) : rtool === "Read" ? READ_TOOLS.has(tool) : tool === rtool;
  if (!applies) return false;
  if (spec === null || spec === "*" || spec === "**") return true;
  if (tool === "Bash") return matchBash(spec, subCommand ?? String(toolInput.command ?? ""));
  if (EDIT_TOOLS.has(tool) || READ_TOOLS.has(tool)) return matchPath(spec, String(toolInput.file_path ?? toolInput.path ?? ""));
  if (tool === "WebFetch") return matchDomain(spec, String(toolInput.url ?? ""));
  return false;
}

export interface RuleRef {
  kind: "allow" | "ask" | "deny";
  rule: string;
  source: string; // which settings layer contributed it
}

export interface Decision {
  decision: "allow" | "ask" | "deny";
  rule: string | null;
  source: string | null;
  reason: string;
  subCommand: string | null;
}

const DECISION_RANK: Record<string, number> = { deny: 0, ask: 1, allow: 2 };
const KINDS = ["deny", "ask", "allow"] as const;

function evaluateSingle(rules: RuleRef[], tool: string, toolInput: Json, mode: string, sub: string | null): Decision {
  // Order matters and is the #1 exam fact: deny is checked first, then ask, then allow.
  for (const kind of KINDS) {
    for (const ref of rules) {
      if (ref.kind === kind && ruleMatches(ref.rule, tool, toolInput, sub)) {
        return { decision: kind, rule: ref.rule, source: ref.source, reason: `matched ${kind} rule`, subCommand: sub };
      }
    }
  }
  // No rule matched -> the permission mode decides.
  const d = (decision: Decision["decision"], reason: string): Decision => ({ decision, rule: null, source: null, reason, subCommand: sub });
  if (READ_TOOLS.has(tool)) return d("allow", "read-only tool: no prompt by default");
  if (mode === "plan") return d("deny", "plan mode: no edits or commands");
  if (mode === "bypassPermissions") return d("allow", "bypassPermissions mode");
  if (mode === "acceptEdits" && EDIT_TOOLS.has(tool)) return d("allow", "acceptEdits mode auto-approves file edits");
  return d("ask", `no rule matched; '${mode}' mode prompts the user`);
}

export function evaluate(rules: RuleRef[], tool: string, toolInput: Json, mode = "default"): Decision {
  if (tool !== "Bash") return evaluateSingle(rules, tool, toolInput, mode, null);
  const subs = splitCompound(String(toolInput.command ?? ""));
  const decisions = subs.length
    ? subs.map((s) => evaluateSingle(rules, tool, toolInput, mode, s))
    : [{ decision: "ask", rule: null, source: null, reason: "empty command", subCommand: null } as Decision];
  // The most restrictive sub-command decides the whole line (first one wins ties, like Python's min()).
  return decisions.reduce((best, cur) => (DECISION_RANK[cur.decision]! < DECISION_RANK[best.decision]! ? cur : best));
}

export function rulesFromSettings(settings: Json, source: string): RuleRef[] {
  const perms: Json = settings.permissions ?? {};
  const out: RuleRef[] = [];
  for (const kind of ["allow", "ask", "deny"] as const) for (const r of perms[kind] ?? []) out.push({ kind, rule: r, source });
  return out;
}

// ---------------------------------------------------------------------------
// Settings precedence: merge layers with provenance
// ---------------------------------------------------------------------------
/** Lowest -> highest precedence. Managed policy is last so nothing can override it. */
export const PRECEDENCE = ["user", "project", "local", "cli", "managed"];

export interface Effective {
  values: Record<string, unknown>; // dotted key -> value
  source: Record<string, string>; // dotted key -> winning layer
  history: Record<string, [string, unknown][]>; // dotted key -> [(layer, value)]
  rules: RuleRef[];
  hooks: Record<string, Json[]>;
  notes: string[];
}

/** Scalars: highest-precedence layer wins. Permission rules and hooks: combined from every layer. */
export function mergeLayers(layers: [string, Json][]): Effective {
  const eff: Effective = { values: {}, source: {}, history: {}, rules: [], hooks: {}, notes: [] };

  const walk = (node: Json, prefix: string, layer: string): void => {
    for (const [key, value] of Object.entries(node)) {
      if (key.startsWith("_") || key === "$schema") continue;
      const dotted = `${prefix}${key}`;
      if (dotted === "permissions.allow" || dotted === "permissions.ask" || dotted === "permissions.deny") {
        const kind = dotted.split(".")[1] as RuleRef["kind"];
        for (const r of value as string[]) eff.rules.push({ kind, rule: r, source: layer });
      } else if (dotted === "hooks") {
        for (const [event, entries] of Object.entries(value as Record<string, Json[]>)) {
          (eff.hooks[event] ??= []).push(...entries.map((e) => ({ ...e, _source: layer })));
        }
      } else if (value && typeof value === "object" && !Array.isArray(value)) {
        walk(value as Json, dotted + ".", layer);
      } else {
        eff.values[dotted] = value;
        eff.source[dotted] = layer;
        (eff.history[dotted] ??= []).push([layer, value]);
      }
    }
  };

  const ordered = [...layers].sort((a, b) => PRECEDENCE.indexOf(a[0]) - PRECEDENCE.indexOf(b[0]));
  for (const [layer, settings] of ordered) walk(settings, "", layer);

  // Managed policy can switch bypass mode off for everyone, whatever lower layers ask for.
  if (eff.values["permissions.disableBypassPermissionsMode"] === "disable" && eff.values["permissions.defaultMode"] === "bypassPermissions") {
    const requestedBy = eff.source["permissions.defaultMode"];
    eff.values["permissions.defaultMode"] = "default";
    eff.source["permissions.defaultMode"] = "managed";
    eff.notes.push(`'${requestedBy}' asked for bypassPermissions; managed policy disables it -> 'default'`);
  }
  return eff;
}

// ---------------------------------------------------------------------------
// Linters
// ---------------------------------------------------------------------------
const mask = (text: string): string => (text.length > 8 ? text.slice(0, 8) + "..." : "***");

export function lintSettings(settings: Json, target: string, projectDir: string | null = null): Finding[] {
  const out: Finding[] = [];
  const perms: Json = settings.permissions ?? {};
  const rules = rulesFromSettings(settings, target);
  const mode: string = perms.defaultMode ?? "default";

  for (const ref of rules) {
    const parsed = parseRule(ref.rule);
    if (!parsed) {
      out.push(finding("ERROR", "S008", target, `invalid rule syntax "${ref.rule}" - it is silently ignored`));
      continue;
    }
    const [tool, spec] = parsed;
    if (!tool.startsWith("mcp__") && !KNOWN_TOOLS.has(tool)) {
      out.push(finding("WARN", "S008", target, `unknown tool "${tool}" in "${ref.rule}" (tool names are case-sensitive)`));
    }
    if (ref.kind === "allow" && tool === "Bash" && (spec === null || ["*", ":*", "**"].includes(spec.trim()))) {
      out.push(finding("ERROR", "S001", target, `allow "${ref.rule}" lets ANY shell command run without approval`));
    } else if (ref.kind === "allow" && EDIT_TOOLS.has(tool) && spec === null) {
      out.push(finding("WARN", "S001", target, `allow "${ref.rule}" approves edits to every file; scope it, e.g. Edit(./src/**)`));
    }
  }

  const allowSet = new Set(rules.filter((r) => r.kind === "allow").map((r) => r.rule));
  for (const r of rules) {
    if ((r.kind === "deny" || r.kind === "ask") && allowSet.has(r.rule)) {
      out.push(finding("WARN", "S005", target, `"${r.rule}" is in allow AND ${r.kind}; ${r.kind} wins, the allow entry is dead`));
    }
  }

  // Probe the policy with the evaluator itself: can secrets be read? are destructive commands blocked?
  const readable = [".env", ".env.production", "secrets/prod.key"].filter(
    (p) => evaluate(rules, "Read", { file_path: p }, mode).decision !== "deny",
  );
  if (readable.length) out.push(finding("ERROR", "S002", target, `secret files readable (no deny rule): ${readable.join(", ")}`));
  for (const cmd of ["rm -rf /", "git push --force origin main", "git reset --hard HEAD~3"]) {
    const d = evaluate(rules, "Bash", { command: cmd }, mode);
    if (d.decision === "allow") out.push(finding("ERROR", "S003", target, `destructive command "${cmd}" would run without approval`));
    else if (d.decision === "ask") out.push(finding("WARN", "S003", target, `destructive command "${cmd}" is not explicitly denied`));
  }

  if (mode === "bypassPermissions") {
    out.push(finding("ERROR", "S004", target, "defaultMode bypassPermissions in shared settings disables every prompt"));
  }

  for (const [key, value] of Object.entries((settings.env ?? {}) as Record<string, unknown>)) {
    if (findSecrets(String(value)).length || isLiteralSecret(key, String(value))) {
      out.push(finding("ERROR", "S009", target, `env.${key} holds a literal secret (${mask(String(value))}); use a secret manager or apiKeyHelper`));
    }
  }

  for (const [event, entries] of Object.entries((settings.hooks ?? {}) as Record<string, Json[]>)) {
    if (!HOOK_EVENTS.has(event)) out.push(finding("WARN", "S010", target, `unknown hook event "${event}" - it will never fire`));
    for (const entry of entries) {
      for (const hook of (entry.hooks ?? []) as Json[]) {
        const cmd = String(hook.command ?? "");
        if (NETWORK_IN_HOOK.test(cmd)) out.push(finding("ERROR", "S006", target, `${event} hook calls the network (data exfiltration risk): ${cmd.slice(0, 60)}`));
        if (DESTRUCTIVE_IN_HOOK.test(cmd)) out.push(finding("ERROR", "S007", target, `${event} hook runs a destructive command: ${cmd.slice(0, 60)}`));
        const m = /\$CLAUDE_PROJECT_DIR\/([^"'\s]+)/.exec(cmd);
        if (projectDir !== null && m && !existsSync(path.join(projectDir, m[1]!))) {
          out.push(finding("ERROR", "S011", target, `${event} hook script missing: ${m[1]}`));
        }
      }
    }
  }
  return out;
}

/** '${VAR:-https://x}' -> 'https://x'; '${VAR}' -> null (unknown until runtime). */
function resolveDefault(value: string): string | null {
  const m = /^\$\{[A-Za-z_][A-Za-z0-9_]*(?::-(.*))?\}$/.exec(value);
  if (m) return m[1] ?? null;
  return value;
}

export function lintMcp(config: Json, target: string): Finding[] {
  const out: Finding[] = [];
  for (const [name, srv] of Object.entries((config.mcpServers ?? {}) as Record<string, Json>)) {
    const stype: string = srv.type ?? ("url" in srv ? "http" : "stdio");
    const label = `${target}#${name}`;
    if (stype === "stdio" && !srv.command) out.push(finding("ERROR", "M004", label, "stdio server has no command"));
    if ((stype === "http" || stype === "sse") && !srv.url) out.push(finding("ERROR", "M004", label, `${stype} server has no url`));

    // .mcp.json is committed to git: every credential must come from ${ENV_VAR} expansion.
    for (const sectionName of ["headers", "env"]) {
      for (const [key, value] of Object.entries((srv[sectionName] ?? {}) as Record<string, unknown>)) {
        if (findSecrets(String(value)).length || isLiteralSecret(key, String(value))) {
          out.push(finding("ERROR", "M001", label, `${sectionName}.${key} is a literal secret (${mask(String(value))}); use \${ENV_VAR} expansion`));
        }
      }
    }
    const strings: [string, string][] = [["url", String(srv.url ?? "")], ["command", String(srv.command ?? "")]];
    ((srv.args ?? []) as unknown[]).forEach((a, i) => strings.push([`args[${i}]`, String(a)]));
    for (const [where, value] of strings) {
      for (const kind of findSecrets(value)) out.push(finding("ERROR", "M001", label, `${kind} in ${where} (${mask(value)})`));
    }

    const url = resolveDefault(String(srv.url ?? ""));
    if (url) {
      let protocol = "";
      let host = "";
      try {
        const u = new URL(url);
        protocol = u.protocol;
        host = u.hostname;
      } catch {
        /* not a URL - nothing to check */
      }
      if (protocol === "http:" && host !== "localhost" && host !== "127.0.0.1") {
        out.push(finding("WARN", "M002", label, `remote server over plain http: ${url}`));
      }
    }
    if (["npx", "uvx", "bunx"].includes(srv.command)) {
      for (const arg of (srv.args ?? []) as string[]) {
        if (arg.startsWith("-")) continue;
        const bare = arg.startsWith("@") ? arg.slice(1) : arg;
        if (arg.endsWith("@latest") || !bare.includes("@")) {
          out.push(finding("INFO", "M003", label, `unpinned package "${arg}" - pin a version for reproducible, reviewable installs`));
        }
        break;
      }
    }
  }
  return out;
}

export function lintAgent(text: string, target: string): Finding[] {
  const [meta] = parseFrontmatter(text);
  const out: Finding[] = [];
  if (!meta.name || !meta.description) {
    out.push(finding("ERROR", "A001", target, "subagent needs frontmatter name AND description (description drives delegation)"));
  }
  const tools = (meta.tools ?? "").split(",").map((t) => t.trim()).filter(Boolean);
  if (!tools.length) out.push(finding("WARN", "A002", target, "no tools field: the subagent inherits ALL tools (least privilege violated)"));
  const isReviewer = /review|audit/i.test(`${meta.name ?? ""} ${meta.description ?? ""}`);
  const writeTools = tools.filter((t) => WRITE_CAPABLE.has(t.split("(")[0]!));
  if (isReviewer && (writeTools.length || !tools.length)) {
    out.push(finding("WARN", "A003", target, `review/audit agent can modify code (${writeTools.join(", ") || "inherited tools"}); make it read-only`));
  }
  const model = meta.model ?? "";
  if (model && !["sonnet", "opus", "haiku", "inherit"].includes(model) && !model.startsWith("claude-")) {
    out.push(finding("WARN", "A004", target, `unrecognised model "${model}"`));
  }
  return out;
}

export function lintSkill(text: string, target: string, folderName: string | null = null): Finding[] {
  const [meta] = parseFrontmatter(text);
  const out: Finding[] = [];
  const name = meta.name ?? "";
  const desc = meta.description ?? "";
  if (!name || !desc) out.push(finding("ERROR", "K001", target, "SKILL.md needs frontmatter name and description"));
  if (name && (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(name) || name.length > 64)) {
    out.push(finding("ERROR", "K002", target, `skill name "${name}" must be lowercase letters, digits, hyphens (<= 64 chars)`));
  }
  if (desc && (desc.length < 40 || !desc.toLowerCase().includes("when"))) {
    out.push(finding("WARN", "K003", target, "description should say WHAT the skill does and WHEN to use it - it is all Claude sees before loading"));
  }
  if (folderName && name && folderName !== name) out.push(finding("WARN", "K004", target, `folder "${folderName}" does not match skill name "${name}"`));
  return out;
}

export function lintCommand(text: string, target: string): Finding[] {
  const [meta, body] = parseFrontmatter(text);
  const out: Finding[] = [];
  if (!meta.description) out.push(finding("WARN", "X001", target, "no description frontmatter (shown in the / menu)"));
  if (body.includes("$ARGUMENTS") && !meta["argument-hint"]) out.push(finding("INFO", "X002", target, "uses $ARGUMENTS but has no argument-hint"));
  if (/deploy|production|publish|push/i.test(body) && !meta["allowed-tools"]) {
    out.push(finding("WARN", "X003", target, "high-impact command without allowed-tools scoping or a confirmation step"));
  }
  return out;
}

const VAGUE = /\b(be helpful|write good code|follow best practices|be careful)\b/i;

export function lintMemory(text: string, target: string, baseDir: string): Finding[] {
  const out: Finding[] = [];
  for (const kind of findSecrets(text)) {
    out.push(finding("ERROR", "C001", target, `${kind} in memory file - CLAUDE.md is committed and sent to the model every session`));
  }
  if (!/^#+\s.*command/im.test(text)) {
    out.push(finding("WARN", "C002", target, "no Commands section (build/test/lint commands are the highest-value memory)"));
  }
  const scrubbed = text.replace(/`[^`]*`/g, ""); // ignore @mentions inside inline code
  for (const m of scrubbed.matchAll(/(?<![\w@])@([\w./-]+\.md)\b/g)) {
    if (!existsSync(path.join(baseDir, m[1]!))) out.push(finding("ERROR", "C003", target, `@import target not found: ${m[1]}`));
  }
  const lines = splitLines(text);
  if (lines.length > 200) {
    out.push(finding("WARN", "C004", target, `${lines.length} lines - memory is loaded every session; move detail into imports or skills`));
  }
  for (const line of lines) {
    if (VAGUE.test(line)) out.push(finding("WARN", "C005", target, `vague instruction "${line.trim().slice(0, 50)}" - replace with a specific, checkable rule`));
  }
  return out;
}
