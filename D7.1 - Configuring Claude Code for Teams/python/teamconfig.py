"""
Team configuration toolkit for Claude Code (D7.1).

Pure, dependency-free logic that the demo in main.py drives:
  * a linter for settings.json, .mcp.json, subagents, skills, slash commands, CLAUDE.md
  * a settings-layer merger that records provenance (which scope won each key)
  * a permission-rule evaluator (deny > ask > allow, then the permission mode default)

This is a *teaching model* of Claude Code's behaviour, faithful to the documented rules
(rule syntax, precedence, deny-wins) but simplified where the real product has more
nuance (e.g. full gitignore path semantics, every built-in tool, per-tool modes).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}

# Obvious credential shapes. Real teams add a dedicated secret scanner in CI as well.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic-api-key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{10,}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password-in-uri", re.compile(r"://[^/\s:@]+:[^/\s@$]+@")),
]
SENSITIVE_KEY = re.compile(r"(token|secret|password|passwd|api[_-]?key|authorization)", re.IGNORECASE)

KNOWN_TOOLS = {
    "Bash", "Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Glob", "Grep", "LS",
    "WebFetch", "WebSearch", "Task", "Agent", "Skill", "TodoWrite",
}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}   # Edit(...) rules cover all of these
READ_TOOLS = {"Read", "Grep", "Glob", "LS"}                    # Read(...) rules cover all of these
WRITE_CAPABLE = {"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash"}
HOOK_EVENTS = {
    "PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SubagentStop",
    "SessionStart", "SessionEnd", "Notification", "PreCompact",
}
NETWORK_IN_HOOK = re.compile(r"\b(curl|wget|nc|ncat|Invoke-WebRequest|Invoke-RestMethod|iwr)\b|https?://", re.IGNORECASE)
DESTRUCTIVE_IN_HOOK = re.compile(r"rm\s+-rf|Remove-Item\b.*-Recurse|git\s+reset\s+--hard|git\s+clean\s+-[a-z]*f|del\s+/s", re.IGNORECASE)
RULE_RE = re.compile(r"^(?P<tool>[A-Za-z][A-Za-z0-9_-]*)(?:\((?P<spec>.*)\))?$")


@dataclass
class Finding:
    severity: str  # ERROR | WARN | INFO
    code: str
    target: str
    message: str


def find_secrets(text: str) -> list[str]:
    return [name for name, rx in SECRET_PATTERNS if rx.search(text or "")]


def is_literal_secret(key: str, value: str) -> bool:
    """A sensitive-looking key holding a literal (not ${VAR}-expanded) value."""
    return bool(SENSITIVE_KEY.search(key)) and bool(value) and "${" not in value


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """gitignore-flavoured glob: '**/' spans directories, '*' and '?' stay within one segment."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Tiny YAML-frontmatter reader (flat `key: value` pairs only - enough for agents/skills/commands)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    meta: dict[str, str] = {}
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == "---":
            return meta, "\n".join(lines[idx + 1 :])
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return {}, text  # unterminated frontmatter -> treat as plain body


# ---------------------------------------------------------------------------
# Permission rules
# ---------------------------------------------------------------------------
def parse_rule(rule: str) -> tuple[str, str | None] | None:
    m = RULE_RE.match(rule.strip())
    if not m:
        return None
    return m.group("tool"), m.group("spec")


def split_compound(command: str) -> list[str]:
    """Split a shell line on && || ; | and newlines (outside quotes).

    Claude Code evaluates each sub-command separately, so an allow rule for
    `git status` can never smuggle in `git status && curl ... | sh`.
    """
    parts, buf, quote, i = [], [], "", 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        two = command[i : i + 2]
        if two in ("&&", "||"):
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in (";", "|", "\n"):
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [" ".join(p.split()) for p in parts if p.strip()]


def _match_bash(spec: str, command: str) -> bool:
    if spec.endswith(":*"):                       # legacy prefix syntax: Bash(npm run test:*)
        return command.startswith(spec[:-2])
    if "*" in spec:                               # wildcard syntax: Bash(npm run *)
        rx = "^" + ".*".join(re.escape(p) for p in spec.split("*")) + "$"
        return re.match(rx, command) is not None
    return command == spec                        # exact match: Bash(make lint)


def _match_path(spec: str, file_path: str) -> bool:
    path = file_path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if spec.startswith("~/") or spec.startswith("//"):
        return glob_to_regex(spec).match(path) is not None   # home / absolute: rarely matches project paths
    if spec.startswith("./"):
        anchored = spec[2:]
    elif spec.startswith("/"):
        anchored = spec[1:]                       # relative to the settings file's project root
    elif "/" not in spec:
        anchored = "**/" + spec                   # bare name -> any depth (gitignore style)
    else:
        anchored = spec
    return glob_to_regex(anchored).match(path) is not None


def _match_domain(spec: str, url: str) -> bool:
    if not spec.startswith("domain:"):
        return False
    host = (urlparse(url).hostname or "").lower()
    domain = spec[len("domain:") :].lower()
    return host == domain or host.endswith("." + domain)


def rule_matches(rule: str, tool: str, tool_input: dict[str, Any], sub_command: str | None = None) -> bool:
    parsed = parse_rule(rule)
    if not parsed:
        return False  # invalid rules are ignored by the product - the linter flags them
    rtool, spec = parsed
    if rtool.startswith("mcp__"):
        if spec is not None:
            return False
        if rtool.endswith("__*"):
            return tool.startswith(rtool[:-1])
        return tool == rtool or tool.startswith(rtool + "__")
    if rtool == "Edit":
        applies = tool in EDIT_TOOLS
    elif rtool == "Read":
        applies = tool in READ_TOOLS
    else:
        applies = tool == rtool
    if not applies:
        return False
    if spec is None or spec in ("*", "**"):
        return True
    if tool == "Bash":
        return _match_bash(spec, sub_command if sub_command is not None else tool_input.get("command", ""))
    if tool in EDIT_TOOLS or tool in READ_TOOLS:
        return _match_path(spec, str(tool_input.get("file_path") or tool_input.get("path") or ""))
    if tool == "WebFetch":
        return _match_domain(spec, str(tool_input.get("url", "")))
    return False


@dataclass
class RuleRef:
    kind: str     # allow | ask | deny
    rule: str
    source: str   # which settings layer contributed it


@dataclass
class Decision:
    decision: str            # allow | ask | deny
    rule: str | None
    source: str | None
    reason: str
    sub_command: str | None = None


DECISION_RANK = {"deny": 0, "ask": 1, "allow": 2}


def _evaluate_single(rules: list[RuleRef], tool: str, tool_input: dict[str, Any], mode: str, sub: str | None) -> Decision:
    # Order matters and is the #1 exam fact: deny is checked first, then ask, then allow.
    for kind in ("deny", "ask", "allow"):
        for ref in rules:
            if ref.kind == kind and rule_matches(ref.rule, tool, tool_input, sub):
                return Decision(kind, ref.rule, ref.source, f"matched {kind} rule", sub)
    # No rule matched -> the permission mode decides.
    if tool in READ_TOOLS:
        return Decision("allow", None, None, "read-only tool: no prompt by default", sub)
    if mode == "plan":
        return Decision("deny", None, None, "plan mode: no edits or commands", sub)
    if mode == "bypassPermissions":
        return Decision("allow", None, None, "bypassPermissions mode", sub)
    if mode == "acceptEdits" and tool in EDIT_TOOLS:
        return Decision("allow", None, None, "acceptEdits mode auto-approves file edits", sub)
    return Decision("ask", None, None, f"no rule matched; '{mode}' mode prompts the user", sub)


def evaluate(rules: list[RuleRef], tool: str, tool_input: dict[str, Any], mode: str = "default") -> Decision:
    if tool != "Bash":
        return _evaluate_single(rules, tool, tool_input, mode, None)
    subs = split_compound(str(tool_input.get("command", "")))
    decisions = [_evaluate_single(rules, tool, tool_input, mode, s) for s in subs] or [
        Decision("ask", None, None, "empty command")
    ]
    # The most restrictive sub-command decides the whole line.
    return min(decisions, key=lambda d: DECISION_RANK[d.decision])


def rules_from_settings(settings: dict[str, Any], source: str) -> list[RuleRef]:
    perms = settings.get("permissions") or {}
    return [RuleRef(kind, r, source) for kind in ("allow", "ask", "deny") for r in perms.get(kind) or []]


# ---------------------------------------------------------------------------
# Settings precedence: merge layers with provenance
# ---------------------------------------------------------------------------
# Lowest -> highest precedence. Managed policy is last so nothing can override it.
PRECEDENCE = ["user", "project", "local", "cli", "managed"]


@dataclass
class Effective:
    values: dict[str, Any] = field(default_factory=dict)                     # dotted key -> value
    source: dict[str, str] = field(default_factory=dict)                     # dotted key -> winning layer
    history: dict[str, list[tuple[str, Any]]] = field(default_factory=dict)  # dotted key -> [(layer, value)]
    rules: list[RuleRef] = field(default_factory=list)
    hooks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def merge_layers(layers: list[tuple[str, dict[str, Any]]]) -> Effective:
    """Scalars: highest-precedence layer wins. Permission rules and hooks: combined from every layer."""
    eff = Effective()

    def walk(node: dict[str, Any], prefix: str, layer: str) -> None:
        for key, value in node.items():
            if key.startswith("_") or key == "$schema":
                continue
            dotted = f"{prefix}{key}"
            if dotted in ("permissions.allow", "permissions.ask", "permissions.deny"):
                kind = dotted.split(".")[1]
                eff.rules.extend(RuleRef(kind, r, layer) for r in value)
            elif dotted == "hooks":
                for event, entries in value.items():
                    eff.hooks.setdefault(event, []).extend({**e, "_source": layer} for e in entries)
            elif isinstance(value, dict):
                walk(value, dotted + ".", layer)
            else:
                eff.values[dotted] = value
                eff.source[dotted] = layer
                eff.history.setdefault(dotted, []).append((layer, value))

    for layer, settings in sorted(layers, key=lambda lv: PRECEDENCE.index(lv[0])):
        walk(settings, "", layer)

    # Managed policy can switch bypass mode off for everyone, whatever lower layers ask for.
    if eff.values.get("permissions.disableBypassPermissionsMode") == "disable" and \
            eff.values.get("permissions.defaultMode") == "bypassPermissions":
        requested_by = eff.source["permissions.defaultMode"]
        eff.values["permissions.defaultMode"] = "default"
        eff.source["permissions.defaultMode"] = "managed"
        eff.notes.append(f"'{requested_by}' asked for bypassPermissions; managed policy disables it -> 'default'")
    return eff


# ---------------------------------------------------------------------------
# Linters
# ---------------------------------------------------------------------------
def _mask(text: str) -> str:
    return text[:8] + "..." if len(text) > 8 else "***"


def lint_settings(settings: dict[str, Any], target: str, project_dir: Path | None = None) -> list[Finding]:
    out: list[Finding] = []
    perms = settings.get("permissions") or {}
    rules = rules_from_settings(settings, target)
    mode = perms.get("defaultMode", "default")

    for ref in rules:
        parsed = parse_rule(ref.rule)
        if parsed is None:
            out.append(Finding("ERROR", "S008", target, f'invalid rule syntax "{ref.rule}" - it is silently ignored'))
            continue
        tool, spec = parsed
        if not tool.startswith("mcp__") and tool not in KNOWN_TOOLS:
            out.append(Finding("WARN", "S008", target, f'unknown tool "{tool}" in "{ref.rule}" (tool names are case-sensitive)'))
        if ref.kind == "allow" and tool == "Bash" and (spec is None or spec.strip() in ("*", ":*", "**")):
            out.append(Finding("ERROR", "S001", target, f'allow "{ref.rule}" lets ANY shell command run without approval'))
        elif ref.kind == "allow" and tool in EDIT_TOOLS and spec is None:
            out.append(Finding("WARN", "S001", target, f'allow "{ref.rule}" approves edits to every file; scope it, e.g. Edit(./src/**)'))

    allow_set = {r.rule for r in rules if r.kind == "allow"}
    for r in rules:
        if r.kind in ("deny", "ask") and r.rule in allow_set:
            out.append(Finding("WARN", "S005", target, f'"{r.rule}" is in allow AND {r.kind}; {r.kind} wins, the allow entry is dead'))

    # Probe the policy with the evaluator itself: can secrets be read? are destructive commands blocked?
    readable = [p for p in (".env", ".env.production", "secrets/prod.key")
                if evaluate(rules, "Read", {"file_path": p}, mode).decision != "deny"]
    if readable:
        out.append(Finding("ERROR", "S002", target, f"secret files readable (no deny rule): {', '.join(readable)}"))
    for cmd in ("rm -rf /", "git push --force origin main", "git reset --hard HEAD~3"):
        d = evaluate(rules, "Bash", {"command": cmd}, mode)
        if d.decision == "allow":
            out.append(Finding("ERROR", "S003", target, f'destructive command "{cmd}" would run without approval'))
        elif d.decision == "ask":
            out.append(Finding("WARN", "S003", target, f'destructive command "{cmd}" is not explicitly denied'))

    if mode == "bypassPermissions":
        out.append(Finding("ERROR", "S004", target, "defaultMode bypassPermissions in shared settings disables every prompt"))

    for key, value in (settings.get("env") or {}).items():
        if find_secrets(str(value)) or is_literal_secret(key, str(value)):
            out.append(Finding("ERROR", "S009", target, f"env.{key} holds a literal secret ({_mask(str(value))}); use a secret manager or apiKeyHelper"))

    for event, entries in (settings.get("hooks") or {}).items():
        if event not in HOOK_EVENTS:
            out.append(Finding("WARN", "S010", target, f'unknown hook event "{event}" - it will never fire'))
        for entry in entries:
            for hook in entry.get("hooks", []):
                cmd = str(hook.get("command", ""))
                if NETWORK_IN_HOOK.search(cmd):
                    out.append(Finding("ERROR", "S006", target, f"{event} hook calls the network (data exfiltration risk): {cmd[:60]}"))
                if DESTRUCTIVE_IN_HOOK.search(cmd):
                    out.append(Finding("ERROR", "S007", target, f"{event} hook runs a destructive command: {cmd[:60]}"))
                m = re.search(r"\$CLAUDE_PROJECT_DIR/([^\"'\s]+)", cmd)
                if project_dir is not None and m and not (project_dir / m.group(1)).exists():
                    out.append(Finding("ERROR", "S011", target, f"{event} hook script missing: {m.group(1)}"))
    return out


def _resolve_default(value: str) -> str | None:
    """'${VAR:-https://x}' -> 'https://x'; '${VAR}' -> None (unknown until runtime)."""
    m = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-(.*))?\}", value)
    if m:
        return m.group(1)
    return value


def lint_mcp(config: dict[str, Any], target: str) -> list[Finding]:
    out: list[Finding] = []
    for name, srv in (config.get("mcpServers") or {}).items():
        stype = srv.get("type") or ("http" if "url" in srv else "stdio")
        label = f"{target}#{name}"
        if stype == "stdio" and not srv.get("command"):
            out.append(Finding("ERROR", "M004", label, "stdio server has no command"))
        if stype in ("http", "sse") and not srv.get("url"):
            out.append(Finding("ERROR", "M004", label, f"{stype} server has no url"))

        # .mcp.json is committed to git: every credential must come from ${ENV_VAR} expansion.
        for section in ("headers", "env"):
            for key, value in (srv.get(section) or {}).items():
                if find_secrets(str(value)) or is_literal_secret(key, str(value)):
                    out.append(Finding("ERROR", "M001", label, f"{section}.{key} is a literal secret ({_mask(str(value))}); use ${{ENV_VAR}} expansion"))
        strings = [("url", str(srv.get("url", ""))), ("command", str(srv.get("command", "")))]
        strings += [(f"args[{i}]", str(a)) for i, a in enumerate(srv.get("args") or [])]
        for where, value in strings:
            for kind in find_secrets(value):
                out.append(Finding("ERROR", "M001", label, f"{kind} in {where} ({_mask(value)})"))

        url = _resolve_default(str(srv.get("url", "")))
        if url:
            parsed = urlparse(url)
            if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1"):
                out.append(Finding("WARN", "M002", label, f"remote server over plain http: {url}"))
        if srv.get("command") in ("npx", "uvx", "bunx"):
            for arg in srv.get("args") or []:
                if arg.startswith("-"):
                    continue
                bare = arg[1:] if arg.startswith("@") else arg
                if arg.endswith("@latest") or "@" not in bare:
                    out.append(Finding("INFO", "M003", label, f'unpinned package "{arg}" - pin a version for reproducible, reviewable installs'))
                break
    return out


def lint_agent(text: str, target: str) -> list[Finding]:
    meta, body = parse_frontmatter(text)
    out: list[Finding] = []
    if not meta.get("name") or not meta.get("description"):
        out.append(Finding("ERROR", "A001", target, "subagent needs frontmatter name AND description (description drives delegation)"))
    tools = [t.strip() for t in meta.get("tools", "").split(",") if t.strip()]
    if not tools:
        out.append(Finding("WARN", "A002", target, "no tools field: the subagent inherits ALL tools (least privilege violated)"))
    is_reviewer = re.search(r"review|audit", meta.get("name", "") + " " + meta.get("description", ""), re.IGNORECASE)
    write_tools = [t for t in tools if t.split("(")[0] in WRITE_CAPABLE]
    if is_reviewer and (write_tools or not tools):
        out.append(Finding("WARN", "A003", target, f"review/audit agent can modify code ({', '.join(write_tools) or 'inherited tools'}); make it read-only"))
    model = meta.get("model", "")
    if model and model not in ("sonnet", "opus", "haiku", "inherit") and not model.startswith("claude-"):
        out.append(Finding("WARN", "A004", target, f'unrecognised model "{model}"'))
    return out


def lint_skill(text: str, target: str, folder_name: str | None = None) -> list[Finding]:
    meta, _ = parse_frontmatter(text)
    out: list[Finding] = []
    name, desc = meta.get("name", ""), meta.get("description", "")
    if not name or not desc:
        out.append(Finding("ERROR", "K001", target, "SKILL.md needs frontmatter name and description"))
    if name and (not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) or len(name) > 64):
        out.append(Finding("ERROR", "K002", target, f'skill name "{name}" must be lowercase letters, digits, hyphens (<= 64 chars)'))
    if desc and (len(desc) < 40 or "when" not in desc.lower()):
        out.append(Finding("WARN", "K003", target, "description should say WHAT the skill does and WHEN to use it - it is all Claude sees before loading"))
    if folder_name and name and folder_name != name:
        out.append(Finding("WARN", "K004", target, f'folder "{folder_name}" does not match skill name "{name}"'))
    return out


def lint_command(text: str, target: str) -> list[Finding]:
    meta, body = parse_frontmatter(text)
    out: list[Finding] = []
    if not meta.get("description"):
        out.append(Finding("WARN", "X001", target, "no description frontmatter (shown in the / menu)"))
    if "$ARGUMENTS" in body and not meta.get("argument-hint"):
        out.append(Finding("INFO", "X002", target, "uses $ARGUMENTS but has no argument-hint"))
    if re.search(r"deploy|production|publish|push", body, re.IGNORECASE) and not meta.get("allowed-tools"):
        out.append(Finding("WARN", "X003", target, "high-impact command without allowed-tools scoping or a confirmation step"))
    return out


VAGUE = re.compile(r"\b(be helpful|write good code|follow best practices|be careful)\b", re.IGNORECASE)


def lint_memory(text: str, target: str, base_dir: Path) -> list[Finding]:
    out: list[Finding] = []
    for kind in find_secrets(text):
        out.append(Finding("ERROR", "C001", target, f"{kind} in memory file - CLAUDE.md is committed and sent to the model every session"))
    if not re.search(r"^#+\s.*command", text, re.IGNORECASE | re.MULTILINE):
        out.append(Finding("WARN", "C002", target, "no Commands section (build/test/lint commands are the highest-value memory)"))
    scrubbed = re.sub(r"`[^`]*`", "", text)  # ignore @mentions inside inline code
    for ref in re.findall(r"(?<![\w@])@([\w./-]+\.md)\b", scrubbed):
        if not (base_dir / ref).exists():
            out.append(Finding("ERROR", "C003", target, f"@import target not found: {ref}"))
    lines = text.splitlines()
    if len(lines) > 200:
        out.append(Finding("WARN", "C004", target, f"{len(lines)} lines - memory is loaded every session; move detail into imports or skills"))
    for line in lines:
        if VAGUE.search(line):
            out.append(Finding("WARN", "C005", target, f'vague instruction "{line.strip()[:50]}" - replace with a specific, checkable rule'))
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
