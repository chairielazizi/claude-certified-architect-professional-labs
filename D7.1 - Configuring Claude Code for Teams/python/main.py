"""
D7.1 - Configuring Claude Code for Teams

A "team config linter + permission simulator" for the example Claude Code setup in
../example-project:
  1. inventory of the shared team setup (settings, hooks, CLAUDE.md, MCP, subagent, skill, command)
  2. lint the team config (expected: clean) and a deliberately risky config (expected: many findings)
  3. merge five settings layers (user < project < local < CLI < managed) with provenance
  4. evaluate sample tool requests: PreToolUse hook -> deny > ask > allow -> permission mode
  5. run the real hook scripts with Claude Code's stdin/exit-code contract
  6. bonus: Claude drafts a CLAUDE.md from a repo summary, then the linter checks the draft

Nothing here touches your real Claude Code configuration; everything is read from this lab folder.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import MODEL_OPUS, MockRequest, banner, get_client, section, text_of, usage_line  # noqa: E402

from teamconfig import (  # noqa: E402
    SEVERITY_ORDER, Finding, evaluate, lint_agent, lint_command, lint_mcp, lint_memory, lint_settings,
    lint_skill, load_json, merge_layers, rule_matches,
)

LAB = Path(__file__).resolve().parents[1]
PROJECT = LAB / "example-project"
DATA = LAB / "data"
OUT = LAB / "labs_output"


# ---------------------------------------------------------------------------
# Offline responder: canned CLAUDE.md draft (live mode calls the real model)
# ---------------------------------------------------------------------------
CANNED_CLAUDE_MD = """# orders-platform

Order management monorepo: Python API in `api/`, TypeScript web client in `web/`.

## Commands
- Setup: `make setup`
- Lint: `make lint` (run before every commit)
- Tests: `make test`; web unit tests: `npm run test:unit` inside `web/`
- Migrations: `make migrate` (ask a human first); seed data: `make seed-db`

## Architecture
- `api/` service code, `web/` client, `migrations/` SQL, `tests/` test suites
- Module boundaries: @docs/architecture.md

## Conventions
- Conventional Commits for every commit message.
- Money is stored as integer cents.
- Never edit applied migrations; add a new migration instead.
- Follow best practices for security.

## Guard rails
- Never read `.env`, `secrets/`, or `deploy/certs/`.
"""


def responder(req: MockRequest) -> str:
    if "CLAUDE.md" in req.system_text:
        return CANNED_CLAUDE_MD
    return "ok"


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("  (no findings)")
        return
    for f in sorted(findings, key=lambda x: (SEVERITY_ORDER[x.severity], x.code)):
        print(f"  [{f.severity:<5}] {f.code}  {f.target}: {f.message}")


def counts(findings: list[Finding]) -> str:
    c = {s: sum(1 for f in findings if f.severity == s) for s in ("ERROR", "WARN", "INFO")}
    return f"ERROR={c['ERROR']} WARN={c['WARN']} INFO={c['INFO']}"


def compact(value: Any) -> str:
    """JSON without spaces - identical to JavaScript's JSON.stringify output."""
    return json.dumps(value, separators=(",", ":"))


def short(value: Any, width: int) -> str:
    text = value if isinstance(value, str) else compact(value)
    return text if len(text) <= width else text[: width - 3] + "..."


# ---------------------------------------------------------------------------
# Hook runner: emulates how Claude Code invokes a command hook
# ---------------------------------------------------------------------------
HOOK_SCRIPT_RE = re.compile(r"\$CLAUDE_PROJECT_DIR/([^\"'\s]+)")


def run_hook_script(script: Path, payload: str, project_dir: Path) -> tuple[int, str, str]:
    """stdin = JSON payload; exit 0 = continue, 2 = block (stderr goes back to Claude), other = non-blocking error.

    We run the script with this interpreter instead of shell-expanding the settings command,
    so the demo behaves the same on Windows, macOS and Linux.
    """
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(project_dir))
    proc = subprocess.run([sys.executable, str(script)], input=payload, capture_output=True,
                          text=True, env=env, timeout=30)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def matcher_matches(matcher: str, tool: str) -> bool:
    return matcher in ("", "*") or re.fullmatch(matcher, tool) is not None


def run_pre_tool_hooks(hooks: dict[str, list[dict[str, Any]]], tool: str, tool_input: dict[str, Any]) -> tuple[str, str]:
    for entry in hooks.get("PreToolUse", []):
        if not matcher_matches(str(entry.get("matcher", "")), tool):
            continue
        for hook in entry.get("hooks", []):
            m = HOOK_SCRIPT_RE.search(str(hook.get("command", "")))
            if not m:
                continue
            payload = json.dumps({
                "session_id": "lab-session", "cwd": str(PROJECT), "hook_event_name": "PreToolUse",
                "tool_name": tool, "tool_input": tool_input,
            })
            code, _, err = run_hook_script(PROJECT / m.group(1), payload, PROJECT)
            if code == 2:
                return "block", err
    return "pass", ""


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_inventory() -> None:
    section("1. The shared team setup (example-project/)")
    items = [
        (".claude/settings.json", "project (shared)", "permissions allow/ask/deny, env, hooks, MCP opt-in"),
        (".claude/hooks/protect_paths.py", "project (shared)", "PreToolUse guard: exit 2 blocks protected paths"),
        (".claude/hooks/format_changed_file.py", "project (shared)", "PostToolUse formatter for edited files"),
        ("CLAUDE.md", "project memory", "team commands, architecture, conventions"),
        ("web/CLAUDE.md", "nested memory", "loaded when Claude works inside web/"),
        ("docs/architecture.md", "memory import", "pulled in via @docs/architecture.md"),
        (".mcp.json", "project MCP", "servers with ${ENV_VAR} secrets (approved per user)"),
        (".claude/agents/code-reviewer.md", "subagent", "read-only reviewer (tools: Read, Grep, Glob)"),
        (".claude/skills/api-endpoint-checklist/SKILL.md", "skill", "loaded on demand by description"),
        (".claude/commands/pr-ready.md", "slash command", "/pr-ready [base-branch]"),
        (".gitignore", "repo hygiene", "keeps settings.local.json and .env out of git"),
    ]
    for rel, scope, purpose in items:
        status = "present" if (PROJECT / rel).exists() else "MISSING"
        print(f"  {status:<8} {rel:<48} {scope:<17} {purpose}")


def lint_project_dir(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    findings += lint_settings(load_json(root / ".claude" / "settings.json"), ".claude/settings.json", root)
    findings += lint_mcp(load_json(root / ".mcp.json"), ".mcp.json")
    for agent in sorted((root / ".claude" / "agents").glob("*.md")):
        findings += lint_agent(agent.read_text(encoding="utf-8"), f".claude/agents/{agent.name}")
    for skill in sorted((root / ".claude" / "skills").glob("*/SKILL.md")):
        findings += lint_skill(skill.read_text(encoding="utf-8"), f".claude/skills/{skill.parent.name}/SKILL.md", skill.parent.name)
    for cmd in sorted((root / ".claude" / "commands").glob("*.md")):
        findings += lint_command(cmd.read_text(encoding="utf-8"), f".claude/commands/{cmd.name}")
    findings += lint_memory((root / "CLAUDE.md").read_text(encoding="utf-8"), "CLAUDE.md", root)
    findings += lint_memory((root / "web" / "CLAUDE.md").read_text(encoding="utf-8"), "web/CLAUDE.md", root)
    return findings


def step_lint() -> tuple[list[Finding], list[Finding]]:
    section("2. Lint the team config (expected: clean)")
    good = lint_project_dir(PROJECT)
    print_findings(good)
    print(f"  summary: {counts(good)}")

    section("3. Lint a deliberately risky config (data/risky/)")
    risky_dir = DATA / "risky"
    bad: list[Finding] = []
    bad += lint_settings(load_json(risky_dir / "settings.risky.json"), "settings.risky.json")
    bad += lint_mcp(load_json(risky_dir / "mcp.risky.json"), "mcp.risky.json")
    bad += lint_agent((risky_dir / "agent-reviewer.risky.md").read_text(encoding="utf-8"), "agent-reviewer.risky.md")
    bad += lint_skill((risky_dir / "skill.risky.md").read_text(encoding="utf-8"), "skill.risky.md")
    bad += lint_command((risky_dir / "command.risky.md").read_text(encoding="utf-8"), "command.risky.md")
    bad += lint_memory((risky_dir / "memory.risky.md").read_text(encoding="utf-8"), "memory.risky.md", PROJECT)
    print_findings(bad)
    print(f"  summary: {counts(bad)}")
    print("  -> Run this linter in CI on every PR that touches .claude/, .mcp.json or CLAUDE.md.")
    return good, bad


def load_layers() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("user", load_json(DATA / "layers" / "user.settings.json")),
        ("project", load_json(PROJECT / ".claude" / "settings.json")),
        ("local", load_json(DATA / "layers" / "local.settings.json")),
        ("cli", load_json(DATA / "layers" / "cli-args.json")["settings"]),
        ("managed", load_json(DATA / "layers" / "managed.settings.json")),
    ]


def step_precedence():
    section("4. Settings precedence: merge five layers with provenance")
    print("  precedence (highest first): managed policy > command-line args > local project")
    print("  (.claude/settings.local.json) > shared project (.claude/settings.json) > user (~/.claude/settings.json)")
    print(f"  CLI layer simulates: {' '.join(load_json(DATA / 'layers' / 'cli-args.json')['argv'])}")
    layers = load_layers()
    eff = merge_layers(layers)
    print("\n  effective scalar settings (winner <- overridden values):")
    for key in sorted(eff.values):
        hist = eff.history[key]
        lost = [f"{layer}={short(v, 18)}" for layer, v in hist[:-1]]
        lost_txt = f"   overrides: {', '.join(lost)}" if lost else ""
        print(f"    {key:<42} = {short(eff.values[key], 20):<20} [{eff.source[key]}]{lost_txt}")
    print("\n  permission rules are COMBINED across layers (deny anywhere wins at evaluation):")
    for kind in ("allow", "ask", "deny"):
        by_src: dict[str, int] = {}
        for r in eff.rules:
            if r.kind == kind:
                by_src[r.source] = by_src.get(r.source, 0) + 1
        detail = ", ".join(f"{src}={n}" for src, n in by_src.items())
        print(f"    {kind:<5} {sum(by_src.values()):>2} rules  ({detail})")
    print(f"    hooks: {', '.join(f'{ev} x{len(v)}' for ev, v in eff.hooks.items())}")

    # What-if: a developer tries to switch on bypass mode locally.
    what_if = [(n, s) if n != "local" else (n, {**s, "permissions": {**s["permissions"], "defaultMode": "bypassPermissions"}})
               for n, s in layers]
    eff_bypass = merge_layers(what_if)
    print("\n  what-if: local settings request defaultMode=bypassPermissions")
    for note in eff_bypass.notes:
        print(f"    -> {note}")

    OUT.mkdir(parents=True, exist_ok=True)
    dump = {"values": eff.values, "source": eff.source,
            "rules": [{"kind": r.kind, "rule": r.rule, "source": r.source} for r in eff.rules]}
    (OUT / "effective-settings.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"\n  wrote labs_output/effective-settings.json")
    return eff


def describe_input(tool: str, tool_input: dict[str, Any]) -> str:
    for key in ("command", "file_path", "url"):
        if key in tool_input:
            return str(tool_input[key])
    return compact(tool_input)


def step_simulate(eff) -> tuple[int, int]:
    section("5. Permission simulator: PreToolUse hook -> deny > ask > allow -> mode")
    mode = eff.values.get("permissions.defaultMode", "default")
    print(f"  effective defaultMode = {mode}  (hooks run first; a hook exit 2 blocks even an allowed call)")
    requests = load_json(DATA / "tool_requests.json")
    ok = 0
    for i, req in enumerate(requests, 1):
        tool, tool_input, expect = req["tool"], req["input"], req["expect"]
        print(f"\n  #{i:02d} {tool:<34} {short(describe_input(tool, tool_input), 56)}")
        hook_state, hook_msg = run_pre_tool_hooks(eff.hooks, tool, tool_input)
        if hook_state == "block":
            outcome = "blocked-by-hook"
            print(f"      hook  : BLOCK (exit 2) -> {short(hook_msg, 90)}")
        else:
            d = evaluate(eff.rules, tool, tool_input, mode)
            outcome = d.decision
            via = f"{d.rule} [{d.source}]" if d.rule else d.reason
            sub = f" on sub-command '{d.sub_command}'" if d.sub_command and d.sub_command != tool_input.get("command") else ""
            print(f"      result: {d.decision.upper():<5} via {via}{sub}")
            losers = [r for r in eff.rules if r.kind != d.decision and d.rule
                      and rule_matches(r.rule, tool, tool_input, d.sub_command)]
            if losers:
                print("      note  : overrides " + ", ".join(f"{r.kind} {r.rule} [{r.source}]" for r in losers))
        match = outcome == expect
        ok += match
        print(f"      expect: {expect:<15} {'OK' if match else 'MISMATCH'}")
    print(f"\n  simulator agreed with expectations on {ok}/{len(requests)} requests")
    return ok, len(requests)


def step_hooks() -> None:
    section("6. Hooks up close: the stdin / exit-code contract")
    guard = PROJECT / ".claude" / "hooks" / "protect_paths.py"
    cases = [
        ("Edit", {"file_path": "api/routes/orders.py"}),
        ("Write", {"file_path": "secrets/stripe.key"}),
        ("Edit", {"file_path": "../../outside-the-repo.txt"}),
        ("Edit", {"file_path": "web/package-lock.json"}),
    ]
    meaning = {0: "continue", 2: "BLOCK (stderr -> Claude)"}
    for tool, tool_input in cases:
        payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input, "cwd": str(PROJECT)})
        code, _, err = run_hook_script(guard, payload, PROJECT)
        print(f"  PreToolUse {tool:<5} {tool_input['file_path']:<28} exit={code} {meaning.get(code, 'non-blocking error')}")
        if err:
            print(f"      stderr: {short(err, 96)}")
    code, _, err = run_hook_script(guard, "this is not json", PROJECT)
    print(f"  PreToolUse malformed payload                 exit={code} {meaning.get(code, 'non-blocking error')} ({err})")

    # PostToolUse formatter, run against a scratch file in labs_output (never the example project).
    sandbox = OUT / "hook-sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    target = sandbox / "notes.md"
    target.write_bytes(b"# Release notes   \n\n- fixed refund rounding  \n- added audit log\t\n\n\n")
    before = target.read_text(encoding="utf-8")
    payload = json.dumps({"hook_event_name": "PostToolUse", "tool_name": "Write", "tool_input": {"file_path": "notes.md"}, "cwd": str(sandbox)})
    code, out, _ = run_hook_script(PROJECT / ".claude" / "hooks" / "format_changed_file.py", payload, sandbox)
    after = target.read_text(encoding="utf-8")

    def trailing(t: str) -> int:
        return sum(1 for line in t.split("\n") if line != line.rstrip())

    print(f"\n  PostToolUse Write notes.md -> exit={code}, hook said: '{out}'")
    print(f"      before: {len(before)} chars, {trailing(before)} lines with trailing whitespace")
    print(f"      after : {len(after)} chars, {trailing(after)} lines with trailing whitespace")


def step_generate_claude_md() -> list[Finding]:
    section("7. Bonus: Claude drafts a CLAUDE.md from a repo summary")
    client = get_client(responder)
    summary = (DATA / "repo_summary.json").read_text(encoding="utf-8")
    system = (
        "You write CLAUDE.md memory files for Claude Code. Rules: under 60 lines; sections Commands, "
        "Architecture, Conventions, Guard rails; only facts present in the repo summary; never include "
        "secrets; specific, checkable instructions in imperative voice."
    )
    msg = client.messages.create(
        model=MODEL_OPUS,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=system,
        messages=[{"role": "user", "content": f"<repo_summary>\n{summary}\n</repo_summary>\nDraft the CLAUDE.md."}],
    )
    draft = text_of(msg)
    print(f"  model={msg.model} stop_reason={msg.stop_reason} {usage_line(msg)}")
    if msg.stop_reason == "max_tokens":
        print("  WARNING: draft truncated - raise max_tokens before using it")
    for line in draft.splitlines()[:8]:
        print(f"    | {line}")
    print(f"    | ... ({len(draft.splitlines())} lines total)")
    findings = lint_memory(draft, "CLAUDE.generated.md", PROJECT)
    print("  linter on the generated draft (human review still required):")
    print_findings(findings)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "CLAUDE.generated.md").write_text(draft, encoding="utf-8")
    print("  wrote labs_output/CLAUDE.generated.md")
    return findings


def main() -> None:
    banner("D7.1 - Configuring Claude Code for Teams")
    step_inventory()
    good, bad = step_lint()
    eff = step_precedence()
    ok, total = step_simulate(eff)
    step_hooks()
    draft_findings = step_generate_claude_md()

    section("8. Takeaways")
    print(f"  team config findings: {counts(good)} | risky config: {counts(bad)} | draft: {counts(draft_findings)}")
    print(f"  permission simulator: {ok}/{total} decisions matched expectations")
    print("  - Commit .claude/settings.json, CLAUDE.md, .mcp.json; keep settings.local.json personal.")
    print("  - Deny wins over ask and allow, in every layer; managed policy cannot be overridden.")
    print("  - Hooks are deterministic guard rails (exit 2 blocks); prompts and memory are guidance.")
    print("  - Secrets: deny reads of .env/secrets, and use ${ENV_VAR} expansion in .mcp.json.")


if __name__ == "__main__":
    main()
