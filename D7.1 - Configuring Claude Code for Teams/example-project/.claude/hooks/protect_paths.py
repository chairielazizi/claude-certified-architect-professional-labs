#!/usr/bin/env python3
"""PreToolUse guard hook: block edits to protected paths.

Claude Code runs this before every Edit/Write/MultiEdit (see the "matcher" in
.claude/settings.json). Contract:
  * stdin  : JSON payload with tool_name, tool_input, cwd, hook_event_name, ...
  * exit 0 : allow the tool call to continue to the normal permission checks
  * exit 2 : BLOCK the tool call; stderr is fed back to Claude so it can adapt
  * other  : non-blocking error (shown to the user, tool call proceeds)

Why a hook and not only a deny rule? Hooks are deterministic code: they can
apply logic (path normalisation, escaping the project root, generated-file
markers) that a static permission pattern cannot express. Keep hooks fast,
dependency-free, and side-effect free - this one only reads stdin.
"""

from __future__ import annotations

import json
import os
import re
import sys

# Team policy: files that must only change through a reviewed, human-driven process.
PROTECTED_PATTERNS = [
    ".env",
    ".env.*",
    "secrets/**",
    "**/*.pem",
    ".git/**",
    ".claude/settings.json",      # changing the guard rails needs a human PR
    ".claude/hooks/**",
    "migrations/applied/**",      # applied migrations are immutable history
    "web/package-lock.json",      # lockfiles change only via the package manager
]


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Minimal gitignore-style glob: ** spans directories, * stays within one segment."""
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


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read().lstrip("\ufeff"))  # tolerate a BOM from some shells
    except json.JSONDecodeError:
        print("protect_paths: could not parse hook payload", file=sys.stderr)
        return 1  # non-blocking error: never wedge the session on a malformed payload

    tool_input = payload.get("tool_input") or {}
    target = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not target:
        return 0

    project = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    absolute = os.path.abspath(os.path.join(project, target))
    rel = os.path.relpath(absolute, os.path.abspath(project)).replace(os.sep, "/")

    if rel == ".." or rel.startswith("../"):
        print(f"Blocked by team policy: '{target}' is outside the project root.", file=sys.stderr)
        return 2

    for pattern in PROTECTED_PATTERNS:
        if glob_to_regex(pattern).match(rel):
            print(
                f"Blocked by team policy: '{rel}' is protected (rule '{pattern}'). "
                "Propose the change in your reply instead, and a maintainer will apply it.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
