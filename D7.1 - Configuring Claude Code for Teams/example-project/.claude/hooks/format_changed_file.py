#!/usr/bin/env python3
"""PostToolUse formatting hook: tidy the file Claude just edited.

Runs after every successful Edit/Write/MultiEdit. It never blocks (always exits 0)
because formatting is a convenience, not a guard rail. It only touches the single
file named in the payload, only inside the project, and never uses the network:
  * *.py  -> `ruff format` if ruff is already installed locally
  * other -> stdlib normalisation: strip trailing whitespace, ensure final newline

Why a hook? Deterministic formatting in a hook is cheaper and more reliable than
asking the model to "remember to format" - the harness guarantees it runs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

TEXT_SUFFIXES = {".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ts", ".tsx", ".css", ".sql"}


def normalise(path: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        original = fh.read()
    lines = [line.rstrip() for line in original.splitlines()]
    tidy = "\n".join(lines).rstrip("\n") + "\n"
    if tidy != original:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(tidy)
        return True
    return False


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read().lstrip("\ufeff"))  # tolerate a BOM from some shells
    except json.JSONDecodeError:
        return 0
    target = (payload.get("tool_input") or {}).get("file_path") or ""
    project = os.path.abspath(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())
    path = os.path.abspath(os.path.join(project, target))
    if not target or not path.startswith(project + os.sep) or not os.path.isfile(path):
        return 0

    rel = os.path.relpath(path, project).replace(os.sep, "/")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".py" and shutil.which("ruff"):
        subprocess.run(["ruff", "format", "--quiet", path], check=False, timeout=25)
        print(f"format hook: ruff format {rel}")
    elif suffix in TEXT_SUFFIXES or suffix == ".py":
        changed = normalise(path)
        print(f"format hook: {'normalised' if changed else 'already tidy'} {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
