"""
Run every lab (or a filtered subset) and report pass/fail for Python and TypeScript.

Usage (from the repo root, with the venv active or via its python):
    python run_all.py                 # all labs, both languages, offline
    python run_all.py D3              # only Domain 3
    python run_all.py D2.5 --py       # one lab, Python only
    python run_all.py D1 --ts -v      # Domain 1, TypeScript only, show lab output
    python run_all.py --live          # use the real API (needs ANTHROPIC_API_KEY)

Labs live in folders named "D<domain>.<n> - <title>" containing
python/main.py and typescript/main.ts.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAB_DIR_RE = re.compile(r"^D(\d+)\.(\d+) - ")


def discover(prefix: str | None) -> list[Path]:
    labs = [p for p in ROOT.iterdir() if p.is_dir() and LAB_DIR_RE.match(p.name)]
    labs.sort(key=lambda p: tuple(int(x) for x in LAB_DIR_RE.match(p.name).groups()))  # type: ignore[union-attr]
    if prefix:
        want = prefix.rstrip(".")
        labs = [p for p in labs if p.name.split(" - ")[0] == want or p.name.startswith(want + ".")]
    return labs


def run(cmd: list[str], env: dict[str, str], verbose: bool) -> tuple[bool, float, str]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0, "TIMEOUT after 180s"
    secs = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    if verbose:
        print(out)
    return proc.returncode == 0, secs, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filter", nargs="?", help="e.g. D3 or D3.5")
    ap.add_argument("--py", action="store_true", help="Python only")
    ap.add_argument("--ts", action="store_true", help="TypeScript only")
    ap.add_argument("--live", action="store_true", help="call the real API instead of the offline mock")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each lab's output")
    args = ap.parse_args()

    do_py = args.py or not args.ts
    do_ts = args.ts or not args.py
    env = dict(os.environ, PYTHONIOENCODING="utf-8", LABS_MODE="live" if args.live else "offline")
    node = shutil.which("node")
    if do_ts and not node:
        print("WARNING: `node` not found on PATH - TypeScript labs will be skipped (install Node.js 24+).")

    labs = discover(args.filter)
    if not labs:
        print("No labs matched.")
        return 1

    results: list[tuple[str, str, bool, float, str]] = []
    for lab in labs:
        if do_py and (lab / "python" / "main.py").exists():
            ok, secs, out = run([sys.executable, str(lab / "python" / "main.py")], env, args.verbose)
            results.append((lab.name, "python", ok, secs, out))
            print(f"[{'PASS' if ok else 'FAIL'}] {lab.name}  (python, {secs:.1f}s)")
        if do_ts and node and (lab / "typescript" / "main.ts").exists():
            ok, secs, out = run([node, str(lab / "typescript" / "main.ts")], env, args.verbose)
            results.append((lab.name, "typescript", ok, secs, out))
            print(f"[{'PASS' if ok else 'FAIL'}] {lab.name}  (typescript, {secs:.1f}s)")

    failed = [r for r in results if not r[2]]
    print(f"\n{len(results) - len(failed)}/{len(results)} runs passed.")
    for name, lang, _, _, out in failed:
        tail = "\n".join(out.strip().splitlines()[-15:])
        print(f"\n--- FAILED: {name} ({lang}) ---\n{tail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
