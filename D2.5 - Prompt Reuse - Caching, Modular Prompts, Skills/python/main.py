"""
D2.5 - Prompt Reuse: Caching, Modular Prompts, Skills
=====================================================

Three reuse strategies for an insurance-claims assistant:

  (b) Modular prompts - a versioned prompt library (persona / safety / policy /
      reference / format components, semver-pinned per use case), composed in a
      deterministic order with dependency checks and a fingerprint.
  (a) Prompt caching - exam sample question 2: the same ~7.5K-token system prompt
      (handbook + coverage schedule) on every request -> static content first,
      cache_control at the end of it. Measured cache writes vs reads and cost,
      TTL break-even, and two silent anti-patterns (timestamp in the system
      prompt; automatic caching on single-shot requests with a unique tail).
      Plus cross-use-case reuse with a breakpoint after the shared prefix.
  (c) Agent Skills - SKILL.md folders with YAML frontmatter, a local loader with
      validation and progressive disclosure (name+description -> SKILL.md ->
      referenced file) driven by real tool use, and the Claude API Skills
      request code (container.skills + code execution), live-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_OPUS,
    PRICING,
    MockRequest,
    banner,
    cost_usd,
    get_client,
    is_live,
    section,
    text_block,
    text_of,
    tool_use_block,
)

DATA = Path(__file__).resolve().parents[1] / "data"
LIBRARY_DIR = DATA / "prompt_library"
SKILLS_DIR = DATA / "skills"
USE_CASES: dict[str, list[str]] = json.loads((DATA / "use_cases.json").read_text(encoding="utf-8"))["use_cases"]
CLAIMS: dict[str, Any] = json.loads((DATA / "claims.json").read_text(encoding="utf-8"))
SCHEDULE: dict[str, Any] = json.loads((DATA / "coverage_schedule.json").read_text(encoding="utf-8"))
MODEL = MODEL_OPUS
ONE_HOUR_WRITE_MULT = 2.0  # 1h TTL writes cost 2x base input (5m writes: 1.25x, in PRICING)


# ---------------------------------------------------------------------------
# Frontmatter (shared by prompt components and SKILL.md files)
# ---------------------------------------------------------------------------
def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML-frontmatter parser for flat `key: value` headers (no external deps)."""
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter")
    end = text.index("\n---\n", 4)
    meta: dict[str, str] = {}
    for line in text[4:end].split("\n"):
        if line.strip():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[end + 5:].strip()


# ---------------------------------------------------------------------------
# (b) Modular prompt library
# ---------------------------------------------------------------------------
LAYER_ORDER = ["safety", "policy", "reference", "persona", "format"]  # most shared/stable first


class CompositionError(ValueError):
    pass


@dataclass(frozen=True)
class Component:
    id: str
    version: str
    layer: str
    requires: tuple[str, ...]
    summary: str
    body: str
    source: str

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    def render(self) -> str:
        return f'<component id="{self.id}" version="{self.version}">\n{self.body}\n</component>'


def vtuple(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def matches(version: str, constraint: str) -> bool:
    if constraint == "*":
        return True
    if constraint.startswith("^"):
        return version.split(".")[0] == constraint[1:]
    return version == constraint


def round5(x: int) -> int:
    return (x + 2) // 5 * 5


def render_schedule(data: dict[str, Any]) -> str:
    """Reference tables rendered from structured data must be deterministic: same data -> same bytes."""
    lines = [
        f"# Coverage schedule {data['version']} (per-claim deductible and limit by rating region and product)",
        "| region | product | deductible | limit | CAT surge | flood |",
        "|---|---|---|---|---|---|",
    ]
    for r in data["regions"]:
        for p in data["products"]:
            ded = round5(p["base_deductible"] * r["multiplier_pct"] // 100)
            lim = p["base_limit"] * r["multiplier_pct"] // 100 // 1000 * 1000
            surge = r["cat_surge_pct"]
            flood = "n/a" if not p["property"] else ("excluded" if r["flood_excluded"] else "covered")
            lines.append(f"| {r['code']} {r['name']} | {p['product']} | ${ded:,} | ${lim:,} | "
                         f"x{surge // 100}.{surge % 100:02d} | {flood} |")
    return "\n".join(lines)


class PromptLibrary:
    def __init__(self, components: list[Component]) -> None:
        self.components = components

    @classmethod
    def load(cls, folder: Path, generated: list[Component]) -> "PromptLibrary":
        comps: list[Component] = []
        for path in sorted(folder.glob("*.md")):  # sorted: directory order is not guaranteed
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
            requires = tuple(r.strip() for r in meta.get("requires", "").split(",") if r.strip())
            comps.append(Component(meta["id"], meta["version"], meta["layer"], requires, meta["summary"], body, path.name))
        return cls(comps + generated)

    def resolve(self, ref: str) -> Component:
        cid, _, constraint = ref.partition("@")
        candidates = [c for c in self.components if c.id == cid and matches(c.version, constraint or "*")]
        if not candidates:
            raise CompositionError(f"no version of {cid} matches '{constraint}'")
        return max(candidates, key=lambda c: vtuple(c.version))


@dataclass
class Composition:
    components: list[Component]

    @property
    def text(self) -> str:
        return "\n\n".join(c.render() for c in self.components)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    def blocks(self, split: int | None = None, breakpoint: str = "end") -> list[dict[str, Any]]:
        """System blocks. split = number of leading (shared) components in block 1.
        breakpoint: 'end' (cache everything), 'split' (cache only the shared prefix), 'none'."""
        n = len(self.components) if split is None else split
        parts = [self.components[:n], self.components[n:]]
        blocks: list[dict[str, Any]] = []
        for i, group in enumerate(parts):
            if not group:
                continue
            block: dict[str, Any] = {"type": "text", "text": "\n\n".join(c.render() for c in group)}
            if (breakpoint == "split" and i == 0) or (breakpoint == "end" and i == len(parts) - 1) or \
               (breakpoint == "end" and i == 0 and not parts[1]):
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        return blocks


def compose(library: PromptLibrary, refs: list[str], canonical: bool = True) -> Composition:
    comps = [library.resolve(r) for r in refs]
    ids = [c.id for c in comps]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise CompositionError(f"duplicate components: {', '.join(dupes)}")
    for c in comps:
        for req in c.requires:
            if req not in ids:
                raise CompositionError(f"{c.ref} requires {req}, which the manifest does not include")
    if canonical:
        # Deterministic order = stable cache key, whatever order the manifest lists things in.
        comps = sorted(comps, key=lambda c: (LAYER_ORDER.index(c.layer), c.id))
    return Composition(comps)


def common_prefix(a: Composition, b: Composition) -> int:
    n = 0
    for x, y in zip(a.components, b.components):
        if x.ref != y.ref:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# (c) Agent Skills: loader, validation, progressive disclosure
# ---------------------------------------------------------------------------
class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    folder: Path
    body: str

    @property
    def skill_md(self) -> str:
        return (self.folder / "SKILL.md").read_text(encoding="utf-8")


NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")


def validate_skill_meta(meta: dict[str, str], folder_name: str) -> list[str]:
    errors = []
    name = meta.get("name", "")
    desc = meta.get("description", "")
    if not NAME_RE.match(name):
        errors.append("name must be 1-64 chars of lowercase letters, digits, hyphens")
    if "anthropic" in name or "claude" in name:
        errors.append("name must not contain reserved words 'anthropic' or 'claude'")
    if name and name != folder_name:
        errors.append(f"name '{name}' must match folder '{folder_name}'")
    if not desc:
        errors.append("description is required (it is how the model decides to load the skill)")
    if len(desc) > 1024:
        errors.append("description must be <= 1024 chars")
    if re.search(r"<[^>]+>", desc):
        errors.append("description must not contain XML tags")
    return errors


def load_skills(root: Path) -> list[Skill]:
    skills = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        meta, body = parse_frontmatter((folder / "SKILL.md").read_text(encoding="utf-8"))
        errors = validate_skill_meta(meta, folder.name)
        if errors:
            raise SkillError(f"{folder.name}: " + "; ".join(errors))
        skills.append(Skill(meta["name"], meta["description"], folder, body))
    return skills


def skills_index(skills: list[Skill]) -> str:
    """Level 1: only name + description sit in context (~100 tokens per skill)."""
    lines = ["<available_skills>"]
    lines += [f'<skill name="{s.name}">{s.description}</skill>' for s in skills]
    lines.append("</available_skills>")
    lines.append("When a request matches a skill, call load_skill with its name and follow the loaded instructions. "
                 "Load referenced files with read_skill_file only when a step needs them. Otherwise answer directly.")
    return "\n".join(lines)


def read_skill_file(skills: dict[str, Skill], skill: str, rel_path: str) -> str:
    """Level 3 loader with a path-traversal guard: model-supplied paths are untrusted."""
    if skill not in skills:
        raise SkillError(f"unknown skill '{skill}'")
    base = skills[skill].folder.resolve()
    target = (base / rel_path).resolve()
    if base != target and base not in target.parents:
        raise SkillError(f"path '{rel_path}' escapes the skill directory")
    if not target.is_file():
        raise SkillError(f"no such file '{rel_path}' in skill '{skill}'")
    return target.read_text(encoding="utf-8")


def skill_tools(names: list[str]) -> list[dict[str, Any]]:
    # Tools render first in the prompt: keep them deterministic (sorted) so they stay cacheable.
    enum = sorted(names)
    return sorted([
        {"name": "load_skill", "strict": True,
         "description": "Load the full SKILL.md instructions of one available skill before handling a matching request.",
         "input_schema": {"type": "object", "properties": {"name": {"type": "string", "enum": enum}},
                          "required": ["name"], "additionalProperties": False}},
        {"name": "read_skill_file", "strict": True,
         "description": "Read a supporting file that a loaded skill references, e.g. references/red_flags.md.",
         "input_schema": {"type": "object", "properties": {"skill": {"type": "string", "enum": enum},
                                                           "path": {"type": "string"}},
                          "required": ["skill", "path"], "additionalProperties": False}},
    ], key=lambda t: t["name"])


def run_skill_agent(client: Any, system: list[dict[str, Any]], tools: list[dict[str, Any]],
                    skills: dict[str, Skill], user_text: str, max_turns: int = 6) -> tuple[str, list[str]]:
    messages: list[Any] = [{"role": "user", "content": user_text}]
    trace: list[str] = []
    for turn in range(1, max_turns + 1):
        msg = client.messages.create(model=MODEL, max_tokens=2048, system=system, tools=tools, messages=messages,
                                     output_config={"effort": "medium"})
        in_tok = msg.usage.input_tokens + (msg.usage.cache_read_input_tokens or 0) + (msg.usage.cache_creation_input_tokens or 0)
        if msg.stop_reason != "tool_use":
            trace.append(f"turn {turn}: {msg.stop_reason:<9} final answer                                   in={in_tok} tok")
            return text_of(msg), trace
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for block in msg.content:
            if block.type != "tool_use":
                continue
            args = ", ".join(f"{k}={v}" for k, v in block.input.items())
            try:
                if block.name == "load_skill":
                    content = skills[block.input["name"]].skill_md       # level 2
                else:
                    content = read_skill_file(skills, block.input["skill"], block.input["path"])  # level 3
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            except (SkillError, KeyError) as exc:
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True})
            trace.append(f"turn {turn}: tool_use  {block.name}({args})".ljust(66) + f" in={in_tok} tok")
        messages.append({"role": "user", "content": results})  # all results in ONE user message
    return "(stopped: max turns)", trace


# ---------------------------------------------------------------------------
# Offline responder
# ---------------------------------------------------------------------------
FRAUD_REPORT = (
    "Score: 55/100 -> refer to SIU\n"
    "Flags: RF-01 (20), RF-02 (20), RF-04 (15)\n"
    "Information needed: camera purchase date (RF-03), police report reference and filing time (RF-08), "
    "prior claims history (RF-07)\n"
    "Handling note: refer under H9.2 and H3.1 (loss in the first 14 days); request proof of ownership "
    "neutrally and do not mention fraud to the claimant."
)


def responder(req: MockRequest) -> Any:
    user = req.last_user_text
    if "load_skill" in req.tool_names:
        results = req.tool_results
        if not results:
            if "suspicious" in user or "fraud" in user:
                return [text_block("This matches the fraud-screening skill; loading it."),
                        tool_use_block("load_skill", {"name": "claims-fraud-screening"})]
            return "Motor claims must be notified within 7 days of the loss (handbook H2.1)."
        last = str(results[-1].get("content", ""))
        if "## Procedure" in last:
            return [tool_use_block("read_skill_file", {"skill": "claims-fraud-screening", "path": "references/red_flags.md"})]
        if "RED FLAG CATALOGUE" in last:
            return FRAUD_REPORT
        return "I could not load the skill; proceeding with the standard handbook checks."
    if "persona.underwriting_analyst" in req.system_text:
        return CLAIMS["memo_offline"]
    for c in CLAIMS["triage_requests"]:
        if c["claim_id"] in user:
            return json.dumps(c["offline_decision"])
    return "{}"


# ---------------------------------------------------------------------------
# Live-only: Agent Skills on the Claude API
# ---------------------------------------------------------------------------
SKILLS_API_SNIPPET = '''\
skill_dir = Path("data/skills/claims-fraud-screening")
files = [(f"{skill_dir.name}/{p.relative_to(skill_dir).as_posix()}", p.read_bytes(), "text/markdown")
         for p in sorted(skill_dir.rglob("*")) if p.is_file()]
skill = client.skills.create(files=files, display_name="Claims fraud screening")  # Skills API (no beta header)

resp = client.beta.messages.create(
    model="claude-opus-5-5", max_tokens=16000,
    betas=["code-execution-2025-08-25"],
    container={"skills": [
        {"type": "custom", "skill_id": skill.id, "version": "latest"},
        {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},   # Anthropic-managed skill
    ]},
    tools=[{"type": "code_execution_20260521", "name": "code_execution"}],  # skills run in the container
    messages=[{"role": "user", "content": "Screen claim CLM-7731 for fraud and export the flags to a spreadsheet."}],
)
# generated files come back as file ids -> client.beta.files.download(file_id)'''


def run_live_skills(client: Any, fraud_text: str) -> None:
    if os.environ.get("LABS_RUN_SKILLS") != "1":
        print("  Live: listing skills (read-only). Set LABS_RUN_SKILLS=1 to upload the skill and run a container")
        print("  (creates an org resource and bills code-execution time).")
        for i, s in enumerate(client.skills.list()):
            if i >= 5:
                break
            print(f"    {s.id}  {s.display_name}  source={s.source}")
        return
    skill_dir = SKILLS_DIR / "claims-fraud-screening"
    files = [(f"{skill_dir.name}/{p.relative_to(skill_dir).as_posix()}", p.read_bytes(), "text/markdown")
             for p in sorted(skill_dir.rglob("*")) if p.is_file()]
    skill = client.skills.create(files=files, display_name="Claims fraud screening")
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000, betas=["code-execution-2025-08-25"],
        container={"skills": [{"type": "custom", "skill_id": skill.id, "version": "latest"},
                              {"type": "anthropic", "skill_id": "xlsx", "version": "latest"}]},
        tools=[{"type": "code_execution_20260521", "name": "code_execution"}],
        messages=[{"role": "user", "content": fraud_text + " Export the flags to a spreadsheet."}],
    )
    print(f"  uploaded {skill.id}; container run stop_reason={resp.stop_reason}")
    print("  " + text_of(resp)[:400].replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


def fnol(claim: dict[str, Any]) -> str:
    return f'<fnol claim_id="{claim["claim_id"]}">\n{claim["text"]}\n</fnol>'


def usage_cells(m: Any) -> list[str]:
    u = m.usage
    return [str(u.input_tokens), str(u.cache_creation_input_tokens or 0), str(u.cache_read_input_tokens or 0),
            f"${cost_usd(MODEL, u):.5f}"]


def main() -> None:
    banner("D2.5 - Prompt Reuse - Caching, Modular Prompts, Skills")
    client = get_client(responder)
    schedule = Component("reference.coverage_schedule", SCHEDULE["version"], "reference", (),
                         "Generated from coverage_schedule.json", render_schedule(SCHEDULE), "generated")
    library = PromptLibrary.load(LIBRARY_DIR, [schedule])

    # ---------------------------------------------------------------- 1
    section("1. Prompt library: reusable, versioned components")
    table(["component", "layer", "requires", "~tokens", "source"],
          [[c.ref, c.layer, ",".join(c.requires) or "-", str(len(c.body) // 4), c.source] for c in library.components])

    # ---------------------------------------------------------------- 2
    section("2. Compose per use case: resolve semver, check deps, canonical order")
    triage = compose(library, USE_CASES["claims_triage"])
    memo = compose(library, USE_CASES["underwriting_memo"])
    for name, comp in (("claims_triage", triage), ("underwriting_memo", memo)):
        tok = client.messages.count_tokens(model=MODEL, system=comp.blocks(breakpoint="none"),
                                           messages=[{"role": "user", "content": "x"}]).input_tokens
        print(f"  {name:<18} fingerprint={comp.fingerprint}  ~{tok:,} tok")
        print("    order: " + " > ".join(c.ref for c in comp.components))
    shared_n = common_prefix(triage, memo)
    shared_text = "\n\n".join(c.render() for c in triage.components[:shared_n])
    print(f"  shared leading components: {shared_n} (~{len(shared_text) // 4:,} tok) -> one cache entry can serve both")
    naive = compose(library, USE_CASES["claims_triage"], canonical=False)
    shuffled = compose(library, list(reversed(USE_CASES["claims_triage"])))
    pinned = compose(library, [r.replace("safety.core@^2", "safety.core@2.2.0") for r in USE_CASES["claims_triage"]])
    print(f"  manifest order, no canonical sort : {naive.fingerprint}  {'(differs -> cache miss)' if naive.fingerprint != triage.fingerprint else ''}")
    print(f"  reversed manifest, canonical sort : {shuffled.fingerprint}  {'(identical -> cache hit)' if shuffled.fingerprint == triage.fingerprint else '(differs!)'}")
    print(f"  pin safety.core@2.2.0             : {pinned.fingerprint}  (version change = deliberate cache reset)")
    try:
        compose(library, ["safety.pii@^1", "persona.claims_assistant@^1"])
    except CompositionError as exc:
        print(f"  dependency check                  : CompositionError -> {exc}")

    # ---------------------------------------------------------------- 3
    section("3. Exam Q2: same large system prompt every request -> cache it")
    claims = CLAIMS["triage_requests"]
    cached_rows, uncached_cost, cached_cost = [], 0.0, 0.0
    for c in claims:
        m_un = client.messages.create(model=MODEL, max_tokens=512, system=triage.blocks(breakpoint="none"),
                                      messages=[{"role": "user", "content": fnol(c)}], output_config={"effort": "low"})
        m_ca = client.messages.create(model=MODEL, max_tokens=512, system=triage.blocks(breakpoint="end"),
                                      messages=[{"role": "user", "content": fnol(c)}], output_config={"effort": "low"})
        uncached_cost += cost_usd(MODEL, m_un.usage)
        cached_cost += cost_usd(MODEL, m_ca.usage)
        decision = json.loads(text_of(m_ca)).get("decision", "?")
        cached_rows.append([c["claim_id"], decision] + usage_cells(m_ca) + [str(m_un.usage.input_tokens), f"${cost_usd(MODEL, m_un.usage):.5f}"])
    table(["claim", "decision", "in", "cache_write", "cache_read", "cost (cached)", "in (no cache)", "cost (no cache)"], cached_rows)
    prefix_tokens = int(cached_rows[0][3])
    print(f"  5 requests: ${cached_cost:.4f} cached vs ${uncached_cost:.4f} uncached -> {1 - cached_cost / uncached_cost:.0%} saved "
          "(and cached prefixes cut time-to-first-token on real traffic).")
    print("  Rule: static content (tools -> system -> stable docs) first, breakpoint at its end, dynamic content after.")

    # ---------------------------------------------------------------- 4
    section("4. TTL economics on Opus 5.5 (cost of the cached prefix only)")
    p = PRICING[MODEL]
    rows = []
    even_5m = even_1h = None
    for n in (1, 2, 3, 5, 10, 100):
        unc = n * prefix_tokens * p["input"] / 1e6
        c5 = (prefix_tokens * p["cache_write"] + (n - 1) * prefix_tokens * p["cache_read"]) / 1e6
        c1 = (prefix_tokens * p["input"] * ONE_HOUR_WRITE_MULT + (n - 1) * prefix_tokens * p["cache_read"]) / 1e6
        even_5m = even_5m or (n if c5 < unc else None)
        even_1h = even_1h or (n if c1 < unc else None)
        rows.append([str(n), f"${unc:.4f}", f"${c5:.4f}", f"${c1:.4f}"])
    table(["requests in TTL", "no cache", "5-min TTL (1.25x write)", "1-hour TTL (2x write)"], rows)
    print(f"  Break-even: 5-min TTL from request {even_5m}, 1-hour TTL from request {even_1h}. Reads refresh the TTL;")
    print("  use 1h only when requests sharing the prefix arrive 5-60 minutes apart.")

    # ---------------------------------------------------------------- 5
    section("5. Silent invalidator: a timestamp at the top of the system prompt")
    t0 = 9 * 3600  # fake clock (seconds since midnight) - real code would call datetime.now()

    def stamp(i: int) -> str:
        s = t0 + 37 * i
        return f"2026-10-11T{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}Z"

    rows = []
    for i, c in enumerate(claims[:3]):
        bad_system = [{"type": "text", "text": f"Current UTC time: {stamp(i)}"}] + triage.blocks(breakpoint="end")
        m = client.messages.create(model=MODEL, max_tokens=512, system=bad_system,
                                   messages=[{"role": "user", "content": fnol(c)}], output_config={"effort": "low"})
        rows.append(["timestamp in system", c["claim_id"]] + usage_cells(m))
    for i, c in enumerate(claims[3:5], start=3):
        m = client.messages.create(model=MODEL, max_tokens=512, system=triage.blocks(breakpoint="end"),
                                   messages=[{"role": "user", "content": f"<request_time>{stamp(i)}</request_time>\n{fnol(c)}"}],
                                   output_config={"effort": "low"})
        rows.append(["timestamp in user turn", c["claim_id"]] + usage_cells(m))
    table(["variant", "claim", "in", "cache_write", "cache_read", "cost"], rows)
    print("  Every request rewrote the cache (1.25x price) and never read it - no error, just a bigger bill.")
    print("  Same family: UUIDs/request ids, unsorted JSON, per-user names, conditional sections, varying tools.")

    # ---------------------------------------------------------------- 6
    section("6. Wrong tool: top-level automatic caching on single-shot requests")
    rows = []
    for q in CLAIMS["memo_requests"]:
        m = client.messages.create(model=MODEL, max_tokens=512, system=memo.blocks(breakpoint="none"),
                                   cache_control={"type": "ephemeral"},  # breakpoint lands on the unique question
                                   messages=[{"role": "user", "content": q}], output_config={"effort": "low"})
        rows.append([q[:48] + "..."] + usage_cells(m))
    table(["memo request", "in", "cache_write", "cache_read", "cost"], rows)
    print("  Automatic caching places the breakpoint on the LAST block - here the unique question - so each")
    print("  request writes a new entry nobody reads. It shines for multi-turn chats (the breakpoint moves")
    print("  forward with the conversation); for shared-prefix + unique-suffix, mark the shared prefix explicitly.")

    # ---------------------------------------------------------------- 7
    section("7. Cross-use-case reuse: breakpoint after the shared components")
    m1 = client.messages.create(model=MODEL, max_tokens=512, system=triage.blocks(split=shared_n, breakpoint="split"),
                                messages=[{"role": "user", "content": fnol(claims[0])}], output_config={"effort": "low"})
    m2 = client.messages.create(model=MODEL, max_tokens=512, system=memo.blocks(split=shared_n, breakpoint="split"),
                                messages=[{"role": "user", "content": CLAIMS["memo_requests"][0]}], output_config={"effort": "low"})
    table(["use case", "in", "cache_write", "cache_read", "cost"],
          [["claims_triage"] + usage_cells(m1), ["underwriting_memo"] + usage_cells(m2)])
    print("  The memo assistant read the prefix the triage assistant wrote: canonical ordering + shared components")
    print("  turn a prompt library into a cache-sharing strategy (up to 4 breakpoints per request).")

    # ---------------------------------------------------------------- 8
    section("8. Agent Skills: validation + progressive disclosure")
    skills = load_skills(SKILLS_DIR)
    by_name = {s.name: s for s in skills}
    for s in skills:
        print(f"  {s.name:<24} OK  description {len(s.description)} chars, SKILL.md {len(s.skill_md):,} chars")
    bad = validate_skill_meta({"name": "Claude-Helper", "description": ""}, "claude-helper")
    print("  invalid example 'Claude-Helper': " + "; ".join(bad))
    try:
        read_skill_file(by_name, "claims-fraud-screening", "../../use_cases.json")
    except SkillError as exc:
        print(f"  path traversal attempt          : SkillError -> {exc}")
    base = compose(library, USE_CASES["skills_demo"]).text
    tools = skill_tools(list(by_name))
    everything = base + "\n\n" + "\n\n".join(
        s.skill_md + "".join("\n\n" + f.read_text(encoding="utf-8") for f in sorted(s.folder.rglob("*.md")) if f.name != "SKILL.md")
        for s in skills)
    index_system = [{"type": "text", "text": base},
                    {"type": "text", "text": skills_index(skills), "cache_control": {"type": "ephemeral"}}]
    fraud_text = CLAIMS["skill_requests"][0]["text"]
    tok_all = client.messages.count_tokens(model=MODEL, system=everything, messages=[{"role": "user", "content": fraud_text}]).input_tokens
    tok_idx = client.messages.count_tokens(model=MODEL, system=index_system, tools=tools,
                                           messages=[{"role": "user", "content": fraud_text}]).input_tokens
    tok_base = client.messages.count_tokens(model=MODEL, system=base, messages=[{"role": "user", "content": fraud_text}]).input_tokens
    tok_base_tools = client.messages.count_tokens(model=MODEL, system=base, tools=tools,
                                                  messages=[{"role": "user", "content": fraud_text}]).input_tokens
    print(f"  all skills inlined in system : {tok_all:,} tok on EVERY request")
    print(f"  index only (+2 loader tools) : {tok_idx:,} tok -> {1 - tok_idx / tok_all:.0%} smaller; full text loads on demand")
    n = len(skills)
    inline_per = (tok_all - tok_base) / n
    index_per = (tok_idx - tok_base_tools) / n
    print(f"  per skill: ~{inline_per:.0f} tok inlined vs ~{index_per:.0f} tok as an index entry -> with 40 skills: "
          f"~{40 * inline_per:,.0f} vs ~{40 * index_per + (tok_base_tools - tok_base):,.0f} tok per request")
    answer, trace = run_skill_agent(client, index_system, tools, by_name, fraud_text)
    print("  Request 1 (fraud check) - the model pulls in only what it needs:")
    for line in trace:
        print("    " + line)
    for line in answer.splitlines():
        print("    | " + line)
    answer2, trace2 = run_skill_agent(client, index_system, tools, by_name, CLAIMS["skill_requests"][1]["text"])
    print("  Request 2 (no matching skill) - nothing extra is loaded:")
    for line in trace2:
        print("    " + line)
    print("    | " + answer2)

    # ---------------------------------------------------------------- 9
    section("9. Agent Skills on the Claude API (live-only)")
    for line in SKILLS_API_SNIPPET.splitlines():
        print("    " + line)
    if is_live():
        run_live_skills(client, fraud_text)
    else:
        print("  [live-only] The offline mock has no Skills API or code-execution container. The loader above")
        print("  mirrors what the platform does: name+description in context, SKILL.md and files on demand.")

    section("Takeaways")
    print("  * Cache the big static prefix (exam Q2): static first, breakpoint at its end, verify cache_read > 0.")
    print("  * Hunt silent invalidators: timestamps, ids, unsorted JSON, varying tools; pick the right breakpoint.")
    print("  * Modular prompts: versioned components + canonical order = reuse AND a stable, shareable cache key.")
    print("  * Skills: package procedures as SKILL.md; only descriptions cost tokens until a task needs them.")


if __name__ == "__main__":
    main()
