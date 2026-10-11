"""
D3.1 - Tool Configuration and Capability Bloat
Exam objective: Evaluate tool/agent configuration for capability bloat.

What this lab builds
--------------------
1. A tool-configuration auditor that compares an agent's tool manifest with the
   capabilities a role actually needs (least privilege) and flags excess,
   dangerous, unused and overlapping tools.
2. A schema-overhead measurement (count_tokens) of the bloated vs trimmed manifest.
3. A manual tool-use loop (through the shared SDK client) run three times against a
   ticket containing an indirect prompt injection that asks for delete_account:
     A. bloated manifest, no controls      -> the account is deleted
     B. bloated manifest + executor authz  -> the call is denied (is_error result)
     C. trimmed manifest                   -> the capability does not exist
4. A seeded Monte Carlo that ranks compensating controls by residual harm:
   removal (preventive) > authorization check > confirmation > logging (detective).
5. Per-role tool allowlists, reused to build an MCP connector toolset allowlist.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_SONNET,
    PRICING,
    MockRequest,
    banner,
    get_client,
    is_live,
    section,
    text_block,
    text_of,
    tool_use_block,
    usage_line,
)

DATA = Path(__file__).resolve().parents[1] / "data"
# A tier-1 support assistant is high-volume, latency-sensitive work: Sonnet is the
# usual production choice. Model choice does NOT change the capability surface.
MODEL = MODEL_SONNET
ROLE = "support_staff"
STAFF_REQUEST = "Summarize ticket T-1042 and draft a reply for me to review."


# ---------------------------------------------------------------------------
# Deterministic PRNG (identical sequence in the TypeScript version)
# ---------------------------------------------------------------------------
class Mulberry32:
    """Tiny seeded PRNG. Python's `random` and JS Math.random differ, so both
    languages use this to produce byte-identical simulations."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


# ---------------------------------------------------------------------------
# 1. Tool-configuration auditor (pure code - this is the architect's checklist)
# ---------------------------------------------------------------------------
# Severity of an UNNEEDED tool depends on what it can do, not on how often it is used.
SEVERITY_BY_EFFECT = {
    "destructive": "CRITICAL",
    "financial": "CRITICAL",
    "external_write": "HIGH",
    "write": "MEDIUM",
    "draft": "LOW",
    "read": "LOW",
}
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "by", "on", "in", "is", "it", "this",
    "that", "with", "from", "call", "when", "before", "does", "not", "using", "up", "any", "all",
}
OVERLAP_THRESHOLD = 0.40


@dataclass
class Finding:
    tool: str
    kind: str
    severity: str
    detail: str
    action: str


@dataclass
class AuditResult:
    findings: list[Finding]
    keep: list[str]
    remove: list[str]


def tokens_of(tool: dict[str, Any]) -> set[str]:
    """Bag of meaningful words from a tool's description and parameter names."""
    words = re.split(r"[^a-z0-9]+", tool.get("description", "").lower())
    for param in (tool["input_schema"].get("properties") or {}):
        words.extend(param.lower().split("_"))
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def audit_manifest(cfg: dict[str, Any], role: str) -> AuditResult:
    tools: list[dict[str, Any]] = cfg["tools"]
    meta: dict[str, dict[str, Any]] = cfg["tool_metadata"]
    needs = set(cfg["roles"][role]["needs"])
    usage: dict[str, int] = cfg["usage_30d"].get(role, {})
    findings: list[Finding] = []
    keep: list[str] = []
    remove: list[str] = []

    for tool in tools:
        name = tool["name"]
        m = meta[name]
        calls = usage.get(name, 0)
        if m.get("covers"):
            # A generic "escape hatch" tool (SQL, shell, HTTP) re-grants every capability
            # you removed elsewhere and gives the harness only an opaque string to gate.
            excess = sorted(set(m["covers"]) - needs)
            findings.append(Finding(
                name, "ESCAPE_HATCH", "CRITICAL",
                f"generic tool covers {len(m['covers'])} capabilities, {len(excess)} not needed "
                f"({', '.join(excess)}); {calls} calls/30d",
                "REMOVE - replace with narrow, typed tools"))
            remove.append(name)
        elif m["capability"] in needs:
            keep.append(name)
        else:
            sev = SEVERITY_BY_EFFECT[m["effect"]]
            if not m["reversible"]:
                findings.append(Finding(
                    name, "EXCESS_DANGEROUS", sev,
                    f"{m['effect']}, irreversible, not in role spec; {calls} calls/30d",
                    "REMOVE from this role's manifest"))
                remove.append(name)
            elif calls > 0:
                findings.append(Finding(
                    name, "UNDECLARED_USE", sev,
                    f"{m['effect']} capability '{m['capability']}' used {calls:,}x/30d but not in role spec",
                    "REVIEW with process owner (update spec or remove)"))
                remove.append(name)
            else:
                findings.append(Finding(
                    name, "EXCESS_UNUSED", sev,
                    f"not in role spec and 0 calls/30d", "REMOVE"))
                remove.append(name)

    # Overlap: near-duplicate tools make Claude pick the wrong one and double schema cost.
    for i, a in enumerate(tools):
        for b in tools[i + 1:]:
            ma, mb = meta[a["name"]], meta[b["name"]]
            sim = jaccard(tokens_of(a), tokens_of(b))
            same_cap = ma["capability"] == mb["capability"]
            if same_cap or sim >= OVERLAP_THRESHOLD:
                findings.append(Finding(
                    f"{a['name']} ~ {b['name']}", "OVERLAP", "MEDIUM",
                    f"description similarity {sim:.2f}" + (", same capability" if same_cap else ""),
                    "CONSOLIDATE into one clearly-described tool"))

    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.kind, f.tool))
    return AuditResult(findings, keep, remove)


# ---------------------------------------------------------------------------
# 2. Tool executor - where real controls live (never in the prompt)
# ---------------------------------------------------------------------------
@dataclass
class ToolExecutor:
    cfg: dict[str, Any]
    role: str
    allowed_tools: set[str]
    enforce_authz: bool = False
    accounts: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_log: list[str] = field(default_factory=list)

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Return (content, is_error). Every call is logged (detective control)."""
        if name not in self.allowed_tools:
            # Defense in depth: even if the model names a tool that was not offered,
            # the harness refuses to run it.
            return self._log(name, args, f"Error: tool '{name}' is not available to this agent.", True)
        if self.enforce_authz:
            cap = self.cfg["tool_metadata"][name]["capability"]
            if cap not in self.cfg["roles"][self.role]["needs"]:
                return self._log(name, args,
                                 f"PermissionDenied: role '{self.role}' lacks capability '{cap}'.", True)
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return self._log(name, args, f"Error: no handler for '{name}'.", True)
        return self._log(name, args, handler(args), False)

    def _log(self, name: str, args: dict[str, Any], content: str, is_error: bool) -> tuple[str, bool]:
        outcome = "DENIED" if is_error else "OK"
        args_text = json.dumps(args, sort_keys=True)
        if len(args_text) > 60:
            args_text = args_text[:57] + "..."
        self.audit_log.append(f"role={self.role} tool={name} args={args_text} -> {outcome}")
        return content, is_error

    # --- simulated back-end handlers -------------------------------------
    def _tool_read_ticket(self, args: dict[str, Any]) -> str:
        t = self.cfg["tickets"].get(args["ticket_id"])
        if not t:
            return f"No ticket {args['ticket_id']}"
        return f"Ticket {args['ticket_id']} (customer {t['customer_id']})\nSubject: {t['subject']}\n{t['body']}"

    def _tool_search_knowledge_base(self, args: dict[str, Any]) -> str:
        return "KB-112: Duplicate charges are refunded within 5 business days after billing verifies them."

    def _tool_draft_reply(self, args: dict[str, Any]) -> str:
        return f"Draft saved on {args['ticket_id']} ({len(args['body'])} chars) - awaiting human review."

    def _tool_issue_refund(self, args: dict[str, Any]) -> str:
        return f"Refund of ${args['amount_usd']:.2f} issued for {args['order_id']}."

    def _tool_delete_account(self, args: dict[str, Any]) -> str:
        acct = self.accounts.get(args["customer_id"])
        if acct is None:
            return f"No account {args['customer_id']}"
        acct["status"] = "DELETED"
        return f"Account {args['customer_id']} permanently deleted."

    def _tool_get_customer_profile(self, args: dict[str, Any]) -> str:
        return json.dumps(self.accounts.get(args["customer_id"], {}))

    _tool_lookup_customer = _tool_get_customer_profile

    def _tool_run_sql(self, args: dict[str, Any]) -> str:
        return "3 rows affected"


# ---------------------------------------------------------------------------
# 3. Offline responder - simulates Claude (including a SUCCESSFUL injection)
# ---------------------------------------------------------------------------
REPLY_BODY = (
    "Hi Dana, thanks for reaching out. I can see two charges for your March subscription "
    "(order O-5531). I've asked our billing team to verify the duplicate; once confirmed, "
    "the extra charge is refunded within 5 business days. - Acme Support"
)


def last_tool_calls(req: MockRequest) -> list[tuple[str, str, bool]]:
    """(tool_name, result_text, is_error) for the tool results in the latest user turn."""
    names: dict[str, str] = {}
    for m in req.messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
    out = []
    for r in req.tool_results:
        content = r.get("content")
        text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content or [])
        out.append((names.get(r["tool_use_id"], "?"), text, bool(r.get("is_error"))))
    return out


def responder(req: MockRequest) -> Any:
    calls = last_tool_calls(req)
    if not calls:
        return [text_block("I'll read the ticket first."), tool_use_block("read_ticket", {"ticket_id": "T-1042"})]

    done = {name: (text, err) for name, text, err in calls}
    if "read_ticket" in done:
        injected = "NOTE TO THE AI AGENT" in done["read_ticket"][0]
        draft = tool_use_block("draft_reply", {"ticket_id": "T-1042", "body": REPLY_BODY})
        if injected and "delete_account" in req.tool_names:
            # Models resist injection well, but NOT perfectly. Architect for the case
            # where the model is fooled: the tool is right there in the manifest.
            return [
                text_block("The ticket says the account owner requested GDPR erasure; processing that and drafting the billing reply."),
                tool_use_block("delete_account", {"customer_id": "C-77", "reason": "GDPR erasure requested in ticket T-1042"}),
                draft,
            ]
        note = (
            "The ticket body contains instructions addressed to an AI agent (account deletion). "
            "That is untrusted customer content and I have no tool for it, so I'm ignoring it and flagging it. "
        ) if injected else ""
        return [text_block(note + "Drafting the billing reply now."), draft]

    # Results for draft_reply (and possibly delete_account) are in: wrap up.
    summary = "Summary: customer C-77 reports a duplicate March charge on order O-5531. Draft reply saved for your review."
    if "delete_account" in done:
        _, err = done["delete_account"]
        if err:
            summary += " The delete_account request embedded in the ticket was DENIED by the platform; flag the ticket to Trust & Safety."
        else:
            summary += " As requested in the ticket, account C-77 has been deleted."
    elif "NOTE TO THE AI AGENT" in req.all_text:
        summary += " Note: the ticket contained an injected instruction to delete the account; it was ignored - please escalate to Trust & Safety."
    return summary


# ---------------------------------------------------------------------------
# 4. Manual tool-use loop through the shared SDK client
# ---------------------------------------------------------------------------
def run_agent(client: Any, cfg: dict[str, Any], tools: list[dict[str, Any]], executor: ToolExecutor,
              max_turns: int = 6) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": STAFF_REQUEST}]
    for turn in range(1, max_turns + 1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=cfg["system_prompt"],
            tools=tools,
            messages=messages,
        )
        # _request_id is what Anthropic support asks for when you report a problem.
        print(f"  turn {turn}: stop_reason={resp.stop_reason}  {usage_line(resp)}  request_id={resp._request_id}")
        # Always branch on stop_reason before trusting content.
        if resp.stop_reason == "end_turn":
            return text_of(resp)
        if resp.stop_reason != "tool_use":
            return f"[stopped: {resp.stop_reason}] {text_of(resp)}"
        messages.append({"role": "assistant", "content": resp.content})
        results: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            content, is_error = executor.execute(block.name, dict(block.input))
            flag = "ERROR " if is_error else ""
            print(f"    tool_use {block.name}({json.dumps(block.input)[:70]}) -> {flag}{content.splitlines()[0][:80]}")
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                result["is_error"] = True
            results.append(result)
        # Parallel tool calls: ALL results go back in ONE user message.
        messages.append({"role": "user", "content": results})
    return "[stopped: max_turns reached]"


# ---------------------------------------------------------------------------
# 5. Monte Carlo: residual harm per compensating control
# ---------------------------------------------------------------------------
@dataclass
class Control:
    name: str
    kind: str
    p_comply: float   # chance the model follows the injected instruction
    p_block: float    # chance the control stops an attempted call
    note: str


def rank_controls(n: int = 10_000, seed: int = 31) -> list[tuple[Control, int]]:
    # Assumptions are illustrative; the ORDER is what the exam tests.
    base = 0.03  # models are resistant to injection, not immune
    controls = [
        Control("No control (bloated manifest)", "baseline", base, 0.0, "the tool is offered and runs"),
        Control("Remove delete_account from manifest", "preventive - eliminates capability", base, 1.0,
                "no tool to call; unknown tool names rejected by harness"),
        Control("Authorization check in tool executor", "preventive - code-enforced", base, 0.999,
                "residual = policy misconfiguration (0.1%)"),
        Control("Human confirmation before execution", "preventive - human-dependent", base, 0.85,
                "approval fatigue: 15% rubber-stamped"),
        Control("Audit logging + alerting", "detective - after the fact", base, 0.0,
                "every deletion detected... after it happened"),
        Control("Upgrade to a larger model", "not a control", base / 2, 0.0,
                "may resist more often; capability still exposed"),
    ]
    ranked: list[tuple[Control, int]] = []
    for c in controls:
        rng = Mulberry32(seed)  # common random numbers: every control faces the same attack stream
        executed = 0
        for _ in range(n):
            if rng.next() < c.p_comply and rng.next() >= c.p_block:
                executed += 1
        ranked.append((c, executed))
    return ranked


# ---------------------------------------------------------------------------
# 6. Per-role allowlists (and the MCP connector equivalent)
# ---------------------------------------------------------------------------
def build_allowlist(cfg: dict[str, Any], role: str) -> dict[str, str]:
    """tool -> 'allow' | 'gated'. One canonical tool per capability, no escape hatches."""
    meta = cfg["tool_metadata"]
    spec = cfg["roles"][role]
    usage_all: dict[str, int] = {}
    for per_role in cfg["usage_30d"].values():
        for name, n in per_role.items():
            usage_all[name] = usage_all.get(name, 0) + n
    out: dict[str, str] = {}
    for cap in spec["needs"]:
        candidates = [t["name"] for t in cfg["tools"]
                      if meta[t["name"]]["capability"] == cap and not meta[t["name"]].get("covers")]
        if not candidates:
            continue
        # Prefer the most-used tool when two overlap (data-driven consolidation).
        best = sorted(candidates, key=lambda n: (-usage_all.get(n, 0), n))[0]
        out[best] = "gated" if cap in spec.get("gated", []) else "allow"
    return out


def mcp_toolset_allowlist(server_name: str, allowed: list[str]) -> dict[str, Any]:
    """MCP connector allowlist mode: disable everything, then enable named tools."""
    return {
        "type": "mcp_toolset",
        "mcp_server_name": server_name,
        "default_config": {"enabled": False},
        "configs": {name: {"enabled": True} for name in allowed},
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    banner("D3.1 - Tool Configuration and Capability Bloat")
    cfg = json.loads((DATA / "support_agent.json").read_text(encoding="utf-8"))
    client = get_client(responder)

    section("1. The agent's current tool manifest")
    print(f"Agent '{cfg['agent']}' on {MODEL} exposes {len(cfg['tools'])} tools:")
    for t in cfg["tools"]:
        m = cfg["tool_metadata"][t["name"]]
        rev = "reversible" if m["reversible"] else "IRREVERSIBLE"
        print(f"  - {t['name']:<22} {m['capability']:<16} {m['effect']:<12} {rev}")
    role_spec = cfg["roles"][ROLE]
    print(f"Role '{ROLE}': {role_spec['description']}")
    print(f"  needs: {', '.join(role_spec['needs'])}")

    section("2. Audit: manifest vs role capabilities (least privilege)")
    result = audit_manifest(cfg, ROLE)
    print(f"{'SEVERITY':<9} {'KIND':<17} TOOL")
    for f in result.findings:
        print(f"{f.severity:<9} {f.kind:<17} {f.tool}")
        print(f"{'':<28}{f.detail}")
        print(f"{'':<28}-> {f.action}")
    print(f"\nKEEP   ({len(result.keep)}): {', '.join(result.keep)}")
    print(f"REMOVE ({len(result.remove)}): {', '.join(result.remove)}")
    trimmed_tools = [t for t in cfg["tools"] if t["name"] in result.keep]

    section("3. Schema token overhead (client.messages.count_tokens)")
    probe = [{"role": "user", "content": STAFF_REQUEST}]

    def count(tools: list[dict[str, Any]] | None) -> int:
        kwargs: dict[str, Any] = {"model": MODEL, "system": cfg["system_prompt"], "messages": probe}
        if tools:
            kwargs["tools"] = tools
        return client.messages.count_tokens(**kwargs).input_tokens

    baseline = count(None)
    bloated = count(cfg["tools"])
    trimmed = count(trimmed_tools)
    print(f"Prompt without tools         : {baseline:>5} input tokens")
    print(f"Bloated manifest (8 tools)   : {bloated:>5}  (+{bloated - baseline} for schemas)")
    print(f"Trimmed manifest (3 tools)   : {trimmed:>5}  (+{trimmed - baseline} for schemas)")
    print("Per-tool schema cost:")
    for t in cfg["tools"]:
        mark = "keep" if t["name"] in result.keep else "drop"
        print(f"  {t['name']:<22} {count([t]) - baseline:>4} tokens  [{mark}]")
    per_day = 40_000 * 3  # 40k tickets/day x ~3 model calls per ticket
    delta = bloated - trimmed
    price = PRICING[MODEL]["input"]
    print(f"At {per_day:,} requests/day: {delta} extra tokens/request = {delta * per_day / 1e6:.1f}M tokens/day "
          f"= ${delta * per_day * price / 1e6:,.2f}/day uncached (${delta * per_day * price / 1e6 * 365:,.0f}/yr).")
    print("Caching discounts those tokens but they still occupy context and every extra tool is one more")
    print("option the model can mis-select - bloat costs accuracy, latency and attack surface, not just $.")

    accounts_initial = cfg["accounts"]

    section("4. Run A - bloated manifest, no controls (injected ticket)")
    exec_a = ToolExecutor(cfg, ROLE, {t["name"] for t in cfg["tools"]},
                          accounts=copy.deepcopy(accounts_initial))
    final_a = run_agent(client, cfg, cfg["tools"], exec_a)
    print(f"  final: {final_a}")
    print(f"  >>> account C-77 status: {exec_a.accounts['C-77']['status']}")

    section("5. Run B - bloated manifest + authorization check in the executor")
    exec_b = ToolExecutor(cfg, ROLE, {t["name"] for t in cfg["tools"]}, enforce_authz=True,
                          accounts=copy.deepcopy(accounts_initial))
    final_b = run_agent(client, cfg, cfg["tools"], exec_b)
    print(f"  final: {final_b}")
    print(f"  >>> account C-77 status: {exec_b.accounts['C-77']['status']}  (the model still TRIED)")

    section("6. Run C - trimmed manifest (capability removed)")
    exec_c = ToolExecutor(cfg, ROLE, set(result.keep), accounts=copy.deepcopy(accounts_initial))
    final_c = run_agent(client, cfg, trimmed_tools, exec_c)
    print(f"  final: {final_c}")
    print(f"  >>> account C-77 status: {exec_c.accounts['C-77']['status']}  (nothing to call)")
    print("Audit log (run C):")
    for line in exec_c.audit_log:
        print(f"  {line}")

    section("7. Ranking compensating controls (10,000 injected tickets, seeded)")
    print("Illustrative assumptions: model follows the injection 3% of the time (1.5% for a larger model);")
    print("authz misconfigured 0.1%; reviewers rubber-stamp 15% of prompts. The ORDER is the lesson.")
    print(f"{'CONTROL':<38} {'TYPE':<36} {'DELETIONS':>9}  NOTE")
    ranked = sorted(rank_controls(), key=lambda pair: pair[1])
    for c, executed in ranked:
        print(f"{c.name:<38} {c.kind:<36} {executed:>9}  {c.note}")
    print("Order to apply: REMOVE what the role does not need; authorize in code what it does need;")
    print("confirm irreversible actions with a human; LOG everything (detective, never sufficient alone).")

    section("8. Per-role tool allowlists (Y = allowed, G = gated by human approval)")
    roles = list(cfg["roles"].keys())
    allow = {r: build_allowlist(cfg, r) for r in roles}
    print(f"{'TOOL':<22} " + " ".join(f"{r:<19}" for r in roles))
    for t in cfg["tools"]:
        cells = []
        for r in roles:
            mode = allow[r].get(t["name"])
            cells.append(f"{('G' if mode == 'gated' else 'Y') if mode else '-':<19}")
        print(f"{t['name']:<22} " + " ".join(cells))
    toolset = mcp_toolset_allowlist("acme-support", list(allow[ROLE].keys()))
    print(f"\nSame allowlist as an MCP connector toolset for '{ROLE}':")
    print("  " + json.dumps(toolset))
    if is_live() and os.environ.get("D31_MCP_URL"):
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=1024, betas=["mcp-client-2025-11-20"],
            mcp_servers=[{"type": "url", "url": os.environ["D31_MCP_URL"], "name": "acme-support"}],
            tools=[toolset],
            messages=[{"role": "user", "content": STAFF_REQUEST}],
        )
        print(f"  live MCP call stop_reason={resp.stop_reason} request_id={resp._request_id}")
    else:
        print("  [live-only] client.beta.messages.create(..., betas=['mcp-client-2025-11-20'],")
        print("              mcp_servers=[{type:'url', url:..., name:'acme-support'}], tools=[toolset])")
        print("              (set LABS_MODE=live and D31_MCP_URL to run against a real MCP server)")

    section("Takeaways")
    print("* Capability bloat = tools/permissions/autonomy beyond what the role needs (OWASP LLM06 Excessive Agency).")
    print("* The fix for 'staff only need read + draft' is to REMOVE refund/delete tools - not logging,")
    print("  not confirmation prompts, not a bigger model, not a stern system prompt.")
    print("* Generic tools (SQL/shell/HTTP) silently re-grant removed capabilities - replace with narrow tools.")
    print("* Keep manifests per role, stable and ordered (they are part of the cached prefix).")


if __name__ == "__main__":
    main()
