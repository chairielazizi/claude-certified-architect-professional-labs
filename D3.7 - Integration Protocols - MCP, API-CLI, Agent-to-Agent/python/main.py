"""
D3.7 - Integration Protocols: MCP vs direct API/CLI vs agent-to-agent (A2A)

Three ways to connect Claude to capabilities, each implemented for real with the stdlib:

  (a) MCP      - a minimal MCP server (mcp_server.py) spawned as a subprocess and driven over
                 stdio with JSON-RPC 2.0: initialize/version negotiation, tools/list, tools/call,
                 resources. MCP tool schemas are converted into Claude tool definitions and a
                 Claude tool loop forwards each call to the server.
  (b) API/CLI  - plain function-calling tools: one wraps a local "REST" endpoint, one runs an
                 allowlisted CLI (git --version). No protocol, no extra process for the API tool.
  (c) A2A      - two agents over local HTTP on 127.0.0.1 (ephemeral port): agent-card discovery,
                 bearer auth, JSON-RPC message/send + tasks/get, task lifecycle with input-required.

Plus the Claude API MCP connector request shape (live-only) and a decision matrix.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_OPUS,
    MODEL_SONNET,
    MockRequest,
    banner,
    get_client,
    is_live,
    section,
    text_block,
    text_of,
    tool_use_block,
)

HERE = Path(__file__).resolve().parent


def compact(obj: Any) -> str:
    """Compact JSON (same bytes as JSON.stringify in the TS port, so wire sizes match)."""
    return json.dumps(obj, separators=(",", ":"))


# =============================================================================
# Shared: a manual Claude tool loop used by all three integration styles
# =============================================================================
Executor = Callable[[str, dict[str, Any]], tuple[str, bool]]


def run_tool_loop(client: anthropic.Anthropic, system: str, tools: list[dict[str, Any]], user_text: str,
                  execute: Executor, max_turns: int = 6) -> str:
    """Integration-agnostic agent loop. The ONLY thing that changes between MCP, direct API/CLI and A2A
    is `execute` - which is the architectural point: the protocol lives behind the tool boundary."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    for turn in range(1, max_turns + 1):
        resp = client.messages.create(model=MODEL_OPUS, max_tokens=2048, system=system, tools=tools, messages=messages)
        calls = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not calls:
            print(f"    turn {turn}: stop_reason={resp.stop_reason}")
            return text_of(resp).strip()
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for c in calls:  # parallel calls -> all results go back in ONE user message
            print(f"    turn {turn}: {c.name}({json.dumps(c.input, sort_keys=True)})")
            out, is_error = execute(c.name, c.input)
            print(f"            -> {'ERROR ' if is_error else ''}{out[:96]}{'...' if len(out) > 96 else ''}")
            res: dict[str, Any] = {"type": "tool_result", "tool_use_id": c.id, "content": out}
            if is_error:
                res["is_error"] = True
            results.append(res)
        messages.append({"role": "user", "content": results})
    return "(max turns reached)"


# =============================================================================
# (a) MCP client over stdio
# =============================================================================
class McpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code


class StdioMcpClient:
    """Minimal MCP client: spawns the server and speaks newline-delimited JSON-RPC 2.0 over its stdio."""

    def __init__(self, name: str, argv: list[str]) -> None:
        self.name = name
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.next_id = 1
        self.messages = 0
        self.bytes = 0

    def _send(self, msg: dict[str, Any]) -> None:
        data = (compact(msg) + "\n").encode("utf-8")
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()
        self.messages += 1
        self.bytes += len(data)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        mid = self.next_id
        self.next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise ConnectionError("MCP server closed the stream")
            self.messages += 1
            self.bytes += len(line)
            resp = json.loads(line)
            if resp.get("id") == mid:  # (server-initiated notifications would be dispatched here)
                break
        if "error" in resp:
            raise McpError(resp["error"]["code"], resp["error"]["message"])
        return resp["result"]

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def initialize(self) -> dict[str, Any]:
        result = self.request("initialize", {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "d37-lab-client", "version": "1.0.0"},
        })
        self.notify("notifications/initialized")
        return result

    def close(self) -> list[str]:
        assert self.proc.stdin is not None and self.proc.stderr is not None
        self.proc.stdin.close()  # stdio shutdown = close the server's stdin, then wait
        log = self.proc.stderr.read().decode("utf-8").splitlines()
        self.proc.wait(timeout=10)
        return log


def mcp_to_claude_tools(server: str, mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP `inputSchema` (camelCase) -> Claude `input_schema`; names are namespaced per server so two
    servers can both expose e.g. `search` without colliding."""
    return [{"name": f"{server}__{t['name']}", "description": t.get("description", ""), "input_schema": t["inputSchema"]}
            for t in mcp_tools]


# =============================================================================
# (b) Direct API + CLI tools
# =============================================================================
FX_RATES = {"USD/EUR": 0.92, "USD/GBP": 0.79, "EUR/USD": 1.087}


def pricing_api(method: str, path: str) -> tuple[int, dict[str, Any]]:
    """Stand-in for an internal REST service (GET /v1/fx?base=..&quote=..)."""
    url = urllib.parse.urlsplit(path)
    qs = dict(urllib.parse.parse_qsl(url.query))
    if method != "GET" or url.path != "/v1/fx":
        return 404, {"error": "not found"}
    pair = f"{qs.get('base')}/{qs.get('quote')}"
    if pair not in FX_RATES:
        return 404, {"error": f"no rate for {pair}"}
    return 200, {"base": qs["base"], "quote": qs["quote"], "rate": FX_RATES[pair], "as_of": "2026-10-10"}


CLI_ALLOWLIST: dict[str, list[str]] = {"git": ["git", "--version"], "runtime": [sys.executable, "--version"]}


def run_cli(program: str) -> tuple[str, bool]:
    """Allowlisted, argument-fixed, shell-free subprocess with a timeout - never pass model text to a shell."""
    if program not in CLI_ALLOWLIST:
        return f"blocked: '{program}' is not in the CLI allowlist", True
    argv = CLI_ALLOWLIST[program]
    exe = shutil.which(argv[0]) if program == "git" else argv[0]
    if exe is None:
        return f"{program} not found on PATH", True
    proc = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=10)
    return (proc.stdout or proc.stderr).strip(), proc.returncode != 0


DIRECT_TOOLS: list[dict[str, Any]] = [
    {"name": "get_fx_rate",
     "description": "Get the current FX rate from the internal pricing REST API (GET /v1/fx). Use for any currency conversion.",
     "input_schema": {"type": "object", "properties": {"base": {"type": "string"}, "quote": {"type": "string"}},
                      "required": ["base", "quote"]}},
    {"name": "cli_version",
     "description": "Report the installed version of an allowlisted command-line tool on this build agent.",
     "input_schema": {"type": "object", "properties": {"program": {"type": "string", "enum": ["git", "runtime"]}},
                      "required": ["program"]}},
]


def direct_execute(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    if name == "get_fx_rate":
        path = f"/v1/fx?base={args['base']}&quote={args['quote']}"
        status, body = pricing_api("GET", path)
        return compact(body), status != 200
    if name == "cli_version":
        return run_cli(args["program"])
    return f"unknown tool {name}", True


# =============================================================================
# (c) Agent-to-agent (A2A-style) over local HTTP
# =============================================================================
DEMO_TOKEN = "demo-token-123"
EU = {"DE", "FR", "NL", "IE", "ES", "IT"}


class LogisticsAgent:
    """The REMOTE agent. Owned by another team: its own model, tools and policies - callers only see
    the agent card and the task protocol, never its internals."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client
        self.tasks: dict[str, dict[str, Any]] = {}
        self.base_url = ""

    def card(self) -> dict[str, Any]:
        return {
            "protocolVersion": "0.3.0",
            "name": "Logistics Agent",
            "description": "Quotes and books parcel shipping for Acme Audio orders.",
            "url": self.base_url + "/a2a",
            "preferredTransport": "JSONRPC",
            "version": "2.1.0",
            "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": True},
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "security": [{"bearer": []}],
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["application/json", "text/plain"],
            "skills": [
                {"id": "shipping_quote", "name": "Shipping quote",
                 "description": "Quote carrier, price and ETA for a parcel. Needs items with weights, origin warehouse and destination country.",
                 "tags": ["shipping", "quote"], "examples": ["Quote 2 x HP-200 (0.6 kg each) from WH-EU to DE"]},
                {"id": "book_shipment", "name": "Book shipment",
                 "description": "Book a quoted shipment and return a tracking number.", "tags": ["shipping", "booking"]},
            ],
        }

    @staticmethod
    def _set_state(task: dict[str, Any], state: str, text: str | None = None) -> None:
        task["status"] = {"state": state}
        if text:
            task["status"]["message"] = {"role": "agent", "kind": "message", "parts": [{"kind": "text", "text": text}]}
        task["metadata"]["stateTransitions"].append(state)

    def handle_rpc(self, req: dict[str, Any]) -> dict[str, Any]:
        mid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        if method == "message/send":
            return {"jsonrpc": "2.0", "id": mid, "result": self.message_send(params["message"])}
        if method == "tasks/get":
            task = self.tasks.get(params.get("id", ""))
            if task is None:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32001, "message": "TaskNotFoundError"}}
            return {"jsonrpc": "2.0", "id": mid, "result": task}
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def message_send(self, message: dict[str, Any]) -> dict[str, Any]:
        task_id = message.get("taskId")
        if task_id and task_id in self.tasks:
            task = self.tasks[task_id]
        else:
            n = len(self.tasks) + 1
            task = {"id": f"task-{n:04d}", "contextId": f"ctx-{n:04d}", "kind": "task", "status": {},
                    "history": [], "artifacts": [], "metadata": {"stateTransitions": []}}
            self.tasks[task["id"]] = task
            self._set_state(task, "submitted")
        task["history"].append(message)
        skill = (message.get("metadata") or {}).get("skill_id", "shipping_quote")
        if skill != "shipping_quote":
            self._set_state(task, "rejected", f"Skill '{skill}' is not available for this request.")
            return task
        text = " ".join(p.get("text", "") for m in task["history"] for p in m["parts"] if p.get("kind") == "text")
        items = re.search(r"(\d+)\s*x\s*([A-Z]{2}-\d{3})", text)
        weight = re.search(r"(\d+(?:\.\d+)?)\s*kg", text)
        origin = re.search(r"\b(WH-[A-Z]+)\b", text)
        dest = re.search(r"\b(?:[Dd]estination|to)\b\s*:?\s*([A-Z]{2})\b", text)
        if not dest:
            self._set_state(task, "input-required", "Which destination country (ISO 3166 alpha-2 code) should I quote for?")
            return task
        self._set_state(task, "working")
        qty = int(items.group(1)) if items else 1
        kg = qty * float(weight.group(1)) if weight else 1.0
        intra_eu = (origin.group(1) if origin else "") == "WH-EU" and dest.group(1) in EU
        carrier, base, per_kg, eta = (("DHL Parcel", 900, 400, "2-4 business days") if intra_eu
                                      else ("UPS Worldwide", 2400, 900, "5-8 business days"))
        quote = {"carrier": carrier, "price_usd": round(base + per_kg * kg) / 100, "eta": eta,
                 "weight_kg": round(kg * 10) / 10, "destination": dest.group(1)}
        # The remote agent uses ITS OWN model to word the result - invisible to the caller.
        msg = self.client.messages.create(
            model=MODEL_SONNET, max_tokens=512,
            system="You are the Logistics Agent's quoting model. Summarize the quote in one sentence.",
            messages=[{"role": "user", "content": f"Quote JSON: {compact(quote)}"}])
        task["artifacts"] = [{"artifactId": "quote-1", "name": "shipping-quote",
                              "parts": [{"kind": "data", "data": quote}, {"kind": "text", "text": text_of(msg).strip()}]}]
        self._set_state(task, "completed", "Quote ready.")
        return task


def make_handler(agent: LogisticsAgent) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep console output deterministic
            pass

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            data = compact(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - discovery is public
            if self.path == "/.well-known/agent-card.json":
                self._reply(200, agent.card())
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            # Always drain the request body first, even when rejecting: closing a socket
            # with unread bytes makes Windows send a TCP RST, and the client then sees
            # ConnectionAbortedError (WinError 10053) instead of the 401 response.
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if self.headers.get("Authorization") != f"Bearer {DEMO_TOKEN}":
                self._reply(401, {"error": "missing or invalid bearer token"})
                return
            req = json.loads(raw.decode("utf-8"))
            self._reply(200, agent.handle_rpc(req))

    return Handler


def http_json(method: str, url: str, body: dict[str, Any] | None = None, token: str | None = None) -> tuple[int, dict[str, Any]]:
    data = compact(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class A2AClient:
    """Caller side: discover the agent card, turn skills into a Claude tool, run the task protocol."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url, self.token = base_url, token
        self.card: dict[str, Any] = {}
        self.rpc_id = 0
        self.msg_id = 0

    def discover(self) -> dict[str, Any]:
        _, self.card = http_json("GET", self.base_url + "/.well-known/agent-card.json")
        return self.card

    def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.rpc_id += 1
        _, body = http_json("POST", self.card["url"], {"jsonrpc": "2.0", "id": self.rpc_id, "method": method,
                                                       "params": params}, self.token)
        return body

    def delegate_tool(self) -> dict[str, Any]:
        skills = self.card["skills"]
        return {"name": "a2a_delegate",
                "description": (f"Delegate a task to the remote '{self.card['name']}': {self.card['description']} "
                                "Skills: " + "; ".join(f"{s['id']} - {s['description']}" for s in skills) +
                                " If the task comes back input-required, call again with the same task_id and the answer."),
                "input_schema": {"type": "object", "properties": {
                    "skill_id": {"type": "string", "enum": [s["id"] for s in skills]},
                    "message": {"type": "string"},
                    "task_id": {"type": "string", "description": "Continue an existing task"}},
                    "required": ["skill_id", "message"]}}

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if name != "a2a_delegate":
            return f"unknown tool {name}", True
        self.msg_id += 1
        message: dict[str, Any] = {"role": "user", "kind": "message", "messageId": f"msg-{self.msg_id:04d}",
                                   "parts": [{"kind": "text", "text": args["message"]}],
                                   "metadata": {"skill_id": args["skill_id"]}}
        if args.get("task_id"):
            message["taskId"] = args["task_id"]
        resp = self.rpc("message/send", {"message": message, "configuration": {"blocking": True,
                                                                             "acceptedOutputModes": ["application/json", "text/plain"]}})
        if "error" in resp:
            return compact(resp["error"]), True
        task = resp["result"]
        status = task["status"]
        summary: dict[str, Any] = {"task_id": task["id"], "state": status["state"]}
        if status.get("message"):
            summary["agent_message"] = status["message"]["parts"][0]["text"]
        if task["artifacts"]:
            summary["artifacts"] = [p.get("data", p.get("text")) for p in task["artifacts"][0]["parts"]]
        print(f"            a2a {task['id']} transitions so far: {' -> '.join(task['metadata']['stateTransitions'])}")
        return compact(summary), status["state"] in ("failed", "rejected")


# =============================================================================
# Offline responders: orchestrator (Opus) and the remote agent's own model (Sonnet)
# =============================================================================
def tool_history(req: MockRequest) -> list[tuple[str, dict[str, Any], str, bool]]:
    uses: dict[str, tuple[str, dict[str, Any]]] = {}
    out = []
    for m in req.messages:
        if not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_use":
                uses[b["id"]] = (b["name"], b["input"])
            elif b.get("type") == "tool_result" and b["tool_use_id"] in uses:
                name, inp = uses[b["tool_use_id"]]
                c = b.get("content")
                out.append((name, inp, c if isinstance(c, str) else "".join(x.get("text", "") for x in c or []),
                            bool(b.get("is_error"))))
    return out


def orchestrator_responder(req: MockRequest) -> Any:
    hist = tool_history(req)
    if req.system_text.startswith("You are the fulfilment assistant"):
        if not hist:
            return [text_block("Checking stock for both SKUs first."),
                    tool_use_block("inventory__get_stock", {"sku": "HP-200"}),
                    tool_use_block("inventory__get_stock", {"sku": "SP-100"})]
        stock = {json.loads(h[2])["sku"]: json.loads(h[2])["available"] for h in hist
                 if h[0] == "inventory__get_stock" and not h[3]}
        ranked = sorted(stock["HP-200"].items(), key=lambda kv: (-kv[1], kv[0]))
        reserves = [h for h in hist if h[0] == "inventory__reserve_stock"]
        if not reserves:
            return [tool_use_block("inventory__reserve_stock", {"sku": "HP-200", "quantity": 60, "warehouse": ranked[0][0]})]
        if all(h[3] for h in reserves):
            first = ranked[0][1]
            return [text_block(f"{ranked[0][0]} only has {first} units, so I'll split the reservation."),
                    tool_use_block("inventory__reserve_stock", {"sku": "HP-200", "quantity": first, "warehouse": ranked[0][0]}),
                    tool_use_block("inventory__reserve_stock", {"sku": "HP-200", "quantity": 60 - first, "warehouse": ranked[1][0]})]
        done = [json.loads(h[2]) for h in reserves if not h[3]]
        parts = " and ".join(f"{r['reservation_id']} ({r['quantity']} units from {r['warehouse']})" for r in done)
        sp = ", ".join(f"{k} {v}" for k, v in stock["SP-100"].items())
        return (f"Reserved 60 x HP-200 for ORD-10452: {parts}; the first attempt failed because no single warehouse "
                f"had 60 units. SP-100 availability: {sp}.")
    if req.system_text.startswith("You are the finance ops assistant"):
        if not hist:
            return [tool_use_block("get_fx_rate", {"base": "USD", "quote": "EUR"}), tool_use_block("cli_version", {"program": "git"})]
        fx = json.loads(next(h[2] for h in hist if h[0] == "get_fx_rate"))
        cli = next(h for h in hist if h[0] == "cli_version")
        git = f"the build agent reports '{cli[2]}'" if not cli[3] else f"git is unavailable ({cli[2]})"
        return f"250 USD is {250 * fx['rate']:.2f} EUR at {fx['rate']} (rate as of {fx['as_of']}); {git}."
    if req.system_text.startswith("You are the order orchestrator"):
        if not hist:
            return [text_block("I'll delegate the quote to the Logistics Agent."),
                    tool_use_block("a2a_delegate", {"skill_id": "shipping_quote",
                                                    "message": "Quote shipping for order ORD-10452: 2 x HP-200, 0.6 kg each, from WH-EU."})]
        last = json.loads(hist[-1][2])
        if last["state"] == "input-required":
            return [tool_use_block("a2a_delegate", {"skill_id": "shipping_quote", "message": "Destination: DE (Germany).",
                                                    "task_id": last["task_id"]})]
        quote = last["artifacts"][0]
        return (f"Shipping quote for ORD-10452 from the Logistics Agent ({last['task_id']}): {quote['carrier']}, "
                f"USD {quote['price_usd']:.2f}, {quote['eta']} for {quote['weight_kg']} kg to {quote['destination']}.")
    return "OK"


def remote_agent_responder(req: MockRequest) -> Any:
    quote = json.loads(req.last_user_text.split("Quote JSON: ", 1)[1])
    return (f"{quote['carrier']} can deliver {quote['weight_kg']} kg to {quote['destination']} in {quote['eta']} "
            f"for USD {quote['price_usd']:.2f}.")


# =============================================================================
# Demo
# =============================================================================
MCP_CONNECTOR_SNIPPET = """client.beta.messages.create(
    model="claude-opus-5-5", max_tokens=1024,
    betas=["mcp-client-2025-11-20"],
    mcp_servers=[{"type": "url", "url": MCP_SERVER_URL, "name": "inventory",
                  "authorization_token": MCP_TOKEN}],
    tools=[{"type": "mcp_toolset", "mcp_server_name": "inventory",
            "default_config": {"enabled": False},                 # allowlist mode
            "configs": {"get_stock": {"enabled": True}}}],
    messages=[{"role": "user", "content": "How many HP-200 are in WH-EU?"}],
)"""


def main() -> None:
    banner("D3.7 - Integration Protocols: MCP, API/CLI, Agent-to-Agent")
    client = get_client(orchestrator_responder)

    # -------------------------------------------------------------------------
    section("(a) MCP over stdio: lifecycle and discovery")
    server_cmd = [sys.executable, str(HERE / "mcp_server.py")]
    mcp = StdioMcpClient("inventory", server_cmd)
    print("  spawned MCP server subprocess: python mcp_server.py  (stdio transport)")
    try:
        mcp.request("tools/list")
    except McpError as exc:
        print(f"  -> tools/list before initialize   <- JSON-RPC error {exc}")
    init = mcp.initialize()
    caps = ", ".join(init["capabilities"])
    print(f"  -> initialize (client wants 2025-11-25)  <- protocolVersion={init['protocolVersion']} "
          f"server={init['serverInfo']['name']}/{init['serverInfo']['version']} capabilities=[{caps}]")
    print("  -> notifications/initialized (notification: no id, no response)")
    mcp_tools = mcp.request("tools/list")["tools"]
    for t in mcp_tools:
        hints = ", ".join(f"{k}={compact(v)}" for k, v in (t.get("annotations") or {}).items())
        print(f"  tool  {t['name']:<16} {hints}")
    res = mcp.request("resources/list")["resources"][0]
    body = mcp.request("resources/read", {"uri": res["uri"]})["contents"][0]["text"]
    print(f"  resource {res['uri']} -> \"{body.splitlines()[1][:70]}...\"")
    try:
        mcp.request("prompts/list")
    except McpError as exc:
        print(f"  -> prompts/list (not offered)     <- JSON-RPC error {exc}")
    print("  Tools are model-controlled, resources application-controlled, prompts user-controlled.")

    section("(a) MCP tools -> Claude tools, then a Claude tool loop")
    claude_tools = mcp_to_claude_tools("inventory", mcp_tools)
    print("  converted: " + ", ".join(t["name"] for t in claude_tools))
    print(f"  example  : {compact(claude_tools[0])[:110]}...")

    def mcp_execute(name: str, args: dict[str, Any]) -> tuple[str, bool]:
        server, _, tool = name.partition("__")
        if server != "inventory":
            return f"no MCP server '{server}'", True
        try:
            result = mcp.request("tools/call", {"name": tool, "arguments": args})
        except McpError as exc:  # protocol-level failure -> still a tool_result the model can read
            return f"MCP error {exc}", True
        text = "".join(c.get("text", "") for c in result["content"] if c.get("type") == "text")
        return text, bool(result.get("isError"))  # MCP isError -> Claude is_error

    answer = run_tool_loop(
        client,
        "You are the fulfilment assistant. Use the inventory tools; never reserve without checking stock first.",
        claude_tools,
        "Check stock for HP-200 and SP-100, then reserve 60 units of HP-200 for order ORD-10452 "
        "from the warehouse with the most stock.",
        mcp_execute)
    print(f"  answer: {answer}")
    mcp_msgs, mcp_bytes = mcp.messages, mcp.bytes
    server_log = [ln for ln in mcp.close() if ln.startswith("[inventory-mcp]")]
    init_line = next((ln for ln in server_log if "initialize from" in ln), "")
    print(f"  session: {mcp_msgs} JSON-RPC messages, {mcp_bytes} bytes over stdio; server exited cleanly")
    print(f"  server stderr (logging never goes to stdout): {len(server_log)} lines, e.g. \"{init_line}\"")

    # -------------------------------------------------------------------------
    section("(b) Direct API + CLI tools (no protocol layer)")
    status, fx = pricing_api("GET", "/v1/fx?base=USD&quote=EUR")
    print(f"  REST stand-in: GET /v1/fx?base=USD&quote=EUR -> {status} {compact(fx)}")
    blocked, _ = run_cli("curl")
    print(f"  CLI guard   : run_cli('curl') -> {blocked}")
    answer = run_tool_loop(
        client,
        "You are the finance ops assistant. Use tools for live data; do not guess rates or versions.",
        DIRECT_TOOLS,
        "What is 250 USD in EUR, and which git version is installed on this build agent?",
        direct_execute)
    print(f"  answer: {answer}")
    print("  Direct tools: 0 extra processes for the API call, 0 protocol bytes, schemas live in this codebase.")
    print("  Right when one app owns the integration; MCP pays off when many hosts reuse the same tools.")

    # -------------------------------------------------------------------------
    section("(c) Agent-to-agent: discovery, auth, task lifecycle")
    remote = LogisticsAgent(get_client(remote_agent_responder))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(remote))
    httpd.daemon_threads = True
    remote.base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print("  remote Logistics Agent listening on 127.0.0.1:<ephemeral port>")
    try:
        a2a = A2AClient(remote.base_url, DEMO_TOKEN)
        card = a2a.discover()
        print(f"  GET /.well-known/agent-card.json -> {card['name']} v{card['version']} (A2A {card['protocolVersion']}, "
              f"{card['preferredTransport']})")
        print(f"    skills: {', '.join(s['id'] for s in card['skills'])} | auth: {', '.join(card['securitySchemes'])} | "
              f"streaming={compact(card['capabilities']['streaming'])}")
        status, body = http_json("POST", card["url"], {"jsonrpc": "2.0", "id": 0, "method": "tasks/get", "params": {"id": "x"}})
        print(f"  POST without bearer token -> HTTP {status} {compact(body)}")
        tool = a2a.delegate_tool()
        print(f"  agent card -> Claude tool '{tool['name']}' (skill enum: {'|'.join(tool['input_schema']['properties']['skill_id']['enum'])})")
        answer = run_tool_loop(
            client,
            "You are the order orchestrator for Acme Audio. Delegate shipping work to the remote Logistics Agent with "
            "a2a_delegate; when a task is input-required, continue it with its task_id.",
            [tool],
            "Customer order ORD-10452: 2 x HP-200 (0.6 kg each) shipping from WH-EU to a customer in Germany. "
            "Get me a shipping quote.",
            a2a.execute)
        print(f"  answer: {answer}")
        final = a2a.rpc("tasks/get", {"id": "task-0001"})["result"]
        print(f"  tasks/get task-0001 -> state={final['status']['state']}; lifecycle: "
              f"{' -> '.join(final['metadata']['stateTransitions'])}")
        missing = a2a.rpc("tasks/get", {"id": "task-9999"})
        print(f"  tasks/get task-9999 -> JSON-RPC error {compact(missing['error'])}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
    print("  HTTP server shut down cleanly.")

    # -------------------------------------------------------------------------
    section("Claude API MCP connector (live-only)")
    print("Anthropic's API can act as the MCP client for REMOTE servers (Streamable HTTP) - no client loop:")
    for line in MCP_CONNECTOR_SNIPPET.splitlines():
        print("    " + line)
    if is_live() and os.environ.get("MCP_SERVER_URL"):
        msg = client.beta.messages.create(
            model=MODEL_OPUS, max_tokens=1024, betas=["mcp-client-2025-11-20"],
            mcp_servers=[{"type": "url", "url": os.environ["MCP_SERVER_URL"], "name": "inventory",
                          "authorization_token": os.environ.get("MCP_TOKEN")}],
            tools=[{"type": "mcp_toolset", "mcp_server_name": "inventory"}],
            messages=[{"role": "user", "content": "How many HP-200 are in WH-EU?"}],
        )
        print("  live result:", "".join(b.text for b in msg.content if b.type == "text")[:200])
    else:
        print("  [live-only] set LABS_MODE=live, ANTHROPIC_API_KEY and MCP_SERVER_URL (a reachable remote MCP")
        print("  server) to run it. Local stdio servers need a client-side bridge like section (a).")

    # -------------------------------------------------------------------------
    section("Decision matrix")
    rows = [
        ("", "MCP (stdio/HTTP)", "Direct API tool", "CLI subprocess", "A2A"),
        ("unit of reuse", "tools+resources", "one function", "existing CLI", "a whole agent"),
        ("standardization", "open protocol", "app-specific", "CLI contract", "open protocol"),
        ("discovery", "tools/list", "static in code", "--help/docs", "agent card"),
        ("auth", "proc creds/OAuth", "service creds", "local creds", "card schemes"),
        ("latency/overhead", "+hop, JSON-RPC", "lowest", "spawn per call", "multi-turn task"),
        ("who owns it", "tool provider", "your app team", "CLI vendor", "remote team"),
        ("pick it when", "many hosts reuse", "1 app, few tools", "dev/ops tools", "delegate tasks"),
    ]
    for r in rows:
        print("  " + r[0].ljust(18) + "".join(c.ljust(18) for c in r[1:]).rstrip())
    print("\n  Rule of thumb: function calling for app-owned capabilities; MCP when the same capability must")
    print("  plug into many Claude surfaces (Claude Code, Desktop, API connector, Agent SDK); A2A when you hand a")
    print("  goal to an autonomous agent you do not control and only need its task contract.")


if __name__ == "__main__":
    main()
