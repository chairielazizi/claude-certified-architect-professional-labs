/**
 * D3.7 - Integration Protocols: MCP vs direct API/CLI vs agent-to-agent (TypeScript port of python/main.py)
 *
 *  (a) MCP     - spawns mcpServer.ts (node, stdio), JSON-RPC 2.0 handshake, tools/list, tools/call,
 *                resources; MCP schemas -> Claude tools; Claude tool loop forwards calls to the server.
 *  (b) API/CLI - plain function-calling tools: a local "REST" endpoint and an allowlisted CLI.
 *  (c) A2A     - two agents over local HTTP (127.0.0.1, ephemeral port): agent card discovery, bearer
 *                auth, message/send + tasks/get, task lifecycle with input-required; clean shutdown.
 * Plus the Claude API MCP connector request shape (live-only) and a decision matrix.
 */

import Anthropic from "@anthropic-ai/sdk";
import { spawn, spawnSync } from "node:child_process";
import type { ChildProcessWithoutNullStreams } from "node:child_process";
import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MODEL_OPUS, MODEL_SONNET, banner, getClient, isLive, section, textBlock, textOf, toolUseBlock } from "../../shared/typescript/claudeClient.ts";
import type { Json, MockRequest, ResponderResult } from "../../shared/typescript/claudeClient.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** JSON formatted like Python's json.dumps(sort_keys=...) so trace lines match the Python port. */
function pyJson(value: unknown, sortKeys = false): string {
  if (Array.isArray(value)) return `[${value.map((v) => pyJson(v, sortKeys)).join(", ")}]`;
  if (value && typeof value === "object") {
    const keys = Object.keys(value as Json);
    if (sortKeys) keys.sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}: ${pyJson((value as Json)[k], sortKeys)}`).join(", ")}}`;
  }
  return JSON.stringify(value);
}
const compact = (obj: unknown): string => JSON.stringify(obj);

// =============================================================================
// Shared: a manual Claude tool loop used by all three integration styles
// =============================================================================
type Executor = (name: string, args: Json) => Promise<[string, boolean]>;

/** Integration-agnostic agent loop: only `execute` changes between MCP, direct API/CLI and A2A. */
async function runToolLoop(client: Anthropic, system: string, tools: Anthropic.Tool[], userText: string, execute: Executor, maxTurns = 6): Promise<string> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: userText }];
  for (let turn = 1; turn <= maxTurns; turn++) {
    const resp = await client.messages.create({ model: MODEL_OPUS, max_tokens: 2048, system, tools, messages });
    const calls = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
    if (resp.stop_reason !== "tool_use" || calls.length === 0) {
      console.log(`    turn ${turn}: stop_reason=${resp.stop_reason}`);
      return textOf(resp).trim();
    }
    messages.push({ role: "assistant", content: resp.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const c of calls) { // parallel calls -> all results go back in ONE user message
      console.log(`    turn ${turn}: ${c.name}(${pyJson(c.input, true)})`);
      const [out, isError] = await execute(c.name, c.input as Json);
      console.log(`            -> ${isError ? "ERROR " : ""}${out.slice(0, 96)}${out.length > 96 ? "..." : ""}`);
      results.push({ type: "tool_result", tool_use_id: c.id, content: out, ...(isError ? { is_error: true } : {}) });
    }
    messages.push({ role: "user", content: results });
  }
  return "(max turns reached)";
}

// =============================================================================
// (a) MCP client over stdio
// =============================================================================
class McpError extends Error {
  readonly code: number;
  constructor(code: number, message: string) {
    super(`${code} ${message}`);
    this.code = code;
  }
}

/** Minimal MCP client: spawns the server and speaks newline-delimited JSON-RPC 2.0 over its stdio. */
class StdioMcpClient {
  readonly name: string;
  readonly proc: ChildProcessWithoutNullStreams;
  nextId = 1;
  messages = 0;
  bytes = 0;
  private buffer = "";
  private readonly pending = new Map<number, (msg: Json) => void>();
  private readonly stderr: string[] = [];
  private readonly exited: Promise<number | null>;

  constructor(name: string, command: string, args: string[]) {
    this.name = name;
    this.proc = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => this.onData(chunk));
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (chunk: string) => this.stderr.push(chunk));
    this.exited = new Promise((resolve) => this.proc.on("close", (code) => resolve(code)));
  }

  private onData(chunk: string): void {
    this.buffer += chunk;
    let nl: number;
    while ((nl = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, nl);
      this.buffer = this.buffer.slice(nl + 1);
      if (!line.trim()) continue;
      this.messages += 1;
      this.bytes += Buffer.byteLength(line, "utf8") + 1;
      const msg = JSON.parse(line) as Json;
      const waiter = this.pending.get(msg.id); // (server-initiated notifications would be dispatched here)
      if (waiter) {
        this.pending.delete(msg.id);
        waiter(msg);
      }
    }
  }

  private send(msg: Json): void {
    const data = compact(msg) + "\n";
    this.proc.stdin.write(data);
    this.messages += 1;
    this.bytes += Buffer.byteLength(data, "utf8");
  }

  async request(method: string, params?: Json): Promise<Json> {
    const id = this.nextId++;
    const msg: Json = { jsonrpc: "2.0", id, method };
    if (params !== undefined) msg.params = params;
    const reply = new Promise<Json>((resolve) => this.pending.set(id, resolve));
    this.send(msg);
    const resp = await reply;
    if (resp.error) throw new McpError(resp.error.code, resp.error.message);
    return resp.result as Json;
  }

  notify(method: string): void {
    this.send({ jsonrpc: "2.0", method });
  }

  async initialize(): Promise<Json> {
    const result = await this.request("initialize", {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "d37-lab-client", version: "1.0.0" },
    });
    this.notify("notifications/initialized");
    return result;
  }

  async close(): Promise<string[]> {
    this.proc.stdin.end(); // stdio shutdown = close the server's stdin, then wait for exit
    await this.exited;
    return this.stderr.join("").split(/\r?\n/).filter((l) => l.length > 0);
  }
}

/** MCP `inputSchema` (camelCase) -> Claude `input_schema`; names namespaced per server to avoid collisions. */
function mcpToClaudeTools(server: string, mcpTools: Json[]): Anthropic.Tool[] {
  return mcpTools.map((t) => ({ name: `${server}__${t.name}`, description: t.description ?? "", input_schema: t.inputSchema }));
}

// =============================================================================
// (b) Direct API + CLI tools
// =============================================================================
const FX_RATES: Record<string, number> = { "USD/EUR": 0.92, "USD/GBP": 0.79, "EUR/USD": 1.087 };

/** Stand-in for an internal REST service (GET /v1/fx?base=..&quote=..). */
function pricingApi(method: string, pathAndQuery: string): [number, Json] {
  const url = new URL(pathAndQuery, "http://pricing.internal");
  if (method !== "GET" || url.pathname !== "/v1/fx") return [404, { error: "not found" }];
  const base = url.searchParams.get("base");
  const quote = url.searchParams.get("quote");
  const pair = `${base}/${quote}`;
  if (!(pair in FX_RATES)) return [404, { error: `no rate for ${pair}` }];
  return [200, { base, quote, rate: FX_RATES[pair], as_of: "2026-10-10" }];
}

const CLI_ALLOWLIST: Record<string, string[]> = { git: ["git", "--version"], runtime: [process.execPath, "--version"] };

/** Allowlisted, argument-fixed, shell-free subprocess with a timeout - never pass model text to a shell. */
function runCli(program: string): [string, boolean] {
  const argv = CLI_ALLOWLIST[program];
  if (!argv) return [`blocked: '${program}' is not in the CLI allowlist`, true];
  const proc = spawnSync(argv[0]!, argv.slice(1), { encoding: "utf8", timeout: 10_000, shell: false });
  if (proc.error) return [`${program} not found on PATH`, true];
  return [(proc.stdout || proc.stderr).trim(), proc.status !== 0];
}

const DIRECT_TOOLS: Anthropic.Tool[] = [
  {
    name: "get_fx_rate",
    description: "Get the current FX rate from the internal pricing REST API (GET /v1/fx). Use for any currency conversion.",
    input_schema: { type: "object", properties: { base: { type: "string" }, quote: { type: "string" } }, required: ["base", "quote"] },
  },
  {
    name: "cli_version",
    description: "Report the installed version of an allowlisted command-line tool on this build agent.",
    input_schema: { type: "object", properties: { program: { type: "string", enum: ["git", "runtime"] } }, required: ["program"] },
  },
];

async function directExecute(name: string, args: Json): Promise<[string, boolean]> {
  if (name === "get_fx_rate") {
    const [status, body] = pricingApi("GET", `/v1/fx?base=${args.base}&quote=${args.quote}`);
    return [compact(body), status !== 200];
  }
  if (name === "cli_version") return runCli(String(args.program));
  return [`unknown tool ${name}`, true];
}

// =============================================================================
// (c) Agent-to-agent (A2A-style) over local HTTP
// =============================================================================
const DEMO_TOKEN = "demo-token-123";
const EU = new Set(["DE", "FR", "NL", "IE", "ES", "IT"]);

/** The REMOTE agent: its own model, tools and policies - callers only see the card and the task protocol. */
class LogisticsAgent {
  readonly client: Anthropic;
  readonly tasks = new Map<string, Json>();
  baseUrl = "";

  constructor(client: Anthropic) {
    this.client = client;
  }

  card(): Json {
    return {
      protocolVersion: "0.3.0",
      name: "Logistics Agent",
      description: "Quotes and books parcel shipping for Acme Audio orders.",
      url: this.baseUrl + "/a2a",
      preferredTransport: "JSONRPC",
      version: "2.1.0",
      capabilities: { streaming: false, pushNotifications: false, stateTransitionHistory: true },
      securitySchemes: { bearer: { type: "http", scheme: "bearer" } },
      security: [{ bearer: [] }],
      defaultInputModes: ["text/plain"],
      defaultOutputModes: ["application/json", "text/plain"],
      skills: [
        {
          id: "shipping_quote", name: "Shipping quote",
          description: "Quote carrier, price and ETA for a parcel. Needs items with weights, origin warehouse and destination country.",
          tags: ["shipping", "quote"], examples: ["Quote 2 x HP-200 (0.6 kg each) from WH-EU to DE"],
        },
        { id: "book_shipment", name: "Book shipment", description: "Book a quoted shipment and return a tracking number.", tags: ["shipping", "booking"] },
      ],
    };
  }

  private static setState(task: Json, state: string, text?: string): void {
    task.status = { state };
    if (text) task.status.message = { role: "agent", kind: "message", parts: [{ kind: "text", text }] };
    task.metadata.stateTransitions.push(state);
  }

  async handleRpc(req: Json): Promise<Json> {
    const { id, method } = req;
    const params: Json = req.params ?? {};
    if (method === "message/send") return { jsonrpc: "2.0", id, result: await this.messageSend(params.message) };
    if (method === "tasks/get") {
      const task = this.tasks.get(params.id ?? "");
      return task ? { jsonrpc: "2.0", id, result: task } : { jsonrpc: "2.0", id, error: { code: -32001, message: "TaskNotFoundError" } };
    }
    return { jsonrpc: "2.0", id, error: { code: -32601, message: `Method not found: ${method}` } };
  }

  async messageSend(message: Json): Promise<Json> {
    let task = message.taskId ? this.tasks.get(message.taskId) : undefined;
    if (!task) {
      const n = String(this.tasks.size + 1).padStart(4, "0");
      task = { id: `task-${n}`, contextId: `ctx-${n}`, kind: "task", status: {}, history: [], artifacts: [], metadata: { stateTransitions: [] } };
      this.tasks.set(task.id, task);
      LogisticsAgent.setState(task, "submitted");
    }
    task.history.push(message);
    const skill = message.metadata?.skill_id ?? "shipping_quote";
    if (skill !== "shipping_quote") {
      LogisticsAgent.setState(task, "rejected", `Skill '${skill}' is not available for this request.`);
      return task;
    }
    const text = (task.history as Json[]).flatMap((m) => (m.parts as Json[]).filter((p) => p.kind === "text").map((p) => String(p.text))).join(" ");
    const items = /(\d+)\s*x\s*([A-Z]{2}-\d{3})/.exec(text);
    const weight = /(\d+(?:\.\d+)?)\s*kg/.exec(text);
    const origin = /\b(WH-[A-Z]+)\b/.exec(text);
    const dest = /\b(?:[Dd]estination|to)\b\s*:?\s*([A-Z]{2})\b/.exec(text);
    if (!dest) {
      LogisticsAgent.setState(task, "input-required", "Which destination country (ISO 3166 alpha-2 code) should I quote for?");
      return task;
    }
    LogisticsAgent.setState(task, "working");
    const qty = items ? Number(items[1]) : 1;
    const kg = weight ? qty * Number(weight[1]) : 1.0;
    const intraEu = (origin ? origin[1] : "") === "WH-EU" && EU.has(dest[1]!);
    const [carrier, base, perKg, eta] = intraEu ? ["DHL Parcel", 900, 400, "2-4 business days"] as const : ["UPS Worldwide", 2400, 900, "5-8 business days"] as const;
    const quote = { carrier, price_usd: Math.round(base + perKg * kg) / 100, eta, weight_kg: Math.round(kg * 10) / 10, destination: dest[1] };
    // The remote agent uses ITS OWN model to word the result - invisible to the caller.
    const msg = await this.client.messages.create({
      model: MODEL_SONNET, max_tokens: 512,
      system: "You are the Logistics Agent's quoting model. Summarize the quote in one sentence.",
      messages: [{ role: "user", content: `Quote JSON: ${compact(quote)}` }],
    });
    task.artifacts = [{ artifactId: "quote-1", name: "shipping-quote", parts: [{ kind: "data", data: quote }, { kind: "text", text: textOf(msg).trim() }] }];
    LogisticsAgent.setState(task, "completed", "Quote ready.");
    return task;
  }
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function reply(res: ServerResponse, status: number, body: Json): void {
  const data = compact(body);
  res.writeHead(status, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) });
  res.end(data);
}

async function httpJson(method: string, url: string, body?: Json, token?: string): Promise<[number, Json]> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const resp = await fetch(url, { method, headers, ...(body !== undefined ? { body: compact(body) } : {}) });
  return [resp.status, (await resp.json()) as Json];
}

/** Caller side: discover the agent card, turn skills into a Claude tool, run the task protocol. */
class A2AClient {
  readonly baseUrl: string;
  readonly token: string;
  card: Json = {};
  rpcId = 0;
  msgId = 0;

  constructor(baseUrl: string, token: string) {
    this.baseUrl = baseUrl;
    this.token = token;
  }

  async discover(): Promise<Json> {
    [, this.card] = await httpJson("GET", this.baseUrl + "/.well-known/agent-card.json");
    return this.card;
  }

  async rpc(method: string, params: Json): Promise<Json> {
    this.rpcId += 1;
    const [, body] = await httpJson("POST", this.card.url, { jsonrpc: "2.0", id: this.rpcId, method, params }, this.token);
    return body;
  }

  delegateTool(): Anthropic.Tool {
    const skills = this.card.skills as Json[];
    return {
      name: "a2a_delegate",
      description: `Delegate a task to the remote '${this.card.name}': ${this.card.description} ` +
        "Skills: " + skills.map((s) => `${s.id} - ${s.description}`).join("; ") +
        " If the task comes back input-required, call again with the same task_id and the answer.",
      input_schema: {
        type: "object",
        properties: {
          skill_id: { type: "string", enum: skills.map((s) => s.id) },
          message: { type: "string" },
          task_id: { type: "string", description: "Continue an existing task" },
        },
        required: ["skill_id", "message"],
      },
    };
  }

  readonly execute: Executor = async (name, args) => {
    if (name !== "a2a_delegate") return [`unknown tool ${name}`, true];
    this.msgId += 1;
    const message: Json = {
      role: "user", kind: "message", messageId: `msg-${String(this.msgId).padStart(4, "0")}`,
      parts: [{ kind: "text", text: args.message }], metadata: { skill_id: args.skill_id },
    };
    if (args.task_id) message.taskId = args.task_id;
    const resp = await this.rpc("message/send", { message, configuration: { blocking: true, acceptedOutputModes: ["application/json", "text/plain"] } });
    if (resp.error) return [compact(resp.error), true];
    const task = resp.result as Json;
    const summary: Json = { task_id: task.id, state: task.status.state };
    if (task.status.message) summary.agent_message = task.status.message.parts[0].text;
    if (task.artifacts.length) summary.artifacts = (task.artifacts[0].parts as Json[]).map((p) => p.data ?? p.text);
    console.log(`            a2a ${task.id} transitions so far: ${(task.metadata.stateTransitions as string[]).join(" -> ")}`);
    return [compact(summary), task.status.state === "failed" || task.status.state === "rejected"];
  };
}

// =============================================================================
// Offline responders: orchestrator (Opus) and the remote agent's own model (Sonnet)
// =============================================================================
function toolHistory(req: MockRequest): [string, Json, string, boolean][] {
  const uses = new Map<string, [string, Json]>();
  const out: [string, Json, string, boolean][] = [];
  for (const m of req.messages) {
    if (!Array.isArray(m.content)) continue;
    for (const b of m.content as Json[]) {
      if (b.type === "tool_use") uses.set(b.id, [b.name, b.input]);
      else if (b.type === "tool_result" && uses.has(b.tool_use_id)) {
        const [name, inp] = uses.get(b.tool_use_id)!;
        const text = typeof b.content === "string" ? b.content : ((b.content ?? []) as Json[]).map((x) => String(x.text ?? "")).join("");
        out.push([name, inp, text, Boolean(b.is_error)]);
      }
    }
  }
  return out;
}

function orchestratorResponder(req: MockRequest): ResponderResult {
  const hist = toolHistory(req);
  if (req.systemText.startsWith("You are the fulfilment assistant")) {
    if (!hist.length) {
      return [textBlock("Checking stock for both SKUs first."),
        toolUseBlock("inventory__get_stock", { sku: "HP-200" }), toolUseBlock("inventory__get_stock", { sku: "SP-100" })];
    }
    const stock: Record<string, Record<string, number>> = {};
    for (const h of hist) if (h[0] === "inventory__get_stock" && !h[3]) stock[JSON.parse(h[2]).sku] = JSON.parse(h[2]).available;
    const ranked = Object.entries(stock["HP-200"]!).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    const reserves = hist.filter((h) => h[0] === "inventory__reserve_stock");
    if (!reserves.length) return [toolUseBlock("inventory__reserve_stock", { sku: "HP-200", quantity: 60, warehouse: ranked[0]![0] })];
    if (reserves.every((h) => h[3])) {
      const first = ranked[0]![1];
      return [textBlock(`${ranked[0]![0]} only has ${first} units, so I'll split the reservation.`),
        toolUseBlock("inventory__reserve_stock", { sku: "HP-200", quantity: first, warehouse: ranked[0]![0] }),
        toolUseBlock("inventory__reserve_stock", { sku: "HP-200", quantity: 60 - first, warehouse: ranked[1]![0] })];
    }
    const done = reserves.filter((h) => !h[3]).map((h) => JSON.parse(h[2]) as Json);
    const parts = done.map((r) => `${r.reservation_id} (${r.quantity} units from ${r.warehouse})`).join(" and ");
    const sp = Object.entries(stock["SP-100"]!).map(([k, v]) => `${k} ${v}`).join(", ");
    return `Reserved 60 x HP-200 for ORD-10452: ${parts}; the first attempt failed because no single warehouse ` +
      `had 60 units. SP-100 availability: ${sp}.`;
  }
  if (req.systemText.startsWith("You are the finance ops assistant")) {
    if (!hist.length) return [toolUseBlock("get_fx_rate", { base: "USD", quote: "EUR" }), toolUseBlock("cli_version", { program: "git" })];
    const fx = JSON.parse(hist.find((h) => h[0] === "get_fx_rate")![2]) as Json;
    const cli = hist.find((h) => h[0] === "cli_version")!;
    const git = !cli[3] ? `the build agent reports '${cli[2]}'` : `git is unavailable (${cli[2]})`;
    return `250 USD is ${(250 * fx.rate).toFixed(2)} EUR at ${fx.rate} (rate as of ${fx.as_of}); ${git}.`;
  }
  if (req.systemText.startsWith("You are the order orchestrator")) {
    if (!hist.length) {
      return [textBlock("I'll delegate the quote to the Logistics Agent."),
        toolUseBlock("a2a_delegate", { skill_id: "shipping_quote", message: "Quote shipping for order ORD-10452: 2 x HP-200, 0.6 kg each, from WH-EU." })];
    }
    const last = JSON.parse(hist[hist.length - 1]![2]) as Json;
    if (last.state === "input-required") {
      return [toolUseBlock("a2a_delegate", { skill_id: "shipping_quote", message: "Destination: DE (Germany).", task_id: last.task_id })];
    }
    const quote = last.artifacts[0] as Json;
    return `Shipping quote for ORD-10452 from the Logistics Agent (${last.task_id}): ${quote.carrier}, ` +
      `USD ${Number(quote.price_usd).toFixed(2)}, ${quote.eta} for ${quote.weight_kg} kg to ${quote.destination}.`;
  }
  return "OK";
}

function remoteAgentResponder(req: MockRequest): ResponderResult {
  const quote = JSON.parse(req.lastUserText.split("Quote JSON: ")[1]!) as Json;
  return `${quote.carrier} can deliver ${quote.weight_kg} kg to ${quote.destination} in ${quote.eta} for USD ${Number(quote.price_usd).toFixed(2)}.`;
}

// =============================================================================
// Demo
// =============================================================================
const MCP_CONNECTOR_SNIPPET = `await client.beta.messages.create({
  model: "claude-opus-5-5", max_tokens: 1024,
  betas: ["mcp-client-2025-11-20"],
  mcp_servers: [{ type: "url", url: MCP_SERVER_URL, name: "inventory",
                  authorization_token: MCP_TOKEN }],
  tools: [{ type: "mcp_toolset", mcp_server_name: "inventory",
            default_config: { enabled: false },              // allowlist mode
            configs: { get_stock: { enabled: true } } }],
  messages: [{ role: "user", content: "How many HP-200 are in WH-EU?" }],
});`;

banner("D3.7 - Integration Protocols: MCP, API/CLI, Agent-to-Agent");
const client = getClient(orchestratorResponder);

// -----------------------------------------------------------------------------
section("(a) MCP over stdio: lifecycle and discovery");
const mcp = new StdioMcpClient("inventory", process.execPath, [path.join(HERE, "mcpServer.ts")]);
console.log("  spawned MCP server subprocess: node mcpServer.ts  (stdio transport)");
try {
  await mcp.request("tools/list");
} catch (exc) {
  if (!(exc instanceof McpError)) throw exc;
  console.log(`  -> tools/list before initialize   <- JSON-RPC error ${exc.message}`);
}
const init = await mcp.initialize();
console.log(
  `  -> initialize (client wants 2025-11-25)  <- protocolVersion=${init.protocolVersion} ` +
    `server=${init.serverInfo.name}/${init.serverInfo.version} capabilities=[${Object.keys(init.capabilities).join(", ")}]`,
);
console.log("  -> notifications/initialized (notification: no id, no response)");
const mcpTools = (await mcp.request("tools/list")).tools as Json[];
for (const t of mcpTools) {
  const hints = Object.entries((t.annotations ?? {}) as Json).map(([k, v]) => `${k}=${compact(v)}`).join(", ");
  console.log(`  tool  ${String(t.name).padEnd(16)} ${hints}`);
}
const res0 = ((await mcp.request("resources/list")).resources as Json[])[0]!;
const resBody = String(((await mcp.request("resources/read", { uri: res0.uri })).contents as Json[])[0]!.text);
console.log(`  resource ${res0.uri} -> "${resBody.split("\n")[1]!.slice(0, 70)}..."`);
try {
  await mcp.request("prompts/list");
} catch (exc) {
  if (!(exc instanceof McpError)) throw exc;
  console.log(`  -> prompts/list (not offered)     <- JSON-RPC error ${exc.message}`);
}
console.log("  Tools are model-controlled, resources application-controlled, prompts user-controlled.");

section("(a) MCP tools -> Claude tools, then a Claude tool loop");
const claudeTools = mcpToClaudeTools("inventory", mcpTools);
console.log("  converted: " + claudeTools.map((t) => t.name).join(", "));
console.log(`  example  : ${compact(claudeTools[0]).slice(0, 110)}...`);

const mcpExecute: Executor = async (name, args) => {
  const sep = name.indexOf("__");
  const server = sep >= 0 ? name.slice(0, sep) : name;
  const tool = sep >= 0 ? name.slice(sep + 2) : "";
  if (server !== "inventory") return [`no MCP server '${server}'`, true];
  let result: Json;
  try {
    result = await mcp.request("tools/call", { name: tool, arguments: args });
  } catch (exc) { // protocol-level failure -> still a tool_result the model can read
    if (!(exc instanceof McpError)) throw exc;
    return [`MCP error ${exc.message}`, true];
  }
  const text = (result.content as Json[]).filter((c) => c.type === "text").map((c) => String(c.text)).join("");
  return [text, Boolean(result.isError)]; // MCP isError -> Claude is_error
};

let answer = await runToolLoop(
  client,
  "You are the fulfilment assistant. Use the inventory tools; never reserve without checking stock first.",
  claudeTools,
  "Check stock for HP-200 and SP-100, then reserve 60 units of HP-200 for order ORD-10452 " +
    "from the warehouse with the most stock.",
  mcpExecute,
);
console.log(`  answer: ${answer}`);
const [mcpMsgs, mcpBytes] = [mcp.messages, mcp.bytes];
const serverLog = (await mcp.close()).filter((l) => l.startsWith("[inventory-mcp]"));
const initLine = serverLog.find((l) => l.includes("initialize from")) ?? "";
console.log(`  session: ${mcpMsgs} JSON-RPC messages, ${mcpBytes} bytes over stdio; server exited cleanly`);
console.log(`  server stderr (logging never goes to stdout): ${serverLog.length} lines, e.g. "${initLine}"`);

// -----------------------------------------------------------------------------
section("(b) Direct API + CLI tools (no protocol layer)");
const [fxStatus, fx] = pricingApi("GET", "/v1/fx?base=USD&quote=EUR");
console.log(`  REST stand-in: GET /v1/fx?base=USD&quote=EUR -> ${fxStatus} ${compact(fx)}`);
console.log(`  CLI guard   : run_cli('curl') -> ${runCli("curl")[0]}`);
answer = await runToolLoop(
  client,
  "You are the finance ops assistant. Use tools for live data; do not guess rates or versions.",
  DIRECT_TOOLS,
  "What is 250 USD in EUR, and which git version is installed on this build agent?",
  directExecute,
);
console.log(`  answer: ${answer}`);
console.log("  Direct tools: 0 extra processes for the API call, 0 protocol bytes, schemas live in this codebase.");
console.log("  Right when one app owns the integration; MCP pays off when many hosts reuse the same tools.");

// -----------------------------------------------------------------------------
section("(c) Agent-to-agent: discovery, auth, task lifecycle");
const remote = new LogisticsAgent(getClient(remoteAgentResponder));
const server = createServer((req, res) => {
  void (async () => {
    try {
      if (req.method === "GET") {
        if (req.url === "/.well-known/agent-card.json") reply(res, 200, remote.card()); // discovery is public
        else reply(res, 404, { error: "not found" });
        return;
      }
      if (req.headers.authorization !== `Bearer ${DEMO_TOKEN}`) {
        reply(res, 401, { error: "missing or invalid bearer token" });
        return;
      }
      reply(res, 200, await remote.handleRpc(JSON.parse(await readBody(req)) as Json));
    } catch (exc) { // never let a handler failure become an unhandled rejection
      reply(res, 500, { error: exc instanceof Error ? exc.message : String(exc) });
    }
  })();
});
await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
remote.baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
console.log("  remote Logistics Agent listening on 127.0.0.1:<ephemeral port>");
try {
  const a2a = new A2AClient(remote.baseUrl, DEMO_TOKEN);
  const card = await a2a.discover();
  console.log(`  GET /.well-known/agent-card.json -> ${card.name} v${card.version} (A2A ${card.protocolVersion}, ${card.preferredTransport})`);
  console.log(
    `    skills: ${(card.skills as Json[]).map((s) => s.id).join(", ")} | auth: ${Object.keys(card.securitySchemes).join(", ")} | ` +
      `streaming=${compact(card.capabilities.streaming)}`,
  );
  const [st, body] = await httpJson("POST", card.url, { jsonrpc: "2.0", id: 0, method: "tasks/get", params: { id: "x" } });
  console.log(`  POST without bearer token -> HTTP ${st} ${compact(body)}`);
  const tool = a2a.delegateTool();
  const skillEnum = ((tool.input_schema.properties as Json).skill_id.enum as string[]).join("|");
  console.log(`  agent card -> Claude tool '${tool.name}' (skill enum: ${skillEnum})`);
  answer = await runToolLoop(
    client,
    "You are the order orchestrator for Acme Audio. Delegate shipping work to the remote Logistics Agent with " +
      "a2a_delegate; when a task is input-required, continue it with its task_id.",
    [tool],
    "Customer order ORD-10452: 2 x HP-200 (0.6 kg each) shipping from WH-EU to a customer in Germany. " +
      "Get me a shipping quote.",
    a2a.execute,
  );
  console.log(`  answer: ${answer}`);
  const final = (await a2a.rpc("tasks/get", { id: "task-0001" })).result as Json;
  console.log(`  tasks/get task-0001 -> state=${final.status.state}; lifecycle: ${(final.metadata.stateTransitions as string[]).join(" -> ")}`);
  const missing = await a2a.rpc("tasks/get", { id: "task-9999" });
  console.log(`  tasks/get task-9999 -> JSON-RPC error ${compact(missing.error)}`);
} finally {
  server.closeAllConnections(); // drop keep-alive sockets so close() resolves immediately
  await new Promise<void>((resolve, reject) => server.close((e) => (e ? reject(e) : resolve())));
}
console.log("  HTTP server shut down cleanly.");

// -----------------------------------------------------------------------------
section("Claude API MCP connector (live-only)");
console.log("Anthropic's API can act as the MCP client for REMOTE servers (Streamable HTTP) - no client loop:");
for (const line of MCP_CONNECTOR_SNIPPET.split("\n")) console.log("    " + line);
if (isLive() && process.env.MCP_SERVER_URL) {
  const msg = await client.beta.messages.create({
    model: MODEL_OPUS, max_tokens: 1024, betas: ["mcp-client-2025-11-20"],
    mcp_servers: [{ type: "url", url: process.env.MCP_SERVER_URL, name: "inventory", authorization_token: process.env.MCP_TOKEN ?? null }],
    tools: [{ type: "mcp_toolset", mcp_server_name: "inventory" }],
    messages: [{ role: "user", content: "How many HP-200 are in WH-EU?" }],
  });
  const text = msg.content.filter((b): b is Anthropic.Beta.BetaTextBlock => b.type === "text").map((b) => b.text).join("");
  console.log("  live result:", text.slice(0, 200));
} else {
  console.log("  [live-only] set LABS_MODE=live, ANTHROPIC_API_KEY and MCP_SERVER_URL (a reachable remote MCP");
  console.log("  server) to run it. Local stdio servers need a client-side bridge like section (a).");
}

// -----------------------------------------------------------------------------
section("Decision matrix");
const rows: string[][] = [
  ["", "MCP (stdio/HTTP)", "Direct API tool", "CLI subprocess", "A2A"],
  ["unit of reuse", "tools+resources", "one function", "existing CLI", "a whole agent"],
  ["standardization", "open protocol", "app-specific", "CLI contract", "open protocol"],
  ["discovery", "tools/list", "static in code", "--help/docs", "agent card"],
  ["auth", "proc creds/OAuth", "service creds", "local creds", "card schemes"],
  ["latency/overhead", "+hop, JSON-RPC", "lowest", "spawn per call", "multi-turn task"],
  ["who owns it", "tool provider", "your app team", "CLI vendor", "remote team"],
  ["pick it when", "many hosts reuse", "1 app, few tools", "dev/ops tools", "delegate tasks"],
];
for (const r of rows) console.log(("  " + r[0]!.padEnd(18) + r.slice(1).map((c) => c.padEnd(18)).join("")).trimEnd());
console.log("\n  Rule of thumb: function calling for app-owned capabilities; MCP when the same capability must");
console.log("  plug into many Claude surfaces (Claude Code, Desktop, API connector, Agent SDK); A2A when you hand a");
console.log("  goal to an autonomous agent you do not control and only need its task contract.");
