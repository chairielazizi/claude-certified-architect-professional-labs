/**
 * Minimal MCP server over stdio, written from scratch (Node built-ins only) for the D3.7 lab.
 * Spawned by main.ts as `node mcpServer.ts` (Node 24 runs .ts directly via type stripping).
 *
 * Transport : newline-delimited JSON-RPC 2.0 on stdin/stdout; stdout is reserved for protocol
 *             messages, all logging goes to stderr.
 * Lifecycle : initialize (version negotiation + capabilities) -> notifications/initialized -> requests.
 * Errors    : protocol problems -> JSON-RPC `error`; tool execution failures -> result.isError = true.
 * Mirrors python/mcp_server.py byte-for-byte on the wire.
 */

import { createInterface } from "node:readline";

type Json = Record<string, any>;

const SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]; // newest first
const SERVER_INFO = { name: "inventory-mcp", version: "1.2.0" };

const STOCK: Record<string, Record<string, number>> = {
  "HP-200": { "WH-EAST": 25, "WH-WEST": 48, "WH-EU": 12 },
  "SP-100": { "WH-EAST": 140, "WH-WEST": 0, "WH-EU": 61 },
};
const RESERVATIONS: Json[] = [];

const TOOLS: Json[] = [
  {
    name: "get_stock",
    title: "Get stock levels",
    description: "Return available units of one SKU in every warehouse. Read-only. Call before reserving.",
    inputSchema: {
      type: "object", properties: { sku: { type: "string", description: "Product SKU, e.g. HP-200" } },
      required: ["sku"], additionalProperties: false,
    },
    annotations: { readOnlyHint: true, openWorldHint: false },
  },
  {
    name: "reserve_stock",
    title: "Reserve stock",
    description: "Reserve units of a SKU in ONE warehouse for an order. Fails if that warehouse lacks stock.",
    inputSchema: {
      type: "object",
      properties: {
        sku: { type: "string" },
        quantity: { type: "integer", description: "Units to reserve (> 0)" },
        warehouse: { type: "string", enum: ["WH-EAST", "WH-WEST", "WH-EU"] },
      },
      required: ["sku", "quantity", "warehouse"], additionalProperties: false,
    },
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false },
  },
  {
    name: "list_warehouses",
    title: "List warehouses",
    description: "List warehouse codes with their region.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false },
    annotations: { readOnlyHint: true },
  },
];
const RESOURCES = [{
  uri: "inventory://policies/reservations",
  name: "reservations-policy",
  title: "Reservation policy",
  mimeType: "text/markdown",
  description: "Rules the inventory team applies to stock reservations.",
}];
const RESOURCE_TEXT =
  "# Reservation policy\nReservations hold stock for 48 hours. A single reservation cannot exceed the " +
  "units available in one warehouse; split large orders across warehouses.";

/** A tool ran but failed in a way the model should see and handle. */
class ToolError extends Error {}

function toolGetStock(args: Json): string {
  const sku = args.sku;
  if (!(sku in STOCK)) throw new ToolError(`Unknown SKU ${sku}`);
  return JSON.stringify({ sku, available: STOCK[sku] });
}

function toolReserveStock(args: Json): string {
  const { sku, quantity: qty, warehouse: wh } = args;
  const levels = STOCK[sku];
  if (!levels || !(wh in levels) || !Number.isInteger(qty) || qty <= 0) {
    throw new ToolError(`Invalid reservation request: ${JSON.stringify(args)}`);
  }
  if (levels[wh]! < qty) throw new ToolError(`Insufficient stock in ${wh}: ${levels[wh]} available, ${qty} requested`);
  levels[wh] = levels[wh]! - qty;
  const res = {
    reservation_id: `RSV-${String(RESERVATIONS.length + 1).padStart(4, "0")}`,
    sku, quantity: qty, warehouse: wh, remaining: levels[wh],
  };
  RESERVATIONS.push(res);
  return JSON.stringify(res);
}

function toolListWarehouses(): string {
  return JSON.stringify({ "WH-EAST": "North America East", "WH-WEST": "North America West", "WH-EU": "Europe" });
}

const HANDLERS: Record<string, (args: Json) => string> = {
  get_stock: toolGetStock, reserve_stock: toolReserveStock, list_warehouses: toolListWarehouses,
};
const STATE = { initialized: false, protocol: "" };

const log = (msg: string): void => {
  process.stderr.write(`[inventory-mcp] ${msg}\n`);
};
const ok = (id: unknown, result: Json): Json => ({ jsonrpc: "2.0", id, result });
const err = (id: unknown, code: number, message: string): Json => ({ jsonrpc: "2.0", id, error: { code, message } });

function handle(msg: Json): Json | null {
  const method = msg.method;
  const id = msg.id;
  const params: Json = msg.params ?? {};
  if (!("id" in msg)) { // notification: never answered
    if (method === "notifications/initialized") {
      STATE.initialized = true;
      log("client sent notifications/initialized");
    }
    return null;
  }
  if (method === "initialize") {
    const requested = params.protocolVersion;
    const version = SUPPORTED_VERSIONS.includes(requested) ? requested : SUPPORTED_VERSIONS[0]!;
    STATE.protocol = version;
    log(`initialize from ${params.clientInfo?.name} (requested ${requested}, using ${version})`);
    return ok(id, {
      protocolVersion: version,
      capabilities: { tools: { listChanged: false }, resources: { subscribe: false, listChanged: false } },
      serverInfo: SERVER_INFO,
      instructions: "Inventory tools. Always call get_stock before reserve_stock.",
    });
  }
  if (method === "ping") return ok(id, {});
  if (!STATE.initialized) return err(id, -32600, "Server not initialized: send initialize and notifications/initialized first");
  if (method === "tools/list") return ok(id, { tools: TOOLS });
  if (method === "tools/call") {
    const name = params.name;
    const args: Json = params.arguments ?? {};
    const handler = HANDLERS[name];
    if (!handler) return err(id, -32602, `Unknown tool: ${name}`);
    try {
      return ok(id, { content: [{ type: "text", text: handler(args) }], isError: false });
    } catch (exc) {
      if (!(exc instanceof ToolError)) throw exc;
      return ok(id, { content: [{ type: "text", text: exc.message }], isError: true });
    }
  }
  if (method === "resources/list") return ok(id, { resources: RESOURCES });
  if (method === "resources/read") {
    if (params.uri !== RESOURCES[0]!.uri) return err(id, -32602, `Unknown resource: ${params.uri}`);
    return ok(id, { contents: [{ uri: RESOURCES[0]!.uri, mimeType: "text/markdown", text: RESOURCE_TEXT }] });
  }
  return err(id, -32601, `Method not found: ${method}`);
}

log("ready on stdio");
const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (raw) => {
  const line = raw.trim();
  if (!line) return;
  let resp: Json | null;
  try {
    resp = handle(JSON.parse(line) as Json);
  } catch (exc) {
    if (!(exc instanceof SyntaxError)) throw exc;
    resp = err(null, -32700, "Parse error");
  }
  if (resp !== null) process.stdout.write(JSON.stringify(resp) + "\n");
});
rl.on("close", () => log("stdin closed - shutting down"));
