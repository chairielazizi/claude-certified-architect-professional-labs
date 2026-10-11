"""
Minimal MCP server over stdio, written from scratch (stdlib only) for the D3.7 lab.

Transport : newline-delimited JSON-RPC 2.0 on stdin/stdout (one message per line, no embedded
            newlines). stdout is reserved for protocol messages - all logging goes to stderr.
Lifecycle : initialize (version negotiation + capabilities) -> notifications/initialized -> requests.
Features  : tools/list, tools/call, resources/list, resources/read, ping.

Errors follow the MCP split:
  * protocol problems (unknown method, bad params, not initialized) -> JSON-RPC `error` object
  * tool execution failures (e.g. insufficient stock)               -> result with isError: true,
    so the model can read the message and recover.
"""

from __future__ import annotations

import json
import sys
from typing import Any

SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]  # newest first
SERVER_INFO = {"name": "inventory-mcp", "version": "1.2.0"}

STOCK: dict[str, dict[str, int]] = {
    "HP-200": {"WH-EAST": 25, "WH-WEST": 48, "WH-EU": 12},
    "SP-100": {"WH-EAST": 140, "WH-WEST": 0, "WH-EU": 61},
}
RESERVATIONS: list[dict[str, Any]] = []

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_stock",
        "title": "Get stock levels",
        "description": "Return available units of one SKU in every warehouse. Read-only. Call before reserving.",
        "inputSchema": {"type": "object", "properties": {"sku": {"type": "string", "description": "Product SKU, e.g. HP-200"}},
                        "required": ["sku"], "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "reserve_stock",
        "title": "Reserve stock",
        "description": "Reserve units of a SKU in ONE warehouse for an order. Fails if that warehouse lacks stock.",
        "inputSchema": {"type": "object", "properties": {
            "sku": {"type": "string"},
            "quantity": {"type": "integer", "description": "Units to reserve (> 0)"},
            "warehouse": {"type": "string", "enum": ["WH-EAST", "WH-WEST", "WH-EU"]}},
            "required": ["sku", "quantity", "warehouse"], "additionalProperties": False},
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    },
    {
        "name": "list_warehouses",
        "title": "List warehouses",
        "description": "List warehouse codes with their region.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True},
    },
]
RESOURCES = [{
    "uri": "inventory://policies/reservations",
    "name": "reservations-policy",
    "title": "Reservation policy",
    "mimeType": "text/markdown",
    "description": "Rules the inventory team applies to stock reservations.",
}]
RESOURCE_TEXT = ("# Reservation policy\nReservations hold stock for 48 hours. A single reservation cannot exceed the "
                 "units available in one warehouse; split large orders across warehouses.")


class ToolError(Exception):
    """A tool ran but failed in a way the model should see and handle."""


def tool_get_stock(args: dict[str, Any]) -> str:
    sku = args.get("sku")
    if sku not in STOCK:
        raise ToolError(f"Unknown SKU {sku}")
    return json.dumps({"sku": sku, "available": STOCK[sku]}, separators=(",", ":"))


def tool_reserve_stock(args: dict[str, Any]) -> str:
    sku, qty, wh = args.get("sku"), args.get("quantity"), args.get("warehouse")
    if sku not in STOCK or wh not in STOCK[sku] or not isinstance(qty, int) or qty <= 0:
        raise ToolError(f"Invalid reservation request: {json.dumps(args, separators=(',', ':'))}")
    if STOCK[sku][wh] < qty:
        raise ToolError(f"Insufficient stock in {wh}: {STOCK[sku][wh]} available, {qty} requested")
    STOCK[sku][wh] -= qty
    res = {"reservation_id": f"RSV-{len(RESERVATIONS) + 1:04d}", "sku": sku, "quantity": qty, "warehouse": wh,
           "remaining": STOCK[sku][wh]}
    RESERVATIONS.append(res)
    return json.dumps(res, separators=(",", ":"))


def tool_list_warehouses(_args: dict[str, Any]) -> str:
    return json.dumps({"WH-EAST": "North America East", "WH-WEST": "North America West", "WH-EU": "Europe"},
                      separators=(",", ":"))


HANDLERS = {"get_stock": tool_get_stock, "reserve_stock": tool_reserve_stock, "list_warehouses": tool_list_warehouses}
STATE = {"initialized": False, "protocol": ""}


def log(msg: str) -> None:
    print(f"[inventory-mcp] {msg}", file=sys.stderr, flush=True)


def ok(mid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def err(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if "id" not in msg:  # notification: never answered
        if method == "notifications/initialized":
            STATE["initialized"] = True
            log("client sent notifications/initialized")
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        STATE["protocol"] = version
        log(f"initialize from {params.get('clientInfo', {}).get('name')} (requested {requested}, using {version})")
        return ok(mid, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": "Inventory tools. Always call get_stock before reserve_stock.",
        })
    if method == "ping":
        return ok(mid, {})
    if not STATE["initialized"]:
        return err(mid, -32600, "Server not initialized: send initialize and notifications/initialized first")
    if method == "tools/list":
        return ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in HANDLERS:
            return err(mid, -32602, f"Unknown tool: {name}")
        try:
            text = HANDLERS[name](args)
            return ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})
        except ToolError as exc:
            return ok(mid, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
    if method == "resources/list":
        return ok(mid, {"resources": RESOURCES})
    if method == "resources/read":
        if params.get("uri") != RESOURCES[0]["uri"]:
            return err(mid, -32602, f"Unknown resource: {params.get('uri')}")
        return ok(mid, {"contents": [{"uri": RESOURCES[0]["uri"], "mimeType": "text/markdown", "text": RESOURCE_TEXT}]})
    return err(mid, -32601, f"Method not found: {method}")


def main() -> None:
    log("ready on stdio")
    for raw in sys.stdin.buffer:  # binary I/O: no newline translation on Windows
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp: dict[str, Any] | None = err(None, -32700, "Parse error")
        else:
            resp = handle(msg)
        if resp is not None:
            sys.stdout.buffer.write((json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
    log("stdin closed - shutting down")


if __name__ == "__main__":
    main()
