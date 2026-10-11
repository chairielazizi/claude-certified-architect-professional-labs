/**
 * D3.6 - Retrieval Strategies by Data Shape and Query Pattern (TypeScript port of python/main.py)
 *
 * data shapes : structured table (orders.csv), semi-structured JSON (catalog.json), prose
 *               (policies.md), source code (data/code), knowledge graph (graph.json + joins)
 * router      : regex pre-router for IDs -> Claude Haiku classifier with structured output (zod)
 *               -> deterministic fallback rules on low confidence / errors
 * strategies  : key lookup, aggregation plan (Python uses sqlite3; here the SAME validated plan runs
 *               in memory), attribute filter, BM25, vector (local stand-in embeddings), hybrid RRF,
 *               code-aware BM25, graph traversal, agentic tool-use loop
 */

import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { MODEL_HAIKU, MODEL_OPUS, banner, getClient, section, textBlock, textOf, toolUseBlock } from "../../shared/typescript/claudeClient.ts";
import type { Json, MockRequest, ResponderResult } from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
interface QueryItem { id: string; query: string; expected: string[] }
const QUERIES = JSON.parse(readFileSync(path.join(DATA, "queries.json"), "utf8")) as { as_of: string; queries: QueryItem[] };
const AS_OF = QUERIES.as_of; // fixed "today" so warranty math is deterministic

/** JSON text formatted like Python's json.dumps (", " and ": " separators) so both ports print identically. */
function pyJson(value: unknown, sortKeys = false): string {
  if (Array.isArray(value)) return `[${value.map((v) => pyJson(v, sortKeys)).join(", ")}]`;
  if (value && typeof value === "object") {
    const keys = Object.keys(value as Json);
    if (sortKeys) keys.sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}: ${pyJson((value as Json)[k], sortKeys)}`).join(", ")}}`;
  }
  return JSON.stringify(value);
}

// =============================================================================
// 1. Load the five data shapes
// =============================================================================
interface Order {
  order_id: string; order_date: string; customer_id: string; customer_name: string; region: string;
  sku: string; quantity: number; unit_price: string; unit_cents: number; status: string;
}
interface Product {
  sku: string; name: string; category: string; price: number; warranty_months: number; supplier_id: string;
  attributes: { wireless: boolean; battery_h: number | null; noise_cancelling: boolean; waterproof: boolean; weight_g: number };
  description: string;
}

function loadOrders(): Order[] {
  const lines = readFileSync(path.join(DATA, "orders.csv"), "utf8").replace(/\r\n/g, "\n").trim().split("\n");
  const header = lines[0]!.split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(","); // simple CSV: no quoted fields in this fixture
    const r: Record<string, string> = {};
    header.forEach((h, i) => (r[h] = cells[i]!));
    return {
      order_id: r.order_id!, order_date: r.order_date!, customer_id: r.customer_id!, customer_name: r.customer_name!,
      region: r.region!, sku: r.sku!, quantity: Number(r.quantity), unit_price: r.unit_price!,
      unit_cents: Math.round(Number(r.unit_price) * 100), // money as integer cents: exact sums
      status: r.status!,
    };
  });
}

const ORDERS = loadOrders();
const ORDERS_BY_ID = new Map(ORDERS.map((o) => [o.order_id, o] as [string, Order]));
const CATALOG = (JSON.parse(readFileSync(path.join(DATA, "catalog.json"), "utf8")) as { products: Product[] }).products;
const CATALOG_BY_SKU = new Map(CATALOG.map((p) => [p.sku, p] as [string, Product]));
const GRAPH_DATA = JSON.parse(readFileSync(path.join(DATA, "graph.json"), "utf8")) as {
  suppliers: { id: string; name: string; country: string }[];
  edges: { src: string; rel: string; dst: string }[];
};

interface TextDoc { doc_id: string; kind: string; title: string; text: string }

function loadTextDocs(): TextDoc[] {
  const docs: TextDoc[] = [];
  const body = readFileSync(path.join(DATA, "policies.md"), "utf8").replace(/\r\n/g, "\n");
  for (const block of body.split("\n## ").slice(1)) {
    const nl = block.indexOf("\n");
    const title = (nl >= 0 ? block.slice(0, nl) : block).trim();
    const text = nl >= 0 ? block.slice(nl + 1).trim() : "";
    docs.push({ doc_id: `policy:${title.toLowerCase().replace(/ /g, "-")}`, kind: "policy", title, text });
  }
  for (const p of CATALOG) docs.push({ doc_id: `product:${p.sku}`, kind: "product", title: p.name, text: `${p.name}. ${p.description}` });
  return docs;
}

interface CodeChunk { file: string; start: number; end: number; symbol: string; text: string }

const TOP_LEVEL = /^(def |async def |class |export function |function |export class |CREATE )/;
const SYMBOL = /(?:def|function|class|VIEW)\s+(\w+)/;

/** Code-aware chunking: one chunk per top-level definition (leading comments attached). */
function chunkCode(file: string, text: string): CodeChunk[] {
  const lines = text.replace(/\r\n/g, "\n").replace(/\n+$/, "").split("\n");
  const starts: number[] = [];
  lines.forEach((line, i) => {
    if (TOP_LEVEL.test(line)) {
      let j = i;
      while (j > 0 && ["//", "--", "#"].some((p) => lines[j - 1]!.trim().startsWith(p))) j -= 1;
      starts.push(j);
    }
  });
  if (!starts.length || starts[0]! > 0) starts.unshift(0);
  const chunks: CodeChunk[] = [];
  starts.forEach((s, n) => {
    let e = (n + 1 < starts.length ? starts[n + 1]! : lines.length) - 1;
    while (e > s && !lines[e]!.trim()) e -= 1;
    const seg = lines.slice(s, e + 1).join("\n");
    if (!seg.trim()) return;
    const defLine = lines.slice(s, e + 1).find((ln) => TOP_LEVEL.test(ln));
    const m = defLine ? SYMBOL.exec(defLine) : null;
    chunks.push({ file, start: s + 1, end: e + 1, symbol: m ? m[1]! : "(module header)", text: seg });
  });
  return chunks;
}

function loadCode(): CodeChunk[] {
  const dir = path.join(DATA, "code");
  return readdirSync(dir).sort().flatMap((f) => chunkCode(f, readFileSync(path.join(dir, f), "utf8")));
}

// =============================================================================
// 2. Text retrieval primitives (identical to the Python version)
// =============================================================================
const STOPWORDS = new Set(
  `a an and are as at be been but by can could do does don for from had has have how i if in into is it its
  just me my of on or our should so than that the their them then there these they this to too up us was we
  were what when where which while who why will with would you your t s anything good`.split(/\s+/).filter(Boolean),
);

function stem(tok: string): string {
  if (tok.length > 4 && tok.endsWith("ies")) return tok.slice(0, -3) + "y";
  if (tok.length > 3 && tok.endsWith("s") && !tok.endsWith("ss")) return tok.slice(0, -1);
  return tok;
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => !STOPWORDS.has(t)).map(stem);
}

/** Identifier-aware: calculateShippingCost -> calculate shipping cost; RESTOCKING_FEE -> restocking fee. */
function codeTokenize(text: string): string[] {
  return tokenize(text.replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/_/g, " "));
}

const CONCEPT_GROUPS: Record<string, string[]> = {
  return: ["return", "send", "back", "exchange"],
  open: ["open", "opened", "unsealed", "used"],
  dislike: ["like", "dislike", "unhappy"],
  charge: ["charge", "charging", "charger", "battery", "power"],
  broken: ["broken", "stopped", "dead", "defect", "defective", "faulty", "failure", "dropout"],
  warranty: ["warranty", "guarantee", "claim", "repair", "replace"],
  water: ["water", "waterproof", "rain", "wet", "sweat", "splash"],
  sport: ["sport", "running", "run", "gym", "workout", "cycling", "hiking", "outdoor"],
  ship: ["ship", "shipping", "delivery", "deliver", "shipped", "courier"],
  abroad: ["international", "abroad", "overseas"],
  price: ["price", "cost", "fee", "surcharge", "usd"],
  headphone: ["headphone", "headset", "earphone"],
  earbud: ["earbud", "bud"],
  quiet: ["noise", "cancelling", "anc", "quiet"],
};
const CONCEPTS = new Map<string, string>();
for (const [canon, toks] of Object.entries(CONCEPT_GROUPS)) for (const t of toks) CONCEPTS.set(t, canon);

/** Concept-mapped tokens; digit tokens collapse to '<num>' - mimics how dense embeddings blur IDs/numbers. */
function embedFeatures(text: string): string[] {
  return tokenize(text).map((t) => CONCEPTS.get(t) ?? (/\d/.test(t) ? "<num>" : t));
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (const b of Buffer.from(s, "utf8")) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

const EMBED_DIM = 256;

/** Deterministic local stand-in for an embedding model (Anthropic has no embeddings endpoint). */
class Embedder {
  readonly idf = new Map<string, number>();
  readonly defaultIdf: number;

  constructor(corpus: string[]) {
    const df = new Map<string, number>();
    for (const text of corpus) for (const f of new Set(embedFeatures(text))) df.set(f, (df.get(f) ?? 0) + 1);
    const n = corpus.length;
    for (const [f, c] of df) this.idf.set(f, Math.log((n + 1) / (c + 1)) + 1.0);
    this.defaultIdf = Math.log(n + 1) + 1.0;
  }

  embed(text: string): number[] {
    const counts = new Map<string, number>();
    for (const f of embedFeatures(text)) counts.set(f, (counts.get(f) ?? 0) + 1);
    const vec = new Array<number>(EMBED_DIM).fill(0);
    for (const [f, tf] of counts) {
      const h = fnv1a("d36:" + f);
      const sign = ((h >>> 16) & 1) === 0 ? 1.0 : -1.0;
      vec[h % EMBED_DIM] += sign * (1.0 + Math.log(tf)) * (this.idf.get(f) ?? this.defaultIdf);
    }
    let sq = 0.0;
    for (const x of vec) sq += x * x;
    const norm = Math.sqrt(sq);
    return norm > 0 ? vec.map((x) => x / norm) : vec;
  }
}

function dot(a: number[], b: number[]): number {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

type Ranked = [string, number][];
const byScore = (a: [string, number], b: [string, number]): number => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0);

class BM25 {
  readonly tokenizer: (t: string) => string[];
  readonly k1: number;
  readonly b: number;
  readonly tf = new Map<string, Map<string, number>>();
  readonly len = new Map<string, number>();
  readonly df = new Map<string, number>();
  readonly avgdl: number;

  constructor(docs: Map<string, string>, tokenizer: (t: string) => string[] = tokenize, k1 = 1.2, b = 0.75) {
    this.tokenizer = tokenizer;
    this.k1 = k1;
    this.b = b;
    for (const [did, text] of docs) {
      const toks = tokenizer(text);
      const tf = new Map<string, number>();
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      this.tf.set(did, tf);
      this.len.set(did, toks.length);
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    }
    let total = 0;
    for (const v of this.len.values()) total += v;
    this.avgdl = total / Math.max(1, this.len.size);
  }

  search(query: string, k: number, allowed: Set<string> | null = null): Ranked {
    const terms = [...new Set(this.tokenizer(query))];
    const n = this.tf.size;
    const scored: Ranked = [];
    for (const [did, tf] of this.tf) {
      if (allowed !== null && !allowed.has(did)) continue;
      let s = 0.0;
      for (const t of terms) {
        const f = tf.get(t) ?? 0;
        if (f === 0) continue;
        const df = this.df.get(t)!;
        const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
        s += (idf * f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * this.len.get(did)!) / this.avgdl));
      }
      if (s > 0) scored.push([did, s]);
    }
    return scored.sort(byScore).slice(0, k);
  }
}

class VectorIndex {
  readonly embedder: Embedder;
  readonly vecs = new Map<string, number[]>();

  constructor(docs: Map<string, string>, embedder: Embedder) {
    this.embedder = embedder;
    for (const [did, text] of docs) this.vecs.set(did, embedder.embed(text));
  }

  search(query: string, k: number, allowed: Set<string> | null = null): Ranked {
    const qv = this.embedder.embed(query);
    const scored: Ranked = [];
    for (const [did, v] of this.vecs) if (allowed === null || allowed.has(did)) scored.push([did, dot(qv, v)]);
    return scored.sort(byScore).slice(0, k);
  }
}

function rrf(rankings: Ranked[], k = 60, top = 10): Ranked {
  const scores = new Map<string, number>();
  for (const ranking of rankings) ranking.forEach(([did], i) => scores.set(did, (scores.get(did) ?? 0.0) + 1.0 / (k + i + 1)));
  return [...scores.entries()].sort(byScore).slice(0, top);
}

const TEXT_DOCS = loadTextDocs();
const TEXT_BY_ID = new Map(TEXT_DOCS.map((d) => [d.doc_id, d] as [string, TextDoc]));
const TEXT_MAP = new Map(TEXT_DOCS.map((d) => [d.doc_id, d.text] as [string, string]));
const TEXT_EMBEDDER = new Embedder(TEXT_DOCS.map((d) => d.text));
const TEXT_BM25 = new BM25(TEXT_MAP);
const TEXT_VEC = new VectorIndex(TEXT_MAP, TEXT_EMBEDDER);
const CODE_CHUNKS = loadCode();
const CODE_BM25 = new BM25(new Map(CODE_CHUNKS.map((c) => [`${c.file}:${c.start}`, c.text] as [string, string])), codeTokenize);

/** Metadata scoping: the router's data_source becomes a pre-filter on doc kind. */
function scope(source: string): Set<string> | null {
  const kind = ({ policies: "policy", catalog: "product" } as Record<string, string>)[source];
  return kind === undefined ? null : new Set(TEXT_DOCS.filter((d) => d.kind === kind).map((d) => d.doc_id));
}

function textSearch(query: string, mode: string, source: string, k = 2): Ranked {
  const allowed = scope(source);
  if (mode === "bm25") return TEXT_BM25.search(query, k, allowed);
  if (mode === "vector") return TEXT_VEC.search(query, k, allowed);
  return rrf([TEXT_BM25.search(query, 20, allowed), TEXT_VEC.search(query, 20, allowed)], 60, k);
}

// =============================================================================
// 3. Structured strategies: key lookup, aggregation plan, attribute filter
// =============================================================================
const ID_RE = /\b(ORD-\d{5}|[A-Z]{2}-\d{3})\b/g;

function keyLookup(query: string): string[] {
  return [...query.matchAll(ID_RE)].map((m) => m[1]!).map((key) => {
    const o = ORDERS_BY_ID.get(key);
    if (o) return `${key}: status=${o.status} placed=${o.order_date} customer=${o.customer_name} item=${o.sku} x${o.quantity}`;
    const p = CATALOG_BY_SKU.get(key);
    return p ? `${key}: ${p.name} $${p.price.toFixed(2)}` : `${key}: not found`;
  });
}

const GROUP_COLUMNS: Record<string, keyof Order> = { region: "region", customer: "customer_name", product: "sku", sku: "sku", status: "status" };
interface AggPlan { group_by: keyof Order; date_from: string; date_to: string; exclude_status: string[] }

/** Turn the question into a VALIDATED plan (allowlisted columns, bound parameters) - never raw SQL. */
function planAggregation(query: string): AggPlan {
  const q = query.toLowerCase();
  const m = /\bby (region|customer|product|sku|status)\b/.exec(q);
  const group = m ? GROUP_COLUMNS[m[1]!]! : "region";
  const qm = /\bq([1-4]) (\d{4})\b/.exec(q);
  let dateFrom = "0000-01-01";
  let dateTo = "9999-12-31";
  if (qm) {
    const startMonth = (Number(qm[1]) - 1) * 3 + 1;
    const endMonth = startMonth + 2;
    const lastDay = ({ 3: 31, 6: 30, 9: 30, 12: 31 } as Record<number, number>)[endMonth]!;
    dateFrom = `${qm[2]}-${String(startMonth).padStart(2, "0")}-01`;
    dateTo = `${qm[2]}-${String(endMonth).padStart(2, "0")}-${lastDay}`;
  }
  return { group_by: group, date_from: dateFrom, date_to: dateTo, exclude_status: ["cancelled", "returned"] };
}

function planToSql(plan: AggPlan): [string, string[]] {
  const col = plan.group_by;
  if (!Object.values(GROUP_COLUMNS).includes(col)) throw new Error(`column not allowed: ${col}`);
  const sql =
    `SELECT ${col} AS grp, SUM(quantity * unit_cents) AS revenue_cents, COUNT(*) AS n_orders ` +
    "FROM orders WHERE order_date BETWEEN ? AND ? AND status NOT IN (?, ?) " +
    "GROUP BY grp ORDER BY revenue_cents DESC, grp ASC";
  return [sql, [plan.date_from, plan.date_to, ...plan.exclude_status]];
}

/** Same plan, executed in memory (Python runs it through sqlite3) - identical semantics and ordering. */
function runAggregation(plan: AggPlan): [string, number, number][] {
  const groups = new Map<string, [number, number]>();
  for (const o of ORDERS) {
    if (o.order_date < plan.date_from || o.order_date > plan.date_to || plan.exclude_status.includes(o.status)) continue;
    const key = String(o[plan.group_by]);
    const [cents, n] = groups.get(key) ?? [0, 0];
    groups.set(key, [cents + o.quantity * o.unit_cents, n + 1]);
  }
  return [...groups.entries()]
    .map(([g, [c, n]]) => [g, c, n] as [string, number, number])
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}

interface FilterPlan { category?: string; wireless?: boolean; waterproof?: boolean; noise_cancelling?: boolean; max_price?: number; min_battery_h?: number }

function planFilter(query: string): FilterPlan {
  const q = query.toLowerCase();
  const plan: FilterPlan = {};
  for (const cat of ["headphones", "earbuds", "speakers", "accessories"]) {
    if (q.includes(cat) || q.includes(cat.slice(0, -1))) {
      plan.category = cat;
      break;
    }
  }
  if (q.includes("wireless")) plan.wireless = true;
  if (q.includes("waterproof")) plan.waterproof = true;
  if (q.includes("noise cancelling") || q.split(/\s+/).includes("anc")) plan.noise_cancelling = true;
  let m = /(?:under|below|less than) \$?(\d+(?:\.\d+)?)/.exec(q);
  if (m) plan.max_price = Number(m[1]);
  m = /at least (\d+) hours/.exec(q);
  if (m) plan.min_battery_h = Number(m[1]);
  return plan;
}

function runFilter(plan: FilterPlan): Product[] {
  const ok = (p: Product): boolean => {
    const a = p.attributes;
    if (plan.category !== undefined && p.category !== plan.category) return false;
    if (plan.wireless && !a.wireless) return false;
    if (plan.waterproof && !a.waterproof) return false;
    if (plan.noise_cancelling && !a.noise_cancelling) return false;
    if (plan.max_price !== undefined && p.price > plan.max_price) return false;
    if (plan.min_battery_h !== undefined && (a.battery_h === null || a.battery_h < plan.min_battery_h)) return false;
    return true;
  };
  return CATALOG.filter(ok).sort((a, b) => a.price - b.price || (a.sku < b.sku ? -1 : a.sku > b.sku ? 1 : 0));
}

// =============================================================================
// 4. Graph traversal
// =============================================================================
class Graph {
  readonly names = new Map<string, string>();
  readonly out = new Map<string, string[]>();
  readonly inc = new Map<string, string[]>();
  edgeCount = 0;

  node(nid: string, name: string): void {
    if (!this.names.has(nid)) this.names.set(nid, name);
  }

  edge(src: string, rel: string, dst: string): void {
    const key = `${src}|${rel}`;
    if (!this.out.has(key)) this.out.set(key, []);
    const lst = this.out.get(key)!;
    if (lst.includes(dst)) return;
    lst.push(dst);
    const ikey = `${dst}|${rel}`;
    if (!this.inc.has(ikey)) this.inc.set(ikey, []);
    this.inc.get(ikey)!.push(src);
    this.edgeCount += 1;
  }

  neighbors(nid: string, rel: string, direction: "in" | "out"): string[] {
    return (direction === "out" ? this.out : this.inc).get(`${nid}|${rel}`) ?? [];
  }

  /** Follow a typed path pattern hop by hop, keeping every path (so results are explainable). */
  traverse(start: string, steps: [string, "in" | "out"][]): string[][] {
    let paths: string[][] = [[start]];
    for (const [rel, direction] of steps) {
      paths = paths.flatMap((p) => this.neighbors(p[p.length - 1]!, rel, direction).map((nb) => [...p, nb]));
    }
    return paths;
  }
}

function buildGraph(): Graph {
  const g = new Graph();
  for (const s of GRAPH_DATA.suppliers) {
    g.node(`supplier:${s.id}`, s.name);
    g.node(`country:${s.country}`, s.country);
    g.edge(`supplier:${s.id}`, "LOCATED_IN", `country:${s.country}`);
  }
  for (const p of CATALOG) {
    g.node(`product:${p.sku}`, p.sku);
    g.edge(`product:${p.sku}`, "SUPPLIED_BY", `supplier:${p.supplier_id}`);
  }
  for (const o of ORDERS) {
    if (o.status === "delivered" || o.status === "in_transit") {
      g.node(`customer:${o.customer_id}`, o.customer_name);
      g.node(`product:${o.sku}`, o.sku);
      g.edge(`customer:${o.customer_id}`, "PURCHASED", `product:${o.sku}`);
    }
  }
  for (const e of GRAPH_DATA.edges) g.edge(e.src, e.rel, e.dst);
  return g;
}

const GRAPH = buildGraph();

function graphQuery(query: string): [string, string[]] {
  const m = /suppliers? (?:based|located) in ([A-Za-z]+)/.exec(query);
  const country = m ? m[1]! : "Vietnam";
  const paths = GRAPH.traverse(`country:${country}`, [["LOCATED_IN", "in"], ["SUPPLIED_BY", "in"], ["PURCHASED", "in"]]);
  const via = new Map<string, string[]>();
  const suppliers: string[] = [];
  for (const [, supplier, product, customer] of paths) {
    const sName = GRAPH.names.get(supplier!)!;
    if (!suppliers.includes(sName)) suppliers.push(sName);
    const cName = GRAPH.names.get(customer!)!;
    if (!via.has(cName)) via.set(cName, []);
    const pName = GRAPH.names.get(product!)!;
    if (!via.get(cName)!.includes(pName)) via.get(cName)!.push(pName);
  }
  const lines = [...via.keys()].sort().map((name) => `${name.padEnd(15)} via ${via.get(name)!.join(", ")}`);
  const patternText = "(country)<-LOCATED_IN-(supplier)<-SUPPLIED_BY-(product)<-PURCHASED-(customer)";
  return [`${patternText}  [${country}: suppliers ${suppliers.join(", ")}; ${paths.length} paths]`, lines];
}

// =============================================================================
// 5. Agentic iterative retrieval (tool use) for cross-source multi-hop questions
// =============================================================================
const AGENT_SYSTEM =
  "You are Acme Audio's support analyst agent. Answer questions that need facts from several systems " +
  "by calling tools step by step: orders database, product catalog, and policy search. Verify each fact " +
  `with a tool before using it. Today is ${AS_OF}. Answer in two or three sentences and name the order.`;
const AGENT_TOOLS: Anthropic.Tool[] = [
  {
    name: "query_orders",
    description: "Look up orders in the order database. Use it for a customer's orders, order dates or status. " +
      "Filter by customer_name and/or sku. Returns a JSON list of orders.",
    input_schema: {
      type: "object",
      properties: {
        customer_name: { type: "string", description: "Full customer name, e.g. 'Dana Lee'" },
        sku: { type: "string", description: "Product SKU, e.g. 'HP-200'" },
      },
    },
  },
  {
    name: "get_product",
    description: "Get catalog details for ONE SKU: name, category, price, warranty_months and attributes. " +
      "Call it once per SKU you need (parallel calls are fine).",
    input_schema: { type: "object", properties: { sku: { type: "string" } }, required: ["sku"] },
  },
  {
    name: "search_policies",
    description: "Search customer policy text (returns, warranty, shipping, price match). Returns the best sections.",
    input_schema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
];

/** Execute one tool call. Failures come back as [message, isError=true] - never throw into the loop. */
function runAgentTool(name: string, args: Record<string, string>): [string, boolean] {
  if (name === "query_orders") {
    const rows = ORDERS.filter((o) =>
      (!args.customer_name || o.customer_name.toLowerCase() === args.customer_name.toLowerCase()) && (!args.sku || o.sku === args.sku));
    return [pyJson(rows.map((o) => ({ order_id: o.order_id, order_date: o.order_date, sku: o.sku, quantity: o.quantity, status: o.status }))), false];
  }
  if (name === "get_product") {
    const p = CATALOG_BY_SKU.get(args.sku ?? "");
    if (!p) return [`SKU ${args.sku} not found in the catalog (discontinued or invalid).`, true];
    return [pyJson({ sku: p.sku, name: p.name, category: p.category, price: p.price, warranty_months: p.warranty_months }), false];
  }
  if (name === "search_policies") {
    const hits = textSearch(args.query ?? "", "hybrid", "policies", 1);
    return [hits.map(([d]) => `[${TEXT_BY_ID.get(d)!.title}] ${TEXT_BY_ID.get(d)!.text}`).join("\n"), false];
  }
  return [`unknown tool ${name}`, true];
}

function addMonths(iso: string, months: number): string {
  const [y, m, d] = iso.split("-").map(Number) as [number, number, number];
  const idx = m - 1 + months;
  const y2 = y + Math.floor(idx / 12);
  const m2 = (idx % 12) + 1;
  const leap = y2 % 4 === 0 && (y2 % 100 !== 0 || y2 % 400 === 0);
  const dim = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return `${String(y2).padStart(4, "0")}-${String(m2).padStart(2, "0")}-${String(Math.min(d, dim[m2 - 1]!)).padStart(2, "0")}`;
}

/** Manual tool loop: check stop_reason every turn, return ALL tool results of a turn in ONE user message. */
async function agenticLoop(client: Anthropic, question: string, maxTurns = 6): Promise<string> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: question }];
  for (let turn = 1; turn <= maxTurns; turn++) {
    const resp = await client.messages.create({ model: MODEL_OPUS, max_tokens: 2048, system: AGENT_SYSTEM, tools: AGENT_TOOLS, messages });
    const calls = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
    if (resp.stop_reason !== "tool_use" || calls.length === 0) {
      console.log(`    turn ${turn}: stop_reason=${resp.stop_reason} -> final answer`);
      return textOf(resp).trim();
    }
    messages.push({ role: "assistant", content: resp.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const c of calls) {
      const [out, isError] = runAgentTool(c.name, c.input as Record<string, string>);
      const shown = out.replace(/\n/g, " ");
      console.log(`    turn ${turn}: ${c.name}(${pyJson(c.input, true)}) -> ${isError ? "ERROR " : ""}${shown.slice(0, 70)}${shown.length > 70 ? "..." : ""}`);
      results.push({ type: "tool_result", tool_use_id: c.id, content: out, ...(isError ? { is_error: true } : {}) });
    }
    messages.push({ role: "user", content: results });
  }
  return "(stopped: max turns reached)";
}

// =============================================================================
// 6. Router: regex pre-router -> Haiku structured output -> fallback rules
// =============================================================================
const PATTERNS = ["exact_lookup", "aggregation", "filtered_search", "keyword", "conceptual", "mixed", "multi_hop", "code_search"] as const;
const SOURCES = ["orders", "catalog", "policies", "code", "graph", "multiple", "all"] as const;
const RouteDecision = z.object({
  pattern: z.enum(PATTERNS),
  data_source: z.enum(SOURCES),
  confidence: z.number(),
  reason: z.string(),
});

const ROUTER_SYSTEM = `You are a query router for Acme Audio's assistant. Classify the user's query; do not answer it.

pattern:
- exact_lookup: fetch one record by its identifier (order ID, SKU)
- aggregation: totals, counts, averages, rankings over many records
- filtered_search: find items matching attribute constraints (price, battery, category, flags)
- keyword: a specific named term or phrase that must appear verbatim in documents
- conceptual: a natural-language question about meaning or policy, paraphrase-heavy
- mixed: conceptual question that also contains specific key terms
- multi_hop: needs facts chained across relations or across several systems
- code_search: where/how something is implemented in source code

data_source: orders (table), catalog (product JSON), policies (prose), code, graph (supplier/customer
relations), multiple (several systems), all (unsure - search everything).
Give a confidence between 0 and 1 and a short reason.`;

/** What the offline mock 'Haiku' answers. Live, the real model classifies from ROUTER_SYSTEM. */
function haikuClassify(query: string): Json {
  const q = query.toLowerCase();
  if (/\bin the code\b|\bfunction\b|\bcodebase\b|\bimplemented\b/.test(q))
    return { pattern: "code_search", data_source: "code", confidence: 0.95, reason: "asks where logic lives in source code" };
  if (/\b(total|revenue|average|sum|how many)\b/.test(q))
    return { pattern: "aggregation", data_source: "orders", confidence: 0.94, reason: "aggregate over many order rows" };
  if (/\bwhich customers\b.*\bsuppliers?\b/.test(q))
    return { pattern: "multi_hop", data_source: "graph", confidence: 0.91, reason: "customer-product-supplier-country chain" };
  if (/\b[A-Z][a-z]+ [A-Z][a-z]+'s\b/.test(query) && /warranty|return/.test(q))
    return { pattern: "multi_hop", data_source: "multiple", confidence: 0.88, reason: "needs orders + catalog + policy" };
  if (/\$\d+|\bunder \d|\bat least\b/.test(q))
    return { pattern: "filtered_search", data_source: "catalog", confidence: 0.93, reason: "numeric attribute constraints" };
  if (/^what is the [a-z ]+ (fee|rate|window|threshold)\??$/.test(q))
    return { pattern: "keyword", data_source: "policies", confidence: 0.86, reason: "asks for a named policy term" };
  if (q.includes("warranty") && /stopped|broken|dead|not working/.test(q))
    return { pattern: "mixed", data_source: "policies", confidence: 0.83, reason: "policy question with specific symptoms" };
  if (/^(can|may|do|does|how) i\b|\bcan i\b/.test(q))
    return { pattern: "conceptual", data_source: "policies", confidence: 0.9, reason: "paraphrased policy question" };
  return { pattern: "conceptual", data_source: "catalog", confidence: 0.41, reason: "vague; products or policy unclear" };
}

const FALLBACK_RULES: [RegExp, string, string][] = [
  [/\b(total|sum|average|avg|count|how many|revenue)\b/, "aggregation", "orders"],
  [/\b(under|below|less than|at least|more than)\b.*\d/, "filtered_search", "catalog"],
  [/\b(code|function|implemented|module)\b/, "code_search", "code"],
  [/\bwhich (customers|suppliers|products)\b/, "multi_hop", "graph"],
];

interface Route { pattern: string; source: string; confidence: number; via: string; reason: string }
const ORDER_ID_ONLY = /\bORD-\d{5}\b/;

class Router {
  readonly client: Anthropic;
  llmCalls = 0;
  llmInputTokens = 0;

  constructor(client: Anthropic) {
    this.client = client;
  }

  async route(query: string): Promise<Route> {
    // 1) Cheap deterministic pre-router: unambiguous identifiers never need an LLM.
    if (ORDER_ID_ONLY.test(query) && query.split(/\s+/).length <= 10) {
      return { pattern: "exact_lookup", source: "orders", confidence: 1.0, via: "rule:id-regex", reason: "order ID present" };
    }
    // 2) LLM classifier with structured output (schema-validated by the SDK's zod helper).
    let d: z.infer<typeof RouteDecision>;
    try {
      const msg = await this.client.messages.parse({
        model: MODEL_HAIKU,
        max_tokens: 1024,
        output_config: { effort: "low", format: zodOutputFormat(RouteDecision) },
        system: ROUTER_SYSTEM,
        messages: [{ role: "user", content: query }],
      });
      this.llmCalls += 1;
      this.llmInputTokens += msg.usage.input_tokens;
      if (!msg.parsed_output) throw new Error("no parsed output");
      d = msg.parsed_output;
    } catch (exc) {
      if (!(exc instanceof Error)) throw exc;
      return Router.fallback(query, `fallback(error: ${exc.constructor.name})`);
    }
    // 3) Low confidence -> deterministic rules (or ask a clarifying question in a chat UI).
    if (d.confidence < 0.6) return Router.fallback(query, `fallback(llm confidence ${d.confidence.toFixed(2)})`);
    return { pattern: d.pattern, source: d.data_source, confidence: d.confidence, via: "llm:haiku", reason: d.reason };
  }

  static fallback(query: string, via: string): Route {
    for (const [rx, pattern, source] of FALLBACK_RULES) {
      if (rx.test(query.toLowerCase())) return { pattern, source, confidence: 0.5, via, reason: `rule ${rx.source}` };
    }
    return { pattern: "mixed", source: "all", confidence: 0.5, via, reason: "default: hybrid over all text" };
  }
}

function pickStrategy(r: Route): string {
  const table: Record<string, string> = {
    exact_lookup: "key_lookup", aggregation: "sql_aggregation", filtered_search: "attribute_filter", keyword: "bm25",
    conceptual: "vector", mixed: "hybrid", code_search: "code_bm25",
  };
  if (r.pattern === "multi_hop") return r.source === "graph" ? "graph_traversal" : "agentic_loop";
  return table[r.pattern] ?? "hybrid";
}

// =============================================================================
// 7. Offline responder (router classifications + agent turns)
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

function responder(req: MockRequest): ResponderResult {
  if (req.systemText.startsWith("You are a query router")) return JSON.stringify(haikuClassify(req.lastUserText));
  if (req.systemText.startsWith("You are Acme Audio's support analyst")) {
    const question = String(req.messages[0]!.content);
    const history = toolHistory(req);
    const called = history.map((h) => h[0]);
    if (!called.includes("query_orders")) {
      const m = /([A-Z][a-z]+ [A-Z][a-z]+)'s/.exec(question);
      return [textBlock("I'll start with the customer's order history."), toolUseBlock("query_orders", { customer_name: m ? m[1] : "" })];
    }
    const orders = JSON.parse(history.find((h) => h[0] === "query_orders")![2]) as Json[];
    if (!called.includes("get_product")) {
      const skus = [...new Set(orders.map((o) => String(o.sku)))];
      return [textBlock("Checking the category and warranty of each product ordered."), ...skus.map((s) => toolUseBlock("get_product", { sku: s }))];
    }
    if (!called.includes("search_policies")) return [toolUseBlock("search_policies", { query: "warranty period start date proof of purchase" })];
    const products = new Map<string, Json>();
    for (const h of history) if (h[0] === "get_product" && !h[3]) products.set(JSON.parse(h[2]).sku, JSON.parse(h[2]));
    const missing = history.filter((h) => h[0] === "get_product" && h[3]).map((h) => String(h[1].sku));
    const heads = orders.filter((o) => products.get(o.sku)?.category === "headphones");
    const latest = heads.reduce((best, o) => (o.order_date > best.order_date ? o : best));
    const p = products.get(latest.sku)!;
    const end = addMonths(latest.order_date, p.warranty_months);
    const status = AS_OF <= end ? "is still under warranty" : "is no longer under warranty";
    const newer = orders.filter((o) => o.order_date > latest.order_date && products.has(o.sku));
    const last = newer[newer.length - 1];
    const note = last ? ` Her newer order ${last.order_id} is a ${String(products.get(last.sku)!.category).slice(0, -1)}, not headphones.` : "";
    const miss = missing.length ? ` (${missing.join(", ")} is no longer in the catalog, so it was ignored.)` : "";
    const who = /([A-Z][a-z]+ [A-Z][a-z]+)'s/.exec(question);
    return (
      `${who ? who[1] : "The customer"}'s most recent headphone order is ${latest.order_id} (${p.name}, ${p.sku}) placed on ` +
      `${latest.order_date}. It has a ${p.warranty_months}-month warranty starting on the order date, ` +
      `so coverage runs until ${end} and it ${status} as of ${AS_OF}; a claim needs the order number as ` +
      `proof of purchase.${note}${miss}`
    );
  }
  return "OK";
}

// =============================================================================
// Demo
// =============================================================================
const STRATEGY_WHY: Record<string, string> = {
  key_lookup: "an ID is a primary key: O(1) exact fetch, no ranking, no LLM",
  sql_aggregation: "totals need ALL matching rows - top-k retrieval can never sum correctly",
  attribute_filter: "numeric/boolean constraints are predicates, not similarity",
  bm25: "a named term must match verbatim; lexical scoring is precise and explainable",
  vector: "paraphrase with no shared keywords: semantic similarity",
  hybrid: "meaning + specific terms: fuse lexical and semantic ranks (RRF)",
  code_bm25: "identifiers split into words; chunks are whole functions",
  graph_traversal: "explicit relations: follow typed edges hop by hop",
  agentic_loop: "facts in different systems + reasoning: let Claude iterate with tools",
};

function printMatrix(): void {
  const rows = [
    ["Structured table", "key lookup", "SQL group-by", "SQL WHERE", "-", "-", "SQL joins / agent"],
    ["Semi-structured JSON", "key lookup", "SQL/JSON agg", "metadata filter", "BM25 on fields", "vector on desc", "graph / agent"],
    ["Unstructured prose", "-", "-", "metadata filter", "BM25", "vector / hybrid", "agentic multi-query"],
    ["Source code", "symbol index", "-", "path/lang filter", "code BM25", "code embeddings", "call-graph walk"],
    ["Knowledge graph", "node lookup", "graph agg", "property filter", "-", "-", "graph traversal"],
  ];
  const head = ["data shape \\ pattern", "exact ID", "aggregation", "attr filter", "keyword", "conceptual", "multi-hop"];
  const widths = [22, 14, 14, 18, 16, 17, 20];
  const fmt = (cells: string[]) => ("  " + cells.map((c, i) => c.padEnd(widths[i]!)).join("")).trimEnd();
  console.log(fmt(head));
  console.log("  " + "-".repeat(widths.reduce((a, b) => a + b, 0) - 2));
  for (const r of rows) console.log(fmt(r));
}

async function runQuery(client: Anthropic, router: Router, qid: string, query: string): Promise<Route> {
  console.log(`\n[${qid}] "${query}"`);
  const r = await router.route(query);
  const strategy = pickStrategy(r);
  console.log(`  route   : ${r.pattern} / ${r.source}  via ${r.via} (conf ${r.confidence.toFixed(2)}) -> ${strategy}`);
  console.log(`  why     : ${STRATEGY_WHY[strategy]}`);
  if (strategy === "key_lookup") {
    for (const line of keyLookup(query)) console.log(`  result  : ${line}`);
  } else if (strategy === "sql_aggregation") {
    const plan = planAggregation(query);
    const [sql, params] = planToSql(plan);
    console.log(`  plan    : ${pyJson(plan, true)}`);
    console.log(`  sql     : ${sql}`);
    console.log(`  params  : [${params.map((p) => `'${p}'`).join(", ")}]`);
    for (const [grp, cents, n] of runAggregation(plan)) {
      console.log(`  result  : ${grp.padEnd(14)} $${(cents / 100).toFixed(2).padStart(9)}  (${n} orders)`);
    }
  } else if (strategy === "attribute_filter") {
    const plan = planFilter(query);
    console.log(`  plan    : ${pyJson(plan, true)}`);
    for (const p of runFilter(plan)) {
      console.log(`  result  : ${p.sku} ${p.name.padEnd(30)} $${p.price.toFixed(2).padStart(7)}  battery=${p.attributes.battery_h}h`);
    }
  } else if (strategy === "bm25" || strategy === "vector" || strategy === "hybrid") {
    const scopeNote = r.source === "policies" || r.source === "catalog" ? r.source : "all text";
    console.log(`  scope   : ${scopeNote} (metadata pre-filter on doc kind)`);
    for (const [did, s] of textSearch(query, strategy, r.source)) {
      console.log(`  result  : ${did.padEnd(26)} ${s.toFixed(3)}  "${TEXT_BY_ID.get(did)!.text.slice(0, 62)}..."`);
    }
  } else if (strategy === "code_bm25") {
    for (const [key, s] of CODE_BM25.search(query, 2)) {
      const c = CODE_CHUNKS.find((x) => `${x.file}:${x.start}` === key)!;
      console.log(`  result  : ${c.file}:${c.start}-${c.end}  ${c.symbol}  (${s.toFixed(3)})`);
    }
    console.log(`  tokens  : 'calculateShippingCost' -> [${codeTokenize("calculateShippingCost").map((t) => `'${t}'`).join(", ")}]`);
  } else if (strategy === "graph_traversal") {
    const [header, lines] = graphQuery(query);
    console.log(`  pattern : ${header}`);
    for (const line of lines) console.log(`  result  : ${line}`);
  } else if (strategy === "agentic_loop") {
    console.log("  trace   :");
    console.log(`  answer  : ${await agenticLoop(client, query)}`);
  }
  return r;
}

banner("D3.6 - Retrieval Strategies by Data Shape and Query Pattern");
const client = getClient(responder);

section("1. Data shapes");
console.log(`  structured table     orders.csv     ${ORDERS.length} rows (order_id, date, customer, region, sku, qty, price, status)`);
console.log(`  semi-structured JSON catalog.json   ${CATALOG.length} products with nested attributes`);
console.log(`  unstructured prose   policies.md    ${TEXT_DOCS.filter((d) => d.kind === "policy").length} sections (+ product descriptions as text docs)`);
console.log(`  source code          data/code      ${new Set(CODE_CHUNKS.map((c) => c.file)).size} files -> ${CODE_CHUNKS.length} definition-level chunks`);
console.log(`  knowledge graph      graph.json     ${GRAPH.names.size} nodes, ${GRAPH.edgeCount} typed edges (joined from catalog + orders)`);

section("2. Decision matrix: data shape x query pattern -> strategy");
printMatrix();

section("3. Route and retrieve");
console.log("Router = regex pre-router (IDs) -> Claude Haiku with structured output -> fallback rules.");
const router = new Router(client);
const routes: [QueryItem, Route][] = [];
for (const q of QUERIES.queries) routes.push([q, await runQuery(client, router, q.id, q.query)]);

section("4. Anti-pattern: one vector index for everything");
console.log("Serialize every order row, product and policy into text and run pure vector search:");
const naiveDocs = new Map<string, string>(
  ORDERS.map((o) => [o.order_id, `Order ${o.order_id} placed ${o.order_date} by ${o.customer_name} in ${o.region}: ` +
    `${o.quantity} x ${o.sku} at USD ${o.unit_price}, status ${o.status}.`] as [string, string]),
);
for (const d of TEXT_DOCS) naiveDocs.set(d.doc_id, d.text);
const naive = new VectorIndex(naiveDocs, new Embedder([...naiveDocs.values()]));
const [nq1, nq2, nq3] = [0, 1, 2].map((i) => QUERIES.queries[i]!.query) as [string, string, string];
const top1 = naive.search(nq1, 3).map(([d]) => d);
console.log(`  ${nq1}\n    top-3: ${top1.join(", ")} -> ${top1[0] === "ORD-10447" ? "correct" : "WRONG record"}: embeddings blur IDs and numbers`);
const q3Orders = ORDERS.filter((o) => o.order_date >= "2026-07-01" && o.order_date <= "2026-09-30" && o.status !== "cancelled" && o.status !== "returned");
const q3Ids = new Set(q3Orders.map((o) => o.order_id));
const got = naive.search(nq2, 5).filter(([d]) => q3Ids.has(d)).length;
console.log(`  ${nq2}\n    top-5 holds ${got} of the ${q3Orders.length} Q3 orders -> any total computed from it is wrong`);
const good = new Set(runFilter(planFilter(nq3)).map((p) => `product:${p.sku}`));
const top3 = naive.search(nq3, 5).map(([d]) => d).filter((d) => d.startsWith("product:"));
console.log(`  ${nq3}\n    products in top-5: ${top3.join(", ")} -> ${top3.filter((d) => !good.has(d)).length} violate the price/battery constraints`);

section("5. Router report");
console.log(`  ${"id".padEnd(5)}${"expected".padEnd(17)}${"routed".padEnd(17)}${"via".padEnd(32)}ok`);
let correct = 0;
for (const [q, r] of routes) {
  const ok = q.expected.includes(r.pattern);
  if (ok) correct += 1;
  console.log(`  ${q.id.padEnd(5)}${q.expected.join("/").padEnd(17)}${r.pattern.padEnd(17)}${r.via.padEnd(32)}${ok ? "yes" : "NO"}`);
}
console.log(
  `  accuracy ${correct}/${routes.length} | Haiku router calls: ${router.llmCalls} ` +
    `(avg ${Math.floor(router.llmInputTokens / Math.max(1, router.llmCalls))} input tokens) | ` +
    `regex short-circuits: ${routes.filter(([, r]) => r.via.startsWith("rule")).length} | ` +
    `fallbacks: ${routes.filter(([, r]) => r.via.startsWith("fallback")).length}`,
);
console.log("  Routing on Haiku costs a fraction of a cent per query and keeps the expensive model for the");
console.log("  one question that truly needs agentic reasoning (Q9). Track router accuracy as its own eval.");

section("Takeaways");
console.log("  * Match the retriever to the data shape AND the query pattern; no single index serves all.");
console.log("  * Aggregations and filters belong to databases (validated plans, bound parameters), not to RAG.");
console.log("  * Regex for unambiguous IDs, a small model for fuzzy intent, deterministic fallback for doubt.");
console.log("  * Graphs answer explicit relationship chains; agentic loops handle cross-system reasoning.");
