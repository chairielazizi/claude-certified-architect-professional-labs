/**
 * D5.3 - Human-in-the-Loop Validation (TypeScript)
 *
 * Equivalent to python/main.py:
 *   (a) confidence/risk-based routing, (b) approval gates in an agent tool loop,
 *   (c) stratified QA sampling + reviewer agreement, (d) feedback -> corrections
 *   dataset, (e) operational metrics (automation rate, review load, SLA, latency).
 *
 * Only the LLM decisions go through the shared client; routing/QA/metrics are real,
 * deterministic code.
 *
 * Run (from repo root):
 *   node "D5.3 - Human-in-the-Loop Validation/typescript/main.ts"
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type Anthropic from "@anthropic-ai/sdk";
import {
  MODEL_OPUS,
  MockRequest,
  banner,
  getClient,
  section,
  type Json,
  type ResponderResult,
} from "../../shared/typescript/claudeClient.ts";

const DATA = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "data");
const REVIEW_LATENCIES = [1.5, 2.0, 3.0, 3.5]; // deterministic time-in-queue (hours)

// Deterministic PRNG (mulberry32) for stable stratified sampling.
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const SAMPLE_RNG = mulberry32(13); // seed chosen so the sample includes the flagged item

// ===========================================================================
// Types & config
// ===========================================================================
interface Item {
  id: string;
  user: string;
  amount: number;
  category: string;
  support: number;
  fraud_signal: boolean;
  text: string;
}
interface Config {
  auto_amount_limit: number;
  support_min: number;
  confidence_min: number;
  review_sla_hours: number;
}
interface ModelDecision {
  decision: string;
  confidence: number;
  rationale: string;
}
interface Routed {
  itemId: string;
  route: string;
  modelDecision: string;
  confidence: number;
  triggers: string[];
  slaHours: number;
}

const DECISION_SCHEMA = {
  type: "object",
  properties: {
    decision: { type: "string", enum: ["approve", "deny"] },
    confidence: { type: "number" },
    rationale: { type: "string" },
  },
  required: ["decision", "confidence", "rationale"],
  additionalProperties: false,
};

// ===========================================================================
// (a) Confidence/risk-based routing
// ===========================================================================
async function modelDecide(client: Anthropic, item: Item): Promise<ModelDecision> {
  const payload = JSON.stringify({ id: item.id, amount: item.amount, category: item.category, support: item.support, text: item.text });
  const msg = await client.messages.create({
    model: MODEL_OPUS,
    max_tokens: 256,
    output_config: { effort: "low", format: { type: "json_schema", schema: DECISION_SCHEMA } },
    system:
      "You are a refund adjudicator. Return a decision, a calibrated confidence in [0,1], " +
      "and a one-line rationale. Base it only on the provided fields.",
    messages: [{ role: "user", content: `Adjudicate this refund:\n${payload}` }],
  });
  const text = msg.content.filter((b): b is Anthropic.TextBlock => b.type === "text").map((b) => b.text).join("");
  return JSON.parse(text) as ModelDecision;
}

function route(item: Item, decision: ModelDecision, cfg: Config, regulated: string[]): Routed {
  const base: Routed = { itemId: item.id, route: "", modelDecision: decision.decision, confidence: decision.confidence, triggers: [], slaHours: 0 };
  if (item.fraud_signal) return { ...base, route: "REJECT", triggers: ["fraud_signal"] };
  const triggers: string[] = [];
  if (item.amount > cfg.auto_amount_limit) triggers.push(`amount>${cfg.auto_amount_limit}`);
  if (regulated.includes(item.category)) triggers.push(`regulated:${item.category}`);
  if (item.support < cfg.support_min) triggers.push("low_retrieval_support");
  if (decision.confidence < cfg.confidence_min) triggers.push("low_confidence");
  return { ...base, route: triggers.length ? "HUMAN_REVIEW" : "AUTO_APPROVE", triggers };
}

// ===========================================================================
// (b) Approval gates in an agent tool loop
// ===========================================================================
const AGENT_TOOLS = [
  { name: "lookup_order", description: "look up an order", input_schema: { type: "object" as const, properties: { order_id: { type: "string" } } } },
  { name: "issue_refund", description: "issue a refund", input_schema: { type: "object" as const, properties: { order_id: { type: "string" }, amount: { type: "number" } } } },
  { name: "send_external_email", description: "email a customer", input_schema: { type: "object" as const, properties: { to: { type: "string" }, body: { type: "string" } } } },
];
const REFUND_APPROVAL_LIMIT = 200;

function isHighRisk(name: string, args: Json): boolean {
  if (name === "send_external_email") return true;
  if (name === "issue_refund" && Number(args.amount ?? 0) > REFUND_APPROVAL_LIMIT) return true;
  return false;
}

interface HumanDecision {
  action: string;
  editedArgs?: Json;
  note?: string;
}

async function runAgentWithGates(client: Anthropic, scripted: HumanDecision[], corrections: Json[]): Promise<string[]> {
  const messages: Anthropic.MessageParam[] = [{ role: "user", content: "AGENT: refund order ORD-5 for $250 and email the customer a confirmation." }];
  const transcript: string[] = [];
  let si = 0;
  for (let step = 0; step < 6; step++) {
    const resp = await client.messages.create({ model: MODEL_OPUS, max_tokens: 512, tools: AGENT_TOOLS, messages });
    if (resp.stop_reason !== "tool_use") {
      transcript.push("agent: " + resp.content.filter((b): b is Anthropic.TextBlock => b.type === "text").map((b) => b.text).join(""));
      break;
    }
    messages.push({ role: "assistant", content: resp.content });
    const results: Anthropic.ToolResultBlockParam[] = [];
    for (const tu of resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use")) {
      let args = { ...(tu.input as Json) };
      if (isHighRisk(tu.name, args)) {
        const hd = scripted[si++]!;
        if (hd.action === "deny") {
          transcript.push(`GATE  ${tu.name}(${JSON.stringify(args)}) -> human DENY (${hd.note})`);
          corrections.push({ source: "approval_gate", tool: tu.name, from: args, action: "deny", note: hd.note });
          results.push({ type: "tool_result", tool_use_id: tu.id, content: "Denied by human reviewer.", is_error: true });
          continue;
        }
        if (hd.action === "edit" && hd.editedArgs) {
          corrections.push({ source: "approval_gate", tool: tu.name, from: args, to: hd.editedArgs, action: "edit", note: hd.note });
          args = { ...args, ...hd.editedArgs };
          transcript.push(`GATE  ${tu.name} -> human EDIT to ${JSON.stringify(args)} (${hd.note})`);
        } else {
          transcript.push(`GATE  ${tu.name}(${JSON.stringify(args)}) -> human APPROVE`);
        }
        results.push({ type: "tool_result", tool_use_id: tu.id, content: `executed ${tu.name} with ${JSON.stringify(args)}` });
      } else {
        transcript.push(`auto  ${tu.name}(${JSON.stringify(args)}) -> executed (low risk)`);
        results.push({ type: "tool_result", tool_use_id: tu.id, content: `executed ${tu.name} with ${JSON.stringify(args)}` });
      }
    }
    messages.push({ role: "user", content: results });
  }
  return transcript;
}

// ===========================================================================
// (c) QA sampling + reviewer agreement
// ===========================================================================
function sampleK<T>(arr: T[], k: number, rng: () => number): T[] {
  const a = [...arr];
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(rng() * (a.length - i));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
  return a.slice(0, k);
}

function stratifiedSample(auto: Item[], frac: number): Item[] {
  const byCat = new Map<string, Item[]>();
  for (const it of auto) (byCat.get(it.category) ?? byCat.set(it.category, []).get(it.category)!).push(it);
  const sample: Item[] = [];
  for (const items of byCat.values()) {
    const k = Math.max(1, Math.round(items.length * frac));
    sample.push(...sampleK(items, Math.min(k, items.length), SAMPLE_RNG));
  }
  return sample.sort((a, b) => a.id.localeCompare(b.id));
}

function qaReviewer(item: Item): string {
  // Deterministic: reviewer judges the duplicate-charge item's evidence insufficient.
  return item.id === "C08" ? "deny" : "approve";
}

// ===========================================================================
// Offline responder
// ===========================================================================
function buildResponder(): (req: MockRequest) => ResponderResult {
  return (req: MockRequest): ResponderResult => {
    const q = req.lastUserText;
    const names = req.toolNames;

    if (q.includes("AGENT:") || req.toolResults.length) {
      const didRefund = req.allText.includes("issue_refund");
      const didEmail = req.allText.includes("send_external_email");
      if (!didRefund && names.length) return [{ type: "tool_use", id: "t_refund", name: "issue_refund", input: { order_id: "ORD-5", amount: 250 } }];
      if (didRefund && !didEmail && names.length) return [{ type: "tool_use", id: "t_email", name: "send_external_email", input: { to: "jane.doe@example.com", body: "Your refund is processed." } }];
      return "Done: refund issued (as approved/edited) and confirmation email sent.";
    }

    const parts = q.split("Adjudicate this refund:\n");
    if (parts.length < 2) return JSON.stringify({ decision: "deny", confidence: 0.5, rationale: "unparseable" });
    const item = JSON.parse(parts[1]!) as Item;
    if (item.support < 0.5) return JSON.stringify({ decision: "deny", confidence: 0.55, rationale: "weak evidence / no order record" });
    if (item.amount > 300) return JSON.stringify({ decision: "approve", confidence: 0.62, rationale: "large amount, approve with lower confidence" });
    return JSON.stringify({ decision: "approve", confidence: 0.9, rationale: "clear, well-supported claim" });
  };
}

// ===========================================================================
// Main
// ===========================================================================
async function main(): Promise<void> {
  banner("D5.3 - Human-in-the-Loop Validation");
  const data = JSON.parse(readFileSync(path.join(DATA, "decisions.json"), "utf-8")) as Json;
  const cfg = data.thresholds as Config;
  const regulated = data.regulated_categories as string[];
  const items = data.items as Item[];
  const client = getClient(buildResponder());
  const corrections: Json[] = [];

  section("(a) Confidence/risk-based routing");
  console.log(`  ${"id".padEnd(5)}${"route".padEnd(14)}${"model".padEnd(9)}${"conf".padEnd(7)}triggers`);
  console.log("  " + "-".repeat(70));
  const routed: Routed[] = [];
  let reviewSeen = 0;
  for (const item of items) {
    const dec = await modelDecide(client, item);
    const r = route(item, dec, cfg, regulated);
    if (r.route === "HUMAN_REVIEW") {
      r.slaHours = REVIEW_LATENCIES[reviewSeen % REVIEW_LATENCIES.length]!;
      reviewSeen += 1;
    }
    routed.push(r);
    console.log(`  ${r.itemId.padEnd(5)}${r.route.padEnd(14)}${r.modelDecision.padEnd(9)}${String(r.confidence).padEnd(7)}${r.triggers.join(", ") || "-"}`);
  }

  const auto = items.filter((_, i) => routed[i]!.route === "AUTO_APPROVE");
  const review = routed.filter((r) => r.route === "HUMAN_REVIEW");
  const rejected = routed.filter((r) => r.route === "REJECT");

  section("(b) Approval gates in an agent tool loop (scripted human)");
  const scripted: HumanDecision[] = [
    { action: "edit", editedArgs: { amount: 200 }, note: "capped at policy limit $200" },
    { action: "approve", note: "customer confirmation email ok" },
  ];
  for (const line of await runAgentWithGates(client, scripted, corrections)) console.log("  " + line);

  section("(c) QA sampling of auto-approved items (stratified) + agreement");
  const sample = stratifiedSample(auto, 0.6);
  let agree = 0;
  for (const it of sample) {
    const reviewer = qaReviewer(it);
    const match = reviewer === "approve";
    agree += match ? 1 : 0;
    console.log(`  ${it.id.padEnd(5)} auto=approve  reviewer=${reviewer.padEnd(8)} ${match ? "AGREE" : "DISAGREE -> correction"}`);
    if (!match) corrections.push({ source: "qa_sample", item: it.id, auto: "approve", reviewer });
  }
  const agreement = sample.length ? agree / sample.length : 1;
  console.log(`  sampled ${sample.length} of ${auto.length} auto-approved; reviewer agreement = ${Math.round(agreement * 100)}%`);

  section("(d) Feedback capture -> corrections dataset");
  if (!corrections.length) console.log("  (none)");
  for (const c of corrections) console.log("  " + JSON.stringify(c));
  console.log(`  ${corrections.length} correction(s) captured for fine-tuning / prompt-improvement / policy review.`);

  section("(e) Operational metrics");
  const total = routed.length;
  const latencies = review.map((r) => r.slaHours);
  const withinSla = latencies.filter((h) => h <= cfg.review_sla_hours).length;
  const pct = (n: number) => `${Math.round((n / total) * 100)}%`;
  console.log(`  total decisions .......... ${total}`);
  console.log(`  automation rate .......... ${pct(auto.length)}  (${auto.length} auto-approved)`);
  console.log(`  human review load ........ ${pct(review.length)}  (${review.length} queued)`);
  console.log(`  reject rate .............. ${pct(rejected.length)}  (${rejected.length} rejected)`);
  if (latencies.length) {
    const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    console.log(`  avg escalation latency ... ${avg.toFixed(1)}h  (SLA ${cfg.review_sla_hours}h)`);
    console.log(`  SLA attainment ........... ${Math.round((withinSla / latencies.length) * 100)}%  (${withinSla}/${latencies.length} within SLA)`);
  }

  section("Summary");
  console.log("  Routing automates the easy, grounded, low-value cases and escalates the rest.");
  console.log("  High-risk tool actions are gated; every human override becomes training signal.");
  console.log("  QA sampling keeps auto-approvals honest and measures reviewer agreement.");
  console.log("  See README for automation-bias and reviewer-fatigue failure modes.");
}

await main();
