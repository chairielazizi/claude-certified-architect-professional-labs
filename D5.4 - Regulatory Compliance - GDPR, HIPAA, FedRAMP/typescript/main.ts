/**
 * D5.4 - Regulatory Compliance: GDPR, HIPAA, FedRAMP (TypeScript)
 *
 * EDUCATIONAL ONLY - NOT LEGAL ADVICE. See README. Authorization claims are phrased
 * as "verify ..." on purpose. Equivalent to python/main.py:
 *   (a) compliance policy engine, (b) PHI/PII de-identification + reversible vault +
 *   the Claude call + re-identification, (c) GDPR erasure/export, (d) hash-chained
 *   immutable audit log with tamper detection.
 *
 * Run (from repo root):
 *   node "D5.4 - Regulatory Compliance - GDPR, HIPAA, FedRAMP/typescript/main.ts"
 */

import { createHash } from "node:crypto";
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

interface Workload {
  name: string;
  data_types: string[];
  user_region: string;
  government_customer: boolean;
  retention_days: number;
  needs_zdr: boolean;
}

// ===========================================================================
// (a) Compliance policy engine
// ===========================================================================
interface Control {
  name: string;
  required: boolean;
  note: string;
}

function evaluatePolicy(w: Workload): Control[] {
  const dt = new Set(w.data_types);
  const controls: Control[] = [];
  controls.push({ name: "Encryption in transit + at rest", required: true, note: "TLS to the API; encrypt stored logs/indexes" });
  controls.push({ name: "Access controls (least privilege, RBAC)", required: true, note: "scope who can read prompts/outputs/logs" });
  controls.push({ name: "Immutable audit logging", required: true, note: "hash-chained, tamper-evident (see part d)" });

  const sensitive = ["PHI", "PII", "CUI", "PCI"].some((t) => dt.has(t));
  if (sensitive) {
    controls.push({ name: "Data minimization", required: true, note: "send only what the task needs; strip the rest" });
    controls.push({ name: "De-identification / pseudonymization", required: true, note: "tokenize identifiers before the model (see part b)" });
  }
  if (dt.has("PHI")) {
    controls.push({ name: "HIPAA: Business Associate Agreement (BAA)", required: true, note: "verify a signed BAA is in place with the provider before sending PHI" });
    controls.push({ name: "HIPAA: minimum necessary standard", required: true, note: "limit PHI to the minimum necessary" });
    controls.push({ name: "HIPAA: breach-notification process", required: true, note: "documented IR + notification workflow" });
  }
  if (dt.has("PII") && w.user_region === "EU") {
    controls.push({ name: "GDPR: lawful basis + DPA", required: true, note: "document lawful basis; Data Processing Addendum with the provider" });
    controls.push({ name: "GDPR: data residency (inference_geo='eu')", required: true, note: "pin inference geography - 1P / Claude Platform on AWS; on Bedrock/Vertex/Foundry residency is the platform region, inference_geo is unavailable (verify)" });
    controls.push({ name: "GDPR: data-subject rights (erasure, export, access)", required: true, note: "implement DSR workflows (see part c)" });
    controls.push({ name: "GDPR: storage limitation", required: true, note: `retention ${w.retention_days}d must match a documented purpose` });
  }
  if (dt.has("CUI") || w.government_customer) {
    controls.push({ name: "FedRAMP: authorized environment", required: true, note: "use a government-authorized cloud (e.g., Bedrock GovCloud / Vertex gov) - verify the current FedRAMP authorization level with the provider" });
    controls.push({ name: "FedRAMP: US data residency", required: true, note: "keep data + inference in US/gov regions - verify per platform" });
    controls.push({ name: "FedRAMP: continuous monitoring + boundary", required: true, note: "ConMon, SSP, authorization boundary documented" });
  }
  if (dt.has("PCI")) {
    controls.push({ name: "PCI DSS: tokenize PAN, minimize scope", required: true, note: "never send raw card data; tokenize and reduce CDE scope" });
  }
  if (w.needs_zdr) {
    controls.push({ name: "Zero data retention (ZDR) arrangement", required: true, note: "ZDR only if expressly authorized by Anthropic; note Fable 5.1 requires 30-day retention and is NOT available under ZDR unless authorized - choose a different model for ZDR workloads" });
  } else {
    controls.push({ name: "Retention policy", required: true, note: `delete logs/indexes after ${w.retention_days}d per policy` });
  }
  return controls;
}

function recommendPlatform(w: Workload): [string, string] {
  const dt = new Set(w.data_types);
  if (dt.has("CUI") || w.government_customer) return ["Government cloud (Bedrock GovCloud / Vertex gov)", "verify current FedRAMP authorization level + US residency"];
  if (dt.has("PII") && w.user_region === "EU") return ["Claude API / Claude Platform on AWS with inference_geo='eu'", "EU residency via inference_geo (1P/P-AWS); verify DPA"];
  if (dt.has("PHI")) return ["Any BAA-covered deployment", "verify signed BAA before sending PHI"];
  return ["Claude API (standard)", "standard controls"];
}

function inferenceGeoFor(w: Workload): string {
  const dt = new Set(w.data_types);
  if (dt.has("CUI") || w.government_customer) return "us";
  if (w.user_region === "EU") return "eu";
  return "global";
}

// ===========================================================================
// (b) De-identification + reversible tokenization vault
// ===========================================================================
const NAME_DICTIONARY = ["Jane Doe", "John Smith", "Maria Garcia"];
const DEID_PATTERNS: Array<[string, RegExp]> = [
  ["EMAIL", /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g],
  ["MRN", /\bMRN-\d{3,6}\b/g],
  ["SSN", /\b\d{3}-\d{2}-\d{4}\b/g],
  ["PHONE", /(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g],
];

class TokenVault {
  readonly toToken = new Map<string, string>();
  readonly toValue = new Map<string, string>();
  private counts = new Map<string, number>();

  tokenize(kind: string, value: string): string {
    const existing = this.toToken.get(value);
    if (existing) return existing;
    const n = (this.counts.get(kind) ?? 0) + 1;
    this.counts.set(kind, n);
    const token = `[${kind}_${n}]`;
    this.toToken.set(value, token);
    this.toValue.set(token, value);
    return token;
  }

  reidentify(text: string): string {
    let out = text;
    for (const [token, value] of this.toValue) out = out.split(token).join(value);
    return out;
  }
}

function deIdentify(text: string, vault: TokenVault): string {
  let out = text;
  for (const name of NAME_DICTIONARY) if (out.includes(name)) out = out.split(name).join(vault.tokenize("NAME", name));
  for (const [kind, pattern] of DEID_PATTERNS) {
    const matches = [...new Set(out.match(pattern) ?? [])].sort((a, b) => b.length - a.length);
    for (const m of matches) out = out.split(m).join(vault.tokenize(kind, m));
    pattern.lastIndex = 0;
  }
  return out;
}

// ===========================================================================
// (c) GDPR data-subject rights
// ===========================================================================
class DataStores {
  conversationLogs: Array<{ subject: string; text: string }>;
  vectorIndex: Array<{ subject: string; vec_id: string }>;
  promptCache: Map<string, { subject: string }>;
  constructor(logs: Array<{ subject: string; text: string }>, vectors: Array<{ subject: string; vec_id: string }>, cache: Map<string, { subject: string }>) {
    this.conversationLogs = logs;
    this.vectorIndex = vectors;
    this.promptCache = cache;
  }
  eraseSubject(subjectId: string): { logs: number; vectors: number; cache: number } {
    const before = [this.conversationLogs.length, this.vectorIndex.length, this.promptCache.size] as const;
    this.conversationLogs = this.conversationLogs.filter((r) => r.subject !== subjectId);
    this.vectorIndex = this.vectorIndex.filter((r) => r.subject !== subjectId);
    for (const [k, v] of [...this.promptCache]) if (v.subject === subjectId) this.promptCache.delete(k);
    return { logs: before[0] - this.conversationLogs.length, vectors: before[1] - this.vectorIndex.length, cache: before[2] - this.promptCache.size };
  }
  exportSubject(subjectId: string): Json {
    return {
      subject: subjectId,
      conversation_logs: this.conversationLogs.filter((r) => r.subject === subjectId),
      vector_index: this.vectorIndex.filter((r) => r.subject === subjectId),
      prompt_cache: [...this.promptCache.values()].filter((v) => v.subject === subjectId),
    };
  }
}

// ===========================================================================
// (d) Immutable audit log with hash chaining
// ===========================================================================
interface AuditEntry {
  seq: number;
  action: string;
  detail: Json;
  prev_hash: string;
  hash: string;
}

class AuditChain {
  readonly entries: AuditEntry[] = [];
  private static hash(seq: number, action: string, detail: Json, prevHash: string): string {
    // Fixed key order in both append() and verify() -> consistent, deterministic hash.
    // (A plain stringify is enough here; a mutation of `detail` changes the digest,
    // which is exactly the tamper signal we want.)
    const body = JSON.stringify({ seq, action, detail, prev_hash: prevHash });
    return createHash("sha256").update(body).digest("hex");
  }
  append(action: string, detail: Json): void {
    const prevHash = this.entries.length ? this.entries[this.entries.length - 1]!.hash : "GENESIS";
    const seq = this.entries.length + 1;
    this.entries.push({ seq, action, detail, prev_hash: prevHash, hash: AuditChain.hash(seq, action, detail, prevHash) });
  }
  verify(): [boolean, number] {
    let prevHash = "GENESIS";
    for (const e of this.entries) {
      const recomputed = AuditChain.hash(e.seq, e.action, e.detail, prevHash);
      if (recomputed !== e.hash || e.prev_hash !== prevHash) return [false, e.seq];
      prevHash = e.hash;
    }
    return [true, -1];
  }
}

// ===========================================================================
// Offline responder
// ===========================================================================
function buildResponder(): (req: MockRequest) => ResponderResult {
  return (req: MockRequest): ResponderResult => {
    const q = req.lastUserText;
    const name = q.includes("[NAME_1]") ? "[NAME_1]" : "the patient";
    const mrn = q.includes("[MRN_1]") ? " (MRN [MRN_1])" : "";
    const contact = q.includes("[EMAIL_1]") ? " Follow-up via [EMAIL_1]." : "";
    return `Summary: ${name}${mrn} presented with mild symptoms; routine follow-up advised.${contact}`;
  };
}

// ===========================================================================
// Main
// ===========================================================================
async function main(): Promise<void> {
  banner("D5.4 - Regulatory Compliance: GDPR, HIPAA, FedRAMP");
  console.log("  DISCLAIMER: educational demonstration only - NOT legal advice. Verify all");
  console.log("  authorizations (BAA, FedRAMP level, residency, ZDR) with the provider.");
  const data = JSON.parse(readFileSync(path.join(DATA, "workloads.json"), "utf-8")) as { workloads: Workload[] };

  section("(a) Compliance policy engine - checklist per workload");
  for (const w of data.workloads) {
    const [platform, pnote] = recommendPlatform(w);
    console.log(`\n  [${w.name}]  data=${JSON.stringify(w.data_types)} region=${w.user_region} gov=${w.government_customer} retention=${w.retention_days}d zdr=${w.needs_zdr}`);
    console.log(`    recommended platform: ${platform}  (${pnote})`);
    console.log(`    inference_geo: '${inferenceGeoFor(w)}'`);
    for (const c of evaluatePolicy(w)) {
      console.log(`      [${c.required ? "x" : " "}] ${c.name}`);
      console.log(`            - ${c.note}`);
    }
  }

  section("(b) PHI/PII de-identification -> Claude -> re-identification");
  const client = getClient(buildResponder());
  const vault = new TokenVault();
  const note = "Jane Doe (MRN MRN-4471), email jane.doe@example.com, phone (555) 555-0100, SSN 000-00-0000, presented with a mild cough.";
  const deid = deIdentify(note, vault);
  console.log(`  original (inside trust boundary):\n    ${note}`);
  console.log(`  de-identified (this is what LEAVES the boundary to the model):\n    ${deid}`);
  const msg = await client.messages.create({ model: MODEL_OPUS, max_tokens: 256, messages: [{ role: "user", content: `Summarize this clinical note:\n${deid}` }] });
  const modelText = msg.content.filter((b): b is Anthropic.TextBlock => b.type === "text").map((b) => b.text).join("");
  console.log(`  model output (still tokenized - safe to log):\n    ${modelText}`);
  console.log(`  re-identified (inside trust boundary only):\n    ${vault.reidentify(modelText)}`);
  console.log(`  vault entries: ${JSON.stringify([...vault.toValue.keys()])}  (map never leaves the boundary)`);

  section("(c) GDPR data-subject rights - erasure + export");
  const stores = new DataStores(
    [
      { subject: "subj_eu_1", text: "chat 1" },
      { subject: "subj_eu_1", text: "chat 2" },
      { subject: "subj_us_9", text: "chat 3" },
    ],
    [
      { subject: "subj_eu_1", vec_id: "v1" },
      { subject: "subj_us_9", vec_id: "v2" },
    ],
    new Map([
      ["ck1", { subject: "subj_eu_1" }],
      ["ck2", { subject: "subj_us_9" }],
    ]),
  );
  console.log(`  data export for subj_eu_1: ${JSON.stringify(stores.exportSubject("subj_eu_1"))}`);
  const deleted = stores.eraseSubject("subj_eu_1");
  console.log(`  right to erasure for subj_eu_1 -> deleted ${JSON.stringify(deleted)}`);
  console.log(`  remaining logs=${stores.conversationLogs.length} vectors=${stores.vectorIndex.length} cache=${stores.promptCache.size} (other subjects untouched)`);

  section("(d) Immutable audit log (hash chaining + tamper detection)");
  const audit = new AuditChain();
  audit.append("deidentify", { workload: "Clinical note summarizer", tokens: vault.toValue.size });
  audit.append("model_call", { model: MODEL_OPUS, pii_sent: false });
  audit.append("gdpr_export", { subject: "subj_eu_1" });
  audit.append("gdpr_erasure", { subject: "subj_eu_1", deleted });
  const [ok, broken] = audit.verify();
  console.log(`  chain length ${audit.entries.length}; verify -> ${ok ? "OK" : `BROKEN at seq ${broken}`}`);
  for (const e of audit.entries) {
    console.log(`    seq ${e.seq} ${e.action.padEnd(13)} hash=${e.hash.slice(0, 12)}... prev=${e.prev_hash === "GENESIS" ? "GENESIS" : e.prev_hash.slice(0, 12)}`);
  }
  audit.entries[1]!.detail.pii_sent = true;
  const [ok2, broken2] = audit.verify();
  console.log(`  after tampering seq 2's detail -> verify = ${ok2 ? "OK" : `BROKEN at seq ${broken2}`} (tamper detected)`);

  section("Summary");
  console.log("  The policy engine turns a workload profile into an auditable control set.");
  console.log("  De-identification keeps raw PHI/PII inside the trust boundary; only tokens are sent.");
  console.log("  GDPR DSR workflows erase/export across every store a subject touches.");
  console.log("  The hash-chained audit log makes any after-the-fact edit detectable.");
}

await main();
