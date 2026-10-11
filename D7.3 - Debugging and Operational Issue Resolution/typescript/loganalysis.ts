/**
 * Deterministic incident analysis (D7.3) - twin of python/loganalysis.py.
 *
 * Parse logs, cluster error signatures, build a timeline, and correlate the incident with recent
 * deploys BEFORE (and independently of) any model call. It is cheap, reproducible, and becomes the
 * ground truth used to check the triage agent's claims.
 */

export type Json = Record<string, any>;

const LINE_RE = /^(\S+)\s+(DEBUG|INFO|WARN|ERROR)\s+(\S+)\s+(.*)$/;
const KV_RE = /(\w+)=("[^"]*"|\S+)/g;
/** Per-occurrence values that would split one problem into many clusters. */
const VOLATILE = new Set([
  "request_id", "latency_ms", "retry_after", "attempt", "delay_ms", "input_tokens", "output_tokens",
  "cache_read_input_tokens", "cache_creation_input_tokens", "error_rate", "p95_ms", "threshold", "threshold_ms",
]);

export const epoch = (ts: string): number => Math.floor(Date.parse(ts) / 1000);

export interface LogRecord {
  ts: string;
  epoch: number;
  time: string; // HH:MM:SS
  level: string;
  component: string;
  event: string;
  fields: Record<string, string>;
  raw: string;
}

export function parseLog(text: string): [LogRecord[], number] {
  const records: LogRecord[] = [];
  let unparsed = 0;
  const lines = text.split(/\r?\n/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  for (const line of lines) {
    const m = LINE_RE.exec(line);
    if (!m) {
      unparsed++;
      continue;
    }
    const [, ts, level, component, rest] = m as unknown as [string, string, string, string, string];
    const firstKv = /(\w+)=("[^"]*"|\S+)/.exec(rest);
    const event = firstKv ? rest.slice(0, firstKv.index).trim() : rest.trim();
    const fields: Record<string, string> = {};
    for (const kv of rest.matchAll(KV_RE)) fields[kv[1]!] = kv[2]!.replace(/^"+|"+$/g, "");
    records.push({ ts, epoch: epoch(ts), time: ts.slice(11, 19), level, component, event, fields, raw: line });
  }
  return [records, unparsed];
}

export function signature(rec: LogRecord): string {
  const stable = Object.entries(rec.fields).filter(([k]) => !VOLATILE.has(k)).map(([k, v]) => `${k}=${v}`);
  return [rec.component, rec.event, ...stable].join(" ");
}

export interface Cluster {
  signature: string;
  level: string;
  count: number;
  first: string;
  last: string;
}

export function clusterErrors(records: LogRecord[]): Cluster[] {
  const clusters = new Map<string, Cluster>();
  for (const r of records) {
    if (r.level !== "WARN" && r.level !== "ERROR") continue;
    const sig = signature(r);
    const c = clusters.get(sig) ?? { signature: sig, level: r.level, count: 0, first: r.time, last: r.time };
    c.count++;
    c.last = r.time;
    clusters.set(sig, c);
  }
  return [...clusters.values()].sort((a, b) => b.count - a.count || (a.first < b.first ? -1 : a.first > b.first ? 1 : 0));
}

export function bar(value: number, maxValue: number, width: number): string {
  const n = maxValue > 0 ? Math.floor((value / maxValue) * width + 0.5) : 0;
  return "#".repeat(n) + " ".repeat(width - n);
}

export const mean = (values: number[]): number => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);

export interface Facts {
  deployVersion: string;
  deployFinished: string;
  first429: string;
  first529: string;
  secondsDeployToFirst429: number;
  retriesIgnoringRetryAfter: number;
  maxRetryAfterS: number;
  cacheRatioBefore: number;
  cacheRatioAfter: number;
  uncachedInputPerCallBefore: number;
  uncachedInputPerCallAfter: number;
  errorRateBefore: number;
  errorRateAfter: number;
  p95BeforeMs: number;
  p95AfterMs: number;
  itpmPeak: number;
  itpmLimit: number;
  inFlightBefore: number;
  inFlightAfter: number;
}

export function correlate(records: LogRecord[], metrics: Json, deploys: Json[]): Facts {
  const f: Facts = {
    deployVersion: "", deployFinished: "", first429: "", first529: "", secondsDeployToFirst429: 0,
    retriesIgnoringRetryAfter: 0, maxRetryAfterS: 0, cacheRatioBefore: 0, cacheRatioAfter: 0,
    uncachedInputPerCallBefore: 0, uncachedInputPerCallAfter: 0, errorRateBefore: 0, errorRateAfter: 0,
    p95BeforeMs: 0, p95AfterMs: 0, itpmPeak: 0, itpmLimit: 0, inFlightBefore: 0, inFlightAfter: 0,
  };
  const errors429 = records.filter((r) => r.fields.status === "429" && r.component === "claude-client");
  const errors529 = records.filter((r) => r.fields.status === "529" && r.component === "claude-client");
  const errEpochs = [...errors429, ...errors529].map((r) => r.epoch);
  const firstErrEpoch = errEpochs.length ? Math.min(...errEpochs) : 0;
  // Suspect deploy = the most recent deploy that finished before the first error.
  const candidates = deploys.filter((d) => epoch(d.finished) <= firstErrEpoch);
  const suspect = candidates.reduce<Json | null>((best, d) => (!best || epoch(d.finished) > epoch(best.finished) ? d : best), null);
  if (suspect) {
    f.deployVersion = suspect.version;
    f.deployFinished = String(suspect.finished).slice(11, 19);
    if (errors429.length) f.secondsDeployToFirst429 = errors429[0]!.epoch - epoch(suspect.finished);
  }
  f.first429 = errors429.length ? errors429[0]!.time : "";
  f.first529 = errors529.length ? errors529[0]!.time : "";

  // Retry storm: a retry scheduled sooner than the retry-after the API just asked for.
  let lastRetryAfter = 0;
  for (const r of records) {
    if (r.component !== "claude-client") continue;
    if (r.fields.status === "429") {
      lastRetryAfter = Number(r.fields.retry_after ?? "0");
      f.maxRetryAfterS = Math.max(f.maxRetryAfterS, lastRetryAfter);
    } else if (r.event === "retrying" && r.fields.reason === "rate_limit_error") {
      if (Number(r.fields.delay_ms ?? "0") < lastRetryAfter * 1000) f.retriesIgnoringRetryAfter++;
    }
  }

  const cut = suspect ? epoch(suspect.finished) : 0;
  const responses = records.filter((r) => r.component === "claude-client" && r.event === "response");
  const ratio = (rs: LogRecord[]): number => {
    const read = rs.reduce((n, r) => n + Number(r.fields.cache_read_input_tokens ?? "0"), 0);
    const uncached = rs.reduce((n, r) => n + Number(r.fields.input_tokens ?? "0"), 0);
    return read + uncached ? read / (read + uncached) : 0;
  };
  const before = responses.filter((r) => r.epoch < cut);
  const after = responses.filter((r) => r.epoch >= cut);
  f.cacheRatioBefore = ratio(before);
  f.cacheRatioAfter = ratio(after);
  f.uncachedInputPerCallBefore = mean(before.map((r) => Number(r.fields.input_tokens)));
  f.uncachedInputPerCallAfter = mean(after.map((r) => Number(r.fields.input_tokens)));

  const pts = metrics.points as Json[];
  const pre = pts.filter((p) => p.minute < "13:59");
  const post = pts.filter((p) => p.minute >= "14:00");
  f.errorRateBefore = mean(pre.map((p) => p.error_rate));
  f.errorRateAfter = mean(post.map((p) => p.error_rate));
  f.p95BeforeMs = mean(pre.map((p) => p.p95_latency_ms));
  f.p95AfterMs = mean(post.map((p) => p.p95_latency_ms));
  f.itpmPeak = Math.max(...pts.map((p) => p.input_tokens_per_min));
  f.itpmLimit = metrics.limits.claude_opus_input_tokens_per_min;
  f.inFlightBefore = mean(pre.map((p) => p.in_flight));
  f.inFlightAfter = mean(post.map((p) => p.in_flight));
  return f;
}
