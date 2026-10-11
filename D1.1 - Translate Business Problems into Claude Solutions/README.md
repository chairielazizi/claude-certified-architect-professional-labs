# D1.1 - Translate Business Problems into Claude Solutions

> **Exam objective:** "Translate business problems into Claude-based AI solutions"

## Why it matters

Most failed AI projects fail before the first prompt is written: they start from "let's build an agent"
instead of from a measurable business pain. An architect's first deliverable is a **framing** (who hurts,
how much, with what data, under which constraints, measured by which KPI), then a **suitability call**
(should an LLM be involved at all, and at which level of autonomy), and only then a **blueprint** that maps
the problem onto concrete Claude capabilities. Getting this wrong is expensive in both directions: an agent
where a single call would do burns money and predictability; an LLM where a rules engine belongs creates
errors you cannot audit.

## Key concepts

### 1. Problem framing - the minimum fields

| Field | Why the architect needs it |
|---|---|
| Pain point + current process | Defines the baseline you will be measured against |
| Primary users | Determines UX, trust and human-in-the-loop (HITL) design |
| Volume + minutes per item + loaded cost | Sizes the value (and the token bill) |
| Data shape and modalities | Structured -> code; unstructured text/images/PDF -> LLM territory |
| Are the steps predictable? | Workflow (yes) vs agent (no) |
| Latency budget | LLM calls take hundreds of ms to seconds; sub-100 ms budgets rule them out at runtime |
| Error severity + can a human review? | Drives guardrails, HITL and sometimes "no" |
| Success KPIs (with baseline) | Business KPIs, not model metrics: "triage minutes per claim", not "accuracy" |

Use Claude to *extract* this framing from interview notes with **structured outputs**
(`client.messages.parse(output_format=Model)` in Python, `output_config.format` + `zodOutputFormat` in
TypeScript). This turns messy prose into a typed contract the rest of your tooling can consume.

### 2. The four suitability criteria

Anthropic's guidance on when to build agents boils down to four questions. The lab scores each 1-5 in code:

| Criterion | Question | Signals |
|---|---|---|
| **Complexity** | Does the task actually need language understanding or judgment? | Deterministic rules = 1, summarise = 2, judgment over language = 3, open-ended investigation = 5; +1 for multimodal |
| **Value** | Is the prize big enough to justify the tier's build + run cost? | Annual human hours x loaded cost; the agent tier needs more value than a single call |
| **Viability** | Can Claude meet the hard constraints with the data available? | Latency budget, data/system access, integration readiness |
| **Cost of error** | What happens when it is wrong, and will anyone notice in time? | Severity, reversibility, whether a human checkpoint is feasible ("error discovery") |

### 3. The tier ladder - choose the simplest that works

| Tier | Use when | Example in the lab |
|---|---|---|
| **Not an LLM problem** | Deterministic logic over structured data, exact arithmetic, sub-100 ms budgets, unreviewable volume with critical errors | P2 sales tax -> rules engine |
| **Single LLM call** | One well-scoped transformation (summarise, classify, extract) | P4 feedback digest |
| **Workflow** | Steps are known in advance, but several capabilities or systems are involved; code owns the control flow | P1 claims triage |
| **Agent** | Open-ended, the path differs every time, value is high, and errors can be caught (read-only tools, approval gates) | P3 incident investigation |

Anthropic's "Building effective agents" recommends finding the simplest solution possible and only
increasing complexity when needed. Complexity is a cost: every step up the ladder trades predictability,
latency and spend for flexibility.

### 4. Capability mapping - business need to platform feature

| Business need | Claude capability / platform feature |
|---|---|
| Free text -> fields a system can use | Extraction with **structured outputs** (`output_config.format`) or `strict: true` tools |
| Decide a category, severity, route | **Classification** (often a small, fast model; see D2.1) |
| Photos, screenshots, scanned forms | **Vision** (image blocks) |
| Reports, contracts, statements | **PDF support** (document blocks), Files API for reuse |
| Answers must follow policy / cite sources | **RAG** + **citations** |
| Read or act on live systems | **Tool use** (client tools, MCP) |
| Unknown sequence of steps | **Agent loop**: tool use until `stop_reason == "end_turn"`, with step limits |
| Non-urgent high volume | **Message Batches API** (50% discount, async) |
| Large repeated context | **Prompt caching** (see D2.5) |

### 5. When the answer is "don't use an LLM"

Say no (at least on the runtime path) when the task is **deterministic** (a correct answer is computable),
**exact** (cents, dosages, legal thresholds), **latency-critical** (below LLM inference time), **fully
structured**, and **unreviewable at volume**. Claude can still add value *off* the hot path: Claude Code
maintaining rule tables and tests, or Claude explaining rule changes to humans. If an LLM-based flow needs
an exact calculation, give Claude a **tool** that runs the deterministic code instead of asking it to do the
arithmetic.

## Architecture

```
 interview notes (4 problems)
          |
          v
 +-----------------------------+      Claude Sonnet + structured outputs
 | 1. Framing extraction       |----> ProblemFraming (typed: volume, data shape,
 +-----------------------------+      latency, severity, KPIs, ...)
          |
          v
 +-----------------------------+      deterministic code
 | 2. Suitability rubric       |----> complexity / value / viability / cost of error (1-5)
 +-----------------------------+
          |
          v
 +-----------------------------+      deterministic code
 | 3. Capability map + tier    |----> NOT_LLM | SINGLE_CALL | WORKFLOW | AGENT  (rule id shown)
 +-----------------------------+
          |
          v
 +-----------------------------+      deterministic code            Claude Opus
 | 4. Blueprint                |----> components, controls, -----> one-pager for the
 +-----------------------------+      models, KPIs                 top candidate
          |
          v
 5. Contrast: rules engine vs LLM for P2 (exactness, latency, cost at volume)
```

## What the code demonstrates

| Step | Code section (both languages) | What to notice |
|---|---|---|
| 1 | `problems.json`, section "1. Intake" | Four contrasting problems, written as raw notes |
| 2 | `ProblemFraming` model/schema, `messages.parse(...)` | Structured outputs return a validated object; `stop_reason` is checked before use; `effort` combined with the format in `output_config` |
| 3 | `assess()` | Transparent rubric: every score has a printed reason |
| 4 | `map_capabilities()` / `decide_tier()` | Ordered decision table; first matching rule wins and the rule id is printed |
| 5 | `build_blueprint()` | Capabilities grouped into the fewest sensible steps; controls derived from constraints (audit, PII, regulator) |
| 6 | Opus one-pager | Claude writes the narrative from the *computed* payload - "do not invent numbers" |
| 7 | `rules_engine_tax_cents()` vs Haiku | The LLM answer is plausible but one cent off; latency and monthly cost at 60M calls make the case |

## Run it

From the repository root:

```bash
.venv/Scripts/python "D1.1 - Translate Business Problems into Claude Solutions/python/main.py"
node "D1.1 - Translate Business Problems into Claude Solutions/typescript/main.ts"
```

Offline by default (deterministic canned replies). Set `ANTHROPIC_API_KEY` to run the framing extraction and
one-pager against the live API.

## Exam tips & traps

- **"Use an agent" is rarely the right first answer.** If the steps can be written down in advance, the
  expected answer is a workflow (or a single call). Agents are for open-ended paths with high value and
  catchable errors.
- **Exact, deterministic, structured, low-latency = code.** Distractors often propose a bigger model or
  more few-shot examples for arithmetic or rule tables. The right answer is a deterministic service, or a
  tool that Claude calls.
- **Value must be sized at volume.** Multiply tokens per task by volume before recommending a model; a cheap
  per-call price can still be a large monthly bill (and see D1.6 / D4.5).
- **Cost of error includes error *discovery*.** If mistakes are hard to detect, the design needs HITL,
  evals and audit trails, or a lower autonomy tier.
- **KPIs are business outcomes with a baseline** (minutes per claim, misrouting rate), not "model accuracy"
  alone. Evaluation metrics come later (D4.1).
- **Structured outputs over "please reply in JSON".** On Opus 5.5 / Sonnet 5.5 forced `tool_choice`
  (`any`/`tool`) returns a 400; use `output_config.format` / `messages.parse`, or `strict: true` tools.
- **Regulated decisions stay with accountable humans.** Claude triages, extracts and recommends; the
  licensed adjuster decides.

### Practice questions

**Q1.** A parcel carrier wants to "use Claude" to compute shipping fees for 5 million parcels a day from a
published rate card (weight, zone, service level). Fees must be exact and the checkout latency budget is
80 ms. What should the architect recommend?

- A. A Claude Opus agent with a calculator tool and adaptive thinking
- B. Claude Haiku with few-shot examples of correct fee calculations
- C. A deterministic rate engine at runtime; use Claude off the hot path (e.g., to help maintain rate-card parsing and tests)
- D. Claude Sonnet with structured outputs and human review of every fee

**Answer: C.** The task is deterministic, exact, structured, high-volume and latency-critical: all four
signals of "not an LLM problem" at runtime. A and B still put probabilistic generation and hundreds of ms of
inference on the hot path; D is unreviewable at 5M/day.

**Q2.** A bank's card-dispute team handles 8,000 disputes a month. Each dispute needs the customer's
narrative and the merchant's evidence PDFs read, mapped to a reason code through a fixed, documented
procedure, and decided by an analyst. Which design best fits?

- A. An autonomous agent with write access to the ledger that resolves disputes end to end
- B. A workflow: PDF + text extraction with structured outputs -> reason-code classification -> rules check -> analyst review queue
- C. One prompt containing all documents that asks Claude for the final decision
- D. Not an LLM problem - keep the manual process

**Answer: B.** The steps are predictable and multi-capability (PDF, extraction, classification, rules), the
value is meaningful, and the high cost of error is mitigated by keeping the analyst as the decision maker.
A adds autonomy and risk with no benefit; C skips the controls and auditability; D ignores a strong fit.

## Further reading

- Anthropic engineering blog: "Building effective agents" (workflows vs agents, start simple)
- Claude docs: "Structured outputs" (JSON outputs and strict tool use)
- Claude docs: "Vision" and "PDF support"
- Claude docs: "Citations" and "Tool use overview"
- Claude docs: "Define your success criteria" and "Create strong empirical evaluations"
- Claude docs: use-case guides (ticket routing, customer support agent, content moderation, legal summarization)
- Related labs: D1.3 (patterns), D1.6 (business value), D2.1 (model selection), D6.1 (discovery)
