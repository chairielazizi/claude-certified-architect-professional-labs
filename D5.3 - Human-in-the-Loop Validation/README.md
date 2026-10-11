# D5.3 - Human-in-the-Loop Validation

> **Exam objective:** Apply human-in-the-loop validation strategies.

## Why it matters

Full automation is rarely the right target. The architecture question is not "human
or machine" but **where the human sits in the loop** and **which cases reach them**.
A good HITL design automates the easy, grounded, low-stakes cases, routes the risky or
low-confidence ones to people, gates irreversible actions behind explicit approval, and
turns every human correction into training signal - while measuring automation rate and
review load so the human capacity is sized correctly. Done badly, HITL becomes a
rubber-stamp (automation bias) or a bottleneck (reviewer fatigue).

## Key concepts

### Confidence/risk-based routing

Two independent inputs drive routing:

1. **Model self-reported confidence** - a *useful signal*, returned via structured
   output alongside the decision. It is **not ground truth**: a model can be
   confidently wrong, so confidence is one input, never the whole gate.
2. **Deterministic risk rules** - code you own: amount thresholds, regulated
   categories, retrieval-support floors, fraud signals. These are auditable and do
   not drift.

| Route | When | Rationale |
|---|---|---|
| **AUTO_APPROVE** | no risk trigger and confidence >= floor | cheap, grounded, low-value cases |
| **HUMAN_REVIEW** | amount over limit, regulated category, low support, or low confidence | risk or uncertainty warrants a person |
| **REJECT** | hard rule (e.g., fraud signal) | an upstream signal overrides the model entirely |

### Approval gates in an agent loop (approve / edit / deny)

High-risk **tool calls** (irreversible or external side effects - refunds over a
limit, external email, data deletion) must pause for a human before execution. The
human can **approve**, **edit** the arguments (e.g., cap a refund at policy), or
**deny**. The decision is fed back as the tool's `tool_result`, so the model continues
with the real outcome. You do **not** need a manual loop for this - you can gate inside
a tool-runner `run()` function too - but the manual loop makes the gate explicit.

### QA sampling of auto-approved items

Auto-approvals are not reviewed one-by-one (that would defeat automation). Instead a
**stratified random sample** (stratify by category so each segment is represented) is
re-decided by an independent reviewer, and **reviewer agreement** is tracked. Falling
agreement is an early warning that the auto-approve bar is set too low.

### Feedback loop -> corrections dataset

Every human override - an edited tool argument, a denial, a QA disagreement - is a
labeled correction. Captured systematically, these become a dataset for prompt
improvement, eval cases, or fine-tuning. The feedback loop is what makes the system
get *better* instead of just *monitored*.

### Metrics that size the human capacity

- **Automation rate** = auto-approved / total (the efficiency win).
- **Review load** = queued / total (staffing driver).
- **Reject rate** (and why).
- **SLA attainment** and **escalation latency** for the review queue.

### Two human-factors failure modes (discuss in any HITL design)

- **Automation bias**: reviewers over-trust the AI's recommendation and approve
  without real scrutiny - the "human in the loop" becomes a rubber stamp. Mitigate by
  surfacing confidence and rationale, hiding the AI's recommendation until the
  reviewer forms an opinion for high-stakes cases, QA-sampling the reviewers
  themselves, and rotating/randomizing.
- **Reviewer fatigue**: too many low-value escalations dull attention, so real issues
  slip through. Mitigate by tuning thresholds to keep review load sustainable,
  batching, prioritizing by risk, and monitoring agreement/throughput over time.

## Architecture

```
  item --> Claude (structured: decision + confidence)
              |
              v
      +-------------------------- deterministic risk rules --------------------------+
      | fraud? amount>limit? regulated? support<floor? confidence<floor?             |
      +-----------------------------------------------------------------------------+
         |                    |                           |
      REJECT            HUMAN_REVIEW                  AUTO_APPROVE
                             |                            |
                        review queue                 QA sample (stratified)
                        (SLA, latency)                    | reviewer re-decides
                                                          v
      agent tool loop:  high-risk tool_use --> [approve | edit | deny] --> tool_result
                                                          |
                          all overrides + QA disagreements --> corrections dataset
                                                          |
                                                    metrics (automation rate,
                                                    review load, SLA, latency)
```

## What the code demonstrates

| Step | Code (Python / TypeScript) |
|---|---|
| Structured model decision + confidence | `model_decide` / `modelDecide` |
| Deterministic risk routing | `route` |
| Approval gates (approve/edit/deny) in a tool loop | `run_agent_with_gates` / `runAgentWithGates`, `is_high_risk` |
| Stratified QA sampling | `stratified_sample` / `stratifiedSample` |
| Reviewer agreement | `qa_reviewer` / `qaReviewer` |
| Corrections dataset | `corrections` accumulation |
| Metrics | `(e)` section in `main` |

## Run it

From the repo root:

```bash
.venv/Scripts/python "D5.3 - Human-in-the-Loop Validation/python/main.py"
node "D5.3 - Human-in-the-Loop Validation/typescript/main.ts"
```

Offline, deterministic (fixed sampling seed and latency table), no API key.

## Exam tips & traps

- **Model confidence is an input, not the decision.** Combine it with deterministic
  risk rules; never auto-approve on self-reported confidence alone.
- **Gate the irreversible/external actions, not every action.** Low-risk reads run
  automatically; refunds-over-limit, external email, and deletes pause for a human.
- **"Edit" is a first-class human outcome**, not just approve/deny - and the edited
  arguments are the most valuable correction signal.
- **QA-sample the automated path** (and the reviewers) - unmonitored auto-approvals
  silently drift.
- **Design against automation bias and reviewer fatigue explicitly** - they are the
  two ways a HITL design quietly fails.
- **Capture every override as data** - a HITL system that doesn't feed corrections
  back is just expensive monitoring.
- You can gate inside a **tool-runner `run()`** function or in a **manual loop** -
  both support HITL; forced tool choice is not required (and 400s on 5.5-class models).

### Practice questions

**Q1.** A fraud-review assistant reports 0.97 confidence on a $9,000 refund in a
regulated insurance category. The router auto-approves it because confidence exceeds
the 0.75 floor. What is the design flaw?

- A. The confidence floor is too low.
- B. Routing relied on model confidence alone; deterministic risk rules (amount threshold, regulated category) should have escalated it regardless of confidence.
- C. The model should have been forced to use a tool.
- D. Nothing - high confidence justifies auto-approval.

**Answer: B.** Confidence is one input. High-value and regulated cases must escalate
on deterministic rules no matter how confident the model is; otherwise a confidently
wrong model bypasses all human oversight.

**Q2.** In an agent loop, Claude requests `issue_refund(order, amount=250)` and your
policy caps auto-refunds at $200. The reviewer wants to allow $200. What is the correct
HITL mechanism?

- A. Deny the tool call and tell the user to resubmit.
- B. Approve as-is; the model knows best.
- C. The reviewer EDITs the arguments to amount=200; the edited call executes and the edit is captured as a correction fed back as the tool_result.
- D. Remove the issue_refund tool from the agent.

**Answer: C.** Approve/**edit**/deny is the gate contract; editing to the policy cap
executes the corrected action and records a high-value training signal, while the model
continues from the real tool_result.

## Further reading

- Anthropic docs: "Tool use" (approval gates, tool-runner hooks), "Handling stop reasons"
- Anthropic docs: "Structured outputs" (self-reported fields, schema guarantees)
- Anthropic engineering: "Building effective agents" (human oversight in agent loops)
- Human-factors literature: automation bias, complacency, and vigilance decrement
- NIST AI RMF: "Measure" and "Manage" functions (human oversight controls)
