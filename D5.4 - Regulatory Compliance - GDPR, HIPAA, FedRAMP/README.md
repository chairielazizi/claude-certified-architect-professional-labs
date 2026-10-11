# D5.4 - Regulatory Compliance: GDPR, HIPAA, FedRAMP

> **Exam objective:** Ensure compliance with regulations (e.g., GDPR, HIPAA, FedRAMP).

> **DISCLAIMER:** This lab is an **educational demonstration, not legal advice**. The
> policy rules are simplified. Confirm all authorizations - signed BAA, current FedRAMP
> authorization level, data-residency guarantees, ZDR eligibility - **directly with the
> provider** for your specific deployment. Nothing here asserts a specific certification
> for a specific platform.

## Why it matters

Compliance is an **architecture** property, not a checkbox added at the end. The same
Claude model may be deployable for one workload and prohibited for another depending on
data types, user geography, and customer type. An architect translates a workload's
regulatory profile into a concrete, auditable control set: what must be de-identified,
where inference may run, which platform is eligible, what retention applies, and how
every access is logged. Getting it wrong is not a bug - it is a breach.

## Key concepts

### Regulation -> control mapping (simplified)

| Regulation | Trigger | Representative controls |
|---|---|---|
| **HIPAA** | PHI (health data) | Signed **BAA** (verify), minimum-necessary, encryption, access controls, breach notification, de-identification |
| **GDPR** | PII of EU data subjects | Lawful basis + **DPA**, **data residency** (`inference_geo='eu'`), data-subject rights (erasure/export/access), storage limitation, pseudonymization |
| **FedRAMP** | CUI / government customer | **Government-authorized cloud** (verify authorization level), US residency, continuous monitoring, documented authorization boundary |
| **PCI DSS** | Cardholder data | Tokenize PAN, minimize CDE scope, never send raw card data |

### Data residency on the Claude platform

- **`inference_geo`** is a top-level request parameter (e.g., `"us"`, `"eu"`,
  `"global"`); `usage.inference_geo` reports where inference actually ran.
- Availability: **first-party Claude API and Claude Platform on AWS only**. On Amazon
  Bedrock, Google Vertex AI, and Microsoft Foundry the parameter is **not** available -
  residency there is governed by the **platform region** you deploy into (e.g., an EU
  Bedrock region, a GovCloud region). Design accordingly.

### Retention and ZDR

- **Zero Data Retention (ZDR)** is only available if **expressly authorized by
  Anthropic**.
- **Fable 5.1 (and the Mythos 5.x / Fable 5 family) require 30-day retention** and are
  **not** available under ZDR unless specifically authorized - a ZDR org gets a
  `400 invalid_request_error` on every request to those models. For a ZDR workload,
  choose a model that supports it rather than fighting the 400.

### De-identification and the trust boundary

The highest-leverage control for PHI/PII is to **not send it at all**. De-identify
*before* the model call:

1. Detect identifiers (regex + dictionaries; production adds NER).
2. Replace each with a stable token via a **reversible vault** that lives **inside the
   trust boundary** and never leaves it.
3. Send only the **de-identified** text to the model.
4. The model's output references the tokens (safe to log/store).
5. **Re-identify** only inside the trust boundary, for the authorized user.

This bounds exposure: even if a log, cache, or the provider retained the prompt, it
contains tokens, not raw identifiers.

### GDPR data-subject rights across every store

Erasure and export must span **everywhere** a subject's data lands: conversation logs,
the vector/RAG index, and prompt caches. A deletion that misses the vector index is not
compliant. Design a subject-keyed data model so a single erasure fans out to all stores.

### Tamper-evident audit logging

Compliance requires proving *what happened*. A **hash-chained** audit log makes any
after-the-fact edit detectable: each entry stores the hash of the previous entry plus
its own content, so altering entry N changes its hash and breaks every link from N
onward. Verification recomputes the chain and reports the first broken link.

## Architecture

```
  workload profile ---> POLICY ENGINE ---> controls + inference_geo + platform + retention
  (data types,                              (HIPAA/GDPR/FedRAMP/PCI rules)
   region, gov,
   retention, zdr)

  TRUST BOUNDARY
  +-----------------------------------------------------------------+
  | raw PHI/PII --de-identify--> tokens --+                         |
  |      ^                                 |                         |
  |      | reidentify                      v                         |
  |   vault (token<->value, never leaves) [de-identified text] -----|---> Claude (shared client)
  |      ^                                                           |         |
  |      +------------------- tokens in output <--------------------|---------+
  +-----------------------------------------------------------------+

  GDPR DSR:  erase/export subject  -->  { conversation_logs, vector_index, prompt_cache }
  AUDIT:     entry.hash = sha256(seq + action + detail + prev_hash)   (chain -> tamper-evident)
```

## What the code demonstrates

| Step | Code (Python / TypeScript) |
|---|---|
| Workload -> controls | `evaluate_policy` / `evaluatePolicy` |
| Platform + residency recommendation | `recommend_platform`, `inference_geo_for` |
| Detectors + reversible vault | `TokenVault`, `de_identify` / `deIdentify` |
| Model call on de-identified text + re-identify | part (b) in `main` |
| GDPR erasure + export across stores | `DataStores` |
| Hash-chained audit log + tamper detection | `AuditChain` |

## Run it

From the repo root:

```bash
.venv/Scripts/python "D5.4 - Regulatory Compliance - GDPR, HIPAA, FedRAMP/python/main.py"
node "D5.4 - Regulatory Compliance - GDPR, HIPAA, FedRAMP/typescript/main.ts"
```

Offline, deterministic, no API key. All data is obviously fictional.

## Exam tips & traps

- **De-identify before the boundary, re-identify inside it.** The vault map must never
  leave the trust boundary; only tokens go to the model.
- **`inference_geo` is first-party / Claude Platform on AWS only.** On Bedrock / Vertex
  / Foundry, residency is the platform region - don't claim `inference_geo` works there.
- **ZDR is authorization-gated**, and **Fable 5.1 cannot run under ZDR** (30-day
  retention required) - a ZDR org 400s on those models.
- **GDPR erasure must reach the vector index and caches**, not just the chat log.
- **Audit logs must be tamper-evident** (hash chaining / WORM), not just append-only in
  a mutable table.
- **Never assert a specific FedRAMP/HIPAA authorization for a specific platform from
  memory** - phrase as "verify current authorization/BAA with the provider."
- **Minimum necessary / data minimization** is itself a control - send the least data
  the task needs, which also shrinks the compliance surface.

### Practice questions

**Q1.** A healthcare workload must process PHI with Claude and operate under a **zero
data retention** arrangement. Which statement is correct?

- A. Use Fable 5.1 under ZDR for the strongest reasoning.
- B. ZDR is on by default, so no action is needed.
- C. ZDR requires express Anthropic authorization, and Fable 5.1 cannot run under ZDR (it requires 30-day retention and 400s for ZDR orgs) - pick a ZDR-eligible model and confirm a signed BAA.
- D. PHI can be sent freely as long as the response is deleted.

**Answer: C.** ZDR is authorization-gated and Fable 5.1 is excluded from it; a PHI
workload also needs a BAA and should de-identify before sending.

**Q2.** An EU-facing assistant must keep inference in the EU. The team deploys on Amazon
Bedrock and sets `inference_geo="eu"` on each request. What happens?

- A. Inference is pinned to the EU as intended.
- B. `inference_geo` is a first-party / Claude Platform on AWS feature and is not available on Bedrock; EU residency must come from deploying in an EU Bedrock region instead.
- C. It raises a 529 overloaded error.
- D. It silently falls back to a US region but is still compliant.

**Answer: B.** `inference_geo` is unavailable on Bedrock/Vertex/Foundry; residency on
those platforms is controlled by the deployment region, not the request parameter.

## Further reading

- Anthropic Trust Center / data-usage & privacy documentation
- Anthropic docs: "Data residency" (`inference_geo`) and retention / ZDR
- Anthropic docs: platform availability (first-party vs Bedrock/Vertex/Foundry)
- HHS HIPAA Security Rule; EU GDPR (Arts. 5, 15-20, 32); FedRAMP authorization process
- NIST SP 800-53 / 800-171 (control families for FedRAMP / CUI)
