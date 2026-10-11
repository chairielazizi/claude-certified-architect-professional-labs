# Claims Correspondence Copilot - architecture overview (v1.5.0, production)

**Owner:** AI Delivery Team  
**Summary:** Drafts outbound claim letters for adjusters; every letter is approved by a licensed adjuster before it is sent.

## System overview

The Claims Correspondence Copilot drafts outbound claim letters for Harbor Mutual adjusters inside their existing workspace. For each request it builds a claim-scoped context from ClaimCore and DocVault, drafts with Claude Sonnet 5.5 using the full policy wording as a cached prompt prefix, checks the draft with guardrails and a field validator, records an audit trail, and saves the draft to ClaimCore as pending correspondence. A licensed adjuster edits and approves every letter; sending stays in the existing PostMaster workflow, which the copilot never calls.

## C4 level 1 - context

| Element | Type | Description | Owner |
|---|---|---|---|
| Claims adjuster | Person | Requests, edits and approves drafts | - |
| Team lead | Person | Monitors draft quality for the team | - |
| **Claims Correspondence Copilot** | Software system | Drafts outbound claim letters for adjusters; every letter is approved by a licensed adjuster before it is sent. | AI Delivery Team |
| ClaimCore | External system | Claims system of record (REST API) | Claims Platform |
| DocVault | External system | Policy wording and claim documents (S3) | Claims Platform |
| PostMaster | External system | Outbound correspondence service - NOT called by the copilot (by design) | Claims Platform |
| Okta | External system | Workforce SSO | Corporate IT |
| Claude API | External system | Anthropic Messages API (US region) | Anthropic (vendor) |

## C4 level 2 - containers

| Container | Type | Technology | Responsibility | Owner | Depends on |
|---|---|---|---|---|---|
| Adjuster workspace plug-in (`workspace_ui`) | ui | React micro-frontend | Adds a 'Draft letter' action to the adjuster workspace, streams the draft into an editor, shows guardrail and validator findings, and records the adjuster's edits and approval. | Claims Platform | Drafting API, Okta |
| Drafting API (`drafting_api`) | service | Python, FastAPI | Single entry point for drafting: validates the SSO token, orchestrates context building, drafting, guardrail checks and the field validator, writes the audit record and saves the draft to ClaimCore as pending correspondence. | AI Delivery Team | Context builder, Drafting orchestrator, Guardrail service, Audit store, ClaimCore |
| Context builder (`context_builder`) | service | Python | Assembles a claim-scoped context (claim header, parties, amounts, latest notes from ClaimCore; policy wording by product and version from DocVault), laid out so the policy prefix stays byte-stable for prompt caching. | AI Delivery Team | ClaimCore, DocVault |
| Drafting orchestrator (`drafting_agent`) | agent | Claude API (Sonnet 5.5) with read-only tools | Calls Claude Sonnet 5.5 with the pinned prompt version and read-only tools, streams the draft back, and handles refusals, max_tokens stops and API errors. | AI Delivery Team | Claude API, Prompt registry |
| Guardrail service (`guardrails`) | service | Python rules + Claude Haiku 5.5 classifier | Runs deterministic rules and a Claude Haiku 5.5 classifier over every draft to block coverage determinations, liability admissions and other prohibited content before an adjuster sees it. | AI Delivery Team | Claude API, Prompt registry |
| Eval and judge runner (`eval_runner`) | job | Python, Message Batches API | Samples drafts daily and grades them against the rubric with a Claude Haiku 5.5 judge via the Message Batches API; calibrates the judge monthly against human grades and feeds the quality SLO. | AI Quality | Claude API, Audit store, Eval results |
| Backlog batch drafter (`batch_backlog`) | job | Python, Message Batches API | Drafts backlog letters overnight through the Message Batches API at lower unit cost, reusing the context builder and guardrails; drafts land in ClaimCore as pending correspondence for approval. | AI Delivery Team | Context builder, Guardrail service, Audit store, Claude API |

## Data stores

| Store | Kind | Classification | PII | Retention | Encryption | Residency | Owner |
|---|---|---|---|---|---|---|---|
| Audit store | PostgreSQL (append-only) | confidential | yes | 7 years | AES-256 at rest, TLS 1.2+ in transit | US (us-east) | AI Delivery Team |
| Prompt registry | Git repository with signed tags | internal | no | Indefinite | Repository default | US | AI Delivery Team |
| Eval results | S3 bucket | confidential | yes | 13 months | SSE-KMS | US (us-east) | AI Quality |

## Key design decisions

- **Ground drafts with long-context prompting and prompt caching** (ADR-0007): Simplest operations and lowest cost among the options that meet the latency constraint; citation risk is mitigated by the field validator and clause checks.
- **No send capability in the copilot**: Regulatory constraint from discovery: a licensed adjuster approves every letter, so the copilot writes only pending drafts and never calls PostMaster.
- **Small models for guardrails and judging**: Classification and rubric grading are high-volume, well-specified tasks; Claude Haiku 5.5 keeps them cheap enough to run on every draft or sample.
- **Template fallback mode**: When ClaimCore or the Claude API is degraded, adjusters get the approved template instead of an error, protecting user-facing availability.

## Operational notes

- Prompt and model versions are pinned in configuration; every change ships through shadow and canary stages with eval, latency and cost gates.
- Log the Claude request id with every draft so incidents can be traced end to end.
- Watch the cache-read ratio: a drop usually means the policy prefix stopped being byte-stable.

_Prose drafted by Claude from the spec (structured output); tables rendered from `architecture_spec.json`._
