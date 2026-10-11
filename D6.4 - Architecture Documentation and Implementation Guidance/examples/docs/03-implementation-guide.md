# Implementation guide - Claims Correspondence Copilot v1.5.0

## 1. Environments

| Environment | Purpose | Data | Model access |
|---|---|---|---|
| dev | Feature work with mocked ClaimCore | Synthetic claims only | Shared dev key, low rate limits |
| staging | Integration tests and eval gate | Masked production sample | Staging key |
| prod | Adjuster traffic | Live claims (US region) | Production key, org rate limits |

## 2. Configuration and secrets

| Key | Secret | Source | Default |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Vault: kv/ccc/anthropic | - |
| `CLAIMCORE_CLIENT_SECRET` | yes | Vault: kv/ccc/claimcore | - |
| `DRAFT_MODEL` | no | env | claude-sonnet-5-5 |
| `DRAFT_EFFORT` | no | env | low |
| `PROMPT_VERSION_PIN` | no | env | 1.4.1 |
| `JUDGE_SAMPLE_RATE` | no | env | 0.05 |
| `FALLBACK_MODE` | no | feature flag | auto |

## 3. Models and Claude API conventions

| Id | Model | Purpose | Effort | Thinking | max_tokens | Fallback |
|---|---|---|---|---|---:|---|
| m_draft | `claude-sonnet-5-5` | Draft letters from claim context and cached policy wording | low | adaptive | 1200 | Template mode (no AI draft) |
| m_guard | `claude-haiku-5-5` | Prohibited-content classification (coverage and liability language) | low | adaptive | 300 | Block draft and route to manual review |
| m_judge | `claude-haiku-5-5` | Sampled quality judging against the rubric | medium | adaptive | 600 | Defer judging (batch retry next day) |

- Model ids come from configuration (`DRAFT_MODEL`), never hard-coded; no date suffixes on current aliases.
- Thinking: `{"type": "adaptive"}`; control depth with `output_config.effort` per route (drafting `low`, judging `medium`).
- No `temperature`/`top_p`/`top_k` and no assistant prefill on current models - use structured outputs (`output_config.format`) for machine-readable results.
- Tools: `tool_choice` auto with `strict: true` schemas; read-only tools for the agent, writes only through the API layer.
- Stream user-facing drafts (time to first token matters more than total latency for adjusters).
- Prompt caching: keep the system prompt and policy prefix byte-stable (no timestamps or request ids); claim context goes after the cache breakpoint.
- Bulk and evaluation work uses the Message Batches API (asynchronous, lower cost).
- Log the response `request-id`, model, prompt version and token usage with every draft (audit and cost attribution).

## 4. Tools and least privilege

| Tool | Component | Side effect | Auth scope | Human approval / write mode |
|---|---|---|---|---|
| get_claim | drafting_agent | read | `claimcore:claims.read` | n/a (read) |
| get_claim_notes | drafting_agent | read | `claimcore:notes.read` | n/a (read) |
| get_policy_wording | drafting_agent | read | `docvault:policies.read` | n/a (read) |
| save_draft | drafting_api | write | `claimcore:correspondence.write_pending` | writes pending only |

## 5. Prompt management

| Prompt | Version | Owner | Component | Path | Eval suite |
|---|---|---|---|---|---|
| p_draft_status | 1.4.1 | Claims Ops content lead | drafting_agent | `prompts/draft_status.md` | `evals/status_letters.jsonl` |
| p_draft_payment | 1.4.1 | Claims Ops content lead | drafting_agent | `prompts/draft_payment.md` | `evals/payment_letters.jsonl` |
| p_guard | 0.9.0 | Compliance | guardrails | `prompts/guard_classifier.md` | `evals/prohibited_content.jsonl` |
| p_judge | 1.1.0 | AI Quality | eval_runner | `prompts/judge_rubric.md` | `evals/judge_calibration.jsonl` |

- Prompts live in the prompt registry (Git, signed tags) and are versioned with semver: major = behaviour change, minor = wording, patch = typo.
- Production pins a version (`PROMPT_VERSION_PIN`); no prompt change reaches production without passing its eval suite in staging.
- Every change runs shadow, then canary with latency, cost and quality gates (see rollout plan); rollback = re-pin the previous version.
- The prompt owner approves wording changes; the AI Delivery Team approves structural and model changes.
- Record the prompt version and model id with every draft in the audit store.

## 6. Integrations

| System | Protocol | Auth | Direction | Limits |
|---|---|---|---|---|
| ClaimCore | REST over HTTPS | OAuth2 client credentials | read/write (pending drafts only) | 50 rps shared with other apps |
| DocVault | S3 API | IAM role (read-only) | read | - |
| Okta | OIDC | Authorization code + PKCE | read | - |
| Claude API | HTTPS (Messages API, Message Batches API) | API key from Vault | read/write | Org rate limits per model |

## 7. Guardrails

| Id | Type | Control | Implemented in |
|---|---|---|---|
| G1 | input | Claim-scoped context only: no data from other claims or customers | context_builder |
| G2 | output | Prohibited-content check: no coverage determinations or liability admissions | guardrails |
| G3 | output | Field validator: claim number, dates and amounts must match ClaimCore | drafting_api |
| G4 | process | Licensed adjuster approves every letter; sending stays in PostMaster workflow | workspace_ui |
| G5 | data | PII masking in application logs | drafting_api |

## 8. Error handling

| Condition | Detection | Handling |
|---|---|---|
| 429 rate_limit_error | HTTP 429 | SDK retries with exponential backoff (max_retries); shed batch load first; alert on sustained 429s |
| 529 overloaded_error / 5xx | HTTP 529 / 500-504 | Retry with jitter; after repeated failures switch to template fallback mode |
| 400 invalid_request_error | HTTP 400 | Do not retry; log request id; a spike after a deploy usually means an unsupported parameter |
| stop_reason = max_tokens | Response field | Never show a truncated draft; raise the per-letter-type max_tokens or ask for a shorter letter |
| stop_reason = refusal | Response field | Show 'manual drafting needed', log the category; do not auto-retry with a reworded prompt |
| Structured output does not parse | Parse error | Treat as a failed attempt; retry once, then fall back |
| Guardrail block (G2) | Guardrail service | Do not display the draft; route to manual drafting with the reason |
| Field validator mismatch (G3) | Drafting API | Block the draft and highlight the mismatched fields against ClaimCore |
| ClaimCore timeout / 5xx | Circuit breaker | Open the breaker, switch to template mode, page via AL-6 |

## 9. Rollout plan

| Phase | Traffic | Entry criteria | Exit criteria | Rollback trigger |
|---|---|---|---|---|
| Shadow | 0% (drafts generated, not shown) | Eval suite pass rate >= 90% on the golden set | 1 week with no guardrail misses on audited drafts | n/a (not user-visible) |
| Canary | 1 adjuster team (about 5%) | Shadow exit met; on-call briefed | 2 weeks: quality lower bound >= 88%, p95 <= 8 s | Quality lower bound < 85% or p95 > 10 s for 2 h -> flag off |
| Ramp | 25% then 50% | Canary exit met; sponsor sign-off | 2 weeks within all SLOs | Any SLO burn page -> previous traffic level |
| General availability | 100% | Ramp exit met; operations acceptance signed | Steady state; monthly service review | Prompt or model pin to last good version |

## 10. Observability

| SLO | Target | Alert |
|---|---|---|
| SLO-1 Availability | >= 99.5% of valid requests succeed (business hours, 30 days) | AL-1 |
| SLO-2 Latency | >= 95% of drafts complete within 8 s | AL-2 |
| SLO-3 Draft quality | >= 90% of sampled drafts score >= 4/5 (95% CI reported) | AL-3 |
| SLO-4 Cost per completed letter | <= $0.06 | AL-4 |

Dashboards: SLO overview (availability, latency, quality); Cost per completed letter; Guardrail blocks and refusals; Prompt version canary comparison.
