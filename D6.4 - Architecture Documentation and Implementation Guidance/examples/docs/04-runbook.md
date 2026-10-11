# Operational runbook - Claims Correspondence Copilot

**On-call:** AI Delivery Team (primary), Claims Platform (secondary)

Flow for every alert: acknowledge -> diagnose (steps below) -> remediate with the safest reversible action -> escalate if not mitigated in 30 minutes -> postmortem if the error budget impact is above 20%.

| Alert | Severity | Component | Owner |
|---|---|---|---|
| AL-1 Availability fast burn | PAGE | drafting_api | AI Delivery Team |
| AL-2 Latency SLO burn | PAGE | drafting_agent | AI Delivery Team |
| AL-3 Quality at risk | TICKET | eval_runner | AI Quality |
| AL-5 Guardrail block spike | PAGE | guardrails | AI Delivery Team |
| AL-6 ClaimCore dependency errors | PAGE | context_builder | AI Delivery Team |
| AL-4 Cost per letter drift | TICKET | drafting_agent | AI Delivery Team |

### AL-1 - Availability fast burn (PAGE)

- **Condition:** Burn rate >= 14.4 over 1 h (min 30 requests and 3 failures)
- **Component / owner:** drafting_api / AI Delivery Team
- **Protects:** SLO-1 Availability (>= 99.5% of valid requests succeed (business hours, 30 days))

**Likely causes**

- ClaimCore degradation or timeouts (check AL-6)
- Claude API errors: 429 rate limits, 529 overloaded, 5xx
- Bad deploy of the drafting API or a new prompt version

**Diagnosis**

1. Open the SLO overview dashboard and identify the failing status codes
2. Check whether AL-6 is also firing (dependency problem)
3. Search logs by Claude request id for error types (rate_limit_error, overloaded_error, api_error)
4. Compare the start time with the last deploy and prompt version change

**Remediation**

1. Dependency: confirm template fallback mode is active and notify Claims Platform
2. Claude 429/529: pause the backlog batch drafter and let SDK retries with backoff absorb bursts
3. Deploy regression: roll back to the last good release or prompt pin

**Escalation:** AI Delivery Team on-call; Claims Platform for ClaimCore; vendor support with request ids for sustained API errors

### AL-2 - Latency SLO burn (PAGE)

- **Condition:** Share of drafts <= 8 s below 90% over 1 h
- **Component / owner:** drafting_agent / AI Delivery Team
- **Protects:** SLO-2 Latency (>= 95% of drafts complete within 8 s)

**Likely causes**

- Prompt change producing longer drafts
- Cache misses on the policy prefix (prefix no longer byte-stable)
- Higher effort setting or model change

**Diagnosis**

1. Compare output tokens per draft by prompt version
2. Check the cache-read ratio on the dashboard
3. Compare DRAFT_MODEL and DRAFT_EFFORT with the pinned values

**Remediation**

1. Roll back the prompt version if output length jumped
2. Restore a byte-stable prefix (no timestamps, fixed section order) and pre-warm the cache
3. Revert effort or model changes

**Escalation:** AI Delivery Team on-call

### AL-3 - Quality at risk (TICKET)

- **Condition:** Weekly judge pass-rate lower 95% bound below 85%
- **Component / owner:** eval_runner / AI Quality
- **Protects:** SLO-3 Draft quality (>= 90% of sampled drafts score >= 4/5 (95% CI reported))

**Likely causes**

- Recent prompt or model change
- Shift in letter mix or claim data
- Judge drift

**Diagnosis**

1. Break the judge pass rate down by letter type and prompt version
2. Read 10 failing samples next to their rubric scores
3. Check the latest judge calibration agreement

**Remediation**

1. Roll back the offending prompt version
2. Add the failing cases to the eval suite before fixing the prompt
3. Re-calibrate the judge if agreement fell below 85%

**Escalation:** AI Quality lead; Claims Ops content lead for prompt changes

### AL-5 - Guardrail block spike (PAGE)

- **Condition:** Prohibited-content blocks above 2% of drafts over 1 h
- **Component / owner:** guardrails / AI Delivery Team

**Likely causes**

- Drafting prompt change introducing coverage-like phrasing
- Classifier prompt change (p_guard) over-blocking
- Unusual letter mix, such as many near-denial letters

**Diagnosis**

1. Sample 20 blocked drafts and label true vs false positives
2. Check recent p_draft_* and p_guard version changes
3. Compare the block rate by letter type

**Remediation**

1. True positives: roll back the drafting prompt
2. False positives: roll back p_guard and add the cases to its eval suite
3. Never disable the guardrail to restore throughput

**Escalation:** Compliance (notify within 4 hours) and AI Delivery Team

### AL-6 - ClaimCore dependency errors (PAGE)

- **Condition:** ClaimCore 5xx or timeouts above 5% over 10 min
- **Component / owner:** context_builder / AI Delivery Team

**Likely causes**

- ClaimCore outage or degradation
- Shared ClaimCore rate limit exhausted (50 rps)
- Expired OAuth2 client secret

**Diagnosis**

1. Check ClaimCore status and error codes (5xx vs 429 vs 401)
2. Compare request rate with the shared 50 rps limit
3. Verify the client secret in Vault

**Remediation**

1. Confirm the circuit breaker opened and template mode is active
2. Throttle the batch drafter
3. Rotate the secret if authentication fails

**Escalation:** Claims Platform on-call (ClaimCore owner)

### AL-4 - Cost per letter drift (TICKET)

- **Condition:** Daily cost per completed letter above $0.05 for 2 consecutive days
- **Component / owner:** drafting_agent / AI Delivery Team
- **Protects:** SLO-4 Cost per completed letter (<= $0.06)

**Likely causes**

- Lower cache hit rate
- Longer outputs or more regenerations per letter
- Model or effort change

**Diagnosis**

1. Split spend into cache reads, cache writes, uncached input and output
2. Check regenerations per completed letter
3. Confirm the pinned model and effort

**Remediation**

1. Fix prefix stability to restore cache reads
2. Tighten output length budgets per letter type
3. Move eligible backlog volume to the batch drafter

**Escalation:** AI Delivery Team (product owner informed)

_Diagnosis and remediation drafted by Claude from the spec; alert metadata, owners and SLO links rendered from the spec._
