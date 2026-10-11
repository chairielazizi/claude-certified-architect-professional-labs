# Stakeholder briefs - ADR-0007 (Option B: Long context + prompt caching (Sonnet 5.5))

One decision, three framings. Each brief came from the same prompt template parameterized by an audience profile (`data/audiences.json`) and was checked by code before release.

## For: Executive sponsor / steering committee

**Recommendation: approve Option B - long-context grounding with Claude Sonnet 5.5.**

- **Value:** fastest route to the handling-time goal - production-ready in 2 weeks, 3-4 weeks sooner than the alternatives.
- **Cost:** $454 per month (about $0.050 per letter), leaving $7,546 of headroom under the $8,000 budget. Letters for the same product reuse the loaded policy, which keeps each letter cheap.
- **Quality:** 91% of pilot letters passed the quality rubric, and an adjuster still approves every letter.
- **Risk:** citations are less precise than the agentic option (88% vs 95%). We add an automated clause check and review it monthly with Compliance.
- **Robustness:** the recommendation held in 99.5% of 5,000 what-if weightings; only a much stronger emphasis on auditability would change it.

**Decision needed:** approve Option B so the build can start this sprint.

> Checks (after 1 revision(s)): 129/160 words; Flesch reading ease 53.4; 10 figures checked against the matrix; required topics, jargon and grounding all passed.

## For: Engineering lead / platform team

**Decision:** Option B - long-context grounding with prompt caching on Claude Sonnet 5.5 (ADR-0007, Proposed).

**Design**
- Request layout: stable system prompt, then the full policy wording for the claim's product as the cached prefix, then claim context and the drafting instruction after the cache breakpoint.
- Measured profile: $0.0378 per request and $454 per month at the planned volume; most input is billed at the cache-read rate.
- Keep the prefix byte-stable: no timestamps or request IDs in the system prompt, deterministic ordering of policy sections, one cache entry per product and policy version.

**Interfaces**
- Read: ClaimCore REST (claim data, notes) and DocVault (policy wording by product and version).
- Write: the draft goes back to ClaimCore as pending correspondence. No call to PostMaster.

**Ops and failure modes**
- Latency: p95 8.9 s in the pilot against the 10 s SLO, so headroom is thin. Stream the draft, pre-warm the cache per product each morning, and alert on p95 and on the cache-read ratio.
- Fallback: if p95 breaches for a product, route that product to Option C (agentic search, p95 9.4 s). Rollback is a per-product config flag.
- Citations: 88% clause accuracy in the pilot. Add a code check that every cited clause exists in the policy text and block drafts that fail it.
- Evals: the 200-letter golden set runs in CI on every prompt or policy change; regressions block the release.

**Next steps**
1. Prompt layout and cache breakpoints.
2. Clause-check validator and eval gate.
3. Canary with one adjuster team, then widen.

> Checks (first draft): 249/320 words; Flesch reading ease 68.0; 6 figures checked against the matrix; required topics, jargon and grounding all passed.

## For: Security, privacy and legal reviewers

**Decision:** Option B - long-context grounding with prompt caching on Claude Sonnet 5.5 (ADR-0007, Proposed).

**Data flow**
1. At request time the copilot reads the claim from ClaimCore and the policy wording for the claim's product from DocVault.
2. Policy wording and claim context are sent to Claude Sonnet 5.5 in a US region; the draft returns to ClaimCore as pending correspondence for adjuster approval.
3. No new index or embedding store is created. Unlike RAG (Option A), there is no extra copy of policy or claim content to secure, classify and delete.

**Controls**
- Residency: all processing stays in a US region, matching the discovery constraint.
- PII: claim context contains personal data. It is sent only in the uncached part of the request and is masked in application logs.
- Prompt cache: holds policy wording only (no customer personal data) and expires automatically; confirm the provider's cache retention terms in the vendor review.
- Access: the service account reads ClaimCore and DocVault with read-only scopes and writes only pending drafts. It cannot send letters.
- Audit: every draft stores the approver, model version, prompt version and policy version for seven years.

**Residual risks**
- Wrong clause cited (88% citation accuracy in the pilot): mitigated by a code-side clause check and mandatory adjuster approval.
- Another customer's data in a draft: mitigated by claim-scoped requests and approval before sending.

**Decisions needed from you**
- Confirm that keeping customer personal data out of the cached prefix satisfies the retention policy.
- Approve the seven-year audit retention design.

> Checks (first draft): 247/280 words; Flesch reading ease 47.8; 1 figures checked against the matrix; required topics, jargon and grounding all passed.
