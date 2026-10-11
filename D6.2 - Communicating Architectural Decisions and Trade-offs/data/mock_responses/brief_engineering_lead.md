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
