# ADR-0007: Ground drafted letters with long-context prompting and prompt caching on Claude Sonnet 5.5

## Status

Proposed - 2026-10-08. Awaiting approval by the deciders below; becomes Accepted when signed off.

- **Deciders:** Dana Whitfield (VP Claims Operations (sponsor)), Priya Raman (Head of Compliance / DPO), Marcus Oyelaran (Director of Claims Platform)
- **Related requirements:** FR-01 (draft letters), NFR-10 (p95 <= 10 s), NFR-13 (AI spend <= $8,000/month), NFR-03 (US data residency)

## Context

The Claims Correspondence Copilot drafts about 9,000 letters per month (12,000 drafting requests including regenerations), and every draft must reflect the customer's actual policy wording. Adjusters approve every letter, so the system must make that review fast and trustworthy. Must-meet constraints from discovery are a p95 full-draft latency of at most 10 s, AI run cost of at most $8,000 per month, and US data residency. Four grounding approaches ran the same 200-letter golden set: hybrid RAG (A), long context with prompt caching (B), agentic search on Sonnet 5.5 (C) and agentic search on Opus 5.5 (D).

**Decision drivers**

- Fastest time to value: 2 weeks to production readiness versus 5-6 weeks for the alternatives.
- Lowest run cost among eligible options: $454 per month, well inside the $8,000 cap.
- Simplest operations: no index pipeline or vector store to run and patch.
- Fewer copies of sensitive content than RAG, which would add an embedding index of policy and claim text.

## Decision

We will ground each draft by loading the full policy wording for the claim's product as a cached prompt prefix and generating with Claude Sonnet 5.5 (Option B: long context + prompt caching). Claim context is appended after the cached prefix so the cache is reused across letters for the same product and policy version. Agentic search (Option C) remains the documented fallback if citation accuracy becomes the dominant concern.

## Consequences

**Positive**

- Production-ready in about 2 weeks and $301 per month (39.9%) cheaper than agentic search on Sonnet.
- No index or vector store to operate; policy updates reach drafts at the next cache build.
- Budget headroom of $7,546 per month for growth and evaluation traffic.

**Negative (accepted trade-offs)**

- Citation accuracy is 7.0 pp lower than agentic search, so cited clauses need a code-side check against the policy text.
- Answer quality is 1.0 pp lower than Option C.
- Large cached prompts make latency sensitive to cache misses: p95 is 8.9 s against a 10 s ceiling.
- A product whose policy wording outgrows a single prompt would force a redesign.

## Considered options

| Option | Weighted score (0-10) | Must-meet | Outcome |
|---|---:|---|---|
| A - Hybrid RAG + Claude Sonnet 5.5 | 5.10 | PASS | rejected: lowest weighted score (5.10) and the highest operational load |
| B - Long context + prompt caching (Sonnet 5.5) | 6.56 | PASS | chosen |
| C - Agentic search (Sonnet 5.5 + document tools) | 5.95 | PASS | rejected: weighted score 5.95; preferred by Compliance for auditability, kept as the fallback |
| D - Agentic search (Opus 5.5 + document tools) | 5.83 | FAIL: MM-1 (16 vs <= 10) | rejected: fails must-meet MM-1 (latency); candidate for the overnight backlog batch |

**A - Hybrid RAG + Claude Sonnet 5.5**

- Pros: Fastest at p95 (6.8 s); Strong citation accuracy (94%)
- Cons: Index pipeline and vector store to operate; Creates an additional copy of policy and claim content; Policy updates take up to an hour to appear

**B - Long context + prompt caching (Sonnet 5.5)**

- Pros: Best time to value (2 weeks); Lowest monthly cost ($454); Fewest moving parts
- Cons: Lowest citation accuracy of the eligible options (88%); Thin latency headroom against the SLO (8.9 s p95)

**C - Agentic search (Sonnet 5.5 + document tools)**

- Pros: Highest eligible quality (92%) and citation accuracy (95%); Always-fresh content
- Cons: Highest eligible cost ($755 per month); p95 of 9.4 s leaves little SLO headroom; Tool servers and multi-turn loops to operate

**D - Agentic search (Opus 5.5 + document tools)**

- Pros: Best quality (94.5%) and citation accuracy (96%)
- Cons: p95 of 16 s breaks the 10 s SLO; Most expensive option ($1,355 per month)

## Trade-offs (quantified, computed from the decision matrix)

- B over C: saves $301/month (-39.9%); 0.5 s faster at p95; 1.0 pp lower answer quality; 7.0 pp lower citation accuracy; ships 3 weeks sooner.
- B over A: saves $142/month (-23.8%); 2.1 s slower at p95; 2.5 pp higher answer quality; 6.0 pp lower citation accuracy; ships 4 weeks sooner.
- B over D: saves $901/month (-66.5%); 7.1 s faster at p95; 3.5 pp lower answer quality; 8.0 pp lower citation accuracy; ships 3 weeks sooner. (D is not eligible: fails MM-1 (16 vs <= 10))
- Budget: $454 of the $8,000/month cap ($7,546 headroom), about $0.050 per letter.

## Sensitivity

- Dana Whitfield's weights -> Option B (6.59)
- Priya Raman's weights -> Option C (7.20)
- Marcus Oyelaran's weights -> Option B (6.74)
- Monte Carlo (5,000 weight perturbations of +/-50%): A wins 0.0%, B wins 99.5%, C wins 0.5%
- Break-even: the auditability weight would need to rise x2.25 (14.0 -> 31.5 points) for Option C to win.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cache misses after prompt or policy changes push p95 above 10 s | Keep the policy prefix byte-stable, pre-warm the cache per product, alert on p95 and on the cache-read ratio |
| Wrong clause cited in a draft | Validate each cited clause against the policy text in code; adjuster approval stays mandatory |
| Compliance prefers agentic search for auditability | Review citation accuracy monthly with Compliance; the auditability weight would need to rise x2.25 to change the decision |

## Review triggers (reopen this ADR if...)

- Citation accuracy below 88% for two consecutive weeks
- p95 full-draft latency above 10 s in any week
- A product's policy wording no longer fits in one prompt
- A new Claude model or a price change alters the cost or latency comparison
- Compliance raises the auditability weight toward the break-even (x2.25)

---
_Drafted by Claude from the decision matrix (structured output). Status, scores, trade-offs and sensitivity are computed by code; 34 figures in Claude's text were checked against the matrix (after 1 repair round(s): 0 ungrounded)._
