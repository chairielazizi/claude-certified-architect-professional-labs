# Monthly Service Review - Claims Correspondence Copilot - April 2027

**Provider:** AI Delivery Team (managed service) | **Customer:** Harbor Mutual Claims Operations (fictional)

## 1. Executive summary

April missed the availability SLO (99.48% against 99.5%) and used 105% of the error budget, mostly because of a two-hour ClaimCore degradation on Apr 13 (INC-0413, 70% of the budget). The contractual SLA held at 99.83% because outages of customer-owned dependencies are excluded, so no service credit is due ($0.00). A prompt change (v1.4.0, Apr 15-19) made drafts longer and slower and pushed rejections and escalations to 15.4% that week; it was rolled back in v1.4.1. Draft quality is estimated at 91.8% of sampled drafts meeting the rubric, but the 95% confidence interval (89.0%-93.9%) dips below the 90.0% target, so we report it as at risk rather than met.

**Highlights**

- Latency held: 97.8% of drafts completed within 8 s (target 95.0%), with a p95 of 7.17 s.
- Cost per completed letter was $0.0437, well inside the $0.06 target.
- v1.5.0 added a ClaimCore circuit breaker and a template fallback mode, closing the outage feedback.
- 8 of 20 actionable feedback items were closed within the month (40%).

**Concerns**

- Error budget exhausted (105%): the feature freeze applies until 30-day availability is back above 99.5%.
- Prompt v1.4.0 shipped without a latency and tone gate: the week of Apr 15-21 had a p95 of 8.24 s and 15.4% rejections and escalations.
- Two compliance-sensitive reports (a draft implying coverage) need a prohibited-content guard, now P1 (BL-25).
- The judge flagged factual errors in 10 of 499 sampled drafts (2.0%); a field validator against ClaimCore is P1 (BL-21).

## 2. SLO and SLA scorecard

| SLO | Objective | Actual | Target | Status |
|---|---|---|---|---|
| SLO-1 | Availability | 99.48% | 99.5% | **MISSED** |
| SLO-2 | Drafts <= 8 s | 97.8% | 95.0% | **MET** |
| SLO-3 | Quality (judge >= 4/5) | 91.8% [89.0%-93.9%] n=499 | 90.0% | **AT RISK** |
| SLO-4 | Cost per completed letter | $0.0437 | <= $0.06 | **MET** |
| SLO-5 | Rejection + escalation | 12.7% | <= 15% | **MET** |

| SLA term | Actual | Commitment | Result |
|---|---|---|---|
| Availability (excl. 34 dependency-caused failures) | 99.83% | >= 99.0% | met |
| Monthly p95 full-draft latency | 7.17 s | <= 12 s | met |
| **Service credit** | 0% | fee $18,000 | **$0.00** |

SLA exclusions: 4xx client errors (invalid requests); Scheduled maintenance announced at least 72 hours ahead; Outages of customer-owned dependencies (ClaimCore) declared by Harbor Mutual IT.

## 3. Error budget and incidents

- Budget: 48.8 failed requests allowed (0.5% of 9,756 valid); 51 failed -> **105% consumed**.
- INC-0413 (Apr 13, 10:00 for 2 h): ClaimCore API degradation (customer-owned dependency); drafts timed out; used 70% of the budget.

| Alert | Rule | Window | Duration | Peak burn |
|---|---|---|---|---:|
| PAGE | fast burn | Apr 13 10:00 -> Apr 13 11:00 | 1 h | 100.0x |
| PAGE | medium burn | Apr 13 10:00 -> Apr 13 12:00 | 2 h | 37.0x |
| TICKET | slow burn | Apr 13 10:00 -> Apr 13 17:00 | 7 h | 5.3x |
| TICKET | slow burn | Apr 14 11:00 -> Apr 14 17:00 | 6 h | 5.3x |
| TICKET | slow burn | Apr 16 09:00 -> Apr 16 15:00 | 6 h | 4.2x |
| TICKET | slow burn | Apr 16 18:00 -> Apr 17 10:00 | 6 h | 3.8x |

Error-budget policy actions triggered:

- **EBP-1** Feature freeze: only reliability and security changes ship until 30-day availability is back above the SLO.
- **EBP-2** Blameless postmortem within 5 business days for any incident that consumed more than 20% of the monthly budget.
- **EBP-3** Double the judge sample next month and run every prompt change against the eval suite before release.

## 4. Weekly trend

| Week | Availability | Drafts <= 8 s | p95 | Quality [95% CI] (n) | Rejected + escalated | $/letter |
|---|---:|---:|---:|---|---:|---:|
| Apr 1-7 | 99.74% | 98.8% | 6.73 s | 91.5% [85.0%-95.3%] (117) | 11.9% | $0.0429 |
| Apr 8-14 | 98.24% | 99.1% | 6.76 s | 92.2% [85.9%-95.9%] (116) | 12.0% | $0.0429 |
| Apr 15-21 | 99.86% | 93.9% | 8.24 s | 89.7% [82.5%-94.2%] (107) | 15.4% | $0.0464 |
| Apr 22-28 | 99.87% | 98.6% | 6.88 s | 92.2% [85.8%-95.8%] (115) | 12.2% | $0.0431 |
| Apr 29-30 | 100.00% | 99.3% | 6.54 s | 95.5% [84.9%-98.7%] (44) | 11.5% | $0.0427 |

Weekly quality samples are below the 400-sample minimum, so weekly quality is indicative only; the monthly figure is the one we commit to.

## 5. How quality is measured (and how to read it)

- SLI: share of sampled drafts scoring >= 4/5 on the agreed rubric (calibrated LLM judge).
- Judge: claude-haiku-5-5 on a 5% random sample. Calibrated monthly against 50 human-graded drafts; April agreement 88% (target >= 85%).
- Status rule: MET if the lower bound of the 95% Wilson interval is >= target; AT RISK if only the point estimate is; MISSED otherwise.
- Quality figures are estimates from a random sample, not a guarantee for every letter: with 499 judged drafts, the true pass rate is likely between 89.0% and 93.9%.
- Wording varies between regenerations of the same draft; adjuster approval remains the control for every letter.
- A small share of drafts will contain factual slips (2.0% flagged this month); the planned field validator reduces this risk but cannot eliminate it.

## 6. Stakeholder feedback

| Theme | Items | Severity sum | Corroborated by telemetry | Open |
|---|---:|---:|---|---:|
| availability | 3 | 9 | 3/3 | 0 |
| latency | 3 | 6 | 3/3 | 0 |
| verbosity_tone | 3 | 7 | 2/3 | 1 |
| factual_error | 3 | 10 | 2/3 | 3 |
| missing_context | 2 | 6 | n/a (no telemetry signal) | 2 |
| policy_risk | 2 | 8 | n/a (no telemetry signal) | 2 |
| formatting | 2 | 3 | n/a (no telemetry signal) | 2 |
| feature_request | 2 | 4 | n/a (no telemetry signal) | 2 |
| praise | 4 | 4 | n/a (no telemetry signal) | 0 |

| Feedback | Day | Theme | Sev | Summary | Loop status |
|---|---|---|---:|---|---|
| FB-01 | Apr 2 | praise | 1 | Acknowledgement draft was accurate and saved time | acknowledged |
| FB-02 | Apr 6 | factual_error | 3 | Wrong incident date in draft | open BL-21 |
| FB-03 | Apr 9 | missing_context | 3 | Latest adjuster note ignored on claims with many notes | open BL-24 |
| FB-04 | Apr 13 | availability | 3 | Two-hour drafting outage; manual fallback | closed v1.5.0 (Apr 27) |
| FB-05 | Apr 13 | availability | 2 | Draft unavailable errors | closed v1.5.0 (Apr 27) |
| FB-06 | Apr 13 | availability | 4 | Sponsor escalation about outage impact before renewal season | closed v1.5.0 (Apr 27) |
| FB-07 | Apr 15 | verbosity_tone | 2 | Status updates became much longer | closed v1.4.1 (Apr 19) |
| FB-08 | Apr 15 | latency | 2 | Generation noticeably slower | closed v1.4.1 (Apr 19) |
| FB-09 | Apr 16 | verbosity_tone | 3 | Chatty tone doubled rejections on near-denial letters | closed v1.4.1 (Apr 19) |
| FB-10 | Apr 16 | latency | 2 | Slow generation for a short acknowledgement | closed v1.4.1 (Apr 19) |
| FB-11 | Apr 16 | factual_error | 4 | Wrong deductible amount in draft | open BL-21 |
| FB-12 | Apr 19 | praise | 1 | Letters shorter again after the fix | acknowledged |
| FB-13 | Apr 19 | latency | 2 | Still slow in the morning | closed v1.4.1 (Apr 19) |
| FB-14 | Apr 20 | policy_risk | 4 | Draft implied coverage before the coverage review | open BL-25 |
| FB-15 | Apr 21 | formatting | 1 | PDF export shows asterisks instead of bullets | open BL-28 |
| FB-16 | Apr 22 | feature_request | 2 | Request for Spanish-language letters | open BL-27 |
| FB-17 | Apr 22 | missing_context | 3 | Old mailing address used despite a recent note | open BL-24 |
| FB-18 | Apr 23 | praise | 1 | Team lead reports good quality and lower handling time | acknowledged |
| FB-19 | Apr 26 | factual_error | 3 | Payment amount rounded in draft | open BL-21 |
| FB-20 | Apr 27 | feature_request | 2 | Request for bulk backlog mode | open BL-27 |
| FB-21 | Apr 28 | verbosity_tone | 2 | Robotic tone in condolence-type letters | open BL-22 |
| FB-22 | Apr 28 | policy_risk | 4 | Compliance asks for prevention of coverage-language incidents | open BL-25 |
| FB-23 | Apr 29 | praise | 1 | Template fallback mode worked during a ClaimCore blip | acknowledged |
| FB-24 | Apr 30 | formatting | 2 | Wrong branch phone number in letter header | open BL-28 |

## 7. What changed this month - you said, we did

- **v1.4.0** (Apr 15, prompt change): Warmer tone and more explanation of next steps in every letter type.
- **v1.4.1** (Apr 19, rollback + fix): Rolled back the v1.4.0 wording for status and payment letters, added a per-letter-type output length budget, kept the warmer tone for acknowledgements only. _You said: FB-07, FB-08, FB-09, FB-10, FB-13 - we did: this release._
- **v1.5.0** (Apr 27, reliability): ClaimCore circuit breaker with retries and jitter; 'AI unavailable' template fallback in the adjuster UI. _You said: FB-04, FB-05, FB-06 - we did: this release._

Loop closure: 8/20 actionable items closed (40%).

## 8. What's next

| Priority | Item | Score | Feedback |
|---|---|---:|---|
| P1 (risk override) | BL-25 Prohibited-content guard on every draft (coverage and liability language) | 2.93 | FB-14, FB-22 |
| P1 | BL-21 Field validator: check dates, amounts and deductibles against ClaimCore before a draft is shown | 5.50 | FB-02, FB-11, FB-19 |
| P2 | BL-24 Always include the latest adjuster notes and contact changes in context | 3.30 | FB-03, FB-17 |
| P2 | BL-28 Template and export fixes (PDF bullets, branch contact data) | 3.30 | FB-15, FB-24 |
| P3 | BL-22 Tone and length checks per letter type in the eval suite (incl. condolence letters) | 1.50 | FB-21 |
| P3 | BL-27 Discovery: Spanish-language letters and bulk backlog mode | 0.80 | FB-16, FB-20 |

- Ship the prohibited-content guard (BL-25) and the ClaimCore field validator (BL-21).
- Add a latency and tone gate to the canary stage so changes like v1.4.0 are caught before full rollout.
- Double the judge sample so the quality estimate has a tighter confidence interval.
- Hold the feature freeze until availability recovers above the 99.5% SLO.

## 9. Known limitations (measured)

- **LIM-1 Wording is nondeterministic.** Two drafts for the same claim can differ in phrasing; adjusters should review the draft in front of them, not a previous one.
- **LIM-2 Factual slips are rare, not impossible.** The judge flags drafts with a factual error against the claim file; adjuster approval remains the control. April: 10 of 499 sampled drafts flagged (2.0%, 95% CI 1.1%-3.6%).
- **LIM-3 Quality is measured on a sample.** About 5% of drafts are judged; the true pass rate lies within the confidence interval shown, not exactly at the point estimate. April: 91.8% with a 95% CI of 89.0%-93.9%.
- **LIM-4 Long letters take longer.** Latency grows with letter length; the slowest 1% of drafts take noticeably longer than the median. April: p50 4.10 s, p99 8.86 s.
- **LIM-5 Depends on ClaimCore.** If ClaimCore is degraded, drafting falls back to template mode (since v1.5.0).

## 10. What the system will NOT do

- Send letters. Every letter is approved by a licensed adjuster and sent through the existing PostMaster workflow.
- Make or imply coverage determinations, or admit liability.
- Handle injury-claim correspondence (phase 2, after the DPIA).
- Guarantee identical wording for identical inputs - outputs are probabilistic, so regenerating a draft can change its wording.
- Act as the system of record. Claim numbers, dates and amounts come from ClaimCore and are checked there.
- Answer customers directly. It drafts letters for adjusters only.
- Draft letters in languages other than English (Spanish is in discovery, not committed).

## Appendix - SLI definitions

| SLO | SLI | Window |
|---|---|---|
| SLO-1 Availability | good requests / valid requests during business hours (4xx client errors are not valid requests) | 30 days |
| SLO-2 Latency | share of successful drafts completed within 8 s (streamed; full draft) | 30 days (tracked weekly) |
| SLO-3 Draft quality | share of sampled drafts scoring >= 4/5 on the agreed rubric (calibrated LLM judge) | 30 days |
| SLO-4 Cost per completed letter | model spend (drafting + judging) / letters approved as-is or after edits | 30 days |
| SLO-5 Rejection and escalation rate | (drafts rejected + escalated) / successful drafts | 30 days (tracked weekly) |

_Narrative sections drafted by Claude from the computed facts; every figure was checked against telemetry. All tables are computed from the request logs by code._
