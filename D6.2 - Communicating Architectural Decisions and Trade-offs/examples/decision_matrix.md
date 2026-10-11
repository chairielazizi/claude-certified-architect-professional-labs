# Decision matrix - How should the copilot ground each drafted letter in the customer's policy wording and claim documents?

As of 2026-10-08. Scores use anchored 0-10 scales (worst -> 0, best -> 10), so adding or removing an option cannot reorder the others. Must-meet constraints are applied before scoring.

## Raw measurements

| Criterion | Unit | Worst / best anchor | A | B | C | D |
|---|---|---|---:|---:|---:|---:|
| Answer quality | % | 80 / 100 | 88.5 | 91 | 92 | 94.5 |
| Responsiveness | s | 10 / 2 | 6.8 | 8.9 | 9.4 | 16 |
| Run cost | USD/month | 2000 / 0 | $596 | $454 | $755 | $1,355 |
| Time to value | weeks | 12 / 1 | 6 | 2 | 5 | 5 |
| Operational simplicity | 1-5 | 5 / 1 | 4 | 1 | 3 | 3 |
| Auditability | % | 80 / 100 | 94 | 88 | 95 | 96 |
| Content freshness | min | 1440 / 0 | 60 | 5 | 0 | 0 |
| Data minimization | 1-5 | 5 / 1 | 4 | 2 | 2 | 2 |
| Must-meet | - | - | PASS | PASS | PASS | FAIL |

## Weighted scores

| Criterion | Weight | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| Answer quality | 20.5 | 4.25 | 5.50 | 6.00 | 7.25 |
| Responsiveness | 12.0 | 4.00 | 1.37 | 0.75 | 0.00 |
| Run cost | 12.0 | 7.02 | 7.73 | 6.23 | 3.23 |
| Time to value | 14.5 | 5.45 | 9.09 | 6.36 | 6.36 |
| Operational simplicity | 11.0 | 2.50 | 10.00 | 5.00 | 5.00 |
| Auditability | 14.0 | 7.00 | 4.00 | 7.50 | 8.00 |
| Content freshness | 6.5 | 9.58 | 9.97 | 10.00 | 10.00 |
| Data minimization | 9.5 | 2.50 | 7.50 | 7.50 | 7.50 |
| **Weighted total** | 100 | **5.10** | **6.56** | **5.95** | **5.83** |

**Recommendation:** Option B - Long context + prompt caching (Sonnet 5.5).

## Stakeholder weights

| Stakeholder | Influence | Answer quality | Responsiveness | Run cost | Time to value | Operational simplicity | Auditability | Content freshness | Data minimization | Own winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dana Whitfield (VP Claims Operations (sponsor)) | 0.40 | 25 | 15 | 15 | 25 | 5 | 5 | 5 | 5 | B (6.59) |
| Priya Raman (Head of Compliance / DPO) | 0.30 | 20 | 0 | 5 | 5 | 5 | 35 | 10 | 20 | C (7.20) |
| Marcus Oyelaran (Director of Claims Platform) | 0.30 | 15 | 20 | 15 | 10 | 25 | 5 | 5 | 5 | B (6.74) |

## Sensitivity

- One-at-a-time (each weight x0.5 / x1.5): winner never changes
- Break-even: auditability weight x2.25 (14.0 -> 31.5 points) -> Option C
- Monte Carlo (5,000 trials, seed 42): A 0.0%, B 99.5%, C 0.5% -> **ROBUST**
