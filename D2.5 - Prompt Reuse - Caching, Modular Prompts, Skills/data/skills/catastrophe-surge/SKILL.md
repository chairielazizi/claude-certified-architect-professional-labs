---
name: catastrophe-surge
description: Applies Northwind's catastrophe (CAT) event procedures - surge deductibles, fast-track thresholds and inspection waivers - to claims that reference a declared CAT event code such as CAT-2026-014. Use when a claim mentions a CAT code or a declared regional disaster.
---

# Catastrophe (CAT) surge procedure

## Inputs
- CAT event code and declaration date
- Claim region and product (to read the schedule row)

## Steps
1. Confirm the loss date falls inside the CAT event window (declaration date to declaration date + 14 days).
2. Deductible = standard deductible x the region's CAT surge factor, capped at 2x standard (handbook H4.2).
3. Fast-track: property claims under $10,000 with photos skip the site inspection during a CAT event.
4. Claims above 80% of the limit still require inspection (H4.3) even during a CAT event.
5. Tag the claim with the CAT code so reinsurance recoveries can be aggregated.

## Output
One line per step with the value used and the schedule row or handbook section cited.
