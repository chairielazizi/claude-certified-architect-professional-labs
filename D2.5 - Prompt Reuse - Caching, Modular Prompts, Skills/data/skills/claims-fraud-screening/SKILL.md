---
name: claims-fraud-screening
description: Screens an insurance claim for fraud indicators with Northwind's weighted red-flag checklist and returns a 0-100 score with a refer/proceed recommendation. Use when a claim looks suspicious, mentions recent policy changes or missing documents, or the user asks for a fraud check.
---

# Claims fraud screening

Use this procedure to produce a consistent, auditable fraud score for a single claim. The score informs triage
(handbook H9); it is never shown to the claimant.

## Procedure

1. Read the claim and list the facts that are actually stated. Do not assume facts that are not given.
2. Open `references/red_flags.md` and check every flag in the catalogue against the stated facts.
3. For each flag that applies, add its weight. Flags that cannot be assessed from the facts score zero and
   go into the "information needed" list.
4. Cap the total at 100.
5. Recommendation: score 40 or more -> "refer to SIU"; 20-39 -> "proceed with enhanced verification";
   below 20 -> "proceed".

## Output format

Return a short report with these lines, in this order:

- Score: <n>/100 -> <recommendation>
- Flags: <flag id> (<weight>) for each flag that applies, comma separated
- Information needed: what would confirm or clear the flags
- Handling note: one sentence for the handler; never mention fraud in claimant-facing text

## Rules

- Score only on stated facts; list unknowns instead of guessing.
- Two flags describing the same fact count once (use the higher weight).
- If the claim involves bodily injury, add "refer regardless of score" to the handling note (handbook H1.3).
