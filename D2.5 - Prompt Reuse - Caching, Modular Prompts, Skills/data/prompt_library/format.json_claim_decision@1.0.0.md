---
id: format.json_claim_decision
version: 1.0.0
layer: format
requires:
summary: JSON decision record consumed by the claims workflow engine.
---
Return only a JSON object with these keys: claim_id, decision (approve | refer | deny), handbook_sections
(array of section ids), deductible_usd (integer or null), rationale (one or two sentences).
