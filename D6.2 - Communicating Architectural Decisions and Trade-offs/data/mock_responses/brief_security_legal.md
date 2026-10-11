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
