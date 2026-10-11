# Doc-lint report

| Rule | Description |
|---|---|
| DOC-001 | Every component has an accountable owner |
| DOC-002 | Every dependency resolves to a known component, system or data store |
| DOC-003 | Every tool declares a least-privilege auth scope |
| DOC-004 | Write tools either write to a pending state or require human approval |
| DOC-005 | Every SLO is backed by an existing alert |
| DOC-006 | Every alert points at an existing component |
| DOC-007 | Every prompt has a semantic version and an eval suite |
| DOC-008 | PII data stores define retention, encryption and residency |
| DOC-009 | Models use current model ids, are used by a component and have a fallback |
| DOC-010 | Every agent has an output guardrail on itself or its caller |
| DOC-011 | Every alert has a runbook entry |
| DOC-012 | Secrets are sourced from the vault |
| DOC-013 | Every request-flow participant exists in the spec |

## Draft spec: 7 errors, 0 warnings - publication blocked

| Rule | Severity | Ref | Finding |
|---|---|---|---|
| DOC-001 | ERROR | context_builder | component has no owner |
| DOC-002 | ERROR | batch_backlog | depends on unknown 'letter_queue' |
| DOC-003 | ERROR | get_policy_wording | tool has no auth scope |
| DOC-004 | ERROR | send_letter | 'final' write without human approval |
| DOC-005 | ERROR | SLO-4 | SLO has no alert |
| DOC-007 | ERROR | p_guard | prompt has no eval suite |
| DOC-008 | ERROR | eval_store | PII store missing retention |

## Fixes applied - Pull request #142 'Close doc-lint findings' - reviewed by the architecture board

- `set components[id=context_builder].owner` - Every component needs an accountable owner for on-call and change approval.
- `set components[id=batch_backlog].depends_on` - letter_queue was never built; the batch drafter reads the backlog from ClaimCore via the context builder and calls the Batches API directly.
- `set tools[name=get_policy_wording].auth_scope` - Least-privilege scope for the policy wording read.
- `remove tools[name=send_letter]` - Discovery constraint: the copilot must not call PostMaster; sending stays in the adjuster-approved workflow.
- `set prompts[id=p_guard].eval_suite` - Guardrail prompts need a regression suite (positives and negatives) like any other prompt.
- `set data_stores[id=eval_store].retention` - Eval results contain PII from sampled drafts; retention aligned with the audit policy for samples.
- `set slos[id=SLO-4].alert` - An SLO without an alert is a wish.
- `append alerts` - Alert backing SLO-4 (warns before the $0.06 SLO is breached).

## Fixed spec: 0 errors, 0 warnings

Runbook coverage (DOC-011): complete.
