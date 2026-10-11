# Lifecycle timeline - Claims Correspondence Copilot (simulated)

> Simulated project timeline; model release and retirement dates are fictional for the exercise - always check the official model deprecations page.

## 1. Phase model

| Phase | Goal | Exit criteria (gate) | Reviewers | Next |
|---|---|---|---|---|
| Discovery | Agree the problem, measurable requirements and a feasible first use case | D1 Requirements baseline approved<br>D2 No open high-severity stakeholder conflicts<br>D3 Every NFR measurable<br>D4 At least one use case rated GO | Executive sponsor, Solution architect | design |
| Design | Architecture, decisions, SLOs and privacy controls agreed | DS1 Architecture spec lint-clean<br>DS2 Key decisions recorded as accepted ADRs<br>DS3 SLIs, SLOs and SLA approved<br>DS4 Privacy / security review approved<br>DS5 Model API compatibility scan performed | Architecture board, Security and privacy | build |
| Build / prototype | Working system in staging with an eval suite and versioned prompts | B1 Prototype running in staging<br>B2 Eval suite with >= 200 cases<br>B3 All prompts versioned and pinned<br>B4 Route configs compatible with their models (no blocking API issues) | Engineering lead, AI quality lead | evaluation |
| Evaluation | Evidence that quality, safety, latency and cost meet targets - and nothing regressed | E1 Eval pass rate >= 90%<br>E2 No critical safety failures<br>E3 p95 latency <= 8 s under load<br>E4 Cost per task <= $0.06<br>E5 No regressions vs the frozen baseline | AI quality lead, Product owner, Compliance | handoff, build |
| Handoff | Operations can run, monitor and change the system without the build team | H1 Runbook covers every alert<br>H2 Eval suite and baseline results stored<br>H3 Dashboards live (SLOs, cost, guardrails)<br>H4 Named owners: service, prompts, on-call<br>H5 Prompt registry with pinned versions<br>H6 Model inventory with deprecation watch<br>H7 Operations acceptance signed | Operations lead, Service owner | monitoring |
| Monitoring | Run against SLOs; detect drift, cost creep and model lifecycle events | M1 Iteration trigger raised by a monitoring signal | Service owner | iteration |
| Iteration | Scope the change and assess impact before re-entering design | I1 Change request approved with scope<br>I2 Impact assessment completed | Product owner, Solution architect | design |

## 2. Timeline

| Week | Date | Phase | Type | Event |
|---|---|---|---|---|
| W01 | 2026-09-07 | discovery | START | Kick-off; structured interview guide issued |
| W03 | 2026-09-21 | discovery | ART | req_doc (draft): Requirements draft v0.1: 2 high-severity conflicts, 4 vague NFRs |
| W04 | 2026-09-28 | discovery | ART | req_doc (approved): Steering decisions recorded; NFRs made measurable; baseline v1.0 approved |
| W04 | 2026-09-28 | discovery | ART | use_case_assessment (approved): UC-1 rated GO (pilot) |
| W04 | 2026-09-28 | discovery | GATE | Discovery -> Design: PASSED (4/4 exit criteria) |
| W05 | 2026-10-05 | design | ART | arch_spec (draft): Architecture spec draft (doc-lint: 7 errors) |
| W06 | 2026-10-12 | design | ART | arch_spec (approved): Doc-lint clean after review |
| W06 | 2026-10-12 | design | ART | adr_log (approved): ADR-0004..0007 accepted (model choice, grounding strategy, HITL) |
| W06 | 2026-10-12 | design | ART | slo_doc (approved): SLIs, SLOs and SLA agreed with the sponsor |
| W07 | 2026-10-19 | design | ART | dpia (approved): Privacy review: US residency, PII masking, 7-year audit trail |
| W07 | 2026-10-19 | design | SCAN | Baseline API-compatibility scan of route configs: 0 blocking, 0 tuning |
| W07 | 2026-10-19 | design | GATE | Design -> Build / prototype: PASSED (5/5 exit criteria) |
| W09 | 2026-11-02 | build | ART | prototype (approved): Prototype deployed to staging |
| W09 | 2026-11-02 | build | ART | eval_suite (approved): Golden set: 240 cases incl. adversarial and fairness slices |
| W09 | 2026-11-02 | build | ART | prompt_registry (approved): Prompts versioned (2.1.0) and pinned |
| W10 | 2026-11-09 | build | GATE | Build / prototype -> Evaluation: PASSED (4/4 exit criteria) |
| W11 | 2026-11-16 | evaluation | ART | eval_report (approved): Offline eval on 240 cases + load test |
| W11 | 2026-11-16 | evaluation | GATE | Evaluation -> Handoff: PASSED (5/5 exit criteria) |
| W12 | 2026-11-23 | handoff | ART | handoff_package v1: Handoff package v1 submitted to operations (runbook missing AL-4, AL-5; 2 dashboards; 1 owner missing) |
| W13 | 2026-11-30 | handoff | GATE | Handoff -> Monitoring: BLOCKED (2/7) - H1 runbook_coverage 67% (need >= 100%); H3 dashboards 2 (need >= 3); H4 owners_missing 1 (need <= 0); H6 model_inventory_ok = no; H7 ops_signoff = no |
| W14 | 2026-12-07 | handoff | ART | handoff_package v2: Runbook completed, dashboards added, prompt owner named, deprecation watch on, ops sign-off |
| W15 | 2026-12-14 | handoff | GATE | Handoff -> Monitoring: PASSED (7/7 exit criteria) |
| W16 | 2026-12-21 | monitoring | NOTE | Monitoring baseline from evaluation: quality 0.910, cost/task $0.044 |
| W16 | 2026-12-21 | monitoring | SIGNAL | quality 0.915 [OK] cost $0.044 x1.00 [OK] p95 7.2s [OK] |
| W17 | 2026-12-28 | monitoring | SIGNAL | quality 0.918 [OK] cost $0.045 x1.02 [OK] p95 7.3s [OK] |
| W18 | 2027-01-04 | monitoring | SIGNAL | quality 0.912 [OK] cost $0.046 x1.05 [OK] p95 7.1s [OK] |
| W19 | 2027-01-11 | monitoring | SIGNAL | quality 0.897 [WATCH] cost $0.046 x1.05 [OK] p95 8.2s [INCIDENT] - prompt v2.2 regression rolled back (incident INC-019) |
| W20 | 2027-01-18 | monitoring | SIGNAL | quality 0.914 [OK] cost $0.049 x1.11 [WATCH] p95 7.0s [OK] |
| W21 | 2027-01-25 | monitoring | SIGNAL | quality 0.916 [OK] cost $0.050 x1.14 [WATCH] p95 7.1s [OK] |
| W22 | 2027-02-01 | monitoring | SIGNAL | quality 0.917 [OK] cost $0.051 x1.16 [WATCH] p95 7.2s [OK] |
| W22 | 2027-02-01 | monitoring | SIGNAL | SIG-R released: claude-sonnet-5-5 (successor of claude-sonnet-5), claude-haiku-5-5 (successor of claude-haiku-4-5) [EVALUATE] |
| W22 | 2027-02-01 | monitoring | SIGNAL | SIG-D claude-sonnet-5 retires 2027-06-30 (149 days) [TRIGGER] |
| W22 | 2027-02-01 | monitoring | EVENT | Iteration ticket ITR-001 raised: SIG-R successor models released: evaluate upgrade (optional) / SIG-D claude-sonnet-5 retires 2027-06-30 (149 days): mandatory migration / SIG-C cost watch x1.16 (fold into the same iteration) |
| W22 | 2027-02-01 | monitoring | GATE | Monitoring -> Iteration: PASSED (1/1 exit criteria) |
| W23 | 2027-02-08 | iteration | ART | change_request (approved): CR-031: migrate draft_letter (Sonnet 5 -> Sonnet 5.5) and classify_guard (Haiku 4.5 -> Haiku 5.5) |
| W23 | 2027-02-08 | iteration | ART | impact_assessment: 7 breaking changes, 4 tuning tasks; monthly model cost $483.60 -> $457.50 |
| W23 | 2027-02-08 | iteration | GATE | Iteration -> Design: PASSED (2/2 exit criteria) |
| W24 | 2027-02-15 | design | SCAN | Compatibility scan against the target models: 7 blocking, 4 tuning |
| W24 | 2027-02-15 | design | ART | adr_log (approved): ADR-0011 'Migrate to Sonnet 5.5 / Haiku 5.5' accepted; supersedes ADR-0004 |
| W24 | 2027-02-15 | design | GATE | Design -> Build / prototype: PASSED (5/5 exit criteria) |
| W25 | 2027-02-22 | build | SCAN | Route configs updated per scan findings; re-scan: 0 blocking, 1 tuning task(s) left |
| W25 | 2027-02-22 | build | GATE | Build / prototype -> Evaluation: PASSED (4/4 exit criteria) |
| W26 | 2027-03-01 | evaluation | EVAL | round 1: claude-sonnet-5-5 @ prompt 2.1.0 5/6 vs baseline 5/6; regressions: C-03; improved: C-05 |
| W26 | 2027-03-01 | evaluation | EVAL | round 2: claude-sonnet-5-5 @ prompt 2.1.1 6/6 vs baseline 5/6; regressions: none; improved: C-05 - Prompt patch 2.1.1: explicit length budget for status letters |
| W26 | 2027-03-01 | evaluation | ART | eval_report (approved): Load test on the new models |
| W26 | 2027-03-01 | evaluation | GATE | Evaluation -> Handoff: PASSED (5/5 exit criteria) |
| W27 | 2027-03-08 | handoff | ART | handoff_package v3: Model inventory and runbook updated; canary plan approved (inventory: draft_letter=claude-sonnet-5-5, classify_guard=claude-haiku-5-5) |
| W27 | 2027-03-08 | handoff | GATE | Handoff -> Monitoring: PASSED (7/7 exit criteria) |
| W28 | 2027-03-15 | monitoring | NOTE | Canary 5% -> 100% over one week; legacy model ids removed from config |

## 3. Gate reviews

### W04 Discovery -> Design: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| D1 Requirements baseline approved | pass | req_doc approved |
| D2 No open high-severity stakeholder conflicts | pass | open_high_conflicts 0 (need <= 0) |
| D3 Every NFR measurable | pass | unmeasurable_nfrs 0 (need <= 0) |
| D4 At least one use case rated GO | pass | go_count 1 (need >= 1) |

### W07 Design -> Build / prototype: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| DS1 Architecture spec lint-clean | pass | lint_errors 0 (need <= 0) |
| DS2 Key decisions recorded as accepted ADRs | pass | accepted 3 (need >= 2) |
| DS3 SLIs, SLOs and SLA approved | pass | slo_doc approved |
| DS4 Privacy / security review approved | pass | dpia approved |
| DS5 Model API compatibility scan performed | pass | compat_scan present |

### W10 Build / prototype -> Evaluation: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| B1 Prototype running in staging | pass | prototype present |
| B2 Eval suite with >= 200 cases | pass | cases 240 (need >= 200) |
| B3 All prompts versioned and pinned | pass | unversioned 0 (need <= 0) |
| B4 Route configs compatible with their models (no blocking API issues) | pass | blocking 0 (need <= 0) |

### W11 Evaluation -> Handoff: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| E1 Eval pass rate >= 90% | pass | pass_rate 91% (need >= 90%) |
| E2 No critical safety failures | pass | critical_failures 0 (need <= 0) |
| E3 p95 latency <= 8 s under load | pass | p95_s 7.4 (need <= 8) |
| E4 Cost per task <= $0.06 | pass | cost_per_task $0.044 (need <= $0.060) |
| E5 No regressions vs the frozen baseline | pass | regressions 0 (need <= 0) |

### W13 Handoff -> Monitoring: BLOCKED

| Criterion | Result | Evidence |
|---|---|---|
| H1 Runbook covers every alert | **FAIL** | runbook_coverage 67% (need >= 100%) |
| H2 Eval suite and baseline results stored | pass | baseline_recorded = yes |
| H3 Dashboards live (SLOs, cost, guardrails) | **FAIL** | dashboards 2 (need >= 3) |
| H4 Named owners: service, prompts, on-call | **FAIL** | owners_missing 1 (need <= 0) |
| H5 Prompt registry with pinned versions | pass | prompt_registry_pinned = yes |
| H6 Model inventory with deprecation watch | **FAIL** | model_inventory_ok = no |
| H7 Operations acceptance signed | **FAIL** | ops_signoff = no |

### W15 Handoff -> Monitoring: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| H1 Runbook covers every alert | pass | runbook_coverage 100% (need >= 100%) |
| H2 Eval suite and baseline results stored | pass | baseline_recorded = yes |
| H3 Dashboards live (SLOs, cost, guardrails) | pass | dashboards 4 (need >= 3) |
| H4 Named owners: service, prompts, on-call | pass | owners_missing 0 (need <= 0) |
| H5 Prompt registry with pinned versions | pass | prompt_registry_pinned = yes |
| H6 Model inventory with deprecation watch | pass | model_inventory_ok = yes |
| H7 Operations acceptance signed | pass | ops_signoff = yes |

### W22 Monitoring -> Iteration: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| M1 Iteration trigger raised by a monitoring signal | pass | iteration_ticket present |

### W23 Iteration -> Design: PASSED

| Criterion | Result | Evidence |
|---|---|---|
| I1 Change request approved with scope | pass | change_request approved |
| I2 Impact assessment completed | pass | impact_assessment present |

### W24 Design -> Build / prototype: PASSED (loop 2)

| Criterion | Result | Evidence |
|---|---|---|
| DS1 Architecture spec lint-clean | pass | lint_errors 0 (need <= 0) |
| DS2 Key decisions recorded as accepted ADRs | pass | accepted 4 (need >= 2) |
| DS3 SLIs, SLOs and SLA approved | pass | slo_doc approved |
| DS4 Privacy / security review approved | pass | dpia approved |
| DS5 Model API compatibility scan performed | pass | compat_scan present |

### W25 Build / prototype -> Evaluation: PASSED (loop 2)

| Criterion | Result | Evidence |
|---|---|---|
| B1 Prototype running in staging | pass | prototype present |
| B2 Eval suite with >= 200 cases | pass | cases 240 (need >= 200) |
| B3 All prompts versioned and pinned | pass | unversioned 0 (need <= 0) |
| B4 Route configs compatible with their models (no blocking API issues) | pass | blocking 0 (need <= 0) |

### W26 Evaluation -> Handoff: PASSED (loop 2)

| Criterion | Result | Evidence |
|---|---|---|
| E1 Eval pass rate >= 90% | pass | pass_rate 100% (need >= 90%) |
| E2 No critical safety failures | pass | critical_failures 0 (need <= 0) |
| E3 p95 latency <= 8 s under load | pass | p95_s 6.9 (need <= 8) |
| E4 Cost per task <= $0.06 | pass | cost_per_task $0.041 (need <= $0.060) |
| E5 No regressions vs the frozen baseline | pass | regressions 0 (need <= 0) |

### W27 Handoff -> Monitoring: PASSED (loop 2)

| Criterion | Result | Evidence |
|---|---|---|
| H1 Runbook covers every alert | pass | runbook_coverage 100% (need >= 100%) |
| H2 Eval suite and baseline results stored | pass | baseline_recorded = yes |
| H3 Dashboards live (SLOs, cost, guardrails) | pass | dashboards 4 (need >= 3) |
| H4 Named owners: service, prompts, on-call | pass | owners_missing 0 (need <= 0) |
| H5 Prompt registry with pinned versions | pass | prompt_registry_pinned = yes |
| H6 Model inventory with deprecation watch | pass | model_inventory_ok = yes |
| H7 Operations acceptance signed | pass | ops_signoff = yes |

## 4. Handoff package validation

| Check | v1 (submitted) | v2 (after fixes) | v3 (migration delta) |
|---|---|---|---|
| runbook_coverage | 67% | 100% | 100% |
| baseline_recorded | yes | yes | yes |
| dashboards | 2 | 4 | 4 |
| owners_missing | 1 | 0 | 0 |
| prompt_registry_pinned | yes | yes | yes |
| model_inventory_ok | no | yes | yes |
| ops_signoff | no | yes | yes |

## 5. Monitoring signals

| Week | Quality (2-wk mean) | Quality signal | Cost/task (x baseline) | Cost signal | p95 | Latency signal | Notes |
|---|---|---|---|---|---|---|---|
| W16 | 0.915 (0.915) | OK | $0.044 (x1.00) | OK | 7.2 s | OK | - |
| W17 | 0.918 (0.917) | OK | $0.045 (x1.02) | OK | 7.3 s | OK | - |
| W18 | 0.912 (0.915) | OK | $0.046 (x1.05) | OK | 7.1 s | OK | - |
| W19 | 0.897 (0.905) | WATCH | $0.046 (x1.05) | OK | 8.2 s | INCIDENT | prompt v2.2 regression rolled back (incident INC-019) |
| W20 | 0.914 (0.905) | OK | $0.049 (x1.11) | WATCH | 7.0 s | OK | - |
| W21 | 0.916 (0.915) | OK | $0.050 (x1.14) | WATCH | 7.1 s | OK | - |
| W22 | 0.917 (0.917) | OK | $0.051 (x1.16) | WATCH | 7.2 s | OK | SIG-R released: claude-sonnet-5-5 (successor of claude-sonnet-5), claude-haiku-5-5 (successor of claude-haiku-4-5) [EVALUATE]; SIG-D claude-sonnet-5 retires 2027-06-30 (149 days) [TRIGGER] |

Signal rules: SIG-Q Quality drift: 2-week rolling judge pass rate < baseline - 0.03 -> iteration; SIG-C Cost creep: weekly cost per task > baseline x 1.2 (watch above x 1.1) -> iteration; SIG-L Latency: weekly p95 > 8 s -> incident process (rollback), not a lifecycle iteration; SIG-D Model deprecation: a model in use retires within 180 days -> migration task (mandatory); SIG-R New model release: a successor to a model in use is released -> evaluate upgrade (optional).

Iteration ticket ITR-001 reasons:

- SIG-R successor models released: evaluate upgrade (optional)
- SIG-D claude-sonnet-5 retires 2027-06-30 (149 days): mandatory migration
- SIG-C cost watch x1.16 (fold into the same iteration)

## 6. Iteration: model migration

### Impact assessment

| Route | Current model | Target model | Monthly cost now | Monthly cost after |
|---|---|---|---:|---:|
| draft_letter | `claude-sonnet-5` | `claude-sonnet-5-5` | $453.60 | $453.60 |
| classify_guard | `claude-haiku-4-5` | `claude-haiku-5-5` | $30.00 | $3.90 |
| **Total** | | | **$483.60** | **$457.50** |

### API compatibility scan (against the target models)

| Route | Rule | Severity | Finding |
|---|---|---|---|
| draft_letter | GEN-1 | BLOCKS | Update the model id to the target (exact id, no date suffix) |
| draft_letter | SON55-1 | BLOCKS | thinking {type: disabled} returns 400 on Sonnet 5.5 - use adaptive thinking at low effort (between_tools only if thinking must stay off) |
| draft_letter | TC-1 | BLOCKS | Forced tool_choice returns 400 - use auto + strict tool schemas, steer in the prompt and check that the call happened (or use structured outputs) |
| draft_letter | SON55-2 | TUNE | Effort levels are recalibrated on Sonnet 5.5 - re-run the effort sweep against the eval suite |
| draft_letter | GEN-2 | TUNE | Run the regression eval against the frozen baseline before switching; re-baseline latency and cost |
| classify_guard | GEN-1 | BLOCKS | Update the model id to the target (exact id, no date suffix) |
| classify_guard | HAI55-1 | BLOCKS | Manual thinking budgets return 400 on Haiku 5.5 - adaptive thinking only, depth via effort |
| classify_guard | HAI55-2 | BLOCKS | Non-default sampling parameters return 400 on Haiku 5.5 - remove temperature/top_p/top_k |
| classify_guard | HAI55-3 | BLOCKS | Assistant prefill returns 400 on Haiku 5.5 - use structured outputs (output_config.format) |
| classify_guard | HAI55-4 | TUNE | The same text counts as about 30% more tokens - re-baseline cost (the per-token price is much lower) |
| classify_guard | GEN-2 | TUNE | Run the regression eval against the frozen baseline before switching; re-baseline latency and cost |

Config before -> after the fixes:

```
draft_letter:
  before: {"model": "claude-sonnet-5", "max_tokens": 1200, "thinking": {"type": "disabled"}, "effort": "high", "tool_choice": {"type": "tool", "name": "emit_letter"}}
  after:  {"model": "claude-sonnet-5-5", "max_tokens": 1200, "thinking": {"type": "adaptive"}, "effort": "low", "tool_choice": {"type": "auto"}, "strict_tools": true}
classify_guard:
  before: {"model": "claude-haiku-4-5", "max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 2048}, "temperature": 0.2, "prefill": "{"}
  after:  {"model": "claude-haiku-5-5", "max_tokens": 4096, "thinking": {"type": "adaptive"}, "effort": "low", "output_format": "json_schema"}
```

Remaining after the fixes: TUNE draft_letter SON55-2: Effort levels are recalibrated on Sonnet 5.5 - re-run the effort sweep against the eval suite.

### Regression eval (old vs new model)

Frozen baseline: claude-sonnet-5 @ prompt 2.1.0 (re-run). Programmatic checks per case (must include / must not include / word budget).

**Round 1** - `claude-sonnet-5-5` with prompt 2.1.0: 5/6 pass; regressions: C-03; improved: C-05

| Case | Type | Baseline | Candidate | Words | Verdict | Detail |
|---|---|---|---|---:|---|---|
| C-01 | status update | PASS | PASS | 41 | same | - |
| C-02 | document request | PASS | PASS | 38 | same | - |
| C-03 | status update | PASS | FAIL | 156 | REGRESSION | 156 words > 120 |
| C-04 | payment confirmation | PASS | PASS | 37 | same | - |
| C-05 | status update | FAIL | PASS | 39 | improved | missing 'under review'; contains 'covered' |
| C-06 | acknowledgement | PASS | PASS | 25 | same | - |

**Round 2** - `claude-sonnet-5-5` with prompt 2.1.1: 6/6 pass; regressions: none; improved: C-05

| Case | Type | Baseline | Candidate | Words | Verdict | Detail |
|---|---|---|---|---:|---|---|
| C-01 | status update | PASS | PASS | 41 | same | - |
| C-02 | document request | PASS | PASS | 38 | same | - |
| C-03 | status update | PASS | PASS | 63 | same | - |
| C-04 | payment confirmation | PASS | PASS | 37 | same | - |
| C-05 | status update | FAIL | PASS | 39 | improved | missing 'under review'; contains 'covered' |
| C-06 | acknowledgement | PASS | PASS | 25 | same | - |

Decision: switch only after a round with zero regressions against the frozen baseline; then canary, then remove the legacy model ids. Fix regressions in the prompt or config - never by loosening the check.

## 7. Phase durations

| Phase | Loop | Weeks | Duration |
|---|---:|---|---:|
| Discovery | 1 | W01-W04 | 4 |
| Design | 1 | W05-W07 | 3 |
| Build / prototype | 1 | W08-W10 | 3 |
| Evaluation | 1 | W11-W11 | 1 |
| Handoff | 1 | W12-W15 | 4 |
| Monitoring | 1 | W16-W22 | 7 |
| Iteration | 1 | W23-W23 | 1 |
| Design | 2 | W24-W24 | 1 |
| Build / prototype | 2 | W25-W25 | 1 |
| Evaluation | 2 | W26-W26 | 1 |
| Handoff | 2 | W27-W27 | 1 |
| Monitoring | 2 | W28-W28 | 1 |

```
                            1234567890123456789012345678
Discovery                   ####........................
Design                      ....###.....................
Build / prototype           .......###..................
Evaluation                  ..........#.................
Handoff                     ...........####.............
Monitoring                  ...............#######......
Iteration                   ......................#.....
Design (loop 2)             .......................#....
Build / prototype (loop 2)  ........................#...
Evaluation (loop 2)         .........................#..
Handoff (loop 2)            ..........................#.
Monitoring (loop 2)         ...........................#
```
