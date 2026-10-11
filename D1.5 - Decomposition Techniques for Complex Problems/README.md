# D1.5 - Decomposition Techniques for Complex Problems

> **Exam objective:** "Apply decomposition techniques for complex problem solving"

## Why it matters

A single giant prompt fails in ways that are hard to see: the output is truncated at `max_tokens`, a
requirement in the middle of a long instruction list is silently skipped, an arithmetic slip early on
poisons everything after it, and when something goes wrong you must re-run (and re-pay for) the whole
thing. Decomposition turns one opaque call into small steps that are each **checkable, retryable,
right-sized (model per step) and often parallelisable**. Choosing the right decomposition is a core
architect skill.

## Key concepts

### Techniques at a glance

| Technique | Shape | Use when | Lab section |
|---|---|---|---|
| **Prompt chaining** | A -> B -> C (fixed) | Known sequence of transformations | (D1.3) |
| **Task DAG / plan-and-execute** | Planner emits tasks + dependencies; executor runs in topological order | Sub-tasks with real data dependencies; some can run in parallel | 1 |
| **Map-reduce** | Split input -> map each chunk -> reduce | Long inputs (contracts, logs, transcripts) where chunks are independent | 2 |
| **Least-to-most** | Simplest sub-question first; each answer feeds the next | Multi-step reasoning or calculation where later steps build on earlier ones | 3 |
| **Orchestrator-workers** | LLM decides sub-tasks at runtime | Sub-tasks not knowable in advance | (D1.3, D1.4) |

### Plans are untrusted output - validate them in code

An LLM-generated plan is data, not truth. Before executing a DAG, check for **unknown dependencies,
duplicate ids and cycles** (DFS colouring or Kahn's algorithm), return specific feedback ("cycle: t2 -> t5
-> t2") to the planner, re-plan a bounded number of times, then escalate. Execute in **topological levels**:
every task in a level depends only on earlier levels, so a level can run concurrently.

### Context scoping

Each DAG task receives **only its dependencies' outputs**, not the whole history. Less noise, fewer
tokens, better focus, and a failure is attributable to one node.

### Map-reduce on long documents

- **Chunk on structure** (headings, then paragraphs); never mid-sentence. Keep chunks under a size budget.
- Include **document-level context** in each map call (title, section heading) so chunks are interpretable.
- **Reduce in code** whenever the reduction is deterministic (merge, de-duplicate, sort, count, sum). Use an
  LLM only for the narrative summary of the reduced data.
- Trade-off: map-reduce loses cross-chunk relationships (a definition in section 1 that changes section 7).
  When cross-references matter and the document fits, a long-context single pass (1M-token models) with
  prompt caching can be better - decomposition is a choice, not a default.

### Validation gates between steps

| Gate type | Example in the lab |
|---|---|
| Type/range | every numeric field non-negative; volume within 1x-2x last year |
| Identity | deflected + remaining == volume |
| Cross-check against inputs | deflected share must match the 38% pilot rate |
| Consistency with earlier answers | surplus == capacity - remaining |

Only gated values flow forward; a failing step is retried with the gate's feedback (bounded), then
escalated. Here the gate catches a classic swap (deflected vs remaining) at step 2 instead of letting it
produce a confidently wrong final answer.

### Monolithic vs decomposed

| | Monolithic | Decomposed |
|---|---|---|
| Calls / latency | 1 call | many calls; parallel levels and map steps recover latency |
| Token cost | lower | higher (repeated instructions), offset by cheaper models per step |
| Output limits | one `max_tokens` budget for everything | each step has its own budget |
| Failure visibility | silent (truncation, omissions) | per step (`stop_reason`, validators, gates) |
| Retry cost | everything | the failed step only |
| Best for | short, single-skill tasks that fit comfortably | multi-part deliverables, long inputs, multi-step reasoning |

## Architecture

```
                         GOAL: launch-readiness brief
                                    |
     +------------------------------+-------------------------------+
     |                              |                               |
 1. TASK DAG                   2. MAP-REDUCE                  3. LEAST-TO-MOST
 Opus planner (JSON)           chunk contract by section      decompose into sub-questions
   -> validate_dag()             (9 chunks)                    Q1 volume
      (unknown deps, cycles)   map: Haiku extracts             Q2 deflected/remaining --gate--
   -> re-plan with feedback      obligations per chunk         Q3 capacity            --gate--
   -> topo levels:             reduce in CODE: sort,           Q4 surplus/utilisation --gate--
      L1 t1                      count, penalties              final answer (Sonnet)
      L2 t2 t3 t4 (parallel)   Sonnet summary
      L3 t5
      L4 t6 go/no-go
     |                              |                               |
     +------------------------------+-------------------------------+
                                    |
             4. assemble brief -> coverage check (8/8) vs monolithic (4/8, truncated)
```

## What the code demonstrates

| Section | Code | What to notice |
|---|---|---|
| 0 | monolithic call + `coverage()` | `stop_reason=max_tokens`; 4/8 sections; "Pricing" missing even before the truncation point |
| 1 | `validate_dag`, `find_cycle`, `topo_levels`, `run_dag` | First plan has an unknown dependency and a cycle; validator feedback fixes it; level 2 has three independent tasks |
| 2 | `chunk_by_section`, `run_map_reduce` | Oversized section 4 split on paragraphs; deterministic reduce: 15 obligations, tightest deadlines, penalties |
| 3 | `step_schema`, `gate`, `run_least_to_most` | Per-step JSON schema built from the planned fields; the swap error fails the gate and is corrected on retry |
| 4 | assembly + ledger | 25 calls vs 1, roughly 3x tokens, but full coverage and per-step failure visibility |

## Run it

From the repository root:

```bash
.venv/Scripts/python "D1.5 - Decomposition Techniques for Complex Problems/python/main.py"
node "D1.5 - Decomposition Techniques for Complex Problems/typescript/main.ts"
```

## Exam tips & traps

- **`stop_reason: "max_tokens"` = truncated output.** The fix is to decompose (or raise `max_tokens` and
  stream large outputs), not "be concise" or sampling tweaks (`temperature` returns 400 on current models).
- **Validate LLM-generated plans structurally** before executing them: unknown ids, duplicates, cycles.
  "Execute in listed order" or "let the executor figure it out" are traps.
- **Map steps are independent -> parallelise them and use a cheaper model**; do deterministic reductions
  (counts, sums, sorting) in code, not with another LLM call.
- **Least-to-most orders sub-problems from simplest to hardest** and feeds verified answers forward; gates
  stop early errors from compounding.
- **Know when NOT to decompose**: short tasks, or long documents where cross-section reasoning matters and the
  content fits a long-context window (consider prompt caching for repeated queries).
- **Decomposition trades calls and tokens for reliability and observability**; recover latency by running
  DAG levels and map steps concurrently.

### Practice questions

**Q1.** A team asks Claude to write a 40-section compliance report in one request. Responses end abruptly
and several mid-list sections are missing; `stop_reason` is `"max_tokens"`. What is the best fix?

- A. Add "be concise and cover every section" to the prompt
- B. Plan the sections, generate each section (or small groups) in separate calls with a required-sections check, then assemble
- C. Set `temperature` to 0 for more deterministic output
- D. Switch to a faster model

**Answer: B.** The deliverable exceeds a single output budget and the omissions are invisible in one pass.
Decomposition gives each part its own budget and makes coverage checkable. A does not remove the limit; C
returns a 400 on current models and does not address length; D changes speed, not truncation.

**Q2.** A planner returns a task graph in which task B depends on D and task D depends on B. What should the
system do?

- A. Execute the tasks in the order listed
- B. Validate the graph before execution, return specific feedback ("cycle: B -> D -> B") to the planner, re-plan with a bounded retry count, and escalate if it is still invalid
- C. Ignore dependencies and run every task in parallel
- D. Let the executing model resolve the dependency at runtime

**Answer: B.** Plans are untrusted output and must be checked structurally; specific feedback lets the
planner repair them. A and C produce wrong inputs for dependent tasks; D hides the problem.

## Further reading

- Claude docs prompt-engineering guide: "Chain complex prompts for stronger performance"
- Claude docs: "Long context prompting tips"
- Claude docs: "Handling stop reasons" and "Structured outputs"
- Anthropic engineering blog: "Building effective agents" (prompt chaining, orchestrator-workers)
- Research paper: "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models" (Zhou et al.)
- Related labs: D1.3 (patterns), D1.4 (multi-agent), D2.4 (context windows), D3.5 (chunking for RAG)
