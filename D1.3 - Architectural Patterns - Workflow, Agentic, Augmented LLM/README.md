# D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM

> **Exam objective:** "Select appropriate architectural patterns (workflow, agentic, augmented LLM)"

## Why it matters

Pattern choice fixes the cost, latency, predictability and testability of a system long before any prompt
tuning. Anthropic's "Building effective agents" distinguishes **workflows** (LLMs and tools orchestrated
through predefined code paths) from **agents** (the LLM dynamically directs its own process and tool use),
both built from one unit: the **augmented LLM**. The exam expects you to pick the *simplest* pattern that
meets the requirement and to justify it in terms of trade-offs, not fashion.

## Key concepts

### The building block: augmented LLM

An LLM enhanced with **retrieval** (search/RAG over your data), **tools** (typed actions via tool use / MCP)
and **memory** (state that persists across turns or sessions, e.g. the client-side memory tool). For many
use cases a single augmented call is the whole solution. Design the interface well: clear tool descriptions
that say *when* to call, typed and strict schemas, and documents wrapped and cited.

### Workflows (code owns control flow)

| Pattern | Use when | Trade-off | Lab example |
|---|---|---|---|
| **Prompt chaining** | Task splits into a fixed sequence of simpler steps; add programmatic **gates** between steps | More latency (sum of steps) for higher accuracy per step | changelog -> bullets (gate: no ticket ids/internal changes) -> release note |
| **Routing** | Distinct input categories are better handled separately (different prompt, tools or model) | Router errors misdirect; cheap router keeps cost ~1x | Haiku router -> Sonnet refunds / Opus technical / Haiku general |
| **Parallelization - sectioning** | Independent, *known in advance* subtasks | N calls of cost; latency = slowest branch | security / performance / readability reviews concurrently |
| **Parallelization - voting** | Same task run several times for confidence or recall | N x cost; needs a tie/split policy | 3 reviewer perspectives -> majority; split vote -> human |
| **Orchestrator-workers** | Subtasks cannot be predicted; the LLM decides them from the input | Less predictable, more tokens (plan + workers + synthesis) | feature request -> orchestrator picks frontend/backend/scheduler work |
| **Evaluator-optimizer** | Clear evaluation criteria exist and iteration measurably helps | 2 calls per iteration; must cap iterations | code checks first (cheap), LLM judge for factual accuracy |

**Parallelization vs orchestrator-workers** is a classic trap: both fan out, but parallelization's
subtasks are fixed in code; orchestrator-workers' subtasks are chosen by the model at runtime.

### Agents (model owns control flow)

Use an agent when the number and order of steps cannot be known in advance and the model must decide what to
do next from **ground truth in the environment** (tool results). An agent is "just" a loop:

```
while steps < limit:
    response = messages.create(tools=..., messages=history)
    if stop_reason == "end_turn": done
    if stop_reason == "tool_use": run every tool_use block, append ALL tool_results in ONE user turn
    else (max_tokens / refusal / pause_turn): handle explicitly
escalate to a human when the limit is hit
```

Non-negotiables: stopping conditions (step and token budgets), least-privilege tools with **policy
enforced in tool code** (the lab's `create_replacement` refuses when ineligible even if the model asks),
`is_error: true` tool results so the model can recover, transcripts for audit, and human checkpoints for
irreversible actions. The SDK **tool runner** implements this loop for you; write the manual loop when you
need full control (this lab writes it out to make the mechanics visible).

### Trade-off summary

| | Augmented LLM | Workflows | Agent |
|---|---|---|---|
| Predictability / testability | high | high-medium | low |
| Cost per task | lowest | 1-10x | 5-20x+ (tool schemas + growing history each step) |
| Latency | lowest | sum or max of steps | unbounded without limits |
| Flexibility on novel inputs | low | medium | high |
| Failure modes | wrong answer | wrong branch / gate miss | compounding errors, loops, unsafe actions |

Framework caution: agent frameworks are fine, but you must understand the underlying calls; extra
abstraction hides prompts and responses and makes debugging harder.

## Architecture

```
            +------------------- AUGMENTED LLM -------------------+
 query ---->| retrieval (TF-IDF) + memory (facts) + tools         |----> answer + memory write
            +-----------------------------------------------------+

 WORKFLOWS (code decides the path)
   chaining:      step1 --gate--> step2
   routing:       router --> {refund: Sonnet | technical: Opus | general: Haiku}
   sectioning:    [security || performance || readability] --> merge
   voting:        [strict || lenient || literal] --> majority (split -> human)
   orch-workers:  Opus plan --> worker_1..n (Sonnet, isolated context) --> Opus synthesis
   eval-optimize: generate --> code checks --> LLM judge --(feedback)--> generate ... (max 3)

 AGENT (model decides the path)
   task --> [Claude --tool_use--> tools --tool_result--> Claude ...] --end_turn--> summary
                        bounded by max_steps; tool code enforces policy
```

## What the code demonstrates

| # | Function | What to notice |
|---|---|---|
| 1 | `run_augmented` | Retrieval scores printed; two tool calls in one turn, both results returned in ONE user message; memory updated |
| 2 | `run_chaining` / `gate_bullets` | The gate (code, not LLM) rejects ticket ids and internal changes; step 1 reruns with gate feedback |
| 3 | `run_routing` | Structured-output router with a safe default; each route has its own model and prompt |
| 4 | `run_parallel` | Async client: wall-clock is roughly the slowest branch, not the sum; voting diversity comes from perspectives because sampling parameters such as `temperature` are not available on current models |
| 5 | `run_orchestrator` | The plan is model-generated (structured output); workers see only their subtask |
| 6 | `run_evaluator_optimizer` | Cheap deterministic checks before an LLM judge; capped iterations with human fallback |
| 7 | `run_agent` | Tool use until `end_turn`; `in=` tokens grow each step; run B hits the step limit and escalates |
| - | Cost table | Agent and orchestrator rows dominate token spend |
| 8 | `select_pattern` | Ordered decision rules map scenario traits to a pattern with predictability/cost/latency notes |

## Run it

From the repository root:

```bash
.venv/Scripts/python "D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM/python/main.py"
node "D1.3 - Architectural Patterns - Workflow, Agentic, Augmented LLM/typescript/main.ts"
```

## Exam tips & traps

- **Simplest pattern first.** If one augmented call works, a workflow is over-engineering; if a workflow
  works, an agent is over-engineering. "Use an autonomous agent" is the most common wrong answer.
- **Known vs unknown subtasks** decides sectioning vs orchestrator-workers.
- **Routing is also a cost pattern**: send easy traffic to Haiku and hard traffic to Opus (model details in D2.1).
- **Evaluator-optimizer needs articulable criteria** and evidence that feedback improves results; always cap
  iterations and put cheap deterministic checks before the LLM judge.
- **Agents need guardrails in code**: step limits, budgets, least-privilege tools, approval for irreversible
  actions, and `stop_reason` handling. Prompt instructions alone are not a control.
- **Parallel tool calls**: return every `tool_result` in a single user message, matched by `tool_use_id`;
  failed tools return `is_error: true` instead of raising.
- **No sampling knobs**: `temperature`/`top_p`/`top_k` return 400 on current models, so voting diversity
  comes from varied prompts/perspectives, and determinism comes from structure (schemas, gates), not temperature 0.

### Practice questions

**Q1.** Every pull request must be checked for security issues, licence compliance and test coverage. The
checks are independent and well defined, and developers want feedback quickly. Which pattern fits best?

- A. An autonomous agent that decides which checks to run
- B. Orchestrator-workers with an Opus planner
- C. Parallelization (sectioning): run the three checks concurrently and merge the results
- D. Prompt chaining: security, then licences, then coverage

**Answer: C.** The subtasks are known in advance and independent, so run them concurrently; latency is the
slowest check. B adds a planning call to rediscover subtasks you already know; D serialises them (latency =
sum); A adds unpredictability with no benefit.

**Q2.** A coding assistant implements GitHub issues. Which files must change, and how many, cannot be known
until the codebase is inspected; once identified, each change is a well-scoped task. Which pattern best fits?

- A. Routing to a per-language prompt
- B. Orchestrator-workers: a planner decides the subtasks at runtime and delegates each to a worker
- C. Parallelization (voting) across three identical prompts
- D. A single augmented LLM call with the whole repository in context

**Answer: B.** Dynamic, input-dependent subtasks are the defining trait of orchestrator-workers. Routing
selects one handler; voting repeats one task; a single call does not scale to unknown multi-file changes.

## Further reading

- Anthropic engineering blog: "Building effective agents" (augmented LLM, five workflows, agents)
- Anthropic engineering blog: "Writing effective tools for agents"
- Claude docs: "Tool use overview" and "How to implement tool use" (parallel tool use, `is_error`)
- Claude docs: "Memory tool" and "Structured outputs"
- Claude docs: SDK tool runner (Python and TypeScript SDK helpers)
- Anthropic cookbook: agent pattern notebooks (basic workflows, orchestrator-workers, evaluator-optimizer)
- Related labs: D1.4 (multi-agent), D1.5 (decomposition), D3.1 (capability bloat), D3.5/D3.6 (retrieval)
