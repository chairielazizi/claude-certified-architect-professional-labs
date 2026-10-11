# D2.5 - Prompt Reuse - Caching, Modular Prompts, Skills

> **Exam objective:** "Implement prompt reuse strategies (caching, modular prompts, Skills)"

## Why it matters

Most production prompts are mostly static: role, policies, reference tables, examples, tool definitions. Only
a small tail changes per request. Reuse attacks that waste at three levels:

* **Prompt caching** makes the static prefix cheap and fast to re-read on every request (the platform
  mechanism).
* **Modular prompts** let many assistants reuse the same governed, versioned components, and keep the cache key
  stable (the engineering discipline).
* **Agent Skills** package procedures and knowledge as folders the model loads **only when relevant**
  (progressive disclosure, the context-efficiency mechanism).

This lab maps directly to **exam sample question 2**: *the same 8,000-token system prompt and policy document
are sent with every request.* The answer is to put the static content first and enable prompt caching.

## Key concepts

### Prompt caching: the one invariant

**Caching is a prefix match.** The prompt renders as `tools` → `system` → `messages`. A `cache_control`
breakpoint caches everything before it. Change one byte early in the prefix and everything after it misses.

| Item | Value (Oct 2026) |
|---|---|
| Marker | `{"type": "ephemeral"}` on a block (explicit), or top-level `cache_control` (automatic: last cacheable block) |
| TTL | 5 minutes (default, write = **1.25×** input) or `"ttl": "1h"` (write = **2×** input). Every read refreshes the TTL |
| Read price | ~0.1× input. **0.05× on Opus 5.5** ($0.20/MTok), **0.025× on Fable 5.1** |
| Breakpoints | Max **4** per request. Each looks back at most 20 blocks for a prior entry |
| Minimum prefix | 512 tokens on current models (the offline mock uses 1,024). Shorter prefixes are **silently** not cached |
| Scope | Per model and per workspace. Switching model = cold cache |
| Verify | `usage.cache_creation_input_tokens` (write), `usage.cache_read_input_tokens` (read). `input_tokens` is only the **uncached remainder** |

**Break-even** (lab, Opus 5.5): a 5-minute TTL pays off from the **2nd** request, a 1-hour TTL from the **3rd**.
Use 1h only when requests sharing the prefix arrive 5-60 minutes apart.

### Placement rules

| Pattern | Breakpoint |
|---|---|
| Big shared system prompt (exam Q2) | On the **last static system block**. User content follows uncached |
| Shared prefix + unique question (single-shot) | **Explicit** marker at the end of the shared part. Automatic caching would land on the unique tail and write every time without ever reading (lab section 6) |
| Multi-turn chat or agent loop | Top-level automatic caching (the breakpoint moves forward), plus an explicit marker on the static system prefix |
| Several assistants sharing components | Breakpoint after the **shared** components (lab section 7: the memo assistant reads what the triage assistant wrote) |

### Silent invalidators (no error, just a bigger bill)

`datetime.now()` or a UUID in the system prompt (lab section 5), unsorted JSON serialization, per-user values
in the system prompt, conditional system sections, tool sets that vary per request, switching model or
top-level effort mid-conversation, or editing earlier history. **Fix:** move dynamic content after the last
breakpoint (into the user turn, or a `role: "system"` message mid-conversation), serialize deterministically,
pin tools and model per route. **Monitor** `cache_read_input_tokens` continuously, because regressions are
silent. Also note that parallel requests can't read a cache entry until the first response starts streaming.
Fire one request first, or pre-warm with `max_tokens: 0`.

### Modular prompts

* A **prompt library** of components (persona, safety, policy, reference data, output format), each with an
  id, a **semver** version, a layer and declared **dependencies**. Each use case has a **manifest** with
  version constraints (`^2`, `*`, exact pins).
* The **composer** resolves versions, rejects missing dependencies and duplicates, and renders components in a
  **canonical order**: most shared and most stable first (safety → policy → reference → persona → format). The
  same components in any manifest order produce the **same bytes**, so the same cache key.
* A **fingerprint** of the composed prompt goes into logs and evals. A version bump changes it on purpose, as a
  controlled cache reset.
* Generated reference content (the coverage table here) must render **deterministically** from its source
  data.

### Agent Skills

A skill is a **folder with a `SKILL.md`**: YAML frontmatter plus instructions, optionally with reference files
and scripts.

```markdown
---
name: claims-fraud-screening          # <=64 chars, lowercase/digits/hyphens, no "anthropic"/"claude"
description: Screens a claim ... Use when a claim looks suspicious ...   # what it does AND when to use it (<=1024 chars)
---
# Instructions ... (see references/red_flags.md)
```

**Progressive disclosure**, the reason skills scale:

| Level | What enters context | When |
|---|---|---|
| 1. Metadata | `name` + `description` (~100 tokens per skill) | Always, so the model knows the skill exists |
| 2. Instructions | The `SKILL.md` body | When the task matches the description |
| 3. Resources | Referenced files. Scripts are *executed*, so only their output enters context | Only when a step needs them |

Where skills run: the **Claude API** (`container={"skills": [...]}` plus the code-execution tool, beta
`code-execution-2025-08-25`; custom skills uploaded with the **Skills API** `client.skills.create`, no beta
header; Anthropic-managed skills such as `pptx`, `xlsx`, `docx`, `pdf`), **Claude Code** (`.claude/skills/`),
the Claude apps, the Agent SDK and Managed Agents. Skills can contain code, so **only install skills from
trusted sources**.

### Which reuse strategy when?

| Need | Strategy |
|---|---|
| The same large static context on every request | **Prompt caching** |
| Many assistants sharing governed policies, personas and formats | **Modular prompt library** (+ caching of the shared prefix) |
| Many specialized procedures, only a few relevant per task | **Agent Skills** (progressive disclosure) |
| Many tools, only a few relevant per task | Tool search / progressive tool discovery (see **D3.8**) |
| Reusable examples | Few-shot block in the cached prefix (see **D2.3**) |

## Architecture

```mermaid
flowchart TB
    subgraph Library[Prompt library - versioned components]
      S1[safety.core@2.3.0] --- S2[safety.pii@1.0.0] --- P[policy.claims_handbook@3.2.0] --- R[reference.coverage_schedule<br/>generated from JSON]
      PA[persona.claims_assistant] --- PB[persona.underwriting_analyst] --- F1[format.json] --- F2[format.memo]
    end
    M1[manifest: claims_triage] --> C[Composer<br/>semver resolve, deps,<br/>canonical order, fingerprint]
    M2[manifest: underwriting_memo] --> C
    Library --> C
    C --> SYS["system blocks: [shared prefix | cache_control] [use-case tail]"]
    SYS --> API[Claude Opus 5.5]
    API --> U[usage: cache_creation vs cache_read]
    subgraph Skills[data/skills]
      K1[claims-fraud-screening/SKILL.md<br/>+ references/red_flags.md]
      K2[subrogation-letter/SKILL.md]
      K3[catastrophe-surge/SKILL.md]
    end
    Skills -->|L1: name + description| IDX[available_skills index]
    IDX --> API
    API -->|tool_use load_skill| L2[SKILL.md body]
    API -->|tool_use read_skill_file| L3[referenced file<br/>path-traversal guarded]
```

## What the code demonstrates

| Step | Code | What to look for |
|---|---|---|
| 1 | `PromptLibrary.load` | 9 components (two versions of `safety.core`), including one generated component |
| 2 | `compose`, `common_prefix` | Canonical order. A reversed manifest gives the same fingerprint, no sort gives a different one, a version pin resets it. Dependency error. 4 shared components (~7.4K tokens) |
| 3 | Exam Q2 loop | ~7.5K-token prefix: request 1 writes, requests 2-5 read. 67% cheaper over just 5 requests |
| 4 | TTL table | Uncached vs 5-minute vs 1-hour costs and break-even points |
| 5 | Timestamp anti-pattern | `cache_read=0` on every request, then reads resume once the timestamp moves to the user turn |
| 6 | Top-level automatic caching | Single-shot requests with unique questions write every time and never read |
| 7 | Shared breakpoint | The underwriting assistant reads the prefix the triage assistant cached |
| 8 | `load_skills`, `validate_skill_meta`, `read_skill_file`, `run_skill_agent` | Validation, a path-traversal guard, index vs inlined tokens (and a 40-skill projection), a 3-turn progressive-disclosure loop, and a non-matching request that loads nothing |
| 9 | Snippet | Skills API upload plus `container.skills` with code execution (live-only; actual upload gated by `LABS_RUN_SKILLS=1`) |

## Run it

```bash
.venv/Scripts/python "D2.5 - Prompt Reuse - Caching, Modular Prompts, Skills/python/main.py"
node "D2.5 - Prompt Reuse - Caching, Modular Prompts, Skills/typescript/main.ts"
```

Live mode lists your workspace's skills (read-only). Set `LABS_RUN_SKILLS=1` as well to upload the sample skill
and run a code-execution container (this creates an org resource and bills container time).

## Exam tips & traps

* **Exam Q2 pattern:** the same large instructions or policy on every request → **static content first + prompt
  caching**. Distractors: "shorten the policy", "switch to a smaller model", "send the policy only every 10th
  request".
* **Order matters:** tools → system → messages. A timestamp or user ID at the top of the system prompt silently
  disables caching for everything after it.
* **Verify with usage fields.** A rising `cache_read_input_tokens` is the only proof caching works.
  `input_tokens` alone looks deceptively small.
* **Automatic caching isn't always right.** For single-shot requests with a unique tail, place an explicit
  breakpoint at the end of the shared prefix.
* **Caches are model-scoped and workspace-scoped.** A model switch or traffic split across workspaces lowers hit
  rates.
* **Skills ≠ bigger system prompts.** Only name and description are always loaded, and the body loads on demand.
  A vague `description` means the skill never triggers.
* **Skills on the API need the code-execution tool and a container** (`container.skills`). Custom skills are
  uploaded through the Skills API.

### Practice questions

**Q1.** A claims assistant sends the same 7,500-token handbook and coverage table with every request, followed
by a short claim description. Costs are high and time-to-first-token is slow. After the team added
`cache_control`, `cache_read_input_tokens` stays at 0. The system prompt starts with
`"Current time: {now}"`. What should the architect do?

- A. Switch to the 1-hour TTL
- B. Move the timestamp out of the system prompt (into the user turn), keep the static handbook first with the
  breakpoint at its end, and confirm reads in the usage fields
- C. Add a second `cache_control` breakpoint on the user message
- D. Reduce the handbook to under 512 tokens

**Answer: B.** The timestamp changes the prefix on every request, so nothing can ever be read. A changes the
lifetime, not the invalidation. C caches unique content that's never reused. D throws away required context.

**Q2.** An enterprise has 35 specialized procedures (fraud screening, subrogation letters, catastrophe
handling...). Putting all of them in the system prompt costs ~40K tokens per request, but any given request
needs at most one or two. Which approach fits best?

- A. Fine-tune a model on all 35 procedures
- B. Package each procedure as an Agent Skill with a precise `description`. Only names and descriptions stay in
  context, and the matching `SKILL.md` (and its referenced files) load on demand
- C. Keep everything in the system prompt and enable caching only
- D. Split the procedures across 35 separate system prompts and pick one at random

**Answer: B.** Progressive disclosure keeps the always-on cost to roughly 100 tokens per skill. C helps cost,
but still fills context and attention with 34 irrelevant procedures (and every edit invalidates the cache). A
isn't how Claude platform customization works for procedures, and it's slow to update. D doesn't route by
relevance.

## Further reading

* Claude docs: *Prompt caching* (automatic vs explicit breakpoints, TTLs, pricing, invalidation)
* Claude docs: *Pricing* (cache write/read multipliers)
* Claude docs: *Agent Skills overview*, *Skills guide* (Skills API), *Skill authoring best practices*
* Claude docs: *Code execution tool* (container, skills)
* Claude docs: *Prompt templates and variables* / Console prompt management
* Anthropic engineering: *Equipping agents for the real world with Agent Skills*
* Related labs: D2.2 (templates and versioning), D2.3 (few-shot caching), D3.8 (progressive discovery for tools), D4.5 (cost-performance optimization)
