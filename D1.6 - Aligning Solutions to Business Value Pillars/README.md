# D1.6 - Aligning Solutions to Business Value Pillars

> **Exam objective:** "Align solutions to business value pillars (efficiency, transformation, productivity, cost, performance SLAs)"

## Why it matters

Executives do not buy "an agent"; they buy hours back, more output per person, new revenue, a lower cost
per unit, and a service that meets its SLAs. An architect must turn each solution option into numbers on
those pillars, show the cost side honestly (Claude API + platform + people + build), check that the chosen
model can actually meet the performance SLAs, and present a recommendation with a KPI tree the business can
hold the team to. Getting the arithmetic wrong (or letting an LLM do it) destroys credibility.

## Key concepts

### The five pillars and how to measure them

| Pillar | Question | Metrics used in the lab |
|---|---|---|
| **Efficiency** | How much faster / less effort per unit? | agent hours saved per month, blended AHT, first-response (cycle) time |
| **Productivity** | How much more output per person? | tickets resolved per FTE needed |
| **Transformation** | What becomes possible that was not before? | new capability (24/7 multilingual) and risk-adjusted new revenue |
| **Cost** | What does it cost to build and run? | Claude API $/task (from `usage`), infra, people, build -> ROI, payback, 3-year net value |
| **Performance SLAs** | Can it meet the service commitments? | p95 latency, composite availability, quality on the eval set |

### Claude API cost per task

```
cost per call = (input x in_price + output x out_price + cache_write x 1.25 in_price + cache_read x 0.1 in_price) / 1M
cost per task = cost per call x calls per task          (agents: turns per task; workflows: steps per task)
monthly API   = cost per task x volume
```

- Measure tokens from `response.usage` in a pilot with production-size prompts (`cost_usd(model, usage)` in
  the shared client); do not guess from prompt length.
- Model choice moves cost by more than an order of magnitude (Haiku 5.5 $0.10/$0.50, Sonnet 5.5 $2/$10,
  Opus 5.5 $4/$20 per million tokens in this repo's pricing table) - and cached system prompts/knowledge
  bases cut input cost sharply. Optimisation techniques are D4.5.
- In most enterprise cases the API bill is **small next to people, integration and platform costs** (the lab:
  about $670/month API vs about $33,000/month run cost for option B). Value cases fail on adoption and
  realisation, rarely on token prices.

### Turning hours into money honestly

- **Realisation rate**: freed hours only become dollars through attrition, avoided hiring or redeployment
  to valuable work. The lab counts 70% of freed hours.
- **Risk-adjust transformation value** (the lab weights new revenue at 50% confidence) and keep it separate
  from efficiency savings so finance can see what is certain and what is upside.
- **ROI (year 1)** = (12 x monthly benefit - (12 x run cost + build)) / (12 x run cost + build);
  **payback** = build / monthly net benefit; also report a multi-year net value.
- A single **north-star metric** (cost per resolved ticket) with a **guardrail** (CSAT must not drop) prevents
  optimising one pillar at the expense of the customer.

### Performance SLAs pick the model

| SLA | What drives it | Architect levers |
|---|---|---|
| **p95 latency** | time to first token + output tokens x per-token time + tool round trips | smaller model, shorter outputs, fewer tool hops, streaming (improves perceived latency only), see D3.3 |
| **Availability** | product of serial components; redundancy = 1 - (1 - a)^n | multi-AZ app tier, model fallback path (second region/platform), graceful degradation to humans |
| **Quality** | eval-set accuracy for the task | better prompts/retrieval, bigger model, HITL |

Rule: **choose the cheapest model that meets every SLA**, not the most capable one. In the lab, Opus misses
the 6 s p95 target, Haiku misses the 0.92 quality bar, Sonnet passes all three. Availability is an
architecture property, not a model property: without a fallback path the option B design reaches only
99.84% against a 99.9% target. Use p95/p99, never averages, for latency SLAs. Ongoing SLA tracking is D6.3;
model trade-offs in depth are D2.1.

### Scorecards and KPI trees

- Agree pillar **weights with stakeholders before scoring** (otherwise weights get tuned to the preferred answer).
- Normalise each pillar (0-5 relative to the best option) and show the weighted total *and* the raw numbers.
- A **KPI tree** links the north star to driver KPIs (AHT, FRT, tickets/FTE, API $/ticket, payback, p95, availability,
  quality) with baseline -> target, so the pilot has a success contract.
- Let Claude write the **executive summary from computed numbers** and instruct it not to calculate; LLMs are
  great at narrative and unreliable at arithmetic.

## Architecture

```
 value_model.json (baseline, options, assumptions, SLA targets, weights)
        |
        v
 pilot calls per option x candidate model ---> usage ---> cost_usd() ---> $/call, $/ticket, $/month
        |
        v
 SLA check per candidate: simulated p50/p95 (seeded, common random numbers)
                          composite availability (serial x, redundant 1-(1-a)^2)
                          eval quality vs target
        |   cheapest model passing ALL SLAs
        v
 pillar calculator: efficiency | productivity | transformation | cost/ROI | SLA
        |
        v
 weighted scorecard -> ranking -> KPI tree (north star + guardrail)
        |
        v
 Claude Opus: 5-line executive summary from the computed scorecard JSON
```

## What the code demonstrates

| Section | Code | What to notice |
|---|---|---|
| 1 | `value_model.json` | All assumptions are explicit inputs, not buried constants |
| 2 | pilot calls + `cost_usd` | API cost per call from `usage` (including cache reads), times calls per ticket, times volume |
| 3 | `latency_percentiles`, `composite_availability`, `check_sla` | Deterministic LCG simulation identical in Python and TypeScript; model fallback is what makes 99.9% feasible; cheapest passing model is selected |
| 4 | `compute_pillars` | Realisation rate, risk-adjusted transformation value, ROI, payback, cost per resolved ticket |
| 5 | scorecard | Normalised pillar scores with stakeholder weights |
| 6 | KPI tree | North star + guardrail + driver KPIs with baseline -> target |
| 7 | Opus summary | The prompt carries pre-formatted numbers; the model is told not to compute |

## Run it

From the repository root:

```bash
.venv/Scripts/python "D1.6 - Aligning Solutions to Business Value Pillars/python/main.py"
node "D1.6 - Aligning Solutions to Business Value Pillars/typescript/main.ts"
```

All numbers are illustrative planning inputs; edit `data/value_model.json` to model your own case.

## Exam tips & traps

- **Hours saved are not dollars saved** until capacity is removed or redeployed; expect answers that apply a
  realisation rate and include run + build costs to beat answers that multiply hours by wage.
- **Token cost is rarely the deciding cost** for enterprise cases; people, integration, evaluation and platform
  usually dominate. But at very high volume (millions of calls), model choice and caching matter a lot.
- **SLAs are percentiles**: p95/p99 latency, not the mean. Latency grows with output length and tool hops;
  streaming improves time to first token, not total completion time.
- **Composite availability**: serial components multiply (each one lowers the total); redundancy is how you
  climb back above the target.
- **Pick the cheapest model that meets all SLAs** (quality, latency, cost) - "always use the most capable
  model" and "always use the cheapest" are both traps.
- **Transformation value is the least certain pillar**: risk-adjust it and keep it separate from efficiency.
- **Let the LLM narrate, not calculate**: compute in code, pass numbers in, instruct the model not to derive new ones.

### Practice questions

**Q1.** A customer-facing support agent must meet a p95 time-to-complete-reply of 6 s and at least 0.92
resolution quality on the eval set. Pilot results: Haiku 2.7 s / 0.86, Sonnet 4.8 s / 0.93, Opus 7.5 s /
0.96. Which model should the architect choose?

- A. Opus, because it has the highest quality
- B. Haiku, because it is the cheapest and fastest
- C. Sonnet, the cheapest model that meets both SLAs
- D. Opus with streaming enabled to bring the p95 completion time under 6 s

**Answer: C.** Only Sonnet satisfies both constraints. A violates the latency SLA, B violates the quality
SLA, and D is a trap: streaming shortens time to first token, not total completion time.

**Q2.** A business case claims about $1.28M a year of savings: 2,800 agent hours saved per month x $38 per
hour x 12 months. Finance challenges it. What is the most defensible revision?

- A. Assume full automation to increase the hours saved
- B. Apply a realisation rate for how freed hours become money (attrition, avoided hiring, redeployment), subtract run costs (API, infrastructure, people) and the build cost, and report ROI and payback
- C. Report only the Claude API token cost, since it is the only new cost
- D. Drop the efficiency pillar and argue only transformation value

**Answer: B.** It turns gross hours into realisable net value with the full cost picture. A inflates the
claim; C ignores the dominant costs; D swaps a measurable benefit for the least certain one.

## Further reading

- Claude docs: "Pricing" (model prices, prompt-caching multipliers, batch discount)
- Claude docs: "Prompt caching" and "Batch processing" (Message Batches API)
- Claude docs: "Reducing latency" and "Streaming messages"
- Claude docs: "Service tiers" and "Rate limits"
- Claude docs: "Define your success criteria"
- Related labs: D1.1 (value sizing), D2.1 (model selection), D3.3 (accuracy vs latency), D4.5 (cost optimisation), D6.3 (SLA tracking)
