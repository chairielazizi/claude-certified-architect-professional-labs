# D5.5 - Ethical AI: Bias, Fairness, Transparency

> **Exam objective:** Address ethical AI considerations (bias, fairness, transparency).

## Why it matters

When an LLM influences decisions about people - hiring, lending, benefits, moderation -
"it scored 92% accurate" is not enough. A model can be accurate in aggregate while
systematically disadvantaging a group. An architect must be able to **measure** bias
(not just assert its absence), **mitigate** it in the pipeline, and make the system
**transparent and accountable** - disclosure, explanations, and a model/system card.
These are increasingly legal requirements (EU AI Act, US EEOC guidance on automated
hiring), not just good practice.

## Key concepts

### Counterfactual testing beats aggregate accuracy

Hold everything constant and flip **only** a protected attribute (a fictional name,
pronoun, or age marker). If the decision changes, the model is treating the attribute as
a cause - **disparate treatment** - even if overall accuracy looks fine. Counterfactual
pairs expose exactly what aggregate metrics hide.

### Fairness metrics (compute them, don't eyeball them)

Let privileged = GroupA, unprivileged = GroupB; `qualified` is the ground-truth label.

| Metric | Formula | Fair when |
|---|---|---|
| **Demographic parity difference (DPD)** | P(advance\|A) - P(advance\|B) | near 0 (e.g., \|DPD\| <= 0.10) |
| **Equal opportunity difference (EOD)** | TPR(A) - TPR(B), over *qualified* only | near 0 (e.g., \|EOD\| <= 0.10) |
| **Disparate impact ratio (DIR)** | P(advance\|B) / P(advance\|A) | >= 0.80 (**four-fifths rule**) |

- **Demographic parity** asks for equal selection rates. **Equal opportunity** asks for
  equal true-positive rates among the genuinely qualified - often the fairer target when
  base rates differ. They can conflict; choosing which to optimize is a value decision an
  architect must make explicit, not a formula.
- The **four-fifths rule** (DIR >= 0.8) is the classic US adverse-impact screen.

### Mitigation (defense in depth, again)

1. **Attribute redaction** - strip the protected attribute (and proxies) before the
   model. If the model never sees group, it cannot condition on it. (Beware *proxies*:
   zip code, school, name - redaction must cover indirect signals too.)
2. **Prompt-level instruction** - tell the model to ignore protected attributes. Useful,
   but weaker than redaction; do not rely on it alone.
3. **Rubric-based, criteria-cited decisions** - force the decision onto explicit,
   reviewable criteria, and have the model cite which criteria drove it.
4. **Human review for adverse decisions** - route every reject to a person (ties back to
   D5.3 HITL); adverse outcomes are where bias does the most harm.

In the lab, redaction + rubric drives DPD and EOD to 0 and DIR to 1.0 (four-fifths PASS).

### Transparency

- **AI disclosure** - tell affected users a decision was AI-assisted.
- **Decision explanations** - cite the specific criteria behind each decision (the
  `cited_criteria` field makes "group_adjusted_bar" visible when bias is present).
- **Model/system card** - document intended use, limitations, evaluation results (the
  fairness metrics), and contacts. Generated here via **structured output** so it is
  consistent and machine-checkable.

### Claude-platform note

- Structured output (`output_config.format` / `messages.parse`) gives schema-valid
  decisions and system cards - the right tool for auditable, consistent fields.
- No forced tool use on 5.5-class models; steer with the prompt. Keep protected
  attributes out of the prompt entirely rather than trusting the model to ignore them.

## Architecture

```
  candidates (fictional; group, score, qualified)
        |
        +--> COUNTERFACTUAL: matched pairs, flip only the attribute --> count flips
        |
        +--> BIASED pipeline (attribute visible) ----+
        |                                             +--> outcomes --> FAIRNESS METRICS
        +--> MITIGATED pipeline                       |                 DPD / EOD / DIR
             (redact attribute -> fairness prompt ----+                 |
              -> rubric decision -> cite criteria)                       v
                                                                   FAIRNESS REPORT
             adverse (reject) --> human review queue
             outputs --> AI disclosure + cited criteria + model/system card (structured)
```

## What the code demonstrates

| Step | Code (Python / TypeScript) |
|---|---|
| Screen one candidate (redact in mitigated mode) | `screen` |
| Counterfactual pairs + flip count | `(a)` section in `main` |
| DPD / EOD / DIR | `fairness_metrics` / `fairnessMetrics` |
| Fairness report | `print_fairness_report` / `printFairnessReport` |
| Human review for adverse decisions | `(c)` section in `main` |
| Disclosure + cited criteria | `AI_DISCLOSURE`, `cited_criteria` |
| System card via structured output | `generate_system_card` / `generateSystemCard` |

## Run it

From the repo root:

```bash
.venv/Scripts/python "D5.5 - Ethical AI - Bias, Fairness, Transparency/python/main.py"
node "D5.5 - Ethical AI - Bias, Fairness, Transparency/typescript/main.ts"
```

Offline, deterministic, no API key. All people are clearly fictional.

## Exam tips & traps

- **Accuracy is not fairness.** A model can be highly accurate overall and still fail the
  four-fifths rule for a subgroup - measure per-group.
- **Counterfactual testing is the sharpest bias probe** - flip only the attribute and
  look for decision changes.
- **Know the three metrics and what each asks**: parity (equal selection), equal
  opportunity (equal TPR among qualified), disparate impact (four-fifths ratio). They can
  conflict.
- **Redaction beats "please ignore it"** - and must cover *proxies*, not just the literal
  attribute.
- **Route adverse decisions to humans** - that is where bias causes the most harm and
  where oversight matters most.
- **Transparency is a deliverable**: disclosure, cited criteria, and a model/system card
  (intended use, limitations, evaluation results, contacts).
- **Document which fairness definition you chose and why** - it is a value decision, and
  optimizing one metric can worsen another.

### Practice questions

**Q1.** A screening model advances 75% of GroupA and 55% of GroupB applicants. What is
the disparate impact ratio, and does it pass the four-fifths rule?

- A. 0.20 difference, so it passes.
- B. DIR = 0.55 / 0.75 ≈ 0.73, which is below 0.80 - it **fails** the four-fifths rule.
- C. DIR = 0.75 / 0.55 ≈ 1.36, which passes.
- D. You cannot compute it without accuracy.

**Answer: B.** DIR = rate(unprivileged)/rate(privileged) = 0.55/0.75 ≈ 0.73 < 0.80, so
it fails adverse-impact screening. (A confuses DPD with DIR; C inverts the ratio.)

**Q2.** Your team adds "Do not consider the applicant's group" to the system prompt but
still sends the applicant's name and neighborhood. Counterfactual tests still show flips.
What is the most effective next step?

- A. Increase the model's effort level.
- B. Redact the protected attribute **and its proxies** (name, neighborhood) before the model, decide from a blind rubric, and keep measuring counterfactual flips.
- C. Force a tool call to log the decision.
- D. Raise `max_tokens` so the model can reason more.

**Answer: B.** A prompt instruction is weak if the model can still see the attribute or
its proxies; redact the inputs (including proxies), decide from a blind rubric, and
re-run the counterfactual tests.

## Further reading

- Anthropic docs: "Structured outputs" (system-card / decision schemas), "Reduce bias"
- Anthropic Responsible Scaling Policy and usage policies
- Model Cards (Mitchell et al.); Datasheets for Datasets (Gebru et al.)
- Fairness definitions: demographic parity, equalized odds / equal opportunity (Hardt et al.)
- EEOC Uniform Guidelines (four-fifths rule); EU AI Act (high-risk systems, transparency)
