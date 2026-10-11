"""
D6.5 - Lifecycle Phases - Discovery to Iteration

A lifecycle state machine for a Claude solution:

    discovery -> design -> build -> evaluation -> handoff -> monitoring -> iteration -> design ...
                                        ^   |
                                        +---+ (evaluation can send work back to build)

Each phase has exit criteria evaluated against an artifact registry, so a gate review is
computed (PASSED, or BLOCKED with reasons) rather than argued. The simulated project
(data/project_events.json) walks through every phase and shows:

  * a FAILED handoff gate (incomplete handoff package) that passes after fixes
  * monitoring signals (quality drift, cost creep, latency, model release, deprecation)
  * an iteration loop triggered by a model deprecation: impact assessment, API
    compatibility scan against the target models (breaking changes from the migration
    guide), config fixes, and a regression eval of old vs new model through the shared
    Claude client - round 1 finds a regression, a prompt patch fixes it, round 2 passes
  * the timeline, gate log and phase durations (labs_output/lifecycle_timeline.md)
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import PRICING, MockRequest, banner, get_client, section, text_of  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
DATA = LAB / "data"
OUT = LAB / "labs_output"


def load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


MOCK_OUTPUTS: dict[str, str] = load("mock_responses/regression_outputs.json")


def responder(req: MockRequest) -> str:
    """Offline stand-in: canned letters per (model, prompt version, case)."""
    pv = re.search(r"Prompt version: (\d+\.\d+\.\d+)", req.system_text)
    case = re.search(r"Case: (C-\d+)", req.last_user_text)
    if pv and case:
        for key in (f"{req.model}|{pv.group(1)}|{case.group(1)}", f"{req.model}|*|{case.group(1)}"):
            if key in MOCK_OUTPUTS:
                return MOCK_OUTPUTS[key]
    return "Dear Customer,\n\n(no canned reply)\n"


# ---------------------------------------------------------------------------
# Small helpers (mirrored in TypeScript)
# ---------------------------------------------------------------------------
_MISSING = object()


def get_path(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def fmt_val(metric: str, v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if metric.endswith("coverage") or metric.endswith("rate"):
        return f"{v * 100:.0f}%"
    if metric == "cost_per_task":
        return f"${v:.3f}"
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def py_json(v: Any) -> str:
    return json.dumps(v)


def words_of(text: str) -> int:
    return len(re.findall(r"\S+", text))


@dataclass
class Artifact:
    id: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Row:
    week: int
    date: str
    phase: str
    kind: str
    text: str


# ---------------------------------------------------------------------------
# Lifecycle engine: phases, gates, artifact registry, timeline
# ---------------------------------------------------------------------------
class Project:
    def __init__(self, lifecycle: dict[str, Any]) -> None:
        self.lc = lifecycle
        self.phases = {p["id"]: p for p in lifecycle["phases"]}
        y, m, d = (int(x) for x in lifecycle["start_date"].split("-"))
        self.start = date(y, m, d)
        self.phase = ""
        self.loop = 1
        self.artifacts: dict[str, Artifact] = {}
        self.rows: list[Row] = []
        self.gates: list[dict[str, Any]] = []
        self.segments: list[list[Any]] = []  # [phase, start_week, end_week, loop]

    def date_of(self, week: int) -> str:
        return (self.start + timedelta(days=7 * (week - 1))).isoformat()

    def log(self, week: int, kind: str, text: str) -> None:
        self.rows.append(Row(week, self.date_of(week), self.phase, kind, text))
        print(f"  W{week:02d} {self.date_of(week)} [{self.phase:<10}] {kind:<6} {text}")

    def enter(self, phase: str, week: int, first: bool = False) -> None:
        if phase == "design" and self.phase == "iteration":
            self.loop += 1  # iteration re-enters design: a new loop through the lifecycle
        if self.segments:
            self.segments[-1][2] = week
        self.phase = phase
        self.segments.append([phase, week if first else week + 1, week, self.loop])

    def upsert(self, art_id: str, status: str, metrics: Optional[dict[str, Any]] = None) -> None:
        art = self.artifacts.get(art_id) or Artifact(art_id, status)
        art.status = status
        art.metrics.update(metrics or {})
        self.artifacts[art_id] = art

    def check(self, c: dict[str, Any]) -> tuple[str, bool, str]:
        art = self.artifacts.get(c["artifact"])
        if art is None:
            return c["id"], False, f"{c['artifact']} missing"
        kind = c["check"]
        if kind == "exists":
            return c["id"], True, f"{c['artifact']} present"
        if kind == "approved":
            return c["id"], art.status == "approved", f"{c['artifact']} {art.status}"
        metric = c["metric"]
        val = art.metrics.get(metric)
        if kind in ("gte", "lte"):
            ok = val is not None and (val >= c["value"] if kind == "gte" else val <= c["value"])
            op = ">=" if kind == "gte" else "<="
            return c["id"], ok, f"{metric} {fmt_val(metric, val)} (need {op} {fmt_val(metric, c['value'])})"
        return c["id"], val is True, f"{metric} = {fmt_val(metric, val)}"

    def gate(self, week: int, to: str) -> bool:
        cur = self.phases[self.phase]
        allowed = to in self.lc["transitions"][self.phase]
        results = [self.check(c) for c in cur["exit"]]
        passed = allowed and all(ok for _, ok, _ in results)
        n_ok = sum(1 for _, ok, _ in results if ok)
        self.gates.append({"week": week, "from": self.phase, "to": to, "passed": passed, "allowed": allowed,
                           "results": results, "loop": self.loop})
        if passed:
            self.log(week, "GATE", f"{cur['name']} -> {self.phases[to]['name']}: PASSED ({n_ok}/{len(results)} exit criteria)")
            self.enter(to, week)
        else:
            reasons = [f"{cid} {detail}" for cid, ok, detail in results if not ok]
            if not allowed:
                reasons.insert(0, f"transition {self.phase} -> {to} not allowed")
            to_name = self.phases[to]["name"] if to in self.phases else to
            self.log(week, "GATE", f"{cur['name']} -> {to_name}: BLOCKED ({n_ok}/{len(results)}) - " + "; ".join(reasons))
        return passed


# ---------------------------------------------------------------------------
# Model-migration tooling: compatibility scan + fixes + cost impact
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    route: str
    rule: str
    severity: str
    message: str
    fix: Optional[dict[str, Any]]
    target: str


def scan(config: dict[str, Any], mode: str) -> list[Finding]:
    """mode 'migration': check legacy routes against their TARGET model's rules.
       mode 'current'  : check each route against the rules of the model it runs on."""
    findings: list[Finding] = []
    for r in config["routes"]:
        p = r["params"]
        if mode == "migration":
            if p["model"] not in config["migration_targets"]:
                continue
            target, migrating = config["migration_targets"][p["model"]], True
        else:
            target, migrating = p["model"], False
        for rule in config["rules"]:
            if "*" not in rule["targets"] and target not in rule["targets"]:
                continue
            w = rule["when"]
            if w["op"] == "migrating":
                hit = migrating
            else:
                val = get_path(p, w["path"])
                hit = (val is not _MISSING and (
                    (w["op"] == "eq" and val == w["value"]) or (w["op"] == "in" and val in w["value"]) or w["op"] == "present"))
            if hit:
                findings.append(Finding(r["route"], rule["id"], rule["severity"], rule["message"], rule["fix"], target))
    return findings


def apply_fixes(config: dict[str, Any], findings: list[Finding]) -> None:
    routes = {r["route"]: r for r in config["routes"]}
    for f in findings:
        if f.severity != "BLOCKS" or not f.fix:
            continue
        p = routes[f.route]["params"]
        if f.fix.get("set_model"):
            p["model"] = f.target
        for key in f.fix.get("unset", []):
            p.pop(key, None)
        for key, val in f.fix.get("set", {}).items():
            p[key] = copy.deepcopy(val)


def price_of(model: str, config: dict[str, Any]) -> dict[str, float]:
    return PRICING[model] if model in PRICING else config["legacy_pricing"][model]


def monthly_cost(route: dict[str, Any], model: str, config: dict[str, Any]) -> float:
    p, t = price_of(model, config), route["tokens"]
    factor = config["tokenizer_factor"].get(model, 1.0)
    per_call = (t["input"] * p["input"] + t["cache_read"] * p["cache_read"] + t["cache_write"] * p["cache_write"]
                + t["output"] * p["output"]) * factor / 1_000_000
    return per_call * route["monthly_calls"]


# ---------------------------------------------------------------------------
# Regression eval through the shared Claude client
# ---------------------------------------------------------------------------
def grade(text: str, case: dict[str, Any]) -> tuple[bool, list[str], int]:
    low = text.lower()
    n = words_of(text)
    fails = [f"missing '{t}'" for t in case["must_include"] if t.lower() not in low]
    fails += [f"contains '{t}'" for t in case["must_not_include"]
              if re.search(r"(?<![a-z])" + re.escape(t.lower()) + r"(?![a-z])", low)]
    if n > case["max_words"]:
        fails.append(f"{n} words > {case['max_words']}")
    return not fails, fails, n


def run_suite(client: anthropic.Anthropic, suite: dict[str, Any], model: str, prompt: str, migrated: bool,
              config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for case in suite["cases"]:
        extra: dict[str, Any] = {}
        if migrated:  # the migrated route config: adaptive thinking, effort low
            extra = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
        resp = client.messages.create(
            model=model, max_tokens=800, system=suite["prompts"][prompt],
            messages=[{"role": "user", "content": f"Case: {case['id']}\n{case['input']}"}], **extra,
        )
        ok, fails, n = grade(text_of(resp), case)
        p = price_of(model, config)
        cost = (resp.usage.input_tokens * p["input"] + resp.usage.output_tokens * p["output"]) / 1_000_000
        results[case["id"]] = {"ok": ok, "fails": fails, "words": n, "out_tokens": resp.usage.output_tokens, "cost": cost}
    return results


def main() -> None:
    banner("D6.5 - Lifecycle Phases - Discovery to Iteration")
    lifecycle = load("lifecycle.json")
    events = load("project_events.json")["events"]
    package = load("handoff_package.json")
    monitoring = load("monitoring.json")
    config = load("model_config.json")
    suite = load("regression_suite.json")
    client = get_client(responder)

    section("1. Lifecycle model: phases, exit criteria, transitions")
    for p in lifecycle["phases"]:
        nxt = ", ".join(lifecycle["transitions"][p["id"]])
        print(f"  {p['name']:<18} -> {nxt:<18} exit: " + "; ".join(f"{c['id']} {c['text']}" for c in p["exit"]))
    print("  Gates are computed from the artifact registry: evidence, not opinion.")

    section("2. Simulated project timeline")
    proj = Project(lifecycle)
    pkg = copy.deepcopy(package["v1"])
    pkg_history: list[tuple[str, dict[str, Any]]] = []
    mon_rows: list[dict[str, Any]] = []
    baseline: dict[str, float] = {}
    ticket_reasons: list[str] = []
    quality_hist: list[float] = []
    impact: dict[str, Any] = {}
    scan_before: list[Finding] = []
    scan_after: list[Finding] = []
    config_before = copy.deepcopy(config["routes"])
    eval_rounds: list[dict[str, Any]] = []
    frozen_baseline: Optional[dict[str, dict[str, Any]]] = None
    baseline_source = ""
    th = monitoring["thresholds"]

    def package_metrics(p: dict[str, Any]) -> dict[str, Any]:
        alerts, covered = p["runbook"]["alerts"], p["runbook"]["covered"]
        return {
            "runbook_coverage": sum(1 for a in alerts if a in covered) / len(alerts),
            "baseline_recorded": p["eval"]["baseline_recorded"],
            "dashboards": len(p["dashboards"]),
            "owners_missing": sum(1 for v in p["owners"].values() if not v),
            "prompt_registry_pinned": p["prompt_registry"]["pinned"],
            "model_inventory_ok": all(m["deprecation_watch"] for m in p["model_inventory"]),
            "ops_signoff": p["ops_signoff"],
        }

    for ev in events:
        wk, kind = ev["week"], ev["type"]
        if kind == "start":
            proj.enter(ev["phase"], wk, first=True)
            proj.log(wk, "START", ev["note"])
        elif kind == "artifact":
            proj.upsert(ev["id"], ev["status"], ev.get("metrics"))
            proj.log(wk, "ART", f"{ev['id']} ({ev['status']}): {ev['note']}")
        elif kind == "gate":
            proj.gate(wk, ev["to"])
        elif kind == "handoff_package":
            met = package_metrics(pkg)
            pkg_history.append(("v1 (submitted)", met))
            proj.upsert("handoff_package", "submitted", met)
            missing = [a for a in pkg["runbook"]["alerts"] if a not in pkg["runbook"]["covered"]]
            proj.log(wk, "ART", f"handoff_package v1: {ev['note']} (runbook missing {', '.join(missing)}; "
                                f"{met['dashboards']} dashboards; {met['owners_missing']} owner missing)")
        elif kind == "handoff_fix":
            fx = package["fix"]
            pkg["runbook"]["covered"] += fx["runbook_covered_add"]
            pkg["dashboards"] += fx["dashboards_add"]
            pkg["owners"].update(fx["owners"])
            for m in pkg["model_inventory"]:
                m["deprecation_watch"] = fx["deprecation_watch"]
            pkg["ops_signoff"] = fx["ops_signoff"]
            met = package_metrics(pkg)
            pkg_history.append(("v2 (after fixes)", met))
            proj.upsert("handoff_package", "approved", met)
            proj.log(wk, "ART", f"handoff_package v2: {ev['note']}")
        elif kind == "handoff_delta":
            dl = package["delta"]
            pkg["model_inventory"] = copy.deepcopy(dl["model_inventory"])
            pkg["prompt_registry"]["prompts"].update(dl["prompt_registry"])
            met = package_metrics(pkg)
            pkg_history.append(("v3 (migration delta)", met))
            proj.upsert("handoff_package", "approved", met)
            proj.log(wk, "ART", f"handoff_package v3: {ev['note']} (inventory: "
                                + ", ".join(f"{m['route']}={m['model']}" for m in pkg["model_inventory"]) + ")")
        elif kind == "monitoring_week":
            if not baseline:
                er = proj.artifacts["eval_report"].metrics
                baseline = {"quality": er["pass_rate"], "cost": er["cost_per_task"]}
                proj.log(wk, "NOTE", f"Monitoring baseline from evaluation: quality {baseline['quality']:.3f}, "
                                     f"cost/task ${baseline['cost']:.3f}")
            row = next(x for x in monitoring["weeks"] if x["week"] == wk)
            quality_hist.append(row["quality"])
            roll = sum(quality_hist[-2:]) / len(quality_hist[-2:])
            q_stat = ("TRIGGER" if roll < baseline["quality"] - th["quality_drop"]
                      else "WATCH" if row["quality"] < baseline["quality"] - th["quality_watch"] else "OK")
            ratio = row["cost_per_task"] / baseline["cost"]
            c_stat = "TRIGGER" if ratio > th["cost_trigger"] else "WATCH" if ratio > th["cost_watch"] else "OK"
            l_stat = "INCIDENT" if row["p95_s"] > th["p95_max_s"] else "OK"
            ext: list[str] = []
            triggers: list[str] = []
            if q_stat == "TRIGGER":
                triggers.append(f"SIG-Q quality drift (2-week mean {roll:.3f})")
            if c_stat == "TRIGGER":
                triggers.append(f"SIG-C cost x{ratio:.2f}")
            in_use = [r["params"]["model"] for r in config["routes"]]
            for e in monitoring["external_events"]:
                if e["week"] != wk:
                    continue
                if e["type"] == "deprecation_notice" and e["model"] in in_use:
                    y, m, d = (int(x) for x in e["retirement_date"].split("-"))
                    y0, m0, d0 = (int(x) for x in proj.date_of(wk).split("-"))
                    days = (date(y, m, d) - date(y0, m0, d0)).days
                    status = "TRIGGER" if days <= th["deprecation_horizon_days"] else "WATCH"
                    ext.append(f"SIG-D {e['model']} retires {e['retirement_date']} ({days} days) [{status}]")
                    if status == "TRIGGER":
                        triggers.append(f"SIG-D {e['model']} retires {e['retirement_date']} ({days} days): mandatory migration")
                elif e["type"] == "model_release":
                    succ = [f"{new} (successor of {old})" for new, old in e["successor_of"].items() if old in in_use]
                    if succ:
                        ext.append(f"SIG-R released: {', '.join(succ)} [EVALUATE]")
                        triggers.append("SIG-R successor models released: evaluate upgrade (optional)")
            mon_rows.append({"week": wk, "quality": row["quality"], "roll": roll, "q": q_stat, "cost": row["cost_per_task"],
                             "ratio": ratio, "c": c_stat, "p95": row["p95_s"], "l": l_stat, "note": row.get("note", ""), "ext": ext})
            proj.log(wk, "SIGNAL", f"quality {row['quality']:.3f} [{q_stat}] cost ${row['cost_per_task']:.3f} x{ratio:.2f} [{c_stat}] "
                                   f"p95 {row['p95_s']:.1f}s [{l_stat}]" + (f" - {row['note']}" if row.get("note") else ""))
            for x in ext:
                proj.log(wk, "SIGNAL", x)
            mandatory = [t for t in triggers if "SIG-R" not in t]
            if mandatory and "iteration_ticket" not in proj.artifacts:
                if c_stat == "WATCH":
                    triggers.append(f"SIG-C cost watch x{ratio:.2f} (fold into the same iteration)")
                ticket_reasons = triggers
                proj.upsert("iteration_ticket", "open", {"reasons": len(triggers)})
                proj.log(wk, "EVENT", "Iteration ticket ITR-001 raised: " + " | ".join(triggers))
        elif kind == "impact_assessment":
            pre = scan(config, "migration")
            rows = []
            for r in config["routes"]:
                old = r["params"]["model"]
                new = config["migration_targets"].get(old, old)
                rows.append((r["route"], old, new, monthly_cost(r, old, config), monthly_cost(r, new, config)))
            before, after = sum(x[3] for x in rows), sum(x[4] for x in rows)
            impact = {"rows": rows, "before": before, "after": after,
                      "blocking": sum(1 for f in pre if f.severity == "BLOCKS"), "tune": sum(1 for f in pre if f.severity == "TUNE")}
            proj.upsert("impact_assessment", "approved", {"blocking": impact["blocking"]})
            proj.log(wk, "ART", f"impact_assessment: {impact['blocking']} breaking changes, {impact['tune']} tuning tasks; "
                                f"monthly model cost ${before:,.2f} -> ${after:,.2f}")
            for route, old, new, cb, ca in rows:
                print(f"           {route:<15} {old:<17} -> {new:<18} ${cb:>8,.2f} -> ${ca:>8,.2f}/month")
        elif kind == "compat_scan":
            findings = scan(config, ev["mode"])
            blocking = sum(1 for f in findings if f.severity == "BLOCKS")
            tune = sum(1 for f in findings if f.severity == "TUNE")
            if ev["mode"] == "migration":
                scan_before = findings
            proj.upsert("compat_scan", "approved", {"blocking": blocking, "tune": tune})
            proj.log(wk, "SCAN", f"{ev['note']}: {blocking} blocking, {tune} tuning")
            for f in findings:
                print(f"           {f.severity:<6} {f.route:<15} {f.rule:<8} {f.message}")
        elif kind == "apply_migration_fixes":
            apply_fixes(config, scan(config, "migration"))
            scan_after = scan(config, "current")
            blocking = sum(1 for f in scan_after if f.severity == "BLOCKS")
            tune = sum(1 for f in scan_after if f.severity == "TUNE")
            proj.upsert("compat_scan", "approved", {"blocking": blocking, "tune": tune})
            proj.log(wk, "SCAN", f"{ev['note']}; re-scan: {blocking} blocking, {tune} tuning task(s) left")
            for old_r, new_r in zip(config_before, config["routes"]):
                print(f"           {old_r['route']}: {py_json(old_r['params'])}")
                print(f"           {' ' * len(old_r['route'])}  -> {py_json(new_r['params'])}")
        elif kind == "regression_eval":
            route = next(r for r in config["routes"] if r["route"] == "draft_letter")
            new_model = route["params"]["model"]
            legacy_model = suite["recorded_baseline"]["model"]
            if frozen_baseline is None:
                # Freeze the reference ONCE: later rounds compare against the same baseline outputs.
                try:
                    frozen_baseline = run_suite(client, suite, legacy_model, suite["recorded_baseline"]["prompt"], False, config)
                    baseline_source = f"{legacy_model} @ prompt {suite['recorded_baseline']['prompt']} (re-run)"
                except anthropic.APIStatusError as err:  # e.g. the legacy model is already retired (404)
                    rec = suite["recorded_baseline"]
                    frozen_baseline = {cid: {"ok": ok, "fails": [] if ok else ["recorded failure"], "words": 0,
                                             "out_tokens": 0, "cost": 0.0} for cid, ok in rec["results"].items()}
                    baseline_source = f"recorded W{rec['recorded_week']} results ({type(err).__name__} calling {legacy_model})"
            cand = run_suite(client, suite, new_model, ev["prompt"], True, config)
            regressions = [c for c in cand if frozen_baseline[c]["ok"] and not cand[c]["ok"]]
            improved = [c for c in cand if not frozen_baseline[c]["ok"] and cand[c]["ok"]]
            n_pass = sum(1 for c in cand.values() if c["ok"])
            b_pass = sum(1 for c in frozen_baseline.values() if c["ok"])
            eval_rounds.append({"round": ev["round"], "prompt": ev["prompt"], "model": new_model, "cand": cand,
                                "regressions": regressions, "improved": improved})
            proj.upsert("eval_report", "approved", {"pass_rate": n_pass / len(cand), "regressions": len(regressions)})
            proj.log(wk, "EVAL", f"round {ev['round']}: {new_model} @ prompt {ev['prompt']} {n_pass}/{len(cand)} vs baseline "
                                 f"{b_pass}/{len(cand)}; regressions: {', '.join(regressions) or 'none'}; improved: {', '.join(improved) or 'none'}"
                                 + (f" - {ev['note']}" if ev.get("note") else ""))
            print(f"           baseline: {baseline_source}")
            print(f"           {'case':<6}{'type':<22}{'baseline':<10}{'candidate':<11}{'verdict':<12}detail")
            for case in suite["cases"]:
                cid = case["id"]
                b, c = frozen_baseline[cid], cand[cid]
                verdict = "REGRESSION" if cid in regressions else "improved" if cid in improved else "same"
                print(f"           {cid:<6}{case['type']:<22}{'PASS' if b['ok'] else 'FAIL':<10}{'PASS' if c['ok'] else 'FAIL':<11}"
                      f"{verdict:<12}{'; '.join(c['fails'] or b['fails'])}")
            avg_out = sum(c["out_tokens"] for c in cand.values()) / len(cand)
            avg_cost = sum(c["cost"] for c in cand.values()) / len(cand)
            print(f"           candidate avg output {avg_out:.0f} tokens, ${avg_cost:.5f}/case (offline token counts are estimates)")
        elif kind == "note":
            proj.log(wk, "NOTE", ev["note"])

    last_week = max(e["week"] for e in events)
    proj.segments[-1][2] = last_week

    section("3. Phase timeline (Gantt) and gate summary")
    print(f"  {'':<28}" + "".join(str(w % 10) for w in range(1, last_week + 1)))
    for ph, s, e, loop in proj.segments:
        label = f"{proj.phases[ph]['name']}" + (f" (loop {loop})" if loop > 1 else "")
        print(f"  {label:<28}" + "".join("#" if s <= w <= e else "." for w in range(1, last_week + 1)) + f"  W{s:02d}-W{e:02d}")
    passed = sum(1 for g in proj.gates if g["passed"])
    print(f"\n  Gates: {len(proj.gates)} reviewed, {passed} passed, {len(proj.gates) - passed} blocked; "
          f"iteration loops: {proj.loop - 1}; current phase: {proj.phases[proj.phase]['name']}")
    for g in proj.gates:
        if not g["passed"]:
            print(f"  Blocked gate W{g['week']:02d} {g['from']} -> {g['to']}: " +
                  "; ".join(f"{cid} {d}" for cid, ok, d in g["results"] if not ok))

    report = render_report(lifecycle, proj, pkg_history, mon_rows, monitoring, ticket_reasons, impact, scan_before,
                           config_before, config, scan_after, eval_rounds, frozen_baseline or {}, baseline_source, suite, last_week)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lifecycle_timeline.md").write_text(report, encoding="utf-8")
    print(f"\n  wrote labs_output/lifecycle_timeline.md ({len(report.splitlines())} lines); sample in examples/")


def render_report(lifecycle, proj: Project, pkg_history, mon_rows, monitoring, ticket_reasons, impact, scan_before,
                  config_before, config, scan_after, eval_rounds, frozen, baseline_source, suite, last_week) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Lifecycle timeline - {lifecycle['project']} (simulated)")
    a("")
    a(f"> {lifecycle['note']}")
    a("")
    a("## 1. Phase model")
    a("")
    a("| Phase | Goal | Exit criteria (gate) | Reviewers | Next |")
    a("|---|---|---|---|---|")
    for p in lifecycle["phases"]:
        a(f"| {p['name']} | {p['goal']} | " + "<br>".join(f"{c['id']} {c['text']}" for c in p["exit"])
          + f" | {', '.join(p['reviewers'])} | {', '.join(lifecycle['transitions'][p['id']])} |")
    a("")
    a("## 2. Timeline")
    a("")
    a("| Week | Date | Phase | Type | Event |")
    a("|---|---|---|---|---|")
    for r in proj.rows:
        a(f"| W{r.week:02d} | {r.date} | {r.phase} | {r.kind} | {r.text.replace('|', '/')} |")
    a("")
    a("## 3. Gate reviews")
    a("")
    for g in proj.gates:
        a(f"### W{g['week']:02d} {proj.phases[g['from']]['name']} -> {proj.phases[g['to']]['name']}: "
          f"{'PASSED' if g['passed'] else 'BLOCKED'}{' (loop ' + str(g['loop']) + ')' if g['loop'] > 1 else ''}")
        a("")
        a("| Criterion | Result | Evidence |")
        a("|---|---|---|")
        crit = {c["id"]: c["text"] for c in proj.phases[g["from"]]["exit"]}
        for cid, ok, detail in g["results"]:
            a(f"| {cid} {crit[cid]} | {'pass' if ok else '**FAIL**'} | {detail} |")
        a("")
    a("## 4. Handoff package validation")
    a("")
    metrics = list(pkg_history[0][1].keys()) if pkg_history else []
    a("| Check | " + " | ".join(label for label, _ in pkg_history) + " |")
    a("|---|" + "---|" * len(pkg_history))
    for mk in metrics:
        a(f"| {mk} | " + " | ".join(fmt_val(mk, met[mk]) for _, met in pkg_history) + " |")
    a("")
    a("## 5. Monitoring signals")
    a("")
    a("| Week | Quality (2-wk mean) | Quality signal | Cost/task (x baseline) | Cost signal | p95 | Latency signal | Notes |")
    a("|---|---|---|---|---|---|---|---|")
    for m in mon_rows:
        notes = "; ".join([x for x in [m["note"]] if x] + m["ext"])
        a(f"| W{m['week']:02d} | {m['quality']:.3f} ({m['roll']:.3f}) | {m['q']} | ${m['cost']:.3f} (x{m['ratio']:.2f}) | {m['c']} | "
          f"{m['p95']:.1f} s | {m['l']} | {notes or '-'} |")
    a("")
    a("Signal rules: " + "; ".join(f"{s['id']} {s['name']}: {s['rule']} -> {s['action']}" for s in monitoring["signals"]) + ".")
    a("")
    a("Iteration ticket ITR-001 reasons:")
    a("")
    for t in ticket_reasons:
        a(f"- {t}")
    a("")
    a("## 6. Iteration: model migration")
    a("")
    a("### Impact assessment")
    a("")
    a("| Route | Current model | Target model | Monthly cost now | Monthly cost after |")
    a("|---|---|---|---:|---:|")
    for route, old, new, cb, ca in impact.get("rows", []):
        a(f"| {route} | `{old}` | `{new}` | ${cb:,.2f} | ${ca:,.2f} |")
    if impact:
        a(f"| **Total** | | | **${impact['before']:,.2f}** | **${impact['after']:,.2f}** |")
    a("")
    a("### API compatibility scan (against the target models)")
    a("")
    a("| Route | Rule | Severity | Finding |")
    a("|---|---|---|---|")
    for f in scan_before:
        a(f"| {f.route} | {f.rule} | {f.severity} | {f.message} |")
    a("")
    a("Config before -> after the fixes:")
    a("")
    a("```")
    for old_r, new_r in zip(config_before, config["routes"]):
        a(f"{old_r['route']}:")
        a(f"  before: {py_json(old_r['params'])}")
        a(f"  after:  {py_json(new_r['params'])}")
    a("```")
    a("")
    a("Remaining after the fixes: " + ("; ".join(f"{f.severity} {f.route} {f.rule}: {f.message}" for f in scan_after) or "none") + ".")
    a("")
    a("### Regression eval (old vs new model)")
    a("")
    a(f"Frozen baseline: {baseline_source}. Programmatic checks per case (must include / must not include / word budget).")
    a("")
    for rnd in eval_rounds:
        cand = rnd["cand"]
        n_pass = sum(1 for c in cand.values() if c["ok"])
        a(f"**Round {rnd['round']}** - `{rnd['model']}` with prompt {rnd['prompt']}: {n_pass}/{len(cand)} pass; "
          f"regressions: {', '.join(rnd['regressions']) or 'none'}; improved: {', '.join(rnd['improved']) or 'none'}")
        a("")
        a("| Case | Type | Baseline | Candidate | Words | Verdict | Detail |")
        a("|---|---|---|---|---:|---|---|")
        for case in suite["cases"]:
            cid = case["id"]
            b, c = frozen[cid], cand[cid]
            verdict = "REGRESSION" if cid in rnd["regressions"] else "improved" if cid in rnd["improved"] else "same"
            a(f"| {cid} | {case['type']} | {'PASS' if b['ok'] else 'FAIL'} | {'PASS' if c['ok'] else 'FAIL'} | {c['words']} | "
              f"{verdict} | {'; '.join(c['fails'] or b['fails']) or '-'} |")
        a("")
    a("Decision: switch only after a round with zero regressions against the frozen baseline; then canary, then remove "
      "the legacy model ids. Fix regressions in the prompt or config - never by loosening the check.")
    a("")
    a("## 7. Phase durations")
    a("")
    a("| Phase | Loop | Weeks | Duration |")
    a("|---|---:|---|---:|")
    for ph, s, e, loop in proj.segments:
        a(f"| {proj.phases[ph]['name']} | {loop} | W{s:02d}-W{e:02d} | {e - s + 1} |")
    a("")
    a("```")
    a(f"{'':<28}" + "".join(str(w % 10) for w in range(1, last_week + 1)))
    for ph, s, e, loop in proj.segments:
        label = f"{proj.phases[ph]['name']}" + (f" (loop {loop})" if loop > 1 else "")
        a(f"{label:<28}" + "".join("#" if s <= w <= e else "." for w in range(1, last_week + 1)))
    a("```")
    a("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
