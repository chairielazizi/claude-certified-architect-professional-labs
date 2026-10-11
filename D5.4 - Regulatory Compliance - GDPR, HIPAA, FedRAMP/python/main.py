"""
D5.4 - Regulatory Compliance: GDPR, HIPAA, FedRAMP (Python)

EDUCATIONAL ONLY - NOT LEGAL ADVICE. See README disclaimer. Platform authorization
claims are phrased as "verify ..." on purpose; confirm current BAA / FedRAMP / data-
residency arrangements with the provider.

Four parts:
  (a) a data-classification + compliance POLICY ENGINE: workload profile -> required
      controls + platform/residency constraints (inference_geo, ZDR/retention);
  (b) PHI/PII DE-IDENTIFICATION before the model call (regex + dictionary detectors
      + a reversible tokenization vault that stays inside the trust boundary), the
      Claude call on de-identified text, then re-identification;
  (c) GDPR DATA-SUBJECT RIGHTS: right to erasure across logs/vector index/caches, and
      data export;
  (d) an IMMUTABLE AUDIT LOG with hash chaining (+ tamper detection).

Only the LLM call goes through the shared client; everything else is real code.

Run (from repo root):
    .venv/Scripts/python "D5.4 - Regulatory Compliance - GDPR, HIPAA, FedRAMP/python/main.py"
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "python"))
from claude_client import (  # noqa: E402
    MODEL_OPUS,
    MockRequest,
    banner,
    get_client,
    section,
)

DATA = Path(__file__).resolve().parents[1] / "data"


# ===========================================================================
# (a) Compliance policy engine
# ===========================================================================
@dataclass
class Control:
    name: str
    required: bool
    note: str


def evaluate_policy(w: dict[str, Any]) -> list[Control]:
    """Map a workload profile to required controls. Deterministic, auditable rules.
    Authorization-dependent items are phrased as 'verify'."""
    dt = set(w["data_types"])
    region = w["user_region"]
    gov = w["government_customer"]
    controls: list[Control] = []

    # Baseline (every workload)
    controls.append(Control("Encryption in transit + at rest", True, "TLS to the API; encrypt stored logs/indexes"))
    controls.append(Control("Access controls (least privilege, RBAC)", True, "scope who can read prompts/outputs/logs"))
    controls.append(Control("Immutable audit logging", True, "hash-chained, tamper-evident (see part d)"))

    sensitive = bool(dt & {"PHI", "PII", "CUI", "PCI"})
    if sensitive:
        controls.append(Control("Data minimization", True, "send only what the task needs; strip the rest"))
        controls.append(Control("De-identification / pseudonymization", True, "tokenize identifiers before the model (see part b)"))

    # HIPAA (PHI)
    if "PHI" in dt:
        controls.append(Control("HIPAA: Business Associate Agreement (BAA)", True, "verify a signed BAA is in place with the provider before sending PHI"))
        controls.append(Control("HIPAA: minimum necessary standard", True, "limit PHI to the minimum necessary"))
        controls.append(Control("HIPAA: breach-notification process", True, "documented IR + notification workflow"))

    # GDPR (PII of EU subjects)
    if "PII" in dt and region == "EU":
        controls.append(Control("GDPR: lawful basis + DPA", True, "document lawful basis; Data Processing Addendum with the provider"))
        controls.append(Control("GDPR: data residency (inference_geo='eu')", True, "pin inference geography - 1P / Claude Platform on AWS; on Bedrock/Vertex/Foundry residency is the platform region, inference_geo is unavailable (verify)"))
        controls.append(Control("GDPR: data-subject rights (erasure, export, access)", True, "implement DSR workflows (see part c)"))
        controls.append(Control("GDPR: storage limitation", True, f"retention {w['retention_days']}d must match a documented purpose"))

    # FedRAMP / government (CUI or government customer)
    if "CUI" in dt or gov:
        controls.append(Control("FedRAMP: authorized environment", True, "use a government-authorized cloud (e.g., Bedrock GovCloud / Vertex gov) - verify the current FedRAMP authorization level with the provider"))
        controls.append(Control("FedRAMP: US data residency", True, "keep data + inference in US/gov regions - verify per platform"))
        controls.append(Control("FedRAMP: continuous monitoring + boundary", True, "ConMon, SSP, authorization boundary documented"))

    # PCI DSS
    if "PCI" in dt:
        controls.append(Control("PCI DSS: tokenize PAN, minimize scope", True, "never send raw card data; tokenize and reduce CDE scope"))

    # Retention / ZDR
    if w["needs_zdr"]:
        controls.append(Control("Zero data retention (ZDR) arrangement", True, "ZDR only if expressly authorized by Anthropic; note Fable 5.1 requires 30-day retention and is NOT available under ZDR unless authorized - choose a different model for ZDR workloads"))
    else:
        controls.append(Control("Retention policy", True, f"delete logs/indexes after {w['retention_days']}d per policy"))

    return controls


def recommend_platform(w: dict[str, Any]) -> tuple[str, str]:
    dt = set(w["data_types"])
    if "CUI" in dt or w["government_customer"]:
        return ("Government cloud (Bedrock GovCloud / Vertex gov)", "verify current FedRAMP authorization level + US residency")
    if "PII" in dt and w["user_region"] == "EU":
        return ("Claude API / Claude Platform on AWS with inference_geo='eu'", "EU residency via inference_geo (1P/P-AWS); verify DPA")
    if "PHI" in dt:
        return ("Any BAA-covered deployment", "verify signed BAA before sending PHI")
    return ("Claude API (standard)", "standard controls")


def inference_geo_for(w: dict[str, Any]) -> str:
    if "CUI" in dt(w) or w["government_customer"]:
        return "us"
    if w["user_region"] == "EU":
        return "eu"
    return "global"


def dt(w: dict[str, Any]) -> set[str]:
    return set(w["data_types"])


# ===========================================================================
# (b) De-identification + reversible tokenization vault
# ===========================================================================
NAME_DICTIONARY = ["Jane Doe", "John Smith", "Maria Garcia"]  # fictional names
DEID_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("MRN", re.compile(r"\bMRN-\d{3,6}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]


class TokenVault:
    """Reversible pseudonymization. The map lives INSIDE the trust boundary and never
    leaves it; only de-identified text is sent to the model."""

    def __init__(self) -> None:
        self.to_token: dict[str, str] = {}
        self.to_value: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def tokenize(self, kind: str, value: str) -> str:
        if value in self.to_token:
            return self.to_token[value]
        self._counts[kind] = self._counts.get(kind, 0) + 1
        token = f"[{kind}_{self._counts[kind]}]"
        self.to_token[value] = token
        self.to_value[token] = value
        return token

    def reidentify(self, text: str) -> str:
        out = text
        for token, value in self.to_value.items():
            out = out.replace(token, value)
        return out


def de_identify(text: str, vault: TokenVault) -> str:
    out = text
    # Dictionary (names) first, then regex detectors.
    for name in NAME_DICTIONARY:
        if name in out:
            out = out.replace(name, vault.tokenize("NAME", name))
    for kind, pattern in DEID_PATTERNS:
        for m in sorted(set(pattern.findall(out)), key=len, reverse=True):
            out = out.replace(m, vault.tokenize(kind, m))
    return out


# ===========================================================================
# (c) GDPR data-subject rights
# ===========================================================================
@dataclass
class DataStores:
    """Stand-ins for the places a subject's data accumulates."""

    conversation_logs: list[dict[str, Any]] = field(default_factory=list)
    vector_index: list[dict[str, Any]] = field(default_factory=list)
    prompt_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def erase_subject(self, subject_id: str) -> dict[str, int]:
        before = (len(self.conversation_logs), len(self.vector_index), len(self.prompt_cache))
        self.conversation_logs = [r for r in self.conversation_logs if r["subject"] != subject_id]
        self.vector_index = [r for r in self.vector_index if r["subject"] != subject_id]
        self.prompt_cache = {k: v for k, v in self.prompt_cache.items() if v["subject"] != subject_id}
        after = (len(self.conversation_logs), len(self.vector_index), len(self.prompt_cache))
        return {"logs": before[0] - after[0], "vectors": before[1] - after[1], "cache": before[2] - after[2]}

    def export_subject(self, subject_id: str) -> dict[str, Any]:
        return {
            "subject": subject_id,
            "conversation_logs": [r for r in self.conversation_logs if r["subject"] == subject_id],
            "vector_index": [r for r in self.vector_index if r["subject"] == subject_id],
            "prompt_cache": [v for v in self.prompt_cache.values() if v["subject"] == subject_id],
        }


# ===========================================================================
# (d) Immutable audit log with hash chaining
# ===========================================================================
@dataclass
class AuditChain:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, action: str, detail: dict[str, Any]) -> None:
        prev_hash = self.entries[-1]["hash"] if self.entries else "GENESIS"
        payload = {"seq": len(self.entries) + 1, "action": action, "detail": detail, "prev_hash": prev_hash}
        payload["hash"] = self._hash(payload)
        self.entries.append(payload)

    @staticmethod
    def _hash(payload: dict[str, Any]) -> str:
        body = json.dumps({k: payload[k] for k in ("seq", "action", "detail", "prev_hash")}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    def verify(self) -> tuple[bool, int]:
        """Returns (ok, first_broken_seq). A single altered entry breaks the chain from
        that link onward - that is the tamper-evidence property."""
        prev_hash = "GENESIS"
        for e in self.entries:
            recomputed = self._hash({"seq": e["seq"], "action": e["action"], "detail": e["detail"], "prev_hash": prev_hash})
            if recomputed != e["hash"] or e["prev_hash"] != prev_hash:
                return False, e["seq"]
            prev_hash = e["hash"]
        return True, -1


# ===========================================================================
# Offline responder
# ===========================================================================
def build_responder() -> Callable[[MockRequest], Any]:
    def responder(req: MockRequest) -> Any:
        q = req.last_user_text
        # Summarize a de-identified clinical note using ONLY the tokens it contains.
        present = [t for t in ("[NAME_1]", "[MRN_1]", "[EMAIL_1]", "[PHONE_1]", "[SSN_1]") if t in q]
        name = "[NAME_1]" if "[NAME_1]" in present else "the patient"
        mrn = " (MRN [MRN_1])" if "[MRN_1]" in present else ""
        contact = " Follow-up via [EMAIL_1]." if "[EMAIL_1]" in present else ""
        return f"Summary: {name}{mrn} presented with mild symptoms; routine follow-up advised.{contact}"

    return responder


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    banner("D5.4 - Regulatory Compliance: GDPR, HIPAA, FedRAMP")
    print("  DISCLAIMER: educational demonstration only - NOT legal advice. Verify all")
    print("  authorizations (BAA, FedRAMP level, residency, ZDR) with the provider.")
    data = json.loads((DATA / "workloads.json").read_text(encoding="utf-8"))

    # --- (a) policy engine: checklist per workload ---
    section("(a) Compliance policy engine - checklist per workload")
    for w in data["workloads"]:
        platform, pnote = recommend_platform(w)
        print(f"\n  [{w['name']}]  data={w['data_types']} region={w['user_region']} gov={w['government_customer']} retention={w['retention_days']}d zdr={w['needs_zdr']}")
        print(f"    recommended platform: {platform}  ({pnote})")
        print(f"    inference_geo: '{inference_geo_for(w)}'")
        for c in evaluate_policy(w):
            print(f"      [{'x' if c.required else ' '}] {c.name}")
            print(f"            - {c.note}")

    # --- (b) de-identification round trip ---
    section("(b) PHI/PII de-identification -> Claude -> re-identification")
    client = get_client(build_responder())
    vault = TokenVault()
    note = ("Jane Doe (MRN MRN-4471), email jane.doe@example.com, phone (555) 555-0100, "
            "SSN 000-00-0000, presented with a mild cough.")
    deid = de_identify(note, vault)
    print(f"  original (inside trust boundary):\n    {note}")
    print(f"  de-identified (this is what LEAVES the boundary to the model):\n    {deid}")
    msg = client.messages.create(model=MODEL_OPUS, max_tokens=256, messages=[{"role": "user", "content": f"Summarize this clinical note:\n{deid}"}])
    model_text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    print(f"  model output (still tokenized - safe to log):\n    {model_text}")
    final = vault.reidentify(model_text)
    print(f"  re-identified (inside trust boundary only):\n    {final}")
    print(f"  vault entries: {list(vault.to_value.keys())}  (map never leaves the boundary)")

    # --- (c) GDPR data-subject rights ---
    section("(c) GDPR data-subject rights - erasure + export")
    stores = DataStores(
        conversation_logs=[
            {"subject": "subj_eu_1", "text": "chat 1"}, {"subject": "subj_eu_1", "text": "chat 2"},
            {"subject": "subj_us_9", "text": "chat 3"},
        ],
        vector_index=[
            {"subject": "subj_eu_1", "vec_id": "v1"}, {"subject": "subj_us_9", "vec_id": "v2"},
        ],
        prompt_cache={"ck1": {"subject": "subj_eu_1"}, "ck2": {"subject": "subj_us_9"}},
    )
    export = stores.export_subject("subj_eu_1")
    print(f"  data export for subj_eu_1: {json.dumps(export)}")
    deleted = stores.erase_subject("subj_eu_1")
    print(f"  right to erasure for subj_eu_1 -> deleted {deleted}")
    print(f"  remaining logs={len(stores.conversation_logs)} vectors={len(stores.vector_index)} cache={len(stores.prompt_cache)} (other subjects untouched)")

    # --- (d) hash-chained audit log ---
    section("(d) Immutable audit log (hash chaining + tamper detection)")
    audit = AuditChain()
    audit.append("deidentify", {"workload": "Clinical note summarizer", "tokens": len(vault.to_value)})
    audit.append("model_call", {"model": MODEL_OPUS, "pii_sent": False})
    audit.append("gdpr_export", {"subject": "subj_eu_1"})
    audit.append("gdpr_erasure", {"subject": "subj_eu_1", "deleted": deleted})
    ok, broken = audit.verify()
    print(f"  chain length {len(audit.entries)}; verify -> {'OK' if ok else f'BROKEN at seq {broken}'}")
    for e in audit.entries:
        print(f"    seq {e['seq']} {e['action']:<13} hash={e['hash'][:12]}... prev={e['prev_hash'][:12] if e['prev_hash'] != 'GENESIS' else 'GENESIS'}")
    # Tamper with an entry and re-verify.
    audit.entries[1]["detail"]["pii_sent"] = True
    ok2, broken2 = audit.verify()
    print(f"  after tampering seq 2's detail -> verify = {'OK' if ok2 else f'BROKEN at seq {broken2}'} (tamper detected)")

    section("Summary")
    print("  The policy engine turns a workload profile into an auditable control set.")
    print("  De-identification keeps raw PHI/PII inside the trust boundary; only tokens are sent.")
    print("  GDPR DSR workflows erase/export across every store a subject touches.")
    print("  The hash-chained audit log makes any after-the-fact edit detectable.")


if __name__ == "__main__":
    main()
