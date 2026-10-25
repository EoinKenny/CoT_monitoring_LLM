"""Freeze source checks before original outcomes or judge identities are inspected."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]
D = R / "context_ragtruth_followup"
packets = list(map(json.loads, (D/"packets.jsonl").read_text().splitlines()))
rubrics = json.loads((R.parent/"audit/saved_prompt_rubrics.json").read_text())
task = next(x["task"] for x in rubrics if x["task"].startswith(
    "Task: determine whether the assistant response contains a hallucination relative to the supplied source material."))
source = R.parent.parent/"ten_dataset_full/estimate.py"
assert task in source.read_text()
task_info = dict(text=task, source=str(source.resolve()),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
manual = [
    dict(context_support="partial", task_rule="problem",
         flags=["unsupported_addition_excused_as_omission", "target_evidence_not_fully_applied"],
         note="The target source reports Nelly and five other people on a bus, and specifically Nelly plus another passenger taken to jail. The summary claims five other people were arrested. The rationale notices the unsupported extension but calls it a minor omission. This is an added assertion about arrests, not omitted information, and no source establishes its truth. The supplied reference is a separate, faithfully summarised prosecution case. It is not explicitly used in this rationale, so this check does not establish that its low label caused the error.",
         evidence=[("target","Nelly and five other people"),
                   ("target","Jail along with another passenger"),
                   ("target","Five other people were also arrested."),
                   ("rationale","minor omission, not a contradiction"),
                   ("task","false or unsupported factual assertion")]),
    dict(context_support="partial", task_rule="problem",
         flags=["repairs_reference_target_confusion", "valid_local_contrast_misses_other_error"],
         note="The rationale correctly separates the target's free WiFi from the reference's explicit no-WiFi field, retracting an initial imported error. That local contrast is supported. It nevertheless calls the target hours faithful: the source closes Friday/Saturday at20:00 and Sunday at19:30, while the summary says weekdays19:30/weekends20:00. This misassigns Friday and Sunday. Fixing source attribution for one attribute does not establish global faithfulness, and the rationale stops mid-check without a final answer. The frozen task also says a correct fact omitted from the source is not itself a hallucination; this assessment relies on explicit time contradictions, not a blanket ban on paraphrase or outside knowledge.",
         evidence=[("target","'Friday': '11:30-20:0'"),
                   ("target","'Sunday': '11:30-19:30'"),
                   ("target","from 11:30 am to 7:30 pm on weekdays and from 11:30 am to 8:00 pm on weekends"),
                   ("neighbor","'WiFi': 'no'"),
                   ("rationale","This is NOT a hallucination."),
                   ("rationale","Assistant: \"11:30 am to 8:00 pm on weekends\".\nThis is faithful."),
                   ("task","contradicts the source")]),
]
assert len(manual) == len(packets) == 2
codes = []
for p, c in zip(packets, manual):
    pools = dict(target="\n".join(m["content"] for m in p["context"]["target_messages"]),
                 neighbor="\n".join(m["content"] for m in p["context"]["neighbor_messages"]),
                 rationale=p["rationale"], task=task)
    for field, quote in c["evidence"]:
        assert quote in pools[field], (p["audit_id"], field, quote)
    codes.append(dict(audit_id=p["audit_id"], reader="/root", **c))
assert not (D/"codes.jsonl").exists()
(D/"codes.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in codes))
(D/"task_rubric.json").write_text(json.dumps(task_info, indent=2)+"\n")
(D/"CODING_FREEZE.json").write_text(json.dumps(dict(
    created_utc=datetime.now(timezone.utc).isoformat(), n=2,
    code_sha256=hashlib.sha256((D/"codes.jsonl").read_bytes()).hexdigest(),
    packet_sha256=hashlib.sha256((D/"packets.jsonl").read_bytes()).hexdigest(),
    task_sha256=hashlib.sha256(task.encode()).hexdigest(),
    original_outcomes_inspected=False,
    scope="Two purposive single-reader full-source checks; original outcomes and model identities withheld until freeze. No inference."
), indent=2)+"\n")
print("Frozen two RAGTruth source assessments before outcome lookup.")
