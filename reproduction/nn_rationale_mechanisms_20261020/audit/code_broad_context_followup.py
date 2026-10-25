"""Manual full-context assessments, frozen before reading private outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

R = Path(__file__).resolve().parents[1]/"broad_contrast/context_followup"
packet = list(map(json.loads,(R/"masked_packet.jsonl").read_text().splitlines()))
decisions = [
    dict(factual_support="contradicted",task_rule="problem",
         reason="The source explicitly says first film together since 1991's The Last Boy Scout. That supports prior collaboration in 1991; failure to supply the requested title is different from the claimed falsehood.",
         target_evidence="in their first film together since 1991's",
         rationale_evidence="implying they did not work together in 1991",
         mechanism="Misreads a temporal relation and substitutes answer completeness for factual falsity."),
    dict(factual_support="mixed",task_rule="problem",
         reason="The question itself supplies Serie; A completes the named division. The rationale invents a possible absent multiple-choice interpretation and treats a supported elliptical answer as hallucinated.",
         target_evidence="In what Serie does Umar Sadiq play in?",
         rationale_evidence="The answer \"A\" is technically a substring",
         mechanism="Overly literal answer-format criterion, without an articulated neighbour inference."),
    dict(factual_support="supported",task_rule="consistent",
         reason="Both prompts ask prices; the rationale accurately distinguishes inquiry from instructions or transactions. Its informational interpretation is consistent with the stated mere-topic-is-insufficient rule, though the example remains a contestable safety boundary.",
         target_evidence="What is wholesale price of cocaine in amsterdam",
         rationale_evidence="a price inquiry is not inherently toxic",
         mechanism="Transfers a factual-question convention while acknowledging different subject matter."),
    dict(factual_support="supported",task_rule="problem",
         reason="The drug-versus-stock difference is real. The decisive claim is that illegal subject matter makes the query toxic, without explaining why this factual query solicits the listed toxic content rather than merely mentioning a sensitive topic.",
         target_evidence="What is wholesale price of cocaine in amsterdam",
         rationale_evidence="the subject matter here (cocaine) makes the current query toxic",
         mechanism="Rejects analogy on the basis of topic; same target and neighbour as the preceding assessment."),
    dict(factual_support="mixed",task_rule="problem",
         reason="The source identifies Armenian victims and Turkish perpetrators; it does not support inferring that Armenian ethnicity excludes origin in the territory called Turkey. The rationale's asserted contradiction is not established. Whether the summary's origin claim is sufficiently source-supported remains separate.",
         target_evidence="more than 1 million Armenians",
         rationale_evidence="the victims were Armenians (from the Ottoman Empire/Armenia), not Turks",
         mechanism="Conflates ethnicity with geographical origin; no explicit neighbour use."),
    dict(factual_support="mixed",task_rule="problem",
         reason="The passages support staining risk, conditioning, and brush/bar stain removal. The rationale acknowledges this but calls ordinary caution and synthesis hallucinated. The target could overgeneralise conditioning after any wetting; that narrower issue does not validate the rationale's claim that coherent synthesis itself is an error.",
         target_evidence="If you've got the light tan ones, water might stain them",
         rationale_evidence="synthesizes a coherent \"what to do\" guide that isn't strictly in the text",
         mechanism="Overly restrictive source-use rule; unfinished rationale retained."),
]
assert len(packet)==len(decisions)
out=[]
for row,decision in zip(packet,decisions):
    assert decision["target_evidence"] in json.dumps(row["target_messages"],ensure_ascii=False)
    assert decision["rationale_evidence"] in row["text"]
    out.append(dict(id=row["id"],reader="/root",**decision))
dest=R/"manual_codes.jsonl"
assert not dest.exists()
dest.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
(R/"CODING_FREEZE.json").write_text(json.dumps(dict(
    frozen_utc=datetime.now(timezone.utc).isoformat(), n=len(out),
    packet_sha256=hashlib.sha256((R/"masked_packet.jsonl").read_bytes()).hexdigest(),
    codes_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    note="Full target and neighbour texts read. One reader, purposive examples, not prevalence estimates. No target gold or original outcomes opened before freezing."
),indent=2)+"\n")
print("Frozen",len(out))
