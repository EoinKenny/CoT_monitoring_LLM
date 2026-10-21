"""Reveal original outcomes only after verifying frozen manual assessments."""
from pathlib import Path
import gzip,hashlib,json

R=Path(__file__).resolve().parents[1]/"context_convention_followup"
f=json.loads((R/"CODING_FREEZE.json").read_text())
assert hashlib.sha256((R/"codes.jsonl").read_bytes()).hexdigest()==f["code_sha256"]
assert hashlib.sha256((R/"packets.jsonl").read_bytes()).hexdigest()==f["packet_sha256"]
codes={r["audit_id"]:r for r in map(json.loads,(R/"codes.jsonl").read_text().splitlines())}
packets={r["audit_id"]:r for r in map(json.loads,(R/"packets.jsonl").read_text().splitlines())}
with gzip.open(R/"private_mapping.jsonl.gz","rt") as src:private=list(map(json.loads,src))
out=[dict(**packets[p["audit_id"]],assessment=codes[p["audit_id"]],outcome=p) for p in private]
(R/"unmasked_examples.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
for row in out:
    print(json.dumps(dict(audit_id=row["audit_id"],assessment=row["assessment"]["task_rule"],**{
        k:row["outcome"][k] for k in ["judge","backbone","true_label","control_probability",
                                     "nn_probability","flip_type","reason_probability","score_valid"]
    })))
