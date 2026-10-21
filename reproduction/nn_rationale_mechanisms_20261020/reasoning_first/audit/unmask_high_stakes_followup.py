"""Join original outcomes only after the source-assessment freeze."""
from pathlib import Path
import gzip
import hashlib
import json

D = Path(__file__).resolve().parents[1]/"context_high_stakes_followup"
freeze = json.loads((D/"CODING_FREEZE.json").read_text())
assert hashlib.sha256((D/"codes.jsonl").read_bytes()).hexdigest() == freeze["code_sha256"]
assert hashlib.sha256((D/"packets.jsonl").read_bytes()).hexdigest() == freeze["packet_sha256"]
codes = {r["audit_id"]: r for r in map(json.loads, (D/"codes.jsonl").read_text().splitlines())}
packets = {r["audit_id"]: r for r in map(json.loads, (D/"packets.jsonl").read_text().splitlines())}
with gzip.open(D/"private_mapping.jsonl.gz", "rt") as f:
    private = list(map(json.loads, f))
out = [dict(**packets[r["audit_id"]], assessment=codes[r["audit_id"]], outcome=r) for r in private]
dest = D/"unmasked_examples.json"
assert not dest.exists()
dest.write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n")
for row in out:
    print(json.dumps(dict(audit_id=row["audit_id"], task_rule=row["assessment"]["task_rule"],
                         **{k: row["outcome"][k] for k in [
                             "judge","backbone","true_label","control_probability","nn_probability",
                             "flip_type","reason_probability","score_valid","finish_reason","generated_tokens"
                         ]})))
