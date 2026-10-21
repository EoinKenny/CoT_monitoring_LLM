"""Join assessed native examples to their own pilot controls, not later runs."""
from pathlib import Path
from datetime import datetime,timezone
import gzip
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"native_contrast/context_aegis_followup"
P=R/"remote_retrieval/reasoning-full-pilot-20261011"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=json.loads((O/"CODING_FREEZE.json").read_text())
assert sha(O/"manual_codes.jsonl")==freeze["codes_sha256"]
assert sha(O/"masked_packet.jsonl")==freeze["packet_sha256"]
prepared=json.loads((P/"PREPARED.json").read_text())
assert sha(P/"OFF_CONTROLS.json")==prepared["off_controls_sha256"]
assert sha(P/"SELECTION.json")==prepared["selection_sha256"]
selected={(x["backbone"],x["sample_id"]):x for x in json.loads((P/"SELECTION.json").read_text())}
off={(x["judge"],x["backbone"],x["sample_id"],x["mode"],x["condition"]):x for x in json.loads((P/"OFF_CONTROLS.json").read_text())}
native={}
for x in map(json.loads,gzip.open(R/"remote_retrieval/native_saved_records.jsonl.gz","rt")):
    if x["run"]=="reasoning-full-pilot-20261011":
        key=x["judge"],x["backbone"],x["sample_id"],x["mode"],x["condition"]
        assert key not in native
        native[key]=x
packet={x["id"]:x for x in map(json.loads,(O/"masked_packet.jsonl").read_text().splitlines())}
codes={x["id"]:x for x in map(json.loads,(O/"manual_codes.jsonl").read_text().splitlines())}
private=list(map(json.loads,gzip.open(O/"private_mapping.jsonl.gz","rt")))
def label(value,mode):
    return int(value) if mode=="hard" else int(float(value)>=50)
out=[]
for p in private:
    for x in p["matched_records"]:
        assert x["run"]=="reasoning-full-pilot-20261011"
        key=x["judge"],x["backbone"],x["sample_id"],x["mode"]
        controls={c:off[key+(c,)] for c in ["baseline","nearest"]}
        gold=selected[(x["backbone"],x["sample_id"])]["true_label"]
        old_c,old_n=[label(controls[c]["value"],x["mode"]) for c in ["baseline","nearest"]]
        new=x["final"]
        assert new["valid"]
        scored=label(new["final"],x["mode"])
        baseline=native[key+("baseline",)]
        nbase=baseline["result"]
        assert nbase["valid"]
        result=dict(**packet[p["id"]],**{k:v for k,v in codes[p["id"]].items() if k!="id"},
                    judge=x["judge"],backbone=x["backbone"],sample_id=x["sample_id"],mode=x["mode"],
                    true_label=gold,old_control_value=controls["baseline"]["value"],
                    old_nn_value=controls["nearest"]["value"],old_control_label=old_c,old_nn_label=old_n,
                    original_flip=old_c!=old_n,new_native_value=int(new["final"]),new_native_label=scored,
                    new_native_correct=scored==gold,changed_from_old_nn=scored!=old_n,
                    native_baseline_value=int(nbase["final"]),native_baseline_label=label(nbase["final"],x["mode"]),
                    source_raw_sha256=x["source_raw_sha256"],source_raw_file=x["source_raw_file"],
                    source_context_sha256=x["context_check"]["parsed_context_sha256"])
        out.append(result)
dest=O/"unmasked_examples.json"
assert not dest.exists()
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
(O/"UNMASKING_RECEIPT.json").write_text(json.dumps(dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),n=len(out),
    original_controls_sha256=sha(P/"OFF_CONTROLS.json"),selection_sha256=sha(P/"SELECTION.json"),
    coding_freeze_sha256=sha(O/"CODING_FREEZE.json"),output_sha256=sha(dest),
    scope="Native pilot controls from exact historical frozen modes. Hard labels are binary; probability decisions use50. Native 2048-token intervention remains separate from later512-token reasoning-first study.",
),indent=2)+"\n")
for x in out:
    print(x["id"],x["judge"],x["mode"],"gold",x["true_label"],
          "old control/NN",x["old_control_value"],x["old_nn_value"],
          "native control/NN",x["native_baseline_value"],x["new_native_value"],
          "correct",x["new_native_correct"],"oldflip",x["original_flip"])
