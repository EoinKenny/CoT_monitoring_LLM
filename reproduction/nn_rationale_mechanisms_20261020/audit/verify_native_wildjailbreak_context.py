from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import gzip
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"native_contrast/context_wildjailbreak_followup"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((O/"unmasked_examples.json").read_text())
freeze=json.loads((O/"CODING_FREEZE.json").read_text())
assert sha(O/"manual_codes.jsonl")==freeze["codes_sha256"]
assert sha(O/"masked_packet.jsonl")==freeze["packet_sha256"]
needed={(x["backbone"],x["sample_id"]) for x in rows}
features={}
for x in map(json.loads,gzip.open(R.parents[1]/"reproduction/nn_mechanisms_20261015/features.jsonl.gz","rt")):
    key=x["backbone"],x["sample_id"]
    if key in needed:
        features[key]=x
for x in rows:
    assert sha(Path(x["source_raw_file"]))==x["source_raw_sha256"]
    ctx={k:x[k] for k in ["task","target_messages","neighbor_messages","nearest_label","nearest_prediction"]}
    assert hashlib.sha256(json.dumps(ctx,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==x["source_context_sha256"]
    f=features[x["backbone"],x["sample_id"]]
    assert f["true_label"]==x["true_label"]
    for k in ["target_messages","neighbor_messages","nearest_label","nearest_prediction"]:
        assert f[k]==x[k]
class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs=[]
        self.ids=[]
        self.cards=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if "href" in d:self.hrefs.append(d["href"])
        if "id" in d:self.ids.append(d["id"])
        if tag=="details":self.cards+=1
h=Links(); h.feed((O/"report.html").read_text())
for link in h.hrefs:
    if link.startswith("#"):assert link[1:] in h.ids
    elif "://" not in link:assert (O/link.split("#")[0]).exists(),link
assert h.cards==len(rows)==8
assert len(set(x["source_context_sha256"] for x in rows))==3
assert sum(x["original_flip"] for x in rows)==1
assert all(x["true_label"]==0 for x in rows)
(O/"QA.json").write_text(json.dumps(dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),
    raw_checksums=len(rows),exact_context_hashes=len(rows),independent_feature_golds=len(rows),
    n_distinct_contexts=3,n_cards=h.cards,links=len(h.hrefs),
    coding_freeze_verified=True,report_sha256=sha(O/"report.html"),
    browser_rendered=False,
),indent=2)+"\n")
print("Verified eight traces, three contexts, source hashes, golds, frozen codes and local HTML links.")
