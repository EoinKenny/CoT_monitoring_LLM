"""Run on server once all annotations are complete; read-only audit plus export."""
from pathlib import Path
import json,hashlib,tarfile,time
B=Path('/home/ubuntu/experiments')
ROOTS={'qwen122':B/'nn-mechanism-two-labels-20261018','gemma':B/'nn-mechanism-two-labels-gemma-20261018','qwen':B/'nn-mechanism-two-labels-qwen-20261018'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
verified={};units=None
for name,r in ROOTS.items():
 c=json.loads((r/'COMPLETED.json').read_text());assert c['n_cases']==c['n_units']==25000
 for f,k in [('annotations.jsonl','annotations_sha256'),('labels_by_case.csv','case_labels_sha256'),('OUTPUT_SHA256.json','output_manifest_sha256'),('PREPARED.json','prepared_sha256')]:assert sha(r/f)==c[k],(name,f)
 manifest=json.loads((r/'OUTPUT_SHA256.json').read_text())
 for f,h in manifest.items():assert sha(r/f)==h,(name,f)
 ids=[json.loads(l)['unit_id'] for l in (r/'annotations.jsonl').open()]
 expected=[json.loads(l)['unit_id'] for l in (r/'units.jsonl').open()]
 assert ids==expected and len(set(ids))==25000
 if units is None:units=ids
 else:assert ids==units
 verified[name]=dict(completed=c,raw_files_checked=len(manifest),root=str(r))
out=B/'nn-mechanism-additional-judges-20261018'/'analysis_export_20261019.tar.gz'
assert not out.exists()
with tarfile.open(out,'w:gz') as t:
 for name,r in ROOTS.items():
  for f in ['annotations.jsonl','labels_by_case.csv','COMPLETED.json','PREPARED.json','PROTOCOL.json','CONFIG.json','OUTPUT_SHA256.json','PARSER_REPAIR.json']:
   if (r/f).exists():t.add(r/f,arcname=name+'/'+f)
 p=ROOTS['qwen122']/'mapping_PRIVATE.jsonl';t.add(p,arcname='mapping_PRIVATE.jsonl')
receipt=dict(time=time.time(),judges=verified,archive_sha256=sha(out),identical_ordered_case_membership=True)
p=out.with_suffix('.verified.json');p.write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(archive=str(out),receipt=str(p),sha256=sha(out),counts={k:v['completed']['status_counts'] for k,v in verified.items()}),indent=2))
