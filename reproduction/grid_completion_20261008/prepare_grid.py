"""Create additive views of authoritative artifacts; never overwrite historical runs."""
from common import *
import os
SRC=Path('/home/ubuntu/qwen122b-fp8-full-20261007')
assert read(SRC/'COMPLETED.json')['rows']==456008
def link(target,dest):
 if dest.is_symlink():assert dest.resolve()==target.resolve()
 else:assert not dest.exists();dest.symlink_to(target)
audit={}
for b in ['llama','qwen','gemma','llama70']:
 original=E/'reproducible-sampled-20261004'/b if b!='llama70' else E/'llama70b-full-20261005/llama70'
 out=ROOT/b;out.mkdir(exist_ok=True);(out/'judges').mkdir(exist_ok=True)
 for p in original.iterdir():
  if p.name in ['judges','prepare_judges.py']:continue
  link(p,out/p.name)
 for p in (original/'judges').iterdir():link(p,out/'judges'/p.name)
 selected=read(original/'selected_flags.json');ids={r['sample_id'] for r in selected};verified=[]
 for m in JUDGES[:-1]:
  for c in CONDITIONS:
   d=original/'judges'/f'{m}_{c}'/'run';mark=read(d/'COMPLETED')
   assert sha(d/'judgments.jsonl')==mark['judgments_sha256']
   rr=[json.loads(l) for l in (d/'judgments.jsonl').open()]
   assert len(rr)==len(selected) and {r['sample_id'] for r in rr}==ids
   verified.append(dict(judge=m,condition=c,n=len(rr),sha256=mark['judgments_sha256']))
 audit[b]=dict(source=str(original),selected=len(selected),verified=verified)
B.mkdir(exist_ok=True)
for ds in DATASETS:
 for source,dest in [('metadata.json',f'{ds}_metadata.json'),('features.npy',f'{ds}_features.npy'),('EXTRACTED.json',f'{ds}_EXTRACTED.json'),('BATCHES.json',f'{ds}_BATCHES.json')]:link(SRC/ds/source,B/dest)
 rows=read(SRC/ds/'metadata.json');splits=dict(source=str(SRC/ds/'metadata.json'),source_sha256=sha(SRC/ds/'metadata.json'),split_records=[{k:r[k] for k in ['immutable_id','split','label','group','input_hash']} for r in rows])
 if (B/f'{ds}_SPLITS.json').exists():assert read(B/f'{ds}_SPLITS.json')==splits
 else:write(B/f'{ds}_SPLITS.json',splits)
write(B/'backbone.json',dict(repo='Qwen/Qwen3.5-122B-A10B-FP8',revision='a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9',hidden_size=3072,block_index=31,quantization='official FP8',source_provenance=read(SRC/'PROVENANCE.json'),authoritative_features=str(SRC)))
link(E/'llama70b-full-20261005/llama70/judge_rubrics.json',B/'judge_rubrics.json')
write(ROOT/'IMPORT_AUDIT.json',audit)
print(json.dumps(dict(historical_judge_runs=60,historical_cells=600,new_backbone='qwen122',new_judge='qwen122',target_cells=900)),flush=True)
