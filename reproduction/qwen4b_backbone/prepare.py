import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,time,hashlib,shutil
from transformers import AutoTokenizer
from tokenize_rows import tokenize_record
R=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');B=R.parent/'ten-dataset-full-20261002'
assert not (R/'NEW_PREPARED').exists()
backbone=json.loads((R/'backbone.json').read_text());tok=AutoTokenizer.from_pretrained(backbone['path'],local_files_only=True)
summary={};start=time.time();prov=R/'provenance';prov.mkdir(exist_ok=True)
for ds in ['wildguard','high_stakes','toxicchat','beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']:
 p=B/f'{ds}_metadata.json';source=json.loads(p.read_text());out=[]
 for i,r in enumerate(source):
  keep={k:r[k] for k in ['id','messages','label','split','group','input_hash','role']}
  keep.update(source_index=i,source_metadata_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if i==0 else out[0]['source_metadata_sha256'])
  keep.update(tokenize_record(tok,keep));out.append(keep)
  if i%2000==0:print(json.dumps(dict(event='tokenizing',dataset=ds,completed=i,total=len(source),seconds=time.time()-start)),flush=True)
 assert [(r['id'],r['split'],r['group'],r['label'],r['input_hash']) for r in source]==[(r['id'],r['split'],r['group'],r['label'],r['input_hash']) for r in out]
 (R/f'{ds}_selected.json').write_text(json.dumps(out))
 summary[ds]=dict(n=len(out),max_tokens=max(r['n_tokens'] for r in out),over_4096=sum(r['n_tokens']>4096 for r in out),tokens=sum(r['n_tokens'] for r in out),source_metadata_sha256=out[0]['source_metadata_sha256'],split_sha256=hashlib.sha256(json.dumps([(r['id'],r['split'],r['group'],r['label']) for r in out]).encode()).hexdigest())
 shutil.copyfile(p,prov/p.name)
 print(json.dumps(dict(event='prepared',dataset=ds,**summary[ds])),flush=True)
for n in ['downloads.json','VERIFIED.json','config.json','README.md','judge_rubrics.json']:
 if (B/n).exists():shutil.copyfile(B/n,prov/('llama_'+n))
shutil.copyfile(B/'judge_rubrics.json',R/'judge_rubrics.json')
(R/'new_inventory.json').write_text(json.dumps(summary,indent=2));(R/'NEW_PREPARED').write_text(json.dumps(dict(seconds=time.time()-start,datasets=summary),indent=2))
