from pathlib import Path
import json,gzip,hashlib,time
E=Path('/home/ubuntu/experiments');R=E/'nn-mechanisms-20261015';P=E/'probability-500-20261012'
pairs=[('llama70','qwen122','nn-ablation-20261013'),('qwen','qwen','nn-ablation-qwen-qwen-20261013'),('gemma','gemma','nn-ablation-gemma-gemma-20261013'),('llama','qwen','nn-ablation-llama-qwen-20261013'),('gemma','qwen122','nn-ablation-gemma-qwen122-20261013')]
cs=['baseline','nearest','text_label','text_prediction','text_only','label_only','prediction_only','label_prediction','random_full'];sources={};out=[]
def read(p):
 raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
for b,m,name in pairs:
 root=E/name;sel=read(root/'score'/b/'selected_flags.json');assert sel==read(P/'score'/b/'selected_flags.json')
 rnd={r['sample_id']:r for r in read(root/'RANDOM_SELECTION.json')};answers={}
 for c in cs:
  run=(P if c in ['baseline','nearest'] else root)/'score'/b/'judges'/f'{m}_{c}'/'run';p=run/'judgments.jsonl';raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();done=read(run/'COMPLETED');assert h==done['judgments_sha256'];sources[str(p)]=h
  rows=[json.loads(x) for x in raw.splitlines()];answers[c]={r['sample_id']:r['positive_probability'] for r in rows}
  assert len(rows)==len(answers[c])==len(sel)==10000
 for r in sel:
  z=rnd[r['sample_id']];out.append(dict(backbone=b,model=m,**r,random_label=z['training_label'],random_prediction=z['training_prediction'],random_id=z['training_id'],scores={c:answers[c][r['sample_id']] for c in cs}))
 print('exported',b,m,flush=True)
with gzip.open(R/'ablation_cases.jsonl.gz','wt') as f:
 for r in out:f.write(json.dumps(r)+'\n')
(R/'ABLATION_EXPORT.json').write_text(json.dumps(dict(time=time.time(),n=len(out),sources=sources,sha256=hashlib.sha256((R/'ablation_cases.jsonl.gz').read_bytes()).hexdigest()),indent=2))
