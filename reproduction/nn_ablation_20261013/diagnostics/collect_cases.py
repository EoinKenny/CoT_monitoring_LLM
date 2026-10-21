import json,hashlib
from pathlib import Path
A=Path('/home/ubuntu/experiments/nn-ablation-20261013');P=Path('/home/ubuntu/experiments/probability-500-20261012');S=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009/llama70')
read=lambda p:json.loads(p.read_text())
rows=read(A/'score/llama70/selected_flags.json');random={x['sample_id']:x for x in read(A/'RANDOM_SELECTION.json')};answers={};hashes={}
for c in ['baseline','nearest','text_prediction','random_full','text_label','label_prediction']:
 root=P if c in ['baseline','nearest'] else A
 run=root/'score/llama70/judges'/('qwen122_'+c)/'run';p=run/'judgments.jsonl';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==read(run/'COMPLETED')['judgments_sha256'];hashes[str(p)]=h
 answers[c]={x['sample_id']:x['positive_probability'] for x in map(json.loads,p.open())}
meta=read(S/'toxicchat_metadata.json');out=[]
for ds in sorted({r['dataset'] for r in rows}):
 refs={x['sample_id']:x for x in read(S/'judge_inputs'/f'{ds}_explanations.json')['records']}
 for r in rows:
  if r['dataset']!=ds:continue
  sid=r['sample_id'];n=refs[sid]['nearest'];ra=random[sid]
  x=dict(r,scores={c:a[sid] for c,a in answers.items()},nearest_label=n['label'],nearest_prediction=n['prediction'],nearest_id=n['immutable_id'],nearest_cosine=n['cosine'],random_label=ra['training_label'],random_prediction=ra['training_prediction'],random_id=ra['training_id'])
  if ds=='toxicchat':x.update(messages=refs[sid]['messages'],nearest_messages=n['messages'],random_messages=meta[ra['training_index']]['messages'])
  out.append(x)
print(json.dumps(dict(sources=hashes,rows=out)))
