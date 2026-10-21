"""Attach existing 2-NN scores to every selected qualitative candidate."""
from pathlib import Path
import gzip,json
R=Path(__file__).resolve().parent;e=json.load(open(R/'results/audit_examples.json'));wanted={}
for x in e:
 for j in x['judge_outcomes']:
  if j['judge'] in ['qwen','qwen122','gemma','glm','mistral','olmo']:wanted.setdefault((j['judge'],x['source']['backbone'],x['source']['sample_id']),[]).append((x,j))
rows=[]
for l in gzip.open(R.parent/'two_nn_20261014/export_verified_20261016/nearest2_scores.jsonl.gz','rt'):
 r=json.loads(l);k=(r['model'],r['backbone'],r['sample_id'])
 for x,j in wanted.get(k,[]):rows.append(dict(audit_id=x['audit_id'],dataset=x['source']['dataset'],judge=j['judge'],gold=x['source']['true_label'],control=j['control'],nearest=j['nearest'],nearest2=r['positive_probability']))
assert len(rows)==sum(map(len,wanted.values()));json.dump(rows,open(R/'results/audit_second_nn_scores.json','w'),indent=2)
for r in rows:
 if r['audit_id'] in [15,24,25,7] and r['judge'] in {15:['qwen'],24:['qwen122'],25:['qwen122'],7:['olmo']}[r['audit_id']]:print(r)
