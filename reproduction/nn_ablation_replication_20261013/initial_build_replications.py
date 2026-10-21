from pathlib import Path
import json,hashlib,statistics,collections
root=Path(__file__).resolve().parent
src=root.parent/'nn_ablation_20261013'
parent=root.parent/'probability_500_20261012/results/pairs.json'
rows=json.loads(parent.read_text()); ranks=[]
for b,j in sorted({(r['backbone'],r['judge']) for r in rows}):
 rr=[r for r in rows if (r['backbone'],r['judge'])==(b,j)]
 baseline=statistics.mean(r['variants']['baseline_joint']['f1'] for r in rr);nn=statistics.mean(r['variants']['nearest_joint']['f1'] for r in rr)
 ranks.append(dict(backbone=b,judge=j,baseline_f1=baseline,nn_f1=nn,gain=nn-baseline))
ranks.sort(key=lambda x:-x['gain'])
selection=dict(rule='Two distinct additional judges excluding already-ablated qwen122; choose each judge best backbone and rank by equal-dataset macro test F1 gain of validation-joint-tuned nearest versus baseline. Exploratory mechanism follow-up selected from observed parent results.',source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),ranking=ranks,selected=[r for r in ranks if (r['backbone'],r['judge']) in [('qwen','qwen'),('gemma','gemma')]])
(root/'SELECTION.json').write_text(json.dumps(selection,indent=2))
for b,j,title in [('qwen','qwen','Qwen3-4B-Instruct-2507 backbone; Qwen3.6-27B judge'),('gemma','gemma','Gemma backbone; Gemma4-31B judge')]:
 d=root/(b+'_'+j);(d/'code').mkdir(parents=True,exist_ok=True)
 remote=f'/home/ubuntu/experiments/nn-ablation-{b}-{j}-20261013'
 for p in (src/'code').glob('*.py'):
  s=p.read_text()
  if p.name!='numerics_source.py':
   s=s.replace('/home/ubuntu/experiments/nn-ablation-20261013',remote).replace('llama70',b).replace('qwen122',j)
   s=s.replace('Llama 3.3 70B backbone; Qwen 3.5 122B judge',title)
   s=s.replace('The pair was chosen for the largest observed parent NN improvement','The pair was chosen for its judge\'s largest observed parent NN improvement'.replace("judge's",'judge-specific'))
  (d/'code'/p.name).write_text(s)
 protocol=json.loads((src/'PROTOCOL.json').read_text());protocol.update(study='NN component ablation replication',backbone=b,judge=j,selection_provenance=selection['rule'],inference='Reuse this judge pinned parent weights, tokenizer and engine settings, batch16, greedy seed20261001, max_tokens8, thinking disabled, scalar regex. Retain context capacity unless random text requires larger capacity within original context limit.')
 protocol['analysis']=protocol['analysis'].replace('Model pair chosen for largest observed parent NN gain','Model pair chosen for this additional judge largest observed parent NN gain')
 (d/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');(d/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
 print(remote)
