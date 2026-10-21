"""Freeze balanced, outcome-masked context audit of existing saved outputs only."""
from pathlib import Path
from collections import defaultdict,Counter
import json,gzip,hashlib,random
R=Path(__file__).resolve().parent
A=R/'context_audit';A.mkdir(exist_ok=True)
(A/'packets').mkdir(exist_ok=True)
(A/'codes').mkdir(exist_ok=True)
seed=20261020
with gzip.open(R/'corpus/rows.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
with gzip.open(R/'corpus/contexts.jsonl.gz','rt') as f:contexts={r['context_id']:r for r in map(json.loads,f)}
rubrics=json.load(open(R/'TASK_RUBRICS.json'))
strata=defaultdict(lambda:defaultdict(list))
for r in rows:
 if not r['reasoning'].strip(): continue
 arm='contrast' if r['contrast']=='yes' else 'similarity_only' if r['similarity']=='yes' else None
 if arm: strata[(arm,r['dataset'],r['flip_type'])][r['context_id']].append(r)
rng=random.Random(seed);selected=[]
for (arm,ds,outcome),groups in sorted(strata.items()):
 n=6 if arm=='contrast' else 4
 keys=rng.sample(sorted(groups),n)
 for cid in keys:
  r=rng.choice(sorted(groups[cid],key=lambda x:x['trace_id']))
  aid='ctx_'+hashlib.sha256(('20261020:'+r['trace_id']).encode()).hexdigest()[:20]
  selected.append(dict(audit_id=aid,arm=arm,dataset=ds,original_outcome=outcome,
   trace_id=r['trace_id'],text_id=r['text_id'],context_id=cid,judge=r['judge'],backbone=r['backbone'],
   original_nn=r['nn_probability'],control=r['control_probability'],new_probability=r['reason_probability'],
   gold=r['true_label'],retains=r['retains_original_nn'],context_stratum_N=len(groups),
   selected_contexts=n,traces_in_context_stratum=len(groups[cid]),
   audit_selection_probability=(n/len(groups))/len(groups[cid])))
rowmap={r['trace_id']:r for r in rows}
assert len(selected)==200 and len({x['trace_id'] for x in selected})==200
packets=[]
for ds in sorted(rubrics):
 out=[]
 for x in selected:
  if x['dataset']!=ds:continue
  r=rowmap[x['trace_id']]
  out.append(dict(audit_id=x['audit_id'],dataset=ds,task=rubrics[ds]['task'],
   context=contexts[x['context_id']],rationale=r['reasoning']))
 rng.shuffle(out)
 p=A/'packets'/('context_'+ds+'.jsonl')
 content=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out)
 if p.exists():assert p.read_text()==content,'Never change frozen selection'
 else:p.write_text(content)
 packets.append(dict(dataset=ds,n=len(out),path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),characters=sum(len(json.dumps(x)) for x in out)))
(A/'selection_private.json').write_text(json.dumps(selected,indent=2)+'\n')
(A/'MANIFEST.json').write_text(json.dumps(dict(seed=seed,n=200,packets=packets,
 source_sha256=json.load(open(R/'CORPUS_MANIFEST.json'))['source_sha256'],
 allocation='Per dataset:6 original helpful contrast,6 original harmful contrast,4 similarity-only helpful,4 similarity-only harmful.',
 sampling='Within each stratum sample context IDs uniformly without replacement; choose one trace uniformly within that context and stratum. Contexts may recur across outcome strata; preserve cluster identity.',
 masked='No target gold, source judge, scores, original outcome, existing annotation or first-reader codes.',
 unique_contexts=len({x['context_id'] for x in selected}),
 intended_use='Independent second reading of expressed codes plus full-context validity audit; exploratory balanced sample. Do not quote unweighted sample frequencies as population prevalence.'),indent=2)+'\n')
print(json.dumps({'n':200,'unique_contexts':len({x['context_id'] for x in selected}),'packet_characters':{p['dataset']:p['characters'] for p in packets}},indent=2))
