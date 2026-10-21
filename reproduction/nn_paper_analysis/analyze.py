from pathlib import Path
import json,collections,numpy as np
B=Path('/home/ubuntu/experiments/ten-dataset-full-20261002');R=B.parent/'ten-dataset-sampled-20261002';OUT=R/'nn_paper_analysis.json'
rows=json.loads((R/'private.json').read_text());byid={r['sample_id']:r for r in rows};ex={}
for line in (B/'judge_inputs/explanations.jsonl').open():
 r=json.loads(line)
 if r['sample_id'] in byid:ex[r['sample_id']]=r
assert set(ex)==set(byid)
for r in rows:
 n=ex[r['sample_id']]['nearest'];r.update(nn_label=n['label'],nn_prediction=n['prediction'],cosine=n['cosine'],nn_wrong=n['label']!=n['prediction'])
models=['qwen','gemma','llama'];pred={}
for m in models:
 pred[m]={}
 for c in ['baseline','saliency','nearest']:
  js=[json.loads(l) for l in (R/'judges'/f'{m}_{c}'/'run/judgments.jsonl').read_text().splitlines()];pred[m][c]={r['sample_id']:r['prediction'] for r in js}
def summarize(rr,m):
 d=collections.Counter();d['n']=len(rr)
 for r in rr:
  y=r['true_label'];b,s,n=[pred[m][c][r['sample_id']] for c in ['baseline','saliency','nearest']];nl=r['nn_label']
  for c,v in [('baseline',b),('saliency',s),('nearest',n)]:
   d[c+'_correct']+=int(v==y);d[c+'_positive']+=int(v==1);d[c+'_matches_nn_label']+=int(v==nl)
  d['nn_label_correct']+=int(nl==y);d['rescue_baseline']+=int(b!=y and n==y);d['harm_baseline']+=int(b==y and n!=y)
  d['rescue_saliency']+=int(s!=y and n==y);d['harm_saliency']+=int(s==y and n!=y)
  d['rescue_both']+=int(b!=y and s!=y and n==y);d['harm_both']+=int(b==y and s==y and n!=y)
  d['baseline_disagrees_nn_label']+=int(b!=nl);d['flips_toward_nn_label']+=int(b!=nl and n==nl);d['flips_away_nn_label']+=int(b==nl and n!=nl)
  d['rescues_toward_nn_label']+=int(b!=y and n==y and n==nl);d['harms_toward_nn_label']+=int(b==y and n!=y and n==nl)
 return dict(d)
x={'n':len(rows),'models':{},'neighbour_error_signal':{},'datasets':{}}
for wrong in [False,True]:
 rr=[r for r in rows if r['nn_wrong']==wrong];x['neighbour_error_signal'][str(wrong)]=dict(n=len(rr),target_probe_errors=sum(r['true_label']==0 for r in rr),nn_label_correct=sum(r['nn_label']==r['true_label'] for r in rr))
for m in models:
 x['models'][m]={'overall':summarize(rows,m),'by_target_label':{str(y):summarize([r for r in rows if r['true_label']==y],m) for y in [0,1]},'by_neighbour_error':{str(w):summarize([r for r in rows if r['nn_wrong']==w],m) for w in [False,True]},'by_neighbour_label_correct':{str(w):summarize([r for r in rows if (r['nn_label']==r['true_label'])==w],m) for w in [False,True]}}
 for w in [False,True]:
  for y in [0,1]:x['models'][m][f'nn_wrong_{w}_target_{y}']=summarize([r for r in rows if r['nn_wrong']==w and r['true_label']==y],m)
base=json.loads((R/'weighted_results.json').read_text())['results']
for ds in sorted(base):
 rr=[r for r in rows if r['dataset']==ds];b=base[ds]['test'];accept={y:np.mean([r['nn_label'] for r in rr if r['true_label']==y]) for y in [0,1]}
 tp=b['tp']*accept[1];fp=b['fp']*accept[0];f1=2*tp/(tp+fp+b['tp']+b['fn'])
 x['datasets'][ds]=dict(n=len(rr),copy_nn_label_f1=float(f1),copy_nn_accuracy=float(np.mean([r['nn_label']==r['true_label'] for r in rr])),signal={str(w):dict(n=sum(r['nn_wrong']==w for r in rr),target_probe_errors=sum(r['nn_wrong']==w and r['true_label']==0 for r in rr)) for w in [False,True]},models={m:summarize(rr,m) for m in models})
# Within-dataset similarity quartiles avoid comparing raw cosine across tasks.
for ds in sorted(base):
 rr=sorted([r for r in rows if r['dataset']==ds],key=lambda r:r['cosine'])
 for q,ix in enumerate(np.array_split(np.arange(len(rr)),4)):
  for i in ix:rr[i]['cosine_quartile']=q
x['cosine_quartiles']={str(q):{m:summarize([r for r in rows if r['cosine_quartile']==q],m) for m in models} for q in range(4)}
OUT.write_text(json.dumps(x,indent=2));print(json.dumps({'n':x['n'],'neighbour_error_signal':x['neighbour_error_signal'],'models':x['models']},indent=2))
