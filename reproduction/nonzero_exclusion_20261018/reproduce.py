"""Descriptive exclusion sensitivity; use original thresholds and metrics unchanged."""
import json,collections,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;C=['baseline','saliency','nearest'];rows=[]
inputs=[P/'probability_500_20261012/results/pairs.json',P/'api_results_20261015/results/matched_results.json']
for r in json.load(open(inputs[0])):
 for c in C:rows.append(dict(scope='local',model=r['judge'],backbone=r['backbone'],dataset=r['dataset'],condition=c,**r['variants'][c+'_joint']))
for r in json.load(open(inputs[1])):
 if r['scope']=='api_condition_matched' and r['policy']=='joint':rows.append(dict(r,scope='api'))
out=[];members=[]
for scope in ['all','local','api']:
 rr=[r for r in rows if scope=='all' or r['scope']==scope];g=collections.defaultdict(dict)
 for r in rr:g[r['model'],r['backbone'],r['dataset']][r['condition']]=r
 keys={k for k,v in g.items() if all(v[c]['probability_threshold']>0 for c in C)}
 pair={k for k,v in g.items() if all(v[c]['probability_threshold']>0 for c in ['baseline','nearest'])}
 for k in sorted(keys):members.append(dict(scope=scope,model=k[0],backbone=k[1],dataset=k[2]))
 for mode in ['original','drop_each','matched_three','matched_nn_control']:
  for c in C:
   if mode=='matched_nn_control' and c=='saliency':continue
   selected=[r for r in rr if r['condition']==c and (mode=='original' or mode=='drop_each' and r['probability_threshold']>0 or mode=='matched_three' and (r['model'],r['backbone'],r['dataset']) in keys or mode=='matched_nn_control' and (r['model'],r['backbone'],r['dataset']) in pair)]
   out.append(dict(scope=scope,mode=mode,condition=c,n=len(selected),**{m:sum(r[m] for r in selected)/len(selected) for m in ['f1','precision','recall']}))
json.dump(out,open(R/'primary_summary.json','w'),indent=2);json.dump(members,open(R/'retained_combinations.json','w'),indent=2)
json.dump({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'method':'Equal-weight mean of existing cell metrics; primary comparison retains exactly the same cells across3conditions, all judge cutoffs>0. Original condition-specific probe cutoffs unchanged. Local full-test estimates and API matched-valid estimates remain separate scopes, with all as a descriptive combined summary. No refitting, new calls, test selection or new significance tests.'},open(R/'VERIFIED.json','w'),indent=2)
