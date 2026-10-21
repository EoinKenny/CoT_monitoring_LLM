"""Recompute the paper's fixed and matched-threshold results from frozen scores.

No model calls. Threshold selection uses validation only. Each anchor supplies
one threshold pair shared across all explanations in that judge/backbone/dataset.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import argparse,ast,collections,csv,gzip,hashlib,json,time
import numpy as np
REPO=Path(__file__).resolve().parents[1]
C=['baseline','saliency','nearest'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];API=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra'];TOX={'aegis','beavertails','toxicchat','toxigen'}
def read(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def savecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True,help='Restored reproduction directory');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--frontier',action='store_true');ap.add_argument('--check-existing',action='store_true');args=ap.parse_args()
 assert not args.out.exists(),'Use a new output directory to preserve earlier results';args.out.mkdir(parents=True)
 spec=read(REPO/'docs/FROZEN_INPUTS.json')
 for rel,h in spec.items():assert sha(args.data/rel)==h,rel
 source=REPO/'reproduction/two_nn_20261014/code/numerics_source.py';ns={'np':np};tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','curve','grid','choose']],type_ignores=[]),str(source),'exec'),ns)
 d=read(args.data/'api_results_20261015/frozen/design.json.gz');z=np.load(args.data/'api_results_20261015/results/scores.npz');cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr]
 groups=collections.defaultdict(list)
 for i,r in enumerate(cases):groups[r['backbone'],r['dataset'],r['split']].append(i)
 valid={m:(z[m]>=0).all(axis=1) for m in API};common=np.logical_and.reduce(list(valid.values()));rows=[];thresholds=[];frontier=[]
 scopes=[('local_full',LOCAL,{m:np.ones(len(cases),bool) for m in LOCAL}),('api_matched',API,valid),('common_valid',LOCAL+API,{m:common for m in LOCAL+API})]
 for scope,models,masks in scopes:
  for b in d['samples']:
   for ds,s in sorted(d['thresholds'][b].items()):
    for m in models:
     vi=[i for i in groups[b,ds,'val'] if masks[m][i]];ti=[i for i in groups[b,ds,'test'] if masks[m][i]];vr=[cases[i] for i in vi];tr=[cases[i] for i in ti];assert vr and tr
     pv=np.array([r['probe_probability'] for r in vr]);pt=np.array([r['probe_probability'] for r in tr]);yv=np.array([r['true_label'] for r in vr]);yt=np.array([r['true_label'] for r in tr]);wv=np.array([1/r['inclusion_probability'] for r in vr]);wt=np.array([1/r['inclusion_probability'] for r in tr])
     vp=s['populations']['val'] if scope=='local_full' else {'n':wv.sum(),'positive':wv@yv};pop=s['populations']['test'] if scope=='local_full' else {'n':wt.sum(),'positive':wt@yt}
     full=np.array(s['thresholds']);positions=np.searchsorted(np.sort(pv),full,side='left');ts=full[np.r_[positions[:-1]!=positions[1:],True]]
     base=dict(scope=scope,model=m,backbone=b,dataset=ds,validation_n=len(vi),test_n=len(ti))
     cache={}
     def evaluate(th,cut,ci):
      key=float(th),int(cut),ci
      if key not in cache:
       ref=pt>=th;yes=ref&(z[m][ti,ci]>=cut);metric=ns['metrics'](wt[yes&(yt==1)].sum(),wt[yes&(yt==0)].sum(),pop['n'],pop['positive'])
       cache[key]=dict(probe_threshold=float(th),judge_threshold=int(cut),test_referred=int(ref.sum()),test_accepted=int(yes.sum()),test_rejected=int((ref&~yes).sum()),referral_rate=float(wt[ref].sum()/pop['n']),**metric)
      return cache[key]
     for ci,c in enumerate(C):rows.append(dict(**base,policy='fixed50',anchor='fixed50',condition=c,**evaluate(.5,50,ci)))
     for ai,anchor in enumerate(C):
      g=ns['grid'](vr,z[m][vi,ai],ts,vp);cut,k=ns['choose'](g['f1'],ts);th=ts[k];thresholds.append(dict(**base,anchor=anchor,probe_threshold=float(th),judge_threshold=cut,validation_f1=float(g['f1'][cut,k])))
      for ci,c in enumerate(C):rows.append(dict(**base,policy='tuned_shared',anchor=anchor,condition=c,**evaluate(th,cut,ci)))
      if args.frontier:
       for bi,beta in enumerate(np.logspace(-1,1,41)):
        if bi==20:beta=1.
        den=beta**2*vp['positive']+g['tp']+g['fp'];obj=np.divide((1+beta**2)*g['tp'],den,out=np.zeros_like(den),where=den>0);fc,fk=ns['choose'](obj,ts)
        for ci,c in enumerate(C):frontier.append(dict(**base,anchor=anchor,condition=c,beta_index=bi,beta=float(beta),**evaluate(ts[fk],fc,ci)))
    print(scope,b,ds,flush=True)
 assert len(rows)==12000 and len(thresholds)==3000
 metrics=['f1','precision','recall','accuracy','referral_rate'];summary=[]
 for cohort in ['all10','toxicity4']:
  subset=[r for r in rows if cohort=='all10' or r['dataset'] in TOX]
  for scope in ['combined','local_full','api_matched','common_valid']:
   scoped=[r for r in subset if (scope=='combined' and r['scope']!='common_valid') or r['scope']==scope]
   for policy in ['fixed50','baseline','nearest','saliency','average_of_three']:
    ss=[r for r in scoped if (policy=='average_of_three' and r['policy']=='tuned_shared') or r['anchor']==policy]
    for dim in ['overall','model','backbone','dataset']:
     for group in ['ALL'] if dim=='overall' else sorted({r[dim] for r in ss}):
      rr=[r for r in ss if dim=='overall' or r[dim]==group];record=dict(cohort=cohort,scope=scope,policy=policy,dimension=dim,group=group)
      for c in C:
       cc=[r for r in rr if r['condition']==c]
       for metric in metrics:record[c+'_'+metric]=float(np.mean([r[metric] for r in cc]))
      for metric in metrics:record['nn_minus_control_'+metric]=record['nearest_'+metric]-record['baseline_'+metric]
      summary.append(record)
 savecsv(args.out/'cells.csv',rows);savecsv(args.out/'thresholds.csv',thresholds);savecsv(args.out/'summary.csv',summary)
 checked=0
 if args.check_existing:
  old=read(args.data/'anchor_transfer_20261018/results/cells.json');index={(r['scope'],r['model'],r['backbone'],r['dataset'],r['anchor'],r['condition']):r for r in rows}
  fixed=[dict(r,anchor='fixed50') for r in read(args.data/'two_policy_evaluation_20261018/results/cells.json') if r['policy']=='fixed_both50']
  for r in old+fixed:
   new=index[tuple(r[k] for k in ['scope','model','backbone','dataset','anchor','condition'])]
   for key in metrics+['probe_threshold','judge_threshold','test_referred']:assert abs(new[key]-r[key])<1e-10,(r,key,new[key])
   checked+=1
  assert checked==12000
 if args.frontier:
  assert len(frontier)==369000
  with gzip.open(args.out/'tradeoff_points.csv.gz','wt',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(frontier[0]));w.writeheader();w.writerows(frontier)
 (args.out/'VERIFIED.json').write_text(json.dumps({'time':time.time(),'new_model_calls':0,'rows':len(rows),'threshold_pairs':len(thresholds),'matched_historical_rows':checked,'frontier_rows':len(frontier),'inputs':spec,'selection':'validation-only; shared thresholds within each anchor; equal-cell macro means; common-valid sensitivity separate','limits':'previously examined test set; descriptive exploratory analysis; no private reasoning claim'},indent=2))
 print('Verified historical rows:',checked,flush=True)
if __name__=='__main__':main()
