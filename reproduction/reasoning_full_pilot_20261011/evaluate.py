from judge import ROOT,sha,save,CS
import json,collections
import numpy as np
def metric(y,p):
 y=np.array(y,dtype=int);p=np.array(p,dtype=int);tp=int(np.sum((y==1)&(p==1)));fp=int(np.sum((y==0)&(p==1)));fn=int(np.sum((y==1)&(p==0)));tn=int(np.sum((y==0)&(p==0)))
 return dict(n=len(y),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,accuracy=(tp+tn)/len(y) if len(y) else 0)
def main():
 prepared=json.loads((ROOT/'PREPARED.json').read_text());assert sha(ROOT/'SELECTION.json')==prepared['selection_sha256'];assert sha(ROOT/'OFF_CONTROLS.json')==prepared['off_controls_sha256']
 selected=json.loads((ROOT/'SELECTION.json').read_text());meta={(r['backbone'],r['sample_id']):r for r in selected};off=json.loads((ROOT/'OFF_CONTROLS.json').read_text());cases=[json.loads(l) for l in (ROOT/'cases.jsonl').open()];ids={r['id'] for r in cases};lookup={};raw_coverage={}
 for j in ['qwen','gemma']:
  d=ROOT/j;done=json.loads((d/'COMPLETED.json').read_text());assert sha(d/'RESULTS.json')==done['results_sha256'];x=json.loads((d/'RESULTS.json').read_text());rr=[]
  for b in x['batches']:
   assert sha(d/(b['tag']+'.raw.json'))==b['raw_sha256'];assert b['config_sha256']==sha(d/'CONFIG.json');rr+=b['records']
  assert len(rr)==len(ids) and {r['id'] for r in rr}==ids
  for r in rr:
   value=int(r['final']) if r['valid'] else None;lookup[j,'on',r['condition'],r['mode'],r['backbone'],r['sample_id']]=None if value is None else (value if r['mode']=='hard' else int(value>=50))
  for r in off:
   if r['judge']==j:lookup[j,'off',r['condition'],r['mode'],r['backbone'],r['sample_id']]=r['value'] if r['mode']=='hard' else int(r['value']>=50)
  raw_coverage[j]=dict(n=len(rr),valid=sum(r['valid'] for r in rr),truncated=sum(r['finish_reason']=='length' for r in rr),seconds=sum(b['seconds'] for b in x['batches']),input_tokens=sum(r['input_tokens'] for r in rr),generated_tokens=sum(r['generated_tokens'] for r in rr))
 def pred(j,thinking,c,mode,r,pipeline):
  if pipeline and r['probe_probability']<r['probe_threshold']:return 0
  return lookup[j,thinking,c,mode,r['backbone'],r['sample_id']]
 summaries=[]
 for j in ['qwen','gemma']:
  for thinking in ['off','on']:
   for mode in ['hard','probability']:
    for c in CS:
     for pipeline in [False,True]:
      for ds in ['ALL']+sorted({r['dataset'] for r in selected}):
       rows=[r for r in selected if ds=='ALL' or r['dataset']==ds];y=[r['true_label'] for r in rows];p=[pred(j,thinking,c,mode,r,pipeline) for r in rows];known=[i for i,z in enumerate(p) if z is not None]
       lower=[(1-y[i]) if z is None else z for i,z in enumerate(p)];upper=[y[i] if z is None else z for i,z in enumerate(p)]
       summaries.append(dict(judge=j,thinking=thinking,mode=mode,condition=c,pipeline=pipeline,dataset=ds,n=len(rows),unresolved=len(rows)-len(known),complete_case=metric([y[i] for i in known],[p[i] for i in known]),worst_case=metric(y,lower),best_case=metric(y,upper)))
 comparisons=[];rng=np.random.default_rng(20261011)
 for j in ['qwen','gemma']:
  for mode in ['hard','probability']:
   for pipeline in [False,True]:
    pairs=[(('on',c),('on','baseline')) for c in ['saliency','nearest']]+[(('on',c),('off',c)) for c in CS]
    for a,b in pairs:
     matched=[r for r in selected if pred(j,*a,mode,r,pipeline) is not None and pred(j,*b,mode,r,pipeline) is not None]
     # Cluster duplicate source groups across backbones; stratify by dataset and gold label.
     groups=collections.defaultdict(lambda:collections.defaultdict(list))
     for i,r in enumerate(matched):groups[r['dataset'],r['true_label']][r['group']].append(i)
     strata=[list(g.values()) for g in groups.values()];y=np.array([r['true_label'] for r in matched]);pa=np.array([pred(j,*a,mode,r,pipeline) for r in matched]);pb=np.array([pred(j,*b,mode,r,pipeline) for r in matched]);diff=[]
     for k in range(2000):
      ix=np.array([i for clusters in strata for h in rng.integers(0,len(clusters),len(clusters)) for i in clusters[h]],dtype=int);diff.append(metric(y[ix],pa[ix])['f1']-metric(y[ix],pb[ix])['f1'])
     comparisons.append(dict(judge=j,mode=mode,pipeline=pipeline,a=a,b=b,n_matched=len(matched),a_metrics=metric(y,pa),b_metrics=metric(y,pb),f1_difference=metric(y,pa)['f1']-metric(y,pb)['f1'],paired_cluster_bootstrap_95=np.quantile(diff,[.025,.975]).tolist()))
 probe=metric([r['true_label'] for r in selected],[r['probe_probability']>=r['probe_threshold'] for r in selected])
 save(ROOT/'RESULTS.json',dict(coverage=raw_coverage,probe_only=probe,summaries=summaries,comparisons=comparisons,notes=['Descriptive balanced test pilot, not a representative full-test F1 estimate. No new thresholds selected on test.','Probability cutoff fixed at50; original validation-selected probe threshold held fixed.','Bootstrap comparisons use paired complete cases; missing-output worst/best bounds are reported separately to expose selection effects.','Native thinking output unconstrained; historical nonthinking controls used constrained final decoding. This is a practical mode comparison, not isolation of reasoning alone.','Only32 examples per dataset across five backbones; per-dataset effects uncertain. No fresh held-out confirmatory claim.']))
 report=['Balanced reasoning pilot completed.','Calls: '+str(raw_coverage),'Probe-only balanced-pilot metrics: '+str(probe),'Aggregate judge-only results (complete cases; inspect bounds where unresolved):']
 for r in summaries:
  if r['dataset']=='ALL' and not r['pipeline']:report.append(f"{r['judge']} thinking={r['thinking']} {r['mode']} {r['condition']}: F1={r['complete_case']['f1']:.4f}, n={r['complete_case']['n']}/{r['n']}, unresolved={r['unresolved']}")
 (ROOT/'REPORT.txt').write_text('\n'.join(report)+'\n');save(ROOT/'ANALYSIS_COMPLETED.json',dict(results_sha256=sha(ROOT/'RESULTS.json'),report_sha256=sha(ROOT/'REPORT.txt')))
if __name__=='__main__':main()
