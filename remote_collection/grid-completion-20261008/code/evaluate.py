"""Exact probe and sampled-flag metrics, design-weighted cascade estimates and paired CIs."""
import sys,csv
from common import *
import numpy as np

def load_predictions(path,selected):
 mark=read(path/'run/COMPLETED');assert sha(path/'run/judgments.jsonl')==mark['judgments_sha256'];rr=[json.loads(l) for l in (path/'run/judgments.jsonl').open()];by={r['sample_id']:r for r in rr}
 assert len(by)==len(rr)==len(selected) and set(by)=={r['sample_id'] for r in selected}
 assert all(by[r['sample_id']]['dataset']==r['dataset'] for r in selected)
 return np.array([by[r['sample_id']]['prediction'] for r in selected],int)
def paired_intervals(base,rows,p0,p1,seed):
 if not rows:return dict(delta_f1=0.,ci95=[0.,0.],replicates=0,scope='No flags')
 a=np.array([r['true_label'] for r in rows]);rng=np.random.default_rng(seed);deltas=[]
 # Census strata remain fixed. Non-census strata are resampled together across conditions.
 # The ordinary bootstrap ignores finite-population correction, conservatively when fractions are large.
 for _ in range(2000):
  ix=[]
  for y in [0,1]:
   s=np.flatnonzero(a==y)
   if len(s):ix.extend(s if rows[s[0]]['inclusion_probability']==1 else rng.choice(s,len(s),replace=True))
  rr=[rows[i] for i in ix];deltas.append(pipeline_counts(base,rr,p1[ix])['f1']-pipeline_counts(base,rr,p0[ix])['f1'])
 return dict(delta_f1=pipeline_counts(base,rows,p1)['f1']-pipeline_counts(base,rows,p0)['f1'],ci95=np.quantile(deltas,[.025,.975]).tolist(),replicates=2000,seed=seed,scope='Paired within-gold-stratum sample bootstrap, census strata fixed; conditional on fitted probes and fixed held-out population. No finite-population correction; approximate, not population-generalization uncertainty.')
def evaluate(b,m):
 R=ROOT/b;selected=read(R/'selected_flags.json');frozen=read(R/'SAMPLE_FROZEN.json');assert sha(R/'selected_flags.json')==frozen['selection_sha256'];results=read(R/'results.json');pred={c:load_predictions(R/'judges'/f'{m}_{c}',selected) for c in CONDITIONS};out=[];comparisons=[]
 for ds in DATASETS:
  ix=[i for i,r in enumerate(selected) if r['dataset']==ds];rows=[selected[i] for i in ix];base=results[ds]['metrics']['test'];gold=np.array([r['true_label'] for r in rows]);exact=all(r['inclusion_probability']==1 for r in rows)
  for c in CONDITIONS:
   p=pred[c][ix];sample=confusion_metrics(sum((gold==1)&(p==1)),sum((gold==0)&(p==1)),sum((gold==1)&(p==0)),sum((gold==0)&(p==0)))
   cascade=pipeline_counts(base,rows,p);out.append(dict(backbone=b,dataset=ds,judge=m,condition=c,sampled_flags=len(rows),selected_tp=int(sum(gold)),selected_fp=int(sum(gold==0)),probe_f1=base['f1'],sample_f1=sample['f1'],pipeline_f1=cascade['f1'],pipeline_accuracy=cascade['accuracy'],delta_f1=cascade['f1']-base['f1'],full_test_exact=exact,**{'pipeline_'+k:cascade[k] for k in ['tp','fp','fn','tn']},sample_metrics=sample,pipeline_metrics=cascade,sampling=frozen['counts'][ds]))
  for c in ['saliency','nearest']:
   seed=int(digest(':'.join([str(SEED),b,m,ds,c]))[:16],16);comparison=paired_intervals(base,rows,pred['baseline'][ix],pred[c][ix],seed)
   flip={str(y):dict(baseline_wrong_explanation_right=int(sum((gold==y)&(pred['baseline'][ix]!=gold)&(pred[c][ix]==gold))),baseline_right_explanation_wrong=int(sum((gold==y)&(pred['baseline'][ix]==gold)&(pred[c][ix]!=gold)))) for y in [0,1]}
   comparisons.append(dict(backbone=b,dataset=ds,judge=m,condition=c,vs='baseline',sample_flips=flip,**comparison))
 write(R/f'{m}_EVALUATION.json',dict(results=out,paired_comparisons=comparisons));return out

def aggregate():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 allresults=[];probes=[];comparisons=[]
 for b in OLD:
  R=ROOT/b;results=read(R/'results.json');fig,axes=plt.subplots(2,5,figsize=(19,7))
  for ds,ax in zip(DATASETS,axes.flat):
   v=results[ds];roc=np.load(R/f'{ds}_roc.npz');ax.plot(roc['fpr'],roc['tpr'],label=f"AUC={v['metrics']['test']['auc']:.3f}");ax.plot([0,1],[0,1],'k--',alpha=.3);ax.set(title=ds,xlabel='False positive rate',ylabel='True positive rate');ax.legend()
   for split in ['train','val','test']:probes.append(dict(backbone=b,dataset=ds,split=split,threshold=v['threshold'],**v['metrics'][split]))
  fig.suptitle(b+' backbone: probe ROC on complete test sets');fig.tight_layout();fig.savefig(ROOT/f'{b}_probe_roc.png',dpi=170);plt.close(fig)
  for m in JUDGES:
   ev=read(R/f'{m}_EVALUATION.json');allresults+=ev['results'];comparisons+=ev['paired_comparisons']
 assert len(allresults)==len(OLD)*len(JUDGES)*30 and len(probes)==len(OLD)*30
 for filename,rows in [('judge_results.csv',allresults),('probe_results.csv',probes)]:
  flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
  with (ROOT/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 write(ROOT/'RESULTS.json',dict(judge_results=allresults,probe_results=probes,paired_comparisons=comparisons))
 lines=['# Revised sampled evaluation','', 'Full eligible training, balanced class weights, frozen validation-F1 thresholds. Pipeline metrics are design-weighted full-test estimates unless all flags were judged. Sample F1 is reported separately.','', 'Training-policy selection on FAVA used test F1 and is exploratory. Historical test/pilot inspection also limits confirmatory claims. FAVA is highly positive and should be compared with the always-positive baseline.','', '| Backbone | Judge | Baseline macro F1 | Saliency macro F1 | NN macro F1 |','|---|---|---:|---:|---:|']
 for b in OLD:
  for m in JUDGES:
   means=[np.mean([r['pipeline_f1'] for r in allresults if r['backbone']==b and r['judge']==m and r['condition']==c]) for c in CONDITIONS];lines.append('| '+b+' | '+m+' | '+' | '.join(f'{v:.4f}' for v in means)+' |')
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');write(ROOT/'COMPLETED',dict(judge_cells=len(allresults),probes=len(OLD)*10,results_sha256=sha(ROOT/'RESULTS.json'),time=time.time()))
if __name__=='__main__':
 if len(sys.argv)==3:evaluate(*sys.argv[1:])
 else:aggregate()
