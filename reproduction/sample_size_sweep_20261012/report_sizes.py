import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/ubuntu/experiments');OUT=ROOT/'sample-size-comparison-20261012';OUT.mkdir(exist_ok=True)
SIZES=[100,200,300,500];Js=['qwen','gemma'];MODES=['hard','joint'];records=[];check=[];cell=[];dataset_records=[]
for n in SIZES:
 p=ROOT/('sample-stability-20261012' if n==100 else f'sample-stability-n{n}-20261012')
 done=json.loads((p/'COMPLETED.json').read_text());assert done['replicates']==10000 and done['pairs']==100
 for name,sha in done['outputs'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha
 files=sorted(p.glob('*.npz'))
 datasets=sorted({f.stem.removeprefix(next(b+'_' for b in ['llama70','qwen122','llama','qwen','gemma'] if f.stem.startswith(b+'_'))) for f in files})
 for ds in datasets:
  zz=[np.load(p/(b+'_'+ds+'.npz')) for b in ['llama','qwen','gemma','llama70','qwen122']]
  for ji,j in enumerate(Js):
   for mi,mode in enumerate(MODES):
    dd=np.mean([z['sampled_test'][:,ji,6+mi]-z['sampled_test'][:,ji,mi] for z in zz],axis=0)*100
    lo,hi=np.quantile(dd,[.025,.975]);dataset_records.append(dict(n=n,dataset=ds,judge=j,mode=mode,mean_gain_pp=float(dd.mean()),low_pp=float(lo),high_pp=float(hi),halfwidth_pp=float((hi-lo)/2)))
 s=json.loads((p/'SUMMARY.json').read_text());details=json.loads((p/'PER_PAIR.json').read_text())
 for j in Js:
  for mode in MODES:
   v=s[j][mode];d=v['nn_minus_baseline'];a=[x for x in details if x['judge']==j and x['mode']==mode]
   record=dict(n=n,judge=j,mode=mode,reference_gain_pp=100*(v['reference_nn']-v['reference_baseline']),mean_gain_pp=100*d['mean'],low_pp=100*d['p025'],high_pp=100*d['p975'],halfwidth_pp=50*(d['p975']-d['p025']),nn_wins=v['nn_positive_fraction'],baseline_tuning_loss_pp=100*(v['reference_baseline']-v['validation_only_baseline_f1']['mean']),nn_tuning_loss_pp=100*(v['reference_nn']-v['validation_only_nn_f1']['mean']),median_cell_halfwidth_pp=float(np.median([50*(x['sampled_delta']['p975']-x['sampled_delta']['p025']) for x in a])),max_cell_halfwidth_pp=max(50*(x['sampled_delta']['p975']-x['sampled_delta']['p025']) for x in a))
   records.append(record)
 a=[r for r in records if r['n']==n];maxw=max(r['halfwidth_pp'] for r in a);maxloss=max(max(r['baseline_tuning_loss_pp'],r['nn_tuning_loss_pp']) for r in a)
 check.append(dict(n=n,max_halfwidth_pp=maxw,max_tuning_loss_pp=maxloss,passes_practical=maxw<=1 and maxloss<=.5,passes_strict=maxw<=.5 and maxloss<=.5,judgments_per_judge=n*2*5*10*3*2))
fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
for j in Js:
 for mode in MODES:
  a=[r for r in records if r['judge']==j and r['mode']==mode];axes[0].plot(SIZES,[r['halfwidth_pp'] for r in a],marker='o',label=j.title()+' '+mode);axes[1].plot(SIZES,[max(r['baseline_tuning_loss_pp'],r['nn_tuning_loss_pp']) for r in a],marker='o',label=j.title()+' '+mode)
axes[0].axhline(1,color='gray',ls='--',label='Practical precision: ±1 pp');axes[0].axhline(.5,color='black',ls=':',label='Strict precision: ±0.5 pp');axes[1].axhline(.5,color='black',ls=':',label='0.5 pp tuning loss');axes[0].set_ylabel('Half-width of central 95% resample range (F1 pp)');axes[1].set_ylabel('Largest mean tuning loss, baseline or NN (F1 pp)')
for ax in axes:ax.set_xlabel('Cases per split (same number for validation and test)');ax.set_xticks(SIZES);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(OUT/'sample_size_precision.png',dpi=180);plt.close(fig)
rows=''.join(f"<tr><td>{r['n']} + {r['n']}</td><td>{r['max_halfwidth_pp']:.2f}</td><td>{r['max_tuning_loss_pp']:.2f}</td><td>{r['passes_practical']}</td><td>{r['passes_strict']}</td><td>{r['judgments_per_judge']:,}</td></tr>" for r in check)
allrows=''.join(f"<tr><td>{r['n']}</td><td>{r['judge']}</td><td>{r['mode']}</td><td>{r['mean_gain_pp']:.2f}</td><td>[{r['low_pp']:.2f}, {r['high_pp']:.2f}]</td><td>{100*r['nn_wins']:.1f}%</td><td>{r['median_cell_halfwidth_pp']:.2f}</td></tr>" for r in records)
html='''<!doctype html><meta charset="utf-8"><title>Sample-size stability comparison</title><style>body{max-width:1150px;margin:35px auto;font:17px system-ui;line-height:1.5}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:8px}img{width:100%}</style><h1>Sample-size stability comparison</h1><p>10,000 paired stratified bootstrap replicates at each size; independently draw validation and test, retune frozen thresholds on validation, preserve design weights. Qwen and Gemma, five backbones, ten datasets. All three explanations and hard, fixed-50 and jointly tuned probability decisions are computed. Primary sizing checks concern NN versus baseline, separately for each judge in hard and joint modes.</p><p>Practical criterion: central 95% resample half-width no greater than 1 F1 percentage point in all four primary comparisons, and mean performance loss from smaller validation sets no greater than 0.5 point. Strict criterion uses 0.5 point half-width. These are explicitly chosen precision targets, not significance tests.</p><table><tr><th>Val + test</th><th>Worst half-width (pp)</th><th>Worst tuning loss (pp)</th><th>Practical passes</th><th>Strict passes</th><th>Judgments per judge</th></tr>'''+rows+'''</table><img src="sample_size_precision.png"><h2>Individual comparisons</h2><table><tr><th>n each split</th><th>Judge</th><th>Mode</th><th>Mean NN gain (pp)</th><th>Central 95% range</th><th>NN wins</th><th>Median per-cell half-width (pp)</th></tr>'''+allrows+'''</table><p>These are conditional empirical stability ranges using the existing 500-case pool, not population confidence intervals or guarantees for new judges. Resampling uses replacement, including at n=500; the uncertainty does not vanish merely because the pool contains 500 cases. Macro resampling is independent across backbone–dataset cells and does not model dependence from shared source examples. Per-cell results remain less precise than the macro average. Near-zero effects may remain sign-unstable even at larger sizes.</p><p><a href="RESULTS.json">Numeric results</a>. No inference or retraining was performed.</p>'''
(OUT/'report.html').write_text(html);(OUT/'RESULTS.json').write_text(json.dumps(dict(criteria=check,comparisons=records,per_dataset=dataset_records),indent=2));(OUT/'COMPLETED.json').write_text(json.dumps(dict(sizes=SIZES,replicates_each=10000,verified_npz_files=200,hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.name!='COMPLETED.json'}),indent=2));print(json.dumps(dict(criteria=check,comparisons=records,per_dataset=dataset_records),indent=2))
