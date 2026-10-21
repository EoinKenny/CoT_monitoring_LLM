import json,hashlib,time,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/home/ubuntu/experiments/sample-stability-20261012')
s=json.loads((O/'SUMMARY.json').read_text());pairs=json.loads((O/'PER_PAIR.json').read_text())
files=sorted(O.glob('*.npz'));zs=[np.load(f) for f in files];sample=np.array([z['sampled_test'] for z in zs]);full=np.array([z['full_test'] for z in zs]);refs=np.array([z['reference'] for z in zs]);fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for ji,j in enumerate(['qwen','gemma']):
 for mi,m in enumerate(['hard','joint']):
  ax=axs[mi,ji];d=(sample[:,:,ji,6+mi]-sample[:,:,ji,mi]).mean(0)*100;df=(full[:,:,ji,6+mi]-full[:,:,ji,mi]).mean(0)*100;ref=(refs[:,ji,6+mi]-refs[:,ji,mi]).mean()*100
  ax.hist(d,bins=55,density=True,alpha=.6,label='100 validation + 100 test');ax.hist(df,bins=55,density=True,histtype='step',linewidth=1.8,label='100 validation + fixed 500 test');ax.axvline(0,color='black',lw=1);ax.axvline(ref,color='firebrick',linestyle='--',label='Original 500 + 500');ax.set_title(j.title()+' — '+('hard labels' if m=='hard' else 'jointly tuned probabilities'));ax.set_xlabel('NN minus no-explanation F1 (percentage points)');ax.set_ylabel('Bootstrap density')
axs[0,0].legend(fontsize=8);fig.savefig(O/'stability.png',dpi=160);plt.close(fig)
rows=[]
for j,mm in s.items():
 for m,v in mm.items():
  d=v['nn_minus_baseline'];f=v['validation_only_nn_minus_baseline'];rows.append(f"<tr><td>{j}</td><td>{m}</td><td>{100*(v['reference_nn']-v['reference_baseline']):.3f}</td><td>{100*d['mean']:.3f}</td><td>[{100*d['p025']:.3f}, {100*d['p975']:.3f}]</td><td>{100*v['nn_positive_fraction']:.2f}%</td><td>[{100*f['p025']:.3f}, {100*f['p975']:.3f}]</td></tr>")
text='''<!doctype html><meta charset="utf-8"><title>100-case sample stability</title><style>body{max-width:1150px;margin:40px auto;font:17px system-ui;line-height:1.6}table{border-collapse:collapse;font-size:15px}td,th{padding:8px;border:1px solid #ddd}img{width:100%}</style><h1>100-case sample stability: 10,000 paired bootstrap replicates</h1><p>Existing Qwen and Gemma judgments, five backbones, ten datasets. Each replicate draws 100 validation cases and independently 100 test cases, with replacement inside the original score-band × gold-label strata. Quotas match the reduced experiment; weights preserve each stratum's population. The same cases are used across judges, conditions and output formats. All thresholds are reselected on validation only, with the original candidate grid and tie rules.</p><p>Each judge average equally weights 50 backbone–dataset cells. The original reference uses all 500 validation and 500 test cases. A second analysis evaluates every 100-case validation fit on the unchanged 500-case test pool, separating threshold instability from test sampling noise.</p><table><tr><th>Judge</th><th>Mode</th><th>Original NN gain (pp)</th><th>Mean 100-case gain (pp)</th><th>Central 95% resample range</th><th>Replicates NN better</th><th>Validation-only range</th></tr>'''+''.join(rows)+'''</table><img src="stability.png"><h2>Interpretation limits</h2><p>These are empirical stability ranges conditional on the existing 500-case pool, not population confidence intervals or probabilities that a hypothesis is true. With-replacement bootstrap draws can contain repeated cases; the actual 100-case study selects distinct cases. Draws are paired across judges and conditions within each backbone–dataset cell, but independent across backbone cells; pooled ranges do not account for shared source examples across backbones. No model calls or retraining were performed.</p><p>Per-cell results: <a href="PER_PAIR.json">PER_PAIR.json</a>. Aggregate results: <a href="SUMMARY.json">SUMMARY.json</a>. Frozen inputs and method: <a href="PROTOCOL.json">PROTOCOL.json</a>. Raw replicate scores and thresholds are retained in the server's per-cell NPZ files.</p>'''
(O/'report.html').write_text(text)
print(json.dumps(s,indent=2))
