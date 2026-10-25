import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import csv,gzip,json,collections
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';O.mkdir(exist_ok=True)
a=np.load(P/'mechanisms_all12_20261023/annotations.npz');M=list(a['models']);units=list(a['unit_ids']);mapping={r['unit_id']:r for r in map(json.loads,open(P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl'))};rr=[mapping[u] for u in units]
d=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rs in d['samples'].items() for r in rs];look={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)};ix=np.array([look[r['backbone'],r['sample_id']] for r in rr]);y=np.array([cases[i]['true_label'] for i in ix]);w0=np.array([1/cases[i]['inclusion_probability'] for i in ix]);cell=np.array([r['backbone']+':'+r['dataset'] for r in rr]);z=np.load(P/'main_performance_all12_20261023/scores.npz')
with gzip.open(P/'nn_mechanisms_20261015/results/joined_features.csv.gz','rt') as f:features={(r['backbone'],r['sample_id']):r for r in csv.DictReader(f)}
cos=np.array([float(features[r['backbone'],r['sample_id']]['nearest_cosine']) for r in rr]);assert np.isfinite(cos).all() and ((cos>=-1)&(cos<=1+1e-8)).all()
parts=[]
for ji,m in enumerate(M):
 s=z[m][ix];valid=(s[:,0]>=0)&(s[:,2]>=0);bc=(s[:,0]>=50)==y;nc=(s[:,2]>=50)==y;ev=np.where(~bc&nc,0,np.where(bc&~nc,2,1));w=np.zeros(len(rr))
 for c in np.unique(cell):
  mask=valid&(cell==c);w[mask]=w0[mask]/w0[mask].sum()
 parts.append((cos[valid],ev[valid],w[valid],a['leave_one_out'][ji,valid],np.where(valid)[0]))
x=np.concatenate([p[0] for p in parts]);ev=np.concatenate([p[1] for p in parts]);w=np.concatenate([p[2] for p in parts]);labels=np.concatenate([p[3] for p in parts]);ids=np.concatenate([p[4] for p in parts]);rng=np.random.default_rng(20261023)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
E=['Wrong → right','No change','Right → wrong'];COL=['#247a59','#8795a8','#bb7168'];stats=[]
fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)
for li,(ax,title) in enumerate(zip(axs,['Labelled similarity = yes','Important contrast = yes'])):
 for ei in range(3):
  mask=(labels[:,li]==1)&(ev==ei);xx=x[mask];ww=w[mask];assert len(xx)>0
  # Weighted histogram silhouettes and summary use every eligible row.
  edges=np.linspace(-1,1,201);h,_=np.histogram(xx,bins=edges,weights=ww);h=np.convolve(h,[.15,.2,.3,.2,.15],mode='same');cent=(edges[:-1]+edges[1:])/2;width=h/h.max()*.34
  ax.fill_betweenx(cent,ei-width,ei+width,color=COL[ei],alpha=.2)
  take=rng.choice(len(xx),min(1600,len(xx)),replace=False);ax.scatter(ei+rng.uniform(-.23,.23,len(take)),xx[take],s=4,alpha=.16,color=COL[ei],edgecolors='none',rasterized=True)
  order=np.argsort(xx);q=np.interp([.25,.5,.75],np.cumsum(ww[order])/ww.sum(),xx[order]);ax.plot([ei,ei],[q[0],q[2]],color='black',lw=3);ax.scatter([ei],[q[1]],s=34,color='white',edgecolor='black',zorder=4)
  ax.text(ei,1.025,f'n={len(xx):,}',ha='center',fontsize=9)
  stats.append(dict(feature=['labelled_similarity','important_contrast'][li],outcome=E[ei],judge_case_pairs=len(xx),unique_contexts=len(np.unique(ids[mask])),mean=float(ww@xx/ww.sum()),q25=q[0],median=q[1],q75=q[2]))
 ax.set_title(title,pad=22);ax.set_xticks(range(3));ax.set_xticklabels(E);ax.set_ylim(max(-1,float(x.min())-.03),1.065);ax.set_xlim(-.5,2.5);ax.grid(axis='y',alpha=.15)
axs[0].set_ylabel('Cosine similarity: target ↔ nearest neighbour');fig.suptitle('NN similarity by decision outcome · all 12 judges',fontweight='bold',y=1.02);fig.tight_layout();fig.savefig(O/'cosine_swarm.png',bbox_inches='tight');fig.savefig(O/'cosine_swarm.pdf',bbox_inches='tight');plt.close(fig)
with open(O/'summary.csv','w') as f:q=csv.DictWriter(f,fieldnames=list(stats[0]));q.writeheader();q.writerows(stats)
body='<h1>NN cosine similarity by decision outcome</h1><p>Two overlapping groups: labelled similarity=yes and important contrast=yes. Labels use the majority of the other 11 annotators, excluding the decision judge. Original control versus 1NN decisions use cutoff50, without probe gating, matching the preceding mechanism analysis.</p><img src="cosine_swarm.png"><p>Dots are a reproducible uniform display sample of at most1,600 actual judge–case pairs per distribution. Shapes use every eligible pair with inverse-inclusion weights normalized to equal judge/backbone/dataset cell mass. White dots are weighted medians; black lines span the weighted interquartile range. Counts refer to all eligible pairs, not displayed dots; a target/backbone context can repeat across judges and across both panels. Unchanged includes both correct and incorrect stable predictions.</p><p>Raw activation cosine similarity can differ in scale across backbones/datasets. These pooled distributions are descriptive; cosine distance to a neighbour is not distance to the probe decision boundary.</p><table><tr><th>Group</th><th>Outcome</th><th>Pairs</th><th>Unique contexts</th><th>Mean</th><th>Median</th></tr>'
for r in stats:body+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [r['feature'],r['outcome'],r['judge_case_pairs'],r['unique_contexts'],f"{r['mean']:.3f}",f"{r['median']:.3f}"])+'</tr>'
body+='</table><p><a href="cosine_swarm.pdf">PDF</a> · <a href="summary.csv">Summary data</a></p>'
(O/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NN cosine distributions</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#182738}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}</style>'+body)
print(json.dumps(stats,indent=2))
