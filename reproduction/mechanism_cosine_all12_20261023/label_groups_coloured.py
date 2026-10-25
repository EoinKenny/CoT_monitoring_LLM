import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import numpy as np,csv,gzip,json
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results'
a=np.load(P/'mechanisms_all12_20261023/annotations.npz');units=list(a['unit_ids']);lab=a['majority'];mapping={r['unit_id']:r for r in map(json.loads,open(P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl'))}
with gzip.open(P/'nn_mechanisms_20261015/results/joined_features.csv.gz','rt') as f:features={(r['backbone'],r['sample_id']):r for r in csv.DictReader(f)}
x=np.array([float(features[mapping[u]['backbone'],mapping[u]['sample_id']]['nearest_cosine']) for u in units]);assert np.isfinite(x).all()
# Outcomes are original control versus one-NN decisions at cutoff 50.
d=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];lookup={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)}
ix=np.array([lookup[mapping[u]['backbone'],mapping[u]['sample_id']] for u in units]);y=np.array([cases[i]['true_label'] for i in ix]);z=np.load(P/'main_performance_all12_20261023/scores.npz');outcomes=[];valids=[]
for m in a['models']:
 scores=z[m][ix];valids.append((scores[:,0]>=0)&(scores[:,2]>=0));bc=(scores[:,0]>=50)==y;nc=(scores[:,2]>=50)==y;outcomes.append(np.where(~bc&nc,0,np.where(bc&~nc,2,1)))
outcomes=np.array(outcomes);valids=np.array(valids)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(8,5));rng=np.random.default_rng(20261023);names=['Labelled similarity','Important contrast'];colors=['#247a59','#8562aa'];stats=[]
for j in range(2):
 v=x[lab[:,j]==1];vi=ax.violinplot([v],positions=[j],widths=.7,showextrema=False)
 for b in vi['bodies']:b.set_facecolor(colors[j]);b.set_edgecolor(colors[j]);b.set_alpha(.2)
 ji,ci=np.where(valids & (lab[:,j]==1)[None,:]);take=rng.choice(len(ci),min(5000,len(ci)),replace=False);cx=ci[take];events=outcomes[ji[take],cx];xx=j+rng.uniform(-.25,.25,len(take))
 for ei,color,name in [(1,'#a6acb4','Unchanged'),(0,'#168452','Wrong → correct'),(2,'#c94540','Correct → wrong')]:
  mask=events==ei;ax.scatter(xx[mask],x[cx[mask]],s=7,alpha=.25 if ei==1 else .55,color=color,edgecolors='none',rasterized=True,label=name if j==0 else None)
 q=np.quantile(v,[.25,.5,.75]);ax.plot([j,j],[q[0],q[2]],color='black',lw=3);ax.scatter([j],[q[1]],color='white',edgecolor='black',s=45,zorder=4);ax.text(j,1.025,f'n={len(v):,}\nmedian={q[1]:.3f}',ha='center',fontsize=10)
 stats.append(dict(label=names[j],n=len(v),mean=float(v.mean()),median=float(q[1]),q25=float(q[0]),q75=float(q[2])))
ax.set_xticks([0,1]);ax.set_xticklabels(names);ax.set_ylabel('NN–target cosine similarity');ax.set_ylim(min(0,float(x.min())-.02),1.12);ax.set_title('Cosine similarity by mechanism label and decision outcome');ax.legend(frameon=False,loc='lower right',fontsize=9);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(O/'cosine_label_groups_coloured.png');fig.savefig(O/'cosine_label_groups_coloured.pdf');plt.close(fig)
with open(O/'cosine_label_groups_coloured.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
overlap=int(((lab[:,0]==1)&(lab[:,1]==1)).sum())
block='<h2>Label groups with dots coloured by decision outcome</h2><img src="cosine_label_groups_coloured.png"><p>All test contexts, regardless of whether the NN changed a decision. Strict all-12 consensus (at least seven votes). Violin shapes and black summaries count each context once per group; '+str(overlap)+' contexts belong to both groups. Distributions and medians use all contexts, unweighted; coloured dots are a uniform sample of up to 5,000 valid judge–context pairs per group, preserving outcome frequencies in expectation. Green means wrong→correct, red correct→wrong, grey unchanged. Original probability-only decisions use cutoff 50 without probe gating. A context can have different outcomes for different judges. White dots show medians and black lines the interquartile range. <a href="cosine_label_groups_coloured.pdf">PDF</a> · <a href="cosine_label_groups_coloured.csv">Summary</a></p>'
p=O/'report.html';s=p.read_text();s=s.replace('<h1>NN cosine similarity by decision outcome</h1>','<h1>NN cosine similarity by decision outcome</h1>'+block);p.write_text(s)
print(json.dumps(dict(groups=stats,overlap=overlap),indent=2))
