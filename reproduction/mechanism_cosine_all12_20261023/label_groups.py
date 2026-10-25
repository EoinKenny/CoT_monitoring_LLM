import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import numpy as np,csv,gzip,json
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results'
a=np.load(P/'mechanisms_all12_20261023/annotations.npz');units=list(a['unit_ids']);lab=a['majority'];mapping={r['unit_id']:r for r in map(json.loads,open(P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl'))}
with gzip.open(P/'nn_mechanisms_20261015/results/joined_features.csv.gz','rt') as f:features={(r['backbone'],r['sample_id']):r for r in csv.DictReader(f)}
x=np.array([float(features[mapping[u]['backbone'],mapping[u]['sample_id']]['nearest_cosine']) for u in units]);assert np.isfinite(x).all()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,ax=plt.subplots(figsize=(8,5));rng=np.random.default_rng(20261023);names=['Labelled similarity','Important contrast'];colors=['#247a59','#8562aa'];stats=[]
for j in range(2):
 v=x[lab[:,j]==1];vi=ax.violinplot([v],positions=[j],widths=.7,showextrema=False)
 for b in vi['bodies']:b.set_facecolor(colors[j]);b.set_edgecolor(colors[j]);b.set_alpha(.2)
 take=rng.choice(len(v),min(2000,len(v)),replace=False);ax.scatter(j+rng.uniform(-.23,.23,len(take)),v[take],s=5,alpha=.13,color=colors[j],edgecolors='none',rasterized=True)
 q=np.quantile(v,[.25,.5,.75]);ax.plot([j,j],[q[0],q[2]],color='black',lw=3);ax.scatter([j],[q[1]],color='white',edgecolor='black',s=45,zorder=4);ax.text(j,1.025,f'n={len(v):,}\nmedian={q[1]:.3f}',ha='center',fontsize=10)
 stats.append(dict(label=names[j],n=len(v),mean=float(v.mean()),median=float(q[1]),q25=float(q[0]),q75=float(q[2])))
ax.set_xticks([0,1]);ax.set_xticklabels(names);ax.set_ylabel('NN–target cosine similarity');ax.set_ylim(min(0,float(x.min())-.02),1.12);ax.set_title('Cosine similarity by mechanism label');ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(O/'cosine_label_groups.png');fig.savefig(O/'cosine_label_groups.pdf');plt.close(fig)
with open(O/'cosine_label_groups.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
overlap=int(((lab[:,0]==1)&(lab[:,1]==1)).sum())
block='<h2>Direct comparison of the two labels</h2><img src="cosine_label_groups.png"><p>All test contexts, regardless of whether the NN changed a decision. Strict all-12 consensus (at least seven votes). Each context counts once per group; '+str(overlap)+' contexts belong to both groups. Distributions and medians use all contexts, unweighted; dots are a reproducible display sample. White dots show medians and black lines the interquartile range. <a href="cosine_label_groups.pdf">PDF</a> · <a href="cosine_label_groups.csv">Summary</a></p>'
p=O/'report.html';s=p.read_text();s=s.replace('<h1>NN cosine similarity by decision outcome</h1>','<h1>NN cosine similarity by decision outcome</h1>'+block);p.write_text(s)
print(json.dumps(dict(groups=stats,overlap=overlap),indent=2))
