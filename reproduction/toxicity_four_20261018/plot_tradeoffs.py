"""Validation-selected, matched-cutoff precision-recall curves for four datasets."""
import os
os.environ['MPLCONFIGDIR']='/tmp/probe-matplotlib'
import json, gzip, hashlib, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R=Path(__file__).resolve().parent; O=R/'results'; S=R.parent/'anchor_transfer_20261018/results'
C=['baseline','saliency','nearest']; A=['baseline','nearest','saliency']
DS=['toxigen','toxicchat','beavertails','aegis']
COL={'baseline':'#64748b','saliency':'#d97706','nearest':'#1676c5'}
N={'baseline':'No explanation','saliency':'Saliency','nearest':'1 NN'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=S/'tradeoff_points.json.gz'
assert sha(source)==json.loads((S/'COMPLETED.json').read_text())['outputs'][source.name]
with gzip.open(source,'rt') as f: rows=[r for r in json.load(f) if r['dataset'] in DS]
assert len(rows)==147600
groups=collections.defaultdict(list)
for r in rows:groups[tuple(r[k] for k in ['scope','model','backbone','dataset','anchor','beta_index'])].append(r)
for rr in groups.values():
    assert len(rr)==3 and {r['condition'] for r in rr}==set(C)
    assert len({(r['probe_threshold'],r['judge_threshold'],r['test_n'],r['test_referred']) for r in rr})==1
agg=collections.defaultdict(list)
for r in rows:
    scope='common_valid' if r['scope']=='common_valid' else 'combined'
    for ds in ['ALL',r['dataset']]:
        for anchor in [r['anchor'],'average']:
            agg[scope,ds,anchor,r['condition'],r['beta_index']].append([r['recall']*100,r['precision']*100,r['f1']*100])
points={k:np.mean(v,axis=0) for k,v in agg.items()}
summary=json.loads((O/'summary.json').read_text())
for r in summary:
    if r['scope'] not in ['combined','common_valid'] or r['policy']=='fixed50' or r['dimension']!='overall':continue
    a='average' if r['policy']=='average_of_three' else r['policy']
    for c in C:
        assert np.allclose(points[r['scope'],'ALL',a,c,20],[r[c][m]*100 for m in ['recall','precision','f1']],rtol=0,atol=1e-10)
data=[dict(scope=k[0],dataset=k[1],anchor=k[2],condition=k[3],beta_index=k[4],beta=10**(-1+k[4]/20),recall=v[0],precision=v[1],mean_cell_f1=v[2],averaged_records=len(agg[k])) for k,v in points.items()]
(O/'tradeoff_aggregates.json').write_text(json.dumps(data,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})
def line(ax,scope,ds,anchor):
    for c in C:
        xy=np.array([points[scope,ds,anchor,c,i] for i in range(41)])
        ax.plot(xy[:,0],xy[:,1],color=COL[c],lw=2.4,marker='.',ms=4,label=N[c])
        ax.scatter(*xy[20,:2],marker='*',s=235,color=COL[c],edgecolors='white',lw=.9,zorder=6)
    ax.set_xlabel('Mean test recall (%)');ax.set_ylabel('Mean test precision (%)')
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)
for scope,suffix in [('combined',''),('common_valid','_common_valid')]:
    fig,axs=plt.subplots(2,2,figsize=(14,10.5),sharex=True,sharey=True)
    fig.subplots_adjust(top=.86,bottom=.18,left=.075,right=.98,hspace=.36,wspace=.20)
    titles=['Tune on no explanation','Tune on 1 NN','Tune on saliency','Average across the three tuning sources']
    for ax,a,title in zip(axs.flat,A+['average'],titles):
        line(ax,scope,'ALL',a);ax.set_title(title,fontsize=14,pad=13)
        if a!='average':ax.text(.02,.035,'Both cutoffs transferred unchanged to all conditions',transform=ax.transAxes,fontsize=9,color='#465669')
        else:ax.text(.02,.035,'Descriptive average; not one deployable threshold pair',transform=ax.transAxes,fontsize=9,color='#465669')
    fig.suptitle('Toxicity and content safety: precision–recall trade-offs',fontsize=19,y=.965)
    fig.text(.5,.92,'ToxiGen • ToxicChat • BeaverTails • Aegis  |  200 judge–backbone–dataset combinations',ha='center',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color=COL[c],lw=3,label=N[c]) for c in C],loc='lower center',bbox_to_anchor=(.5,.103),ncol=3,frameon=False,fontsize=12)
    fig.text(.5,.083,'Stars: validation-F1 selections. Lines: 41 validation Fβ objectives, from precision-oriented to recall-oriented.',ha='center',fontsize=10)
    fig.text(.5,.057,'Same cases and both thresholds within each comparison. Test outcomes never select the points; no smoothing.',ha='center',fontsize=10)
    fig.text(.5,.031,('Original local/API evaluation scopes; zero judge cutoffs retained.' if scope=='combined' else 'Sensitivity analysis: identical valid cases across all ten judges; zero judge cutoffs retained.'),ha='center',fontsize=10)
    save(fig,'precision_recall'+suffix)

fig,axs=plt.subplots(2,2,figsize=(14,10.5),sharex=True,sharey=True)
fig.subplots_adjust(top=.85,bottom=.17,left=.075,right=.98,hspace=.33,wspace=.20)
for ax,ds,title in zip(axs.flat,DS,['ToxiGen','ToxicChat','BeaverTails','Aegis']):
    line(ax,'combined',ds,'average');ax.set_title(title,fontsize=15,pad=12)
fig.suptitle('Precision–recall trade-offs by dataset',fontsize=19,y=.96)
fig.text(.5,.91,'Equal average across three tuning sources and 50 judge–backbone combinations per dataset',ha='center',fontsize=12)
fig.legend(handles=[Line2D([0],[0],color=COL[c],lw=3,label=N[c]) for c in C],loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=12)
fig.text(.5,.068,'Stars: mean of validation-F1-selected points. Each source applies identical probe and judge cutoffs to all conditions.',ha='center',fontsize=10)
fig.text(.5,.040,'Descriptive mean curves, not independently optimised test frontiers. Original local/API scopes; zero cutoffs retained.',ha='center',fontsize=10)
save(fig,'precision_recall_by_dataset')
page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Toxicity precision–recall trade-offs</title><style>body{font:17px/1.5 system-ui;margin:35px auto;max-width:1350px;padding:0 20px;color:#172335}img{width:100%}a{color:#1676c5}</style><h1>Toxicity precision–recall trade-offs</h1><p>Four datasets, ten judges and five backbones. Threshold pairs were selected using validation Fβ, then held identical across no explanation, saliency and NN. The 41 β values range from 0.1 to 10: smaller values favour precision and larger values favour recall. Stars mark β=1, the F1 selections from the tables.</p><p>The points are joined in validation-objective order without smoothing or test-based pruning. These are matched-threshold trade-off paths, not independent test-optimised Pareto envelopes. The average panel averages metric coordinates across three sources and is not a deployable single policy. Original weights and zero-cutoff cases are retained; no new model calls.</p><img src="precision_recall.png" alt="Precision recall curves for three threshold-selection sources and their average"><p><a href="precision_recall.pdf">Download main PDF</a> · <a href="precision_recall.svg">Main SVG</a></p><img src="precision_recall_by_dataset.png" alt="Mean precision recall curves for each of four datasets"><p><a href="precision_recall_by_dataset.pdf">Download dataset PDF</a></p><details><summary>Same-valid-cases sensitivity across all ten judges</summary><img src="precision_recall_common_valid.png" alt="Common valid cases precision recall sensitivity"></details><p><a href="report.html">Full numerical tables and exact F1 thresholds</a> · <a href="tradeoff_aggregates.json">All plotted coordinates</a></p></html>'''
(O/'tradeoffs.html').write_text(page)
(O/'PLOTS_VERIFIED.json').write_text(json.dumps(dict(source_sha256=sha(source),source_rows=len(rows),matched_threshold_groups=len(groups),beta1_replays=True,new_model_calls=0,outputs={p.name:sha(p) for p in O.iterdir() if p.name.startswith(('precision_recall','tradeoff'))},script_sha256=sha(Path(__file__))),indent=2))
for a in A+['average']:
    print(a,{c:points['combined','ALL',a,c,20].round(3).tolist() for c in C})
