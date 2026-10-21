"""Probe-fixed validation trade-offs and matched-recall descriptive interpolation."""
import os
os.environ['MPLCONFIGDIR']='/tmp/probe-matplotlib'
import json,gzip,hashlib,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent;O=R/'results';P=R.parent;S=P/'api_results_20261015'
C=['baseline','saliency','nearest'];A=['baseline','nearest','saliency']
DS=['toxigen','toxicchat','beavertails','aegis'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];API=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra']
COL={'baseline':'#64748b','saliency':'#d97706','nearest':'#1676c5'};N={'baseline':'No explanation','saliency':'Saliency','nearest':'1 NN'}
def read(p):
    with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(P/'anchor_transfer_20261018/results/COMPLETED.json')
for path in [S/'frozen/design.json.gz',S/'results/scores.npz']:assert sha(path)==receipt['inputs'][str(path)]
d=read(S/'frozen/design.json.gz');cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];z=np.load(S/'results/scores.npz')
valid={m:(z[m]>=0).all(axis=1) for m in API};common=np.logical_and.reduce(list(valid.values()));groups=collections.defaultdict(list)
for i,r in enumerate(cases):
    if r['dataset'] in DS:groups[r['backbone'],r['dataset'],r['split']].append(i)
old={(r['scope'],r['model'],r['backbone'],r['dataset'],r['condition']):r for r in read(O/'cells.json') if r['anchor']=='fixed50'}
betas=np.logspace(-1,1,41);betas[20]=1.;cuts=np.arange(102)
rows=[];ceilings=[];replays=0;agg=collections.defaultdict(list)
for scope,models in [('local_full',LOCAL),('api_matched',API),('common_valid',LOCAL+API)]:
    for m in models:
        mask=common if scope=='common_valid' else valid[m] if scope=='api_matched' else np.ones(len(cases),bool)
        for b in d['samples']:
            for ds in DS:
                splitdata={}
                for split in ['val','test']:
                    ix=np.array([i for i in groups[b,ds,split] if mask[i]]);rr=[cases[i] for i in ix]
                    y=np.array([r['true_label'] for r in rr]);w=np.array([1/r['inclusion_probability'] for r in rr]);gate=np.array([r['probe_probability']>=.5 for r in rr])
                    pop=d['thresholds'][b][ds]['populations'][split] if scope=='local_full' else dict(n=w.sum(),positive=w@y)
                    assert len(ix) and (z[m][ix]>=0).all()
                    yes=gate[:,None,None]&(z[m][ix,:,None]>=cuts[None,None,:])
                    tp=np.sum(yes*(w*y)[:,None,None],axis=0);fp=np.sum(yes*(w*(1-y))[:,None,None],axis=0)
                    prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0);rec=tp/pop['positive'];f1=np.divide(2*tp,pop['positive']+tp+fp,out=np.zeros_like(tp),where=pop['positive']+tp+fp>0)
                    acc=(tp+pop['n']-pop['positive']-fp)/pop['n']
                    splitdata[split]=dict(tp=tp,fp=fp,precision=prec,recall=rec,f1=f1,accuracy=acc,pop=pop,n=len(ix),referred=int(gate.sum()))
                v=splitdata['val'];t=splitdata['test'];base=dict(scope=scope,model=m,backbone=b,dataset=ds,probe_threshold=.5,validation_n=v['n'],test_n=t['n'],test_referred=t['referred'])
                assert np.ptp(t['recall'][:,0])<1e-12
                ceilings.append(dict(**base,recall_ceiling=float(t['recall'][0,0])))
                for ci,c in enumerate(C):
                    for metric in ['precision','recall','f1','accuracy']:
                        assert abs(t[metric][ci,50]-old[scope,m,b,ds,c][metric])<1e-10
                    replays+=1
                for ai,a in enumerate(C):
                    for bi,beta in enumerate(betas):
                        den=beta**2*v['pop']['positive']+v['tp'][ai]+v['fp'][ai]
                        objective=np.divide((1+beta**2)*v['tp'][ai],den,out=np.zeros_like(den),where=den>0)
                        cut=int(np.flatnonzero(np.abs(objective-objective.max())<=1e-12)[-1])
                        for ci,c in enumerate(C):
                            r=dict(**base,anchor=a,condition=c,beta_index=bi,beta=float(beta),judge_threshold=cut,**{k:float(t[k][ci,cut]) for k in ['precision','recall','f1','accuracy']})
                            rows.append(r)
                            sc='common_valid' if scope=='common_valid' else 'combined'
                            for anchor in [a,'average']:
                                agg[sc,anchor,c,bi].append([r['recall']*100,r['precision']*100,r['f1']*100])
assert len(rows)==147600 and replays==1200
with gzip.open(O/'fixed_probe50_tradeoff_cells.json.gz','wt') as f:json.dump(rows,f,separators=(',',':'))
means={k:np.mean(rr,axis=0) for k,rr in agg.items()}
points=[dict(scope=k[0],anchor=k[1],condition=k[2],beta_index=k[3],beta=float(betas[k[3]]),recall=x[0],precision=x[1],mean_cell_f1=x[2]) for k,x in means.items()]
(O/'fixed_probe50_tradeoff_aggregates.json').write_text(json.dumps(points,indent=2))
original=read(O/'tradeoff_aggregates.json')
joint={(r['scope'],r['anchor'],r['condition'],r['beta_index']):np.array([r['recall'],r['precision'],r['mean_cell_f1']]) for r in original if r['dataset']=='ALL'}
def line(data,scope,anchor,c):return np.array([data[scope,anchor,c,i] for i in range(41)])
def interp(xy,recall):
    # Only interpolate along monotonic observed paths: never select a test envelope.
    assert np.all(np.diff(xy[:,0])>=-1e-10)
    candidates=[]
    for left,right in zip(xy,xy[1:]):
        x0,y0=left[:2];x1,y1=right[:2]
        if abs(x1-x0)<1e-10:
            if abs(recall-x0)<1e-10:candidates.extend([y0,y1])
        elif x0<=recall<=x1:
            candidates.append(y0+(y1-y0)*(recall-x0)/(x1-x0))
    # A vertical segment has multiple precision values at exactly the same
    # recall: leave it unreported rather than selecting the best test value.
    if not candidates or max(candidates)-min(candidates)>1e-8:return None
    return float(np.mean(candidates))
comparisons=[]
for policy,data in [('joint',joint),('probe_fixed50',means)]:
    for scope in ['combined','common_valid']:
        for target in [60,65,70,75,80,85,90]:
            p={c:interp(line(data,scope,'average',c),target) for c in C}
            if any(x is None for x in p.values()):continue
            comparisons.append(dict(policy=policy,scope=scope,recall=target,**p,nn_minus_control=p['nearest']-p['baseline'],nn_minus_saliency=p['nearest']-p['saliency']))
(O/'matched_recall_precision.json').write_text(json.dumps(comparisons,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
def draw(ax,data,scope='combined',anchor='average'):
    for c in C:
        xy=line(data,scope,anchor,c);ax.plot(xy[:,0],xy[:,1],color=COL[c],lw=2.8,marker='.',ms=5)
        ax.scatter(*xy[20,:2],marker='*',s=230,color=COL[c],edgecolors='white',lw=.9,zorder=8)
    ax.set_xlabel('Mean test recall (%)');ax.set_ylabel('Mean test precision (%)')
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(O/(name+'.'+ext),dpi=190,facecolor='white')
    plt.close(fig)
legend=[Line2D([0],[0],color=COL[c],lw=3,label=N[c]) for c in C]
fig,ax=plt.subplots(figsize=(10.8,7.5));fig.subplots_adjust(left=.10,right=.97,top=.83,bottom=.23)
draw(ax,joint);ax.set_xlim(60,90);ax.set_ylim(71,96)
for rec in [70,80,85]:
    pr={c:interp(line(joint,'combined','average',c),rec) for c in C}
    ax.plot([rec,rec],[pr['baseline'],pr['nearest']],color=COL['nearest'],lw=2.3)
    ax.scatter([rec,rec],[pr['baseline'],pr['nearest']],s=28,color=[COL['baseline'],COL['nearest']],zorder=7)
    ax.annotate(f"+{pr['nearest']-pr['baseline']:.2f} pp",xy=(rec,pr['nearest']),xytext=(0,15),textcoords='offset points',ha='center',fontsize=11,color=COL['nearest'],bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=2))
fig.suptitle('Middle of the average precision–recall curve',fontsize=18,y=.96)
fig.text(.5,.90,'Four toxicity/content-safety datasets • 200 combinations • mean across three tuning sources',ha='center',fontsize=11)
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.12),ncol=3,frameon=False)
fig.text(.5,.089,'Blue annotations: approximate NN precision gain over control at the same mean recall.',ha='center',fontsize=10)
fig.text(.5,.058,'Linear interpolation between observed points; no smoothing or test-based envelope selection.',ha='center',fontsize=10)
fig.text(.5,.027,'Matched-recall comparisons can use different β values; only same-β comparisons share thresholds.',ha='center',fontsize=10)
save(fig,'precision_recall_average_zoom')

ceiling=np.mean([r['recall_ceiling'] for r in ceilings if r['scope']!='common_valid'])*100
fig,axs=plt.subplots(1,2,figsize=(14.5,7));fig.subplots_adjust(left=.07,right=.98,top=.81,bottom=.26,wspace=.25)
draw(axs[0],means);axs[0].set_title('Full plotted range',fontsize=14)
axs[0].axvline(ceiling,ls=':',color='#8a94a2',lw=1.5)
draw(axs[1],means);axs[1].set_xlim(55,min(ceiling+1,100))
yy=[p[1] for c in C for p in line(means,'combined','average',c) if p[0]>=55]
axs[1].set_ylim(min(yy)-2,min(100,max(yy)+2));axs[1].set_title('Closer view of the higher-recall region',fontsize=14)
fig.suptitle('Probe fixed at 50%: tune only the judge threshold',fontsize=18,y=.96)
fig.text(.5,.90,'Mean across control-, NN- and saliency-selected validation thresholds; all three conditions share each selected cutoff',ha='center',fontsize=10.5)
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.14),ncol=3,frameon=False)
fig.text(.5,.11,f'Stars: validation-F1-selected judge thresholds. Mean recall ceiling imposed by the fixed probe: {ceiling:.2f}%.',ha='center',fontsize=10)
fig.text(.5,.078,'The probe referral set is constant; only the judge acceptance threshold changes. No new inference.',ha='center',fontsize=10)
fig.text(.5,.046,'Original local/API scopes and weights; zero judge cutoffs retained. These curves are descriptive means.',ha='center',fontsize=10)
save(fig,'precision_recall_probe_fixed50')

fig,axs=plt.subplots(1,3,figsize=(16,6),sharex=True,sharey=True);fig.subplots_adjust(left=.065,right=.985,top=.78,bottom=.25,wspace=.20)
for ax,a in zip(axs,A):draw(ax,means,anchor=a);ax.set_title('Judge cutoff selected using\n'+N[a],fontsize=12)
fig.suptitle('Probe fixed at 50%: separate validation tuning sources',fontsize=18,y=.95)
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False)
fig.text(.5,.065,'At each objective, all three conditions receive the same judge cutoff. Stars mark validation-F1 choices.',ha='center',fontsize=10)
fig.text(.5,.030,'200 combinations across four datasets; original local/API scopes; no test selection or smoothing.',ha='center',fontsize=10)
save(fig,'precision_recall_probe_fixed50_sources')
method=dict(probe_threshold=.5,judge_grid='Integers 0..101; >= decision; 101 is reject-all',selection='Select highest judge cutoff within 1e-12 of maximum weighted validation F-beta, for beta 0.1..10. Repeat for each selection source; transfer same cutoff to all conditions.',interpolation='Linear interpolation of mean test precision at common mean test recall, along monotonically increasing observed paths. No test-based optimisation. Different conditions can use different beta positions at common recall; this is descriptive, not the paired same-threshold explanation effect.',recall_ceiling=ceiling,fixed50_replays=replays,new_model_calls=0)
(O/'FIXED_PROBE_METHOD.json').write_text(json.dumps(method,indent=2))
tab='<table><tr><th>Recall</th><th>Control precision</th><th>Saliency precision</th><th>NN precision</th><th>NN − control</th><th>NN − saliency</th></tr>'
for r in comparisons:
    if r['policy']=='joint' and r['scope']=='combined':tab+='<tr>'+''.join(f'<td>{r[k]:.2f}</td>' for k in ['recall','baseline','saliency','nearest','nn_minus_control','nn_minus_saliency'])+'</tr>'
tab+='</table>'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Zoomed and fixed-probe precision recall</title><style>body{font:17px/1.5 system-ui;max-width:1350px;margin:30px auto;padding:0 24px;color:#172335}img{width:100%}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ccd4dc;text-align:right}</style><h1>Closer precision–recall comparisons</h1><p>The first plot enlarges the existing joint-tuning average. Both thresholds are matched across conditions within each validation objective. The second holds the probe cutoff at 50% and varies only validation-selected judge cutoffs, again repeating the three selection sources.</p><img src="precision_recall_average_zoom.png" alt="Zoomed mean precision recall curves"><h2>Approximate precision at matched mean recall</h2>'''+tab+'''<p>Interpolated descriptive coordinates, not exact attainable common-recall policies or confidence intervals. These comparisons use different positions on the curves; matched thresholds apply at the same validation objective, not necessarily at matched recall.</p><img src="precision_recall_probe_fixed50.png" alt="Probe fixed at 50 percent precision recall curves"><img src="precision_recall_probe_fixed50_sources.png" alt="Three validation tuning sources with fixed probe"><p><a href="precision_recall_average_zoom.pdf">Zoom PDF</a> · <a href="precision_recall_probe_fixed50.pdf">Fixed-probe PDF</a> · <a href="matched_recall_precision.json">Matched-recall values</a> · <a href="fixed_probe50_tradeoff_cells.json.gz">Per-cell selections and outcomes</a> · <a href="FIXED_PROBE_METHOD.json">Methods</a></p></html>'''
(O/'zoom_and_fixed_probe.html').write_text(page)
(O/'FIXED_PROBE_VERIFIED.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in [S/'frozen/design.json.gz',S/'results/scores.npz',O/'tradeoff_aggregates.json',Path(__file__)]},fixed50_replays=replays,rows=len(rows),recall_ceiling=ceiling,new_model_calls=0),indent=2))
print('CEILING',ceiling)
print('MATCHED',json.dumps([r for r in comparisons if r['scope']=='combined']))
print('FIXED_PROBE_F1',json.dumps({c:means['combined','average',c,20].tolist() for c in C}))
