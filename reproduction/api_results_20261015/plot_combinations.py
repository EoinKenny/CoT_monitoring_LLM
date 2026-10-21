from pathlib import Path
import json, gzip, csv, hashlib, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'plots'; OUT.mkdir(exist_ok=True)
with gzip.open(ROOT/'frozen/design.json.gz','rt') as f: design=json.load(f)
cases=[dict(backbone=b,**r) for b in design['samples'] for r in design['samples'][b]]
scores=np.load(ROOT/'results/scores.npz')
common=np.logical_and.reduce([(scores[m]>=0).all(axis=1) for m in scores.files])
groups=collections.defaultdict(list)
for i,r in enumerate(cases):
    if common[i] and r['split']=='test': groups[r['backbone'],r['dataset']].append(i)
conditions=['baseline','saliency','nearest']
labels={'baseline':'No explanation','saliency':'Saliency','nearest':'1 nearest neighbour'}
palette={'baseline':'#64748b','saliency':'#d88b25','nearest':'#2989ac'}
policies={'fixed50':'Judge cutoff = 50','joint':'Jointly tuned cutoffs'}
rows=[]
for r in json.loads((ROOT/'results/matched_results.json').read_text()):
    if r['scope']!='common_valid_all_judges': continue
    ix=np.array(groups[r['backbone'],r['dataset']]); rr=[cases[i] for i in ix]
    y=np.array([v['true_label'] for v in rr]); w=np.array([1/v['inclusion_probability'] for v in rr])
    p=np.array([v['probe_probability'] for v in rr])
    pred=(p>=r['probe_threshold'])&(scores[r['model']][ix,conditions.index(r['condition'])]>=r['probability_threshold'])
    tp=w[(y==1)&pred].sum(); fp=w[(y==0)&pred].sum(); fn=w[(y==1)&~pred].sum()
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    assert abs(f1-r['f1'])<1e-12
    rows.append({k:r[k] for k in ['model','backbone','dataset','condition','policy','f1','precision','recall','test_n']}|{'accuracy':float(w[y==pred].sum()/w.sum())})
df=pd.DataFrame(rows)
pair=df.groupby(['model','backbone','condition','policy'],as_index=False)[['accuracy','f1']].mean()
assert len(pair)==300 and len(df)==3000
pair.to_csv(OUT/'combination_metrics.csv',index=False)
df.to_csv(OUT/'dataset_metrics.csv',index=False)
sns.set_theme(style='whitegrid',context='notebook',font_scale=1.08)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
foot='50 backbone–judge combinations per condition; each averages 10 datasets. Same valid cases across all 10 judges.\nDesign-weighted test metrics; thresholds selected for validation F1. Points are correlated, not independent replications.'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    plt.close(fig)
summary=[]
for metric in ['accuracy','f1']:
    title='Classification accuracy' if metric=='accuracy' else 'F1 score'
    # Keep scales equal between policies for an honest absolute comparison.
    lo=np.floor(pair[metric].min()*100/5)*5-2; hi=min(100,np.ceil(pair[metric].max()*100/5)*5+2)
    fig,axs=plt.subplots(1,2,figsize=(13,6.5),sharey=True)
    fig.subplots_adjust(bottom=.22,top=.80,wspace=.12)
    for ax,(policy,plabel) in zip(axs,policies.items()):
        sub=pair[pair.policy==policy].copy(); sub[metric]*=100
        sns.swarmplot(data=sub,x='condition',y=metric,hue='condition',order=conditions,hue_order=conditions,palette=palette,legend=False,size=4.5,ax=ax)
        for i,c in enumerate(conditions):
            mean=sub[sub.condition==c][metric].mean()
            ax.plot([i-.23,i+.23],[mean,mean],color='#111827',lw=2.7,zorder=8)
            ax.text(i,hi-1.4,f'{mean:.2f}%',ha='center',va='top',weight='bold',color=palette[c])
            summary.append(dict(metric=metric,policy=policy,condition=c,mean=mean))
        ax.set(title=plabel,xlabel='',ylabel=title+' (%)',ylim=(lo,hi))
        ax.set_xticks(range(3));ax.set_xticklabels([labels[c].replace('1 nearest neighbour','1NN') for c in conditions]);ax.grid(False,axis='x')
    axs[1].set_ylabel('')
    fig.suptitle(f'{title} across backbone–judge combinations',fontsize=19,x=.06,ha='left',y=.97,weight='bold')
    fig.text(.06,.895,'Each dot is one combination. Black bars and coloured numbers show condition means.',fontsize=11)
    fig.text(.06,.055,foot,fontsize=10,color='#536174')
    save(fig,metric+'_swarm')
    fig,axs=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    fig.subplots_adjust(bottom=.18,top=.86,hspace=.38,wspace=.13)
    bins=np.linspace(lo,hi,17)
    for row,(policy,plabel) in enumerate(policies.items()):
        for col,c in enumerate(conditions):
            ax=axs[row,col]; v=100*pair[(pair.policy==policy)&(pair.condition==c)][metric]
            ax.hist(v,bins=bins,color=palette[c],alpha=.85,edgecolor='white')
            ax.axvline(v.mean(),color='#111827',lw=1.7,ls='--')
            ax.set_title(labels[c]+f'\nMean {v.mean():.2f}%',fontsize=12)
            ax.set_xlabel(title+' (%)' if row else '')
            ax.set_ylabel(plabel+'\nNumber of combinations' if col==0 else '')
            ax.grid(False,axis='x')
    fig.suptitle(f'Distribution of {title.lower()}',fontsize=19,x=.06,ha='left',y=.97,weight='bold')
    fig.text(.06,.045,foot+'\nIdentical histogram bins and axes in every panel. Dashed line = mean.',fontsize=10,color='#536174')
    save(fig,metric+'_histograms')
    fig,axs=plt.subplots(1,2,figsize=(13,6.5),sharey=True)
    fig.subplots_adjust(bottom=.22,top=.80,wspace=.12)
    deltas=[]
    for ax,(policy,plabel) in zip(axs,policies.items()):
        sub=pair[pair.policy==policy].pivot(index=['model','backbone'],columns='condition',values=metric)*100
        delta=sub[['saliency','nearest']].subtract(sub.baseline,axis=0)
        for _,r in delta.iterrows(): ax.plot([0,1],r.values,color='#c8ced6',alpha=.6,lw=.7,zorder=1)
        for i,c in enumerate(['saliency','nearest']):
            v=delta[c].values;ax.scatter(np.full(50,i),v,color=palette[c],s=23,alpha=.7,zorder=2)
            ax.plot([i-.14,i+.14],[v.mean()]*2,color='#111827',lw=3,zorder=3)
            ax.text(i,.98,f'Mean {v.mean():+.2f} pp\n{sum(v>1e-10)}/50 improve',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=11,color=palette[c])
        ax.axhline(0,color='#606975',ls='--',lw=1);ax.set_xlim(-.4,1.4)
        ax.set_xticks([0,1]);ax.set_xticklabels(['Saliency','1NN']);ax.set_title(plabel)
        ax.set_ylabel('Change versus no explanation (percentage points)');ax.grid(False,axis='x')
        deltas.extend(delta.values.ravel())
    low=min(deltas);high=max(deltas);pad=max((high-low)*.25,1)
    axs[0].set_ylim(low-pad*.2,high+pad);axs[1].set_ylabel('')
    fig.suptitle(f'Paired changes in {title.lower()}',fontsize=19,x=.06,ha='left',y=.97,weight='bold')
    fig.text(.06,.895,'Each line joins the same backbone and judge. Above zero means improvement over no explanation.',fontsize=11)
    fig.text(.06,.055,foot,fontsize=10,color='#536174')
    save(fig,metric+'_paired')
(OUT/'means.json').write_text(json.dumps(summary,indent=2))
manifest={'scope':'common_valid_all_judges','test_backbone_cases':int(sum(common[i] and r['split']=='test' for i,r in enumerate(cases))), 'points_per_condition':50,'datasets_per_point':10,'accuracy':'sum(weight * correct) / sum(weight), within dataset; equal-dataset and equal-combination means','policy':'Existing validation-F1 thresholds, no refitting for accuracy','inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'frozen/design.json.gz',ROOT/'results/scores.npz',ROOT/'results/matched_results.json']}}
(OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2))
html='<!doctype html><meta charset="utf-8"><title>Explanation comparison plots</title><style>body{font:17px system-ui;max-width:1300px;margin:35px auto;padding:0 20px;color:#243044}img{width:100%}p{line-height:1.5}section{margin:40px 0}a{color:#216d8d}</style><h1>No explanation, saliency and nearest neighbour</h1><p>50 backbone–judge combinations, ten datasets each. All ten judges use the same valid cases within each backbone. Accuracy and F1 are distinct metrics; both use the existing validation-F1-selected thresholds. These are conditional matched-case results, not full-test API population estimates.</p>'
for metric in ['accuracy','f1']:
    for kind in ['swarm','histograms','paired']:
        n=metric+'_'+kind
        html+=f'<section><img src="{n}.png" alt="{n}"><p><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a> · <a href="{n}.png">PNG</a></p></section>'
html+='<p><a href="combination_metrics.csv">Plotted data</a> · <a href="plot_manifest.json">Methods and provenance</a></p>'
(OUT/'gallery.html').write_text(html)
print(json.dumps(summary,indent=2))
