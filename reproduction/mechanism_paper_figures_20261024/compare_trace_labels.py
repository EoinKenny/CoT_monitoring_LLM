import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';P=R.parent
c=pd.read_json(P/'nn_rationale_mechanisms_20261020/results/coded_trace_metadata.jsonl.gz',lines=True)
f=pd.read_pickle(P/'nn_contrast_extremity_20261024/results/joined.pkl')
keys=['judge','backbone','sample_id'];assert not c.duplicated(keys).any()
m=c.merge(f[keys+['similarity','contrast','w']],on=keys,suffixes=('_historical','_current'),validate='one_to_one',how='left',indicator=True);assert m['_merge'].eq('both').all()
rows=[];raw=[];tables={}
for scope,mask in [('Contrast census',m.requested_census),('Contrast census, answer retained',m.requested_census&m.retains_original_nn),('All coded (harm-enriched)',np.ones(len(m),bool))]:
 for source in ['current','historical']:
  for feature in ['similarity','contrast']:
   g=m.loc[mask].copy();v=g[feature+'_'+source];known=v.isin([0,1]) if source=='current' else v.isin(['yes','no']);ok=known&g.prose_assessable&g.relation.isin(['S','C','B','N']);g=g.loc[ok].copy();orig=g[feature+'_'+source];g['original']=orig.eq(1) if source=='current' else orig.eq('yes');g['stated']=g.relation.isin(['S','B'] if feature=='similarity' else ['C','B'])
   for a in [0,1]:
    for b in [0,1]:g[f'p{a}{b}']=g.w*(g.original.eq(bool(a))&g.stated.eq(bool(b)))
   cols=['p00','p01','p10','p11'];cells=g.groupby(['judge','dataset','backbone'])[['w']+cols].sum();rates=cells[cols].div(cells.w,axis=0);joint=rates.groupby(['judge','dataset']).mean().groupby('judge').mean().mean().to_numpy().reshape(2,2)
   den=joint.sum(axis=1);yes=np.divide(joint[:,1],den,out=np.full(2,np.nan),where=den>0);phi_den=np.sqrt(np.prod(joint.sum(axis=0))*np.prod(den));phi=(joint[0,0]*joint[1,1]-joint[0,1]*joint[1,0])/phi_den if phi_den>0 else np.nan
   rows.append(dict(scope=scope,labels=source,feature=feature,eligible=len(g),excluded=int(np.asarray(mask).sum()-len(g)),original_no=int((~g.original).sum()),original_yes=int(g.original.sum()),stated_if_no=100*yes[0],stated_if_yes=100*yes[1],difference_pp=100*(yes[1]-yes[0]),phi=phi))
   if scope=='Contrast census' and source=='current':tables[feature]=(100*np.divide(joint,den[:,None],out=np.full_like(joint,np.nan),where=den[:,None]>0),pd.crosstab(g.original,g.stated).reindex(index=[False,True],columns=[False,True],fill_value=0).to_numpy())
res=pd.DataFrame(rows);res.to_json(O/'trace_label_association.json',orient='records',indent=2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(10,5))
for i,(feature,title) in enumerate([('similarity','Similarity'),('contrast','Contrast')]):
 t,n=tables[feature];ax[i].imshow(t,vmin=0,vmax=100,cmap='Blues')
 for a in [0,1]:
  for b in [0,1]:ax[i].text(b,a,f'{t[a,b]:.1f}%\n(n={n[a,b]:,})',ha='center',va='center',color='white' if t[a,b]>55 else '#243547')
 ax[i].set_xticks([0,1]);ax[i].set_xticklabels(['Not stated','Stated']);ax[i].set_yticks([0,1]);ax[i].set_yticklabels(['Label: no','Label: yes']);ax[i].set_xlabel('Written explanation');ax[i].set_title(title)
ax[0].set_ylabel('Original pair label (independent consensus)');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'trace_label_agreement.{ext}',dpi=190,bbox_inches='tight')
html=f'''<!doctype html><meta charset="utf-8"><title>Pair labels versus written explanations</title><style>body{{font:17px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 25px;color:#263547}}img{{width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}</style><h1>Do original labels correspond to what judges say?</h1><p>Probability-first → explanation traces only. All 17,533 coded traces matched uniquely to original paired scores and independent pair labels. Primary analysis: the 12,219 historical contrast-census traces, restricted to assessable prose and resolved original labels. Six local judges only.</p><img src="trace_label_agreement.png"><p>Row percentages answer: given an original yes/no label, how often is that relation stated in the explanation? Percentages use macro-standardized joint distributions; n values are raw counts and therefore do not reproduce weighted percentages. Not stating a relation is not proof that the model did not use it. <a href="trace_label_agreement.pdf">PDF</a></p><h2>Matched associations and sensitivity checks</h2>{res.to_html(index=False,float_format=lambda x:f'{x:.3f}')}<p>Current = independent leave-one-out consensus used in Experiment 3. Historical = earlier labels that selected the coding corpus. Every contrast-census trace has historical contrast=yes, so a historical contrast correlation within that subset is undefined, not zero. The broader coded corpus includes an extra harmful-flip-enriched sample and cannot estimate whole-study prevalence without accounting for that selection. Answer-retained rows preserve the original probability-only NN classification after requesting an explanation.</p><p>Similarity stated = S or B; contrast stated = C or B. Unassessable prose and U codes excluded, not counted as absent. Four joint binary proportions are design-weighted within judge/backbone/dataset and averaged equally over available backbones, datasets and judges; conditional rates and phi are then derived from that macro joint distribution. Phi is descriptive binary correlation, not causal evidence. No p-values or independent-row standard errors are used because targets repeat. These labels describe different objects: pair properties versus explicitly expressed comparisons.</p>'''
(O/'trace_label_agreement.html').write_text(html)
print(res.to_string(index=False))
