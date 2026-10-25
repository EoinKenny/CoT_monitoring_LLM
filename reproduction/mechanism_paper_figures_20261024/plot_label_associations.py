import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';src=R.parent/'nn_contrast_extremity_20261024/results/joined.pkl';d=pd.read_pickle(src).reset_index(drop=True)
cl,ids=pd.factorize(pd.MultiIndex.from_frame(d[['dataset','sample_id']]),sort=True);rng=np.random.default_rng(20261024);B=1000;mult=np.zeros((B,len(ids)))
for _,g in d.groupby('dataset'):
 u=np.unique(cl[g.index]);mult[:,u]=rng.multinomial(len(u),np.full(len(u),1/len(u)),size=B)
mult=np.vstack([np.ones((1,len(ids))),mult])
ci,cells=pd.factorize(pd.MultiIndex.from_frame(d[['judge','dataset','backbone']]),sort=True);C=len(cells);cf=cells.to_frame(index=False);cf.columns=['judge','dataset','backbone'];jd,jds=pd.factorize(pd.MultiIndex.from_frame(cf[['judge','dataset']]),sort=True);jf=jds.to_frame(index=False);jf.columns=['judge','dataset'];ji,judges=pd.factorize(jf.judge,sort=True);M1=np.eye(len(jds))[jd];M2=np.eye(len(judges))[ji];w=d.w.to_numpy()
def avg(v,M):
 valid=np.isfinite(v);num=np.nan_to_num(v)@M;den=valid.astype(float)@M
 return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
def estimate(mask,val):
 wk=w*np.asarray(mask);v=np.asarray(val,float);mat=sparse.csr_matrix((np.r_[wk,wk*v],(np.tile(cl,2),np.r_[ci,ci+C])),shape=(len(ids),2*C));mat.eliminate_zeros();z=(mat.T@mult.T).T
 rates=np.divide(z[:,C:],z[:,:C],out=np.full_like(z[:,C:],np.nan),where=z[:,:C]>0);byjudge=100*avg(avg(rates,M1),M2);a=np.nanmean(byjudge,axis=1)
 return a,byjudge[0]
def entry(a):return dict(mean=float(a[0]),lo=float(np.nanpercentile(a[1:],2.5)),hi=float(np.nanpercentile(a[1:],97.5)))
prev=[];dots=[]
for label in ['similarity','contrast']:
 for outcome in ['unchanged','helped','harmed']:
  mask=d[label].isin([0,1])&d[outcome].astype(bool);a,j=estimate(mask,d[label].eq(1));prev.append(dict(label=label,outcome=outcome,rows=int(mask.sum()),**entry(a)))
  for name,v in zip(judges,j):dots.append(dict(label=label,outcome=outcome,judge=name,rate=float(v)))
p=pd.DataFrame(prev);q=pd.DataFrame(dots);old=pd.read_csv(O/'label_overall_rates.csv')
for row in prev:
 label={'similarity':'labelled_similarity','contrast':'important_contrast'}[row['label']];out={'helped':'helpful','harmed':'harmful','unchanged':'unchanged'}[row['outcome']];ref=old[(old.label==label)&(old.outcome==out)].rate.iloc[0];assert np.isclose(row['mean'],ref,atol=1e-8)
known=d.similarity.isin([0,1])&d.contrast.isin([0,1]);comb=[];cdots=[]
for title,s,c in [('Neither',0,0),('Similar only',1,0),('Contrasting only',0,1),('Both',1,1)]:
 mask=known&d.similarity.eq(s)&d.contrast.eq(c);h,_=estimate(mask,d.helped);a,_=estimate(mask,d.harmed);net,j=estimate(mask,d.helped.astype(int)-d.harmed.astype(int));assert np.allclose(net,h-a,equal_nan=True)
 comb.append(dict(group=title,rows=int(mask.sum()),targets=int(d.loc[mask,'sample_id'].nunique()),corrects=float(h[0]),misleads=float(a[0]),unchanged=float(100-h[0]-a[0]),**entry(net)))
 cdots.extend(dict(group=title,judge=name,rate=float(v)) for name,v in zip(judges,j))
co=pd.DataFrame(comb);cq=pd.DataFrame(cdots);assert co.rows.sum()==known.sum()
p.to_json(O/'association_prevalence.json',orient='records',indent=2);co.to_json(O/'association_combinations.json',orient='records',indent=2)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(10,5))
colors=['#84909c','#258466','#c66d66']
for i,(lab,title) in enumerate([('similarity','Similar NN'),('contrast','Contrasting NN')]):
 for x,out in enumerate(['unchanged','helped','harmed']):
  row=p[(p.label==lab)&(p.outcome==out)].iloc[0];vals=q[(q.label==lab)&(q.outcome==out)].rate.to_numpy();ax[i].scatter(x+np.linspace(-.18,.18,len(vals)),vals,color=colors[x],alpha=.35,s=25)
  ax[i].vlines(x,row.lo,row.hi,color=colors[x],lw=2);ax[i].plot([x-.18,x+.18],[row['mean']]*2,color=colors[x],lw=4);ax[i].text(x,row.hi+3,f'{row["mean"]:.1f}%',ha='center',fontsize=10)
 ax[i].set_title(title);ax[i].set_xticks(range(3));ax[i].set_xticklabels(['Unchanged','Corrects\njudge','Misleads\njudge']);ax[i].set_ylim(0,105);ax[i].set_box_aspect(1)
ax[0].set_ylabel('Mean label prevalence within outcome (%)');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'label_association_prevalence.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for x,row in co.iterrows():
 vals=cq[cq.group==row.group].rate.to_numpy();ax.scatter(x+np.linspace(-.17,.17,len(vals)),vals,color='#5a72a1',alpha=.35,s=28);ax.vlines(x,row.lo,row.hi,color='#5a72a1',lw=2)
 ax.plot([x-.2,x+.2],[row['mean']]*2,color='#5a72a1',lw=4);ax.text(x,row.hi+1,f'{row["mean"]:+.1f}',ha='center')
ax.axhline(0,color='#777',ls='--',lw=1);ax.set_xticks(range(4));ax.set_xticklabels(['Neither','Similar\nonly','Contrasting\nonly','Both']);ax.set_ylabel('Net accuracy change (pp)');ax.set_ylim(min(-5,cq.rate.min()-3,co.lo.min()-3),max(cq.rate.max()+5,co.hi.max()+5));fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'label_association_combinations.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
html=f'''<!doctype html><meta charset="utf-8"><title>Mechanism labels and performance</title><style>body{{font:17px/1.5 system-ui;max-width:1150px;margin:35px auto;padding:0 25px;color:#263547}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style><h1>Mechanism labels and judge performance</h1><p>All ten datasets and twelve outcome judges; full original probability range. Valid paired control/1-NN test outcomes, fixed-50 judge decisions, no probe-referral gate. Labels are independent majority annotations excluding the outcome judge. These plots do not use the exploratory 50–70 filter.</p><h2>1. Which labels accompany each outcome?</h2><img src="label_association_prevalence.png"><p>Each faded dot is one judge averaged over backbones and datasets. Thick bars show the overall mean; vertical intervals are 95% target-cluster bootstrap intervals. The two labels can overlap. Binary known labels only, separately for each feature.</p>{p.to_html(index=False,float_format=lambda v:f'{v:.2f}')}<p><a href="label_association_prevalence.pdf">PDF</a></p><h2>2. What is the net benefit within each label combination?</h2><img src="label_association_combinations.png"><p>Net gain = percentage corrected minus percentage misled. Unchanged cases remain in the denominator. Only rows with both labels resolved as yes/no enter this comparison; uncertain or missing labels are excluded explicitly. Each faded dot is one judge.</p>{co.to_html(index=False,float_format=lambda v:f'{v:.2f}')}<p>{int(known.sum()):,} paired rows have both labels resolved; {int((~known).sum()):,} of {len(d):,} paired rows are excluded from panel 2 for unresolved labels. The four groups partition resolved rows; target IDs repeat across judges/backbones and can appear in multiple groups.</p><p><a href="label_association_combinations.pdf">PDF</a></p><p>Design-weighted within judge/backbone/dataset and the specified conditioning group; equal averaging over available backbones, then datasets, then judges. Undefined groups omitted. 1,000 resamples of target IDs within dataset keep repeated outcomes across judges/backbones together. Intervals are pointwise, with annotations treated as fixed. Groups may differ in dataset, class and judge composition: these associations are not causal mechanism estimates. The two views have different denominators and therefore answer different questions.</p>'''
(O/'label_associations.html').write_text(html)
(O/'LABEL_ASSOCIATIONS_VERIFIED.json').write_text(json.dumps(dict(rows=len(d),both_known=int(known.sum()),excluded=int((~known).sum()),previous_prevalence_verified=True,combination_partition_verified=True,bootstrap_replicates=B,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),new_model_calls=0),indent=2))
print(p.to_string(index=False));print(co.to_string(index=False))
