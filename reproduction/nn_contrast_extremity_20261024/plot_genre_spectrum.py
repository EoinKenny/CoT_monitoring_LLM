import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';B=1000;SEED=20261024;RADIUS=15.
full=pd.read_pickle(O/'joined.pkl');full=full[full.contrast==1].copy().reset_index(drop=True)
assert full.control.between(0,100).all() and full.nn.between(0,100).all()
assert (full.helped.astype(int)+full.harmed.astype(int)+full.unchanged.astype(int)).eq(1).all()
cluster,cids=pd.factorize(full.sample_id,sort=True);full['cluster']=cluster
rng=np.random.default_rng(SEED);mult=np.zeros((B,len(cids)))
for ds,v in full.groupby('dataset'):
 ids=np.unique(cluster[v.index]);mult[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B)
mult=np.vstack([np.ones((1,len(cids))),mult])
sd=float(full.control.std(ddof=1));low=max(0,50-sd);high=min(100,50+sd)
genres={'Overall':sorted(full.dataset.unique()),'Toxicity / harmful content':['aegis','beavertails','toxicchat','toxigen'],'Safety':['wildguard','wildjailbreak'],'Hallucination':['fava','halueval','ragtruth'],'High stakes':['high_stakes']}
assert set(sum(list(genres.values())[1:],[]))==set(full.dataset.unique())
scopes=[(name,name,full.control.between(50,70)&full.dataset.isin(ds)) for name,ds in genres.items()]
old=pd.read_csv(O/'summary.csv');results={};audit={};allbins=[];allcoarse=[];boot={}
for key,title,mask in scopes:
 g=full.loc[mask].copy().reset_index(drop=True);x=100*g.neighbor_probe_probability.to_numpy();w=g.w.to_numpy();h=g.helped.to_numpy(float);a=g.harmed.to_numpy(float);cl=g.cluster.to_numpy()
 ci,cells=pd.factorize(pd.MultiIndex.from_frame(g[['judge','backbone','dataset']]),sort=True);C=len(cells);cf=cells.to_frame(index=False);cf.columns=['judge','backbone','dataset'];jd,jds=pd.factorize(pd.MultiIndex.from_frame(cf[['judge','dataset']]),sort=True);jf=jds.to_frame(index=False);jf.columns=['judge','dataset'];ji,judges=pd.factorize(jf.judge,sort=True);M1=np.eye(len(jds))[jd];M2=np.eye(len(judges))[ji]
 def avg(v,M):
  valid=np.isfinite(v);num=np.nan_to_num(v)@M;den=valid.astype(float)@M
  return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
 def macro(num,den):
  v=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
  return 100*np.nanmean(avg(avg(v,M1),M2),axis=1)
 def estimate(k):
  wk=w*k;mat=sparse.csr_matrix((np.r_[wk,wk*h,wk*a],(np.tile(cl,3),np.r_[ci,ci+C,ci+2*C])),shape=(len(cids),3*C));mat.eliminate_zeros()
  z=(mat.T@mult.T).T;den=z[:,:C]
  return macro(z[:,C:2*C],den),macro(z[:,2*C:],den)
 hh,aa=estimate(np.ones(len(g)));nn=hh-aa
 allcoarse.append(dict(scope=key,rows=len(g),targets=g.sample_id.nunique(),judges=g.judge.nunique(),datasets=g.dataset.nunique(),helped=float(hh[0]),harmed=float(aa[0]),net=float(nn[0]),net_lo=float(np.nanpercentile(nn[1:],2.5)),net_hi=float(np.nanpercentile(nn[1:],97.5))))
 for left in range(0,100,10):
  k=(x>=left)&((x<left+10) if left<90 else x<=100);hh,aa=estimate(k)
  row=dict(scope=key,left=left,right=left+10,helped=float(hh[0]),harmed=float(aa[0]),unchanged=float(100-hh[0]-aa[0]),net=float(hh[0]-aa[0]),rows=int(k.sum()),targets=int(len(np.unique(cl[k]))))
  for name,val in [('helped',hh),('harmed',aa),('net',hh-aa)]:
   assert np.isfinite(val[0])
   row[name+'_lo'],row[name+'_hi']=map(float,np.nanpercentile(val[1:],[2.5,97.5]))
  allbins.append(row)

b=pd.DataFrame(allbins);summary=pd.DataFrame(allcoarse)
b.to_json(O/'genre_spectrum_bins.json',orient='records',indent=2);summary.to_json(O/'genre_spectrum_summary.json',orient='records',indent=2)
old=pd.read_json(O/'selected_spectrum_bins_ci.json');tox=b[b.scope=='Toxicity / harmful content']
for col in ['helped','harmed','net','rows']:assert np.allclose(old[col],tox[col])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
rate_max=min(100,max(60,np.ceil(b[['helped_hi','harmed_hi']].max().max()/10)*10));net_low=min(-10,np.floor(b.net_lo.min()/10)*10);net_high=max(30,np.ceil(b.net_hi.max()/10)*10)
def draw(ax,bb):
 x=((bb.left+bb.right)/2).to_numpy()
 for k,label,color,off in [('helped','Corrects judge','#258466',-1.8),('harmed','Misleads judge','#c66d66',1.8)]:
  ax[0].plot(x,bb[k].to_numpy(),'-o',color=color,lw=2,label=label,ms=5)
  ax[0].vlines(x+off,bb[k+'_lo'].to_numpy(),bb[k+'_hi'].to_numpy(),color=color,alpha=.7,lw=1.1)
  for suf in ['_lo','_hi']:ax[0].plot(x+off,bb[k+suf].to_numpy(),linestyle='none',marker='_',color=color,ms=6)
 ax[1].plot(x,bb.net.to_numpy(),'-o',color='#5a72a1',lw=2,ms=5);ax[1].vlines(x,bb.net_lo.to_numpy(),bb.net_hi.to_numpy(),color='#5a72a1',alpha=.7,lw=1.1)
 for suf in ['_lo','_hi']:ax[1].plot(x,bb['net'+suf].to_numpy(),linestyle='none',marker='_',color='#5a72a1',ms=6)
 ax[1].axhline(0,color='#777',ls='--',lw=1);ax[0].set_ylim(0,rate_max);ax[1].set_ylim(net_low,net_high)
 ax[0].set_ylabel('Mean outcome rate (%)');ax[1].set_ylabel('Net accuracy change (pp)')
 for a in ax:a.set_xlim(0,100);a.set_xlabel('Neighbour probe probability (%)');a.set_box_aspect(.85)
 ax[0].legend(frameon=False,fontsize=10)
fig,axes=plt.subplots(5,2,figsize=(12,23))
parts=[]
for i,name in enumerate(genres):
 bb=b[b.scope==name];z=summary[summary.scope==name].iloc[0];title=f'{name} · {int(z.rows):,} paired judgments'
 draw(axes[i],bb);axes[i,0].set_title(title,loc='left')
 f,a=plt.subplots(1,2,figsize=(11,5));draw(a,bb);f.suptitle(title+'\nExploratory · Original judge probability 50–70',fontsize=14);f.tight_layout()
 for ext in ['png','pdf']:f.savefig(O/f'genre_spectrum_{i}.{ext}',dpi=175,bbox_inches='tight')
 plt.close(f);parts.append(f'<h2>{name}</h2><p>Datasets: {", ".join(genres[name])}.</p><img src="genre_spectrum_{i}.png"><p><a href="genre_spectrum_{i}.pdf">PDF</a></p>')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/f'genre_spectrum_all.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
html=f"""<!doctype html><meta charset='utf-8'><title>Contrasting NN by dataset genre</title><style>body{{font:17px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 25px;color:#263547}}img{{width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style><h1>Contrasting neighbours: overall and by dataset genre</h1><p>All ten datasets, followed by each genre. The original judge band stays at <b>50–70</b>; it was selected previously on the toxicity subset and is not re-optimized here. Independent Contrasting NN consensus, valid paired test outcomes, fixed-50 judge decisions, no probe-referral filter. These are exploratory descriptive results, not all NN cases or complete pipeline performance.</p><p>Ten-point non-overlapping bins [0,10), …, [90,100]. Rates include unchanged cases in their denominators. Design weights within judge/backbone/dataset, then equal averaging over available backbones, datasets and judges; overall does not equally weight the four genres. Same axes across genres. Error bars are 95% pointwise percentile intervals from 1,000 target-ID resamples within dataset, keeping repeated judge/backbone outcomes paired; they do not adjust for band selection. Sparse-bin composition varies.</p>{summary.to_html(index=False,float_format=lambda v:f'{v:.2f}')}<p><a href='genre_spectrum_all.pdf'>Combined five-row PDF</a></p>{''.join(parts)}<h2>Bin counts and intervals</h2>{b.to_html(index=False,float_format=lambda v:f'{v:.2f}')}"""
(O/'genre_spectrum.html').write_text(html)
(O/'GENRE_SPECTRUM_VERIFIED.json').write_text(json.dumps(dict(genres=genres,judge_band=[50,70],replicates=B,seed=SEED,toxicity_point_estimates_match=True,source_sha256=hashlib.sha256((O/'joined.pkl').read_bytes()).hexdigest(),new_model_calls=0),indent=2))
print(summary.to_string(index=False))
