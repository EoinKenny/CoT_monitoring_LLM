"""Recompute plotted statistics from bundled per-case scores and annotations."""
import collections,gzip,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

def macro(values,cells):
 """Equal backbones, then datasets, then judges; preserve missing-cell support."""
 frame=cells.to_frame(index=False);frame.columns=['judge','dataset','backbone']
 jd,keys=pd.factorize(pd.MultiIndex.from_frame(frame[['judge','dataset']]),sort=True)
 jf=keys.to_frame(index=False);jf.columns=['judge','dataset'];ji,judges=pd.factorize(jf.judge,sort=True)
 def avg(v,M):
  ok=np.isfinite(v);den=ok.astype(float)@M
  return np.divide(np.nan_to_num(v)@M,den,out=np.full((v.shape[0],M.shape[1]),np.nan),where=den>0)
 return np.nanmean(avg(avg(values,np.eye(len(keys))[jd]),np.eye(len(judges))[ji]),axis=1)

def bootstrap_weights(d,seed):
 cl,ids=pd.factorize(pd.MultiIndex.from_frame(d[['dataset','sample_id']]),sort=True)
 rng=np.random.default_rng(seed);multi=np.zeros((1000,len(ids)))
 for _,g in d.groupby('dataset'):
  u=np.unique(cl[g.index]);multi[:,u]=rng.multinomial(len(u),np.full(len(u),1/len(u)),size=1000)
 return cl,np.vstack([np.ones((1,len(ids))),multi])

def label_statistics(d,out):
 cl,multi=bootstrap_weights(d,20261024)
 ci,cells=pd.factorize(pd.MultiIndex.from_frame(d[['judge','dataset','backbone']]),sort=True);nc=len(cells)
 prev=[];cellrows=[]
 for label,long in [('similarity','labelled_similarity'),('contrast','important_contrast')]:
  for outcome,longout in [('unchanged','unchanged'),('helped','helpful'),('harmed','harmful')]:
   keep=d[label].isin([0,1])&d[outcome];w=d.w.to_numpy()*keep;val=d[label].eq(1)
   mat=sparse.csr_matrix((np.r_[w,w*val],(np.tile(cl,2),np.r_[ci,ci+nc])),shape=(multi.shape[1],2*nc));z=(mat.T@multi.T).T
   rates=np.divide(z[:,nc:],z[:,:nc],out=np.full_like(z[:,nc:],np.nan),where=z[:,:nc]>0)
   v=100*macro(rates,cells);lo,hi=np.nanpercentile(v[1:],[2.5,97.5])
   prev.append(dict(label=label,outcome=outcome,rows=int(keep.sum()),mean=v[0],lo=lo,hi=hi))
   for cell,rate in zip(cells,rates[0]):
    cellrows.append(dict(judge=cell[0],dataset=cell[1],backbone=cell[2],label=long,outcome=longout,rate=100*rate))
 pd.DataFrame(prev).to_json(out/'label_prevalence.json',orient='records',indent=2)
 pd.DataFrame(cellrows).to_csv(out/'label_cells.csv.gz',index=False)

def contrast_statistics(d,out):
 # Match the original bootstrap universe: all resolved contrast-yes cases before band selection.
 full=d[d.contrast==1].copy().reset_index(drop=True)
 cluster,cids=pd.factorize(full.sample_id,sort=True);full['cluster']=cluster
 rng=np.random.default_rng(20261024);multi=np.zeros((1000,len(cids)))
 for _,g in full.groupby('dataset'):
  ids=np.unique(cluster[g.index]);multi[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=1000)
 multi=np.vstack([np.ones((1,len(cids))),multi]);rows=[]
 genres={'Overall':sorted(full.dataset.unique()),'Toxicity / harmful content':['aegis','beavertails','toxicchat','toxigen'],'Safety':['wildguard','wildjailbreak'],'Hallucination':['fava','halueval','ragtruth'],'High stakes':['high_stakes']}
 for scope,ds in genres.items():
  g=full[full.control.between(50,70)&full.dataset.isin(ds)].reset_index(drop=True)
  ci,cells=pd.factorize(pd.MultiIndex.from_frame(g[['judge','dataset','backbone']]),sort=True);nc=len(cells)
  x=100*g.neighbor_probe_probability.to_numpy();cl=g.cluster.to_numpy()
  for left in range(0,100,10):
   k=(x>=left)&((x<left+10) if left<90 else x<=100);w=g.w.to_numpy()*k
   mat=sparse.csr_matrix((np.r_[w,w*g.helped,w*g.harmed],(np.tile(cl,3),np.r_[ci,ci+nc,ci+2*nc])),shape=(len(cids),3*nc));z=(mat.T@multi.T).T
   rates=[]
   for start in [nc,2*nc]:
    a=np.divide(z[:,start:start+nc],z[:,:nc],out=np.full_like(z[:,:nc],np.nan),where=z[:,:nc]>0);rates.append(100*macro(a,cells))
   h,a=rates;v=h-a;row=dict(scope=scope,left=left,right=left+10,helped=h[0],harmed=a[0],unchanged=100-h[0]-a[0],net=v[0],rows=int(k.sum()),targets=int(len(np.unique(cl[k]))))
   for name,value in [('helped',h),('harmed',a),('net',v)]:row[name+'_lo'],row[name+'_hi']=np.nanpercentile(value[1:],[2.5,97.5])
   rows.append(row)
 pd.DataFrame(rows).to_json(out/'contrast_bins.json',orient='records',indent=2)

def trace_statistics(d,out):
 rows=[];joints={}
 for f in ['similarity','contrast']:
  g=d[d.status.eq('valid')&d.trace_assessable.eq('yes')&d['trace_'+f].isin(['yes','no'])&d['original_'+f].isin([0,1])].copy()
  g['cat']=2*g['original_'+f]+g['trace_'+f].eq('yes').astype(int)
  q=g.groupby(['judge','dataset','backbone','cat']).weight.sum().unstack('cat').reindex(columns=range(4),fill_value=0).fillna(0)
  rate=q.div(q.sum(axis=1),axis=0);joint=rate.groupby(['judge','dataset']).mean().groupby('judge').mean().mean().to_numpy().reshape(2,2)
  raw=pd.crosstab(g['original_'+f],g['trace_'+f].eq('yes')).reindex(index=[0,1],columns=[False,True],fill_value=0).to_numpy()
  rows.append(dict(scope='All original flips',feature=f,n=len(g),raw_joint=raw.tolist()))
  joints['All original flips|'+f]=joint.tolist()
 pd.DataFrame(rows).to_json(out/'trace_associations.json',orient='records',indent=2)
 (out/'trace_joints.json').write_text(json.dumps(joints,indent=2))

def main_curves(results,out):
 frames=[]
 for policy,file,idx in [('fixed','fixed_probe_curves.csv.gz','judge_threshold'),('tuned','tradeoff_points.csv.gz','beta_index')]:
  d=pd.read_csv(results/file)
  for dim,key in [('overall',None),('model','model'),('dataset','dataset')]:
   keys=([key] if key else [])+['condition',idx]
   g=d.groupby(keys,as_index=False).agg(mean_test_recall=('recall','mean'),mean_test_precision=('precision','mean'),averaged_rows=('recall','size'))
   g=g.rename(columns={idx:'operating_index',key:'group'} if key else {idx:'operating_index'})
   if key is None:g['group']='ALL'
   g['dimension']=dim;g['policy']=policy;frames.append(g)
 curves=pd.concat(frames,ignore_index=True);curves.to_csv(out/'main_curves.csv.gz',index=False)
 groups=json.loads((out/'groups.json').read_text());records=[];cells=pd.read_csv(results/'cells.csv')
 for group in groups['group_figures']:
  kind,key=group['kind'],group['group'];sub=curves[(curves.dimension==('model' if kind=='judge' else 'dataset'))&curves.group.isin([key] if kind=='judge' else group['datasets'])]
  for (policy,condition,idx),rows in sub.groupby(['policy','condition','operating_index']):records.append(dict(kind=kind,group=key,policy=policy,condition=condition,operating_index=idx,recall=100*rows.mean_test_recall.mean(),precision=100*rows.mean_test_precision.mean()))
  cc=cells[cells.model.eq(key) if kind=='judge' else cells.dataset.isin(group['datasets'])]
  means=cc.groupby(['policy','condition','model','dataset']).f1.mean()
  for stat in group['stats']:
   v=100*means.loc[stat['policy'],stat['condition']];stat['mean_f1']=float(v.mean());stat['dots']=len(v)
 pd.DataFrame(records).to_csv(out/'group_curves.csv.gz',index=False);(out/'groups.json').write_text(json.dumps(groups,indent=2))

def ablation_statistics(analysis,out):
 design=json.load(gzip.open(analysis/'design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rr in design['samples'].items() for r in rr]
 c=pd.DataFrame(cases);scores=np.load(analysis/'ablation_scores.npz');conditions=['baseline','nearest','text_label','text_prediction','text_only','label_only','prediction_only','label_prediction','random_full'];rows=[]
 for model in scores.files:
  for (bb,ds),g in c[c.split=='test'].groupby(['backbone','dataset']):
   ii=g.index[(scores[model][g.index]>=0).all(axis=1)];a=scores[model][ii];g=c.loc[ii];w=1/g.inclusion_probability.to_numpy();y=g.true_label.to_numpy();ref=g.probe_probability.to_numpy()>=.5
   pop=design['thresholds'][bb][ds]['populations']['test'];local=model in ['qwen','gemma','mistral','olmo','glm','qwen122'];n=pop['n'] if local else w.sum();pos=pop['positive'] if local else w@y
   from paper.numerics import metrics
   for j,condition in enumerate(conditions):
    p=ref&(a[:,j]>=50);m=metrics(w[p&(y==1)].sum(),w[p&(y==0)].sum(),n,pos)
    rows.append(dict(model=model,backbone=bb,dataset=ds,condition=condition,n_cases=len(ii),**{k:m[k] for k in ['precision','recall','f1','accuracy']}))
 pd.DataFrame(rows).to_csv(out/'ablation_cells.csv.gz',index=False)

def recompute(data,out):
 import shutil
 out.mkdir(parents=True,exist_ok=True);plots=out/'plots';shutil.copytree(data/'plots',plots,dirs_exist_ok=True)
 results=out/'main'
 subprocess.run([sys.executable,'-m','paper.evaluate_main','--data',str(data/'analysis'),'--out',str(results),'--frontier'],check=True,cwd=Path(__file__).resolve().parents[1])
 pd.read_csv(results/'cells.csv').to_csv(plots/'main_cells.csv.gz',index=False)
 main_curves(results,plots);ablation_statistics(data/'analysis',plots)
 d=pd.read_csv(data/'analysis/paired_test.csv.gz');label_statistics(d,plots);contrast_statistics(d,plots)
 trace_statistics(pd.read_csv(data/'analysis/trace_pairs.csv.gz'),plots)
 return plots
