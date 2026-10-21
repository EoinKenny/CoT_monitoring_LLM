import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import ast,json,gzip,hashlib,time,joblib,warnings,shutil
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import ConvergenceWarning
R=Path(__file__).resolve().parent;I=R/'inputs';O=R/'results';SEED=20261019
BB=['llama','qwen','gemma','llama70','qwen122'];DS=['aegis','beavertails','toxicchat','toxigen'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];POL=['fixed50','baseline_anchor','nearest_anchor','saliency_anchor']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def reserve():assert shutil.disk_usage(R).free>20*1024**3
d=read(I/'design.json.gz');scores=np.load(I/'scores.npz');f=pd.read_csv(I/'joined_features.csv.gz');f['source_index']=np.arange(len(f));f['fold']=f.cluster.map(lambda s:int(hashlib.sha256((str(SEED)+':'+s).encode()).hexdigest()[:12],16)%3)
assert not set(f[f.split=='val'].cluster)&set(f[f.split=='test'].cluster)
ths=pd.DataFrame(read(I/'thresholds.json'));ths=ths[ths.scope!='common_valid'];ths={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in ths.itertuples()}
# Reuse the original feature whitelist and exact cutoff/metric rules.
tree=ast.parse((I/'tabular_source.py').read_text());wanted=['features','threshold','metric'];defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(defs)==len(wanted)
savedBB,savedDS=BB,DS;BB=sorted(f.backbone.unique());DS=sorted(f.dataset.unique());exec(compile(ast.Module(body=defs,type_ignores=[]),str(I/'tabular_source.py'),'exec'));FEATURE_BB,FEATURE_DS=BB,DS;BB,DS=savedBB,savedDS
def tabular(g):
 global BB,DS
 oldb,oldd=BB,DS;BB,DS=FEATURE_BB,FEATURE_DS
 try:X,names=features(g)
 finally:BB,DS=oldb,oldd
 assert names==read(I/'features.json');return X
def cases(b):
 a=pd.DataFrame(read(R/b/'mapping.json'));g=f[(f.backbone==b)&f.dataset.isin(DS)].copy();g=g.merge(a[['sample_id','array_row','nearest_id','nearest_label','split','true_label']],on='sample_id',suffixes=('','_export'),validate='one_to_one')
 for c in ['nearest_id','nearest_label','split','true_label']:assert (g[c]==g[c+'_export']).all(),c
 assert len(g)==4000;return g.sort_values('array_row').reset_index(drop=True)
def pop(g,m):
 b,ds,sp=g.backbone.iloc[0],g.dataset.iloc[0],g.split.iloc[0]
 if m in LOCAL:a=d['thresholds'][b][ds]['populations'][sp];return a['n'],a['positive']
 return g.weight.sum(),float(g.weight@g.true_label)
def decisions(g,m,pol):
 pair=np.array([(.5,50) if pol=='fixed50' else ths[m,b,ds,pol.replace('_anchor','')] for b,ds in zip(g.backbone,g.dataset)]);ref=g.probe_probability.to_numpy()>=pair[:,0];ss=scores[m][g.source_index.to_numpy()];return ref,ref&(ss[:,0]>=pair[:,1]),ref&(ss[:,2]>=pair[:,1])
CS=[.001,.01,.1,1.,10.];KS=[151,51,15,5,3,1];DIMS=['raw',16,64,128];REPS=['raw_l2','probe_l2']
def geometry_name(s):return f"{s['rep']}__{s['dim']}__{s['form']}__{'combined' if s['meta'] else 'embedding'}"
def spec_name(s):return geometry_name(s)+'__'+s['algorithm']+'__'+str(s['parameter'])
def all_specs():
 out=[dict(rep='tabular',dim=0,form='tabular',meta=True,algorithm='logistic',parameter=c) for c in CS]
 for rep in REPS:
  for dim in DIMS:
   for form in ['pair','relation']:
    for meta in [False,True]:
     geom=dict(rep=rep,dim=dim,form=form,meta=meta)
     out += [dict(**geom,algorithm='logistic',parameter=c) for c in CS]
     # kNN on concatenated target-neighbour coordinates; nonlinear retrieval already models interactions.
     if form=='pair':out += [dict(**geom,algorithm='knn_'+met,parameter=k) for met in ['euclidean','cosine'] for k in KS]
  for form in ['target','neighbor']:out += [dict(rep=rep,dim=64,form=form,meta=False,algorithm='logistic',parameter=c) for c in CS]
 return out
SPECS=all_specs();SPEC={spec_name(s):s for s in SPECS}
def family(s,v):
 pair=s['form'] in ['pair','relation'];emb=pair and not s['meta'];comb=pair and s['meta'];log=s['algorithm']=='logistic'
 return {'tabular':s['rep']=='tabular','embedding_logistic':emb and log,'combined_logistic':comb and log,'embedding_knn':emb and not log,'combined_knn':comb and not log,'embedding':emb,'combined':comb,'selected':pair or s['rep']=='tabular','embedding_raw':emb and s['dim']=='raw','embedding_pca':emb and s['dim']!='raw','combined_raw':comb and s['dim']=='raw','combined_pca':comb and s['dim']!='raw','target_only':s['form']=='target','neighbor_only':s['form']=='neighbor'}[v]
PRIMARY_VARIANTS=['tabular','embedding_logistic','combined_logistic','embedding_knn','combined_knn','embedding','combined','selected','embedding_raw','embedding_pca','combined_raw','combined_pca','target_only','neighbor_only']
SECOND_VARIANTS=['tabular','embedding_logistic','combined_logistic','embedding_knn','combined_knn','embedding','combined','selected']
def matrix(b,s,fold,g,tab):
 if s['rep']=='tabular':return tab
 base=R/'representations'/b;tag=s['rep'] if s['dim']=='raw' else s['rep']+'__'+str(fold);rows=g.array_row.to_numpy();t=np.asarray(np.load(base/(tag+'__target.npy'),mmap_mode='r')[rows]);n=np.asarray(np.load(base/(tag+'__neighbor.npy'),mmap_mode='r')[rows])
 if s['dim']!='raw':t=t[:,:int(s['dim'])];n=n[:,:int(s['dim'])]
 if s['form']=='pair':parts=[t,n]
 elif s['form']=='relation':parts=[t,n,t*n,np.abs(t-n)]
 elif s['form']=='target':parts=[t]
 else:parts=[n]
 if s['meta']:parts.append(tab)
 X=np.concatenate(parts,axis=1);assert np.isfinite(X).all();return X
def fit_log(s,X,y,w,scaler=None):
 if len(np.unique(y))<2:return dict(constant=float(y[0]) if len(y) else .5)
 w=w/w.mean();scaler=scaler or StandardScaler().fit(X,sample_weight=w);xx=scaler.transform(X)
 solver='liblinear' if X.shape[1]>=1024 else 'lbfgs';extra={'dual':True,'intercept_scaling':100.} if solver=='liblinear' else {}
 for it in [4000,16000]:
  est=LogisticRegression(C=s['parameter'],solver=solver,max_iter=it,tol=1e-6,random_state=SEED,**extra)
  try:
   with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);est.fit(xx,y,sample_weight=w)
   return dict(estimator=est,scaler=scaler,iterations=int(est.n_iter_.max()),solver=solver,max_iter=it)
  except ConvergenceWarning:
   if it==16000:raise
   print('convergence extension',spec_name(s),flush=True)
def predict_log(mm,X):
 if 'constant' in mm:return np.full(len(X),mm['constant'])
 return mm['estimator'].predict_proba(mm['scaler'].transform(X))[:,1]
def fit_knn(X,y,w,scaler=None):
 if len(np.unique(y))<2:return dict(constant=float(y[0]) if len(y) else .5)
 scaler=scaler or StandardScaler().fit(X,sample_weight=w);return dict(scaler=scaler,X=scaler.transform(X),y=y,w=w)
def predict_knn(mm,X,met,ks):
 if 'constant' in mm:return {k:np.full(len(X),mm['constant']) for k in ks}
 n=min(max(ks),len(mm['y']));nn=NearestNeighbors(n_neighbors=n,metric=met,algorithm='brute',n_jobs=1).fit(mm['X']);ids=nn.kneighbors(mm['scaler'].transform(X),return_distance=False);w=mm['w'][ids];a=np.cumsum(w*mm['y'][ids],axis=1);b=np.cumsum(w,axis=1);return {k:a[:,min(k,n)-1]/b[:,min(k,n)-1] for k in ks}
