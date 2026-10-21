import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from common import *
import numpy as np,joblib,warnings,gc,time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,roc_auc_score,confusion_matrix,roc_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
threadpool_limits(8);np.random.seed(SEED);warnings.filterwarnings('error',category=ConvergenceWarning)
R=B;bb=read(B/'backbone.json');results={};start=time.time()
def status(event,**kw):
 write(R/'TRAIN_STATUS.json',dict(event=event,seconds=time.time()-start,**kw));print(json.dumps(dict(event=event,**kw)),flush=True)
for ds in DATASETS:
 reserve();done=R/f'{ds}_TRAIN_COMPLETED.json'
 if done.exists():
  mark=read(done)
  for name,h in mark['sha256'].items():assert sha(R/name)==h
  results[ds]=mark['result'];continue
 assert sha(B/f'{ds}_features.npy')==read(B/f'{ds}_EXTRACTED.json')['sha256']
 rows=read(B/f'{ds}_metadata.json');x=np.load(B/f'{ds}_features.npy',mmap_mode='r');status('fitting',dataset=ds)
 y=np.array([r['label'] for r in rows]);sp=np.array([r['split'] for r in rows]);tr=sp=='train';va=sp=='val';scaler=StandardScaler().fit(x[tr]);z=scaler.transform(x)
 clf=LogisticRegression(C=.001 if ds=='high_stakes' else .01,class_weight='balanced',max_iter=2000,tol=1e-6,solver='lbfgs',random_state=SEED).fit(z[tr],y[tr]);p=clf.predict_proba(z)[:,1]
 precision,recall,ts=precision_recall_curve(y[va],p[va]);f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15);threshold=float(ts[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])
 scores={}
 for s in ['train','val','test']:
  sel=sp==s;tn,fp,fn,tp=confusion_matrix(y[sel],p[sel]>=threshold,labels=[0,1]).ravel();scores[s]=dict(confusion_metrics(tp,fp,fn,tn),n=int(sum(sel)),positive=int(sum(y[sel])),auc=float(roc_auc_score(y[sel],p[sel])))
 result=dict(threshold=threshold,metrics=scores,counts={s+':'+str(c):int(sum((sp==s)&(y==c))) for s in ['train','val','test'] for c in [0,1]},flagged_test=int(sum((sp=='test')&(p>=threshold))),training='all eligible, class_weight balanced',C=clf.C,missing_vectors=0)
 joblib.dump(dict(model=clf,scaler=scaler,threshold=threshold,config=dict(backbone=bb,seed=SEED,C=clf.C,class_weight='balanced')),R/f'{ds}_probe.joblib');np.savez(R/f'{ds}_predictions.npz',labels=y,probabilities=p,splits=sp)
 fpr,tpr,t=roc_curve(y[sp=='test'],p[sp=='test']);np.savez(R/f'{ds}_roc.npz',fpr=fpr,tpr=tpr,thresholds=t)
 names=[f'{ds}_{suffix}' for suffix in ['features.npy','metadata.json','probe.joblib','predictions.npz','SPLITS.json','roc.npz']]
 write(done,dict(result=result,sha256={n:sha(R/n) for n in names}));results[ds]=result;write(R/'partial_results.json',results);status('dataset_complete',dataset=ds,result=result);del x,z,rows;gc.collect()
write(R/'results.json',results);write(R/'COMPLETED',dict(datasets=10,seconds=time.time()-start));status('completed')
