import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from common import *
import numpy as np,joblib,traceback

def metric(y,p,w):
 tp=float(np.sum(w*(y==1)*p));fp=float(np.sum(w*(y==0)*p));fn=float(np.sum(w*(y==1)*(~p)))
 return dict(f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0)
def main():
 results=[]
 for b in BB:
  selected=read(PARENT/'score'/b/'selected_flags.json')
  for ds in DS:
   reserve();d=ROOT/'knn'/b/ds;d.mkdir(parents=True,exist_ok=True);assert not (d/'COMPLETED.json').exists()
   write(ROOT/'knn/STATUS.json',dict(state='running',backbone=b,dataset=ds,time=time.time()))
   meta=read(HARD/b/(ds+'_metadata.json'));X=np.load(HARD/b/(ds+'_features.npy'),mmap_mode='r');pr=joblib.load(HARD/b/(ds+'_probe.joblib'));sc=pr['scaler'];train=np.array([i for i,r in enumerate(meta) if r['split']=='train']);sel=[r for r in selected if r['dataset']==ds];ids=np.array([r['index'] for r in sel]);
   A=sc.transform(np.array(X[train])).astype('float32');B=sc.transform(np.array(X[ids])).astype('float32');assert np.isfinite(A).all() and np.isfinite(B).all()
   labels=np.array([meta[i]['label'] for i in train]);y=np.array([r['true_label'] for r in sel]);w=np.array([1/r['inclusion_probability'] for r in sel]);val=np.array([r['split']=='val' for r in sel]);scores={};timings={};saved={}
   for met in ['cosine','euclidean']:
    t=time.perf_counter();a=A.copy();q=B.copy()
    if met=='cosine':a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-20);q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-20)
    norms=(a*a).sum(1);allix=[];alld=[]
    for start in range(0,len(q),64):
     v=q[start:start+64];dist=1-v@a.T if met=='cosine' else (v*v).sum(1)[:,None]+norms[None,:]-2*v@a.T
     for j,r in enumerate(sel[start:start+64]):
      blocked=np.array([str(meta[i]['group'])==str(meta[r['index']]['group']) for i in train]);dist[j,blocked]=np.inf
     ix=np.argsort(dist,axis=1,kind='stable')[:,:51];allix.append(ix);alld.append(np.take_along_axis(dist,ix,axis=1))
    ix=np.concatenate(allix);dd=np.concatenate(alld);timings[met]=time.perf_counter()-t;saved[met+'_train_indices']=train[ix];saved[met+'_distances']=dd
    for k in [1,5,15,51]:scores[met+'_'+str(k)]=labels[ix[:,:k]].mean(1)
   candidates=[]
   for name,s in scores.items():
    for cutoff in range(102):
     m=metric(y[val],s[val]>=cutoff/100,w[val]);candidates.append((m['f1'],cutoff,name))
   frozen=np.array([r['probe_probability'] for r in sel]);scores['linear_probe']=frozen
   linear_cut=max(range(102),key=lambda c:(metric(y[val],frozen[val]>=c/100,w[val])['f1'],c))
   best=max(candidates,key=lambda x:(x[0],x[1],x[2]));_,cutoff,name=best
   out=dict(backbone=b,dataset=ds,n_train=len(train),n_val=int(val.sum()),n_test=int((~val).sum()),selected_method=name,selected_cutoff=cutoff,validation_f1=best[0],test=metric(y[~val],scores[name][~val]>=cutoff/100,w[~val]),timings_seconds=timings,frame='Design-weighted selected sample across full score range; direct classifier, not cascade',linear_selected_cutoff=linear_cut,linear_test_tuned=metric(y[~val],frozen[~val]>=linear_cut/100,w[~val]),test_fixed50={n:metric(y[~val],s[~val]>=.5,w[~val]) for n,s in scores.items()})
   np.savez_compressed(d/'scores.npz',**scores,**saved);write(d/'mapping.json',sel);write(d/'COMPLETED.json',dict(**out,sha256=sha(d/'scores.npz'),metadata_sha256=sha(HARD/b/(ds+'_metadata.json')),probe_sha256=sha(HARD/b/(ds+'_probe.joblib'))));results.append(out)
 write(ROOT/'knn/COMPLETED.json',dict(cells=results,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'knn/FAILED.json',dict(traceback=traceback.format_exc()));raise
