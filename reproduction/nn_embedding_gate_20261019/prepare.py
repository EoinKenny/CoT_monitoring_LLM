from common import *
from sklearn.decomposition import PCA
import sys
start=time.time();reserve();b=sys.argv[1];g=cases(b);out=R/'representations'/b;out.mkdir(parents=True,exist_ok=True)
if (out/'COMPLETED.json').exists():
 r=read(out/'COMPLETED.json')
 for name,digest in r['outputs'].items():assert sha(out/name)==digest
 print('representations already verified',b,flush=True);sys.exit()
receipt=read(R/b/'COMPLETED.json')
for name,digest in receipt['outputs'].items():assert sha(R/b/name)==digest
T=np.load(R/b/'target.npy');N=np.load(R/b/'neighbor.npy');mapping=pd.DataFrame(read(R/b/'mapping.json'));sc=np.load(R/b/'scalers.npz');sr=mapping.scaler_row.to_numpy();records=[]
for rep in REPS:
 if rep=='raw_l2':t=T.astype(np.float64);n=N.astype(np.float64)
 else:t=(T-sc['mean'][sr])/sc['scale'][sr];n=(N-sc['mean'][sr])/sc['scale'][sr]
 t=(t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-30)).astype(np.float32);n=(n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-30)).astype(np.float32);assert np.isfinite(t).all() and np.isfinite(n).all();np.save(out/(rep+'__target.npy'),t);np.save(out/(rep+'__neighbor.npy'),n)
 for fold in [0,1,2,'full']:
  mask=g.split.eq('val') & (g.fold.ne(fold) if fold!='full' else True);ix=g.loc[mask,'array_row'].to_numpy();fitX=np.concatenate([t[ix],n[ix]]);pca=PCA(n_components=128,svd_solver='randomized',iterated_power=3,random_state=SEED);pca.fit(fitX)
  tag=rep+'__'+str(fold);np.save(out/(tag+'__target.npy'),pca.transform(t).astype(np.float32));np.save(out/(tag+'__neighbor.npy'),pca.transform(n).astype(np.float32));joblib.dump(pca,out/(tag+'__pca.joblib'),compress=3)
  fitclusters=set(g.loc[mask,'cluster']);hold=g[(g.split=='val')&g.fold.eq(fold)] if fold!='full' else g[g.split=='test'];assert not fitclusters&set(hold.cluster)
  records.append(dict(representation=rep,fold=str(fold),validation_contexts=len(ix),fit_array_rows=ix.tolist(),fit_source_indices=g.loc[mask,'source_index'].tolist(),test_fit_rows=0,heldout_target_overlap=0,explained_variance={str(k):float(pca.explained_variance_ratio_[:k].sum()) for k in [16,64,128]}))
  print('PCA',b,tag,round(time.time()-start,1),'s',flush=True)
write(out/'COMPLETED.json',dict(backbone=b,shape=list(T.shape),records=records,outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npy','.joblib']},seconds=time.time()-start));print('prepared',b,flush=True)
