from common import *
import joblib,numpy as np
rows=[]
for b in BB:
 for ds in DS:
  src=HARD/b/(ds+'_probe.joblib');probe=joblib.load(src);sc=probe['scaler'];dest=ROOT/'scalers'/b/(ds+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
  assert sc.with_mean and sc.with_std and np.isfinite(sc.mean_).all() and np.isfinite(sc.scale_).all() and (sc.scale_>0).all()
  np.savez(dest,mean=sc.mean_,scale=sc.scale_,threshold=np.array(probe['threshold']))
  # Diagnostic only: the portable transform implements sklearn's two in-place operations.
  X=np.load(HARD/b/(ds+'_features.npy'),mmap_mode='r');x=np.array(X[:16]);expected=sc.transform(x);actual=x.copy();actual-=sc.mean_;actual/=sc.scale_
  rows.append(dict(backbone=b,dataset=ds,source_sha256=sha(src),export_sha256=sha(dest),max_abs_transform_difference=float(np.max(np.abs(expected-actual)))))
write(ROOT/'SCALER_EXPORTS.json',dict(exports=rows,reason='Deserialize unchanged frozen probes in their original CPU environment; pass only numeric scaler arrays and original threshold to GPU preparation. No fitting or model changes.'))
