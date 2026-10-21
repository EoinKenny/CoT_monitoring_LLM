import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from common import *
import numpy as np,joblib,statistics
out=[]
for b in BB:
 for ds in DS:
  d=read(ROOT/'knn'/b/ds/'COMPLETED.json');sel=read(ROOT/'knn'/b/ds/'mapping.json');probe=joblib.load(HARD/b/(ds+'_probe.joblib'));X=np.load(HARD/b/(ds+'_features.npy'),mmap_mode='r');B=probe['scaler'].transform(np.array(X[[r['index'] for r in sel]])).astype('float32');clf=probe['model']
  # Warm and time the same 1000 pre-extracted, standardized rows. No new LLM work.
  clf.predict_proba(B);times=[]
  for repeat in range(15):
   start=time.perf_counter()
   for k in range(0,len(B),64):clf.predict_proba(B[k:k+64])
   times.append(time.perf_counter()-start)
  measured=statistics.median(times);met=d['selected_method'].rsplit('_',1)[0];search=d['timings_seconds'][met]
  out.append(dict(backbone=b,dataset=ds,n=len(B),dimension=B.shape[1],linear_seconds=measured,linear_ms_per_case=1000*measured/len(B),knn_recorded_seconds=search,knn_ms_per_case=1000*search/len(B),recorded_search_to_linear_ratio=search/measured,linear_repeats=15))
write(ROOT/'knn/LINEAR_TIMING.json',dict(rows=out,method='CPU,8 BLAS threads,batch64,warm median15 linear runs on same standardized embeddings. Compare saved exact kNN sweep timing, including normalization, group exclusion and full sorting to top51. Excludes backbone encoding and disk/scaler setup. Timing ratio is implementation-specific, not a Faiss or GPU benchmark.'))
print(__import__('json').dumps(out))
