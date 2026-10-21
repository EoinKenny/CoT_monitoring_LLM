import os
os.environ['STUDY_ROOT']='/home/ubuntu/experiments/full-threshold-sweep-20261009'
import sys
from pathlib import Path
R=Path(os.environ['STUDY_ROOT']);sys.path.insert(0,str(R/'llama70'))
from runtime import *
import hashlib
A=R/'repair_v6'
model=load_model()
print('LOADED',torch.cuda.memory_allocated()/2**30,flush=True)
selected=read(B/'selected_flags.json');candidates=[]
for ds in DATASETS:
 rows=read(B/f'{ds}_metadata.json');schedule=read(B/f'{ds}_BATCHES.json');ids={r['sample_id'] for r in selected if r['dataset']==ds}
 for bi,ix in enumerate(schedule):
  if not any(rows[i]['immutable_id'] in ids for i in ix):continue
  width=max(rows[i]['n_tokens'] for i in ix)
  candidates.append((len(ix)*width,width,ds,bi,[rows[i] for i in ix]))
 # Keep only batch rows rather than full metadata after this dataset.
 del rows
worst=max(candidates,key=lambda a:(a[0],a[1]));longest=max(candidates,key=lambda a:(a[1],a[0]))
first=next(c for c in candidates if not (B/'saliency'/c[2]/f'{c[3]:06d}.json').exists())
results=[]
for tag,c in [('pending',first),('max_padded_tokens',worst),('longest_sequence',longest)]:
 features,stats=forward(model,c[4]);free,total=torch.cuda.mem_get_info();result=dict(tag=tag,dataset=c[2],batch=c[3],stats=stats,free_gpu_gib=free/2**30,features_sha256=hashlib.sha256(features.tobytes()).hexdigest());results.append(result);print(json.dumps(result),flush=True);write(A/'BENCHMARK.json',dict(gpu_layers=GPU_LAYERS,results=results))
write(A/'BENCHMARK_COMPLETED.json',dict(gpu_layers=GPU_LAYERS,results=results,time=time.time()))
