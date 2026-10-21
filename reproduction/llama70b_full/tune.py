from runtime import *
import traceback
ROOT.mkdir(exist_ok=True);reserve();start=time.time();model=load_model();data=sorted(read(PILOT/'BENCHMARK_INPUTS.json'),key=lambda r:r['n_tokens']);chosen=[data[7],data[len(data)//2],data[-1]]
records=[];policy_candidates=[(32,8192),(64,16384),(128,24576)];eligible=[];warm,_=forward(model,[chosen[0]])
for max_batch,max_tokens in policy_candidates:
 passed=True
 for r in chosen:
  n=min(max_batch,max(1,max_tokens//r['n_tokens']));tag=dict(max_batch=max_batch,max_tokens=max_tokens,length=r['n_tokens'],batch=n)
  try:
   # All padding identical to future right-padded extraction; benchmark labels are unused.
   reference,_=forward(model,[r]);x,stats=forward(model,[r]*n);cos=np.sum(x*reference,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(reference));assert cos.min()>.999
   record=dict(**tag,**stats,min_cosine=float(cos.min()),success=stats['peak_gpu_gib']<40);passed=passed and record['success'];records.append(record);write(ROOT/'TUNING_PARTIAL.json',records);print(json.dumps(record),flush=True)
  except torch.OutOfMemoryError as e:
   records.append(dict(**tag,success=False,error='CUDA OOM'));write(ROOT/'TUNING_PARTIAL.json',records);passed=False;break
 if passed:eligible.append((max_batch,max_tokens))
 else:break
assert eligible,'No tested batch policy fits safely'
# Pick the fastest schedule-aware estimate, using the actual frozen corpus length distribution.
valid=[r for r in records if r.get('success')];fit=np.linalg.lstsq(np.array([[1,r['padded_tokens']] for r in valid]),np.array([r['seconds'] for r in valid]),rcond=None)[0];fit=np.maximum(fit,0)
pop=read(PILOT/'POPULATION.json');cost={p:[0,0] for p in eligible}
for ds in DATASETS:
 rows=read(PRIOR/'llama'/f'{ds}_metadata.json');ratio=pop[ds]['pilot_token_ratio_mean'];minimal=[dict(n_tokens=round(r['n_tokens']*ratio),immutable_id=r['immutable_id']) for r in rows];del rows
 for policy in eligible:
  for ix in batches(minimal,*policy):cost[policy][0]+=1;cost[policy][1]+=len(ix)*max(minimal[i]['n_tokens'] for i in ix)
 estimates=[dict(max_batch=p[0],max_tokens=p[1],batches=v[0],padded_tokens=v[1],hours=float((fit[0]*v[0]+fit[1]*v[1])/3600)) for p,v in cost.items()];winner=min(estimates,key=lambda r:r['hours'])
write(ROOT/'BATCH_POLICY.json',dict(**winner,seed=SEED,selection='Minimum schedule-aware time among policies tested below 40 GiB with cosine > .999 against singleton; no test labels used',measurements=records,alternatives=estimates,seconds_per_batch=float(fit[0]),seconds_per_padded_token=float(fit[1]),estimated_lengths='Prior frozen lengths times per-dataset pilot ratio; replaced by exact new token lengths after preparation',elapsed_seconds=time.time()-start))
print(json.dumps(dict(event='tuned',policy=winner)),flush=True)
