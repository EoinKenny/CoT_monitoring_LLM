"""Single canonical feature matrix with atomic, verified batch commits."""
from runtime import *
import hashlib,fcntl
lock=(ROOT/'gpu-stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
reserve();model=load_model();started=time.time();run_rows=run_batches=0
for ds in DATASETS:
 rows=read(B/f'{ds}_metadata.json');schedule=read(B/f'{ds}_BATCHES.json');path=B/f'{ds}_features.npy';done=B/f'{ds}_EXTRACTED.json';proofs=B/'feature_commits'/ds;proofs.mkdir(parents=True,exist_ok=True)
 if done.exists():assert sha(path)==read(done)['sha256'];continue
 if path.exists():x=np.load(path,mmap_mode='r+')
 else:
  assert shutil.disk_usage(ROOT).free>20*2**30+len(rows)*8192*4
  x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),8192));x.flush()
 assert x.shape==(len(rows),8192) and x.dtype==np.float32
 for bi,ix in enumerate(schedule):
  reserve();proof=proofs/f'{bi:06d}.json';ids=[rows[i]['immutable_id'] for i in ix];h=lambda v:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
  if proof.exists():
   mark=read(proof);assert mark['ids']==ids and mark['feature_sha256']==h(x[ix]);continue
  vals,stats=forward(model,[rows[i] for i in ix]);assert np.isfinite(vals).all()
  if run_batches==0:
   repeated,_=forward(model,[rows[i] for i in ix]);assert np.array_equal(vals,repeated),'Same-batch activations not bitwise reproducible';write(ROOT/('REPEAT_VERIFIED_'+str(time.time_ns())+'.json'),dict(dataset=ds,batch=bi,bitwise_equal=True))
  x[ix]=vals;x.flush()
  with path.open('rb') as f:os.fsync(f.fileno())
  write(proof,dict(ids=ids,indices=ix,feature_sha256=h(vals),metadata_sha256=read(B/f'{ds}_PREPARED.json')['sha256'][f'{ds}_metadata.json'],stats=stats));run_rows+=len(ix);run_batches+=1
  status=dict(event='extracting',dataset=ds,batches_completed=bi+1,dataset_batches=len(schedule),run_rows=run_rows,run_batches=run_batches,elapsed_seconds=time.time()-started,stats=stats,free_disk_gib=shutil.disk_usage(ROOT).free/2**30);write(ROOT/'EXTRACTION_STATUS.json',status)
  if bi%10==0:print(json.dumps(status),flush=True)
 assert np.isfinite(x).all();write(done,dict(rows=len(rows),batches=len(schedule),sha256=sha(path),metadata_sha256=sha(B/f'{ds}_metadata.json'),schedule_sha256=sha(B/f'{ds}_BATCHES.json')));del x,rows
write(ROOT/'EXTRACTION_COMPLETED.json',dict(time=time.time(),datasets=DATASETS))
