from common import *
import subprocess,sys,fcntl,traceback,signal

def stage(name,python,args,j=None):
 reserve();write(ROOT/'STATUS.json',dict(state='running',stage=name,time=time.time()))
 env=dict(os.environ,EXPANSION_ROOT=str(ROOT),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
 env.update(read(ROOT/'judge_specs.json').get('common_environment',{}))
 if j:env.update(specs()[j].get('environment',{}))
 log=ROOT/'logs'/f'{name}_{time.time_ns()}.log'
 with log.open('a') as f:
  p=subprocess.Popen([python,str(ROOT/'code'/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT/'code',start_new_session=True)
  write(ROOT/'CHILD_V4.json',dict(stage=name,pid=p.pid,pgid=p.pid,time=time.time(),log=str(log)))
  last=time.time()
  while p.poll() is None:
   time.sleep(15)
   try:reserve()
   except Exception:
    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=60);raise
   status=ROOT/'models'/j/'STATUS.json' if j else log
   progress=status.stat().st_mtime if status.exists() else log.stat().st_mtime
   last=max(last,progress)
   if time.time()-last>1800:
    write(ROOT/f'STALL_{time.time_ns()}.json',dict(stage=name,pid=p.pid,last_progress=last,log=str(log)))
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=60)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  try:os.killpg(p.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  return p.returncode==0

def main():
 (ROOT/'logs').mkdir(exist_ok=True);f=(ROOT/'controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'COMPLETED.json').exists():return
 for name,h in read(ROOT/'CODE_FROZEN.json').items():assert sha(ROOT/name)==h,name
 assert (ROOT/'PREPARED.json').exists()
 for name,h in read(ROOT/'ARTIFACTS_FROZEN.json').items():assert sha(ROOT/name)==h,name
 failures=[]
 for j in ALL:
  if (ROOT/'models'/j/'COMPLETED.json').exists():continue
  for attempt in range(3):
   if stage(f'full_{j}_attempt{attempt+1}',GPU,['run_judge.py',j],j):break
   write(ROOT/f'WORKER_FAILURE_{j}_{time.time_ns()}.json',dict(judge=j,attempt=attempt+1,time=time.time(),no_resubmission_of_attempted_cases=True))
   time.sleep(15)
  if not (ROOT/'models'/j/'COMPLETED.json').exists():failures.append(j)
 assert stage('coverage',CPU,['coverage.py'])
 coverage=read(ROOT/'COVERAGE.json')
 if not failures and coverage['accounted']==coverage['planned']:
  write(ROOT/'COMPLETED.json',dict(time=time.time(),coverage_sha256=sha(ROOT/'COVERAGE.json'),**coverage))
  write(ROOT/'STATUS.json',dict(state='completed',time=time.time(),coverage=coverage))
 else:write(ROOT/'STATUS.json',dict(state='collection_finished_with_exceptions',time=time.time(),coverage=coverage,failed_judges=failures))
if __name__=='__main__':
 try:main()
 except BaseException:
  x=dict(state='paused_failure',time=time.time(),traceback=traceback.format_exc());write(ROOT/f'FAILED_{time.time_ns()}.json',x);write(ROOT/'STATUS.json',x);raise
