from common import *
import subprocess,fcntl,traceback
C=ROOT/'code'
def call(stage,script,args=(),python=GPU):
 reserve();marker=ROOT/(stage+'_COMPLETED.json')
 if marker.exists():return
 env=dict(os.environ,PYTHONHASHSEED='20261004',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONPATH=str(C));log=ROOT/'logs'/(stage+'.log');write(ROOT/'STATUS.json',dict(state='running',stage=stage,time=time.time(),log=str(log)))
 with log.open('a') as f:
  p=subprocess.Popen([python,'-u',str(C/script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=C);write(ROOT/'CHILD.json',dict(pid=p.pid,stage=stage));rc=p.wait()
 if rc:raise RuntimeError(f'{stage} exited {rc}; saved attempts retained; no automatic retry')
 write(marker,dict(time=time.time(),script_sha256=sha(C/script)))
def main():
 (ROOT/'logs').mkdir(exist_ok=True);lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(ROOT/'CODE_FROZEN.json')=={p.name:sha(p) for p in C.glob('*.py')};assert sha(ROOT/'PROTOCOL.json')==read(ROOT/'PREPARED.json')['protocol_sha256']
 for j in JS:assert (ROOT/'smoke'/j/'run/COMPLETED').exists()
 # Reverify parent inputs before inference and again in evaluation.
 for p,h in read(ROOT/'PARENT_VERIFIED.json')['sources'].items():assert sha(p)==h,p
 write(ROOT/'PID.json',dict(pid=os.getpid(),time=time.time()));write(ROOT/'LAUNCHED.json',dict(time=time.time(),pid=os.getpid()))
 for b in BBS:
  for j in JS:
   for c in CS:call(f'judge_{b}_{j}_{c}','judge.py',[str(ROOT/b/'judges'/f'{j}_{c}')])
 call('evaluate','evaluate.py',python=CPU)
 write(ROOT/'COMPLETED.json',dict(time=time.time(),results_sha256=sha(ROOT/'RESULTS.json')));write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'STATUS.json',dict(state='failed',error=str(e),traceback=traceback.format_exc(),time=time.time()));raise
