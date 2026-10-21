from common import *
import subprocess,fcntl,traceback,sys
def call(stage,script,args=(),j=None,python=GPU):
 reserve();marker=ROOT/(stage+'_COMPLETED.json')
 if marker.exists():return
 env=dict(os.environ,PYTHONHASHSEED='20261011',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONPATH=str(ROOT/'code'))
 if j:
  extra=specs()[j].get('environment',{});env.update(extra)
  if extra.get('PYTHONPATH'):env['PYTHONPATH']=extra['PYTHONPATH']+':'+str(ROOT/'code')
 log=ROOT/'logs'/(stage+'.log');write(ROOT/'STATUS.json',dict(state='running',stage=stage,time=time.time(),log=str(log)))
 with log.open('a') as f:
  p=subprocess.Popen([python,'-u',str(script),*args],cwd=ROOT/'code',env=env,stdout=f,stderr=subprocess.STDOUT);write(ROOT/'CHILD.json',dict(pid=p.pid,stage=stage,time=time.time()));rc=p.wait()
 if rc:raise RuntimeError(f'{stage} failed with exit {rc}; raw attempts retained, no automatic retry')
 write(marker,dict(time=time.time(),script_sha256=sha(script)))
def main():
 (ROOT/'logs').mkdir(exist_ok=True);lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(ROOT/'CODE_FROZEN.json')=={str(p.relative_to(ROOT)):sha(p) for base in [ROOT/'code',ROOT/'evaluation'] for p in base.rglob('*.py')}
 assert sha(ROOT/'judge_specs.json')==read(ROOT/'RUN_MANIFEST.json')['specs_sha256'] and sha(ROOT/'PROTOCOL.json')==read(ROOT/'RUN_MANIFEST.json')['protocol_sha256']
 assert (ROOT/'canonical/PREPARED.json').exists()
 if not (ROOT/'SETUP_COMPLETED.json').exists():call('setup',ROOT/'code/setup_reduced.py',python=CPU)
 write(ROOT/'PID.json',dict(pid=os.getpid(),time=time.time()))
 for j in NEW:
  call('prepare_'+j,ROOT/'code/prepare_judge.py',[j],j=j)
  call('judge_'+j,ROOT/'code/run_judge.py',[j],j=j)
 call('evaluate',ROOT/'evaluation/evaluate_expansion.py',['--hard-root',str(ROOT/'hard'),'--probability-root',str(ROOT/'probability'),'--output',str(ROOT/'analysis')],python=CPU)
 write(ROOT/'COMPLETED.json',dict(time=time.time(),n=462000,judges=ALL));write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/'STATUS.json',dict(state='failed',time=time.time(),traceback=traceback.format_exc()));raise
