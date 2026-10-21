import sys,os,json,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path('/home/ubuntu/experiments/two-nn-20261014');sys.path.insert(0,str(ROOT/'code'))
from common import *
def stage(name,args,j=None,python=CPU):
 reserve();env=dict(os.environ,EXPANSION_ROOT=str(ROOT),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
 if j:env.update(specs()[j].get('environment',{}))
 write(ROOT/'STATUS.json',dict(state='running',stage=name,recovery='20261015',time=time.time()))
 with (ROOT/'logs'/f'{name}_recovery_20261015.log').open('a') as log:
  p=subprocess.Popen([python,str(ROOT/'code'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT/'code')
  write(ROOT/'CHILD.json',dict(pid=p.pid,stage=name,time=time.time()));assert p.wait()==0,name

def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,h in read(ROOT/'CODE_FROZEN.json').items():assert sha(ROOT/name)==h
 for j in ['qwen','gemma']:assert read(ROOT/'models'/j/'COMPLETED.json')['n']==50000
 for j in ['mistral','olmo','glm','qwen122']:
  stage('judge_'+j,['run_judge.py',j],j=j,python=GPU)
  for b in BBS:
   stage('evaluate_'+b+'_'+j,['evaluate.py',b,j]);stage('diagnostics_'+b+'_'+j,['diagnostics.py',b,j])
 stage('aggregate',['aggregate.py']);assert read(ROOT/'COMPLETED.json')['new_judgments']==300000
 write(ROOT/'STATUS.json',dict(state='completed',recovery='20261015',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  x=dict(state='failed',time=time.time(),traceback=traceback.format_exc());write(ROOT/f'FAILED_{time.time_ns()}.json',x);write(ROOT/'STATUS.json',x);raise
