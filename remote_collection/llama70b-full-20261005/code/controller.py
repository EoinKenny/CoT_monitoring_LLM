"""Detached serial controller; no automatic retries of failed stages or judgments."""
from common import *
import subprocess,sys,fcntl,traceback
C=ROOT/'code';env=dict(os.environ,PYTHONPATH=str(C),STUDY_ROOT=str(ROOT),EXPERIMENT_ROOT=str(B),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def stage(name,command):
 reserve();write(ROOT/'STATUS.json',dict(stage=name,state='running',time=time.time(),controller_pid=os.getpid()))
 with (ROOT/(name+'.log')).open('a') as log:
  p=subprocess.Popen(command,cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);write(ROOT/'CHILD.json',dict(stage=name,pid=p.pid,command=command,time=time.time()));code=p.wait()
 if code:raise RuntimeError(f'{name} failed with exit {code}; inspect preserved log, no automatic retry')
 write(ROOT/(name+'_STAGE_COMPLETED.json'),dict(time=time.time(),command=command))
def once(name,command):
 if not (ROOT/(name+'_STAGE_COMPLETED.json')).exists():stage(name,command)
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (ROOT/'BATCH_POLICY.json').exists(),'Tuning must complete before launch'
 provenance=read(ROOT/'SOURCE_MANIFEST.json')
 for name,h in provenance['code'].items():assert sha(C/name)==h,'Frozen code changed'
 once('tests',[CPU,str(C/'test_protocol.py')]);once('cpu_environment',[CPU,str(C/'environment.py'),'cpu']);once('gpu_environment',[GPU,str(C/'environment.py'),'gpu'])
 once('prepare',[CPU,str(C/'prepare.py')]);once('extract',[CPU,str(C/'extract.py')]);once('fit',[CPU,str(C/'fit.py')]);once('select',[CPU,str(C/'select.py'),'llama70']);once('saliency',[CPU,str(C/'saliency.py')]);once('neighbours',[CPU,str(C/'neighbours.py')])
 for m in JUDGES:once('prompts_'+m,[GPU,str(B/'prepare_judges.py'),m])
 # Record actual selected cases/token counts before inference for later ETA updates.
 write(ROOT/'JUDGING_PLAN.json',dict(selected=len(read(B/'selected_flags.json')),judgments=15*len(read(B/'selected_flags.json')),configs={f'{m}_{c}':read(B/'judges'/f'{m}_{c}'/'approved_config.json') for m in JUDGES for c in CONDITIONS}))
 for m in JUDGES:
  for c in CONDITIONS:stage('judge_'+m+'_'+c,[GPU,str(C/'judge.py'),str(B/'judges'/f'{m}_{c}')])
  once('evaluate_'+m,[CPU,str(C/'evaluate.py'),'llama70',m])
 once('aggregate',[CPU,str(C/'evaluate.py')]);once('combined',[CPU,str(C/'combine.py')]);write(ROOT/'STATUS.json',dict(stage='completed',state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  failure=dict(error=str(e),traceback=traceback.format_exc(),time=time.time());write(ROOT/('FAILED_'+str(time.time_ns())+'.json'),failure);write(ROOT/'STATUS.json',dict(stage='halted',state='failed',**failure));raise
