"""Fill missing grid cells serially, preserving historical results and failed attempts."""
from common import *
import subprocess,sys,fcntl,traceback
C=ROOT/'code';EXTRACT='/home/ubuntu/qwen122b-fp8-smoke-20261007/env/bin/python'
env=dict(os.environ,PYTHONPATH=str(C),STUDY_ROOT=str(ROOT),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED=str(SEED))
def stage(name,command,backbone='qwen122',extract=False):
 reserve();write(ROOT/'STATUS.json',dict(stage=name,state='running',time=time.time(),controller_pid=os.getpid()))
 e=dict(env,EXPERIMENT_ROOT=str(ROOT/backbone))
 if extract:e['PYTHONPATH']=str(C)+':/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages'
 with (ROOT/(name+'.log')).open('a') as log:
  p=subprocess.Popen(command,cwd=C,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);write(ROOT/'CHILD.json',dict(stage=name,pid=p.pid,command=command,time=time.time()));code=p.wait()
 if code:raise RuntimeError(f'{name} failed with exit {code}; preserve logs and review before retry')
 write(ROOT/(name+'_STAGE_COMPLETED.json'),dict(time=time.time(),command=command))
def once(name,command,**kw):
 if not (ROOT/(name+'_STAGE_COMPLETED.json')).exists():stage(name,command,**kw)
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=read(ROOT/'CODE_FROZEN.json')
 for name,h in manifest['sha256'].items():assert sha(C/name)==h,'Frozen code changed'
 once('protocol_tests',[CPU,str(C/'test_protocol.py')]);once('prepare_grid',[CPU,str(C/'prepare_grid.py')]);once('fit',[CPU,str(C/'fit.py')]);once('select',[CPU,str(C/'select.py'),'qwen122'])
 once('saliency',[EXTRACT,str(C/'saliency.py')],extract=True);once('neighbours',[CPU,str(C/'neighbours.py')])
 for m in JUDGES[:-1]:once('prompts_qwen122_'+m,[GPU,str(C/'prepare_judges.py'),m])
 # Five established judges fill the missing backbone even if the new large judge is unavailable.
 for m in JUDGES[:-1]:
  for c in CONDITIONS:once('judge_qwen122_'+m+'_'+c,[GPU,str(C/'judge.py'),str(B/'judges'/f'{m}_{c}')])
  once('evaluate_qwen122_'+m,[CPU,str(C/'evaluate.py'),'qwen122',m])
 download=ROOT/'qwen122_judge_download'
 while not (download/'DOWNLOAD_COMPLETED.json').exists():
  if list(download.glob('DOWNLOAD_FAILED_*.json')):raise RuntimeError('Qwen122 judge download failed; existing-judge results retained')
  write(ROOT/'STATUS.json',dict(stage='waiting_for_qwen122_judge_download',state='running',time=time.time()));time.sleep(30)
 for b in OLD:once('prompts_'+b+'_qwen122',[GPU,str(C/'prepare_judges.py'),'qwen122'],backbone=b)
 once('qwen122_judge_smoke',[GPU,str(C/'smoke_judge.py')])
 for b in OLD:
  for c in CONDITIONS:once('judge_'+b+'_qwen122_'+c,[GPU,str(C/'judge.py'),str(ROOT/b/'judges'/f'qwen122_{c}')],backbone=b)
  once('evaluate_'+b+'_qwen122',[CPU,str(C/'evaluate.py'),b,'qwen122'])
 once('aggregate',[CPU,str(C/'evaluate.py')]);write(ROOT/'STATUS.json',dict(stage='completed',state='completed',judge_cells=900,probes=50,time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  failure=dict(error=str(e),traceback=traceback.format_exc(),time=time.time());write(ROOT/f'FAILED_{time.time_ns()}.json',failure);write(ROOT/'STATUS.json',dict(stage='halted',state='failed',**failure));raise
