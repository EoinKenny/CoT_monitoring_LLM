from common import *
import subprocess,sys,traceback,fcntl

def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (ROOT/'COMPLETED.json').exists()
 frozen=read(ROOT/'FROZEN.json')
 for p,h in frozen.items():assert sha(ROOT/p)==h,p
 assert read(ROOT/'SETUP_COMPLETED.json')['canonical_sha256']==sha(ROOT/'canonical/cases.jsonl')
 (ROOT/'logs').mkdir(exist_ok=True)
 for stage,python,args in [('prepare',GPU,['prepare_judge.py','qwen']),('judge',GPU,['run_judge.py','qwen']),('evaluate',CPU,['evaluate.py'])]:
  reserve();write(ROOT/'STATUS.json',dict(state='running',stage=stage,time=time.time()))
  with (ROOT/'logs'/f'{stage}.log').open('a') as log:
   process=subprocess.Popen([python,str(ROOT/'code'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,EXPANSION_ROOT=str(ROOT)))
   write(ROOT/'CHILD.json',dict(pid=process.pid,stage=stage,time=time.time()));assert process.wait()==0,f'{stage} failed: see log'
 write(ROOT/'COMPLETED.json',dict(time=time.time(),new_judgments=70000,reused_judgments=20000,analysis_sha256=sha(ROOT/'analysis/COMPLETED.json')));write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/('FAILED_'+str(time.time_ns())+'.json'),dict(traceback=traceback.format_exc()))
  write(ROOT/'STATUS.json',dict(state='failed',time=time.time(),traceback=traceback.format_exc()));raise
