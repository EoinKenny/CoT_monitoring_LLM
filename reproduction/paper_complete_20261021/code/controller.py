from common import *
import subprocess,fcntl,traceback
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not (ROOT/'PREPARED.json').exists():
  write(ROOT/'STATUS.json',dict(state='preparing',pid=os.getpid(),time=time.time()))
  with (ROOT/'logs/prepare.log').open('x') as f:subprocess.run([CPU,str(ROOT/'code/prepare.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 failures=[];completed=[]
 for index,t in enumerate(read(ROOT/'PREPARED.json')['tasks']):
  write(ROOT/'STATUS.json',dict(state='running',index=index,task=t,pid=os.getpid(),completed_tasks=completed,failed_tasks=failures,time=time.time()))
  with (ROOT/'logs'/f'task_{index:02d}.log').open('x') as f:r=subprocess.run([GPU,str(ROOT/'code/run_task.py'),str(index)],stdout=f,stderr=subprocess.STDOUT)
  (completed if r.returncode==0 else failures).append(index)
 write(ROOT/'COLLECTION_FINISHED.json',dict(completed_tasks=completed,failed_tasks=failures,time=time.time()));write(ROOT/'STATUS.json',dict(state='completed' if not failures else 'finished_with_failures',failed_tasks=failures,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));write(ROOT/'STATUS.json',dict(state='failed',time=time.time()));raise
