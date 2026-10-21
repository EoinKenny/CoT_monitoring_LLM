from common import *
import subprocess,fcntl,sys,traceback
if __name__=='__main__':
 lock=(ROOT/'boot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,EXPANSION_ROOT=str(ROOT),OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
 try:
  if not (ROOT/'PREPARED.json').exists():
   write(ROOT/'STATUS.json',dict(state='preparing',time=time.time()))
   subprocess.run([GPU,str(ROOT/'code/setup.py')],env=env,check=True)
  subprocess.run([CPU,str(ROOT/'code/controller.py')],env=env,check=True)
 except BaseException:
  write(ROOT/f'BOOT_FAILED_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
