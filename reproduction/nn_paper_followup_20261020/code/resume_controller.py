from common import *
import subprocess,fcntl,traceback
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for j in JUDGES:
  d=ROOT/'runs/controlled'/j;doc=read(d/'COMPLETED.json');assert doc['n']==160 and doc['invalid']==0 and sha(d/'responses.jsonl')==doc['responses_sha256'];assert sha(d/'CONFIG.json')==doc['config_sha256']
  for p in (d/'batches').glob('*.json'):
   if p.name.endswith(('.raw.json','.attempt.json')):continue
   x=read(p);assert sha(p.with_suffix('.raw.json'))==x['raw_sha256']
 assert not (ROOT/'PREPARED.json').exists() and not list((ROOT/'runs/natural').glob('*')),'Do not regenerate prepared inputs or attempted natural calls'
 jobs=[('prepare_v2.py',)]+[('run_judge.py',j,'natural') for j in JUDGES]
 for job in jobs:
  reserve();write(ROOT/'STATUS.json',dict(stage='_'.join(job),state='running',recovery='v2',pid=os.getpid(),time=time.time()))
  with open(ROOT/'logs'/('recovery_v2_'+'_'.join(job)+'.log'),'x') as f:subprocess.run([GPU,str(ROOT/'code'/job[0]),*job[1:]],stdout=f,stderr=subprocess.STDOUT,check=True)
 write(ROOT/'COLLECTION_COMPLETED.json',dict(time=time.time(),recovery='v2',human_validation='pending'))
 write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'FAILED_v2.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
