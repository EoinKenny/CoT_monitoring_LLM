from common import *
import sys,subprocess,traceback,fcntl,datetime

def stage(name,python,args,j=None):
 reserve();write(ROOT/'STATUS.json',dict(state='running',stage=name,time=time.time()))
 env=dict(os.environ,EXPANSION_ROOT=str(ROOT),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
 if (ROOT/'judge_specs.json').exists():
  env.update(read(ROOT/'judge_specs.json').get('common_environment',{}))
  if j:env.update(specs()[j].get('environment',{}))
 with (ROOT/'logs'/f'{name}.log').open('a') as f:
  p=subprocess.Popen([python,str(ROOT/'code'/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT/'code')
  write(ROOT/'CHILD.json',dict(stage=name,pid=p.pid,time=time.time()));assert p.wait()==0,f'{name} failed; inspect preserved logs'

def eta():
 models={j:read(ROOT/'models'/j/'SMOKE_PASSED.json') for j in ALL if (ROOT/'models'/j/'SMOKE_PASSED.json').exists()}
 data=dict(time=time.time(),smoked_judges=list(models),complete=len(models)==6,models=models)
 if data['complete']:
  seconds=sum(x['estimated_remaining_seconds'] for x in models.values())
  data.update(remaining_inference_hours=seconds/3600,planning_hours=seconds*1.25/3600,estimated_finish_utc=datetime.datetime.fromtimestamp(time.time()+seconds*1.25,datetime.timezone.utc).isoformat(),note='Two batches per backbone/judge; approximate runtime. 25% planning margin. CPU evaluation additional.')
 write(ROOT/'ETA.json',data)

def main():
 (ROOT/'logs').mkdir(exist_ok=True);f=(ROOT/'controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (ROOT/'COMPLETED.json').exists()
 for name,h in read(ROOT/'RECOVERY_V3_FROZEN.json').items():assert sha(ROOT/name)==h,name
 if not (ROOT/'SETUP_COMPLETED.json').exists():stage('setup',CPU,['setup.py'])
 for j in ALL:
  stage('prepare_'+j,GPU,['prepare_judge.py',j],j)
  if not (ROOT/'models'/j/'SMOKE_PASSED.json').exists():stage('smoke_'+j,GPU,['run_judge_v3.py',j,'smoke'],j)
  eta()
 assert read(ROOT/'ETA.json')['complete'];write(ROOT/'ALL_SMOKES_PASSED.json',dict(time=time.time(),pairs=30,judgments=sum(sum(c['n'] for c in x['cells']) for x in read(ROOT/'ETA.json')['models'].values()),eta_sha256=sha(ROOT/'ETA.json')))
 if not (ROOT/'FULL_LAUNCHED.json').exists():write(ROOT/'FULL_LAUNCHED.json',dict(time=time.time(),eta=read(ROOT/'ETA.json')))
 for j in ALL:
  if not (ROOT/'models'/j/'COMPLETED.json').exists():stage('full_'+j,GPU,['run_judge_v3.py',j,'full'],j)
 for b in BBS:
  for j in ALL:
   stage('evaluate_'+b+'_'+j,CPU,['evaluate_v3.py',b,j])
   stage('agreement_'+b+'_'+j,CPU,['agreement.py',b,j])
 stage('aggregate',CPU,['aggregate.py']);assert read(ROOT/'COMPLETED.json')['new_judgments']==300000
 write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  x=dict(state='failed',time=time.time(),traceback=traceback.format_exc());write(ROOT/f'FAILED_{time.time_ns()}.json',x);write(ROOT/'STATUS.json',x);raise
