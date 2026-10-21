import os,sys,subprocess,time,fcntl,traceback,importlib.metadata
from common import *
C=ROOT/'code';lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();logs=ROOT/'logs';logs.mkdir(exist_ok=True)
write(ROOT/'SUPERVISOR_PID.json',dict(pid=os.getpid(),time=time.time()))
def status(stage,**kw):write(ROOT/'STATUS.json',dict(stage=stage,seconds=time.time()-start,pid=os.getpid(),**kw));print(json.dumps(dict(stage=stage,**kw)),flush=True)
def call(python,script,args=(),b=None):
 reserve();env=os.environ.copy();env.update(STUDY_ROOT=str(ROOT),PYTHONHASHSEED=str(SEED),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
 env['PYTHONPATH']=':'.join(([str(ROOT/b)] if b else [])+[str(C)])
 if b:env['EXPERIMENT_ROOT']=str(ROOT/b)
 stamp=str(time.time_ns());log=logs/(stamp+'_'+Path(script).stem+'.log');status('running',script=str(script),args=list(args),backbone=b,log=str(log))
 with log.open('x') as f:subprocess.run([python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 for name,python in [('cpu',CPU),('gpu',GPU)]:
  if not (ROOT/(name+'_ENVIRONMENT.json')).exists():call(python,C/'environment.py',[name])
 for b in OLD:
  if not (ROOT/b/'COMPLETED').exists():call(GPU if b=='gemma' else CPU,C/'train.py',[b],b)
  if not (ROOT/b/'SAMPLE_FROZEN.json').exists():call(CPU,C/'select.py',[b],b)
 counts={b:read(ROOT/b/'SAMPLE_FROZEN.json')['counts'] for b in OLD};write(ROOT/'SAMPLING_BALANCE.json',counts)
 for b in OLD:
  R=ROOT/b
  if not (R/'judge_inputs/EXTRACTION_COMPLETED').exists():call(GPU if b=='gemma' else CPU,R/'explanations.py',b=b)
  for m in JUDGES:
   marker=R/'judges'/f'{m}_PROMPTS_COMPLETED.json'
   if not marker.exists():
    call(GPU,R/'prepare_judges.py',[m],b);write(marker,dict(configs={c:sha(R/'judges'/f'{m}_{c}/approved_config.json') for c in CONDITIONS}))
 call(CPU,C/'estimate.py')
 for b in OLD:
  for m in JUDGES:
   for c in CONDITIONS:
    directory=ROOT/b/'judges'/f'{m}_{c}'
    if not (directory/'run/COMPLETED').exists():call(GPU,C/'judge.py',[str(directory)],b)
   call(CPU,C/'evaluate.py',[b,m],b)
 call(CPU,C/'evaluate.py');status('completed')
try:
 while not (ROOT/'prepared/NEW_PREPARED').exists():
  live=False
  for p in Path('/proc').glob('[0-9]*/cmdline'):
   try:live=live or str(C/'prepare_full.py').encode() in p.read_bytes()
   except (PermissionError,FileNotFoundError,ProcessLookupError):pass
  assert live,'Preparation exited without completion; inspect prepare.log'
  status('waiting_for_data_preparation');time.sleep(15)
 assert (ROOT/'OLD_CONTROLLER_RETIRED.json').exists()
 code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')};frozen=ROOT/'CODE_FROZEN.json'
 if frozen.exists():assert read(frozen)==code,'Source changed after launch; require documented new run version'
 else:write(frozen,code)
 main()
except Exception as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
