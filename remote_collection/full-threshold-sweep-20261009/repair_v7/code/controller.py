import os,sys,json,subprocess,time,fcntl,traceback,hashlib,shutil
from pathlib import Path
ROOT=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009');C=ROOT/'code'
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python';GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';EXTRACT='/home/ubuntu/qwen122b-fp8-smoke-20261007/env/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(stage,python,script,args=(),b=None,extra=None):
 if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('Below 20 GiB reserve')
 marker=ROOT/(stage+'_COMPLETED.json')
 if marker.exists():return
 env=dict(os.environ,STUDY_ROOT=str(ROOT),PYTHONHASHSEED='20261004',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1')
 env['PYTHONPATH']=':'.join(([str(ROOT/b)] if b else [])+[str(C)])
 if b:env['EXPERIMENT_ROOT']=str(ROOT/b)
 if extra:env.update(extra)
 log=ROOT/'logs'/(stage+'.log');write(ROOT/'STATUS.json',dict(stage=stage,state='running',time=time.time(),log=str(log)))
 with log.open('a') as f:
  p=subprocess.Popen([python,'-u',str(script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=script.parent);write(ROOT/'CHILD.json',dict(pid=p.pid,stage=stage));code=p.wait()
 if code:raise RuntimeError(f'{stage} failed ({code}); preserved log and outputs, no automatic retry')
 write(marker,dict(time=time.time(),script_sha256=sha(script)))
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True);lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(ROOT/'PID.json',dict(pid=os.getpid(),time=time.time()))
 call('setup',CPU,C/'setup.py')
 frozen=ROOT/'CODE_FROZEN.json';code={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not p.is_symlink()}
 if frozen.exists():assert read(frozen)==code
 else:write(frozen,code)
 # Cached token activations first; large-backbone replays follow in isolated stages.
 for b in ['llama','qwen','gemma','llama70','qwen122']:
  R=ROOT/b
  if b in ['llama','qwen','gemma']:
   call('replay_missing_'+b,GPU if b=='gemma' else CPU,C/'replay_missing.py',[b],b=b)
   call('explanations_'+b,GPU if b=='gemma' else CPU,R/'explanations.py',b=b)
  else:
   extra={'PYTHONPATH':str(R)+':/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages'} if b=='qwen122' else None
   call('saliency_'+b,EXTRACT if b=='qwen122' else CPU,R/'saliency.py',b=b,extra=extra);call('neighbours_'+b,CPU,R/'neighbours.py',b=b)
  for m in ['qwen','gemma']:
   call('prompts_'+b+'_'+m,GPU,R/'prepare_judges.py',[m],b=b)
   for c in ['baseline','saliency','nearest']:call('judge_'+b+'_'+m+'_'+c,GPU,C/'judge.py',[str(R/'judges'/f'{m}_{c}')],b=b)
 call('evaluate',CPU,C/'evaluate.py');write(ROOT/'COMPLETED.json',dict(time=time.time(),results_sha256=sha(ROOT/'RESULTS.json')));write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'STATUS.json',dict(state='failed',error=str(e),traceback=traceback.format_exc(),time=time.time()));raise
