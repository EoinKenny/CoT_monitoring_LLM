from pathlib import Path
import subprocess,json,time,os,fcntl,traceback,hashlib,sys
R=Path('/home/ubuntu/experiments/expanded-judges-20261003');PY='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python';start=time.time()
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'PIPELINE_STATUS.json').exists()
state=dict(completed=[],failures={})
def status(stage,**kw):
 state.update(stage=stage,elapsed_seconds=time.time()-start,**kw);p=R/'PIPELINE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));os.replace(t,p);print(json.dumps(state),flush=True)
def call(script,args,log,python=PY):
 with (R/log).open('x') as f:subprocess.run([python,str(R/script)]+args,stdout=f,stderr=subprocess.STDOUT,check=True)
# Independent downloads; GPU inference stays strictly sequential.
handles={};logs={}
status('downloading')
for m in ['glm','olmo']:
 logs[m]=(R/(m+'_download.log')).open('x');handles[m]=subprocess.Popen([PY,str(R/'download.py'),m],stdout=logs[m],stderr=subprocess.STDOUT)
(R/'SOURCE_CHECKSUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},indent=2))
for m in ['glm','olmo']:
 try:
  status('waiting_for_download',model=m)
  assert handles[m].wait()==0, 'Checkpoint download failed; see model download log'
  for b in ['llama','qwen']:
   status('preparing_prompts',model=m,backbone=b);call('prepare_judges.py',[m,b],f'prepare_{m}_{b}.log')
  for b in ['llama','qwen']:
   for c in ['baseline','saliency','nearest']:
    cond=m+'_'+c;status('judging',model=m,backbone=b,condition=c)
    call('judge_run.py',[str(R/b/'judges'/cond)],f'{b}_{cond}.log')
    call('evaluate.py',[str(R/b),cond],f'{b}_{cond}_evaluation.log',CPU)
    state['completed'].append(b+'_'+cond)
 except Exception as e:
  state['failures'][m]=dict(error=str(e),traceback=traceback.format_exc());status('model_failed',model=m)
  # Do not retry a failed condition. Independent second model still gets its scheduled run.
if not state['failures']:
 for b in ['llama','qwen']:call('evaluate.py',[str(R/b)],f'{b}_evaluation.log',CPU)
 combined={}
 for b in ['llama','qwen']:
  src=R.parent/('ten-dataset-sampled-20261002' if b=='llama' else 'qwen4b-backbone-20261002')
  old=json.loads((src/'results.json').read_text());new=json.loads((R/b/'results.json').read_text())
  combined[b]={ds:dict(old['results'][ds],**new['results'][ds]) for ds in old['results']}
 (R/'ALL_FIVE_JUDGES.json').write_text(json.dumps(dict(results=combined,scope='Five judges, two backbones, ten datasets, three judge conditions. Same frozen samples within backbone; exact probes and estimated cascades.'),indent=2))
 status('completed');(R/'PIPELINE_COMPLETED').write_text(json.dumps(dict(conditions=len(state['completed']),elapsed_seconds=time.time()-start)))
else:status('finished_with_failures')
