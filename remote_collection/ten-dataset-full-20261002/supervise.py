from pathlib import Path
import subprocess,time,json,os,traceback,fcntl,hashlib
R=Path('/home/ubuntu/experiments/ten-dataset-full-20261002');CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python';JUDGE='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';start=time.time()
lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'PIPELINE_STATUS.json').exists()
def status(state,**kw):
 d=dict(state=state,seconds=time.time()-start,**kw);p=R/'PIPELINE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p);print(json.dumps(d),flush=True)
def call(args,log):
 with (R/log).open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 status('waiting_for_probes')
 while not (R/'COMPLETED').exists():
  if (R/'FAILED.json').exists():raise RuntimeError('Probe job failed; no automatic retry')
  p=Path('/proc/354238/cmdline')
  if not p.exists() or str(R/'run.py').encode() not in p.read_bytes():raise RuntimeError('Probe process vanished without completion marker')
  time.sleep(30)
 status('auditing')
 call([CPU,str(R/'audit.py')],'audit.log')
 status('estimating_workload')
 call([CPU,str(R/'estimate.py')],'estimate.log')
 status('extracting_explanations')
 call([CPU,str(R/'explanations.py')],'explanations.log')
 for m in ['qwen','gemma','llama']:
  status('preparing_judge_prompts',model=m)
  call([CPU if m=='qwen' else JUDGE,str(R/'prepare_judges.py'),m],f'prepare_{m}.log')
 for c in ['baseline','saliency','nearest']:
  cfg=[json.loads((R/'judges'/f'{m}_{c}'/'approved_config.json').read_text()) for m in ['qwen','gemma','llama']]
  assert len({x['semantic_sha256'] for x in cfg})==1
  assert len({x['n'] for x in cfg})==1
 (R/'PROMPTS_VERIFIED').write_text('All three models have identical semantic instructions, references, and ordered sample IDs per condition.\n')
 for m in ['qwen','gemma','llama']:
  for c in ['baseline','saliency','nearest']:
   condition=f'{m}_{c}';status('judging',condition=condition)
   call([JUDGE,str(R/'judge_run.py'),str(R/'judges'/condition)],f'{condition}.log')
   call([CPU,str(R/'judge_evaluate.py'),str(R/'judges'/condition)],f'{condition}_evaluation.log')
 status('completed',conditions=9)
 (R/'PIPELINE_COMPLETED').write_text(json.dumps(dict(conditions=9,seconds=time.time()-start)))
except Exception as e:
 status('failed',error=str(e),traceback=traceback.format_exc());raise

