from pathlib import Path
import json,subprocess,time,os,traceback,fcntl
R=Path('/home/ubuntu/experiments/judge-llama-awq-20261002')
U=Path('/home/ubuntu/experiments/judge-explanations-20261002')
PY='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
start=time.time()
lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'QUEUE_STATUS.json').exists()
def status(state,**kw):
 d=dict(state=state,seconds=time.time()-start,**kw);p=R/'QUEUE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p);print(json.dumps(d),flush=True)
def call(args,log):
 with (R/log).open('x') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 status('downloading')
 call([CPU,str(R/'download.py')],'download.log')
 status('preparing')
 call([PY,str(R/'prepare_baseline.py')],'prepare_baseline.log')
 call([PY,str(R/'prepare_explanations.py'),'llama'],'prepare_explanations.log')
 # Verify semantic prompt identity against each frozen existing condition.
 for root,ref in [(R,Path('/home/ubuntu/experiments/judge-gemma-20261002'))]+[(R/c,U/f'qwen_{c}') for c in ['saliency','nearest']]:
  a=[json.loads(s) for s in (root/'prompts.jsonl').read_text().splitlines()]
  b=[json.loads(s) for s in (ref/'prompts.jsonl').read_text().splitlines()]
  assert len(a)==len(b)==5190
  assert [(r['sample_id'],r['instruction_sha256']) for r in a]==[(r['sample_id'],r['instruction_sha256']) for r in b]
 status('waiting_for_qwen_and_gemma',upstream=str(U))
 while not (U/'COMPLETED').exists():
  state=json.loads((U/'QUEUE_STATUS.json').read_text())['state']
  if state in ['failed','evaluation_failed']:raise RuntimeError('Upstream queue failed; no automatic continuation')
  cmd=Path('/proc/344285/cmdline')
  if not cmd.exists() or str(U/'supervise.py').encode() not in cmd.read_bytes():raise RuntimeError('Upstream supervisor is no longer running without completion marker')
  time.sleep(30)
 for condition,root in [('baseline',R),('saliency',R/'saliency'),('nearest',R/'nearest')]:
  status('running',condition=condition)
  call([PY,str(R/'run.py'),str(root)],f'{condition}.log')
  call([PY,str(R/'evaluate.py'),str(root)],f'{condition}_evaluation.log')
 status('completed',conditions=3,judgments=15570)
 (R/'COMPLETED').write_text(json.dumps(dict(conditions=3,judgments=15570,seconds=time.time()-start)))
except Exception as e:
 status('failed',error=str(e),traceback=traceback.format_exc());raise
