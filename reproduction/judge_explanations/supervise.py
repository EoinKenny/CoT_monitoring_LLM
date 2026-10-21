from pathlib import Path
import json,subprocess,time,os
R=Path('/home/ubuntu/experiments/judge-explanations-20261002');PY='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python';start=time.time()
assert not (R/'QUEUE_STATUS.json').exists()
def status(state,**kw):
 d=dict(state=state,seconds=time.time()-start,**kw);p=R/'QUEUE_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p);print(json.dumps(d),flush=True)
for condition in ['qwen_saliency','qwen_nearest','gemma_saliency','gemma_nearest']:
 status('running',condition=condition)
 with (R/f'{condition}.log').open('w') as log:
  result=subprocess.run([PY,str(R/'run.py'),str(R/condition)],stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   status('failed',condition=condition,returncode=result.returncode);raise SystemExit(result.returncode)
  result=subprocess.run([PY,str(R/'evaluate.py'),str(R/condition)],stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   status('evaluation_failed',condition=condition,returncode=result.returncode);raise SystemExit(result.returncode)
status('completed',conditions=4,judgments=20760)
(R/'COMPLETED').write_text(json.dumps(dict(conditions=4,judgments=20760,seconds=time.time()-start)))
