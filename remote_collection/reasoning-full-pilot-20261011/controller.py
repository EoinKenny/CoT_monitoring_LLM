from pathlib import Path
import subprocess,json,time,fcntl,hashlib
R=Path('/home/ubuntu/experiments/reasoning-full-pilot-20261011')
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'LAUNCHED.json').exists(),'Controller already launched; inspect durable state before explicit recovery'
(R/'LAUNCHED.json').write_text(json.dumps(dict(time=time.time(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')})))
for tag,cmd in [('qwen',[GPU,str(R/'judge.py'),'qwen']),('gemma',[GPU,str(R/'judge.py'),'gemma']),('evaluate',[CPU,str(R/'evaluate.py')])]:
 with (R/(tag+'.log')).open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (R/(tag+'_EXIT.json')).write_text(json.dumps(dict(time=time.time(),code=p.returncode)))
 if p.returncode:
  (R/'FAILED.json').write_text(json.dumps(dict(time=time.time(),stage=tag,code=p.returncode)));raise SystemExit(p.returncode)
 time.sleep(5)
(R/'COMPLETED.json').write_text(json.dumps(dict(time=time.time(),analysis_sha256=hashlib.sha256((R/'RESULTS.json').read_bytes()).hexdigest())))
