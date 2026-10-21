from pathlib import Path
import subprocess,time,json,os
R=Path('/home/ubuntu/experiments/reasoning-benchmark-two-batches-20261011')
PY='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
assert not (R/'LAUNCHED.json').exists()
(R/'LAUNCHED.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid())))
for mode in ['gemma','glm','gptoss']:
 with (R/(mode+'.log')).open('x') as log:
  p=subprocess.run([PY,str(R/'benchmark.py'),mode],stdout=log,stderr=subprocess.STDOUT)
 (R/(mode+'_EXIT.json')).write_text(json.dumps(dict(time=time.time(),returncode=p.returncode)))
 if mode=='prepare' and p.returncode:break
 time.sleep(5)
(R/'CONTROLLER_FINISHED.json').write_text(json.dumps(dict(time=time.time())))
