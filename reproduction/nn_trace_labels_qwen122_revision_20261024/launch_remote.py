import os,json,time,subprocess,fcntl
from pathlib import Path
R=Path('/home/ubuntu/experiments/nn-trace-labels-qwen122-revision-20261024')
with (R/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'LAUNCH.json').exists(),'Existing launch receipt; inspect before any restart'
 log=(R/'run.log').open('ab')
 p=subprocess.Popen(['/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python','-u',str(R/'run.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=p.pid,time=time.time(),experiment=str(R));(R/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
