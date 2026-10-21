import subprocess,os
from common import *
assert not (ROOT/'SUPERVISOR_PID.json').exists(),'Existing launch: inspect state before explicit resume'
assert (ROOT/'smoke/qwen_baseline/run/COMPLETED').exists()
assert list((ROOT/'smoke/qwen_baseline/run').glob('RESUME_VERIFIED_*.json'))
env=os.environ.copy();env.update(STUDY_ROOT=str(ROOT),PYTHONPATH=str(ROOT/'code'))
with (ROOT/'supervisor.log').open('x') as f:
 p=subprocess.Popen([CPU,'-u',str(ROOT/'code/supervise.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
write(ROOT/'LAUNCH.json',dict(pid=p.pid,time=time.time(),root=str(ROOT),protocol='full eligible training; up to 500 reproducible TP/FP-stratified flags per backbone/dataset'))
print(json.dumps(read(ROOT/'LAUNCH.json')))
