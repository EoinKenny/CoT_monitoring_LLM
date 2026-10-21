"""Explicit recovery queue; no automatic retry of any launched run."""
import importlib.util,os,sys,time,json,fcntl,subprocess,signal,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_queue',R/'queue.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def main():
 lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=q.read(R/'QUEUE_PLAN.json');q.verify(Path(plan['parent']))
 for location in plan['children']:
  child=Path(location)
  if (child/'COMPLETED.json').exists():q.verify(child);continue
  assert not (child/'RECOVERY_LAUNCH_ATTEMPT.json').exists(),'Manual inspection required; no automatic relaunch'
  assert q.sha(child/'run_v2.py')==q.read(child/'PARSER_REPAIR.json')['runner_sha256']
  prior=set(child.glob('FAILED_*.json'))
  q.write(child/'RECOVERY_LAUNCH_ATTEMPT.json',dict(time=time.time(),controller_pid=os.getpid(),repair_sha256=q.sha(child/'PARSER_REPAIR.json')))
  with (child/'run_v2.log').open('ab') as log:
   p=subprocess.Popen([q.PY,'-u',str(child/'run_v2.py'),'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  q.write(child/'RECOVERY_LAUNCH.json',dict(pid=p.pid,time=time.time(),controller_pid=os.getpid()))
  while p.poll() is None:
   q.write(R/'STATUS_V2.json',dict(state='running_child',child=location,pid=p.pid,time=time.time()))
   if set(child.glob('FAILED_*.json'))-prior:
    # Model engine may otherwise remain alive after an exception. This process
    # group was created here exclusively for this child, not for keepalives.
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise RuntimeError('Child failure preserved, engine cleaned up; no retries: '+location)
   time.sleep(15)
  if p.returncode!=0 or not (child/'COMPLETED.json').exists():raise RuntimeError('Child stopped without completion: '+location)
  q.verify(child)
 q.write(R/'COMPLETED_V2.json',dict(time=time.time(),judges=3,total_annotations=75000))
 q.write(R/'STATUS_V2.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  q.write(R/'FAILED_V2.json',dict(time=time.time(),traceback=traceback.format_exc()))
  q.write(R/'STATUS_V2.json',dict(state='failed',time=time.time()))
  raise
