"""Single durable sequential queue; never resubmit a launched child."""
import json,hashlib,time,os,fcntl,subprocess,traceback,shutil
from pathlib import Path
R=Path(__file__).resolve().parent
PY='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 t.replace(p)
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def verify(root):
 c=read(root/'COMPLETED.json');assert c['n_units']==c['n_cases']==25000
 for name,key in [('annotations.jsonl','annotations_sha256'),('labels_by_case.csv','case_labels_sha256'),('OUTPUT_SHA256.json','output_manifest_sha256'),('PREPARED.json','prepared_sha256')]:assert sha(root/name)==c[key],name
 for name,h in read(root/'OUTPUT_SHA256.json').items():assert sha(root/name)==h,name
 expected=[json.loads(x)['unit_id'] for x in (root/'units.jsonl').open()]
 actual=[json.loads(x)['unit_id'] for x in (root/'annotations.jsonl').open()]
 assert len(actual)==len(set(actual))==25000 and actual==expected
 write(R/(root.name+'_VERIFIED.json'),dict(root=str(root),completed_sha256=sha(root/'COMPLETED.json'),status_counts=c['status_counts'],time=time.time()))
def main():
 lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=read(R/'QUEUE_PLAN.json');assert sha(Path(__file__))==plan['script_sha256']
 parent=Path(plan['parent'])
 while not (parent/'COMPLETED.json').exists():
  if list(parent.glob('FAILED_*.json')) or not alive(plan['parent_pid']):raise RuntimeError('Parent stopped without completion; no children launched')
  write(R/'STATUS.json',dict(state='waiting_for_qwen122',parent_status=read(parent/'STATUS.json'),time=time.time()))
  time.sleep(60)
 verify(parent)
 # Wait for the parent's inference process to release its model and GPU locks.
 while alive(plan['parent_pid']):time.sleep(15)
 for location in plan['children']:
  child=Path(location)
  if (child/'COMPLETED.json').exists():verify(child);continue
  if (child/'LAUNCH_ATTEMPT.json').exists():raise RuntimeError('Previous child launch recorded; inspect manually, never auto-resubmit: '+location)
  assert sha(child/'study.py')==read(child/'PREPARED.json')['script_sha256']
  assert sha(child/'SMOKE_CONFIG.json')==read(child/'PREPARED.json')['config_sha256']
  assert shutil.disk_usage(child).free>=20*2**30
  write(child/'LAUNCH_ATTEMPT.json',dict(time=time.time(),controller_pid=os.getpid()))
  with (child/'run.log').open('ab') as log:
   proc=subprocess.Popen([PY,'-u',str(child/'study.py'),'run'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(child/'LAUNCH.json',dict(pid=proc.pid,time=time.time(),controller_pid=os.getpid()))
  write(R/'STATUS.json',dict(state='running_child',child=location,pid=proc.pid,time=time.time()))
  code=proc.wait()
  if code!=0 or not (child/'COMPLETED.json').exists():raise RuntimeError('Child stopped: '+location+' exit='+str(code))
  verify(child)
 write(R/'COMPLETED.json',dict(time=time.time(),judges=3,total_annotations=75000,children=plan['children']))
 write(R/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(R/'FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()))
  write(R/'STATUS.json',dict(state='failed',time=time.time()))
  raise
