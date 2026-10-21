from common import *
import fcntl,subprocess,signal,traceback

def locked(name='controller.lock'):
 with (ROOT/name).open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
  except BlockingIOError:return True

def cmd(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return ''

def main():
 guard=(ROOT/'supervisor.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=read(ROOT/'RECOVERY_V3_LAUNCH.json');pid=launch['pid']
 while locked():
  reserve()
  state=read(ROOT/'STATUS.json');stage=state.get('stage','');j=stage.removeprefix('full_')
  status=ROOT/'models'/j/'STATUS.json'
  last=max(launch['time'],status.stat().st_mtime if status.exists() else (ROOT/'STATUS.json').stat().st_mtime)
  write(ROOT/'SUPERVISOR_STATUS.json',dict(state='watching_existing_controller',pid=pid,last_progress=last,time=time.time()))
  if time.time()-last>1800:
   command=cmd(pid)
   if str(ROOT/'code/controller_v3.py') not in command or os.getpgid(pid)!=pid:raise RuntimeError('Cannot verify stalled controller ownership; manual inspection required')
   write(ROOT/f'STALL_RECOVERY_{time.time_ns()}.json',dict(pid=pid,command=command,time=time.time(),last_progress=last))
   os.killpg(pid,signal.SIGTERM);time.sleep(30)
   if cmd(pid):os.killpg(pid,signal.SIGKILL)
  time.sleep(30)
 # A crashed controller may leave its worker alive; never overlap GPU workers.
 while locked('gpu.lock'):
  child=read(ROOT/'CHILD.json');cpid=child['pid'];command=cmd(cpid)
  if str(ROOT/'code/run_judge_v3.py') not in command or os.getpgid(cpid)!=pid:
   raise RuntimeError('GPU lock owned by an unverified process; no new worker launched')
  newest=max((ROOT/'models'/j/'STATUS.json').stat().st_mtime for j in ALL)
  write(ROOT/'SUPERVISOR_STATUS.json',dict(state='waiting_for_existing_worker',pid=cpid,time=time.time()))
  if time.time()-newest>1800:
   write(ROOT/f'ORPHAN_STALL_{time.time_ns()}.json',dict(pid=cpid,pgid=pid,command=command,time=time.time()))
   os.killpg(pid,signal.SIGTERM);time.sleep(30)
   if cmd(cpid):os.killpg(pid,signal.SIGKILL)
  time.sleep(30)
 if (ROOT/'COMPLETED.json').exists():
  write(ROOT/'SUPERVISOR_STATUS.json',dict(state='already_completed',time=time.time()));return
 reserve()
 with (ROOT/'logs'/'controller_v4.log').open('a') as f:
  p=subprocess.Popen([CPU,str(ROOT/'code/controller_v4.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,EXPANSION_ROOT=str(ROOT)))
  write(ROOT/'RECOVERY_V4_LAUNCH.json',dict(pid=p.pid,time=time.time()))
  write(ROOT/'SUPERVISOR_STATUS.json',dict(state='v4_controller_running',pid=p.pid,time=time.time()))
  rc=p.wait()
 write(ROOT/'SUPERVISOR_STATUS.json',dict(state='v4_controller_exited',returncode=rc,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(ROOT/f'SUPERVISOR_FAILED_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
