"""Preserve additional research folders before retirement; do not stop source jobs."""
import pathlib,subprocess,json,time,os,shutil,shlex,traceback,fcntl
R=pathlib.Path('/home/ubuntu/g6e-retirement-archive-20261008')
M=pathlib.Path('/home/ubuntu/project-migration-20261007')
R.mkdir(exist_ok=True)
def write(name,d):
 p=R/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));os.replace(t,p)
def main():
 lock=(R/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ssh=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o',f'UserKnownHostsFile={M}/known_hosts','-o','StrictHostKeyChecking=yes','-i',str(M/'transfer_key')])
 for folder in ['llm-from-scratch','kmeans-llm','nltk_data','keepalive']:
  assert shutil.disk_usage(R).free>130*2**30
  for stage in ['copy','verify']:
   write('STATUS.json',dict(state=stage,folder=folder,pid=os.getpid(),time=time.time()))
   args=['rsync','-aH','--no-owner','--no-group','--partial','--timeout=1800','--stats','-e',ssh]
   if stage=='verify':args+=['--checksum','--dry-run','--itemize-changes','--out-format=CHECK %i %n%L']
   else:args+=['--info=progress2']
   args+=['ubuntu@10.0.14.197:'+folder+'/',str(R/folder)+'/']
   log=R/f'{folder}_{stage}.log'
   with log.open('w') as fh:subprocess.run(args,stdout=fh,stderr=subprocess.STDOUT,check=True)
   if stage=='verify':
    changes=[line for line in log.read_text().splitlines() if line.startswith('CHECK ') and not line.startswith('CHECK .d')]
    if changes:raise RuntimeError(f'{folder}: source/destination differences remain; inspect {log}; source keepalive may still be writing')
   write(f'{folder}_{stage}_COMPLETED.json',dict(time=time.time(),source=folder,destination=str(R/folder)))
 write('COMPLETED.json',dict(time=time.time(),folders=['llm-from-scratch','kmeans-llm','nltk_data','keepalive'],source_preserved=True))
 write('STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(f'FAILED_{time.time_ns()}.json',dict(error=str(e),traceback=traceback.format_exc()));write('STATUS.json',dict(state='failed',error=str(e),time=time.time()));raise
