from common import *
import subprocess,fcntl,traceback

def main():
 assert (ROOT/'PREPARED.json').exists();lock=(ROOT/'smoke.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 results={}
 for j in JS:
  D=ROOT/'smoke'/j;log=ROOT/'logs'/('smoke_'+j+'.log')
  assert not (D/'run').exists(), 'Smoke attempt exists; inspect rather than silently repeat'
  write(ROOT/'SMOKE_STATUS.json',dict(state='running',judge=j,time=time.time()))
  with log.open('x') as f:
   p=subprocess.Popen([GPU,'-u',str(ROOT/'code/judge.py'),str(D)],cwd=ROOT/'code',stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),HF_HUB_OFFLINE='1',PYTHONHASHSEED='20261004'));write(ROOT/'SMOKE_CHILD.json',dict(pid=p.pid,judge=j));rc=p.wait()
  assert rc==0,f'{j} smoke failed; preserve all attempts'
  d=read(D/'run/COMPLETED');assert sha(D/'run/judgments.jsonl')==d['judgments_sha256'];rr=list(jsonlines(D/'run/judgments.jsonl'));assert len(rr)==96
  by={}
  for c in CS:
   r=[x for x in rr if x['condition']==c];assert len(r)==32
   batches={x['batch_number']:x['batch_seconds'] for x in r};secs=sum(batches.values());tokens=sum(x['input_tokens'] for x in r);fulltokens=sum(read(ROOT/b/'judges'/f'{j}_{c}'/'approved_config.json')['total_input_tokens'] for b in BBS)
   by[c]=dict(n=32,batch_seconds=secs,input_tokens=tokens,full_input_tokens=fulltokens,estimated_full_seconds=secs*fulltokens/tokens,score_range=[min(x['positive_probability'] for x in r),max(x['positive_probability'] for x in r)])
  results[j]=dict(conditions=by,seconds_including_load=d['seconds_this_session']);write(ROOT/'SMOKE_RESULTS.json',results)
 estimate=sum(v['estimated_full_seconds'] for j in results.values() for v in j['conditions'].values())+30*60
 write(ROOT/'SMOKE_COMPLETED.json',dict(time=time.time(),results=results,estimated_hours=estimate/3600,eta_range_hours=[estimate/3600,estimate*1.5/3600],note='Token-volume scaled stratified-length smoke benchmark plus 60s loading per cell; 50% upper margin for batching and dataset variation. Evaluation extra.'))
 write(ROOT/'SMOKE_STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:write(ROOT/'SMOKE_STATUS.json',dict(state='failed',error=str(e),traceback=traceback.format_exc()));raise
