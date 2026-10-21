from common import *
import sys,subprocess,traceback,fcntl

def stage(name,python,args):
 reserve();write(ROOT/'STATUS.json',dict(state='running',stage=name,time=time.time()))
 with (ROOT/'logs'/f'{name}.log').open('a') as log:
  p=subprocess.Popen([python,str(ROOT/'code'/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,EXPANSION_ROOT=str(ROOT),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8'))
  write(ROOT/'CHILD.json',dict(stage=name,pid=p.pid,time=time.time()));assert p.wait()==0,f'{name} failed, see logs'
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'COMPLETED.json').exists();(ROOT/'logs').mkdir(exist_ok=True)
 for path,h in read(ROOT/'CODE_FROZEN.json').items():assert sha(ROOT/path)==h,path
 while not (ROOT/'SETUP_COMPLETED.json').exists():
  pid=read(ROOT/'SETUP_PID.json')['pid'];os.kill(pid,0)
  state=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0];assert state!='Z','Retrieval failed; inspect setup.log'
  reserve();write(ROOT/'STATUS.json',dict(state='retrieving',time=time.time()));time.sleep(30)
 setup=read(ROOT/'SETUP_COMPLETED.json');assert setup['canonical_sha256']==sha(ROOT/'canonical/cases.jsonl')
 for path,h in setup['retrieval_hashes'].items():assert sha(ROOT/path)==h
 stage('contract_tests',CPU,['test_contract.py'])
 # Check longer prompt lengths for all models before any new inference.
 for j in ALL:stage('prepare_'+j,GPU,['prepare_judge.py',j])
 for b in BBS:
  for j in ALL:stage('parent_replay_'+b+'_'+j,CPU,['test_evaluation.py',b,j])
 frozen={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'PROTOCOL.json',ROOT/'SETUP_COMPLETED.json',ROOT/'judge_specs.json',*sorted((ROOT/'prepared').glob('*.json'))]};write(ROOT/'INPUTS_FROZEN.json',frozen)
 for j in ALL:
  stage('judge_'+j,GPU,['run_judge.py',j])
  for b in BBS:
   stage('evaluate_'+b+'_'+j,CPU,['evaluate.py',b,j]);stage('diagnostics_'+b+'_'+j,CPU,['diagnostics.py',b,j])
 stage('aggregate',CPU,['aggregate.py']);assert read(ROOT/'COMPLETED.json')['new_judgments']==300000
 write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  x=dict(state='failed',time=time.time(),traceback=traceback.format_exc());write(ROOT/f'FAILED_{time.time_ns()}.json',x);write(ROOT/'STATUS.json',x);raise
