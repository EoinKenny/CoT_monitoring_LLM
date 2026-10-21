from common import *
import sys,subprocess,fcntl,traceback
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
def run(script,*args):
 reserve();name='_'.join([script]+list(args));write(ROOT/'STATUS.json',dict(stage=name,state='running',time=time.time(),pid=os.getpid()))
 with open(ROOT/'logs'/(name+'.log'),'x') as f:subprocess.run([GPU,str(ROOT/'code'/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(ROOT/'logs').mkdir(exist_ok=True)
 run('controlled.py')
 for j in JUDGES:run('run_judge.py',j,'controlled')
 write(ROOT/'ALL_CONTROLLED_SMOKES_PASSED.json',dict(judges=JUDGES,time=time.time()))
 run('prepare.py')
 for j in JUDGES:run('run_judge.py',j,'natural')
 write(ROOT/'COLLECTION_COMPLETED.json',dict(time=time.time(),note='Human validation remains pending; controlled synthetic diagnostics are separate from natural examples.'))
if __name__=='__main__':
 try:main()
 except BaseException:write(ROOT/'FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
