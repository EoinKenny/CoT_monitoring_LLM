from pathlib import Path
import fcntl,subprocess,sys,time,json,os
R=Path(__file__).resolve().parent
lock=open(R/'evaluation.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (R/'results/EVALUATION_COMPLETED.json').exists():sys.exit()
(R/'EVALUATION_LAUNCH.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))
while not (R/'TRAINING_COMPLETED.json').exists():
 if (R/'FAILED.json').exists():raise RuntimeError('Training failed; evaluation not started')
 time.sleep(15)
with open(R/'logs/evaluate.log','a') as out:
 subprocess.run([sys.executable,'-u',str(R/'evaluate.py')],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=R)
