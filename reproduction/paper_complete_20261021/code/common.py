from pathlib import Path
import json,hashlib,time,os,shutil
ROOT=Path(os.environ.get('COMPLETE_ROOT','/home/ubuntu/experiments/paper-complete-20261021'));BASE=ROOT.parent;PARENT=BASE/'probability-500-20261012';HARD=BASE/'full-threshold-sweep-20261009'
BBS=['llama','qwen','gemma','llama70','qwen122'];JUDGES=['glm','mistral','olmo','qwen','gemma','qwen122']
CS=['text_label','text_prediction','text_only','label_only','prediction_only','label_prediction','random_full']
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
def read(p):return json.loads(Path(p).read_text())
def jl(p):
 with Path(p).open() as f:
  for l in f:yield json.loads(l)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 t.replace(p)
def savejl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();t=p.with_name(p.name+'.tmp');assert not t.exists()
 with t.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 t.replace(p)
def reserve():
 assert shutil.disk_usage(ROOT).free>20*2**30,'Below 20 GiB reserve'
