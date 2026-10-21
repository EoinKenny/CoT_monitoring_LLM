import os,json,hashlib,shutil,time
from pathlib import Path
ROOT=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009')
PARENT=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
BBS=['llama','qwen','gemma','llama70','qwen122'];JS=['qwen','gemma'];CS=['baseline','saliency','nearest']
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def reserve():assert shutil.disk_usage(ROOT).free>=20*2**30,'Below 20 GiB disk reserve'
def jsonlines(p):
 with Path(p).open() as f:
  for line in f:yield json.loads(line)
def lines(p,rows):
 p=Path(p);t=p.with_name(p.name+'.tmp');p.parent.mkdir(parents=True,exist_ok=True)
 with t.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(t,p)
