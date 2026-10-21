"""Two-neighbour study: shared storage and integrity helpers."""
from pathlib import Path
import hashlib,json,os,shutil,time
ROOT=Path(os.environ.get('EXPANSION_ROOT','/home/ubuntu/experiments/nn-reason-after-20261016'))
HARD=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
PROB=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009')
BBS=['llama','qwen','gemma','llama70','qwen122']
CS=['nearest_reason_after'];MODES=['score']
NEW=['qwen','gemma','mistral','olmo','glm','qwen122'];ALL=NEW
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def jl(p):
 with Path(p).open() as f:
  for line in f:yield json.loads(line)
def reserve():
 for p in [ROOT,Path('/opt/dlami/nvme')]:
  if shutil.disk_usage(p).free<20*2**30:raise RuntimeError(f'Free disk below 20 GiB reserve: {p}')
def cell(mode,b,j,c):return ROOT/mode/b/'judges'/f'{j}_{c}'
def specs():return read(ROOT/'judge_specs.json')['models']
def cells(j):
 for b in BBS:
  for c in CS:
   for mode in MODES:yield mode,b,c,cell(mode,b,j,c)

PARENT=Path("/home/ubuntu/experiments/probability-500-20261012")
TOTAL=50000

MAX_OUTPUT=256
REGEX=r"(?:0|[1-9][0-9]?|100)\n[^\n][\s\S]*"
