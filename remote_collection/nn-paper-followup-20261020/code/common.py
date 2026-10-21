from pathlib import Path
import json,hashlib,os,time,shutil
ROOT=Path(os.environ.get('FOLLOWUP_ROOT','/home/ubuntu/experiments/nn-paper-followup-20261020'))
PARENT=ROOT.parent/'probability-500-20261012'
HARD=ROOT.parent/'full-threshold-sweep-20261009'
BB=['qwen','llama70']; DS=['aegis','beavertails','toxicchat','toxigen','fava','halueval','ragtruth']; JUDGES=['qwen122','gemma','glm']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');
 with open(t,'w') as f:json.dump(d,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 t.replace(p)
def reserve():
 for p in [ROOT,Path('/opt/dlami/nvme')]:
  assert shutil.disk_usage(p).free>=20*1024**3, f'Disk reserve: {p}'
def jl(p):
 with open(p) as f:
  for l in f:yield json.loads(l)
def savejl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();t=p.with_name(p.name+'.tmp');assert not t.exists()
 with open(t,'w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
  f.flush();os.fsync(f.fileno())
 t.replace(p)
