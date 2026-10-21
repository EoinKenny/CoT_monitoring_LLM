from pathlib import Path
import os,json,hashlib,time,gzip,shutil,fcntl
ROOT=Path('/home/ubuntu/experiments/nn-mechanism-labels-20261018')
PARENT=Path('/home/ubuntu/experiments/nn-reason-after-20261016')
AFTER=Path('/home/ubuntu/experiments/nn-reason-flips-20261017')
FIRST=Path('/home/ubuntu/experiments/nn-reason-first-512-20261017')
JUDGES=['qwen','gemma','mistral'];BBS=['llama','qwen','gemma','llama70','qwen122']
GPU='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python'
CASE_FIELDS=['target_ambiguity','decision_relevant_similarity','label_convention_cue','decision_relevant_contrast','potentially_misleading_overlap','reference_quality_concern','insufficient_context']
RATIONALE_FIELDS=['uses_analogy','uses_contrast','uses_reference_label','uses_probe_prediction','uses_independent_task_evidence','unsupported_transfer','insufficient_explanation']
SCHEMA={'type':'object','properties':{**{k:{'type':'string','enum':['yes','no','uncertain']} for k in CASE_FIELDS+RATIONALE_FIELDS},'evidence':{'type':'string','maxLength':240}},'required':CASE_FIELDS+RATIONALE_FIELDS+['evidence'],'additionalProperties':False}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 with open(t,'w') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
 t.replace(p)
def jl(p):
 with open(p) as f:
  for line in f:yield json.loads(line)
def reserve():
 for p in [ROOT,Path('/opt/dlami/nvme')]:
  if shutil.disk_usage(p).free<20*2**30:raise RuntimeError('Disk below20GiB reserve')
