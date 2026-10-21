"""Prepare two independent annotation runs from immutable parent inputs; CPU only."""
import json,hashlib,shutil,time,ast
from pathlib import Path
R=Path(__file__).resolve().parent
P=R.parent/'nn-mechanism-two-labels-20261018'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2))
prep=read(P/'PREPARED.json')
script=(P/'study.py').read_text()
assert sha(P/'study.py')==prep['script_sha256']
script=script.replace('Frozen two-label Qwen annotation study','Additional local-judge two-label annotation study')
script=script.replace("range(0,len(units),4)","range(0,len(units),cfg['batch_size'])")
script=script.replace('chunk=units[lo:lo+4]',"chunk=units[lo:lo+cfg['batch_size']]")
script=script.replace('if len(allresults)==4:',"if len(allresults)==cfg['batch_size']:")
script=script.replace("<=262144","<=32768")
script=script.replace("(R/'batches').mkdir(exist_ok=True)","(R/'batches').mkdir(exist_ok=True)")
needle="  if len(allresults)==cfg['batch_size']:write(R/'FULL_LAUNCHED.json'"
assert needle in script
script=script.replace(needle,"  if batch==0:\n   write(R/'FIRST_BATCH_CHECK.json',dict(n=len(rr),valid=sum(r['status']=='valid' for r in rr),statuses=dict(counts),time=time.time()))\n   if not any(r['status']=='valid' for r in rr):raise RuntimeError('First production batch has no valid JSON; retained without retry')\n"+needle)
ast.parse(script)
children=[]
for judge in ['gemma','qwen']:
 child=R.parent/('nn-mechanism-two-labels-'+judge+'-20261018')
 assert not child.exists(),str(child)+' already exists'
 child.mkdir()
 for name,key in [('units.jsonl','units_sha256'),('mapping_PRIVATE.jsonl','mapping_sha256'),('prompt_template.txt','template_sha256')]:
  assert sha(P/name)==prep[key]
  shutil.copyfile(P/name,child/name)
  assert sha(child/name)==prep[key]
 cfg0=read(R.parent/'nn-reason-flips-20261017'/'models'/judge/'parent_config.json')
 assert Path(cfg0['engine']['model']).is_dir()
 cfg={k:cfg0[k] for k in ['judge','model','revision','engine','batch_size','versions']}
 cfg.update(sampling=dict(temperature=0,seed=20261001,max_tokens=128,constrained_decoding=False),native_thinking=False,prompt_template_sha256=prep['template_sha256'])
 write(child/'SMOKE_CONFIG.json',cfg)
 proto=read(P/'PROTOCOL.json')
 proto['decoding']='Local '+cfg['model']+' revision '+cfg['revision']+'; established engine, batch16, native thinking disabled, temperature0, seed20261001, max_tokens128, unconstrained generation. First production batch retained as format preflight. No automatic regeneration.'
 proto['replication']='Same complete ordered 25000 test contexts as Qwen122, identical prompt bytes, no access to other annotations. Additional judges selected by measured local classification performance; that ranking does not guarantee annotation accuracy.'
 write(child/'PROTOCOL.json',proto)
 (child/'study.py').write_text(script)
 cp=dict(prep,protocol_sha256=sha(child/'PROTOCOL.json'),script_sha256=sha(child/'study.py'),parent_root=str(P),parent_prepared_sha256=sha(P/'PREPARED.json'),config_sha256=sha(child/'SMOKE_CONFIG.json'),time=time.time())
 write(child/'PREPARED.json',cp)
 write(child/'STATUS.json',dict(state='queued',total=25000,completed=0,time=time.time()))
 children.append(str(child))
write(R/'QUEUE_PLAN.json',dict(parent=str(P),parent_pid=3757849,children=children,per_judge_cases=25000,total_annotations_including_parent=75000,selection='Gemma31B and Qwen27B: strongest remaining local judges in existing controlled-threshold evaluation.',script_sha256=sha(R/'queue.py'),time=time.time()))
print(json.dumps(read(R/'QUEUE_PLAN.json'),indent=2))
