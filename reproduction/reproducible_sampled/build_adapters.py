"""Freeze minimally adapted, established preparation/explanation/prompt implementations."""
from common import *
import shutil
C=ROOT/'code';raw=ROOT/'prepared';raw.mkdir(exist_ok=True)
p=OLD['llama']/'prepare.py';s=p.read_text();s=s.replace("R=Path('/home/ubuntu/experiments/ten-dataset-full-20261002')",f'R=Path({str(raw)!r})')
s=s.replace("quota=min(len(pools[s,0]),len(pools[s,1])) if s=='train' else None","quota=None").replace("selected+=(bins[0][:n]+bins[1][:n]) if s=='train' else bins[0]+bins[1]","selected+=bins[0]+bins[1]")
# Preserve completed per-dataset preparation on restart; never alter source adapter rules.
s=s.replace("for ds,rows in D.items():","for ds,rows in D.items():\n if (R/(ds+'_selected.json')).exists():continue")
(C/'prepare_full.py').write_text(s)
for b in OLD:
 R=ROOT/b;R.mkdir(exist_ok=True)
 if b=='llama':
  bb=dict(repo='meta-llama/Llama-3.2-1B-Instruct',revision='9213176726f574b556790deb65791e0c5aa438b6',path='/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',block_index=10)
 else:bb=read(OLD[b]/'backbone.json')
 write(R/'backbone.json',bb);shutil.copyfile(OLD['llama']/'judge_rubrics.json',R/'judge_rubrics.json')
 if b in ['qwen','gemma']:
  shutil.copyfile(OLD[b]/'tokenize_rows.py',R/'tokenize_rows.py')
 if b=='gemma':
  shutil.copyfile(OLD[b]/'render_content.py',R/'render_content.py')
  s=(OLD[b]/'backbone_loader.py').read_text().replace("R=Path('/home/ubuntu/experiments/gemma31b-backbone-20261003')",f'R=Path({str(R)!r})');(R/'backbone_loader.py').write_text(s)
 s=(OLD[b]/'explanations.py').read_text()
 start=s.index('results=json.loads');end=s.index("(R/'private.json').write_text",start)
 s=s[:start]+"results=json.loads((B/'results.json').read_text());rows=json.loads((B/'selected_flags.json').read_text())\n"+s[end:]
 # Route all experiment writes to the new root; legacy mask function remains a read-only import.
 s=s.replace("B=Path('/home/ubuntu/experiments/ten-dataset-full-20261002')",f'B=Path({str(R)!r})').replace("B=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002')",f'B=Path({str(R)!r})')
 s=s.replace("B.parent/'logistic-probes-20261001/train.py'",f"Path({str(E/'logistic-probes-20261001/train.py')!r})")
 # Equal similarities choose the smallest immutable identity, not mutable array position.
 s=s.replace("trainix=np.array([i for i,r in enumerate(meta) if r['split']=='train'])","trainix=np.array(sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id']))")
 s=s.replace('T/=np.linalg.norm(T,axis=1)[:,None]','T/=np.maximum(np.linalg.norm(T,axis=1)[:,None],1e-30)').replace('Z/=np.linalg.norm(Z,axis=1)[:,None]','Z/=np.maximum(np.linalg.norm(Z,axis=1)[:,None],1e-30)')
 # Bounded-memory exact cosine search; stable first maximum over identity-sorted training rows.
 s=s.replace('sim=Z@T.T;best=sim.argmax(axis=1)', '''best=np.zeros(len(group),dtype=int);bestscore=np.full(len(group),-np.inf)
  for ts in range(0,len(T),8192):
   sim=Z@T[ts:ts+8192].T;local=sim.argmax(axis=1);values=sim[np.arange(len(group)),local];better=values>bestscore;best[better]=ts+local[better];bestscore[better]=values[better]''').replace('float(sim[k,best[k]])','float(bestscore[k])')
 # Resume only committed explanations; an incomplete append requires explicit inspection.
 s=s.replace("assert not (R/'explanations.jsonl').exists()", "assert not (R/'explanations.jsonl').exists(), 'Partial explanations require inspection'")
 s=s.replace('ties smallest index','ties immutable training ID').replace('for name in results:', 'for name in sorted(results):')
 (R/'explanations.py').write_text(s)
 s=(OLD['gemma']/'prepare_judges.py').read_text().replace("protocol='Full balanced probe training; frozen validation-F1 threshold; positive-only judge; native template'","protocol='Full eligible training with balanced class weights; validation-F1 threshold; reproducible TP/FP stratified flags; native template'")
 s=s.replace("max_input_tokens=max(lengths)","max_input_tokens=max(lengths,default=0)")
 (R/'prepare_judges.py').write_text(s)
write(ROOT/'ADAPTER_SOURCE_MANIFEST.json',dict(original_sources={str(p):sha(p) for b in OLD for p in OLD[b].glob('*.py')},adapted={str(p):sha(p) for b in OLD for p in (ROOT/b).glob('*.py')},preparation_sha256=sha(C/'prepare_full.py')))
