from common import *
from ablation_transform import transform,random_index
import numpy as np,re,collections
EXISTING={('llama70','qwen122'):'nn-ablation-20261013',('qwen','qwen'):'nn-ablation-qwen-qwen-20261013',('gemma','gemma'):'nn-ablation-gemma-gemma-20261013',('llama','qwen'):'nn-ablation-llama-qwen-20261013',('gemma','qwen122'):'nn-ablation-gemma-qwen122-20261013'}
def main():
 reserve();assert not (ROOT/'PREPARED.json').exists();proofs=[];sources={};paths={}
 for (b,j),name in EXISTING.items():
  d=BASE/name;done=read(d/'models'/j/'COMPLETED.json');assert done['n']==70000
  assert sha(d/'canonical/cases.jsonl')==read(d/'SETUP_COMPLETED.json')['canonical_sha256']
  for c in CS:
   run=d/'score'/b/'judges'/f'{j}_{c}'/'run';cm=read(run/'COMPLETED');assert cm['n']==10000 and sha(run/'judgments.jsonl')==cm['judgments_sha256'];proofs.append(dict(backbone=b,judge=j,condition=c,path=str(run),sha256=cm['judgments_sha256']))
  paths.setdefault(b,d/'canonical/cases.jsonl');sources[str(d/'canonical/cases.jsonl')]=sha(d/'canonical/cases.jsonl')
 parent=PARENT/'canonical/cases.jsonl';sources[str(parent)]=sha(parent)
 # Existing prompts stay byte-identical; create only the previously absent Qwen122 backbone ablations.
 canonical={r['sample_id']:r for r in jl(parent) if r['backbone']=='qwen122' and r['condition']=='nearest'};selected=read(PARENT/'score/qwen122/selected_flags.json');assert list(canonical)==[r['sample_id'] for r in selected]
 refs={};names={};private=[]
 for ds in sorted({r['dataset'] for r in selected}):
  names[ds]={}
  for r in canonical.values():
   if r['dataset']==ds:
    for k in ['Dataset label','Probe prediction']:
     m=re.search(r'\n'+k+r': (.+) \(([01])\)',r['messages'][1]['content']);assert m;names[ds][int(m[2])]=m[1]
  assert set(names[ds])=={0,1}
  src=HARD/'qwen122';meta=read(src/(ds+'_metadata.json'));z=np.load(src/(ds+'_predictions.npz'));proof=read((src/(ds+'_probe.joblib')).resolve().parent/(ds+'_TRAIN_COMPLETED.json'))
  for suf in ['metadata.json','predictions.npz','probe.joblib']:
   f=src/(ds+'_'+suf);h=sha(f);assert h==proof['sha256'][f.name];sources[str(f)]=h
  assert np.array_equal(z['labels'],[m['label'] for m in meta]) and np.array_equal(z['splits'],[m['split'] for m in meta]) and np.isfinite(z['probabilities']).all()
  threshold=proof['result']['threshold'];pool=sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id']);ts=[r for r in selected if r['dataset']==ds];assert not {str(meta[i]['group']) for i in pool}&{str(r['group']) for r in ts}
  for r in ts:
   assert meta[r['index']]['immutable_id']==r['sample_id'];k,h,counter=random_index(ds,r['sample_id'],len(pool));i=pool[k];n=meta[i];pred=int(z['probabilities'][i]>=threshold);refs[r['sample_id']]=dict(messages=n['messages'],label=int(n['label']),prediction=pred);private.append(dict(sample_id=r['sample_id'],reference_id=n['immutable_id'],hash=h,counter=counter,threshold=threshold))
 generated=[]
 for c in CS:
  for t in selected:
   r=canonical[t['sample_id']];msgs=transform(r['messages'],c,refs[t['sample_id']],names[r['dataset']]);mh=digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')));pid='nnabl_'+digest(json.dumps([r['sample_id'],c,mh],separators=(',',':')));generated.append(dict(sample_id=r['sample_id'],prompt_id=pid,backbone='qwen122',dataset=r['dataset'],split=r['split'],condition=c,mode='score',messages=msgs,messages_sha256=mh,parent_prompt_id=r['prompt_id']))
 path=ROOT/'canonical/qwen122.jsonl';savejl(path,generated);paths['qwen122']=path;savejl(ROOT/'private/qwen122_random_selection.jsonl',private)
 # Freeze a per-backbone source and exact ordering, checking all existing/new condition coverage.
 specs=read(PARENT/'judge_specs.json');write(ROOT/'judge_specs.json',specs);tasks=[]
 for b in BBS:
  path=paths[b];rr=list(jl(path));expected=[r['sample_id'] for r in read(PARENT/'score'/b/'selected_flags.json')];assert len(rr)==len({r['prompt_id'] for r in rr})==70000
  for c in CS:assert [r['sample_id'] for r in rr if r['condition']==c]==expected
  if b!='qwen122':(ROOT/'canonical'/f'{b}.jsonl').symlink_to(path)
  for j in JUDGES:
   if (b,j) not in EXISTING:tasks.append(dict(study='ablation',backbone=b,judge=j,path=str(ROOT/'canonical'/f'{b}.jsonl'),sha256=sha(path),n=70000))
 units=BASE/'nn-mechanism-two-labels-20261018/units.jsonl';assert sha(units)==read(units.parent/'PREPARED.json')['units_sha256'];rr=[dict(prompt_id=r['unit_id'],backbone=r['backbone'],dataset=r['dataset'],sample_id=r['unit_id'],split='test',condition='mechanism_labels',messages=[dict(role='user',content=r['prompt'])]) for r in jl(units)];assert len(rr)==25000;savejl(ROOT/'canonical/mechanism.jsonl',rr)
 for j in ['glm','mistral','olmo']:tasks.append(dict(study='mechanism',backbone='all',judge=j,path=str(ROOT/'canonical/mechanism.jsonl'),sha256=sha(ROOT/'canonical/mechanism.jsonl'),n=25000))
 # Verify completed companion studies; preserve capped/ambiguous observations, never re-issue them.
 companion={}
 for name in ['nn-reason-flips-20261017','nn-reason-first-512-20261017','two-nn-20261014']:
  companion[name]=read(BASE/name/'COMPLETED.json')
 assert companion['nn-reason-flips-20261017']['accounted']==35919 and companion['nn-reason-first-512-20261017']['accounted']==35919
 tasks.sort(key=lambda t:(JUDGES.index(t['judge']),t['study']!='ablation',BBS.index(t['backbone']) if t['backbone'] in BBS else 99))
 write(ROOT/'REUSED_COVERAGE.json',dict(ablations=proofs,companions=companion,sources=sources));write(ROOT/'PREPARED.json',dict(time=time.time(),tasks=tasks,new_calls=sum(t['n'] for t in tasks),reused_ablation_calls=350000,canonical={b:sha(ROOT/'canonical'/f'{b}.jsonl') for b in BBS},mechanism_sha256=sha(ROOT/'canonical/mechanism.jsonl')));print('PREPARED',sum(t['n'] for t in tasks),flush=True)
if __name__=='__main__':main()
