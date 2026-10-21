from common import *
import numpy as np, re, copy
SOURCE=HARD/'qwen'
REFERENCE_START='You will also receive the most similar training example, retrieved using the activation representation used by the probe.'
USE='Use the reference as supporting context, and independently classify the current conversation.'
FOOTER='Positive-class probability (0-100):'
FIELDS={'text_label':(1,1,0),'text_prediction':(1,0,1),'text_only':(1,0,0),'label_only':(0,1,0),'prediction_only':(0,0,1),'label_prediction':(0,1,1),'random_full':(1,1,1)}
def random_index(ds,target,n):
 limit=2**256-(2**256 % n)
 for counter in range(100):
  key=json.dumps([20261013,ds,target,counter],separators=(',',':'))
  h=digest(key); number=int(h,16)
  if number<limit:return number%n,h,counter
 raise RuntimeError('Hash rejection failed')
def transform(messages,condition,random_ref=None,names=None):
 text,label,pred=FIELDS[condition];system,user=[m['content'] for m in messages]
 assert system.count(REFERENCE_START)==1 and system.count(USE)==1
 prefix,rest=system.split(REFERENCE_START);_,suffix=rest.split(USE)
 random=condition=='random_full'
 intro=('You will also receive a randomly selected training example from the same dataset.' if random else REFERENCE_START if text else 'You will also receive information about the most similar training example, retrieved using the activation representation used by the probe.')
 details=[]
 if label and pred:details.append("The reference example includes its dataset label and the probe's prediction. The probe may disagree with the dataset label.")
 elif label:details.append('The reference information includes its dataset label.')
 elif pred:details.append("The reference information includes the probe's prediction. The probe can be wrong.")
 details.append('The reference and the current conversation may belong to different classes.' if random else 'Similarity does not guarantee that the reference and the current conversation belong to the same class.')
 new_system=prefix+intro+'\n\n'+' '.join(details)+'\n\n'+USE+suffix
 current,ref=user.rsplit('\n\nNearest-neighbour explanation:\n',1)
 body,ending=ref.rsplit('\n\n'+FOOTER,1);assert ending==''
 raw_text,raw_labels=body.removeprefix('The most similar training example was:\n').rsplit('\n\nReference example:\n',1)
 assert raw_labels.count('\nProbe prediction: ')==1
 label_line,pred_line=raw_labels.split(' |\n');assert label_line.startswith('Dataset label: ') and pred_line.startswith('Probe prediction: ')
 if random:
  raw_text=json.dumps(random_ref['messages'],ensure_ascii=True)
  label_line=f"Dataset label: {names[random_ref['label']]} ({random_ref['label']})"
  pred_line=f"Probe prediction: {names[random_ref['prediction']]} ({random_ref['prediction']})"
 parts=[current, 'Random training-reference control:' if random else 'Nearest-neighbour explanation:']
 if text:parts.append(('The randomly selected training example was:\n' if random else 'The most similar training example was:\n')+raw_text)
 if label or pred:
  ls=([label_line] if label else [])+([pred_line] if pred else [])
  parts.append('Reference example:\n'+' |\n'.join(ls))
 parts.append(FOOTER)
 return [{'role':'system','content':new_system},{'role':'user','content':parts[0]+'\n\n'+parts[1]+'\n'+'\n\n'.join(parts[2:])}]

def main():
 reserve();assert read(PARENT/'COMPLETED.json')['n']==900000
 assert not (ROOT/'canonical/cases.jsonl').exists(),'Already set up; do not replace'
 rows=read(PARENT/'score/qwen/selected_flags.json');ids=[r['sample_id'] for r in rows];assert len(set(ids))==len(ids)==10000
 specs_parent=read(PARENT/'judge_specs.json');write(ROOT/'judge_specs.json',specs_parent)
 d=ROOT/'score/qwen';d.mkdir(parents=True,exist_ok=True)
 for name in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json']:
  (d/name).write_bytes((PARENT/'score/qwen'/name).read_bytes())
 canonical={r['sample_id']:r for r in jl(PARENT/'canonical/cases.jsonl') if r['backbone']=='qwen' and r['condition']=='nearest'}
 assert list(canonical)==ids
 names={};pool_manifests={};random_refs={};source_hashes={};selection=[]
 for ds in sorted({r['dataset'] for r in rows}):
  for r in canonical.values():
   if r['dataset']!=ds:continue
   for kind in ['Dataset label','Probe prediction']:
    hit=re.search(r'\n'+kind+r': (.+) \(([01])\)',r['messages'][1]['content'])
    assert hit;names.setdefault(ds,{})[int(hit[2])]=hit[1]
  assert set(names[ds])=={0,1}
  meta=read(SOURCE/f'{ds}_metadata.json');z=np.load(SOURCE/f'{ds}_predictions.npz',allow_pickle=False)
  training=read((SOURCE/f'{ds}_metadata.json').resolve().parent/f'{ds}_TRAIN_COMPLETED.json')
  for suffix in ['metadata.json','predictions.npz','probe.joblib']:
   f=SOURCE/f'{ds}_{suffix}';h=sha(f);assert h==training['sha256'][f.name];source_hashes[str(f.resolve())]=h
  threshold=training['result']['threshold'];assert len(z['probabilities'])==len(meta)
  assert np.array_equal(z['labels'],[r['label'] for r in meta]);assert np.array_equal(z['splits'],[r['split'] for r in meta]);assert np.isfinite(z['probabilities']).all()
  pool=sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id'])
  poolids=[meta[i]['immutable_id'] for i in pool];assert len(poolids)==len(set(poolids))
  selected=[r for r in rows if r['dataset']==ds];assert len(selected)==1000
  train_groups={meta[i]['group'] for i in pool};assert not train_groups & {r['group'] for r in selected}
  pool_manifests[ds]=dict(n=len(pool),ordered_ids_sha256=digest(json.dumps(poolids,separators=(',',':'))),probe_threshold=threshold)
  ef=SOURCE/'judge_inputs/explanations.jsonl'; proof=read(SOURCE/'judge_inputs/EXTRACTION_COMPLETED'); assert sha(ef)==proof['explanations_sha256']; source_hashes[str(ef)]=sha(ef)
  expl={r['sample_id']:r for r in jl(ef) if r['dataset']==ds}
  for r in selected:
   assert meta[r['index']]['immutable_id']==r['sample_id'] and meta[r['index']]['split']==r['split']
   k,h,counter=random_index(ds,r['sample_id'],len(pool));i=pool[k];nr=meta[i]
   prediction=int(z['probabilities'][i]>=threshold)
   random_refs[r['sample_id']]=dict(messages=nr['messages'],label=int(nr['label']),prediction=prediction)
   nn=expl[r['sample_id']]['nearest'];assert meta[nn['index']]['id']==nn['id']; assert meta[nn['index']]['messages']==nn['messages']; nn['immutable_id']=meta[nn['index']]['immutable_id'];assert int(z['probabilities'][nn['index']]>=threshold)==nn['prediction']
   selection.append(dict(sample_id=r['sample_id'],dataset=ds,split=r['split'],training_id=nr['immutable_id'],training_index=i,training_label=int(nr['label']),training_prediction=prediction,training_probability=float(z['probabilities'][i]),pool_n=len(pool),hash=h,counter=counter,nearest_id=nn['immutable_id'],same_as_nearest=nr['immutable_id']==nn['immutable_id']))
  (d/f'{ds}_predictions.npz').symlink_to((SOURCE/f'{ds}_predictions.npz').resolve())
  print('random pool frozen',ds,len(pool),flush=True)
  del meta,z
 write(ROOT/'RANDOM_SELECTION.json',selection);write(ROOT/'TRAINING_POOLS.json',pool_manifests)
 output=ROOT/'canonical/cases.jsonl';output.parent.mkdir(exist_ok=True)
 examples={};count=0
 with output.open('w') as f:
  for condition in CS:
   for sid in ids:
    original=canonical[sid];msg=transform(original['messages'],condition,random_refs[sid],names[original['dataset']])
    # Target and task/output contracts are preserved exactly.
    assert msg[1]['content'].split('\n\n')[0]==original['messages'][1]['content'].split('\n\n')[0]
    assert msg[0]['content'].split('Return only one integer')[1]==original['messages'][0]['content'].split('Return only one integer')[1]
    mh=digest(json.dumps(msg,ensure_ascii=False,separators=(',',':')))
    ident='nnabl_'+digest(json.dumps([sid,condition,mh],separators=(',',':')))
    r=dict(sample_id=sid,prompt_id=ident,backbone='qwen',dataset=original['dataset'],split=original['split'],condition=condition,mode='score',messages=msg,messages_sha256=mh,parent_prompt_id=original['prompt_id'])
    f.write(json.dumps(r,ensure_ascii=False)+'\n');count+=1
    examples.setdefault(condition,msg)
  f.flush();os.fsync(f.fileno())
 assert count==TOTAL
 write(ROOT/'canonical/PREPARED.json',dict(n=count,cases_sha256=sha(output)))
 write(ROOT/'PROMPT_EXAMPLES.json',examples)
 source_hashes[str(PARENT/'canonical/cases.jsonl')]=sha(PARENT/'canonical/cases.jsonl')
 source_hashes[str(PARENT/'score/qwen/selected_flags.json')]=sha(PARENT/'score/qwen/selected_flags.json')
 write(ROOT/'SETUP_COMPLETED.json',dict(n=count,random_references=len(selection),random_equal_nearest=sum(x['same_as_nearest'] for x in selection),canonical_sha256=sha(output),random_selection_sha256=sha(ROOT/'RANDOM_SELECTION.json'),sources=source_hashes))
if __name__=='__main__':main()
