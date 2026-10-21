"""Prepare a new reduced study without modifying any source artifacts."""
from common import *
from reduction import select
from imports import create_imports
import collections,shutil,csv
OLD=Path('/home/ubuntu/experiments/judge-reduced-20261011')
def main():
 reserve();assert not (ROOT/'SETUP_COMPLETED.json').exists()
 selections={};audits={};proof={}
 for b in BBS:
  src=HARD/b;rows=read(src/'selected_flags.json');assert sha(src/'selected_flags.json')==read(src/'SAMPLE_FROZEN.json')['selection_sha256']
  proof[str(src/'selected_flags.json')]=sha(src/'selected_flags.json')
  chosen={};audit={}
  for ds in sorted({r['dataset'] for r in rows}):
   for split in ['val','test']:
    pool=[r for r in rows if r['dataset']==ds and r['split']==split];subset,info=select(b,pool)
    shuffled=list(reversed(pool));assert select(b,shuffled)==(subset,info),'Selection depends on input ordering'
    chosen.update(subset);audit[ds+':'+split]=info
  selected=[chosen[r['sample_id']] for r in rows if r['sample_id'] in chosen]
  assert len(selected)==2000
  assert not {(r['dataset'],r['group']) for r in selected if r['split']=='val'} & {(r['dataset'],r['group']) for r in selected if r['split']=='test'}
  assert {r['sample_id'] for r in selected} <= {r['sample_id'] for r in read(OLD/'hard'/b/'selected_flags.json')}
  selections[b]=selected;audits[b]=audit
  for mode in MODES:
   d=ROOT/mode/b;d.mkdir(parents=True)
   write(d/'selected_flags.json',selected)
   write(d/'SAMPLE_FROZEN.json',dict(selection_sha256=sha(d/'selected_flags.json'),sample_ids=[r['sample_id'] for r in selected],counts={ds:{s:100 for s in ['val','test']} for ds in {r['dataset'] for r in selected}},seed=20261011,scope='Nested 100 validation and 100 test cases, score-band x gold-label strata, two-stage inclusion weights',parent_sha256=sha(src/'selected_flags.json')))
   shutil.copy2(src/'THRESHOLDS.json',d/'THRESHOLDS.json')
   for p in src.glob('*_predictions.npz'):(d/p.name).symlink_to(p.resolve())
   (d/'judge_inputs').symlink_to(src/'judge_inputs',target_is_directory=True)
 write(ROOT/'SAMPLING_AUDIT.json',dict(seed=20261011,per_split=100,groups=audits,parent_checksums=proof))
 # Preserve canonical IDs and verbatim messages. The old private map stays private.
 canonical=ROOT/'canonical';canonical.mkdir()
 out=ROOT/'api_export';out.mkdir(exist_ok=True)
 selected_ids={b:{r['sample_id'] for r in rows} for b,rows in selections.items()};count=0;pids=[];balance=collections.Counter()
 assert sha(OLD/'canonical/cases.jsonl')==read(OLD/'canonical/PREPARED.json')['cases_sha256']
 with (canonical/'cases.jsonl').open('w') as f,(out/'prompts.csv').open('w',newline='') as csvf:
  w=csv.DictWriter(csvf,fieldnames=['prompt_id','output_type','system_prompt','user_prompt'],lineterminator='\n');w.writeheader()
  for r in jl(OLD/'canonical/cases.jsonl'):
   if r['sample_id'] not in selected_ids[r['backbone']]:continue
   f.write(json.dumps(r,ensure_ascii=False)+'\n');w.writerow(dict(prompt_id=r['prompt_id'],output_type=r['mode'],system_prompt=r['messages'][0]['content'],user_prompt=r['messages'][1]['content']))
   pids.append(r['prompt_id']);balance[(r['backbone'],r['dataset'],r['split'],r['condition'],r['mode'])]+=1;count+=1
 assert count==len(set(pids))==60000 and len(balance)==600 and set(balance.values())=={100}
 csv.field_size_limit(100000000)
 with (out/'prompts.csv').open(newline='') as f:
  for row,r in zip(csv.DictReader(f),jl(canonical/'cases.jsonl'),strict=True):
   assert row==dict(prompt_id=r['prompt_id'],output_type=r['mode'],system_prompt=r['messages'][0]['content'],user_prompt=r['messages'][1]['content'])
 write(canonical/'PREPARED.json',dict(total=count,cases_sha256=sha(canonical/'cases.jsonl'),parent_canonical_sha256=sha(OLD/'canonical/cases.jsonl'),selection_audit_sha256=sha(ROOT/'SAMPLING_AUDIT.json')))
 write(out/'export_manifest.json',dict(total=count,prompts_csv_sha256=sha(out/'prompts.csv'),ordered_ids_sha256=digest('\n'.join(pids)+'\n'),groups=600,per_group=100,checks=dict(exact_original_messages=True,stable_original_prompt_ids=True,same_cases_all_conditions=True,csv_roundtrip_exact=True,no_target_labels_exported=True)))
 # Existing Qwen/Gemma outputs are projected, never generated again.
 reused=0
 for mode,src in [('hard',OLD/'hard'),('probability',OLD/'probability')]:
  for b in BBS:
   wanted=selected_ids[b];ids=[r['sample_id'] for r in selections[b]]
   for j in ['qwen','gemma']:
    for c in CS:
     source=src/b/'judges'/f'{j}_{c}';d=cell(mode,b,j,c);d.mkdir(parents=True)
     cfg=read(source/'run/config.json');done=read(source/'run/COMPLETED')
     assert sha(source/'run/config.json')==done['config_sha256'] and sha(source/'run/judgments.jsonl')==done['judgments_sha256']
     assert sha(source/'prompts.jsonl')==cfg['prompt_sha256']
     prompts=[r for r in jl(source/'prompts.jsonl') if r['sample_id'] in wanted];assert [r['sample_id'] for r in prompts]==ids
     with (d/'prompts.jsonl').open('w') as f:
      for r in prompts:f.write(json.dumps(r,ensure_ascii=False)+'\n')
     (d/'run/batches').mkdir(parents=True)
     projected=dict(cfg,n=2000,prompt_sha256=sha(d/'prompts.jsonl'),projection_source_config_sha256=sha(source/'run/config.json'),reused=2000)
     write(d/'approved_config.json',projected);write(d/'run/config.json',projected)
     records=create_imports(source,d,wanted);assert [r['sample_id'] for r in records]==ids
     with (d/'run/judgments.jsonl').open('w') as f:
      for r in records:f.write(json.dumps(r)+'\n')
     write(d/'run/COMPLETED',dict(n=2000,time=time.time(),judgments_sha256=sha(d/'run/judgments.jsonl'),config_sha256=sha(d/'run/config.json'),imports_sha256=sha(d/'run/IMPORTED.json')))
     reused+=len(records)
 assert reused==120000
 write(ROOT/'SETUP_COMPLETED.json',dict(time=time.time(),reused_judgments=reused,maximum_additional=300000,expected_total=420000))
 print('Reduced setup verified: 10000 cases, 60000 prompts per judge, 120000 saved Qwen/Gemma outputs reused',flush=True)
if __name__=='__main__':main()
