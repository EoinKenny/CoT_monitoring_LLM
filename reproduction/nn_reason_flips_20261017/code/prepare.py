from common import *
from parser_source import parse
from output_policy import missing
import collections

def main():
 reserve();assert not (ROOT/'PREPARED.json').exists()
 write(ROOT/'judge_specs.json',read(SOURCE/'judge_specs.json'));summary=[];allinputs={}
 for j in ALL:
  cfg=read(SOURCE/'models'/j/'config.json');write(ROOT/'models'/j/'parent_config.json',cfg)
  assert cfg['sampling']==dict(temperature=0,seed=20261001,max_tokens=256,regex=REGEX)
  for b in BBS:
   selpath=PARENT/f'score/{b}/selected_flags.json';rows=read(selpath);by={x['sample_id']:x for x in rows};answers={}
   for c in ['baseline','nearest']:
    run=PARENT/f'score/{b}/judges/{j}_{c}/run';done=read(run/'COMPLETED');fp=run/'judgments.jsonl'
    assert sha(fp)==done['judgments_sha256'] and sha(run/'config.json')==done['config_sha256']
    rr=list(jl(fp));assert len(rr)==10000 and [x['sample_id'] for x in rr]==[x['sample_id'] for x in rows]
    answers[c]={x['sample_id']:x for x in rr};allinputs[str(fp)]=sha(fp)
   chosen=[]
   for r in rows:
    sid=r['sample_id'];a=answers['baseline'][sid]['positive_probability'];n=answers['nearest'][sid]['positive_probability']
    if r['split']=='test' and (a>=50)!=(n>=50):
     chosen.append(dict(r,control_probability=a,nn_probability=n,flip_type='helped' if int(n>=50)==r['true_label'] else 'harmed',judge=j,backbone=b))
   wanted={x['sample_id'] for x in chosen};src=SOURCE/f'score/{b}/judges/{j}_nearest_reason_after';d=cell('score',b,j,CS[0]);d.mkdir(parents=True,exist_ok=True)
   approved=read(src/'approved_config.json');assert sha(src/'prompts.jsonl')==approved['prompt_sha256'];allinputs[str(src/'prompts.jsonl')]=approved['prompt_sha256']
   pp=list(jl(src/'prompts.jsonl'));prompts=[p for p in pp if p['sample_id'] in wanted];assert [p['sample_id'] for p in prompts]==[x['sample_id'] for x in chosen]
   (d/'prompts.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in prompts));write(d/'selection.json',chosen)
   source_run=src/'run';plan=read(source_run/'PLAN.json');source_conf=sha(source_run/'config.json');assert read(source_run/'config.json')['global_config_sha256']==sha(SOURCE/'models'/j/'config.json')
   reused=[];origins=[]
   for attempt in sorted((source_run/'batches').glob('*.attempt.json')):
    index=int(attempt.name.split('.')[0]);a=read(attempt);chunk=[pp[k] for k in plan[index]['indices']]
    assert a['ids']==[p['prompt_id'] for p in chunk] and a['config_sha256']==source_conf
    if not any(p['sample_id'] in wanted for p in chunk):continue
    raw=attempt.with_name(f'{index:06d}.raw.json');commit=attempt.with_name(f'{index:06d}.json')
    if raw.exists():
     z=read(commit) if commit.exists() else None
     if z:
      assert z['raw_sha256']==sha(raw) and z['records_sha256']==digest(json.dumps(z['records'],sort_keys=True)) and z['config_sha256']==source_conf
     rr=parse(read(raw),chunk,index,z['records'][0]['batch_seconds'] if z else 0.)
     if z:assert rr==z['records']
     origin=dict(raw=str(raw),raw_sha256=sha(raw),committed=bool(z))
    else:
     rr=missing(chunk,index,0.);origin=dict(attempt=str(attempt),attempt_sha256=sha(attempt),ambiguous=True)
    reused.extend(r for r in rr if r['sample_id'] in wanted);origins.append(origin)
   assert len(reused)==len({r['sample_id'] for r in reused})
   write(d/'reused.json',reused);write(d/'reuse_provenance.json',origins)
   already={r['sample_id'] for r in reused};todo=[k for k,p in enumerate(prompts) if p['sample_id'] not in already];bs=cfg['batch_size'];write(d/'PLAN.json',[todo[k:k+bs] for k in range(0,len(todo),bs)])
   frozen=dict(n=len(prompts),prompts_sha256=sha(d/'prompts.jsonl'),selection_sha256=sha(d/'selection.json'),reused_sha256=sha(d/'reused.json'),plan_sha256=sha(d/'PLAN.json'),parent_config_sha256=sha(ROOT/'models'/j/'parent_config.json'))
   write(d/'FROZEN.json',frozen)
   summary.append(dict(judge=j,backbone=b,n=len(prompts),reuse=len(reused),new=len(todo),ambiguous=sum(x.get('output_status')=='ambiguous_no_raw' for x in reused),helped=sum(x['flip_type']=='helped' for x in chosen),harmed=sum(x['flip_type']=='harmed' for x in chosen),datasets=dict(collections.Counter(x['dataset'] for x in chosen))))
 write(ROOT/'INPUT_SHA256.json',allinputs);write(ROOT/'PREPARED.json',dict(time=time.time(),cells=summary,total=sum(x['n'] for x in summary),new=sum(x['new'] for x in summary),reused=sum(x['reuse'] for x in summary),ambiguous=sum(x['ambiguous'] for x in summary),protocol_sha256=sha(ROOT/'PROTOCOL.json')))
 files=[ROOT/'PREPARED.json',ROOT/'judge_specs.json',ROOT/'INPUT_SHA256.json']+list((ROOT/'models').glob('*/parent_config.json'))+list((ROOT/'score').glob('*/judges/*/FROZEN.json'))
 write(ROOT/'ARTIFACTS_FROZEN.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
 print(json.dumps(read(ROOT/'PREPARED.json')),flush=True)
if __name__=='__main__':main()
