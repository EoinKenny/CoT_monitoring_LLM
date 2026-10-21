"""512-token continuation retaining every prior committed/ambiguous attempt."""
from common import *
from parser_source import parse
from output_policy import missing
import shutil,collections
OLD=ROOT.parent/'nn-reason-first-20261017'
def main():
 assert not (ROOT/'PREPARED.json').exists()
 assert (OLD/'PAUSED_FOR_512.json').exists()
 reserve();write(ROOT/'judge_specs.json',read(OLD/'judge_specs.json'));summary=[];sources={};budgets=[]
 for j in ALL:
  oldcfg=read(OLD/'models'/j/'parent_config.json');cfg=dict(oldcfg,sampling=dict(oldcfg['sampling'],max_tokens=512),protocol_sha256=sha(ROOT/'PROTOCOL.json'),parent_config_sha256=sha(OLD/'models'/j/'parent_config.json'))
  prompts={b:list(jl(OLD/f'score/{b}/judges/{j}_{CS[0]}/prompts.jsonl')) for b in BBS}
  need=max(p['input_tokens'] for pp in prompts.values() for p in pp)+512
  engine=dict(cfg['engine']);engine['max_model_len']=max(engine['max_model_len'],((need+255)//256)*256);assert engine['max_model_len']<=specs()[j]['context_limit'];cfg['engine']=engine
  write(ROOT/'models'/j/'parent_config.json',cfg);shutil.copyfile(OLD/'models'/j/'TOKENIZER.json',ROOT/'models'/j/'TOKENIZER.json')
  for b in BBS:
   src=OLD/f'score/{b}/judges/{j}_{CS[0]}';d=cell('score',b,j,CS[0]);d.mkdir(parents=True,exist_ok=True);pp=prompts[b];frozen=read(src/'FROZEN.json');conf=sha(src/'FROZEN.json')
   for file,key in [('prompts.jsonl','prompts_sha256'),('selection.json','selection_sha256'),('PLAN.json','plan_sha256'),('reused.json','reused_sha256')]:assert sha(src/file)==frozen[key]
   for name in ['prompts.jsonl','selection.json']:shutil.copyfile(src/name,d/name)
   plan=read(src/'PLAN.json');records=read(src/'reused.json');origins=[]
   for a in sorted((src/'run/batches').glob('*.attempt.json')):
    number=int(a.name.split('.')[0]);chunk=[pp[k] for k in plan[number]];attempt=read(a);assert attempt['ids']==[p['prompt_id'] for p in chunk] and attempt['config_sha256']==conf
    raw=a.with_name(f'{number:06d}.raw.json');commit=a.with_name(f'{number:06d}.json')
    if commit.exists():
     z=read(commit);assert z['config_sha256']==conf and z['records_sha256']==digest(json.dumps(z['records'],sort_keys=True));rr=z['records']
     if z['raw_available']:assert sha(raw)==z['raw_sha256'] and parse(read(raw),chunk,number,rr[0]['batch_seconds'])==rr
    elif raw.exists():
     timing=a.with_name(f'{number:06d}.timing.json');rr=parse(read(raw),chunk,number,read(timing)['seconds'] if timing.exists() else 0.)
    else:rr=missing(chunk,number,0.)
    records.extend(rr);origins.append(dict(attempt=str(a),attempt_sha256=sha(a),raw=str(raw) if raw.exists() else None,raw_sha256=sha(raw) if raw.exists() else None,source_config_sha256=conf,max_tokens=256))
   used={x['sample_id'] for x in records};assert len(used)==len(records)
   todo=[k for k,p in enumerate(pp) if p['sample_id'] not in used];bs=cfg['batch_size']
   queues={ds:sorted([k for k in todo if pp[k]['dataset']==ds],key=lambda k:digest('20261001:'+pp[k]['sample_id'])) for ds in sorted({pp[k]['dataset'] for k in todo})};smoke=[]
   while len(smoke)<min(2*bs,len(todo)):
    for q in queues.values():
     if q and len(smoke)<min(2*bs,len(todo)):smoke.append(q.pop(0))
   smoke_set=set(smoke);rest=[k for k in todo if k not in smoke_set];newplan=[smoke[k:k+bs] for k in range(0,len(smoke),bs)]+[rest[k:k+bs] for k in range(0,len(rest),bs)]
   assert sorted(k for batch in newplan for k in batch)==sorted(todo)
   write(d/'reused.json',records);write(d/'reuse_provenance.json',origins);write(d/'PLAN.json',newplan)
   write(d/'FROZEN.json',dict(n=len(pp),prompts_sha256=sha(d/'prompts.jsonl'),selection_sha256=sha(d/'selection.json'),reused_sha256=sha(d/'reused.json'),plan_sha256=sha(d/'PLAN.json'),parent_config_sha256=sha(ROOT/'models'/j/'parent_config.json')))
   sources[str(src/'FROZEN.json')]=conf
   for p in pp:budgets.append(dict(judge=j,backbone=b,sample_id=p['sample_id'],prompt_id=p['prompt_id'],max_tokens=256 if p['sample_id'] in used else 512,reused=p['sample_id'] in used))
   summary.append(dict(judge=j,backbone=b,n=len(pp),reused_256=len(records),new_512=len(todo),new_smoke_n=len(smoke)))
 write(ROOT/'TOKEN_BUDGETS.json',budgets);write(ROOT/'INPUT_SHA256.json',sources)
 assert sum(x['n'] for x in summary)==35919
 write(ROOT/'PREPARED.json',dict(total=35919,new=sum(x['new_512'] for x in summary),reused=sum(x['reused_256'] for x in summary),cells=summary,time=time.time(),protocol_sha256=sha(ROOT/'PROTOCOL.json')))
 files=[ROOT/n for n in ['PREPARED.json','judge_specs.json','TOKEN_BUDGETS.json','INPUT_SHA256.json','PROTOCOL.json','OUTPUT.txt']]+list((ROOT/'models').glob('*/parent_config.json'))+list((ROOT/'models').glob('*/TOKENIZER.json'))+list((ROOT/'score').glob('*/judges/*/FROZEN.json'))+list((ROOT/'score').glob('*/judges/*/reuse_provenance.json'))
 write(ROOT/'ARTIFACTS_FROZEN.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
 print(json.dumps(read(ROOT/'PREPARED.json')))
if __name__=='__main__':main()
