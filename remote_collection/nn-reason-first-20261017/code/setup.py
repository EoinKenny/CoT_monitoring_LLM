from common import *
import collections,sys,fcntl,importlib.metadata
AFTER=ROOT.parent/'nn-reason-flips-20261017'
def main():
 reserve();lock=(ROOT/'setup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (ROOT/'PREPARED.json').exists():
  for n,h in read(ROOT/'ARTIFACTS_FROZEN.json').items():assert sha(ROOT/n)==h,n
  return
 for n,h in read(ROOT/'CODE_FROZEN.json').items():assert sha(ROOT/n)==h,n
 done=read(AFTER/'COMPLETED.json');assert done['coverage_sha256']==sha(AFTER/'COVERAGE.json') and done['accounted']==35919
 write(ROOT/'judge_specs.json',read(AFTER/'judge_specs.json'))
 old=(SOURCE/'NEW_OUTPUT.txt').read_text();new=(ROOT/'OUTPUT.txt').read_text().rstrip('\n')
 canonical=SOURCE/'canonical/cases.jsonl';assert sha(canonical)==read(SOURCE/'canonical/PREPARED.json')['cases_sha256']
 all_messages={(r['backbone'],r['sample_id']):r for r in jl(canonical)}
 evidence={str(canonical):sha(canonical)};summary=[]
 for j in ALL:
  s=specs()[j];cfg=read(AFTER/'models'/j/'parent_config.json');cfg=dict(cfg,sampling=dict(temperature=0,seed=20261001,max_tokens=256,regex=REGEX),protocol_sha256=sha(ROOT/'PROTOCOL.json'),parent_config_sha256=sha(AFTER/'models'/j/'parent_config.json'))
  write(ROOT/'models'/j/'parent_config.json',cfg)
  if j=='mistral':
   from vllm.tokenizers.mistral import MistralTokenizer
   tok=MistralTokenizer.from_pretrained(s['model_path'])
  else:
   from transformers import AutoTokenizer
   tok=AutoTokenizer.from_pretrained(s['model_path'],local_files_only=True,trust_remote_code=False)
  for b in BBS:
   src=AFTER/f'score/{b}/judges/{j}_nearest_reason_after';frozen=read(src/'FROZEN.json')
   assert frozen['selection_sha256']==sha(src/'selection.json') and frozen['prompts_sha256']==sha(src/'prompts.jsonl')
   selection=read(src/'selection.json');prior=list(jl(src/'prompts.jsonl'));assert [x['sample_id'] for x in selection]==[x['sample_id'] for x in prior]
   evidence[str(src/'FROZEN.json')]=sha(src/'FROZEN.json');d=cell('score',b,j,CS[0]);d.mkdir(parents=True,exist_ok=True);pp=[]
   for p in prior:
    r=all_messages[b,p['sample_id']];assert r['messages_sha256']==p['messages_sha256'] and r['prompt_id']==p['prompt_id']
    msgs=[dict(x) for x in r['messages']];assert msgs[0]['content'].count(old)==1
    msgs[0]['content']=msgs[0]['content'].replace(old,new,1)
    cue='Positive-class probability (0-100):';assert msgs[1]['content'].endswith(cue)
    msgs[1]['content']=msgs[1]['content'][:-len(cue)]+'Assessment (VERY brief reasoning, then positive-class probability):'
    # Target, neighbour, label, prediction, task and reference text are byte-identical.
    assert msgs[1]['content'].rsplit('Assessment (VERY brief reasoning, then positive-class probability):',1)[0]==r['messages'][1]['content'][:-len(cue)]
    kwargs=dict(s['chat_template_kwargs'],add_generation_prompt=True,tokenize=j=='mistral')
    if j=='mistral':ids=tok.apply_chat_template(msgs,**kwargs);text=tok.decode(ids,skip_special_tokens=False)
    else:text=tok.apply_chat_template(msgs,**kwargs);ids=tok.encode(text,add_special_tokens=False)
    assert ids and len(ids)+256<=cfg['engine']['max_model_len'],(j,b,p['sample_id'],len(ids))
    mh=digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')))
    pp.append(dict(p,prompt_id='nrf_'+digest(json.dumps([b,p['sample_id'],mh],separators=(',',':'))),condition=CS[0],prompt=text,token_ids=ids,input_tokens=len(ids),instruction_sha256=digest(text),messages_sha256=mh,messages=msgs,parent_prompt_id=p['prompt_id']))
   text=''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pp);dest=d/'prompts.jsonl'
   if dest.exists():assert dest.read_text()==text
   else:dest.write_text(text)
   write(d/'selection.json',selection);write(d/'reused.json',[])
   # Two deterministic smoke batches per backbone, spread across available datasets.
   queues={ds:sorted([k for k,p in enumerate(pp) if p['dataset']==ds],key=lambda k:digest('20261001:'+pp[k]['sample_id'])) for ds in sorted({p['dataset'] for p in pp})}
   smoke=[];bs=cfg['batch_size']
   while len(smoke)<min(2*bs,len(pp)):
    for ds,q in queues.items():
     if q and len(smoke)<min(2*bs,len(pp)):smoke.append(q.pop(0))
   used=set(smoke);remain=[k for k in range(len(pp)) if k not in used]
   plan=[smoke[k:k+bs] for k in range(0,len(smoke),bs)]+[remain[k:k+bs] for k in range(0,len(remain),bs)]
   assert sorted(k for ch in plan for k in ch)==list(range(len(pp)))
   write(d/'PLAN.json',plan)
   write(d/'FROZEN.json',dict(n=len(pp),prompts_sha256=sha(dest),selection_sha256=sha(d/'selection.json'),reused_sha256=sha(d/'reused.json'),plan_sha256=sha(d/'PLAN.json'),parent_config_sha256=sha(ROOT/'models'/j/'parent_config.json')))
   summary.append(dict(judge=j,backbone=b,n=len(pp),smoke_n=len(smoke),batch_size=bs,max_input_tokens=max(p['input_tokens'] for p in pp)))
   write(ROOT/'SETUP_STATUS.json',dict(judge=j,backbone=b,cells=len(summary),time=time.time()))
  tokfiles={p.name:sha(p) for p in Path(s['model_path']).iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.model'] and 'safetensors' not in p.name}
  write(ROOT/'models'/j/'TOKENIZER.json',dict(files=tokfiles,chat_template_kwargs=s['chat_template_kwargs'],versions={p:importlib.metadata.version(p) for p in ['transformers','tokenizers','mistral-common']}))
 assert sum(x['n'] for x in summary)==35919
 write(ROOT/'INPUT_SHA256.json',evidence);write(ROOT/'PREPARED.json',dict(total=35919,new=35919,reused=0,cells=summary,time=time.time(),protocol_sha256=sha(ROOT/'PROTOCOL.json')))
 files=[ROOT/'PREPARED.json',ROOT/'judge_specs.json',ROOT/'INPUT_SHA256.json',ROOT/'PROTOCOL.json',ROOT/'OUTPUT.txt']+list((ROOT/'models').glob('*/parent_config.json'))+list((ROOT/'models').glob('*/TOKENIZER.json'))+list((ROOT/'score').glob('*/judges/*/FROZEN.json'))
 write(ROOT/'ARTIFACTS_FROZEN.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
if __name__=='__main__':main()
