"""Re-template identical task messages, preserving all complete input text."""
from common import *
import sys,importlib.metadata
def main(j):
 reserve();s=specs()[j];manifest=ROOT/'prepared'/f'{j}.json'
 canonical=ROOT/'canonical/cases.jsonl';canonical_hash=sha(canonical)
 if manifest.exists():
  saved=read(manifest);assert saved['canonical_sha256']==canonical_hash and saved['spec_sha256']==digest(json.dumps(s,sort_keys=True))
  for x in saved['cells']:assert sha(Path(x['directory'])/'prompts.jsonl')==x['prompt_sha256']
  return
 assert canonical_hash==read(ROOT/'canonical/PREPARED.json')['cases_sha256']
 path=s['model_path']
 if j=='mistral':
  from vllm.tokenizers.mistral import MistralTokenizer
  tok=MistralTokenizer.from_pretrained(path)
 else:
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
 outputs={};expected={b:[r['sample_id'] for r in read(ROOT/'score'/b/'selected_flags.json')] for b in BBS};count=0
 for mode,b,c,d in cells(j):
  d.mkdir(parents=True,exist_ok=True);assert not (d/'prompts.jsonl').exists(),f'Uncommitted preparation exists: {d}'
  outputs[mode,b,c]=dict(directory=str(d),stream=(d/'prompts.jsonl.tmp').open('w'),ids=[],total_input_tokens=0,max_input_tokens=0)
 for r in jl(canonical):
  key=r['mode'],r['backbone'],r['condition'];o=outputs[key]
  kwargs=dict(s['chat_template_kwargs'],add_generation_prompt=True,tokenize=j=='mistral')
  if j=='mistral':
   ids=tok.apply_chat_template(r['messages'],**kwargs)
   text=tok.decode(ids,skip_special_tokens=False)
  else:
   text=tok.apply_chat_template(r['messages'],**kwargs)
   ids=tok.encode(text,add_special_tokens=False)
  assert ids and len(ids)+8<=s['context_limit'],f'Input exceeds model context; never truncate: {j} {r["prompt_id"]} {len(ids)}'
  out=dict(sample_id=r['sample_id'],prompt_id=r['prompt_id'],dataset=r['dataset'],backbone=r['backbone'],condition=r['condition'],mode=r['mode'],split=r['split'],prompt=text,token_ids=ids,input_tokens=len(ids),instruction_sha256=digest(text),messages_sha256=r['messages_sha256'])
  o['stream'].write(json.dumps(out,ensure_ascii=False)+'\n');o['ids'].append(r['sample_id']);o['total_input_tokens']+=len(ids);o['max_input_tokens']=max(o['max_input_tokens'],len(ids));count+=1
  if count%10000==0:reserve();write(ROOT/'PREPARE_STATUS.json',dict(judge=j,completed=count,total=TOTAL,time=time.time()));print(j,count,flush=True)
 assert count==TOTAL
 all_cells=[]
 for (mode,b,c),o in outputs.items():
  assert o['ids']==expected[b]
  f=o.pop('stream');f.flush();os.fsync(f.fileno());f.close();d=Path(o['directory']);(d/'prompts.jsonl.tmp').replace(d/'prompts.jsonl')
  cfg=dict(model=s['repo'],revision=s['revision'],judge=j,condition=c,backbone=b,mode=mode,n=len(o.pop('ids')),prompt_sha256=sha(d/'prompts.jsonl'),canonical_sha256=canonical_hash,protocol_sha256=sha(ROOT/'PROTOCOL.json'),tokenizer_path=path,tokenizer_kwargs=s['chat_template_kwargs'],spec_sha256=digest(json.dumps(s,sort_keys=True)),**o)
  write(d/'approved_config.json',cfg);(d/'model_path.txt').write_text(path);all_cells.append(cfg)
 tokenizer_files={p.name:sha(p) for p in Path(path).iterdir() if p.is_file() and p.suffix in ['.json','.jinja','.model'] and 'safetensors' not in p.name}
 write(manifest,dict(judge=j,n=count,canonical_sha256=canonical_hash,spec_sha256=digest(json.dumps(s,sort_keys=True)),max_input_tokens=max(o['max_input_tokens'] for o in all_cells),total_input_tokens=sum(o['total_input_tokens'] for o in all_cells),cells=all_cells,tokenizer_files=tokenizer_files,versions={p:importlib.metadata.version(p) for p in ['transformers','tokenizers','mistral-common']},time=time.time()))
if __name__=='__main__':main(sys.argv[1])
