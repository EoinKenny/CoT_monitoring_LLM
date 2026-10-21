import os
os.environ.update(OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13',HF_HUB_OFFLINE='1')
from common import *
import sys,re,time,traceback,fcntl,importlib.metadata
REGEX=r'(?:0|[1-9][0-9]?|100)'
def main(j,phase):
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 reserve();s=read(PARENT/'judge_specs.json')['models'][j];source=ROOT/('controlled_prompts.jsonl' if phase=='controlled' else 'natural_prompts.jsonl');records=list(jl(source));d=ROOT/'runs'/phase/j;d.mkdir(parents=True,exist_ok=True)
 lock=(ROOT.parent/'grid-completion-20261008/gpu-stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tok=AutoTokenizer.from_pretrained(s['model_path'],local_files_only=True,trust_remote_code=False);prompts=[]
 for r in records:
  text=tok.apply_chat_template(r['messages'],**dict(s['chat_template_kwargs'],tokenize=False,add_generation_prompt=True));ids=tok.encode(text,add_special_tokens=False);assert len(ids)+8<=s['context_limit'];prompts.append(dict(r,token_ids=ids,input_tokens=len(ids),rendered_sha256=digest(text)))
 savejl(d/'prompts.jsonl',prompts);bs=s['batch_size'];engine=dict(s['engine'],model=s['model_path'],max_model_len=max(read(PARENT/'models'/j/'config.json')['engine']['max_model_len'],((max(p['input_tokens'] for p in prompts)+8+255)//256)*256),max_num_seqs=bs)
 cfg=dict(judge=j,phase=phase,repo=s['repo'],revision=s['revision'],engine=engine,sampling=dict(temperature=0,max_tokens=8,seed=20261001,regex=REGEX),source_sha256=sha(source),prompt_sha256=sha(d/'prompts.jsonl'),protocol_sha256=sha(ROOT/'PROTOCOL.json'),code_sha256=sha(Path(__file__)),versions={p:importlib.metadata.version(p) for p in ['torch','vllm','transformers']});write(d/'CONFIG.json',cfg);ch=sha(d/'CONFIG.json')
 write(d/'STATUS.json',dict(state='loading',total=len(prompts),time=time.time()));llm=LLM(**engine);params=SamplingParams(temperature=0,seed=20261001,max_tokens=8,structured_outputs=StructuredOutputsParams(regex=REGEX));out=[];elapsed=0.;invalid=0
 for no,start in enumerate(range(0,len(prompts),bs)):
  reserve();chunk=prompts[start:start+bs];p=d/'batches'/f'{no:06d}';p.parent.mkdir(exist_ok=True);assert not p.with_suffix('.attempt.json').exists() and not p.with_suffix('.json').exists()
  write(p.with_suffix('.attempt.json'),dict(prompt_ids=[r['prompt_id'] for r in chunk],time=time.time(),config_sha256=ch));t=time.perf_counter();generated=llm.generate([{'prompt_token_ids':r['token_ids']} for r in chunk],params,use_tqdm=False);sec=time.perf_counter()-t;elapsed+=sec
  raw=[dict(prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for o in generated];write(p.with_suffix('.raw.json'),raw);assert len(raw)==len(chunk);rr=[]
  for r,o in zip(chunk,raw):
   assert o['prompt_token_ids']==r['token_ids'] and len(o['outputs'])==1;g=o['outputs'][0];valid=re.fullmatch(REGEX,g['text'].strip()) is not None and g['finish_reason']=='stop';invalid+=int(not valid)
   rr.append(dict(prompt_id=r['prompt_id'],sample_id=r['sample_id'],arm=r['arm'],backbone=r['backbone'],dataset=r['dataset'],split=r['split'],condition=r['condition'],judge=j,positive_probability=int(g['text'].strip()) if valid else None,status='valid' if valid else 'invalid',raw_text=g['text'],finish_reason=g['finish_reason'],input_tokens=r['input_tokens'],output_tokens=len(g['token_ids']),batch_seconds=sec))
  write(p.with_suffix('.json'),dict(records=rr,raw_sha256=sha(p.with_suffix('.raw.json')),config_sha256=ch));out+=rr;write(d/'STATUS.json',dict(state='running',completed=len(out),total=len(prompts),invalid=invalid,inference_seconds=elapsed,time=time.time()))
  if no==1:
   write(d/'SMOKE.json',dict(n=len(out),invalid=invalid,seconds=elapsed,estimated_total_seconds_with_25pct_margin=elapsed/len(out)*len(prompts)*1.25))
   assert invalid==0,'Smoke returned invalid outputs; preserved without retries'
 savejl(d/'responses.jsonl',out);write(d/'COMPLETED.json',dict(n=len(out),invalid=invalid,inference_seconds=elapsed,responses_sha256=sha(d/'responses.jsonl'),config_sha256=ch,time=time.time()))
if __name__=='__main__':
 try:main(*sys.argv[1:])
 except BaseException:write(ROOT/'runs'/sys.argv[2]/sys.argv[1]/'FAILED.json',dict(traceback=traceback.format_exc()));raise
