import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['CUDA_HOME']='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13'
from pathlib import Path
import json,time,hashlib,re,shutil,subprocess,threading,traceback,statistics
R=Path('/home/ubuntu/experiments/gptoss120-smoke-20261011/attempt3_triton')
def save(name,x):
 p=R/name; t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def gpu():
 return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
def guard():assert shutil.disk_usage(R).free>20*1024**3
samples=[];stop=threading.Event()
def monitor():
 with (R/'gpu_samples.jsonl').open('a') as f:
  while not stop.is_set():
   f.write(json.dumps(dict(time=time.time(),gpu=gpu()))+'\n');f.flush();stop.wait(1)
def main():
 from transformers import AutoTokenizer
 import vllm,torch,transformers
 from vllm import LLM,SamplingParams
 guard();assert not (R/'STARTED.json').exists(),'Preserve previous attempt; do not rerun in place'
 save('STARTED.json',dict(time=time.time(),pid=os.getpid()))
 model=(R/'model_path.txt').read_text().strip();tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
 source=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009/smoke/qwen/prompts.jsonl');pool=[json.loads(x) for x in source.open()];selected=[]
 for c in ['baseline','saliency','nearest']:
  rs=sorted([r for r in pool if r['condition']==c],key=lambda r:(r['input_tokens'],r['sample_id']))
  for k in sorted(set(round(i*(len(rs)-1)/7) for i in range(8))):selected.append(rs[k])
 probability_instruction=json.loads(Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009/PROTOCOL.json').read_text())['output_instruction']
 label_instruction='Return exactly one digit: 1 for the positive class or 0 for the negative class. Do not include explanations or other text.'
 prompts=[]
 for row in selected:
  text=row['prompt'];head='<|im_start|>system\n';sep='<|im_end|>\n<|im_start|>user\n';tail='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  assert text.startswith(head) and text.endswith(tail);sy,us=text[len(head):-len(tail)].split(sep,1)
  for mode in ['hard','probability']:
   system=sy;user=us
   if mode=='hard':
    assert probability_instruction in sy;system=sy.replace(probability_instruction,label_instruction,1)
    footer='Classification of the CURRENT conversation (0 or 1):' if row['condition']=='nearest' else 'Classification (0 or 1):'
    assert user.endswith('Positive-class probability (0-100):');user=user[:-len('Positive-class probability (0-100):')]+footer
   messages=[dict(role='system',content=system),dict(role='user',content=user)]
   rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,reasoning_effort='low');ids=tok.encode(rendered,add_special_tokens=False)
   assert 'Reasoning: low' in rendered
   prompts.append(dict(id=row['sample_id']+':'+mode,sample_id=row['source_sample_id'],backbone=row['backbone'],dataset=row['dataset'],condition=row['condition'],mode=mode,messages=messages,prompt=rendered,token_ids=ids,input_tokens=len(ids)))
 (R/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts))
 maxin=max(r['input_tokens'] for r in prompts);maxlen=((maxin+2048+1023)//1024)*1024
 cfg=dict(flashinfer_sampler=False,cuda_home=os.environ['CUDA_HOME'],attention_backend='TRITON_ATTN',model=model,revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a',reasoning_effort='low',seed=20261001,max_tokens=2048,temperature=0,max_model_len=maxlen,max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.90,enable_prefix_caching=False,enforce_eager=True,quantization='official MXFP4 automatic',vllm=vllm.__version__,torch=torch.__version__,transformers=transformers.__version__,n=len(prompts),max_input_tokens=maxin,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prompt_sha256=hashlib.sha256((R/'prompts.jsonl').read_bytes()).hexdigest(),note='Isolated feasibility benchmark; eager execution, no prefix caching. No forced final answer or constrained reasoning; native Harmony and low effort. Repeated inputs across batch-size benchmarks are intentional and individually archived; not evaluation judgments.')
 save('CONFIG.json',cfg);save('STATUS.json',dict(state='loading',time=time.time(),max_input=maxin,max_model_len=maxlen));thread=threading.Thread(target=monitor,daemon=True);thread.start();t=time.perf_counter()
 llm=LLM(attention_backend='TRITON_ATTN',model=model,tokenizer=model,dtype='auto',max_model_len=maxlen,max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.90,enable_prefix_caching=False,enforce_eager=True,seed=20261001,trust_remote_code=False)
 save('LOADED.json',dict(seconds=time.perf_counter()-t,gpu=gpu()));params=SamplingParams(temperature=0,seed=20261001,max_tokens=2048,skip_special_tokens=False)
 batches=[]
 def run_batch(tag,rows):
  guard();attempt=R/(tag+'.attempt.json');assert not attempt.exists();attempt.write_text(json.dumps(dict(time=time.time(),ids=[r['id'] for r in rows])))
  t=time.perf_counter();out=llm.generate([dict(prompt_token_ids=r['token_ids']) for r in rows],params,use_tqdm=False);elapsed=time.perf_counter()-t
  raw=[dict(id=r['id'],text=o.outputs[0].text,token_ids=list(o.outputs[0].token_ids),finish_reason=o.outputs[0].finish_reason,stop_reason=o.outputs[0].stop_reason) for r,o in zip(rows,out)];save(tag+'.raw.json',raw)
  parsed=[]
  for r,o in zip(rows,raw):
   text=o['text'];match=re.search(r'<\|start\|>assistant<\|meta_sep\|>final<\|message\|>(.*?)<\|fim_suffix\|>',text,re.S)
   if not match:match=re.search(r'(?:^|<\|meta_sep\|>)final<\|message\|>(.*?)<\|fim_suffix\|>',text,re.S)
   final=match.group(1).strip() if match else None;valid=bool(final is not None and re.fullmatch(r'\d{1,3}',final) and int(final)<=(1 if r['mode']=='hard' else 100))
   parsed.append(dict(id=r['id'],condition=r['condition'],mode=r['mode'],input_tokens=r['input_tokens'],generated_tokens=len(o['token_ids']),final=final,valid=valid,finish_reason=o['finish_reason']))
  result=dict(tag=tag,n=len(rows),seconds=elapsed,calls_per_second=len(rows)/elapsed,input_tokens=sum(r['input_tokens'] for r in rows),generated_tokens=sum(r['generated_tokens'] for r in parsed),valid=sum(r['valid'] for r in parsed),records=parsed,gpu=gpu());save(tag+'.result.json',result);batches.append(result);save('STATUS.json',dict(state='benchmarking',last=tag,batches=len(batches),time=time.time()));return result
 # Warm-up is separate. The same 48 frozen prompts are then measured at each batch size.
 run_batch('warmup',[prompts[0]])
 for bs in [1,8,16]:
  rows=prompts if bs>1 else [prompts[i] for i in [0,1,16,17,32,33]]
  for i in range(0,len(rows),bs):run_batch(f'batch{bs:02d}_{i//bs:03d}',rows[i:i+bs])
 stop.set();thread.join(timeout=3)
 results=[]
 for bs in [1,8,16]:
  rr=[r for r in batches if r['tag'].startswith(f'batch{bs:02d}_')];sec=sum(r['seconds'] for r in rr);n=sum(r['n'] for r in rr);results.append(dict(batch_size=bs,calls=n,seconds=sec,calls_per_second=n/sec,seconds_per_call=sec/n,valid=sum(r['valid'] for r in rr),generated_tokens=sum(r['generated_tokens'] for r in rr),input_tokens=sum(r['input_tokens'] for r in rr)))
 save('RESULTS.json',dict(benchmarks=results,load=json.loads((R/'LOADED.json').read_text()),peak_gpu_mib=max(int(json.loads(l)['gpu'].split(',')[0]) for l in (R/'gpu_samples.jsonl').open()),max_input_tokens=maxin,free_disk_gib=shutil.disk_usage(R).free/1024**3))
 save('COMPLETED.json',dict(time=time.time(),results_sha256=hashlib.sha256((R/'RESULTS.json').read_bytes()).hexdigest()));save('STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  stop.set();save('FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
