import os
os.environ.update(OMP_NUM_THREADS='8',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13')
from pathlib import Path
import json,time,hashlib,sys,subprocess,re,shutil
P=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009');R=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009');O=Path('/home/ubuntu/experiments/combined-output-smoke-20261012');CS=['baseline','saliency','nearest']
OLD='Return exactly one digit: 1 for the positive class or 0 for the negative class. Do not include explanations or other text.'
NEW='Return two integers separated by a comma: the predicted class (0 or 1), then your estimated probability from 0 to 100 that the CURRENT conversation belongs to the positive class defined by the task. 0 means certainly negative, 50 means equally likely, and 100 means certainly positive. The probability is for the positive class, not confidence in your predicted label. Format: label,probability. Do not include spaces, explanations, or any other text.'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def jl(p):return [json.loads(l) for l in p.open()]
def prepare():
 O.mkdir(exist_ok=False);rows=read(P/'llama/selected_flags.json');selected=[]
 for ds in sorted({r['dataset'] for r in rows}):
  pool=sorted([r for r in rows if r['dataset']==ds and r['split']=='val'],key=lambda r:hashlib.sha256(('combined-smoke-20261012:'+r['sample_id']).encode()).hexdigest());selected+=pool[:3]
 extra=sorted([r for r in rows if r['split']=='val' and r not in selected],key=lambda r:hashlib.sha256(('combined-extra:'+r['sample_id']).encode()).hexdigest())[:2];selected+=extra
 keys=[(r['sample_id'],CS[i%3]) for i,r in enumerate(selected)];write(O/'SELECTION.json',dict(keys=keys,n=32,seed=20261012,backbone='llama',split='val',note='Three cases per dataset plus two hash-selected extras; explanation types rotate. Same cases and batches across formats and judges.'))
 for j in ['qwen','gemma']:
  out={};sources={}
  for mode,root in [('hard',P),('probability',R)]:
   for c in CS:
    f=root/'llama/judges'/f'{j}_{c}'/'prompts.jsonl';sources[str(f)]=sha(f)
    for row in jl(f):
     if (row['sample_id'],c) in keys:out[(mode,row['sample_id'],c)]=row
  records=[]
  for sid,c in keys: records.append(dict(sample_id=sid,condition=c,hard=out['hard',sid,c],probability=out['probability',sid,c]))
  # Balance length mix across both batches; same order for the two judges.
  if j=='qwen':
   ordered=sorted(range(32),key=lambda i:records[i]['hard']['input_tokens']);order=ordered[::2]+ordered[1::2];write(O/'ORDER.json',order)
  order=read(O/'ORDER.json');write(O/f'{j}_inputs.json',[records[i] for i in order]);write(O/f'{j}_sources.json',sources)
 write(O/'PROTOCOL.json',dict(n_per_judge_per_format=32,batches_per_format=2,batch_size=16,formats=['hard','probability','combined'],combined_instruction=NEW,combined_regex='[01],(?:0|[1-9][0-9]?|100)',warmup='One explicit warmup request per format, separately recorded and excluded from steady timing.',order=[['hard','probability','combined'],['combined','probability','hard']],prefix_caching=False,seed=20261001,no_retries=True))
def run(j):
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 D=O/j;D.mkdir(exist_ok=False);rows=read(O/f'{j}_inputs.json');cfg=read(R/'llama/judges'/f'{j}_baseline'/'run/config.json');engine=dict(cfg['engine']);tok=AutoTokenizer.from_pretrained(engine['model'],local_files_only=True)
 for r in rows:
  p=r['hard']['prompt'];assert p.count(OLD)==1;footer='Classification of the CURRENT conversation (0 or 1):' if r['condition']=='nearest' else 'Classification (0 or 1):';idx=p.rindex(footer);new=p[:idx]+'Classification and positive-class probability (label,probability):'+p[idx+len(footer):];new=new.replace(OLD,NEW,1);tokens=tok.encode(new,add_special_tokens=False);r['combined']=dict(prompt=new,token_ids=tokens,input_tokens=len(tokens))
 engine['max_model_len']=max(3584,((max(r[m]['input_tokens'] for r in rows for m in ['hard','probability','combined'])+32+255)//256)*256)
 params={
 'hard':SamplingParams(temperature=0,seed=20261001,max_tokens=1,allowed_token_ids=[tok.encode(str(i),add_special_tokens=False)[0] for i in [0,1]]),
 'probability':SamplingParams(temperature=0,seed=20261001,max_tokens=8,structured_outputs=StructuredOutputsParams(choice=[str(i) for i in range(101)])),
 'combined':SamplingParams(temperature=0,seed=20261001,max_tokens=16,structured_outputs=StructuredOutputsParams(regex=r'[01],(?:0|[1-9][0-9]?|100)'))}
 write(D/'CONFIG.json',dict(engine=engine,model=cfg['model'],revision=cfg['revision'],source_config_sha256=sha(R/'llama/judges'/f'{j}_baseline'/'run/config.json'),code_sha256=sha(Path(__file__)),versions=cfg['versions'],hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True),sampling={m:str(p) for m,p in params.items()}));write(D/'PROMPTS.json',rows)
 assert shutil.disk_usage(O).free>20*1024**3
 write(D/'STATUS.json',dict(state='loading'));t=time.perf_counter();llm=LLM(**engine);load=time.perf_counter()-t;batchresults=[]
 def generate(chunk,mode,name):
  write(D/(name+'.attempt.json'),dict(time=time.time(),mode=mode,ids=[r['sample_id'] for r in chunk]));t=time.perf_counter();res=llm.generate([{'prompt_token_ids':r[mode]['token_ids']} for r in chunk],params[mode],use_tqdm=False);elapsed=time.perf_counter()-t
  raw=[dict(sample_id=r['sample_id'],condition=r['condition'],text=x.outputs[0].text,token_ids=list(x.outputs[0].token_ids),finish_reason=x.outputs[0].finish_reason,input_tokens=len(x.prompt_token_ids)) for r,x in zip(chunk,res)];write(D/(name+'.raw.json'),raw);assert len(raw)==len(chunk)
  pattern={'hard':r'[01]','probability':r'(?:0|[1-9][0-9]?|100)','combined':r'[01],(?:0|[1-9][0-9]?|100)'}[mode]
  assert all(re.fullmatch(pattern,x['text'].strip()) for x in raw)
  if mode!='hard':assert all(x['finish_reason']=='stop' for x in raw)
  rec=dict(mode=mode,name=name,n=len(raw),seconds=elapsed,input_tokens=sum(x['input_tokens'] for x in raw),output_tokens=sum(len(x['token_ids']) for x in raw),raw_sha256=sha(D/(name+'.raw.json')));write(D/(name+'.json'),rec);return rec
 for m in params:generate(rows[:1],m,'warmup_'+m)
 for bi,order in enumerate([['hard','probability','combined'],['combined','probability','hard']]):
  for m in order:
   rec=generate(rows[bi*16:(bi+1)*16],m,f'batch{bi}_{m}');batchresults.append(rec);write(D/'STATUS.json',dict(state='running',completed_batches=len(batchresults)))
 totals={m:{k:sum(r[k] for r in batchresults if r['mode']==m) for k in ['seconds','input_tokens','output_tokens','n']} for m in params};separate=totals['hard']['seconds']+totals['probability']['seconds'];combined=totals['combined']['seconds'];raws={m:sum([read(D/f'batch{i}_{m}.raw.json') for i in [0,1]],[]) for m in params};agreements=sum(int(a['text'])==int(b['text'].split(',')[0]) for a,b in zip(raws['hard'],raws['combined']));mae=sum(abs(int(a['text'])-int(b['text'].split(',')[1])) for a,b in zip(raws['probability'],raws['combined']))/32
 write(D/'COMPLETED.json',dict(judge=j,n_cases=32,load_seconds=load,totals=totals,separate_seconds=separate,combined_seconds=combined,speedup=separate/combined,saving_fraction=1-combined/separate,hard_label_agreement=agreements,probability_mae=mae,batches=batchresults));write(D/'STATUS.json',dict(state='completed'))
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:run(sys.argv[1])
 except Exception as e:
  if O.exists():write(O/('FAILED_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=__import__('traceback').format_exc()))
  raise
