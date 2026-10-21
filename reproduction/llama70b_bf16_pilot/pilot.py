"""Official BF16 Llama-3.3-70B prefix feasibility pilot; never starts the full study."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TOKENIZERS_PARALLELISM']='false'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,time,hashlib,traceback,shutil,fcntl,copy,gc
from pathlib import Path
R=Path('/home/ubuntu/experiments/llama70b-bf16-pilot-20261005');R.mkdir(exist_ok=True)
REPO='meta-llama/Llama-3.3-70B-Instruct';REV='6f6073b423013f6a7d4d9f39144961bfbfbc386b';START=time.time()
def write(name,x):
 p=R/name;t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def status(event,**kw):
 d=dict(event=event,elapsed_seconds=time.time()-START,**kw);write('STATUS.json',d);print(json.dumps(d),flush=True)
def reserve(extra=0):assert shutil.disk_usage(R).free>20*2**30+extra,'Insufficient disk reserve'
def prefix_model(config,layers=55):
 import torch
 from transformers.models.llama.modeling_llama import LlamaModel,LlamaRotaryEmbedding
 config._attn_implementation='sdpa'
 with torch.device('meta'):m=LlamaModel(config)
 m.layers=torch.nn.ModuleList(list(m.layers[:layers]));m.norm=torch.nn.Identity();m.rotary_emb=LlamaRotaryEmbedding(config,device='cuda');return m

def main():
 import torch,numpy as np
 from transformers import AutoConfig,AutoTokenizer
 from safetensors import safe_open
 from huggingface_hub import HfApi,hf_hub_download,snapshot_download
 from accelerate import dispatch_model
 lock=(R/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'COMPLETED.json').exists(),'Pilot already complete; preserve results'
 torch.manual_seed(20261005);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 status('checking_official_model_access',repo=REPO,revision=REV)
 cfgpath=Path(hf_hub_download(REPO,'config.json',revision=REV));root=cfgpath.parent
 idxpath=Path(hf_hub_download(REPO,'model.safetensors.index.json',revision=REV));index=json.loads(idxpath.read_text())['weight_map']
 needed={k:v for k,v in index.items() if k=='model.embed_tokens.weight' or (k.startswith('model.layers.') and int(k.split('.')[2])<55)};shards=sorted(set(needed.values()))
 info=HfApi().model_info(REPO,revision=REV,files_metadata=True);sizes={f.rfilename:f.size for f in info.siblings};missing=sum(sizes[s] for s in shards if not (root/s).exists());reserve(missing)
 manifest=dict(repo=REPO,revision=REV,block_index=54,executed_blocks=55,original_blocks=80,dtype='bfloat16',quantization=None,resident_gpu_blocks=17,weight_keys=sorted(needed),shards=shards,shard_bytes=sum(sizes[s] for s in shards),config_sha256=hashlib.sha256(cfgpath.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 write('PLAN.json',manifest);status('downloading_prefix_shards',shards=len(shards),missing_gib=missing/2**30)
 snapshot_download(REPO,revision=REV,allow_patterns=shards+['tokenizer*','special_tokens_map.json','chat_template*'],max_workers=2)
 reserve();status('preparing_benchmark_examples')
 tok=AutoTokenizer.from_pretrained(root,local_files_only=True);special=set(tok.all_special_ids)
 # Reuse audited content-mask rules with the new tokenizer and native template.
 import ast
 source=Path('/home/ubuntu/experiments/logistic-probes-20261001/train.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='content_mask');namespace=dict(np=np,json=json,tokenizer=tok,special_ids=special,config={'template_date_string':'01 Oct 2026'});exec(compile(ast.Module(body=[fn],type_ignores=[]),'audited_content_mask','exec'),namespace)
 data=[];population={};B=Path('/home/ubuntu/experiments/reproducible-sampled-20261004/llama')
 for ds in ['wildguard','high_stakes','toxicchat','beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']:
  rows=json.loads((B/(ds+'_metadata.json')).read_text());ordered=sorted(rows,key=lambda r:(r['n_tokens'],r['immutable_id']));population[ds]=dict(rows=len(rows),old_tokens=sum(r['n_tokens'] for r in rows));ratios=[]
  for q in [.1,.5,.9,1.]:
   r=ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))];text=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026');ids=tok(text,add_special_tokens=False)['input_ids'];proxy='wildguard' if r['role']=='assistant' else ('toxicchat' if r['role']=='user' else 'high_stakes');mask=namespace['content_mask'](r['messages'],np.array(ids),proxy);assert mask.any();assert not any(ids[i] in special for i in np.flatnonzero(mask))
   data.append(dict(dataset=ds,id=r['immutable_id'],token_ids=ids,mask=np.flatnonzero(mask).tolist(),n_tokens=len(ids)));ratios.append(len(ids)/r['n_tokens'])
  population[ds]['pilot_token_ratio_mean']=sum(ratios)/len(ratios)
  del rows,ordered
 write('BENCHMARK_INPUTS.json',data);write('POPULATION.json',population)
 config=AutoConfig.from_pretrained(root,local_files_only=True);assert config.num_hidden_layers==80 and config.hidden_size==8192
 m=prefix_model(config);expected=set(m.state_dict());loaded=set();status('loading_bf16_prefix')
 for shard in shards:
  with safe_open(str(root/shard),framework='pt',device='cpu') as sf:
   for name in sf.keys():
    if name not in needed:continue
    key=name.removeprefix('model.');assert key in expected;module_name,attr=key.rsplit('.',1);module=m.get_submodule(module_name);v=sf.get_tensor(name);assert v.dtype==torch.bfloat16 and tuple(v.shape)==tuple(getattr(module,attr).shape)
    device='cuda' if key.startswith('embed_tokens.') or (key.startswith('layers.') and int(key.split('.')[1])<17) else 'cpu'
    setattr(module,attr,torch.nn.Parameter(v.to(device),requires_grad=False));loaded.add(key)
  status('loading_bf16_prefix',shard=shard,loaded_tensors=len(loaded),total_tensors=len(expected))
 assert expected==loaded and not any(p.is_meta for p in m.parameters());assert all(p.dtype==torch.bfloat16 for p in m.parameters())
 before=dict(gpu_parameter_gib=sum(p.numel()*p.element_size() for p in m.parameters() if p.is_cuda)/2**30,cpu_parameter_gib=sum(p.numel()*p.element_size() for p in m.parameters() if not p.is_cuda)/2**30)
 devices={'embed_tokens':'cuda:0','norm':'cuda:0','rotary_emb':'cuda:0',**{f'layers.{i}':'cuda:0' if i<17 else 'cpu' for i in range(55)}}
 m=dispatch_model(m,device_map=devices,main_device='cuda:0',force_hooks=True).eval();write('LOADED.json',dict(**before,device_map=devices,seconds=time.time()-START));status('benchmarking',**before)
 def forward(batch):
  width=max(r['n_tokens'] for r in batch);ids=torch.full((len(batch),width),tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
  for j,r in enumerate(batch):ids[j,:r['n_tokens']]=torch.tensor(r['token_ids'],device='cuda');att[j,:r['n_tokens']]=1
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  with torch.inference_mode():h=m(input_ids=ids,attention_mask=att,use_cache=False).last_hidden_state
  assert h.shape[-1]==8192 and torch.isfinite(h).all();features=torch.stack([h[j,r['mask']].float().mean(0) for j,r in enumerate(batch)]).cpu()
  torch.cuda.synchronize();seconds=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()/2**30;del h,ids,att
  return features,dict(seconds=seconds,batch_size=len(batch),padded_tokens=len(batch)*width,useful_tokens=sum(r['n_tokens'] for r in batch),max_length=width,peak_gpu_gib=peak)
 ordered=sorted(data,key=lambda r:(r['n_tokens'],r['id']));test=ordered[:2]
 f0,warm=forward(test);f1,repeated=forward(test);assert torch.equal(f0,f1),'Same-batch repeat not bitwise identical; inspect before full extraction';np.save(R/'repeat_features.npy',f1.numpy())
 measured=[];group=[]
 for r in ordered:
  if group and (len(group)>=8 or (len(group)+1)*r['n_tokens']>8192):
   f,stats=forward(group);measured.append(stats);write('BENCHMARK_PARTIAL.json',measured);status('benchmark_batch',batch=len(measured),**stats);group=[]
  group.append(r)
 if group:f,stats=forward(group);measured.append(stats);status('benchmark_batch',batch=len(measured),**stats)
 # Identical real content at batch sizes 1, 4 and 8 measures transfer amortization.
 mid=ordered[len(ordered)//2];scaling=[]
 for n in [1,4,8]:
  if n*mid['n_tokens']>16384:continue
  f,stats=forward([mid]*n);scaling.append(stats)
  if n==1:reference=f[0]
  else:assert torch.nn.functional.cosine_similarity(reference[None],f,dim=1).min()>.999,'Unexpected batch-size numerical drift'
 total_tokens=sum(x['old_tokens']*x['pilot_token_ratio_mean'] for x in population.values());rate=sum(x['useful_tokens'] for x in measured)/sum(x['seconds'] for x in measured)
 result=dict(model=manifest,loaded=before,warmup=warm,repeat=repeated,repeat_bitwise_equal=True,benchmark_batches=measured,batch_scaling=scaling,estimated_full_tokens=total_tokens,representative_useful_tokens_per_second=rate,rough_full_extraction_hours=total_tokens/rate/3600,scope='Feasibility pilot only; ETA extrapolates 40 length-stratified examples, not a calibrated full-corpus benchmark. Batch scheduling affects PCIe overhead; no full extraction or judge run launched.',free_disk_gib=shutil.disk_usage(R).free/2**30,total_seconds=time.time()-START)
 write('RESULTS.json',result);write('COMPLETED.json',dict(time=time.time(),results_sha256=hashlib.sha256((R/'RESULTS.json').read_bytes()).hexdigest()));status('completed',rough_extraction_hours=result['rough_full_extraction_hours'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  write('FAILED_'+str(time.time_ns())+'.json',dict(error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));status('blocked_access' if type(e).__name__=='GatedRepoError' else 'failed',error_type=type(e).__name__,message=str(e));raise
