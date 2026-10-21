import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TOKENIZERS_PARALLELISM']='false'
import time,json
from common import *
import numpy as np,torch
from transformers import AutoConfig,AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaModel,LlamaRotaryEmbedding
from safetensors import safe_open
from accelerate import dispatch_model

def configure():
 import random
 random.seed(SEED);torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
def load_model():
 configure();plan=read(PILOT/'PLAN.json');assert sha(MODEL/'config.json')==plan['config_sha256'];assert plan['revision']==MODEL.name
 cfg=AutoConfig.from_pretrained(MODEL,local_files_only=True);assert cfg.num_hidden_layers==80 and cfg.hidden_size==8192;cfg._attn_implementation='sdpa'
 with torch.device('meta'):model=LlamaModel(cfg)
 model.layers=torch.nn.ModuleList(list(model.layers[:55]));model.norm=torch.nn.Identity();model.rotary_emb=LlamaRotaryEmbedding(cfg,device='cuda');expected=set(model.state_dict());loaded=set();needed=set(plan['weight_keys'])
 for shard in plan['shards']:
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for name in f.keys():
    if name not in needed:continue
    key=name.removeprefix('model.');module,attr=key.rsplit('.',1);sub=model.get_submodule(module);v=f.get_tensor(name);assert v.dtype==torch.bfloat16 and tuple(v.shape)==tuple(getattr(sub,attr).shape)
    gpu=key.startswith('embed_tokens.') or (key.startswith('layers.') and int(key.split('.')[1])<17)
    setattr(sub,attr,torch.nn.Parameter(v.to('cuda' if gpu else 'cpu'),requires_grad=False));loaded.add(key)
  print(json.dumps(dict(event='load_shard',shard=shard,tensors=len(loaded))),flush=True)
 assert loaded==expected and not any(p.is_meta for p in model.parameters())
 devices={'embed_tokens':'cuda:0','norm':'cuda:0','rotary_emb':'cuda:0',**{f'layers.{i}':'cuda:0' if i<17 else 'cpu' for i in range(55)}}
 return dispatch_model(model,device_map=devices,main_device='cuda:0',force_hooks=True).eval()

def forward(model,rows,token_callback=None):
 width=max(r['n_tokens'] for r in rows);ids=torch.full((len(rows),width),128009,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
 for j,r in enumerate(rows):ids[j,:r['n_tokens']]=torch.tensor(r['token_ids'],device='cuda');att[j,:r['n_tokens']]=1
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 with torch.inference_mode():h=model(input_ids=ids,attention_mask=att,use_cache=False).last_hidden_state
 assert h.shape[-1]==8192 and torch.isfinite(h).all()
 features=torch.stack([h[j,r['mask']].float().mean(0) for j,r in enumerate(rows)]).cpu().numpy()
 if token_callback is not None:
  for j,r in enumerate(rows):token_callback(r,h[j,:r['n_tokens']])
 torch.cuda.synchronize();stats=dict(seconds=time.perf_counter()-t,batch_size=len(rows),padded_tokens=len(rows)*width,useful_tokens=sum(r['n_tokens'] for r in rows),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
 return features,stats

def batches(rows,max_batch,max_tokens):
 order=sorted(range(len(rows)),key=lambda i:(rows[i]['n_tokens'],rows[i].get('immutable_id',rows[i].get('id',''))));out=[];chunk=[]
 for i in order:
  if chunk and (len(chunk)>=max_batch or (len(chunk)+1)*rows[i]['n_tokens']>max_tokens):out.append(chunk);chunk=[]
  chunk.append(i)
 if chunk:out.append(chunk)
 return out
