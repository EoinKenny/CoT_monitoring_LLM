import os
os.environ['OMP_NUM_THREADS']='8';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,time,torch,numpy as np
from transformers import AutoModel,AutoTokenizer
from tokenize_rows import tokenize_record
R=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');B=R.parent/'ten-dataset-full-20261002'
b=json.loads((R/'backbone.json').read_text());tok=AutoTokenizer.from_pretrained(b['path'],local_files_only=True);rows=[];counts={}
for ds in ['wildguard','high_stakes','toxicchat','beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']:
 data=json.loads((B/f'{ds}_metadata.json').read_text());ix=sorted(set([0,len(data)//2,len(data)-1]+np.random.default_rng(20261002).choice(len(data),5,replace=False).tolist()))
 for i in ix:rows.append(dict(dataset=ds,**tokenize_record(tok,data[i])))
 counts[ds]=len(ix)
torch.set_num_threads(8);torch.manual_seed(20261002)
model=AutoModel.from_pretrained(b['path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();assert model.config.hidden_size==2560 and model.config.num_hidden_layers==36
cache={}
def hook(m,i,o):cache['h']=(o[0] if isinstance(o,tuple) else o).detach()
model.layers[24].register_forward_hook(hook)
start=time.time();torch.cuda.reset_peak_memory_stats()
for offset in range(0,len(rows),4):
 batch=rows[offset:offset+4];width=max(r['n_tokens'] for r in batch);ids=torch.full((len(batch),width),tok.pad_token_id,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
 for i,r in enumerate(batch):ids[i,:r['n_tokens']]=torch.tensor(r['token_ids'],device='cuda');att[i,:r['n_tokens']]=1
 with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
 h=cache.pop('h');assert torch.isfinite(h).all()
 for i,r in enumerate(batch):assert h[i,r['mask']].mean(0).shape==(2560,)
 del h,ids,att
report=dict(rows=len(rows),datasets=counts,seconds=time.time()-start,tokens=sum(r['n_tokens'] for r in rows),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,max_tokens=max(r['n_tokens'] for r in rows),block_index=24)
(R/'SMOKE_PASSED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
