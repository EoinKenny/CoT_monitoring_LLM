import os
os.environ['OMP_NUM_THREADS']='8';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,torch,numpy as np,copy
from pathlib import Path
from transformers import AutoTokenizer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from backbone_loader import load_model,CPUEmbedding
R=Path('/home/ubuntu/experiments/gemma31b-backbone-20261003')
torch.set_num_threads(8);torch.manual_seed(20261002)
# Verify prefix execution equals the corresponding block in a full causal stack.
c=Gemma4TextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=60,num_attention_heads=4,num_key_value_heads=2,num_global_key_value_heads=2,head_dim=16,global_head_dim=32,hidden_size_per_layer_input=0,num_kv_shared_layers=0)
c._attn_implementation='sdpa';full=Gemma4TextModel(c).eval();prefix=copy.deepcopy(full);prefix.layers=torch.nn.ModuleList(list(prefix.layers[:41]));prefix.norm=torch.nn.Identity();seen={}
full.layers[40].register_forward_hook(lambda m,i,o:seen.update(h=o.detach()))
x=torch.tensor([[2,14,23,44,1]])
with torch.inference_mode():full(input_ids=x,use_cache=False);actual=prefix(input_ids=x,use_cache=False).last_hidden_state
assert torch.equal(actual,seen['h']);del full,prefix,seen
model=load_model();cache={}
model.layers[40].register_forward_hook(lambda m,i,o:cache.update(h=o.detach()))
tok=AutoTokenizer.from_pretrained(json.loads((R/'backbone.json').read_text())['path'],local_files_only=True)
bench=[];total_tokens=0;total_padded=0;total_cache=0;max_tokens=0
for p in sorted(R.glob('*_selected.json')):
 rows=json.loads(p.read_text());order=sorted(range(len(rows)),key=lambda i:rows[i]['n_tokens']);batches=[];chunk=[]
 for i in order:
  if chunk and (len(chunk)>=4 or (len(chunk)+1)*rows[i]['n_tokens']>2048):batches.append(chunk);chunk=[]
  chunk.append(i)
 if chunk:batches.append(chunk)
 total_padded+=sum(len(ch)*max(rows[i]['n_tokens'] for i in ch) for ch in batches)
 total_tokens+=sum(r['n_tokens'] for r in rows);total_cache+=sum(r['n_tokens']*5376*2 for r in rows if r['split']=='test')
 max_tokens=max(max_tokens,max(r['n_tokens'] for r in rows))
 # Include the longest batch plus spread through each dataset's length distribution.
 chosen=sorted(set(np.linspace(0,len(batches)-1,8,dtype=int).tolist()))
 for k in chosen:bench.append((p.stem,[rows[i] for i in batches[k]]))
del rows,batches
import shutil
assert shutil.disk_usage(R).free>total_cache+20*2**30,'Insufficient disk for full test activation cache and working files'
seconds=0;padded=0;reports=[];torch.cuda.reset_peak_memory_stats()
for ds,batch in bench:
 width=max(r['n_tokens'] for r in batch);ids=torch.full((len(batch),width),tok.pad_token_id,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
 for i,r in enumerate(batch):ids[i,:r['n_tokens']]=torch.tensor(r['token_ids'],device='cuda');att[i,:r['n_tokens']]=1
 torch.cuda.synchronize();start=time.time()
 with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
 h=cache.pop('h');assert torch.isfinite(h).all()
 for i,r in enumerate(batch):assert h[i,r['mask']].float().mean(0).shape==(5376,)
 torch.cuda.synchronize();elapsed=time.time()-start;seconds+=elapsed;padded+=ids.numel()
 reports.append(dict(dataset=ds,rows=len(batch),width=width,seconds=elapsed));print(json.dumps(reports[-1]),flush=True)
 del h,ids,att
report=dict(prefix_equivalence_exact=True,batches=len(bench),seconds=seconds,padded_tokens=padded,estimated_extraction_hours=total_padded/(padded/seconds)/3600,total_tokens=total_tokens,total_padded_tokens=total_padded,test_cache_gib=total_cache/2**30,max_tokens=max_tokens,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,batches_measured=reports)
(R/'SMOKE_PASSED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
