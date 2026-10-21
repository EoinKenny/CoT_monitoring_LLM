"""Small synthetic GPU test of truncation and CPU-offload equivalence, not 70B feasibility."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,copy,json
from transformers import LlamaConfig,LlamaModel
from accelerate import dispatch_model
torch.manual_seed(20261005);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
c=LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=80,num_attention_heads=4,num_key_value_heads=2,vocab_size=64);c._attn_implementation='sdpa'
full=LlamaModel(c).to(device='cuda',dtype=torch.bfloat16).eval();ids=torch.tensor([[1,2,3,4,5,6],[4,5,6,7,8,9]],device='cuda');saved={}
def hook(m,i,o):saved['h']=(o[0] if isinstance(o,tuple) else o).clone()
handle=full.layers[54].register_forward_hook(hook)
with torch.inference_mode():full(input_ids=ids,use_cache=False)
expected=saved['h'];handle.remove();prefix=copy.deepcopy(full);prefix.layers=torch.nn.ModuleList(list(prefix.layers[:55]));prefix.norm=torch.nn.Identity()
with torch.inference_mode():actual=prefix(input_ids=ids,use_cache=False).last_hidden_state
assert torch.equal(expected,actual)
for layer in prefix.layers[17:]:layer.cpu()
devices={'embed_tokens':'cuda:0','norm':'cuda:0','rotary_emb':'cuda:0',**{f'layers.{i}':'cuda:0' if i<17 else 'cpu' for i in range(55)}}
prefix=dispatch_model(prefix,device_map=devices,main_device='cuda:0',force_hooks=True)
with torch.inference_mode():offloaded=prefix(input_ids=ids,use_cache=False).last_hidden_state
assert torch.equal(expected,offloaded)
print(json.dumps(dict(passed=True,full_block55_equals_prefix=True,cpu_offload_equals_gpu=True,dtype='bfloat16',scope='Synthetic 128-wide architecture, not a 70B capacity or throughput measurement')))
