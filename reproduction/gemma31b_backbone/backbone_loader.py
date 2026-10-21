"""Load the exact causal prefix needed for a pre-final-norm layer-41 probe."""
import json,torch,hashlib
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel,Gemma4TextRotaryEmbedding
R=Path('/home/ubuntu/experiments/gemma31b-backbone-20261003')

class CPUEmbedding(torch.nn.Module):
 def __init__(self,embedding):super().__init__();self.embedding=embedding
 def forward(self,ids):return self.embedding(ids.cpu()).to(ids.device)

def load_model():
 b=json.loads((R/'backbone.json').read_text());path=Path(b['path'])
 config=AutoConfig.from_pretrained(path,local_files_only=True).text_config
 assert config.num_hidden_layers==60 and config.hidden_size==5376
 assert not config.hidden_size_per_layer_input and not config.num_kv_shared_layers
 config._attn_implementation='sdpa'
 with torch.device('meta'):model=Gemma4TextModel(config)
 model.layers=torch.nn.ModuleList(list(model.layers[:41]))
 # Final normalization is downstream of the requested block output.
 model.norm=torch.nn.Identity()
 model.rotary_emb=Gemma4TextRotaryEmbedding(config,device='cuda')
 expected=set(model.state_dict());loaded=set();prefix='model.language_model.'
 for shard in b['loaded_shards']:
  with safe_open(str(path/shard),framework='pt',device='cpu') as f:
   for original in f.keys():
    if original not in b['weight_keys']:continue
    key=original[len(prefix):];assert key in expected
    value=f.get_tensor(original)
    module_name,attr=key.rsplit('.',1);module=model.get_submodule(module_name)
    assert tuple(getattr(module,attr).shape)==tuple(value.shape)
    device='cpu' if key=='embed_tokens.weight' else 'cuda'
    if attr in module._parameters:
     assert value.dtype==torch.bfloat16
     setattr(module,attr,torch.nn.Parameter(value.to(device),requires_grad=False))
    else:setattr(module,attr,value.to(device))
    loaded.add(key)
 assert loaded==expected,(len(loaded),len(expected),sorted(expected-loaded))
 assert not any(p.is_meta for p in model.parameters())
 model.embed_tokens.embed_scale=torch.tensor(model.embed_tokens.scalar_embed_scale,dtype=torch.float32)
 assert not any(b.is_meta for b in model.buffers())
 model.embed_tokens=CPUEmbedding(model.embed_tokens)
 model.eval()
 report=dict(loaded_parameters=sum(p.numel() for p in model.parameters()),loaded_parameter_tensors=len(loaded),dtype='bfloat16',quantized=False,original_config_layers=60,executed_blocks=41,block_index=40,embedding_device='cpu',gpu_parameter_gib=sum(p.numel()*p.element_size() for p in model.parameters() if p.is_cuda)/2**30,config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest())
 (R/'PREFIX_LOADER_VERIFIED.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 return model
