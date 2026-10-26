# Adapted from the original 2026-10-07 pinned FP8 prefix loader.
import pathlib, re
import torch
from safetensors import safe_open
from transformers import FineGrainedFP8Config
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel, Qwen3_5MoeTextRotaryEmbedding
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear
from .common import read, write, log

def load_model(plan, output):
    cfgraw=read(pathlib.Path(plan['model_path'])/'config.json');cfg=Qwen3_5MoeTextConfig(**cfgraw['text_config']);cfg._attn_implementation='sdpa';cfg._experts_implementation='grouped_mm'
    assert cfg.num_hidden_layers==48 and cfg.hidden_size==3072
    with torch.device('meta'):
        m=Qwen3_5MoeTextModel(cfg)
        m.layers=torch.nn.ModuleList(list(m.layers[:32]));m.norm=torch.nn.Identity()
        quant=FineGrainedFP8Config(**cfgraw['quantization_config'])
        skip=[n.removeprefix('model.language_model.') for n in quant.modules_to_not_convert]
        m=replace_with_fp8_linear(m,modules_to_not_convert=skip,quantization_config=quant,pre_quantized=True)
    m.to_empty(device='cuda');m.rotary_emb=Qwen3_5MoeTextRotaryEmbedding(cfg,device='cuda')
    params=dict(m.named_parameters());coverage={k:0 for k in params};seen=set();scale_promotions=0;index=read(pathlib.Path(plan['model_path'])/'model.safetensors.index.json')['weight_map']
    needed=set(plan['needed_weight_keys']);files=sorted({index[k] for k in needed})
    with torch.no_grad():
        for fi,name in enumerate(files):
            with safe_open(str(pathlib.Path(plan['model_path'])/name),framework='pt',device='cpu') as f:
                for key in f.keys():
                    if key not in needed:continue
                    assert key not in seen;seen.add(key);short=key.removeprefix('model.language_model.');value=f.get_tensor(key)
                    mt=re.fullmatch(r'(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)',short)
                    if mt:
                        prefix,ei,projection,kind=mt.groups();suffix='' if kind=='weight' else '_scale_inv'
                        target=prefix+('.down_proj' if projection=='down_proj' else '.gate_up_proj')+suffix
                        dest=params[target][int(ei)]
                        if projection!='down_proj':dest=dest[:dest.shape[0]//2] if projection=='gate_proj' else dest[dest.shape[0]//2:]
                        assert dest.shape==value.shape,(key,dest.shape,value.shape)
                        if dest.dtype!=value.dtype:
                            assert kind=='weight_scale_inv' and dest.dtype==torch.float32 and value.dtype in (torch.bfloat16,torch.float16),(key,dest.dtype,value.dtype)
                            scale_promotions+=1  # BF16/FP16 -> FP32 is an exact value-preserving promotion.
                        dest.copy_(value);coverage[target]+=1
                    else:
                        assert short in params,(key,'unexpected key');dest=params[short];assert dest.shape==value.shape
                        # Preserve checkpoint dtype, including FP32 state parameters.
                        if short.endswith('weight_scale_inv') and dest.dtype!=value.dtype:
                            assert dest.dtype==torch.float32 and value.dtype in (torch.bfloat16,torch.float16)
                            dest.copy_(value);scale_promotions+=1
                        elif dest.dtype!=value.dtype:
                            module_name,attr=short.rsplit('.',1);mod=m.get_submodule(module_name);p=torch.nn.Parameter(value.to('cuda'),requires_grad=False);setattr(mod,attr,p);params[short]=p
                        else:dest.copy_(value)
                        coverage[short]+=1
                    del value
            log(output, 'loading_fp8_prefix',shards=fi+1,total_shards=len(files),keys=len(seen))
    assert seen==needed
    for key,count in coverage.items():
        expected=512 if '.experts.gate_up_proj' in key else 256 if '.experts.down_proj' in key else 1
        assert count==expected,(key,count,expected)
    assert all(not p.is_meta for p in m.parameters()) and all(not b.is_meta for b in m.buffers())
    m.eval();m.requires_grad_(False)
    write(output/'LOADER_VERIFIED.json',dict(source_keys=len(seen),target_parameters=len(coverage),coverage=coverage,exact_scale_promotions_to_fp32=scale_promotions,parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()),revision=plan['revision'],block_index=31,remaining_layers_omitted=16,final_norm_omitted=True))
    return m
