"""Pinned downloads and the original pre-final-normalization backbone prefixes."""
import importlib.util
import os
from pathlib import Path
import re
import sys

from .common import ROOT, read, reserve, sha, write


def needed_key(key, spec):
    if not key.startswith(spec['prefix']):
        return False
    key = key[len(spec['prefix']):]
    match = re.match(r'layers\.(\d+)\.', key)
    return key.startswith('embed_tokens.') or bool(match and int(match[1]) <= spec['block_index'])


def model_path(kind, alias, cache, overrides=None):
    spec = read(ROOT/'configs'/('backbones.json' if kind == 'backbone' else 'local_judges.json'))[alias]
    override = (overrides or {}).get(kind, {}).get(alias)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path/'config.json').exists() and not (path/'params.json').exists():
            raise ValueError(f'Invalid model snapshot: {path}')
        if path.name != spec['revision']:
            raise ValueError(f'Model override must be the pinned snapshot directory {spec["revision"]}: {path}')
        # Explicit override is provenance-recorded; never guess a different model.
        return path, spec
    from huggingface_hub import snapshot_download
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Ask the Hub for a pinned inventory before downloading weights. This also
    # checks gated-model authorization without starting inference.
    if kind == 'backbone':
        patterns = ['*.json', '*.model', '*.tiktoken', '*.txt', '*.jinja']
        path = Path(snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=cache, allow_patterns=patterns))
        index = path/'model.safetensors.index.json'
        if index.exists():
            weights = read(index)['weight_map']
            shards = sorted({name for key, name in weights.items() if needed_key(key, spec)})
            if not shards:
                raise ValueError('Pinned checkpoint has no expected prefix weights')
        else:
            shards = ['*.safetensors']
        patterns += shards
    else:
        patterns = ['*.json', '*.safetensors', '*.model', '*.tiktoken', '*.txt', '*.jinja']
    plan = snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=cache, allow_patterns=patterns, dry_run=True)
    extra = sum(item.file_size for item in plan if item.will_download)
    reserve(cache, extra)
    path = Path(snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=cache, allow_patterns=patterns))
    return path, spec


def weight_map(path):
    index = path/'model.safetensors.index.json'
    if index.exists():
        return read(index)['weight_map']
    from safetensors import safe_open
    result = {}
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            result.update({key: shard.name for key in f.keys()})
    return result


def configure(fp8=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR'] = '1'
    import random
    import numpy as np
    import torch
    random.seed(20261004)
    np.random.seed(20261004)
    torch.manual_seed(20261004)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    if fp8:
        torch.set_default_dtype(torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        kernel = ROOT/'open_reproduction/vendor/finegrained_fp8'
        for name, h in read(kernel.parent/'kernel_manifest.json')['files'].items():
            if sha(kernel/name) != h:
                raise ValueError(f'Pinned kernel changed: {name}')
        spec = importlib.util.spec_from_file_location('probe_frozen_fp8', kernel/'__init__.py', submodule_search_locations=[str(kernel)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        from transformers.integrations.hub_kernels import _KERNEL_MODULE_MAPPING
        _KERNEL_MODULE_MAPPING['finegrained-fp8'] = mod


def load_backbone(alias, path, spec, out):
    import torch
    from transformers import AutoConfig
    from safetensors import safe_open
    configure(alias == 'qwen122')
    weights = weight_map(path)
    needed = {k for k in weights if needed_key(k, spec)}
    if alias == 'qwen122':
        from .fp8_prefix import load_model
        return load_model(dict(model_path=str(path), needed_weight_keys=sorted(needed), revision=spec['revision']), out)
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    if alias == 'gemma':
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as Model, Gemma4TextRotaryEmbedding as Rotary
        cfg = cfg.text_config
        if cfg.hidden_size_per_layer_input or cfg.num_kv_shared_layers:
            raise ValueError('Unsupported changed Gemma configuration')
    elif alias == 'qwen':
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as Model, Qwen3RotaryEmbedding as Rotary
    else:
        from transformers.models.llama.modeling_llama import LlamaModel as Model, LlamaRotaryEmbedding as Rotary
    cfg._attn_implementation = 'sdpa'
    with torch.device('meta'):
        model = Model(cfg)
    model.layers = torch.nn.ModuleList(list(model.layers[:spec['block_index']+1]))
    model.norm = torch.nn.Identity()
    model.rotary_emb = Rotary(cfg, device='cuda')
    expected = set(model.state_dict())
    loaded = set()
    for shard in sorted({weights[k] for k in needed}):
        with safe_open(str(path/shard), framework='pt', device='cpu') as f:
            for name in f.keys():
                if name not in needed:
                    continue
                key = name[len(spec['prefix']):]
                if key not in expected:
                    raise ValueError(f'Unexpected checkpoint tensor: {name}')
                module_name, attr = key.rsplit('.', 1)
                module = model.get_submodule(module_name)
                value = f.get_tensor(name)
                if tuple(value.shape) != tuple(getattr(module, attr).shape):
                    raise ValueError(f'Checkpoint shape mismatch: {name}')
                cpu = alias == 'gemma' and key == 'embed_tokens.weight'
                if alias == 'llama70' and key.startswith('layers.'):
                    cpu = int(key.split('.')[1]) >= spec['gpu_layers']
                value = value.to('cpu' if cpu else 'cuda')
                if attr in module._parameters:
                    if value.dtype != torch.bfloat16:
                        raise ValueError(f'Expected unquantized BF16: {name}')
                    setattr(module, attr, torch.nn.Parameter(value, requires_grad=False))
                else:
                    setattr(module, attr, value)
                loaded.add(key)
    if loaded != expected or any(p.is_meta for p in model.parameters()):
        raise ValueError(f'Missing checkpoint weights: {expected-loaded}')
    if alias == 'gemma':
        model.embed_tokens.embed_scale = torch.tensor(model.embed_tokens.scalar_embed_scale, dtype=torch.float32)
        class CPUEmbedding(torch.nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding
            def forward(self, ids):
                return self.embedding(ids.cpu()).to(ids.device)
        model.embed_tokens = CPUEmbedding(model.embed_tokens)
    if alias == 'llama70':
        from accelerate import dispatch_model
        device_map = {'embed_tokens':'cuda:0', 'norm':'cuda:0', 'rotary_emb':'cuda:0'}
        device_map.update({f'layers.{i}': 'cuda:0' if i < spec['gpu_layers'] else 'cpu' for i in range(spec['block_index']+1)})
        model = dispatch_model(model, device_map=device_map, main_device='cuda:0', force_hooks=True)
    if any(b.is_meta for b in model.buffers()):
        raise ValueError('Uninitialized model buffers')
    write(out/'LOADER_VERIFIED.json', dict(revision=spec['revision'], block_index=spec['block_index'], tensors=len(loaded), final_norm_omitted=True))
    return model.eval().requires_grad_(False)


def batches(rows, size, tokens):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['n_tokens'], rows[i]['immutable_id']))
    batch = []
    for i in order:
        if batch and (len(batch) >= size or (len(batch)+1)*rows[i]['n_tokens'] > tokens):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def forward(model, rows, pad):
    import torch
    width = max(r['n_tokens'] for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device='cuda')
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :row['n_tokens']] = torch.tensor(row['token_ids'], device='cuda')
        attention[i, :row['n_tokens']] = 1
    with torch.inference_mode():
        hidden = model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    if not torch.isfinite(hidden).all():
        raise ValueError('Nonfinite backbone activations')
    pooled = torch.stack([hidden[i, r['mask']].float().mean(0) for i, r in enumerate(rows)]).cpu().numpy()
    return pooled, hidden
