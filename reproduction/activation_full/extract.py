"""Resumable full extraction with verified, atomic shards and pinned sources."""
import os
import socket

# This host's IPv6 connections stall; scope the workaround to this process.
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _getaddrinfo(
    host, port, socket.AF_INET, type, proto, flags
)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'

import json
import fcntl
import hashlib
import shutil
import traceback
import sys
from datetime import datetime, timezone
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import save_file, load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path('/home/ubuntu/experiments/activation-full-20261001')
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'activations').mkdir(exist_ok=True)
LOCK = (ROOT / 'run.lock').open('a')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (ROOT / 'COMPLETED').exists():
    raise SystemExit('Already completed; refusing to extract again.')
(ROOT / 'pid.txt').write_text(str(os.getpid()) + '\n')

def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def failure(exc_type, exc, tb):
    atomic_json(ROOT / 'FAILED.json', dict(time=datetime.now(timezone.utc).isoformat(),
        error=str(exc), traceback=''.join(traceback.format_exception(exc_type, exc, tb))))
    atomic_json(ROOT / 'status.json', dict(state='failed', error=str(exc)))
    sys.__excepthook__(exc_type, exc, tb)

sys.excepthook = failure
atomic_json(ROOT / 'status.json', dict(state='preparing', pid=os.getpid(),
    updated_at=datetime.now(timezone.utc).isoformat()))
MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
REVS = {
    MODEL: '9213176726f574b556790deb65791e0c5aa438b6',
    'allenai/wildguardmix': 'd29c47f41c8b51348b5c8e8c81c039b3132b66d1',
    'Arrrlex/models-under-pressure': '940d830f639617752f54fdf2bca0da7e0d856a9e',
    'lmsys/toxic-chat': '29df8e4dba60e1f4af4b4075c0705c5b313548a8',
}


def log(event, **kwargs):
    record = dict(event=event, **kwargs)
    print(json.dumps(record), flush=True)
    with (ROOT / 'journal.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')


def download(repo, filename):
    return hf_hub_download(repo, filename, repo_type='dataset', revision=REVS[repo])


start = time.perf_counter()
log('start', revisions=REVS, torch=torch.__version__)
torch.set_num_threads(8)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVS[MODEL])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
datasets = {'wildguard': [], 'high_stakes': [], 'toxicchat': []}
source_counts = {}


def add(dataset, source, ident, messages, label):
    # Only model-visible content: never include label explanations or target labels.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    datasets[dataset].append(dict(id=str(ident), source=source, text=text, label=str(label), messages=messages, dataset_index=len(datasets[dataset])))
    source_counts[source] = source_counts.get(source, 0) + 1


for filename in ['train/wildguard_train.parquet', 'test/wildguard_test.parquet']:
    df = pd.read_parquet(download('allenai/wildguardmix', filename))
    for idx, r in df.iterrows():
        if pd.isna(r['response']) or r['response_harm_label'] not in ('harmful', 'unharmful'):
            continue
        add('wildguard', 'wildguard/' + filename, idx,
            [{'role': 'user', 'content': r['prompt']}, {'role': 'assistant', 'content': r['response']}],
            r['response_harm_label'])
log('dataset_loaded', dataset='wildguard', examples=len(datasets['wildguard']))

files = HfApi().list_repo_files('Arrrlex/models-under-pressure', repo_type='dataset',
                               revision=REVS['Arrrlex/models-under-pressure'])
for filename in files:
    if not filename.endswith('.jsonl') or not (filename.startswith('training/') or '_balanced/' in filename):
        continue
    with open(download('Arrrlex/models-under-pressure', filename)) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            if r['labels'] not in ('high-stakes', 'low-stakes'):
                continue
            raw = r['inputs']
            try:
                messages = json.loads(raw)
            except (ValueError, TypeError):
                messages = None
            if not isinstance(messages, list):
                messages = [{'role': 'user', 'content': raw}]
            add('high_stakes', 'high_stakes/' + filename, r.get('ids', idx), messages, r['labels'])
log('dataset_loaded', dataset='high_stakes', examples=len(datasets['high_stakes']))

for split in ['train', 'test']:
    filename = f'data/0124/toxic-chat_annotation_{split}.csv'
    df = pd.read_csv(download('lmsys/toxic-chat', filename))
    for idx, r in df.iterrows():
        if r['toxicity'] not in (0, 1) or pd.isna(r['user_input']):
            continue
        add('toxicchat', 'toxicchat/' + split, r.get('conv_id', idx),
            [{'role': 'user', 'content': r['user_input']}], r['toxicity'])
log('dataset_loaded', dataset='toxicchat', examples=len(datasets['toxicchat']))

# Preserve the pilot input formatting and verify the complete token inventory.
inventory = {}
for name, rows in datasets.items():
    t = time.perf_counter()
    for offset in range(0, len(rows), 256):
        part = rows[offset:offset + 256]
        ids = tokenizer([r['text'] for r in part], add_special_tokens=False)['input_ids']
        for row, tokens in zip(part, ids):
            row['tokens'] = tokens
            del row['text']
    lengths = np.array([len(r['tokens']) for r in rows])
    inventory[name] = dict(examples=len(rows), tokens=int(lengths.sum()),
                           mean_tokens=float(lengths.mean()), max_tokens=int(lengths.max()),
                           length_percentiles=np.percentile(lengths, [25, 50, 75, 95, 99]).tolist(),
                           activation_gib=float(lengths.sum() * 2048 * 2 / 2**30),
                           tokenize_seconds=time.perf_counter() - t)
    log('inventory', dataset=name, **inventory[name])
(ROOT / 'inventory.json').write_text(json.dumps(dict(datasets=inventory, sources=source_counts,
                                                    revisions=REVS), indent=2))
log('preparation_done', seconds=time.perf_counter() - start)
expected = {'wildguard': (39643, 18310532), 'high_stakes': (18098, 4025903), 'toxicchat': (10165, 707627)}
for name, (count, tokens) in expected.items():
    assert (inventory[name]['examples'], inventory[name]['tokens']) == (count, tokens), name
configuration = dict(revisions=REVS, block_index=10, representation='post-block residual',
    dtype='bfloat16', batch_size=8, examples_per_shard=64, truncation=False,
    input_format='chat template; no added system instruction; no generation prompt',
    forward='all 16 backbone blocks; no LM head or KV cache',
    splits='same as pilot; no deduplication; retain source and original IDs')
fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
if (ROOT / 'config.json').exists():
    assert json.loads((ROOT / 'config.json').read_text()) == configuration, 'Configuration changed'
else:
    atomic_json(ROOT / 'config.json', configuration)
completed_bytes = sum(p.stat().st_size for p in (ROOT / 'activations').glob('*.safetensors'))
needed = sum(x['tokens'] for x in inventory.values()) * 4100 - completed_bytes + 10 * 2**30
assert shutil.disk_usage(ROOT).free > needed, 'Insufficient free disk space'


t = time.perf_counter()
model_path = snapshot_download(MODEL, revision=REVS[MODEL],
                              allow_patterns=['*.json', '*.safetensors'], max_workers=2)
log('model_download_done', seconds=time.perf_counter() - t)
t = time.perf_counter()
model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16,
                                 attn_implementation='sdpa').to('cuda').eval()
assert model.config.hidden_size == 2048
assert model.config.num_hidden_layers == 16
log('model_loaded', seconds=time.perf_counter() - t, gpu=torch.cuda.get_device_name(),
    hidden_size=model.config.hidden_size, layers=model.config.num_hidden_layers)
captured = {}


def hook(module, inputs, output):
    captured['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()


# Residual output of the 11th transformer block (zero-based index 10).
handle = model.layers[10].register_forward_hook(hook)
with torch.inference_mode():
    for _ in range(2):
        model(input_ids=torch.full((8, 256), tokenizer.eos_token_id, device='cuda'),
              attention_mask=torch.ones((8, 256), device='cuda', dtype=torch.long), use_cache=False)
torch.cuda.synchronize()
captured.clear()
torch.cuda.reset_peak_memory_stats()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


total_examples = sum(len(rows) for rows in datasets.values())
total_tokens = sum(x['tokens'] for x in inventory.values())
completed_examples = completed_tokens = completed_shards = 0
manifest = []
extraction_start = time.perf_counter()


def status(state='running', **extra):
    value = dict(state=state, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(),
        completed_examples=completed_examples, total_examples=total_examples,
        completed_tokens=completed_tokens, total_tokens=total_tokens,
        completed_shards=completed_shards, elapsed_seconds=time.perf_counter()-start, **extra)
    atomic_json(ROOT / 'status.json', value)
    return value


status()
for name, rows in datasets.items():
    ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]['tokens']))
    for shard_no, offset in enumerate(range(0, len(ordered), 64)):
        indices = ordered[offset:offset+64]
        lengths = [len(rows[i]['tokens']) for i in indices]
        path = ROOT / 'activations' / f'{name}-{shard_no:05d}.safetensors'
        meta_path = path.with_suffix('.json')
        partial = path.with_suffix('.partial')
        sample_meta = [{k: rows[i][k] for k in ['id','source','label','dataset_index','messages']} for i in indices]
        expected_tokens = sum(lengths)
        if partial.exists():
            raise RuntimeError(f'Unresolved partial shard requires inspection: {partial}')
        if path.exists() or meta_path.exists():
            assert path.exists() and meta_path.exists(), f'Unpaired shard files: {path}'
            info = json.loads(meta_path.read_text())
            assert info['fingerprint'] == fingerprint and info['samples'] == sample_meta
            assert info['tokens'] == expected_tokens and info['sha256'] == digest(path)
            log('shard_resume_verified', dataset=name, shard=shard_no)
        else:
            t = time.perf_counter()
            pieces = []
            for batch_offset in range(0, len(indices), 8):
                batch = indices[batch_offset:batch_offset+8]
                batch_lengths = [len(rows[i]['tokens']) for i in batch]
                assert len(batch) * max(batch_lengths) <= 32768
                encoded = tokenizer.pad({'input_ids':[rows[i]['tokens'] for i in batch]},
                                        padding=True, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    model(**encoded, use_cache=False)
                hidden = captured.pop('hidden')
                pieces.append(torch.cat([hidden[j,:n] for j,n in enumerate(batch_lengths)], dim=0).cpu())
                del encoded, hidden
            activations = torch.cat(pieces, dim=0).contiguous()
            del pieces
            token_ids = torch.tensor([v for i in indices for v in rows[i]['tokens']], dtype=torch.int32)
            offsets = torch.tensor([0] + np.cumsum(lengths).tolist(), dtype=torch.int64)
            assert activations.shape == (expected_tokens, 2048)
            assert activations.dtype == torch.bfloat16 and torch.isfinite(activations).all()
            tensors = dict(activations=activations, token_ids=token_ids, offsets=offsets)
            save_file(tensors, str(partial), metadata=dict(fingerprint=fingerprint,
                model_revision=REVS[MODEL], block_index='10', dataset=name))
            with partial.open('rb') as f:
                os.fsync(f.fileno())
            restored = load_file(str(partial))
            assert set(restored) == set(tensors)
            for key in tensors:
                assert torch.equal(restored[key], tensors[key]), f'Failed readback: {key}'
            info = dict(dataset=name, shard=shard_no, file=path.name, examples=len(indices),
                tokens=expected_tokens, bytes=partial.stat().st_size, sha256=digest(partial),
                fingerprint=fingerprint, verified=True, samples=sample_meta,
                elapsed_seconds=time.perf_counter()-t)
            os.replace(partial, path)
            atomic_json(meta_path, info)
            sync_dir(path.parent)
            del activations, restored, tensors, token_ids, offsets
            log('shard_committed', dataset=name, shard=shard_no, examples=len(indices),
                tokens=expected_tokens, seconds=info['elapsed_seconds'])
        manifest.append({k: info[k] for k in ['dataset','shard','file','examples','tokens','bytes','sha256','fingerprint']})
        completed_examples += len(indices)
        completed_tokens += expected_tokens
        completed_shards += 1
        status(dataset=name, last_shard=path.name)
    log('dataset_complete', dataset=name, examples=len(rows), tokens=inventory[name]['tokens'])

assert completed_examples == total_examples and completed_tokens == total_tokens
assert len({m['file'] for m in manifest}) == completed_shards
assert len(list((ROOT / 'activations').glob('*.safetensors'))) == completed_shards
assert not list((ROOT / 'activations').glob('*.partial'))
atomic_json(ROOT / 'manifest.json', dict(configuration=configuration, shards=manifest,
    total_examples=total_examples, total_tokens=total_tokens,
    total_file_bytes=sum(m['bytes'] for m in manifest)))
handle.remove()
finished = status('completed', peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  extraction_seconds=time.perf_counter()-extraction_start)
atomic_json(ROOT / 'COMPLETED', finished)
sync_dir(ROOT)
log('completed', **finished)
