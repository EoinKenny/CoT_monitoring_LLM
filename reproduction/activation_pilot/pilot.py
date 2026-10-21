"""Bounded, one-layer activation pilot; never runs full-dataset extraction."""
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
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import save_file, load_file
from transformers import AutoModel, AutoTokenizer

ROOT = Path('/home/ubuntu/experiments/activation-pilot-20261001')
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / 'activations').mkdir(exist_ok=True)
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
    datasets[dataset].append(dict(id=str(ident), source=source, text=text, label=str(label)))
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

# Tokenize the full text inventory, but run the GPU on only 64 examples/dataset.
# This makes disk-size estimates exact for the specified formatting and filters.
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
rng = random.Random(20261001)
results = {}

for name, rows in datasets.items():
    ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]['tokens']))
    bins = np.array_split(np.array(ordered), 8)
    measures = []
    for bin_idx, indexes in enumerate(bins):
        chosen = sorted(rng.sample(indexes.tolist(), 8), key=lambda i: len(rows[i]['tokens']))
        # Split only unusually long batches to limit padding memory; never truncate.
        batches = []
        current = []
        for idx in chosen:
            if current and (len(current) + 1) * len(rows[idx]['tokens']) > 32768:
                batches.append(current)
                current = []
            current.append(idx)
        if current:
            batches.append(current)
        bin_measures = []
        for sub_idx, batch in enumerate(batches):
            lengths = [len(rows[i]['tokens']) for i in batch]
            t = time.perf_counter()
            encoded = tokenizer.pad({'input_ids': [rows[i]['tokens'] for i in batch]},
                                    padding=True, return_tensors='pt').to('cuda')
            torch.cuda.synchronize()
            prepared = time.perf_counter()
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            torch.cuda.synchronize()
            forward = time.perf_counter()
            hidden = captured.pop('hidden')
            activations = torch.cat([hidden[j, :n] for j, n in enumerate(lengths)], dim=0).cpu().contiguous()
            torch.cuda.synchronize()
            transferred = time.perf_counter()
            path = ROOT / 'activations' / f'{name}-{bin_idx}-{sub_idx}.safetensors'
            offsets = torch.tensor([0] + np.cumsum(lengths).tolist(), dtype=torch.int64)
            token_ids = torch.tensor([v for i in batch for v in rows[i]['tokens']], dtype=torch.int32)
            save_file({'activations': activations, 'offsets': offsets, 'token_ids': token_ids}, str(path),
                      metadata={'model_revision': REVS[MODEL], 'block_index': '10',
                                'samples': json.dumps([{k: rows[i][k] for k in ['id', 'source', 'label']} for i in batch])})
            with path.open('rb') as f:
                os.fsync(f.fileno())
            written = time.perf_counter()
            # Verify a complete round-trip; verification time is excluded from throughput.
            restored = load_file(str(path))
            assert torch.equal(restored['activations'], activations)
            assert restored['activations'].shape == (sum(lengths), 2048)
            assert restored['activations'].dtype == torch.bfloat16
            m = dict(dataset=name, bin=bin_idx, sub_batch=sub_idx, examples=len(batch),
                     tokens=sum(lengths), padded_tokens=len(batch) * max(lengths),
                     prepare_seconds=prepared-t, forward_seconds=forward-prepared,
                     transfer_seconds=transferred-forward, write_fsync_seconds=written-transferred,
                     total_seconds=written-t, file_bytes=path.stat().st_size, verified=True)
            measures.append(m)
            bin_measures.append(m)
            log('batch', **m)
            del hidden, activations, restored, encoded
        # Extrapolate within length strata; each bin contains ~1/8 of all examples.
        seconds_per_example = sum(m['total_seconds'] for m in bin_measures) / len(chosen)
        for m in bin_measures:
            m['bin_population'] = len(indexes)
        log('bin_projection', dataset=name, bin=bin_idx,
            projected_seconds=seconds_per_example * len(indexes))
    projected = sum(sum(m['total_seconds'] for m in measures if m['bin'] == b) / 8 * len(bins[b]) for b in range(8))
    totals = {k: sum(m[k] for m in measures) for k in
              ['examples', 'tokens', 'prepare_seconds', 'forward_seconds', 'transfer_seconds',
               'write_fsync_seconds', 'total_seconds', 'file_bytes']}
    results[name] = dict(inventory=inventory[name], measured=totals, batches=measures,
                         projected_seconds=projected,
                         token_rate_projection_seconds=inventory[name]['tokens'] / totals['tokens'] * totals['total_seconds'])
    log('dataset_complete', dataset=name, projected_seconds=projected, **totals)

handle.remove()
report = dict(results=results, revisions=REVS, source_counts=source_counts,
              peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
              wall_seconds=time.perf_counter()-start,
              configuration=dict(seed=20261001, samples_per_dataset=64, batch_size=8,
                  block_index=10, storage_dtype='bfloat16', truncation=False,
                  input_format='chat template; no added system instruction; no generation prompt',
                  forward='all 16 backbone blocks; no LM head or KV cache',
                  sampling='8 random examples from each of 8 length quantiles',
                  duplicates='source splits preserved; no cross-split deduplication in this timing inventory'))
(ROOT / 'results.json').write_text(json.dumps(report, indent=2))
(ROOT / 'COMPLETED').write_text('192 examples extracted; saved tensors verified\n')
log('completed', wall_seconds=report['wall_seconds'], peak_gpu_allocated_gib=report['peak_gpu_allocated_gib'])
