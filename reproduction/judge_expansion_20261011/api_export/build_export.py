#!/usr/bin/env python3
"""Lossless, streaming export of frozen messages; no inference or source writes."""
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path('/home/ubuntu/experiments/judge-expansion-20261011')
CANON = ROOT / 'canonical'
EXPORT = ROOT / 'api_export'
SOURCES = {
    'hard': Path('/home/ubuntu/experiments/full-threshold-sweep-20261009'),
    'probability': Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009'),
}
BACKBONES = ['llama', 'qwen', 'gemma', 'llama70', 'qwen122']
CONDITIONS = ['baseline', 'saliency', 'nearest']
DATASETS = {'aegis','beavertails','fava','halueval','high_stakes','ragtruth','toxicchat','toxigen','wildguard','wildjailbreak'}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read(path):
    return json.loads(path.read_text())

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def split_messages(prompt, config, dataset):
    # Only remove known model-specific chat wrappers, not any message text.
    start = '<|im_start|>system\n'
    middle = '<|im_end|>\n<|im_start|>user\n'
    suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    assert prompt.startswith(start) and prompt.endswith(suffix)
    core = prompt[len(start):-len(suffix)]
    system, sep, user = core.partition(middle)
    assert sep
    assert start + system + middle + user + suffix == prompt
    assert ('\n\n' + config['rubrics'][dataset]) in system
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def main():
    CANON.mkdir(parents=True, exist_ok=True)
    EXPORT.mkdir(parents=True, exist_ok=True)
    assert not (CANON / 'PREPARED.json').exists(), 'Refusing to overwrite frozen export'
    assert shutil.disk_usage(ROOT).free > 20 * 1024 ** 3
    sources = {}
    def source(path):
        sources[str(path)] = sha(path)
        return sources[str(path)]
    for root in SOURCES.values():
        for name in ['COMPLETED.json', 'EVALUATION_COMPLETED.json', 'PROTOCOL.json', 'RESULTS.json']:
            source(root / name)
        done = read(root / 'COMPLETED.json')
        assert sha(root / 'RESULTS.json') == done['results_sha256']
    count = 0
    balance = collections.Counter()
    seen_ids = set()
    max_chars = [0, 0]
    ordered_keys = []
    source_groups = []
    csv_path = EXPORT / 'prompts.csv'
    with (CANON / 'cases.jsonl').open('x', encoding='utf-8') as cf, csv_path.open('x', encoding='utf-8', newline='') as pf, (CANON / 'private_mapping.jsonl').open('x', encoding='utf-8') as private:
        writer = csv.DictWriter(pf, fieldnames=['prompt_id','output_type','system_prompt','user_prompt'], lineterminator='\n')
        writer.writeheader()
        for backbone in BACKBONES:
            parent = SOURCES['hard'] / backbone
            selected = read(parent / 'selected_flags.json')
            ids = [x['sample_id'] for x in selected]
            assert len(ids) == len(set(ids)) == 10000
            selected_by_id = {x['sample_id']: x for x in selected}
            assert {x['dataset'] for x in selected} == DATASETS
            assert collections.Counter((x['dataset'],x['split']) for x in selected) == collections.Counter({(ds, split):500 for ds in DATASETS for split in ['val','test']})
            for mode, root in SOURCES.items():
                source(root / backbone / 'selected_flags.json')
                assert read(root / backbone / 'selected_flags.json') == selected
                for condition in CONDITIONS:
                    d = root / backbone / 'judges' / ('qwen_' + condition)
                    p = d / 'prompts.jsonl'
                    cfg = read(d / 'approved_config.json')
                    assert source(p) == cfg['prompt_sha256']
                    source(d / 'approved_config.json')
                    run = read(d / 'run/config.json')
                    done = read(d / 'run/COMPLETED')
                    assert source(d / 'run/config.json') == done['config_sha256']
                    assert source(d / 'run/judgments.jsonl') == done['judgments_sha256']
                    source(d / 'run/COMPLETED')
                    assert run['prompt_sha256'] == sha(p)
                    parent_cfg = read(parent / 'judges' / ('qwen_' + condition) / 'approved_config.json')
                    observed = []
                    with p.open(encoding='utf-8') as source_file:
                        for line_number, line in enumerate(source_file, 1):
                            row = json.loads(line)
                            observed.append(row['sample_id'])
                            s = selected_by_id[row['sample_id']]
                            assert row['dataset'] == s['dataset']
                            messages = split_messages(row['prompt'], parent_cfg, row['dataset'])
                            # Full probability-prompt content was checksummed by source construction.
                            if mode == 'probability':
                                assert digest(row['prompt']) == row['instruction_sha256']
                            message_json = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',',':'))
                            key = [backbone, row['dataset'], row['sample_id'], s['split'], condition, mode, digest(message_json)]
                            pid = 'p_' + digest(json.dumps(key, ensure_ascii=False, separators=(',',':')))
                            assert pid not in seen_ids
                            seen_ids.add(pid)
                            canonical = dict(prompt_id=pid, id=pid, backbone=backbone, dataset=row['dataset'], sample_id=row['sample_id'], split=s['split'], condition=condition, mode=mode, messages=messages, messages_sha256=digest(message_json), source_prompt_sha256=digest(row['prompt']), source_relative_path=str(p.relative_to(root)), source_root=str(root), source_line=line_number, source_instruction_sha256=row['instruction_sha256'])
                            cf.write(json.dumps(canonical, ensure_ascii=False, separators=(',',':')) + '\n')
                            writer.writerow(dict(prompt_id=pid, output_type=mode, system_prompt=messages[0]['content'], user_prompt=messages[1]['content']))
                            private.write(json.dumps(dict(prompt_id=pid,backbone=backbone,condition=condition,mode=mode,**s), ensure_ascii=False, separators=(',',':')) + '\n')
                            for i in (0,1): max_chars[i] = max(max_chars[i],len(messages[i]['content']))
                            count += 1
                            balance[(backbone,row['dataset'],s['split'],condition,mode)] += 1
                            ordered_keys.append(pid)
                    assert observed == ids
                    source_groups.append(dict(backbone=backbone,mode=mode,condition=condition,n=len(observed),prompt_sha256=sha(p)))
                    print(json.dumps(source_groups[-1]), flush=True)
    assert count == 300000 and len(balance) == 600 and set(balance.values()) == {500}
    # CSV is a transport format, not a spreadsheet: preserve long text and all Unicode verbatim.
    csv.field_size_limit(100000000)
    with csv_path.open(newline='', encoding='utf-8') as pf, (CANON/'cases.jsonl').open(encoding='utf-8') as cf:
        reader = csv.DictReader(pf)
        assert reader.fieldnames == ['prompt_id','output_type','system_prompt','user_prompt']
        verified = 0
        for row, line in zip(reader, cf, strict=True):
            c = json.loads(line)
            assert row['prompt_id'] == c['prompt_id']
            assert row['output_type'] == c['mode']
            assert row['system_prompt'] == c['messages'][0]['content']
            assert row['user_prompt'] == c['messages'][1]['content']
            verified += 1
        assert verified == 300000
    manifest = dict(total=count,cases_sha256=sha(CANON/'cases.jsonl'),prompts_csv_sha256=sha(csv_path),private_mapping_sha256=sha(CANON/'private_mapping.jsonl'),ordered_ids_sha256=digest('\n'.join(ordered_keys)+'\n'),sources=sources,source_groups=source_groups,groups=len(balance),per_group=500,max_message_characters={'system':max_chars[0],'user':max_chars[1]},checks={'unique_prompt_ids':True,'same_selected_cases_and_order':True,'source_prompt_files_match_approved_and_run_configs':True,'completed_judgments_checksums_verified':True,'chat_wrapper_removal_reversible':True,'csv_roundtrip_exact_300000_rows':True,'target_labels_and_strata_omitted_from_export':True},created_unix=time.time())
    dump(CANON/'PREPARED.json',manifest)
    dump(EXPORT/'export_manifest.json',{k:manifest[k] for k in ['total','prompts_csv_sha256','ordered_ids_sha256','groups','per_group','max_message_characters','checks']})
    print(json.dumps({'completed':True,'total':count,'cases_sha256':manifest['cases_sha256'],'csv_bytes':csv_path.stat().st_size}),flush=True)

if __name__ == '__main__':
    main()
