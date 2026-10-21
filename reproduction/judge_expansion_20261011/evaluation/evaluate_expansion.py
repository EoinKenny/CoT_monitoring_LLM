#!/usr/bin/env python3
"""Route the frozen historical estimators to the seven-judge completed grid.

No model calls, training, sampling, or new inferential policy is performed.
Unmodified historical source snapshots are checked before use. Only paths,
judge lists, and expected completion counts are substituted at runtime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

HERE = Path(__file__).resolve().parent
BBS = ['llama', 'qwen', 'gemma', 'llama70', 'qwen122']
JS = ['qwen', 'gemma', 'olmo', 'glm', 'qwen122', 'llama', 'mistral']
CS = ['baseline', 'saliency', 'nearest']
OLD_HARD = '/home/ubuntu/experiments/full-threshold-sweep-20261009'
OLD_PROB = '/home/ubuntu/experiments/confidence-threshold-sweep-20261009'
OLD_ANALYSIS = '/home/ubuntu/experiments/probability-comprehensive-analysis-20261011'
PAIRS = len(BBS) * len(JS) * 10
PER_MODE = len(BBS) * len(JS) * len(CS) * 10000


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def source(name):
    path = HERE / 'upstream' / name
    assert sha(path) == read(HERE / 'UPSTREAM_SHA256.json')[name], path
    return path.read_text()


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def load_stage(stage, hard_root, prob_root, output):
    """Numerical functions are used byte-for-byte from frozen snapshots."""
    namespace = {'__name__': 'frozen_expansion_stage', '__file__': str(HERE / 'upstream' / stage)}
    text = source(stage)
    if stage == 'probability_evaluate.py':
        common = source('probability_common.py')
        common = common.replace(OLD_PROB, str(prob_root)).replace(OLD_HARD, str(hard_root))
        common = once(common, "JS=['qwen','gemma']", f'JS={JS!r}')
        exec(compile(common, str(HERE / 'upstream/probability_common.py'), 'exec'), namespace)
        text = once(text, 'from common import *', '# common namespace injected from pinned snapshot')
        text = text.replace('total=100', f'total={PAIRS}')
        text = once(text, 'len(allresults)==100', f'len(allresults)=={PAIRS}')
        text = once(text, 'pairs=100', f'pairs={PAIRS}')
        text = once(text, 'judgments=300000', f'judgments={PER_MODE}')
    elif stage == 'hard_evaluate.py':
        text = text.replace(OLD_HARD, str(hard_root))
        text = once(text, "for judge in ['qwen','gemma']:", f'for judge in {JS!r}:')
    elif stage == 'analyze.py':
        text = text.replace(OLD_HARD, str(hard_root)).replace(OLD_PROB, str(prob_root)).replace(OLD_ANALYSIS, str(output))
        text = once(text, "JS=['qwen','gemma']", f'JS={JS!r}')
        text = once(text, 'n==300000', f'n=={PER_MODE}')
        text = text.replace('total=100', f'total={PAIRS}')
    else:
        raise ValueError(stage)
    namespace['__adapted_source__'] = text
    exec(compile(text, str(HERE / 'upstream' / stage), 'exec'), namespace)
    return namespace


def verify_inputs(hard_root, prob_root, output):
    """Fail closed before estimation on exact case, prompt, raw, and commit coverage."""
    audited = []
    sources = {}
    for b in BBS:
        hard = hard_root / b
        prob = prob_root / b
        for name in ['selected_flags.json', 'THRESHOLDS.json']:
            assert sha(hard / name) == sha(prob / name), (b, name, 'paired metadata differs')
            sources[str(hard / name)] = sha(hard / name)
            sources[str(prob / name)] = sha(prob / name)
        rows = read(hard / 'selected_flags.json')
        ids = [r['sample_id'] for r in rows]
        assert len(ids) == len(set(ids)) == 10000, b
        assert sha(hard / 'selected_flags.json') == read(hard / 'SAMPLE_FROZEN.json')['selection_sha256']
        vgroups = {(r['dataset'], r['group']) for r in rows if r['split'] == 'val'}
        tgroups = {(r['dataset'], r['group']) for r in rows if r['split'] == 'test'}
        assert not vgroups & tgroups, (b, 'sample group leakage')
        specs = read(hard / 'THRESHOLDS.json')
        assert len(specs) == 10
        for ds in specs:
            for split in ['val', 'test']:
                selected = [r for r in rows if r['dataset'] == ds and r['split'] == split]
                assert len(selected) == 500, (b, ds, split)
                assert all(0 < r['inclusion_probability'] <= 1 for r in selected)
            name = f'{ds}_predictions.npz'
            assert sha(hard / name) == sha(prob / name), (b, name)
            sources[str(prob / name)] = sha(prob / name)
        for root, mode in [(hard_root, 'hard'), (prob_root, 'probability')]:
            for judge in JS:
                for condition in CS:
                    cell = root / b / 'judges' / f'{judge}_{condition}'
                    run = cell / 'run'
                    done = read(run / 'COMPLETED')
                    config = read(run / 'config.json')
                    assert sha(run / 'judgments.jsonl') == done['judgments_sha256'], cell
                    assert sha(run / 'config.json') == done['config_sha256'], cell
                    assert sha(cell / 'prompts.jsonl') == config['prompt_sha256'], cell
                    results = jl(run / 'judgments.jsonl')
                    assert [r['sample_id'] for r in results] == ids, cell
                    field = 'prediction' if mode == 'hard' else 'positive_probability'
                    max_value = 1 if mode == 'hard' else 100
                    assert all(type(r[field]) is int and 0 <= r[field] <= max_value for r in results), cell
                    assert all(r['input_tokens'] >= 0 and r['batch_seconds'] >= 0 for r in results), cell
                    committed = []
                    for batch in sorted((run / 'batches').glob('*.json')):
                        if batch.name.endswith(('.raw.json', '.attempt.json')):
                            continue
                        document = read(batch)
                        assert sha(batch.with_suffix('.raw.json')) == document['raw_sha256'], batch
                        assert document['config_sha256'] == done['config_sha256'], batch
                        digest = hashlib.sha256(json.dumps(document['records'], sort_keys=True).encode()).hexdigest()
                        assert digest == document['records_sha256'], batch
                        committed.extend(document['records'])
                    assert committed == results, (cell, 'commit contents or order differs')
                    sources[str(run / 'judgments.jsonl')] = sha(run / 'judgments.jsonl')
                    sources[str(run / 'config.json')] = sha(run / 'config.json')
                    audited.append(dict(backbone=b, judge=judge, condition=condition, mode=mode, judgments=len(results)))
    assert sum(x['judgments'] for x in audited) == 2 * PER_MODE
    write(output / 'INPUT_VERIFIED.json', dict(time=time.time(), cells=audited, sources=sources))
    write(prob_root / 'PARENT_VERIFIED.json', dict(time=time.time(), sources=sources))
    return audited


def token_costs(hard_root, prob_root, output):
    """Input tokens spent on routed cases, using the existing test design weights.

    These are token counts, not latency/cost estimates or new selection criteria.
    Cross-model tokenizers differ. Cached-token discounts are not inferred.
    """
    pairs = read(output / 'pairs.json')
    result = []
    for b in BBS:
        rows = read(prob_root / b / 'selected_flags.json')
        for judge in JS:
            outputs = {}
            for mode, root in [('hard', hard_root), ('probability', prob_root)]:
                for condition in CS:
                    p = root / b / 'judges' / f'{judge}_{condition}' / 'run/judgments.jsonl'
                    outputs[condition, mode] = {r['sample_id']: r for r in jl(p)}
            for pair in [p for p in pairs if p['backbone'] == b and p['judge'] == judge]:
                ds = pair['dataset']
                tr = [r for r in rows if r['dataset'] == ds and r['split'] == 'test']
                n = pair['populations']['test']['n']
                for variant, policy in pair['variants'].items():
                    if variant == 'probe':
                        tokens = routed = 0.
                    else:
                        condition, suffix = variant.split('_', 1)
                        mode = 'hard' if suffix == 'parent_hard' else 'probability'
                        rmap = outputs[condition, mode]
                        routed_rows = [r for r in tr if r['probe_probability'] >= policy['probe_threshold']]
                        tokens = sum(rmap[r['sample_id']]['input_tokens'] / r['inclusion_probability'] for r in routed_rows)
                        routed = sum(1 / r['inclusion_probability'] for r in routed_rows)
                    result.append(dict(backbone=b, judge=judge, dataset=ds, variant=variant,
                        population_test_n=n, test_sample_n=len(tr),
                        estimated_population_input_tokens=tokens,
                        estimated_input_tokens_per_population_example=tokens / n,
                        estimated_input_tokens_per_1000_population_examples=1000 * tokens / n,
                        weighted_mean_input_tokens_per_referred_case=tokens / routed if routed else 0.,
                        exact_referral_fraction=policy['routing_rate'],
                        f1=policy['f1'], precision=policy['precision'], recall=policy['recall']))
    write(output / 'input_token_costs.json', dict(rows=result,
        note='Design-weighted test estimates at validation-selected probe gates; raw un-discounted model-tokenizer input counts. No assumption that equal token counts imply equal latency, compute or price. Judge cutoff affects decisions but not whether a referred prompt is processed.'))
    with (output / 'input_token_costs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


def summarize(namespace, output):
    # Reuse the exact historical macro, heterogeneity, and paired dataset-block
    # bootstrap summary code, stopping before its two-judge plotting and prose.
    text = source('summarize.py')
    text = text[text.index('from collections import Counter'):text.index('# Precision-recall plot')]
    exec(compile(text, str(HERE / 'upstream/summarize.py'), 'exec'), namespace)
    summary = read(output / 'summary.json')
    h = summary['heterogeneity']['judge']
    variants = ['probe', 'baseline_parent_hard', 'saliency_parent_hard', 'nearest_parent_hard',
                'baseline_fixed50', 'saliency_fixed50', 'nearest_fixed50',
                'baseline_joint', 'saliency_joint', 'nearest_joint']
    records = []
    for judge, values in h.items():
        for variant in variants:
            records.append(dict(judge=judge, variant=variant, **values[variant]))
    with (output / 'judge_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table = '<table><tr><th>Judge</th><th>Policy</th><th>F1 %</th><th>Precision %</th><th>Recall %</th><th>Referral %</th></tr>'
    for r in records:
        table += '<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in [r['judge'], r['variant']] + [f'{100*r[k]:.2f}' for k in ['f1', 'precision', 'recall', 'routing_rate']]) + '</tr>'
    table += '</table>'
    document = ('<!doctype html><meta charset="utf-8"><title>Seven-judge probe study</title>'
        '<style>body{font:16px system-ui;margin:40px;max-width:1200px}table{border-collapse:collapse}th,td{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}</style>'
        '<h1>Seven-judge probe study</h1>'
        '<p>Five backbones, ten datasets, seven local judges. Hard labels and probability-only outputs under baseline, saliency and nearest-neighbour conditions. '
        'All 2,100,000 judgments and paired case memberships verified. Each judge sees the same 50,000 backbone–dataset cases: 25,000 validation and 25,000 test.</p>'
        '<p>Thresholds and configurations are selected using validation only. LLM pipeline metrics are design-weighted held-out test estimates; probe-only metrics use complete test populations. '
        'The study has prior test inspection and is exploratory. Macro averages weight backbone–dataset–judge cells equally, not individual examples. '
        'The same 500 validation and 500 test cases per backbone–dataset pair are reused across conditions and judges; judgments are not independent examples.</p>'
        '<p>parent_hard tunes only the probe gate; fixed50 tunes the gate with the judge cutoff fixed to 50; joint tunes both gate and probability cutoff. '
        'All ties use the original frozen rules. Summary intervals use the unchanged paired dataset-block bootstrap; per-cell and threshold-selection sensitivity analyses retain original strata and weights.</p>'
        + table +
        '<p>See summary.json for overall macro means, heterogeneity, paired comparisons and intervals; input_token_costs.csv for weighted input-token estimates at the validation-selected gates; '
        'runtime.json for actual collection input tokens and measured batch times. Token counts do not establish equal latency or cost across model tokenizers, hardware, quantization, or cache configurations.</p>'
        '<p>Comprehensive outputs include pairs.json, budgets.json, endpoints.json, effects.json, calibration.json, stability.json, support.json, cases.json and neighbour_groups.json. '
        'Nearest-neighbour examples supply labelled information: these comparisons do not isolate interpretability from in-context learning. No claim of universal NN advantage is assumed.</p>')
    (output / 'report.html').write_text(document)
    write(output / 'REPORT_COMPLETED.json', dict(time=time.time(), pairs=PAIRS,
        files={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name != 'REPORT_COMPLETED.json'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hard-root', type=Path, required=True)
    parser.add_argument('--probability-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for path in [args.hard_root, args.probability_root, args.output]:
        path.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(path).free >= 20 * 2**30, 'Below 20 GiB reserve'
    lock_path = args.output / 'EVALUATION.lock'
    import fcntl
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'REPORT_COMPLETED.json').exists():
            completed = read(args.output / 'REPORT_COMPLETED.json')
            assert all(sha(args.output / p) == h for p,h in completed['files'].items())
            print('Completed expansion report already exists and verifies; no recomputation.')
            return
        verify_inputs(args.hard_root, args.probability_root, args.output)
        if args.verify_only:
            print('Verified complete 2,100,000-judgment paired grid.')
            return
        write(args.output / 'ADAPTER_PROVENANCE.json', dict(time=time.time(),
            upstream_sha256=read(HERE / 'UPSTREAM_SHA256.json'), adapter_sha256=sha(__file__),
            judges=JS, backbones=BBS, conditions=CS, hard_root=str(args.hard_root),
            probability_root=str(args.probability_root), mutations='Paths, judge list, and completion counts only; original numerical functions unchanged. Additional token-cost arithmetic uses the same routed test masks and inverse-inclusion weights.'))
        hard = load_stage('hard_evaluate.py', args.hard_root, args.probability_root, args.output)
        hard['run']()
        write(args.hard_root / 'COMPLETED.json', dict(time=time.time(), judgments=PER_MODE,
            results_sha256=sha(args.hard_root / 'RESULTS.json'), scope='evaluation verified all completed per-cell collection artifacts'))
        prob = load_stage('probability_evaluate.py', args.hard_root, args.probability_root, args.output)
        prob['main']()
        write(args.probability_root / 'COMPLETED.json', dict(time=time.time(), judgments=PER_MODE,
            results_sha256=sha(args.probability_root / 'RESULTS.json'), scope='evaluation verified all completed per-cell collection artifacts'))
        analysis = load_stage('analyze.py', args.hard_root, args.probability_root, args.output)
        analysis['run']()
        token_costs(args.hard_root, args.probability_root, args.output)
        summarize(analysis, args.output)
        print(json.dumps(dict(state='completed', pairs=PAIRS, judgments=2 * PER_MODE, report=str(args.output / 'report.html'))))


if __name__ == '__main__':
    main()
