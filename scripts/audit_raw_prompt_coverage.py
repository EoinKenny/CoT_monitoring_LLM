#!/usr/bin/env python3
"""Check raw response identifiers against saved prompts, without inference."""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import lzma
import sys
import zipfile
from pathlib import Path


def catalogue_ids(path):
    """Expand compact identifiers using the same rules as data/prompts/restore.py."""
    strings, identities, previous = {}, [], []
    def encoded(value):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with lzma.open(path, 'rt') as f:
        for line in f:
            record = json.loads(line)
            if not isinstance(record, list):
                continue
            tag, row = record
            if tag == 's':
                strings[row[0]] = row[1]
            elif tag == 'c':
                identities = row
            elif tag == 'p':
                pid, study, recipe, ci, split, condition, mask = row
                if pid is None or isinstance(pid, int):
                    messages = [{'role': role, 'content': ''.join(strings[i] for i in ids)}
                                for role, ids in json.loads(recipe)]
                    digest = hashlib.sha256(encoded(messages)).hexdigest()
                    if pid is None:
                        assert study == 'ablation'
                        backbone, sample_id = identities[ci]
                        old = 'nnabl_' + hashlib.sha256(encoded([sample_id, condition, digest])).hexdigest()
                        pid = 'apiabl_' + hashlib.sha256((backbone + ':' + old).encode()).hexdigest()
                    else:
                        assert study in ('reason_after', 'reason_first')
                        pid = ('apir_' if study == 'reason_after' else 'apif_') + hashlib.sha256(
                            (previous[pid] + ':' + digest).encode()).hexdigest()
                previous.append(pid)
                yield pid


def audit(folder, catalogue):
    csv.field_size_limit(sys.maxsize)
    ids = set(catalogue_ids(catalogue))
    prompt_sources = []
    catalogue_count = len(ids)
    studies = set()
    for asset in sorted(folder.glob('prompt-contexts-*.zip')):
        with zipfile.ZipFile(asset) as z:
            for name in z.namelist():
                if not name.endswith('.jsonl'):
                    continue
                # Canonical messages suffice for ordinary calls. The older
                # reasoning runners saved their inputs in prompts.jsonl.
                selected = ('/canonical/' in name or name.endswith('/units.jsonl')
                            or ('/nn-reason-' in name and name.endswith('/prompts.jsonl')))
                if not selected:
                    continue
                count = 0
                with z.open(name) as f:
                    for line in f:
                        row = json.loads(line)
                        key = row.get('prompt_id') or row.get('unit_id')
                        if key:
                            ids.add(key)
                            count += 1
                studies.add(name.split('/')[1])
                prompt_sources.append({'asset': asset.name, 'path': name, 'rows': count})
        print('Indexed', asset.name, flush=True)
    tables = []
    for asset in sorted(folder.glob('raw-responses-*.zip')):
        with zipfile.ZipFile(asset) as z:
            for name in z.namelist():
                if not (name.endswith('.csv.gz') or name.endswith('.csv')):
                    continue
                study = Path(name).name.removesuffix('.raw.csv.gz')
                core = name.startswith(('api/', 'final_local/')) or study in studies
                if not core:
                    tables.append({'path': name, 'scope': 'historical exploratory; not audited'})
                    continue
                rows = matched = blank = 0
                missing = collections.Counter()
                statuses = collections.Counter()
                with z.open(name) as raw:
                    stream = gzip.GzipFile(fileobj=raw) if name.endswith('.gz') else raw
                    with io.TextIOWrapper(stream, encoding='utf-8-sig', newline='') as f:
                        for row in csv.DictReader(f):
                            rows += 1
                            key = row.get('prompt_id') or row.get('sample_id')
                            statuses[row.get('status', '')] += 1
                            if not key:
                                blank += 1
                            elif key in ids:
                                matched += 1
                            else:
                                missing[key] += 1
                tables.append({'path': name, 'rows': rows, 'matched_rows': matched,
                               'blank_identifier_rows': blank, 'unmatched_rows': sum(missing.values()),
                               'unmatched_ids': dict(missing), 'statuses': dict(statuses)})
                print('Audited', name, rows, matched, 'matched', flush=True)
    traces = []
    for asset in sorted(folder.glob('trace-label-records-*.zip')):
        with zipfile.ZipFile(asset) as z:
            for name in z.namelist():
                if not name.endswith('/annotations.jsonl'):
                    continue
                units_name = name.rsplit('/', 1)[0] + '/units.jsonl'
                with z.open(units_name) as f:
                    unit_ids = {json.loads(line)['unit_id'] for line in f}
                count = matched = 0
                statuses = collections.Counter()
                with z.open(name) as f:
                    for line in f:
                        row = json.loads(line)
                        count += 1
                        matched += row.get('unit_id') in unit_ids
                        statuses[row.get('status', '')] += 1
                traces.append({'path': name, 'rows': count, 'matched_units': matched,
                               'statuses': dict(statuses)})
    result = {'scope': 'Identifier linkage, not an assertion of independent calls or successful outputs.',
              'catalogue_prompt_ids': catalogue_count, 'all_indexed_prompt_ids': len(ids),
              'prompt_sources': prompt_sources, 'response_tables': tables, 'trace_annotations': traces}
    (folder / 'PROMPT_RESPONSE_COVERAGE.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--catalogue', type=Path, default=Path('data/prompts/catalog.jsonl.xz'))
    args = parser.parse_args()
    result = audit(args.folder, args.catalogue)
    print(json.dumps({'tables': len(result['response_tables']),
                      'unmatched_rows': sum(t.get('unmatched_rows', 0) for t in result['response_tables'])}))
