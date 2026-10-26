"""Maintainer tool: package audited dataset inputs, never activations or scores.

Run on the original experiment host. The consumer has no dependency on that host.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

DATASETS = 'wildguard high_stakes toxicchat beavertails aegis toxigen halueval ragtruth fava wildjailbreak'.split()
BACKBONES = 'llama qwen gemma llama70 qwen122'.split()
FIELDS = 'id messages label split group input_hash role immutable_id token_ids mask n_tokens content_spans rendered_sha256 empty_user_template_adapter'.split()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=1, description='Original prepared dataset records, audited split membership, and frozen native tokenization. No activations, fitted probes, scores, or retrieved neighbours.', files={})
    for bb in BACKBONES:
        for ds in DATASETS:
            source = args.source / bb / f'{ds}_metadata.json'
            name = f'{bb}__{ds}.jsonl.gz'
            dest = args.out / name
            receipt = dest.with_suffix('.receipt.json')
            if receipt.exists():
                entry = json.loads(receipt.read_text())
                assert sha(source) == entry['source_sha256'] and sha(dest) == entry['sha256']
                manifest['files'][name] = entry
                continue
            rows = json.loads(source.read_text())
            ids = set()
            groups = {s: set() for s in ['train', 'val', 'test']}
            hashes = {s: set() for s in groups}
            counts = collections.Counter()
            with dest.with_suffix('.tmp').open('wb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=6) as f:
                    for row in rows:
                        row = {k: row[k] for k in FIELDS if k in row}
                        assert row['immutable_id'] not in ids
                        ids.add(row['immutable_id'])
                        if 'token_ids' in row:
                            assert row['n_tokens'] == len(row['token_ids'])
                            assert row['mask'] and max(row['mask']) < row['n_tokens']
                        groups[row['split']].add(str(row['group']))
                        hashes[row['split']].add(row['input_hash'])
                        counts[f"{row['split']}:{row['label']}"] += 1
                        f.write((json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode())
            for a in groups:
                for b in groups:
                    if a != b:
                        assert not groups[a] & groups[b], (bb, ds, 'group leakage')
                        assert not hashes[a] & hashes[b], (bb, ds, 'text leakage')
            dest.with_suffix('.tmp').replace(dest)
            entry = dict(backbone=bb, dataset=ds, rows=len(rows), counts=dict(counts), bytes=dest.stat().st_size, sha256=sha(dest), source_sha256=sha(source))
            receipt.write_text(json.dumps(entry, indent=2)+'\n')
            manifest['files'][name] = entry
            print(json.dumps(entry), flush=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
