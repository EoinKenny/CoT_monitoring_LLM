#!/usr/bin/env python3
"""Verify a downloaded raw-data release without model calls or extraction."""
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath

def verify(folder, deep=False):
    manifest=json.loads((folder/'RAW_DATA_MANIFEST.json').read_text())
    counts={'assets':0,'members':0,'bytes':0}
    for entry in manifest['assets']:
        name=entry['name']
        if Path(name).name != name:
            raise ValueError('Asset name must be a filename')
        path=folder/name
        if path.stat().st_size != entry['bytes']:
            raise ValueError(f'Wrong size: {name}')
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=entry['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
        counts['assets']+=1; counts['bytes']+=entry['bytes']
        if deep:
            with zipfile.ZipFile(path) as z:
                records=json.loads(z.read('MEMBERS.json'))
                names=z.namelist()
                if len(names)!=len(set(names)) or set(names)!={r['path'] for r in records}|{'MEMBERS.json'}:
                    raise ValueError(f'Member inventory mismatch: {name}')
                for r in records:
                    p=PurePosixPath(r['path'])
                    if p.is_absolute() or '..' in p.parts or '\\' in r['path']:
                        raise ValueError(f'Unsafe member path: {name}')
                    if z.getinfo(r['path']).file_size != r['bytes']:
                        raise ValueError(f'Member size mismatch: {name}:{r["path"]}')
                    with z.open(r['path']) as f:
                        if hashlib.file_digest(f,'sha256').hexdigest()!=r['sha256']:
                            raise ValueError(f'Member checksum mismatch: {name}:{r["path"]}')
                    counts['members']+=1
        print('Verified',name,flush=True)
    return counts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder',type=Path)
    ap.add_argument('--deep',action='store_true',help='Also decompress and verify every original member.')
    a=ap.parse_args(); print(json.dumps(verify(a.folder,a.deep),indent=2))
if __name__=='__main__':main()
