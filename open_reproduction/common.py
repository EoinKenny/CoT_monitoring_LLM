import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = 'wildguard high_stakes toxicchat beavertails aegis toxigen halueval ragtruth fava wildjailbreak'.split()
BACKBONES = 'llama qwen gemma llama70 qwen122'.split()
JUDGES = 'qwen gemma glm olmo mistral qwen122'.split()
SEED = 20261004


def read(p):
    return json.loads(Path(p).read_text())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def frozen(p, value):
    if Path(p).exists():
        if read(p) != value:
            raise RuntimeError(f'Configuration/input changed: {p}. Use a NEW output directory.')
    else:
        write(p, value)


def reserve(path, extra=0):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < 20*2**30 + extra:
        raise RuntimeError(f'Need 20 GiB reserve plus {extra/2**30:.1f} GiB for next write: {path}')


def records(p):
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(p, 'rt') as f:
        for line in f:
            yield json.loads(line)


def write_records(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def log(out, stage, **kw):
    value = dict(stage=stage, time=time.time(), **kw)
    write(Path(out)/'STATUS.json', value)
    print(json.dumps(value), flush=True)


def completed(mark, inputs):
    if not Path(mark).exists():
        return False
    saved = read(mark)
    if saved['inputs'] != inputs:
        raise RuntimeError(f'Inputs changed for committed stage {mark}')
    for name, h in saved['outputs'].items():
        if sha(Path(mark).parent/name) != h:
            raise RuntimeError(f'Committed output changed: {name}')
    return True


def commit(mark, inputs, files, **kw):
    mark = Path(mark)
    write(mark, dict(inputs=inputs, outputs={str(Path(p).relative_to(mark.parent)): sha(p) for p in files}, **kw))


@contextlib.contextmanager
def lock(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another controller owns {path}') from None
        yield


def validate_rows(rows):
    if len({r['immutable_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate immutable IDs')
    for field in ['group', 'input_hash']:
        sets = {s: {str(r[field]) for r in rows if r['split'] == s} for s in ['train', 'val', 'test']}
        if any(sets[a] & sets[b] for a, b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError(f'Split leakage: {field}')
    if any(r['label'] not in [0,1] or r['split'] not in ['train','val','test'] for r in rows):
        raise ValueError('Invalid dataset record')


def smoke_rows(rows, per_class=8):
    # Smoke-only deterministic small split; never present this as paper results.
    keep = set()
    for split in ['train', 'val', 'test']:
        for label in [0, 1]:
            pool = [r for r in rows if r['split'] == split and r['label'] == label]
            pool.sort(key=lambda r: (r.get('n_tokens', len(str(r['messages']))), r['immutable_id']))
            keep.update(r['immutable_id'] for r in pool[:per_class])
    return [r for r in rows if r['immutable_id'] in keep]
