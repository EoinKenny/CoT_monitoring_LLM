"""Freeze a 32-call feasibility test and fetch an official, pinned checkpoint."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, shutil, time
from huggingface_hub import hf_hub_download

ROOT = Path('/home/ubuntu/experiments/mistral-small4-smoke-20261011')
SOURCE = Path('/home/ubuntu/experiments/reasoning-full-pilot-20261011')
CACHE = '/opt/dlami/nvme/interpretable-probe-models'
REPO = 'mistralai/Mistral-Small-4-119B-2603-NVFP4'
REV = '45331841b631f4e281df8e959ea3cc9beb84298a'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, d):
    p = Path(p); t = p.with_suffix(p.suffix + '.tmp')
    with t.open('w') as f:
        json.dump(d, f, indent=2); f.flush(); os.fsync(f.fileno())
    t.replace(p)
def rank(s): return hashlib.sha256(('20261011|mistral-smoke|' + s).encode()).hexdigest()

def main():
    assert not (ROOT/'PREPARED.json').exists(), 'Already prepared; do not overwrite'
    info = json.loads((ROOT/'hf_metadata.json').read_text()); assert info['sha'] == REV
    files = [x for x in info['siblings'] if '/' not in x['rfilename'] and x['rfilename'].endswith(('.json','.jinja','.safetensors','.model','.txt'))]
    size = sum(x.get('size', 0) for x in files)
    assert shutil.disk_usage(CACHE).free > size + 25*2**30
    manifest = json.loads((SOURCE/'PREPARED.json').read_text())
    assert sha(SOURCE/'cases.jsonl') == manifest['cases_sha256']
    assert sha(SOURCE/'SELECTION.json') == manifest['selection_sha256']
    meta = json.loads((SOURCE/'SELECTION.json').read_text())
    cases = [json.loads(x) for x in (SOURCE/'cases.jsonl').open()]
    case_map = {r['id']: r for r in cases}
    datasets = sorted({r['dataset'] for r in meta})
    bbs = ['llama','qwen','gemma','llama70','qwen122']
    conditions = ['baseline','saliency','nearest']
    picked = []; used = set()
    for i in range(16):
        ds = datasets[i % 10]; label = i % 2; bb = bbs[i % 5]; c = conditions[i % 3]
        pool = [r for r in meta if r['dataset'] == ds and r['true_label'] == label and r['backbone'] == bb and (bb,r['sample_id']) not in used]
        assert pool
        r = min(pool, key=lambda r:(rank(bb+'|'+r['sample_id']),r['sample_id']))
        used.add((bb,r['sample_id'])); picked.append(dict(r,condition=c))
    assert sum(r['true_label'] for r in picked) == 8
    selected = []
    for mode in ['hard','probability']:
        for r in picked:
            key = f"{r['backbone']}|{r['sample_id']}|{r['condition']}|{mode}"
            selected.append(case_map[key])
    assert len(selected) == 32 and len({r['id'] for r in selected}) == 32
    save(ROOT/'SELECTION.json',picked)
    (ROOT/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    save(ROOT/'PROTOCOL.json',dict(model=REPO,revision=REV,n_calls=32,n_unique_cases=16,batch_size=16,reasoning_effort='none',conditions=conditions,modes=['hard','probability'],seed=20261001,selection_seed=20261011,description='One batch of 16 hard labels and one batch of probabilities on the same 16 cases, covering all ten datasets and five backbones. Eight positives and eight negatives. Three explanation conditions assigned across cases (6/5/5), not a powered condition comparison. Existing frozen task messages are reused verbatim, with the official Mistral chat template. Gold labels remain in selection/evaluation only.',test_only=True,no_full_run_authorized_by_this_script=True,minimum_free_gib=20,hard_max_tokens=1,probability_max_tokens=8,temperature=0))
    save(ROOT/'PREPARED.json',dict(cases_sha256=sha(ROOT/'cases.jsonl'),selection_sha256=sha(ROOT/'SELECTION.json'),source_cases_sha256=sha(SOURCE/'cases.jsonl'),source_selection_sha256=sha(SOURCE/'SELECTION.json')))
    save(ROOT/'DOWNLOAD_MANIFEST.json',dict(repo=REPO,revision=REV,bytes=size,files=files))
    save(ROOT/'STATUS.json',dict(state='downloading',time=time.time(),bytes=size))
    start = time.time()
    def get(x):
        assert shutil.disk_usage(CACHE).free > 20*2**30
        p = hf_hub_download(REPO,x['rfilename'],revision=REV,cache_dir=CACHE)
        print('Downloaded',x['rfilename'],flush=True); return p
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(get,files))
    model_path = str(Path(paths[0]).parent)
    (ROOT/'model_path.txt').write_text(model_path)
    hashes = {}
    for f in files:
        p = Path(model_path)/f['rfilename']; assert p.stat().st_size == f['size']
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(16*2**20),b''): h.update(chunk)
        hashes[f['rfilename']] = h.hexdigest()
        if f.get('lfs',{}).get('sha256'): assert h.hexdigest() == f['lfs']['sha256']
    save(ROOT/'WEIGHTS_SHA256.json',hashes)
    save(ROOT/'DOWNLOAD_COMPLETED.json',dict(path=model_path,seconds=time.time()-start,weights_sha256=sha(ROOT/'WEIGHTS_SHA256.json')))

if __name__ == '__main__':
    try: main()
    except BaseException:
        import traceback
        save(ROOT/'DOWNLOAD_FAILED.json',dict(time=time.time(),traceback=traceback.format_exc())); raise
