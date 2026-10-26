#!/usr/bin/env python3
"""Package saved research records; never run inference or alter source archives."""
import argparse, csv, datetime, gzip, hashlib, io, json, re, shutil, tarfile, zipfile
from pathlib import Path, PurePosixPath

CORE = [
    'probability-500-20261012', 'two-nn-20261014',
    'nn-ablation-20261013', 'nn-ablation-gemma-gemma-20261013',
    'nn-ablation-qwen-qwen-20261013', 'nn-ablation-llama-qwen-20261013',
    'nn-ablation-gemma-qwen122-20261013',
    'nn-reason-flips-20261017', 'nn-reason-first-512-20261017',
    'nn-mechanism-two-labels-20261018',
    'nn-mechanism-two-labels-qwen-20261018',
    'nn-mechanism-two-labels-gemma-20261018', 'paper-complete-20261021',
]
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-proj-[A-Za-z0-9_-]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class Parts:
    """Independent ZIPs, each with a streaming per-member checksum manifest."""
    def __init__(self, out, prefix):
        self.out, self.prefix = out, prefix
        self.assets, self.rows, self.z, self.number = [], [], None, 0

    def start(self):
        self.number += 1
        self.path = self.out / f'{self.prefix}-{self.number:03d}.zip'
        assert not self.path.exists(), self.path
        self.z = zipfile.ZipFile(self.path, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=3, allowZip64=True)
        self.rows = []

    def add(self, name, source, size, provenance, *, stored=False, scan=True):
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts
        if self.z is None:
            self.start()
        if self.z.fp.tell() > 400 * 2**20:
            self.close()
            self.start()
        assert name not in self.z.NameToInfo, name
        info = zipfile.ZipInfo(name, (2026, 10, 26, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        info._compresslevel = 3
        h, n, tail = hashlib.sha256(), 0, b''
        with self.z.open(info, 'w', force_zip64=True) as dest:
            while block := source.read(2**20):
                if scan and SECRET.search(tail + block):
                    raise ValueError(f'Potential credential in {name}; review before publication')
                tail = block[-256:]
                dest.write(block); h.update(block); n += len(block)
        assert n == size, (name, size, n)
        self.rows.append(dict(path=name, bytes=n, sha256=h.hexdigest(), source=provenance))

    def file(self, path, name, provenance=None):
        path = Path(path)
        with path.open('rb') as f:
            self.add(name, f, path.stat().st_size, provenance or str(path),
                     stored=path.suffix in {'.gz', '.zip', '.xz', '.npz'},
                     scan=path.suffix not in {'.gz', '.zip', '.xz', '.npz'})

    def close(self):
        if self.z is None:
            return
        self.z.writestr('MEMBERS.json', json.dumps(self.rows, indent=2) + '\n')
        self.z.close()
        assert self.path.stat().st_size < 2 * 2**30
        self.assets.append(dict(name=self.path.name, bytes=self.path.stat().st_size,
                               sha256=sha(self.path), members=len(self.rows),
                               uncompressed_bytes=sum(r['bytes'] for r in self.rows)))
        print('PACKAGED', self.path.name, self.path.stat().st_size, 'bytes', flush=True)
        self.z = None

def selected_context(name):
    p = PurePosixPath(name); low = p.name.lower()
    if low.endswith(('.pem', '.env', '.log', '.pyc', '.lock')):
        return False
    if p.suffix == '.jsonl' and ('canonical' in p.parts or low in {
        'prompts.jsonl', 'units.jsonl', 'mapping_private.jsonl', 'smoke_units.jsonl'}):
        return True
    if low.endswith('.attempt.json') or low.startswith(('failed', 'interrupted')):
        return p.suffix == '.json'
    if p.suffix in {'.json', '.txt'} and (
        len(p.parts) == 2 or low in {'selected_flags.json','selection.json','reused.json',
        'plan.json','frozen.json','approved_config.json','config.json','parent_config.json',
        'run_config.json','model_config.json','completed','completed.json',
        'sampling.json','instruction.txt','prompt_template.txt'} or
        'config' in low or 'protocol' in low or 'frozen' in low):
        return low != 'backup_manifest.json'
    return False

def build_responses(ws, out):
    parts = Parts(out, 'raw-responses'); inventory = []
    old = ws/'paper_data_backup_20261021/remote_archives'
    final = ws/'paper_data_backup_20261024/remote_archives'
    for p in sorted(old.glob('*.raw.csv.gz')):
        meta = json.loads(p.with_name(p.name.replace('.raw.csv.gz', '.DONE.json')).read_text())
        assert sha(p) == meta['csv_sha256']
        parts.file(p, 'historical_local/'+p.name, 'verified 2026-10-21 snapshot')
        inventory.append(dict(path='historical_local/'+p.name, rows=meta['csv_rows'],
                              scope='historical snapshot; may overlap later records'))
    p=final/'paper-complete-20261021.raw.csv.gz'
    meta=json.loads((final/'paper-complete-20261021.DONE.json').read_text())
    assert sha(p)==meta['csv_sha256']
    parts.file(p,'final_local/'+p.name,'verified final 2026-10-24 snapshot')
    inventory.append(dict(path='final_local/'+p.name,rows=meta['csv_rows'],scope='final collection; includes recovery and superseded attempts'))
    api=ws/'reproduction/api_results_20261015/responses.csv'
    raw=api.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((api.parent/'INPUT_MANIFEST.json').read_text())['csv_sha256']
    parts.add('api/original_four_judges_responses.csv',io.BytesIO(raw),len(raw),str(api.relative_to(ws)))
    inventory.append(dict(path='api/original_four_judges_responses.csv',rows=600000,scope='original four API judges'))
    p=ws/'reproduction/paper_coverage_audit_20261023/six_judges_responses_5col.csv.gz'
    with gzip.open(p,'rb') as f:
        h=hashlib.file_digest(f,'sha256').hexdigest()
    assert h==json.loads((p.parent/'AUDIT_VERIFIED.json').read_text())['decompressed_source_sha256']
    parts.file(p,'api/'+p.name,str(p.relative_to(ws)))
    inventory.append(dict(path='api/'+p.name,rows=2869438,scope='six-judge follow-up API upload; original five columns retained'))
    journal=Path('/Users/eoinkenny/Downloads/request_journal.zip')
    audit=json.loads((api.parent/'JOURNAL_AUDIT.json').read_text())
    assert sha(journal)==audit['zip_sha256']
    with zipfile.ZipFile(journal) as z:
        for m in z.infolist():
            if not m.is_dir():
                with z.open(m) as f:
                    parts.add('api/original_four_judges_request_journal.jsonl',f,m.file_size,
                              'request_journal.zip:'+m.filename)
    for rel in [
        'reproduction/api_results_20261015/INPUT_MANIFEST.json',
        'reproduction/api_results_20261015/JOURNAL_AUDIT.json',
        'reproduction/api_results_20261015/frozen/prompt_index.jsonl.gz',
        'reproduction/paper_coverage_audit_20261023/AUDIT_VERIFIED.json',
        'reproduction/paper_coverage_audit_20261023/CASE_IDENTITIES.json',
        'reproduction/paper_coverage_audit_20261023/api_dynamic_selection.json',
        'reproduction/paper_coverage_audit_20261023/api_recovered_mechanism_labels.jsonl.gz',
        'reproduction/paper_coverage_audit_20261023/api_by_model_study.csv',
        'reproduction/paper_coverage_audit_20261023/api_outcomes.csv',
    ]:
        parts.file(ws/rel,'provenance/'+rel,str(rel))
    parts.close()
    return dict(assets=parts.assets, tables=inventory)

def build_contexts(ws,out):
    parts=Parts(out,'prompt-contexts'); links=[]; sources=[]
    for name in CORE:
        base=ws/('paper_data_backup_20261024' if name=='paper-complete-20261021' else 'paper_data_backup_20261021')/'remote_archives'
        archive=base/(name+'.tar.gz');receipt=json.loads((base/(name+'.DONE.json')).read_text())
        # Archive hashes were verified in the backup audit. Recheck current bytes here.
        assert sha(archive)==receipt['archive_sha256']
        print('READING',name,flush=True)
        n=0
        with tarfile.open(archive,'r|gz') as tf:
            for m in tf:
                if m.issym() or m.islnk():
                    if selected_context(m.name):links.append(dict(path=m.name,target=m.linkname))
                elif m.isfile() and selected_context(m.name):
                    with tf.extractfile(m) as f:
                        parts.add('local/'+m.name,f,m.size,archive.name+':'+m.name)
                    n+=1
        sources.append(dict(study=name,source_archive_sha256=receipt['archive_sha256'],selected_files=n))
        print('READ',name,n,'context/attempt records',flush=True)
    # Do not recreate absolute server symlinks. Record the target; originals are
    # supplied in the corresponding study where available.
    body=json.dumps(links,indent=2).encode()
    parts.add('CONTEXT_SYMLINKS.json',io.BytesIO(body),len(body),'source archive symlink metadata')
    parts.close()
    return dict(assets=parts.assets,studies=sources,context_symlinks=len(links))

def build_traces(ws,out):
    parts=Parts(out,'trace-label-records');base=ws/'paper_data_backup_20261026/trace_runs'
    manifest=json.loads((base.parent/'TRACE_SOURCE_MANIFEST.json').read_text())
    # Include the exact prompt units, annotations and native batch attempts from
    # all saved revision runs; the approved Qwen122 revision is marked in docs.
    for p in sorted(base.rglob('*')):
        if p.is_file() and p.suffix in {'.json','.jsonl','.txt'}:
            parts.file(p,'trace_runs/'+str(p.relative_to(base)),str(p.relative_to(ws)))
    parts.close()
    return dict(assets=parts.assets,source_verification=json.loads((base.parent/'TRACE_VERIFIED.json').read_text()))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--section',choices=['responses','contexts','traces'],required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(a.out).free>10*2**30
    result={'responses':build_responses,'contexts':build_contexts,'traces':build_traces}[a.section](a.workspace.resolve(),a.out)
    result['created_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (a.out/(a.section+'_manifest.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
