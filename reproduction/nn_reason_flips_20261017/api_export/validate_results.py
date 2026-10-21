"""Standard-library-only integrity, routing and response validator; makes no API calls."""
import argparse,collections,csv,gzip,hashlib,json,re
from pathlib import Path

FIELDS=['prompt_id','model','probability','raw_response','status','finish_reason']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def first_probability(raw):
    lines=raw.strip().splitlines()
    s=lines[0].strip() if lines else ''
    return int(s) if re.fullmatch(r'(?:0|[1-9][0-9]?|100)',s) else None
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prompts',default='prompts.csv');ap.add_argument('--responses');ap.add_argument('--manifest');ap.add_argument('--require-accounted',action='store_true');a=ap.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'export_manifest.json').read_text())
    for name,h in json.loads((root/'CHECKSUMS.json').read_text()).items():assert sha(root/name)==h,('Package checksum mismatch',name)
    assert sha(a.prompts)==manifest['prompts_csv_sha256']
    expected=set();ids=set()
    with open(a.prompts,newline='') as f:
        for r in csv.DictReader(f):
            assert r['prompt_id'] not in ids;ids.add(r['prompt_id'])
            models=json.loads(r['models']);assert len(models)==len(set(models)) and models
            for m in models:
                assert m in manifest['models'];expected.add((r['prompt_id'],m))
    assert len(ids)==manifest['unique_prompts'] and len(expected)==manifest['total_calls']
    assert dict(collections.Counter(m for _,m in expected))==manifest['calls_per_model']
    seen=set();counts=collections.Counter();quality=collections.Counter()
    if a.responses:
        opener=gzip.open if str(a.responses).endswith('.gz') else open
        with opener(a.responses,'rt',newline='') as f:
            reader=csv.DictReader(f);assert reader.fieldnames==FIELDS,reader.fieldnames
            for r in reader:
                k=r['prompt_id'],r['model'];assert k in expected and k not in seen,k;seen.add(k)
                assert r['status'] in ['ok','invalid','error','ambiguous','skipped']
                p=first_probability(r['raw_response'])
                if r['status']=='ok':
                    assert p is not None and r['probability']==str(p),k
                    lines=r['raw_response'].strip().splitlines()
                    if not '\n'.join(lines[1:]).strip():quality['blank_explanation']+=1
                else:
                    assert r['probability']=='',k
                    if r['status']=='invalid':assert p is None,k
                if r['finish_reason'] in ['length','max_tokens','max_output_tokens']:quality['capped']+=1
                counts[r['model'],r['status']]+=1
    if a.manifest:
        mm=json.loads(Path(a.manifest).read_text())
        assert set(mm['models'])==set(manifest['models'])
        for m,spec in mm['models'].items():
            assert spec.get('provider') and spec.get('resolved_model_id'),m
            assert spec.get('actual_request_parameters'),m
            assert 'unsupported_requested_settings' in spec,m
    if a.require_accounted:assert seen==expected,('Unaccounted pairs',len(expected-seen))
    report=dict(package_integrity=True,expected_calls=len(expected),unique_prompts=len(ids),accounted=len(seen),unaccounted=len(expected-seen),by_model_status=[dict(model=m,status=s,n=n) for (m,s),n in sorted(counts.items())],quality_observations=dict(quality),all_accounted=seen==expected,model_manifest_checked=bool(a.manifest))
    Path('validation_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
