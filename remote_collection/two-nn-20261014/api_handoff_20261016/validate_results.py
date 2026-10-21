#!/usr/bin/env python3
"""Validate an API handoff/result file. No dependencies or network access."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

FIELDS = ['prompt_id','model','response','status','input_tokens','output_tokens','latency_ms']
STATUSES = {'ok','invalid','error','ambiguous','skipped'}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prompts',default='prompts.csv')
    ap.add_argument('--responses')
    ap.add_argument('--manifest',default='model_manifest.json')
    ap.add_argument('--require-complete',action='store_true')
    args=ap.parse_args()
    csv.field_size_limit(100000000)
    prompt_path=Path(args.prompts)
    export=json.loads((prompt_path.parent/'export_manifest.json').read_text())
    assert sha(prompt_path)==export['prompts_csv_sha256'],'Prompt checksum mismatch'
    prompts={}
    with prompt_path.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['prompt_id','output_type','system_prompt','user_prompt']
        for row in reader:
            assert row['prompt_id'] not in prompts,'Duplicate prompt ID'
            assert row['output_type'] in {'probability'}
            assert row['system_prompt'] and row['user_prompt']
            assert None not in row,'Malformed CSV record'
            prompts[row['prompt_id']]=row['output_type']
    assert len(prompts)==export['total']==50000
    if not args.responses:
        print(json.dumps({'prompts_verified':len(prompts),'sha256':export['prompts_csv_sha256']}))
        return
    manifest=json.loads(Path(args.manifest).read_text())
    assert manifest['prompts_csv_sha256']==export['prompts_csv_sha256']
    aliases=[m['model'] for m in manifest['models']]
    assert aliases and len(aliases)==len(set(aliases))
    assert all(a and not a.startswith('REPLACE') for a in aliases)
    seen=set(); counts=collections.Counter(); problems=[]
    with open(args.responses,encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==FIELDS,'Unexpected response columns/order'
        for line,row in enumerate(reader,2):
            assert None not in row,'Malformed CSV record'
            pid=row['prompt_id']; model=row['model']; status=row['status']
            assert pid in prompts,f'Unknown prompt ID at logical row {line}'
            assert model in aliases,f'Model alias missing from manifest: {model}'
            key=(pid,model)
            assert key not in seen,f'Duplicate response key at logical row {line}'
            seen.add(key)
            assert status in STATUSES,f'Unknown status at logical row {line}'
            if status=='ok':
                pattern=r'(?:0|[1-9][0-9]?|100)'
                if re.fullmatch(pattern,row['response'].strip()) is None:
                    problems.append({'row':line,'problem':'Invalid probability marked ok','prompt_id':pid,'model':model})
            for field in ['input_tokens','output_tokens']:
                if row[field]: assert re.fullmatch(r'[0-9]+',row[field]),f'Invalid {field} at row {line}'
            if row['latency_ms']: assert float(row['latency_ms'])>=0
            counts[(model,status)]+=1
    by_model={}
    for alias in aliases:
        statuses={s:counts[(alias,s)] for s in sorted(STATUSES)}
        statuses['missing']=len(prompts)-sum(statuses.values())
        by_model[alias]=statuses
    report={'expected_per_model':len(prompts),'models':by_model,'invalid_ok_rows':problems,'unique_response_rows':len(seen),'prompts_sha256':export['prompts_csv_sha256'],'responses_sha256':sha(args.responses),'manifest_sha256':sha(args.manifest)}
    Path('validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not problems,'Some malformed outputs were marked ok'
    if args.require_complete:
        assert all(x['ok']==len(prompts) for x in by_model.values()),'Incomplete valid coverage'

if __name__=='__main__':main()
