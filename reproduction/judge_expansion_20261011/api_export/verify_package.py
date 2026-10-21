#!/usr/bin/env python3
"""Independent pairing verification plus deterministic ZIP packaging."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

R=Path('/home/ubuntu/experiments/judge-expansion-20261011')
P=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
Q=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009')
BBS=['llama','qwen','gemma','llama70','qwen122']
CS=['baseline','saliency','nearest']
OLD='Return exactly one digit: 1 for the positive class or 0 for the negative class. Do not include explanations or other text.'
NEW=json.loads((Q/'PROTOCOL.json').read_text())['output_instruction']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def main():
    pairs=0
    for b in BBS:
        for c in CS:
            with (P/b/'judges'/('qwen_'+c)/'prompts.jsonl').open() as h,(Q/b/'judges'/('qwen_'+c)/'prompts.jsonl').open() as p:
                for hl,pl in zip(h,p,strict=True):
                    hard=json.loads(hl);prob=json.loads(pl)
                    assert hard['sample_id']==prob['sample_id'] and hard['dataset']==prob['dataset']
                    assert digest(hard['prompt'])==prob['parent_prompt_sha256']
                    assert hard['instruction_sha256']==prob['parent_instruction_sha256']
                    restored=prob['prompt'].replace(NEW,OLD,1)
                    k=restored.rindex('Positive-class probability (0-100):')
                    footer='Classification of the CURRENT conversation (0 or 1):' if c=='nearest' else 'Classification (0 or 1):'
                    restored=restored[:k]+footer+restored[k+len('Positive-class probability (0-100):'):]
                    assert restored==hard['prompt']
                    pairs+=1
    assert pairs==150000
    report={'paired_hard_probability_rows':pairs,'identical_noncontract_content':True,'verified_unix':time.time()}
    (R/'canonical/PAIRED_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    out=R/'api_export'
    export=json.loads((out/'export_manifest.json').read_text())
    export['checks']['paired_output_contract_only_difference']=True
    export['paired_hard_probability_rows']=pairs
    export['files']={name:sha(out/name) for name in ['prompts.csv','README.txt','responses_template.csv','model_manifest.template.json','validate_results.py']}
    (out/'export_manifest.json').write_text(json.dumps(export,indent=2)+'\n')
    names=['README.txt','prompts.csv','responses_template.csv','model_manifest.template.json','validate_results.py','export_manifest.json']
    archive=out/'external_api_judges_20261011.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name in names:
            info=zipfile.ZipInfo(name,date_time=(2026,10,11,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o100644 << 16)
            with (out/name).open('rb') as src,z.open(info,'w',force_zip64=True) as dst:
                shutil.copyfileobj(src,dst,8*1024*1024)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==names and z.testzip() is None
        for name in names:
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==sha(out/name)
    with (R/'canonical/private_mapping.jsonl').open('rb') as src,gzip.open(out/'private_mapping.jsonl.gz','wb') as dst:
        shutil.copyfileobj(src,dst,8*1024*1024)
    complete={'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'prompts_csv_bytes':(out/'prompts.csv').stat().st_size,'prompts_csv_sha256':sha(out/'prompts.csv'),'rows':300000,'files':names,'private_mapping_gzip_sha256':sha(out/'private_mapping.jsonl.gz'),'archive_members_checksums_verified':True,'paired_rows':pairs,'time':time.time()}
    (out/'PACKAGE_COMPLETED.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(complete),flush=True)

if __name__=='__main__':main()
