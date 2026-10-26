"""Portable raw exports and explicit coverage, including failed/ambiguous calls."""
import collections
import csv
import gzip
import json
import sqlite3

from .common import read, sha, write


def export(out):
    con = sqlite3.connect(out/'responses.sqlite')
    catalog = sqlite3.connect('file:'+str(out/'prompts.sqlite')+'?mode=ro',uri=True)
    dest = out/'exports'
    dest.mkdir(exist_ok=True)
    fields = [r[1] for r in con.execute('PRAGMA table_info(requests)')]
    with gzip.open(dest/'responses.csv.gz','wt',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in con.execute('SELECT * FROM requests ORDER BY model,prompt_id'):
            writer.writerow(row)
    case_fields = ['case_key','backbone','dataset','sample_id','split','true_label','probe_probability','population_weight','inclusion_probability','stratum']
    cases = {}
    with gzip.open(dest/'cases.csv.gz','wt',newline='') as f:
        writer = csv.DictWriter(f,case_fields,extrasaction='ignore')
        writer.writeheader()
        for case,bb,ds,meta in catalog.execute('SELECT * FROM cases ORDER BY case_key'):
            row = dict(json.loads(meta),case_key=case,backbone=bb,dataset=ds)
            cases[case] = row
            writer.writerow(row)
    score_fields = ['judge','prompt_id','study','condition','status','probability','labels_json','finish_reason']+case_fields
    with gzip.open(dest/'scores.csv.gz','wt',newline='') as f:
        writer = csv.DictWriter(f,score_fields,extrasaction='ignore')
        writer.writeheader()
        for model,pid,study,case,condition,status,parsed,finish in con.execute('SELECT model,prompt_id,study,case_key,condition,status,parsed,finish_reason FROM requests ORDER BY model,prompt_id'):
            value = json.loads(parsed) if parsed else None
            writer.writerow(dict(cases.get(case,{}),judge=model,prompt_id=pid,study=study,condition=condition,status=status,probability=value if isinstance(value,(int,float)) else None,labels_json=json.dumps(value) if isinstance(value,dict) else None,finish_reason=finish))
    coverage = []
    total_missing = 0
    for plan_file in sorted((out/'judges').glob('*/*_PLAN.json')):
        alias = plan_file.parent.name
        study = plan_file.name.removesuffix('_PLAN.json')
        expected = read(plan_file)['prompt_ids']
        observed = dict(con.execute('SELECT prompt_id,status FROM requests WHERE model=? AND study=?',(alias,study)))
        missing = [pid for pid in expected if pid not in observed]
        unexpected = sorted(set(observed)-set(expected))
        if unexpected:
            raise ValueError(f'Unexpected judgments: {alias}/{study}')
        counts = dict(collections.Counter(observed.values()))
        coverage.append(dict(judge=alias,study=study,expected=len(expected),attempted=len(observed),missing=len(missing),status_counts=counts))
        total_missing += len(missing)
    write(dest/'coverage.json',dict(cells=coverage,unattempted=total_missing,invalid_and_ambiguous_retained=True,probability_first_scored_from_first_line=True))
    with gzip.open(dest/'trace_sources.csv.gz','wt',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_model','source_prompt','label_model','label_prompt'])
        writer.writerows(con.execute('SELECT * FROM trace_sources ORDER BY source_model,source_prompt,label_model'))
    con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    con.close()
    catalog.close()
    checks = {str(p.relative_to(out)):sha(p) for p in dest.iterdir() if p.is_file()}
    checks['responses.sqlite'] = sha(out/'responses.sqlite')
    checks['prompts.sqlite'] = sha(out/'prompts.sqlite')
    write(out/'OUTPUT_SHA256.json',checks)
    result = dict(attempted=sum(c['attempted'] for c in coverage),unattempted=total_missing,valid=sum(c['status_counts'].get('valid',0) for c in coverage),invalid=sum(c['status_counts'].get('invalid',0) for c in coverage),ambiguous=sum(c['status_counts'].get('dispatched',0)+c['status_counts'].get('ambiguous',0) for c in coverage),smoke=read(out/'RUN.json')['smoke'],coverage_sha256=sha(dest/'coverage.json'))
    write(out/('COMPLETED.json' if total_missing == 0 else 'INCOMPLETE.json'),result)
    return result
