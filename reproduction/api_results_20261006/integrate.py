"""Join returned API data without reparsing failures or imputing classifications."""
import csv, gzip, json, hashlib, collections, math
from pathlib import Path

R = Path(__file__).resolve().parent
P = R.parent / 'api_export_20261005/prompts.csv'
L = R.parent / 'reproducible_sampled/results/judge_results.csv'
csv.field_size_limit(10_000_000)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (R/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
def f1(tp, fp, fn):
    return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.

with P.open() as f: prompts = list(csv.DictReader(f))
byid = {r['row_id']: r for r in prompts}
assert len(byid) == len(prompts) == 40179
with gzip.open(R/'results_condensed.csv.gz', 'rt') as f: returned = list(csv.DictReader(f))
lookup = {}
for r in returned:
    key = r['api_model'], r['row_id']
    assert key not in lookup and r['row_id'] in byid
    if r['status'] == 'ok':
        assert r['prediction'] in ['0', '1'] and r['raw_response'].strip() == r['prediction']
    else: assert r['prediction'] == ''
    lookup[key] = r
models = sorted({r['api_model'] for r in returned})
groups = collections.defaultdict(list)
for r in prompts: groups[r['backbone'], r['dataset'], r['condition']].append(r)
coverage=[]; metrics=[]; gaps=[]
for m in models:
    rr=[r for r in returned if r['api_model']==m]; statuses=collections.Counter(r['status'] for r in rr)
    assert len({r['resolved_model_id'] for r in rr})==1
    coverage.append(dict(judge=m,expected=len(prompts),attempted=len(rr),valid=statuses['ok'],api_error=statuses['api_error'],invalid_response=statuses['invalid_response'],missing=len(prompts)-len(rr),valid_fraction=statuses['ok']/len(prompts),resolved_model_id=rr[0]['resolved_model_id'],run_config_sha256=rr[0]['run_config_sha256']))
    for (b,ds,c), rows in sorted(groups.items()):
        base={k:float(rows[0]['test_'+k]) for k in ['tp','fp','fn','tn']}
        for r in rows: assert all(float(r['test_'+k])==v for k,v in base.items())
        for y,key in [(1,'tp'),(0,'fp')]:
            assert math.isclose(sum(float(r['population_weight']) for r in rows if int(r['gold_label'])==y),base[key],abs_tol=1e-7)
        counts=collections.Counter(); retained={0:0.,1:0.}; unknown={0:0.,1:0.}; sc=collections.Counter()
        for r in rows:
            a=lookup.get((m,r['row_id'])); status=a['status'] if a else 'missing';counts[status]+=1
            y=int(r['gold_label']);w=float(r['population_weight'])
            if status=='ok':
                p=int(a['prediction']);retained[y]+=w*p;sc[y,p]+=1
            else:
                unknown[y]+=w;gaps.append(dict(api_model=m,row_id=r['row_id'],backbone=b,dataset=ds,condition=c,status=status))
        tp,fp=retained[1],retained[0];fn=base['fn']+base['tp']-tp
        lower=f1(tp,fp+unknown[0],fn);upper=f1(tp+unknown[1],fp,fn-unknown[1]);complete=counts['ok']==len(rows)
        metrics.append(dict(backbone=b,dataset=ds,judge=m,condition=c,source='api',sampled_flags=len(rows),valid=counts['ok'],api_error=counts['api_error'],invalid_response=counts['invalid_response'],missing=counts['missing'],coverage_complete=complete,probe_f1=f1(base['tp'],base['fp'],base['fn']),pipeline_f1=f1(tp,fp,fn) if complete else '',pipeline_f1_lower=lower,pipeline_f1_upper=upper,pipeline_accuracy=(tp+base['tn']+base['fp']-fp)/sum(base.values()) if complete else '',pipeline_tp=tp if complete else '',pipeline_fp=fp if complete else '',pipeline_fn=fn if complete else '',pipeline_tn=base['tn']+base['fp']-fp if complete else '',full_test_exact=complete and all(float(r['inclusion_probability'])==1 for r in rows),valid_subset_f1=f1(sc[1,1],sc[0,1],sc[1,0]),unknown_positive_weight=unknown[1],unknown_negative_weight=unknown[0]))

# Reconstruct local metrics independently from the frozen prompt export.
with L.open() as f: local=list(csv.DictReader(f))
for r in local:
    rows=groups[r['backbone'],r['dataset'],r['condition']]
    tp=sum(float(p['population_weight'])*int(p['local_'+r['judge']+'_prediction']) for p in rows if p['gold_label']=='1')
    fp=sum(float(p['population_weight'])*int(p['local_'+r['judge']+'_prediction']) for p in rows if p['gold_label']=='0')
    fn=float(rows[0]['test_fn'])+float(rows[0]['test_tp'])-tp
    assert math.isclose(f1(tp,fp,fn),float(r['pipeline_f1']),abs_tol=1e-10)
    r.update(source='local',coverage_complete=True,valid=len(rows),api_error=0,invalid_response=0,missing=0,pipeline_f1_lower=r['pipeline_f1'],pipeline_f1_upper=r['pipeline_f1'])
assert len(local)==450 and len(metrics)==360
save('coverage.csv',coverage);save('api_metrics.csv',metrics);save('combined_judge_results.csv',local+metrics);save('unresolved_rows.csv',gaps)

# Paired descriptive comparisons, restricted explicitly to jointly valid cases.
paired=[]
for m in models:
    for b,ds,c in sorted(groups):
        if c=='baseline':continue
        base={r['sample_id']:r for r in groups[b,ds,'baseline']};n=wr=rw=0
        for r in groups[b,ds,c]:
            a=lookup.get((m,base[r['sample_id']]['row_id']));e=lookup.get((m,r['row_id']))
            if not a or not e or a['status']!='ok' or e['status']!='ok':continue
            n+=1;y=r['gold_label'];wr+=a['prediction']!=y and e['prediction']==y;rw+=a['prediction']==y and e['prediction']!=y
        paired.append(dict(judge=m,backbone=b,dataset=ds,condition=c,matched_valid=n,expected=len(base),baseline_wrong_explanation_right=wr,baseline_right_explanation_wrong=rw,net_correct_gain=wr-rw,scope='Descriptive matched valid subset; no deployment estimate or significance claim'))
save('paired_valid_case_comparisons.csv',paired)

joined_fields=[k for k in prompts[0] if k not in ['system_prompt','user_prompt']]+[k for k in returned[0] if k!='row_id']
with gzip.open(R/'joined_api_results.csv.gz','wt') as f:
    w=csv.DictWriter(f,fieldnames=joined_fields);w.writeheader()
    for a in returned:
        p=byid[a['row_id']];w.writerow({k:(a[k] if k in a else p[k]) for k in joined_fields})
macro=[]
for m in sorted({r['judge'] for r in local+metrics}):
    for b in sorted({r['backbone'] for r in prompts}):
        for c in ['baseline','saliency','nearest']:
            rr=[r for r in local+metrics if (r['judge'],r['backbone'],r['condition'])==(m,b,c)];assert len(rr)==10
            macro.append(dict(judge=m,backbone=b,condition=c,datasets=10,valid=sum(int(r['valid']) for r in rr),expected=sum(int(r['sampled_flags']) for r in rr),macro_f1_lower=sum(float(r['pipeline_f1_lower']) for r in rr)/10,macro_f1_upper=sum(float(r['pipeline_f1_upper']) for r in rr)/10))
save('macro_comparison.csv',macro)
manifest=dict(inputs={str(p):sha(p) for p in [R/'results_condensed.csv.gz',P,L]},script_sha256=sha(__file__),returned_rows=len(returned),unique_keys=len(lookup),expected_per_model=len(prompts),local_cells=len(local),api_cells=len(metrics),complete_api_cells=sum(r['coverage_complete'] for r in metrics),outputs={p.name:sha(p) for p in R.iterdir() if p.suffix in ['.csv','.gz']},provenance_limit='Run configuration hash supplied, but corresponding run manifest not supplied. Anthropic resolved IDs are Bedrock inference-profile ARNs; underlying model versions not independently verified.',methods='Strict original 0/1 parser; no retries, salvage, failure dropping, or reweighting for missingness. F1 intervals are worst/best missing-classification bounds for the frozen weighted sample, not statistical confidence intervals. Successful-subset metrics are descriptive only.')
(R/'INTEGRATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
lines=['API results integrated with the original three-backbone study.','',f'{len(returned):,} returned records; 450 original cells plus 360 API coverage cells.','',manifest['methods'],'',manifest['provenance_limit'],'','Coverage:']
for r in coverage:lines.append(f"{r['judge']}: {r['valid']:,}/{r['expected']:,} valid; {r['api_error']:,} API errors; {r['invalid_response']:,} invalid; {r['missing']:,} absent.")
lines+=['','The 70B run is separate and is not included in this API export. Raw inputs and all unsuccessful rows are preserved. No API calls were made.','Files: combined_judge_results.csv contains original local metrics plus API metrics and bounds; macro_comparison.csv gives equal-dataset macro F1 bounds; joined_api_results.csv.gz restores all evaluation metadata; unresolved_rows.csv enumerates incomplete cases.']
(R/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(coverage=coverage,complete_api_cells=manifest['complete_api_cells'],macro=[r for r in macro if r['judge'].startswith('claude')]),indent=2))
