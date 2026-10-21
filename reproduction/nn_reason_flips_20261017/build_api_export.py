"""Rebuild the API flip-only export from immutable prior API inputs/results."""
from pathlib import Path
import collections,csv,gzip,hashlib,io,json,re,zipfile

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
SRC=BASE/'api_results_20261015'
OUT=ROOT/'api_export'
MODELS=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()

manifest=read(SRC/'INPUT_MANIFEST.json')
assert sha(SRC/'responses.csv')==manifest['csv_sha256']
frozen=read(SRC/'frozen/EXPORT_MANIFEST.json')
for n in ['design.json.gz','prompt_index.jsonl.gz']:
    assert sha(SRC/'frozen'/n)==frozen['files'][n]
with gzip.open(SRC/'frozen/design.json.gz','rt') as f:design=json.load(f)
cases={(b,r['sample_id']):dict(backbone=b,**r) for b,rs in design['samples'].items() for r in rs if r['split']=='test'}
index={}; pairs=collections.defaultdict(dict)
with gzip.open(SRC/'frozen/prompt_index.jsonl.gz','rt') as f:
    for line in f:
        r=json.loads(line)
        if r['split']=='test' and r['condition'] in ['baseline','nearest']:
            assert r['prompt_id'] not in index
            index[r['prompt_id']]=r
            pairs[r['backbone'],r['sample_id']][r['condition']]=r['prompt_id']
assert len(cases)==len(pairs)==25000
assert all(set(p)=={'baseline','nearest'} for p in pairs.values())

journal=Path('/Users/eoinkenny/Downloads/request_journal.zip')
assert sha(journal)==read(SRC/'JOURNAL_AUDIT.json')['zip_sha256']
dispatch=collections.Counter(); results=collections.Counter(); statuses={}
with zipfile.ZipFile(journal) as z:
    with z.open('request_journal.txt') as f:
        for line in f:
            r=json.loads(line); k=r['model'],r['prompt_id']
            if r['event']=='dispatch':dispatch[k]+=1
            if r['event']=='result':results[k]+=1;statuses[k]=r['status']
scores={}; seen=set()
for r in csv.DictReader((SRC/'responses.csv').open()):
    k=r['model'],r['prompt_id'];assert k not in seen;seen.add(k)
    assert statuses[k]==r['status']
    if r['prompt_id'] in index:
        scores[k]=int(r['response'].strip()) if r['status']=='ok' and re.fullmatch(r'(?:0|[1-9][0-9]?|100)',r['response'].strip()) else None

selected=collections.defaultdict(list); private=[]; counts={}
for model in MODELS:
    c=collections.Counter(candidate_test_cases=len(cases))
    for key,p in pairs.items():
        ks=[(model,p[x]) for x in ['baseline','nearest']]
        if any(k not in scores or scores[k] is None for k in ks):c['excluded_invalid_or_missing_pair']+=1;continue
        if any(dispatch[k]!=1 or results[k]!=1 for k in ks):c['excluded_ambiguous_journal_pair']+=1;continue
        a,b=[scores[k] for k in ks];c['valid_unambiguous_pairs']+=1
        if (a>=50)==(b>=50):c['unchanged_classification']+=1;continue
        case=cases[key];kind='helped' if (b>=50)==case['true_label'] else 'harmed'
        c['selected_calls']+=1;c[kind]+=1
        selected[p['nearest']].append(model)
        private.append(dict(model=model,parent_prompt_id=p['nearest'],baseline_prompt_id=p['baseline'],**case,baseline_probability=a,nn_probability=b,flip_type=kind))
    assert c['candidate_test_cases']==c['excluded_invalid_or_missing_pair']+c['excluded_ambiguous_journal_pair']+c['unchanged_classification']+c['selected_calls']
    counts[model]=dict(c)

old=(BASE/'nn_reason_after_20261016/OLD_OUTPUT.txt').read_text()
new=(BASE/'nn_reason_after_20261016/NEW_OUTPUT.txt').read_text()
prior=BASE/'probability_500_20261012/api_export'
prior_manifest=read(prior/'export_manifest.json')
source_zip=prior/'external_api_probability_150000_20261012.zip'
with zipfile.ZipFile(source_zip) as z:
    h=hashlib.sha256()
    with z.open('prompts.csv') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    assert h.hexdigest()==prior_manifest['prompts_csv_sha256']
    rows=[]; mapping={}
    with z.open('prompts.csv') as f:
        for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8',newline='')):
            pid=r['prompt_id']
            if pid not in selected:continue
            assert r['system_prompt'].count(old)==1
            assert r['user_prompt'].endswith('Positive-class probability (0-100):')
            system=r['system_prompt'].replace(old,new,1)
            assert system.replace(new,old,1)==r['system_prompt']
            msgs=[dict(role='system',content=system),dict(role='user',content=r['user_prompt'])]
            mh=digest(json.dumps(msgs,ensure_ascii=False,separators=(',',':')))
            newpid='apir_'+digest(pid+':'+mh)
            assert pid not in mapping;mapping[pid]=newpid
            rows.append(dict(prompt_id=newpid,models=json.dumps(selected[pid],separators=(',',':')),system_prompt=system,user_prompt=r['user_prompt']))
assert len(rows)==len(mapping)==len(selected)
rows.sort(key=lambda r:r['prompt_id'])
assert len({r['prompt_id'] for r in rows})==len(rows)
OUT.mkdir(exist_ok=True)
with (OUT/'prompts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['prompt_id','models','system_prompt','user_prompt']);w.writeheader();w.writerows(rows)
for r in private:r['prompt_id']=mapping[r['parent_prompt_id']]
with gzip.open(ROOT/'api_selection_PRIVATE.jsonl.gz','wt') as f:
    for r in sorted(private,key=lambda x:(x['model'],x['prompt_id'])):f.write(json.dumps(r,ensure_ascii=False)+'\n')
save(ROOT/'API_EXPORT_SELECTION_AUDIT.json',dict(counts=counts,unique_prompts=len(rows),total_calls=len(private),source_response_sha256=sha(SRC/'responses.csv'),source_prompt_csv_sha256=prior_manifest['prompts_csv_sha256'],source_journal_sha256=sha(journal),source_design_sha256=sha(SRC/'frozen/design.json.gz'),source_index_sha256=sha(SRC/'frozen/prompt_index.jsonl.gz'),builder_sha256=sha(__file__),private_selection_sha256=sha(ROOT/'api_selection_PRIVATE.jsonl.gz'),trigger_amendment='User requested export immediately on 2026-10-17; no dependency on completion of local inference',exact_prior_api_user_messages=True,only_output_instruction_changed=True))
save(OUT/'export_manifest.json',dict(study='API 1-NN probability first, VERY brief explanation after; fixed50 test flips',models=MODELS,unique_prompts=len(rows),total_calls=len(private),calls_per_model={m:counts[m]['selected_calls'] for m in MODELS},selection='Each model independently: prior valid, unambiguous baseline and nearest test classifications disagree at50. Excludes invalid/missing comparisons and repeated/unresolved dispatches. No new sampling.',excluded_pairs_per_model={m:{k:v for k,v in counts[m].items() if k.startswith('excluded')} for m in MODELS},positive_cutoff=50,prompts_csv_sha256=sha(OUT/'prompts.csv'),output_contract_sha256=digest(new),target_labels_exported=False,prior_scores_exported=False,flip_directions_exported=False))
(OUT/'responses_template.csv').write_text('prompt_id,model,probability,raw_response,status,finish_reason\n')
save(OUT/'model_manifest.template.json',dict(study='nn_reason_flips_api_20261017',models={m:dict(provider=None,resolved_model_id=None,model_snapshot_or_fingerprint=None,endpoint_or_api_version=None,temperature=0,seed=20261001,max_output_tokens=256,native_reasoning_enabled=False,actual_request_parameters={},unsupported_requested_settings=[],run_started_utc=None,run_finished_utc=None) for m in MODELS}))
print(json.dumps(dict(counts=counts,unique_prompts=len(rows),total_calls=len(private)),indent=2))
