"""Prepare a separate full-text census from immutable downloaded records; no inference."""
from pathlib import Path
import json,gzip,hashlib,re,csv,collections,datetime
R=Path(__file__).resolve().parent.parent;D=R/'remote_retrieval/nn-reason-first-512-20261017';O=R/'reasoning_first'
def sha(b):return hashlib.sha256(b if isinstance(b,bytes) else b.encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def writej(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
assert not O.exists(), 'Keep immutable existing corpus; create a new version if needed'
for nm in ['corpus','packets','codes','results','audit']:(O/nm).mkdir(parents=True,exist_ok=True)
primary=[json.loads(l) for l in gzip.open(R/'corpus/rows.jsonl.gz','rt')]
prior={(r['judge'],r['backbone'],r['sample_id']):r for r in primary}
prior_codes={}
for p in (R/'codes').glob('*.jsonl'):
 for line in p.read_text().splitlines():
  x=json.loads(line)
  if x['id'] in prior_codes:assert prior_codes[x['id']]==x
  prior_codes[x['id']]=x
rows=[];checks=[];texts={};per_judge=collections.defaultdict(collections.Counter)
for cell in sorted(D.glob('score/*/judges/*')):
 backbone=cell.parts[-3];judge=cell.name.removesuffix('_nearest_reason_first')
 selection={x['sample_id']:x for x in read(cell/'selection.json')}
 saved=[json.loads(l) for l in (cell/'run/judgments.jsonl').read_text().splitlines()]
 done=read(cell/'run/COMPLETED.json');assert done['judgments_sha256']==sha((cell/'run/judgments.jsonl').read_bytes())
 combined=list(read(cell/'reused.json'));batches=0
 for path in sorted((cell/'run/batches').glob('*.json')):
  if not path.stem.isdigit():continue
  x=read(path);assert x['records_sha256']==sha(json.dumps(x['records'],sort_keys=True));assert x['config_sha256']==sha((cell/'FROZEN.json').read_bytes())
  if x['raw_available']:assert x['raw_sha256']==sha(path.with_suffix('.raw.json').read_bytes())
  else:assert x['attempt_sha256']==sha(path.with_suffix('.attempt.json').read_bytes())
  combined.extend(x['records']);batches+=1
 assert len(saved)==len(combined)==len(selection)
 assert {r['sample_id']:r for r in saved}=={r['sample_id']:r for r in combined}
 assert len({r['sample_id'] for r in saved})==len(saved)
 checks.append(dict(judge=judge,backbone=backbone,n=len(saved),batches=batches,reused=len(read(cell/'reused.json')),judgments_sha256=sha((cell/'run/judgments.jsonl').read_bytes()),source=str(cell/'run/judgments.jsonl')))
 for r in saved:
  key=(judge,backbone,r['sample_id']);p=prior[key];sel=selection[r['sample_id']]
  for k in ['dataset','true_label','control_probability','nn_probability','flip_type']:
   assert p[k]==sel[k],(key,k)
  raw=r.get('generated_text') or '';masked=raw.strip();last_number=None
  match=re.search(r'\n(?:0|[1-9][0-9]?|100)\s*\Z',masked)
  if match:
   last_number=masked[match.start():].strip();masked=masked[:match.start()].rstrip()
  # The original saved parser governs validity, independent of the reading-only mask.
  valid=type(r.get('positive_probability')) is int and 0<=r['positive_probability']<=100
  if valid:
   assert r['finish_reason']=='stop' and last_number is not None and int(last_number)==r['positive_probability']
   assert masked==r['reasoning'].strip()
  t=sha(p['dataset']+'\n'+masked);trace=sha('|'.join([judge,backbone,r['sample_id'],'reasoning_first512_mixed256']))
  row=dict(p,trace_id=trace,text_id=t,condition='nearest_reason_first',reasoning=masked,raw_generated_text=raw,saved_reasoning=r.get('reasoning'),reason_probability=r.get('positive_probability'),score_valid=valid,finish_reason=r['finish_reason'],reasoning_complete=r.get('reasoning_complete'),generated_tokens=r.get('generated_tokens'),output_status=r.get('output_status'),retains_original_nn=((r['positive_probability']>=50)==(p['nn_probability']>=50)) if valid else None,rationale_correct=((r['positive_probability']>=50)==p['true_label']) if valid else None,source_judgment_file=str(cell/'run/judgments.jsonl'),source_prompt_id=r['prompt_id'],source_record=r,final_numeric_line_masked=last_number is not None,masked_numeric_line_unscored=last_number is not None and not valid,intervention='reasoning_first_512_with_640_preserved_256')
  rows.append(row);counter=per_judge[judge];counter['rows']+=1;counter['valid']+=valid;counter['capped']+=r['finish_reason']=='length';counter['nonempty_rationale']+=bool(masked.strip())
  if p['requested_census'] or p['similarity_harm']:
   v=texts.setdefault(t,dict(text_id=t,dataset=p['dataset'],rationale=masked,contrast_requested=False,similarity_harm_requested=False,trace_ids=[],context_ids=[]))
   v['contrast_requested']|=p['requested_census'];v['similarity_harm_requested']|=p['similarity_harm'];v['trace_ids'].append(trace);v['context_ids'].append(p['context_id'])
assert len(rows)==len(prior)==35919
for name,values in [('rows',rows),('text_map',list(texts.values()))]:
 with gzip.open(O/'corpus'/f'{name}.jsonl.gz','wt') as f:
  for r in values:f.write(json.dumps(r,ensure_ascii=False)+'\n')
# Contexts unchanged; preserve bytes and source hash rather than rebuild.
(O/'corpus/contexts.jsonl.gz').write_bytes((R/'corpus/contexts.jsonl.gz').read_bytes())
batches=[];reuse=[]
for role in ['contrast','similarity_harm_only']:
 eligible=[t for t in texts.values() if t['rationale'].strip() and (t['contrast_requested'] if role=='contrast' else not t['contrast_requested'])]
 for dataset in sorted({t['dataset'] for t in eligible}):
  selected=sorted((t for t in eligible if t['dataset']==dataset),key=lambda t:t['text_id'])
  fresh=[]
  for t in selected:
   if t['text_id'] in prior_codes:
    reuse.append(dict(role=role,dataset=dataset,text_id=t['text_id'],code=prior_codes[t['text_id']],basis='Exact dataset plus entire masked rationale byte-equivalence; expressed-only code, never context correctness'))
   else:fresh.append(t)
  for i in range(0,len(fresh),100):
   bid=f'{role}_{dataset}_{i//100:03d}';packet=[dict(id=t['text_id'],text=t['rationale']) for t in fresh[i:i+100]];path=O/'packets'/(bid+'.jsonl');path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in packet))
   batches.append(dict(batch_id=bid,dataset=dataset,role=role,n=len(packet),path=str(path),sha256=sha(path.read_bytes())))
with (O/'BATCHES.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['batch_id','dataset','role','n','path','sha256']);w.writeheader();w.writerows(batches)
writej(O/'REUSED_EXPRESSED_CODES.json',reuse);writej(O/'SOURCE_VERIFICATION.json',dict(cells=checks,remote_completion=read(D/'COMPLETED.json'),remote_retrieval_receipt=read(R/'remote_retrieval/nn-reason-first-512-20261017.receipt.json')))
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(rows),valid=sum(r['score_valid'] for r in rows),capped=sum(r['finish_reason']=='length' for r in rows),nonempty_rationales=sum(bool(r['reasoning'].strip()) for r in rows),contrast_rows=sum(r['requested_census'] for r in rows),similarity_harm_rows=sum(r['similarity_harm'] for r in rows),unique_full_texts_to_read=sum(b['n'] for b in batches),contrast_unique_full_texts_to_read=sum(b['n'] for b in batches if b['role']=='contrast'),exact_prior_codes_reused=len(reuse),per_judge={k:dict(v) for k,v in per_judge.items()},source_selection='Same original fixed50 test flips as primary; no unchanged cases. Native thinking disabled. Separate intervention: rationale before final probability,512 cap except640 preserved256 cases.',masking='Only final standalone 0..100 line removed for reading, whether scored or capped; raw_generated_text and source_record retained verbatim. Capped responses remain unscored. All nonempty rationale text included.',blinding='Reading packets contain only hashed text ID and full rationale; no original outcomes, judge, backbone or numeric answer metadata.',primary_corpus_sha256=sha((R/'corpus/rows.jsonl.gz').read_bytes()),rows_sha256=sha((O/'corpus/rows.jsonl.gz').read_bytes()),contexts_sha256=sha((O/'corpus/contexts.jsonl.gz').read_bytes()),dedup='Only exact dataset+full rationale expressed code reuse; context correctness must be separately audited.',scope_note='All contrast and similarity-harm-only rationale texts included, including capped/unscored outputs; no sample called full coverage.')
writej(O/'CORPUS_MANIFEST.json',manifest)
(O/'CENSUS_READING_GUIDE.txt').write_text((R/'CENSUS_READING_GUIDE.txt').read_text()+'\nSECOND INTERVENTION: reasoning first, final probability masked. Read complete rationale including capped/unscored texts. Keep this intervention separate in synthesis.\n')
(O/'TASK_RUBRICS.json').write_bytes((R/'TASK_RUBRICS.json').read_bytes());print(json.dumps(manifest,indent=2))
