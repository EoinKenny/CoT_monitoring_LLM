from pathlib import Path
import json,gzip,hashlib,csv,collections,datetime
R=Path(__file__).resolve().parent;P=R.parents[2];D=R/'nn-reason-after-20261016'
sha=lambda b:hashlib.sha256(b if isinstance(b,bytes) else b.encode()).hexdigest()
promptmap={};prompt_manifest=None
for l in gzip.open(R/'nn-reason-after-20261016.attempted_prompts.jsonl.gz','rt'):
 x=json.loads(l)
 if 'manifest' in x:prompt_manifest=x['manifest'];continue
 r=x['row'];judge=Path(x['source_file']).parts[-2].removesuffix('_nearest_reason_after');k=judge,r['backbone'],r['prompt_id'];assert k not in promptmap;promptmap[k]=x
primary=json.load(gzip.open(P/'reproduction/nn_reason_flips_20261017/FINAL_ANALYSIS_SNAPSHOT.json.gz','rt'))['rows'];prior={(r['judge'],r['backbone'],r['sample_id']):r for r in primary}
anns={(r['backbone'],r['sample_id']):r for r in csv.DictReader(gzip.open(P/'reproduction/nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz','rt'))}
features={(r['backbone'],r['sample_id']):r for r in map(json.loads,gzip.open(P/'reproduction/nn_mechanisms_20261015/features.jsonl.gz','rt'))}
rows=[];cells=[]
for cell in sorted(D.glob('score/*/judges/*')):
 judge=cell.name.removesuffix('_nearest_reason_after');b=cell.parts[-3];plan=json.loads((cell/'run/PLAN.json').read_text());saved=[];committed=set()
 for path in sorted((cell/'run/batches').glob('*.json')):
  if not path.stem.isdigit():continue
  n=int(path.stem);z=json.loads(path.read_text());assert z['records_sha256']==sha(json.dumps(z['records'],sort_keys=True));assert z['config_sha256']==sha((cell/'run/config.json').read_bytes())
  if z.get('raw_available',True):assert z['raw_sha256']==sha(path.with_suffix('.raw.json').read_bytes())
  else:assert z['attempt_sha256']==sha(path.with_suffix('.attempt.json').read_bytes())
  for r in z['records']:saved.append((r,n,str(path),'committed'))
  committed.add(n)
 for path in sorted((cell/'run/batches').glob('*.attempt.json')):
  n=int(path.name.split('.')[0]);a=json.loads(path.read_text())
  if n in committed:continue
  assert not path.with_name(f'{n:06d}.raw.json').exists(),'Uncommitted raw needs separate read-only parsing'
  for pid in a['ids']:
   p=promptmap[judge,b,pid]['row'];saved.append((dict(sample_id=p['sample_id'],dataset=p['dataset'],backbone=b,prompt_id=pid,positive_probability=None,generated_text=None,reasoning='',finish_reason='ambiguous_no_raw',output_status='ambiguous_no_raw'),n,str(path),'attempt_without_raw'))
 c=collections.Counter()
 for r,n,source,status in saved:
  pp=promptmap[judge,b,r['prompt_id']]['row'];key=judge,b,r['sample_id'];f=features[b,r['sample_id']];ann=anns.get((b,r['sample_id']));pr=prior.get(key);raw=r.get('generated_text') or '';valid=type(r.get('positive_probability')) is int and 0<=r['positive_probability']<=100
  assert pp['split']==f['split'];reused=bool(raw and pr and raw==pr.get('generated_text'))
  overlap=pr is not None
  row=dict(judge=judge,backbone=b,sample_id=r['sample_id'],dataset=r['dataset'],split=pp['split'],smoke=plan[n]['smoke'],source_record=r,source_file=source,source_file_sha256=sha(Path(source).read_bytes()),source_status=status,original_fixed50_flip_selected=overlap,exact_output_reused_in_primary=reused,prior_ambiguous_case=bool(pr and pr.get('output_status')=='ambiguous_no_raw'),score_valid=valid,nonempty_raw=bool(raw.strip()),nonempty_rationale=bool((r.get('reasoning') or '').strip()),mechanism_annotation_available=ann is not None,majority_important_contrast=ann['majority_important_contrast'] if ann else None,majority_labelled_similarity=ann['majority_labelled_similarity'] if ann else None,source_prompt_file=promptmap[judge,b,r['prompt_id']]['source_file'],source_prompt_line=promptmap[judge,b,r['prompt_id']]['source_line'],source_prompt_line_sha256=promptmap[judge,b,r['prompt_id']]['line_sha256'])
  rows.append(row)
  for k,v in [('accounted',1),('valid',valid),('raw_text',bool(raw.strip())),('nonempty_rationale',row['nonempty_rationale']),('ambiguous',status=='attempt_without_raw'),('smoke',row['smoke']),('test',row['split']=='test'),('val',row['split']=='val'),('exact_reused_in_primary',reused),('fixed50_flip_overlap',overlap),('new_nonempty_test',row['split']=='test' and bool(raw.strip()) and not reused),('new_important_contrast_test',row['split']=='test' and bool(raw.strip()) and not reused and row['majority_important_contrast']=='yes')]:c[k]+=v
 cells.append(dict(judge=judge,backbone=b,counts=dict(c)))
assert len({(r['judge'],r['backbone'],r['sample_id']) for r in rows})==len(rows)
with gzip.open(R/'broad_saved_records.jsonl.gz','wt') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary=collections.Counter()
for c in cells:summary.update(c['counts'])
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Intentionally paused broad probability-first256 run. All saved responses and attempted-without-raw preserved. Never resumed.',counts=dict(summary),cells=cells,prompt_source_verification=prompt_manifest,records_sha256=sha((R/'broad_saved_records.jsonl.gz').read_bytes()),new_contrast_test_by_dataset=dict(collections.Counter(r['dataset'] for r in rows if r['split']=='test' and r['nonempty_rationale'] and not r['exact_output_reused_in_primary'] and r['majority_important_contrast']=='yes')),validation_annotation_status='Existing approved three-judge mechanism annotations are test-only. Validation rows retained but unlabeled, not silently excluded.',selection_bias='All30smoke cells sampled by dataset and input length, then only Qwen on Llama backbone progressed to production sequentially before user pause. Not a balanced complete experiment.',retrieval_receipt=json.loads((R/'nn-reason-after-20261016.receipt.json').read_text()))
(R/'BROAD_INVENTORY.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ['cells','prompt_source_verification']},indent=2))
