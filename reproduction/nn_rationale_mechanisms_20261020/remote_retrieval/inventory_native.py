from pathlib import Path
import json,gzip,csv,hashlib,collections,datetime
R=Path(__file__).resolve().parent;P=R.parents[2]
anns={(r['backbone'],r['sample_id']):r for r in csv.DictReader(gzip.open(P/'reproduction/nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz','rt'))}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['reasoning-full-pilot-20261011','reasoning-benchmark-two-batches-20261011','reasoning-benchmark-quick-20261011','reasoning-benchmark-20261011','gptoss120-smoke-20261011'];summary=[];allrows=[]
for name in names:
 d=R/name;counts=collections.Counter();bys=collections.defaultdict(collections.Counter);files=[]
 reparsed={x['tag']:x for x in json.loads((d/'parsed_results_v2.json').read_text())} if (d/'parsed_results_v2.json').exists() else {}
 for p in sorted(d.rglob('*.raw.json')):
  raw=json.loads(p.read_text());rp=p.with_name(p.name.replace('.raw.json','.result.json'));results=json.loads(rp.read_text()) if rp.exists() else None
  original_results=results
  repaired=reparsed.get(p.name.replace('.raw.json',''))
  if repaired:results=repaired
  by={r['id']:r for r in results.get('records',[])} if results else {}
  if results and 'raw_sha256' in results:assert results['raw_sha256']==sha(p)
  judge=p.relative_to(d).parts[0] if name!='gptoss120-smoke-20261011' else 'gptoss'
  files.append(dict(path=str(p),sha256=sha(p),n=len(raw),result_available=bool(results)))
  for i,r in enumerate(raw):
   rid=r['id'];parts=rid.split('|');rr=by.get(rid);b=ds=sid=cond=mode=None
   if len(parts)==4:b,sid,cond,mode=parts;ds=sid.split(':')[0]
   else:
    parts=rid.split(':')
    if len(parts)==5:b,cond,ds,h,mode=parts;sid=ds+':'+h
   a=anns.get((b,sid));candidate=bool(a and cond=='nearest' and a['majority_important_contrast']=='yes')
   x=dict(run=name,judge=judge,backbone=b,sample_id=sid,dataset=ds,condition=cond,mode=mode,source_raw_file=str(p),source_raw_sha256=sha(p),source_row=i,raw=r,result=rr,annotation_join_by_id_available=a is not None,potential_contrast_annotation_by_id=candidate,annotation_context_equality_verified=False,authoritative_reparse_source=str(d/'parsed_results_v2.json') if repaired else None,legacy_result_preserved_in_source=bool(repaired))
   allrows.append(x)
   for k,v in [('raw_records',1),('nonempty_raw',bool(r.get('text','').strip())),('parsed_result',rr is not None),('valid_result',bool(rr and rr.get('valid'))),('capped',r.get('finish_reason')=='length'),('nearest_records',cond=='nearest'),('id_matched_test_annotation',a is not None),('nearest_potential_contrast',candidate)]:counts[k]+=v;bys[judge][k]+=v
 summary.append(dict(run=name,counts=dict(counts),by_judge={k:dict(v) for k,v in bys.items()},raw_files=files,completion_markers=[str(p) for p in d.glob('*COMPLETED*')],retrieval_receipt=json.loads((R/(name+'.receipt.json')).read_text())))
with gzip.open(R/'native_saved_records.jsonl.gz','wt') as f:
 for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=summary,records_sha256=sha(R/'native_saved_records.jsonl.gz'),scope='Separate native-thinking pilot interventions, not pooled with output-rationale interventions. Raw traces are generated model text, not evidence of underlying causal reasoning. Full pilot is320balanced base cases x2judges x3conditions x2output contracts. Other roots are timing/feasibility tests with intentional repeated inputs and interrupted earlier pilots.',annotation_join_limit='Potential contrast joins above match backbone/sample_id only. Exact prior target/reference/task context identity has NOT been verified for these earlier pilots; do not treat potential labels as confirmed same-prompt annotations. Do not silently drop unmatched cases.',no_inference=True)
(R/'NATIVE_INVENTORY.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps([{'run':r['run'],'counts':r['counts'],'by_judge':r['by_judge']} for r in summary],indent=2))
