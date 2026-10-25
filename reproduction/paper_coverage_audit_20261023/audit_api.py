import csv,json,sqlite3,collections,hashlib,gzip,re,array,importlib.util,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent
CAT=PROJECT/'reproduction/paper_complete_20261021/api_export_compact'
SRC=pathlib.Path('/Users/eoinkenny/Desktop/six_judges_responses_5col.csv')
s=importlib.util.spec_from_file_location('kit',CAT/'kit.py');kit=importlib.util.module_from_spec(s);s.loader.exec_module(kit)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):(ROOT/n).write_text(json.dumps(x,indent=2)+'\n')
def csvout(name,fields,rows):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(fields);w.writerows(rows)
db=kit.connect();ident=json.loads((ROOT/'CASE_IDENTITIES.json').read_text())
meta=db.execute('select p.prompt_id,p.study,p.case_key,p.split,p.condition,coalesce(a.mask,0) from prompts p left join assignments a using(prompt_id) order by p.rowid').fetchall();lookup={r[0]:i for i,r in enumerate(meta)}
models=[r[0] for r in db.execute('select model from models order by bit')];bits=dict(db.execute('select model,bit from models'));N=len(meta)
seen={m:bytearray(N) for m in models};valid={m:bytearray(N) for m in models};originalok={m:bytearray(N) for m in models};values={m:array.array('h',[-1])*N for m in models};categories={m:{} for m in models}
counts=collections.Counter();groups=collections.Counter();finishes=collections.Counter();anomalies=collections.Counter();examples={};n=0
issues=gzip.open(ROOT/'api_response_issues.csv.gz','wt',newline='');w=csv.writer(issues);w.writerow(['row','model','prompt_id','study','backbone','dataset','split','condition','issue','original_status','finish_reason','response_excerpt'])
with SRC.open(newline='') as f:
 reader=csv.DictReader(f);assert reader.fieldnames==['prompt_id','model','response','status','finish_reason']
 for row in reader:
  n+=1;pid=row['prompt_id'];m=row['model'];st=row['status'];finish=row['finish_reason'];text=row['response'];finishes[m,st,finish]+=1
  if pid not in lookup or m not in seen:
   anomalies['unknown_id_or_model']+=1;w.writerow([n,m,pid,'','','','','','unknown_id_or_model',st,finish,text[:250]]);continue
  i=lookup[pid];_,study,case,split,cond,mask=meta[i];bb,sample,dataset=ident[case]
  if seen[m][i]:
   anomalies['duplicate_keys']+=1;seen[m][i]=2;valid[m][i]=0;w.writerow([n,m,pid,study,bb,dataset,split,cond,'duplicate_key',st,finish,text[:250]]);continue
  seen[m][i]=1;value,parsed=kit.parse(study,text,finish);cat='invalid_format'
  if st=='ambiguous':cat='ambiguous'
  elif st=='error':cat='error_unspecified'
  elif st=='skipped':cat='skipped'
  elif finish in ['content_filter','content_filtered','safety','blocked']:cat='content_blocked'
  elif parsed=='ok':
   cat='valid' if st=='ok' else 'recoverable_status_mismatch';valid[m][i]=1;originalok[m][i]=(st=='ok')
  elif study=='mechanism' and finish in kit.NORMAL:
   match=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',text.strip(),re.S)
   if match:
    value,parsed=kit.parse(study,match.group(1),finish)
    if parsed=='ok':cat='recoverable_fenced_json';valid[m][i]=1
  if not valid[m][i] and cat=='invalid_format':
   if finish in ['length','max_tokens','max_output_tokens']:cat='truncated_unscorable'
   elif not text.strip():cat='empty_response'
   elif re.search(r"(?i)\b(i (?:cannot|can't|am unable)|i’m unable|i'm unable|cannot (?:assist|provide|help)|unable to (?:assist|provide|help))\b",text):cat='possible_text_refusal'
  if st=='ok' and not valid[m][i]:anomalies['declared_ok_but_invalid']+=1
  if valid[m][i] and isinstance(value,int):values[m][i]=value
  if study=='reason_after' and valid[m][i] and finish in ['length','max_tokens','max_output_tokens']:anomalies['scorable_probability_with_truncated_rationale']+=1
  if study=='reason_after' and valid[m][i] and (len(text.strip().split('\n',1))<2 or not text.strip().split('\n',1)[1].strip()):anomalies['scorable_probability_without_rationale']+=1
  categories[m][i]=cat;counts[m,study,cat]+=1;groups[m,study,bb,dataset,split,cond,cat]+=1
  if cat!='valid':
   w.writerow([n,m,pid,study,bb,dataset,split,cond,cat,st,finish,text[:250]])
   examples.setdefault((study,cat),text[:600])
issues.close();print('Read',n,'rows',flush=True)
# Exact static assignment coverage, then independently reconstruct new-model trace selection.
expected={m:bytearray(N) for m in models};coverage=collections.defaultdict(collections.Counter);missing=[];unexpected=[];selection=[]
for m in models:
 bit=1<<bits[m]
 for i,(_,study,case,split,cond,mask) in enumerate(meta):
  if mask&bit:expected[m][i]=1
 if m in kit.NEW:
  c=collections.Counter()
  for case,base,nn,after,first in db.execute('select * from pairs'):
   a,b=lookup[base],lookup[nn]
   if not (originalok[m][a] and originalok[m][b] and seen[m][a]==seen[m][b]==1):c['excluded_unusable_main_pair']+=1;continue
   if (values[m][a]>=50)==(values[m][b]>=50):c['unchanged']+=1;continue
   c['selected_flips']+=1;expected[m][lookup[after]]=expected[m][lookup[first]]=1
  selection.append(dict(model=m,**c,qualification='CSV-derived only; no dispatch journal or selection receipt supplied'))
 for i,(pid,study,case,split,cond,mask) in enumerate(meta):
  if expected[m][i]:
   coverage[m,study]['expected']+=1
   if not seen[m][i]:
    coverage[m,study]['absent']+=1;bb,sid,ds=ident[case];missing.append([m,pid,study,bb,ds,split,cond])
   else:
    coverage[m,study]['present']+=1
    coverage[m,study]['usable']+=bool(valid[m][i] and seen[m][i]==1)
  elif seen[m][i]:
   coverage[m,study]['unexpected_present']+=1;unexpected.append([m,pid,study,categories[m].get(i,'duplicate')])
# Matched completeness within studies, using observed valid rows. Main older models are intentionally absent here.
pairgroups=collections.defaultdict(list)
for i,(pid,study,case,split,cond,mask) in enumerate(meta):pairgroups[study,case].append(i)
paired=[]
for m in models:
 for study in ['main','ablation','two_nn','mechanism']:
  c=collections.Counter()
  for (s,case),inds in pairgroups.items():
   if s!=study or not any(expected[m][i] for i in inds):continue
   c['case_groups']+=1;c['all_expected_conditions_present']+=all(seen[m][i]==1 for i in inds if expected[m][i]);c['all_expected_conditions_usable']+=all(valid[m][i] and seen[m][i]==1 for i in inds if expected[m][i])
  if c:paired.append(dict(model=m,study=study,**c))
coverage_rows=[dict(model=m,study=s,**c) for (m,s),c in sorted(coverage.items())]
csvout('api_by_model_study.csv',['model','study','expected','present','absent','usable','unexpected_present'],[[r.get(k,0) for k in ['model','study','expected','present','absent','usable','unexpected_present']] for r in coverage_rows])
csvout('api_outcomes.csv',['model','study','outcome','count'],[list(k)+[v] for k,v in sorted(counts.items())])
csvout('api_by_dataset_condition.csv',['model','study','backbone','dataset','split','condition','outcome','count'],[list(k)+[v] for k,v in sorted(groups.items())])
csvout('api_missing_rows.csv',['model','prompt_id','study','backbone','dataset','split','condition'],missing)
csvout('api_unexpected_rows.csv',['model','prompt_id','study','outcome'],unexpected)
dump('api_paired_coverage.json',paired);dump('api_dynamic_selection.json',selection)
dump('api_audit.json',dict(source=str(SRC),source_bytes=SRC.stat().st_size,source_sha256=sha(SRC),catalog_sha256=sha(CAT/'prompts.sqlite'),rows=n,static_expected=sum(bool(r[5]&(1<<bits[m])) for r in meta for m in models),coverage=coverage_rows,outcomes=[dict(model=k[0],study=k[1],outcome=k[2],n=v) for k,v in sorted(counts.items())],source_statuses=[dict(model=k[0],status=k[1],finish=k[2],n=v) for k,v in finishes.items()],anomalies=dict(anomalies),missing_rows=len(missing),unexpected_rows=len(unexpected),dynamic_selection=selection,examples={str(k):v for k,v in examples.items()},time=time.time()))
print(json.dumps({'rows':n,'missing':len(missing),'unexpected':len(unexpected),'anomalies':dict(anomalies),'dynamic':selection},indent=2))
