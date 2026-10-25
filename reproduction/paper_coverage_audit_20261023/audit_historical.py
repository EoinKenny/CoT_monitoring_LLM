import gzip,csv,json,pathlib,collections,hashlib,sqlite3
R=pathlib.Path(__file__).resolve().parent;P=R.parent.parent
counts=collections.Counter();groups=collections.Counter();seen=set();duplicates=0
source=P/'reproduction/api_results_20261015/results/combined_judgments.csv.gz'
db=sqlite3.connect(P/'reproduction/paper_complete_20261021/api_export_compact/prompts.sqlite');expected={x[0] for x in db.execute('select prompt_id from prompts where study="main"')}
models=collections.defaultdict(set);unknown=0
with gzip.open(source,'rt',newline='') as f:
 for r in csv.DictReader(f):
  m=r['model'];pid=r['prompt_id'];key=(m,pid)
  if key in seen:duplicates+=1
  seen.add(key);models[m].add(pid);unknown+=pid not in expected
  try:v=int(r['positive_probability']);ok=r['status']=='ok' and 0<=v<=100
  except:ok=False
  outcome='usable' if ok else r['status'];counts[m,outcome]+=1;groups[m,r['dataset'],r['condition'],r['split'],outcome]+=1
main=dict(n=len(seen),duplicates=duplicates,unknown_ids=unknown,models=[dict(model=m,expected=len(expected),present=len(ids),absent=len(expected-ids),extra=len(ids-expected),outcomes={k[1]:v for k,v in counts.items() if k[0]==m}) for m,ids in models.items()])
others=[]
for name in ['nn-reason-flips-20261017','nn-reason-first-512-20261017','two-nn-20261014']:
 path=P/'paper_data_backup_20261021/paper_ready_csv'/(name+'.csv.gz');c=collections.Counter();by=collections.defaultdict(collections.Counter);ss=set();dup=0
 with gzip.open(path,'rt',newline='') as f:
  for r in csv.DictReader(f):
   x=json.loads(r['raw_record_json']);j=r['judge'];key=(j,r['backbone'],r['prompt_id']);dup+=key in ss;ss.add(key)
   v=x.get('positive_probability');ok=type(v) in [int,float] and 0<=v<=100;cat='usable' if ok else 'unscorable';c[cat]+=1;by[j][cat]+=1
   if x.get('finish_reason')=='length':c['capped']+=1;by[j]['capped']+=1
 others.append(dict(study=name,n=len(ss),duplicates=dup,outcomes=dict(c),judges={k:dict(v) for k,v in by.items()},source=str(path)))
out=dict(main=main,companions=others,qualification='Historical main normalized response table and checksum-verified local CSV snapshots; no new inference. Probability fields audited without overwriting original parsing.',main_groups=[list(k)+[v] for k,v in groups.items()])
(R/'historical_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(main=main,companions=others),indent=2))
