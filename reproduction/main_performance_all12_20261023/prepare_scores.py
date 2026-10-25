from pathlib import Path
import numpy as np,gzip,json,csv,re,hashlib,collections
R=Path(__file__).resolve().parent;P=R.parent;C=['baseline','saliency','nearest'];NORMAL={'stop','end_turn','completed','eos','stop_sequence'};INT=r'(?:0|[1-9][0-9]?|100)'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
with gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt') as f:d=json.load(f)
cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];look={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)};pidmap={}
with gzip.open(P/'api_results_20261015/frozen/prompt_index.jsonl.gz','rt') as f:
 for l in f:
  r=json.loads(l);pidmap[r['prompt_id']]=look[r['backbone'],r['sample_id']],C.index(r['condition'])
z=np.load(P/'api_results_20261015/results/scores.npz');scores={m:z[m].copy() for m in z.files};new=['claude-opus-5.5','gpt-5.5']
for m in new:scores[m]=np.full((50000,3),-1,dtype=np.int16)
counts=collections.Counter();changed=[];seen=set();src=Path('/Users/eoinkenny/Desktop/six_judges_responses_5col.csv')
with src.open(newline='') as f:
 for r in csv.DictReader(f):
  if r['prompt_id'] not in pidmap:continue
  m=r['model'];assert m in new;key=m,r['prompt_id'];assert key not in seen;seen.add(key);i,c=pidmap[r['prompt_id']];text=r['response'].strip();value=None
  if r['status'] in ['ok','invalid'] and r['finish_reason'] in NORMAL:
   final=text.splitlines()[-1].strip() if text else ''
   if re.fullmatch(INT,final):value=int(final)
  if value is not None:
   scores[m][i,c]=value;kind='bare_integer' if re.fullmatch(INT,text) else 'final_integer_recovered';counts[m,kind]+=1
   if kind=='final_integer_recovered':changed.append(dict(model=m,prompt_id=r['prompt_id'],value=value,original_status=r['status'],finish_reason=r['finish_reason'],backbone=cases[i]['backbone'],dataset=cases[i]['dataset'],split=cases[i]['split'],condition=C[c]))
  else:counts[m,'unusable']+=1
assert len(seen)==300000
# Legacy API has no finish reason field. Previously valid scores retained. Log otherwise parseable
# invalid tails as unverified completion, rather than infer that a capped output is complete.
legacy=[]
with (P/'api_results_20261015/responses.csv').open(newline='') as f:
 for r in csv.DictReader(f):
  t=r['response'].strip();last=t.splitlines()[-1].strip() if t else ''
  if r['status']=='invalid' and re.fullmatch(INT,last):legacy.append(dict(model=r['model'],prompt_id=r['prompt_id'],reason='No finish metadata; cannot rule out truncation',candidate=int(last)))
np.savez_compressed(R/'scores.npz',**scores)
(R/'PARSING_AUDIT.json').write_text(json.dumps(dict(counts=[dict(model=m,outcome=k,n=n) for (m,k),n in counts.items()],recovered=changed,legacy_candidates_not_rescored=legacy,rule='Final nonempty line must be a standalone integer 0-100. Require normal finish and ok/invalid status for new API data; retain raw text. Explicit blocks, errors, ambiguous requests and capped outputs excluded. Legacy valid records retained; eight invalid legacy tails lack finish metadata.',sources={str(src):sha(src),str(P/'api_results_20261015/results/scores.npz'):sha(P/'api_results_20261015/results/scores.npz')},new_calls=0),indent=2))
print(dict(counts));print('legacy unverified',len(legacy))
