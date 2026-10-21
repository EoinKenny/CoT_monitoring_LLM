"""Outcome-blind nested stratified subsampling, with exact two-stage weights."""
import hashlib,json,collections,math
SEED=20261011
N=100
def rank(backbone,row):
 return hashlib.sha256('\0'.join(map(str,[SEED,'budget110',backbone,row['dataset'],row['stratum'],row['sample_id']])).encode()).hexdigest()
def select(backbone,rows,n=N):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['stratum']].append(r)
 assert len({(r['dataset'],r['split']) for r in rows})==1
 assert len(rows)>=n>=len(groups)
 keys=sorted(groups);q={k:0 for k in keys}
 while sum(q.values())<n:
  for k in keys:
   if q[k]<len(groups[k]):q[k]+=1
   if sum(q.values())==n:break
 chosen={};audit={}
 for k in keys:
  pool=groups[k];oldp=pool[0]['inclusion_probability']
  assert all(abs(r['inclusion_probability']-oldp)<1e-12 for r in pool)
  population=len(pool)/oldp;assert math.isclose(population,round(population),abs_tol=1e-7)
  for r in sorted(pool,key=lambda r:(rank(backbone,r),r['sample_id']))[:q[k]]:
   s=dict(r,parent_inclusion_probability=oldp,parent_selection_hash=r['selection_hash'],selection_hash=rank(backbone,r),conditional_inclusion_probability=q[k]/len(pool),inclusion_probability=oldp*q[k]/len(pool))
   assert math.isclose(s['inclusion_probability'],q[k]/population,abs_tol=1e-12)
   chosen[r['sample_id']]=s
  audit[k]=dict(population=int(round(population)),parent_n=len(pool),selected_n=q[k],probability=q[k]/population)
 assert len(chosen)==n
 return chosen,audit
