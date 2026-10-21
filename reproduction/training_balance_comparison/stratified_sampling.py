"""Deterministic TP/FP evaluation sampling; never used for probe fitting."""
import hashlib

def sample_flags(rows,per_class=250,policy='equal_targets',seed='20261004'):
 assert policy in ['equal_targets','exact_balance'] and per_class>0
 assert len({r['sample_id'] for r in rows})==len(rows)
 assert all(r['true_label'] in [0,1] for r in rows)
 selected=[];counts={}
 for ds in sorted({r['dataset'] for r in rows}):
  pools=[[r for r in rows if r['dataset']==ds and r['true_label']==y] for y in [0,1]]
  quota=min(per_class,*map(len,pools)) if policy=='exact_balance' else per_class
  counts[ds]={}
  for y,pool in enumerate(pools):
   rank=lambda r:hashlib.sha256((seed+'\0'+ds+'\0'+r['sample_id']).encode()).hexdigest()
   chosen=sorted(pool,key=lambda r:(rank(r),r['sample_id']))[:quota]
   selected.extend(chosen);counts[ds][str(y)]=dict(population=len(pool),selected=len(chosen),inclusion_probability=len(chosen)/len(pool) if pool else None,census=len(chosen)==len(pool))
 return selected,dict(policy=policy,seed=seed,target_per_class=per_class,counts=counts,sample_ids=[r['sample_id'] for r in selected],interpretation='Gold labels define evaluation strata only. Sample-only metrics describe this stratified sample; estimate full-test cascades with population class counts. Do not present the sample as uniform across all flags.')

if __name__=='__main__':
 rows=[dict(sample_id=str(i),dataset='x',true_label=int(i>=3)) for i in range(603)]
 a,m=sample_flags(rows);b,n=sample_flags(list(reversed(rows)));assert a==b and m==n
 assert len(a)==253 and m['counts']['x']['0']['census']
 a,m=sample_flags(rows,policy='exact_balance');assert len(a)==6
 a,m=sample_flags(rows[:3]);assert len(a)==3
 assert sample_flags([])[0]==[]
 print('PASS: deterministic order independence, separate 250-per-class quotas, rare-class census, exact-balance alternative, empty classes.')
