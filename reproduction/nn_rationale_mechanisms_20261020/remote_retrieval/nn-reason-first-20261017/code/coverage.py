from common import *
def main():
 details=[]
 for j in ALL:
  for _,b,c,d in cells(j):
   pp=list(jl(d/'prompts.jsonl'));rows=list(read(d/'reused.json'))
   for path in sorted((d/'run/batches').glob('*.json')):
    if not path.stem.isdigit():continue
    z=read(path);assert z['records_sha256']==digest(json.dumps(z['records'],sort_keys=True)) and z['config_sha256']==sha(d/'FROZEN.json')
    if z['raw_available']:assert z['raw_sha256']==sha(path.with_suffix('.raw.json'))
    else:assert z['attempt_sha256']==sha(path.with_suffix('.attempt.json'))
    rows+=z['records']
   ids=[r['sample_id'] for r in rows];assert len(ids)==len(set(ids)) and set(ids)<={p['sample_id'] for p in pp}
   if len(rows)==len(pp):
    done=read(d/'run/COMPLETED.json');assert done['judgments_sha256']==sha(d/'run/judgments.jsonl')
    assert {r['sample_id']:r for r in rows}=={r['sample_id']:r for r in jl(d/'run/judgments.jsonl')}
   valid=[r for r in rows if type(r['positive_probability']) is int and 0<=r['positive_probability']<=100]
   labels={x['sample_id']:x for x in read(d/'selection.json')}
   details.append(dict(judge=j,backbone=b,planned=len(pp),accounted=len(rows),valid=len(valid),invalid=len(rows)-len(valid),ambiguous=sum(r.get('output_status')=='ambiguous_no_raw' for r in rows),capped=sum(r['finish_reason']=='length' for r in rows),blank=sum(not r['reasoning'].strip() for r in rows),reproduced_nn=sum((r['positive_probability']>=50)==(labels[r['sample_id']]['nn_probability']>=50) for r in valid)))
 result={k:sum(x[k] for x in details) for k in ['planned','accounted','valid','invalid','ambiguous','capped','blank','reproduced_nn']};result['cells']=details;result['time']=time.time();write(ROOT/'COVERAGE.json',result)
if __name__=='__main__':main()
