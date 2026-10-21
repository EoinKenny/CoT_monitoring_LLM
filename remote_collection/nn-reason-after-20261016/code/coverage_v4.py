from common import *
def main():
 cells_out=[]
 for j in ALL:
  for _,b,c,d in cells(j):
   run=d/'run';rows=[];expected=[p['sample_id'] for p in jl(d/'prompts.jsonl')]
   for path in sorted((run/'batches').glob('*.json')):
    if not path.stem.isdigit():continue
    z=read(path);assert z['config_sha256']==sha(run/'config.json')
    assert z['records_sha256']==digest(json.dumps(z['records'],sort_keys=True))
    if z.get('raw_available',True):assert z['raw_sha256']==sha(path.with_suffix('.raw.json'))
    else:assert z['attempt_sha256']==sha(path.with_suffix('.attempt.json'))
    rows.extend(z['records'])
   ids=[r['sample_id'] for r in rows];assert len(ids)==len(set(ids)) and set(ids)<=set(expected)
   complete=set(ids)==set(expected)
   if complete:
    done=read(run/'COMPLETED');assert sha(run/'judgments.jsonl')==done['judgments_sha256']
    assert {r['sample_id']:r for r in jl(run/'judgments.jsonl')}=={r['sample_id']:r for r in rows}
   invalid=sum(type(r['positive_probability']) is not int or not 0<=r['positive_probability']<=100 for r in rows)
   cells_out.append(dict(backbone=b,judge=j,complete=complete,n=len(rows),valid=len(rows)-invalid,invalid=invalid,ambiguous=sum(r.get('output_status')=='ambiguous_no_raw' for r in rows),blank_reason=sum(not r['reasoning'].strip() for r in rows),capped=sum(r['finish_reason']=='length' for r in rows)))
 totals={k:sum(x[k] for x in cells_out) for k in ['n','valid','invalid','ambiguous','blank_reason','capped']}
 result=dict(time=time.time(),planned=300000,cells=cells_out,totals=totals)
 write(ROOT/'COVERAGE_V4.json',result)
 if all(x['complete'] for x in cells_out):write(ROOT/'COLLECTION_COMPLETED.json',dict(time=time.time(),coverage_sha256=sha(ROOT/'COVERAGE_V4.json'),**totals))
if __name__=='__main__':main()
