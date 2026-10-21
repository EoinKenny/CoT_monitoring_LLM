"""Explicit audited repair after missing validation token-cache failure; no judge attempts exist."""
import json,hashlib,time,shutil
from pathlib import Path
R=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=R/'repair_v1';archive.mkdir(exist_ok=False)
for name in ['CODE_FROZEN.json','STATUS.json']:
 shutil.copyfile(R/name,archive/name)
report={'time':time.time(),'reason':'Validation cases beyond the former flagged-test-only evaluation have pooled features but lack token activation caches. Replay original extraction batches and store token logits with a frozen-probe pooling check. Preserve 500 completed Llama explanations. No judge outputs exist or are regenerated.','changed':{}}
for b in ['llama','qwen','gemma']:
 p=R/b/'explanations.py';s=p.read_text();(archive/(b+'_explanations.py')).write_text(s)
 s=s.replace("assert not (R/'explanations.jsonl').exists(), 'Partial explanations require inspection'","completed_rows=[json.loads(line) for line in (R/'explanations.jsonl').open()] if (R/'explanations.jsonl').exists() else []; done_ids={r['sample_id'] for r in completed_rows}; assert len(done_ids)==len(completed_rows)")
 s=s.replace('count=0;started=time.time();maxerr=0','count=len(completed_rows);started=time.time();maxerr=max([r[\'pooling_error\'] for r in completed_rows],default=0)')
 s=s.replace(' for r in subset:\n  ix='," for r in subset:\n  if r['sample_id'] in done_ids:continue\n  ix=")
 start=s.index('  with safe_open');end=s.index('  offs=',start);old=s[start:end]
 new="  replay=B/'replayed_scores'/name/(hashlib.sha256(r['sample_id'].encode()).hexdigest()+'.npz')\n  if replay.exists():\n   proof=json.loads(replay.with_suffix('.json').read_text());assert hashlib.sha256(replay.read_bytes()).hexdigest()==proof['sha256'];z=np.load(replay);ids=z['ids'];score=z['score'];mask=np.isin(np.arange(len(ids)),record['mask']);expected=float(model.decision_function(scaler.transform(X[ix:ix+1]))[0]);err=abs(float(score[mask].mean())-expected);assert err<1e-4;maxerr=max(maxerr,err)\n  else:\n"+''.join(' '+line+'\n' for line in old.splitlines())
 s=s[:start]+new+s[end:];compile(s,str(p),'exec');p.write_text(s);report['changed'][str(p)]={'before':sha(archive/(b+'_explanations.py')),'after':sha(p)}
 if b=='gemma':
  src=Path('/home/ubuntu/experiments/reproducible-sampled-20261004/gemma/backbone_loader.py');text=src.read_text().replace("R=Path('/home/ubuntu/experiments/reproducible-sampled-20261004/gemma')",f'R=Path({str(R/b)!r})');(R/b/'backbone_loader.py').write_text(text)
 # Every preserved completed result belongs to the frozen selection and has exact target text.
 if (R/b/'judge_inputs/explanations.jsonl').exists():
  rows=json.loads((R/b/'selected_flags.json').read_text());ix={r['sample_id']:r for r in rows};done=[json.loads(l) for l in (R/b/'judge_inputs/explanations.jsonl').open()];assert len(done)==len({r['sample_id'] for r in done})
  for ds in {r['dataset'] for r in done}:
   meta=json.loads((R/b/f'{ds}_metadata.json').read_text())
   for r in done:
    if r['dataset']==ds:assert r['messages']==meta[ix[r['sample_id']]['index']]['messages']
  report[b+'_preserved_rows']=len(done)
(R/'REPAIR_v1.json').write_text(json.dumps(report,indent=2))
(R/'CODE_FROZEN.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in R.rglob('*.py') if not p.is_symlink()},indent=2))
print(json.dumps(report),flush=True)
