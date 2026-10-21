"""Freeze an additive, full-score-range threshold study from existing probes."""
import os,json,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
SRC=Path('/home/ubuntu/experiments/grid-completion-20261008')
BBS=['llama','qwen','gemma','llama70','qwen122']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def rank(b,ds,sp,band,y,id):return hashlib.sha256('\0'.join(map(str,[20261009,b,ds,sp,band,y,id])).encode()).hexdigest()
def choose(pool,b,ds,sp):
 bins={k:[] for k in sorted({(r['score_band'],r['true_label']) for r in pool})}
 for r in pool:bins[r['score_band'],r['true_label']].append(r)
 for (band,y),v in bins.items():v.sort(key=lambda r:(rank(b,ds,sp,band,y,r['sample_id']),r['sample_id']))
 # Round-robin equal allocation across nonempty score-band x gold-label strata.
 quota={k:0 for k in bins};remaining=min(500,len(pool))
 while remaining:
  for k in bins:
   if remaining and quota[k]<len(bins[k]):quota[k]+=1;remaining-=1
 selected=[];counts={}
 for k,v in bins.items():
  n=quota[k];assert n>0
  counts[str(k)]={'population':len(v),'sampled':n,'inclusion_probability':n/len(v)}
  for r in v[:n]:selected.append(dict(r,stratum=f'{sp}:{k[0]}:{k[1]}',inclusion_probability=n/len(v),selection_hash=rank(b,ds,sp,*k,r['sample_id'])))
 return sorted(selected,key=lambda r:r['sample_id']),counts

def main():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'SAMPLES_FROZEN.json').exists()
 protocol=dict(seed=20261009,backbones=BBS,judges=['qwen','gemma'],conditions=['baseline','saliency','nearest'],training='Frozen existing full-training balanced-weight probes; no retraining',sampling='Up to 500 validation and 500 test cases per backbone/dataset, independently hash-ranked within validation-defined score-decile x gold-label strata. Round-robin equal quotas with shortage redistribution; exact inclusion probabilities retained.',thresholds='All unique validation probe-score values plus 0, original threshold, and nextafter(1,+inf). No test-score-based threshold construction. Highest threshold on max validation estimated-F1 ties within 1e-12.',nn='Complete original training pool, standardized-activation cosine, immutable-ID similarity ties. Reference probe prediction remains frozen at original threshold across sweep.',evaluation='Horvitz-Thompson TP/FP; exact total positives/negatives; validation-selected thresholds evaluated on independent test sample. Paired within-stratum bootstrap, census strata fixed. Test sweep exploratory; no test-selected optimum claimed.',inference='Existing exact prompts and tokenizer versions, batch 16, max_num_seqs16, max_num_batched_tokens4096, greedy restricted 0/1, seed20261001. Raw outputs and failed attempts retained. No silent judge retry.',reuse='Existing input features/probes reused; no judgment reuse unless identical prompt and complete inference and batch context. This independent sample has different batch contexts, so judgments kept as separate fresh runs.',limits='Prior test inspection makes study exploratory. Partial strata can yield noisy estimates at extreme thresholds. Lower-score explanations require additional preparation; judging ETA excludes that work.')
 write(ROOT/'PROTOCOL.json',protocol);summary={};sourcehash={}
 for b in BBS:
  S=SRC/b;R=ROOT/b;R.mkdir(exist_ok=True)
  for p in S.iterdir():
   if p.name in ['backbone.json','results.json','COMPLETED','judge_rubrics.json','render_content.py','feature_commits'] or any(p.name.endswith('_'+s) for s in ['metadata.json','features.npy','probe.joblib','predictions.npz','SPLITS.json','EXTRACTED.json','BATCHES.json']):
    dest=R/p.name
    if not dest.exists():dest.symlink_to(p.resolve())
  selected=[];counts={};spec={};results=read(S/'results.json')
  for ds in sorted(results):
   meta=read(S/f'{ds}_metadata.json');z=np.load(S/f'{ds}_predictions.npz');p=z['probabilities'];labels=z['labels'];splits=z['splits']
   assert len(meta)==len(p) and len({r['immutable_id'] for r in meta})==len(meta)
   assert all(r['split']==splits[i] and r['label']==labels[i] for i,r in enumerate(meta))
   groups={sp:{r['group'] for r in meta if r['split']==sp} for sp in ['train','val','test']}
   assert all(not groups[a]&groups[c] for a,c in [('train','val'),('train','test'),('val','test')]),(b,ds,'group leakage')
   edges=np.unique(np.quantile(p[splits=='val'],np.arange(.1,1,.1)));thresholds=np.unique(np.r_[0.,p[splits=='val'],results[ds]['threshold'],np.nextafter(1.,2.)]).tolist()
   spec[ds]=dict(score_edges=edges.tolist(),thresholds=thresholds,original_threshold=results[ds]['threshold'],populations={sp:dict(n=int(sum(splits==sp)),positive=int(sum(labels[splits==sp]))) for sp in ['val','test']})
   for sp in ['val','test']:
    pool=[dict(sample_id=r['immutable_id'],dataset=ds,index=i,split=sp,true_label=int(labels[i]),probe_probability=float(p[i]),score_band=int(np.searchsorted(edges,p[i],side='right')),group=r['group']) for i,r in enumerate(meta) if r['split']==sp]
    picked,ct=choose(pool,b,ds,sp);assert choose(list(reversed(pool)),b,ds,sp)==(picked,ct)
    selected.extend(picked);counts[ds+':'+sp]=ct
   for suffix in ['predictions.npz','probe.joblib','SPLITS.json']:sourcehash[str(S/f'{ds}_{suffix}')]=sha(S/f'{ds}_{suffix}')
  # Established explanation generators iterate dataset then selection order.
  selected.sort(key=lambda r:(r['dataset'],r['split'],r['sample_id']))
  assert len({r['sample_id'] for r in selected})==len(selected)
  write(R/'selected_flags.json',selected);write(R/'THRESHOLDS.json',spec)
  write(R/'SAMPLE_FROZEN.json',dict(selection_sha256=sha(R/'selected_flags.json'),sample_ids=[r['sample_id'] for r in selected],counts=counts,seed=20261009,scope='Full score range validation and test, despite legacy filename selected_flags'))
  summary[b]=dict(n=len(selected),validation=sum(r['split']=='val' for r in selected),test=sum(r['split']=='test' for r in selected))
  if b in ['llama','qwen','gemma']:
   source=(S/'explanations.py').read_text();old=f"B=Path('/home/ubuntu/experiments/reproducible-sampled-20261004/{b}')";source=source.replace(old,f'B=Path({str(R)!r})')
   # Existing files remain authoritative; partial rows are inspected on failures, never silently regenerated.
   (R/'explanations.py').write_text(source)
  else:
   common=(SRC/'code/common.py').read_text().replace("B=ROOT/'qwen122'",f"B=ROOT/{b!r}")
   (R/'common.py').write_text(common)
   base=SRC/'code' if b=='qwen122' else Path('/home/ubuntu/experiments/llama70b-full-20261005/code')
   for name in (['runtime.py','saliency.py'] if b=='llama70' else ['saliency.py']):(R/name).write_bytes((base/name).read_bytes())
   (R/'neighbours.py').write_bytes((SRC/'code/neighbours.py').read_bytes())
  # Prompt construction unchanged, but stale legacy protocol text corrected.
  s=(SRC/'code/prepare_judges.py').read_text().replace("protocol='Full eligible training with balanced class weights; validation-F1 threshold; reproducible TP/FP stratified flags; native template'","protocol='Frozen probes; reproducible full-score-range validation/test sample; threshold sweep; native template'")
  (R/'prepare_judges.py').write_text(s)
 # Small shortage and census invariance checks independent of real data.
 tiny=[dict(sample_id=str(i),score_band=i%2,true_label=i%2) for i in range(7)]
 a,c=choose(tiny,'toy','toy','test');assert len(a)==7 and all(r['inclusion_probability']==1 for r in a)
 write(ROOT/'SAMPLES_FROZEN.json',dict(summary=summary,sources=sourcehash,protocol_sha256=sha(ROOT/'PROTOCOL.json'),max_judgments=sum(x['n'] for x in summary.values())*6))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
