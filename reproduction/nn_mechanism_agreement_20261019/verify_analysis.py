from pathlib import Path
import json,gzip,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';P=R.parent
source=pd.DataFrame(json.load(open(P/'anchor_transfer_20261018/results/cells.json')));source=source[source.scope!='common_valid']
design=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'))
keep=['policy','model','backbone','dataset','weight','gain'];parts=[]
for c in pd.read_csv(O/'case_outcomes.csv.gz',usecols=keep,chunksize=100000):
 c=c[c.policy.isin(['baseline_anchor','nearest_anchor','saliency_anchor'])].copy();c['numerator']=c.weight*c.gain
 parts.append(c.groupby(['policy','model','backbone','dataset'])[['weight','numerator']].sum())
z=pd.concat(parts).groupby(level=[0,1,2,3]).sum();diffs=[]
old={(r.model,r.backbone,r.dataset,r.anchor,r.condition):r for r in source.itertuples()}
for (policy,m,b,ds),r in z.iterrows():
 anchor=policy.replace('_anchor','');base=old[m,b,ds,anchor,'baseline'];nn=old[m,b,ds,anchor,'nearest']
 n=design['thresholds'][b][ds]['populations']['test']['n'] if base.scope=='local_full' else r.weight
 err=abs(r.numerator/n-(nn.accuracy-base.accuracy));assert err<1e-10,(policy,m,b,ds,err);diffs.append(err)
assert len(diffs)==1500
receipt=dict(replayed_matched_anchor_cells=len(diffs),max_accuracy_delta_error=max(diffs),note='Reconstructed paired outcome numerators reproduce all previous matched-threshold accuracy differences, using each original population denominator. Label-conditioned report rates normalize over the annotated support instead.',time=__import__('time').time())
(O/'REPLAY_VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(receipt)
