import json, hashlib
from pathlib import Path
import numpy as np
base=Path('/home/ubuntu/experiments/logistic-probes-20261001')
weighted=Path('/home/ubuntu/experiments/logistic-weighted-20261001')
out=Path('/home/ubuntu/experiments/judge-pilot-20261001')
out.mkdir(exist_ok=True)
assert not (out/'sample_private.json').exists()
rng=np.random.default_rng(20261001)
rows=[];counts={};hashes={}
for name in ['wildguard','high_stakes','toxicchat']:
    path=weighted/f'{name}_test_predictions.npz'
    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    p=np.load(path)
    meta=json.loads((base/f'{name}_metadata.json').read_text())
    counts[name]={}
    for label,group in [(1,'true_positive'),(0,'false_positive')]:
        pool=np.flatnonzero((p['predictions']==1)&(p['labels']==label))
        counts[name][group]={'available':len(pool),'sampled':50}
        assert len(pool)>=50
        for j in rng.choice(pool,size=50,replace=False):
            i=int(p['indices'][j]);m=meta[i]
            rows.append(dict(dataset=name,index=i,id=m['id'],source=m['source'],messages=m['messages'],true_label=label,probe_prediction=1,probe_probability=float(p['probabilities'][j]),group=group,population_weight=len(pool)/50))
rng.shuffle(rows)
for i,r in enumerate(rows):r['sample_id']=f'pilot-{i+1:03d}'
assert len(set((r['dataset'],r['index']) for r in rows))==300
(out/'sample_private.json').write_text(json.dumps(dict(seed=20261001,counts=counts,prediction_sha256=hashes,records=rows),indent=2))
(out/'judge_inputs.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['sample_id','dataset','messages']})+'\n' for r in rows))
print(json.dumps(counts,indent=2))
