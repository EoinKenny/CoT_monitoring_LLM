import json,hashlib
from pathlib import Path
import numpy as np
R=Path('/home/ubuntu/experiments/ten-dataset-full-20261002');assert (R/'COMPLETED').exists();results=json.loads((R/'results.json').read_text());audits={}
for ds,res in results.items():
 rows=json.loads((R/f'{ds}_metadata.json').read_text());x=np.load(R/f'{ds}_features.npy',mmap_mode='r');p=np.load(R/f'{ds}_predictions.npz');assert x.shape==(len(rows),2048);assert np.isfinite(x).all();assert np.isfinite(p['probabilities']).all()
 assert np.array_equal(p['labels'],[r['label'] for r in rows]);assert np.array_equal(p['splits'],[r['split'] for r in rows]);assert all(r['label'] in [0,1] for r in rows)
 assert res['counts']['train:0']==res['counts']['train:1']
 for a,b in [('train','val'),('train','test'),('val','test')]:
  for key in ['group','input_hash']:assert not ({r[key] for r in rows if r['split']==a}&{r[key] for r in rows if r['split']==b})
 flags=(p['splits']=='test')&(p['probabilities']>=res['threshold']);assert sum(flags)==res['flagged_test']==res['metrics']['test']['tp']+res['metrics']['test']['fp']
 if ds in ['wildguard','high_stakes','toxicchat']:
  old=np.load(R.parent/'logistic-probes-20261001'/f'{ds}_features.npy',mmap_mode='r');assert np.array_equal(x,old[[r['origin_index'] for r in rows]])
 audits[ds]={'rows':len(rows),'balanced_training':True,'disjoint_groups_and_inputs':True,'flags':int(sum(flags)),'test_positive':int(sum(p['labels'][p['splits']=='test'])),'split_sha256':hashlib.sha256(json.dumps([(r['id'],r['split'],r['group'],r['label']) for r in rows]).encode()).hexdigest()}
(R/'VERIFIED.json').write_text(json.dumps(audits,indent=2));print(json.dumps(audits,indent=2))
