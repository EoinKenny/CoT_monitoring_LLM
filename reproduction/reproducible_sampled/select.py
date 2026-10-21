import sys
from common import *
import numpy as np
b=sys.argv[1];R=ROOT/b;allrows=[];counts={};inputs={}
for ds in sorted(DATASETS):
 meta=read(R/f'{ds}_metadata.json');p=np.load(R/f'{ds}_predictions.npz')['probabilities'];t=read(R/'results.json')[ds]['threshold']
 pool=[dict(sample_id=r['immutable_id'],dataset=ds,index=i,true_label=r['label'],probe_probability=float(p[i]),group=r['group']) for i,r in enumerate(meta) if r['split']=='test' and p[i]>=t]
 selected,summary=sample_flags(pool,b,ds);assert sample_flags(list(reversed(pool)),b,ds)==(selected,summary);allrows.extend(selected);counts[ds]=summary
 inputs[ds]=dict(predictions_sha256=sha(R/f'{ds}_predictions.npz'),probe_sha256=sha(R/f'{ds}_probe.joblib'),threshold=t,split_manifest_sha256=sha(R/f'{ds}_SPLITS.json'))
write(R/'selected_flags.json',allrows)
write(R/'SAMPLE_FROZEN.json',dict(seed=SEED,backbone=b,cap=500,target_per_stratum=250,algorithm='SHA256 UTF8(NUL.join(seed,backbone,dataset,TP_or_FP,immutable_id)); ascending hash then ID; fill scarce-stratum shortfall from other stratum',counts=counts,sample_ids=[r['sample_id'] for r in allrows],selection_sha256=sha(R/'selected_flags.json'),inputs=inputs))
print(json.dumps(dict(backbone=b,selected=len(allrows),counts=counts)),flush=True)
