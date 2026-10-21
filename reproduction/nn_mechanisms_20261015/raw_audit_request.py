from pathlib import Path
import json,gzip,numpy as np
R=Path(__file__).resolve().parent;A=R.parent/'api_results_20261015'
d=json.load(gzip.open(A/'frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b in d['samples'] for r in d['samples'][b]];s=np.load(A/'results/scores.npz');out=[]
for y in [0,1]:
 for i,r in enumerate(cases):
  if r['backbone']=='llama' and r['dataset']=='halueval' and r['split']=='test' and r['true_label']==y and (s['gemma'][i,0]>=50)==bool(y) and all((s[m][i,0]>=50)!=bool(y) for m in ['glm','mistral','olmo']):
   out.append(dict(**r,expected_scores={m:s[m][i].tolist() for m in ['glm','mistral','olmo','gemma']}));break
(R/'raw_audit_request.json').write_text(json.dumps(out,indent=2))
