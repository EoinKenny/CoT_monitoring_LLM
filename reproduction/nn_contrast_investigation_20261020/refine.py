"""Refine only regressions whose alternating fixed-effect projection had not converged."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import ast,json,shutil
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;O=R/'results';A=R/'numerical_audit';A.mkdir(exist_ok=True)
# Extract pure fitting functions without rerunning descriptive analysis.
def extract(filename,name,old):
 text=(R/filename).read_text().replace(f'range({old})','range(5000)');module=ast.parse(text);f=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name);env={'pd':pd,'np':np};exec(compile(ast.Module(body=[f],type_ignores=[]),filename,'exec'),env);return env[name]
regress=extract('analyze.py','regress',150);fit=extract('conditional.py','fit',600)
cols=['policy','model','backbone','dataset','cluster','cell','w','aligned','true_label','cosine_rank','probe_margin_rank','cue_margin','label_prediction_agree','judge_margin','nearest_label','flip','cue_agrees_control','attraction','gain','away_flip']+[a+'_'+l for a in ['majority','qwen122','gemma','qwen'] for l in ['important_contrast','labelled_similarity']]
d=pd.concat([x[x.policy.eq('judge50_direct')] for x in pd.read_csv(O/'cases.csv.gz',usecols=cols,chunksize=100000)],ignore_index=True)
receipts=[]
for filename,flag in [('regressions.csv','fe_converged'),('risk_regressions.csv','converged')]:
 p=O/filename;tab=pd.read_csv(p);bad=tab[~tab[flag]]
 if len(bad):
  dst=A/('original_'+filename)
  if not dst.exists():shutil.copy2(p,dst)
  keys=['cohort','annotator','outcome','spec'] if filename=='regressions.csv' else ['cohort','annotator','aligned','control_agrees_neighbor','spec']
  for vals,h in bad.groupby(keys):
   sel=dict(zip(keys,vals));g=d if sel['cohort']=='all10' else d[d.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])]
   ann=sel['annotator'];g=g[g[ann+'_important_contrast'].isin(['yes','no'])&g[ann+'_labelled_similarity'].isin(['yes','no'])]
   if filename=='regressions.csv':res=regress(g,ann,sel['outcome'],sel['spec'])
   else:
    g=g[g.aligned.eq(sel['aligned'])&g.cue_agrees_control.eq(sel['control_agrees_neighbor'])];res=fit(g,ann,True)
   for r in res:
    assert r[flag],(filename,sel,r)
    idx=tab.index[tab.term.eq(r['term']) & np.logical_and.reduce([tab[k].eq(v) for k,v in sel.items()])];assert len(idx)==1
    for k,v in r.items():tab.loc[idx,k]=v
   receipts.append(dict(file=filename,selection=sel,iterations=res[0].get('iterations',res[0].get('fe_iterations')),converged=True))
   print(receipts[-1],flush=True)
  tab.to_csv(p,index=False)
 assert tab[flag].all()
for filename,flag in [('regressions.csv','fe_converged'),('risk_regressions.csv','converged')]:
 orig=pd.read_csv(A/('original_'+filename));final=pd.read_csv(O/filename);assert final[flag].all();receipts.append(dict(file=filename,original_unconverged_rows=int((~orig[flag]).sum()),final_unconverged_rows=0,maximum_final_iterations=int(final['fe_iterations' if flag=='fe_converged' else 'iterations'].max())))
(O/'REGRESSION_REFINEMENT.json').write_text(json.dumps(dict(max_iterations=5000,changes=receipts,source_results_preserved=str(A),all_reported_fits_converged=True),indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x)))
