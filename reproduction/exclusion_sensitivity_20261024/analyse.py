import csv,json,hashlib,collections,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
S=R.parent/'main_performance_1nn_2nn_all12_20261023/results/cells.csv'
rows=list(csv.DictReader(S.open()))
cohorts={'All data':(False,False,600),'Without FAVA/HaluEval':(True,False,480),'Without GLM':(False,True,550),'Without both':(True,True,440)}
out=[]
for name,(drop_ds,drop_glm,n) in cohorts.items():
 for policy in ['fixed50','tuned_shared']:
  groups=collections.defaultdict(list)
  for r in rows:
   if r['policy']!=policy or (drop_ds and r['dataset'] in ['fava','halueval']) or (drop_glm and r['model']=='glm'):continue
   groups[r['model'],r['backbone'],r['dataset'],r['condition']].append(r)
  assert len(groups)==4*n
  means=collections.defaultdict(list)
  for key,rs in groups.items():
   assert len(rs)==(1 if policy=='fixed50' else 3)
   if policy=='tuned_shared':assert {r['anchor'] for r in rs}=={'baseline','saliency','nearest2'}
   for metric in ['precision','recall','f1','accuracy']:
    means[key[-1],metric].append(statistics.mean(float(r[metric]) for r in rs)*100)
  result={'cohort':name,'policy':policy,'cells':n}
  for (condition,metric),values in means.items():result[condition+'_'+metric]=statistics.mean(values)
  for condition in ['saliency','nearest','nearest2']:
   for metric in ['precision','recall','f1','accuracy']:result[condition+'_delta_'+metric]=result[condition+'_'+metric]-result['baseline_'+metric]
  out.append(result)
with (R/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(R/'results.json').write_text(json.dumps(out,indent=2))
(R/'VERIFIED.json').write_text(json.dumps({'source':str(S),'sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'analysis':'Descriptive exclusion sensitivity; no threshold retuning. Frozen within-cell design weights and judge-specific coverage preserved. Equal cell means; tuned means average the three shared-threshold anchors within each cell first.','excluded_datasets':['fava','halueval'],'excluded_judge':'glm','expected_cell_counts':{k:v[2] for k,v in cohorts.items()}},indent=2))
for r in out:
 print(r['cohort'],r['policy'],r['cells'])
 print('F1:',{c:round(r[c+'_f1'],3) for c in ['baseline','saliency','nearest','nearest2']})
 print('2NN gains:',{m:round(r['nearest2_delta_'+m],3) for m in ['precision','recall','f1']})
