"""Aggregate the completed three-backbone, five-judge experiment without new tuning."""
from pathlib import Path
import json,csv,statistics
R=Path('/home/ubuntu/experiments/gemma31b-backbone-20261003');E=R.parent
models=['qwen','gemma','llama','glm','olmo'];conditions=['baseline','saliency','nearest']
allresults=json.loads((E/'expanded-judges-20261003/ALL_FIVE_JUDGES.json').read_text())['results']
allresults['gemma']=json.loads((R/'results.json').read_text())['results']
assert set(allresults)=={'llama','qwen','gemma'}
flat=[];summary=[];verified=0;supp_verified=0
expected=set(json.loads((R/'SAMPLE_FROZEN.json').read_text())['sample_ids'])
for m in models:
 for c in conditions:
  p=R/'judges'/f'{m}_{c}'/'run';rr=[json.loads(l) for l in (p/'judgments.jsonl').open()]
  assert (p/'COMPLETED').exists() and len(rr)==len(expected) and {x['sample_id'] for x in rr}==expected
  assert all(x['prediction'] in [0,1] for x in rr);verified+=len(rr)
manifest=json.loads((R/'SUPPLEMENT_MANIFEST.json').read_text()) if (R/'SUPPLEMENT_MANIFEST.json').exists() else None
if manifest and manifest['frozen']['rows']:
 ids={r['sample_id'] for r in manifest['frozen']['rows']}
 for m in models:
  for c in conditions:
   p=R/'supplement/judges'/f'{m}_{c}'/'run';rr=[json.loads(l) for l in (p/'judgments.jsonl').open()]
   assert (p/'COMPLETED').exists() and len(rr)==len(ids) and {x['sample_id'] for x in rr}==ids
   assert all(x['prediction'] in [0,1] for x in rr);supp_verified+=len(rr)
for b,datasets in allresults.items():
 assert len(datasets)==10
 for ds,d in datasets.items():
  for m in models:
   for c in conditions:
    v=d[m+'_'+c]
    flat.append(dict(backbone=b,dataset=ds,judge=m,condition=c,probe_f1=v['probe']['f1'],cascade_f1=v['cascade_estimate']['f1'],delta_vs_probe=v['f1_change'],ci_low=v['cascade_f1_ci95'][0],ci_high=v['cascade_f1_ci95'][1],sampled_flags=v['sampled_flags'],population_flags=v['population_flags']))
 for m in models:
  means={c:statistics.mean(d[m+'_'+c]['cascade_estimate']['f1'] for d in datasets.values()) for c in conditions}
  probe=statistics.mean(d[m+'_baseline']['probe']['f1'] for d in datasets.values())
  summary.append(dict(backbone=b,judge=m,probe=probe,**means,nn_minus_baseline=means['nearest']-means['baseline'],nn_minus_probe=means['nearest']-probe,nn_helped_datasets=sum(d[m+'_nearest']['cascade_estimate']['f1']>d[m+'_baseline']['cascade_estimate']['f1'] for d in datasets.values())))
for name,rows in [('ALL_RESULTS.csv',flat),('MACRO_RESULTS.csv',summary)]:
 with (R/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/'ALL_BACKBONES_RESULTS.json').write_text(json.dumps(dict(results=allresults,macro=summary),indent=2))
lines=['# Three backbones and five judges','','Macro F1 is the unweighted mean across ten datasets. Probe scores are exact full-test scores; cascade scores are estimates from sampled flags.','','| Backbone | Judge | Probe | Baseline | Saliency | NN | NN − baseline (pp) | NN − probe (pp) | NN improves datasets |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:lines.append('| '+r['backbone']+' | '+r['judge']+' | '+' | '.join(f'{100*r[k]:.2f}' for k in ['probe','baseline','saliency','nearest','nn_minus_baseline','nn_minus_probe'])+f" | {r['nn_helped_datasets']}/10 |")
lines+=['','Supplemental judgments: '+str(supp_verified)+'. Missing nonempty gold-class flag strata were evaluated as separate censuses, under the dated SUPPLEMENT_MANIFEST.json amendment. Original sampled-only metrics exclude supplements; cascade estimates include exact supplemental counts.','','## Interpretation checks','','Compare NN versus baseline separately from NN versus probe. Look for consistency within each judge across backbones, then inspect dataset effects in ALL_RESULTS.csv. No claim of a universal NN benefit is implied by a positive macro average.','','The three backbones share ordered split memberships but can flag different cases. Within a backbone all judge conditions share the same sampled cases. Cross-backbone cascade comparisons are therefore not paired on identical flagged samples. Gemma uses official BF16 weights through block 41; CPU embedding placement and prefix-only execution are engineering adaptations. Backbone family, size, tokenizer and layer differ together. Gemma is also a judge, so inspect the cross-family judges separately.','','Intervals in the per-backbone results are conditional paired row-bootstrap estimates, not training-seed or dataset-population uncertainty. FAVA imbalance, HaluEval length shortcuts, possible benchmark contamination and exploratory reuse of inspected test cases remain limitations. No test-based retuning was performed.']
(R/'ALL_BACKBONES_REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'FINAL_VERIFIED.json').write_text(json.dumps(dict(new_judgments=verified+supp_verified,original_sample_judgments=verified,supplemental_judgments=supp_verified,new_conditions=15,total_backbones=3,judges=5,datasets=10,rows_in_comparison=len(flat),unique_frozen_case_sets_match=True),indent=2))
print(json.dumps(dict(new_judgments=verified+supp_verified,original_sample_judgments=verified,supplemental_judgments=supp_verified,comparison_rows=len(flat))),flush=True)
