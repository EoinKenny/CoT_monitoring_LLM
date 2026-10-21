"""Outcome-independent detection; no judge predictions are inputs to this rule."""
def select_supplement(population, original):
 by={r['sample_id']:r for r in population}
 assert len(by)==len(population)
 ids=[r['sample_id'] for r in original];assert len(ids)==len(set(ids))
 for r in original:
  assert r['sample_id'] in by
  assert (r['dataset'],r['true_label'])==(by[r['sample_id']]['dataset'],by[r['sample_id']]['true_label'])
 present={(r['dataset'],r['true_label']) for r in original}
 absent=sorted({(r['dataset'],r['true_label']) for r in population}-present)
 selected=sorted([r for r in population if (r['dataset'],r['true_label']) in absent],key=lambda r:(r['dataset'],r['true_label'],r['sample_id']))
 assert not set(ids)&{r['sample_id'] for r in selected}
 return selected,[dict(dataset=ds,label=label,population_count=sum(r['dataset']==ds and r['true_label']==label for r in population)) for ds,label in absent]
