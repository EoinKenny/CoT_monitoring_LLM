"""Diagnostic: score uncertainty of cases corrected by NN at a fixed judge cutoff of 50."""
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'results/case_mechanisms.csv.gz');d=d[d.scope.eq('common')].copy()
d['helped']=d.correctness_gain.eq(1)
d['confident_wrong']=(d.true_label.eq(0)&d.base_score.ge(.9))|(d.true_label.eq(1)&d.base_score.le(.1))
d['near50']=d.base_score.between(.4,.6)
rows=[]
for (m,b,ds),g in d.groupby(['model','backbone','dataset']):
 w=g.weight/g.weight.sum()
 rows.append(dict(model=m,backbone=b,dataset=ds,help_mass=float((w*g.helped).sum()),confident_help_mass=float((w*g.helped*g.confident_wrong).sum()),near50_help_mass=float((w*g.helped*g.near50).sum())))
a=pd.DataFrame(rows);a.to_csv(R/'results/boundary_score_cells.csv',index=False)
s=a.groupby('model')[['help_mass','confident_help_mass','near50_help_mass']].sum()
s['confident_fraction_of_corrections']=s.confident_help_mass/s.help_mass
s['near50_fraction_of_corrections']=s.near50_help_mass/s.help_mass
s.to_csv(R/'results/boundary_score_summary.csv')
