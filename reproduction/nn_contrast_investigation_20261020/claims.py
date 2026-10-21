"""Machine-readable provenance for the report's principal numerical claims."""
from pathlib import Path
import pandas as pd,json
R=Path(__file__).resolve().parent;O=R/'results';claims=[]
def claim(name,file,filters,column,scale=1):
 t=pd.read_csv(O/file)
 for key,value in filters.items():t=t[t[key].eq(value)]
 assert len(t)==1,(name,len(t));v=float(t.iloc[0][column])*scale;claims.append(dict(name=name,file=file,filters=filters,column=column,multiplier=scale,value=v));return v
claim('Toxicity harmful contrast flips toward NN class (%)','contrast_flip_decomposition.csv',dict(cohort='toxicity4',scope='all_contrast',outcome='harmed',direction='toward'),'fraction',100)
claim('Same estimate where neighbour label and prediction agree (%)','contrast_flip_decomposition.csv',dict(cohort='toxicity4',scope='cues_agree',outcome='harmed',direction='toward'),'fraction',100)
for co in ['all10','toxicity4']:
 for c in ['yes','no']:
  claim('Second-neighbour accuracy gain: '+co+' contrast '+c+' (pp)','second_nn_sensitivity.csv',dict(policy='judge50_direct',cohort=co,contrast=c,alignment='all'),'delta',100)
 claim('Within-cell contrast difference of second-NN gains: '+co+' (pp)','second_nn_balanced_cells.csv',dict(policy='judge50_direct',cohort=co),'difference',100)
for c in ['yes','no']:
 for col,scale in [('aligned',100),('probe_margin_rank',1)]:claim('Toxicity '+col+' contrast '+c,'summary.csv',dict(policy='judge50_direct',cohort='toxicity4',annotator='majority',contrast=c,similarity='all'),col,scale)
for outcome in ['helped','harmed']:
 for col in ['judge_high_confidence','endpoint_to_opposite_endpoint','mean_absolute_shift']:
  claim('Toxicity contrast '+outcome+' '+col,'confidence_summary.csv',dict(cohort='toxicity4',contrast='yes',outcome=outcome),col,1 if col=='mean_absolute_shift' else 100)
claim('Toxicity shared-anchor net accuracy gain (pp)','shared_anchor_sensitivity.csv',dict(cohort='toxicity4',anchor='three_anchor_average',contrast='yes'),'gain',100)
for co in ['all10','toxicity4']:
 for policy in ['judge50_direct','pipeline50']:
  claim(co+' contrast net gain '+policy+' (pp)','summary.csv',dict(cohort=co,policy=policy,annotator='majority',contrast='yes',similarity='all'),'gain',100)
(O/'CLAIMS.json').write_text(json.dumps(claims,indent=2));print('\n'.join(x['name']+': '+str(round(x['value'],5)) for x in claims))
