from pathlib import Path
import pandas as pd,numpy as np,json
r=Path('reproduction/nn_contrast_extremity_20261024');d=pd.read_pickle(r/'results/joined.pkl');d=d[(d.contrast==1)&d.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])].copy()
def calc(g):
 a=g.assign(wh=g.w*g.helped,wa=g.w*g.harmed).groupby(['judge','dataset','backbone'])[['w','wh','wa']].sum()
 a=a[['wh','wa']].div(a.w,axis=0).groupby(['judge','dataset']).mean().groupby('judge').mean().mean()*100
 return dict(corrects=a.wh,misleads=a.wa,net=a.wh-a.wa,rows=len(g),targets=g.sample_id.nunique(),judges=g.judge.nunique())
rows=[]
for width in [10,15,20,30]:
 for low in range(0,101-width,5):
  g=d[d.control.between(low,low+width)];rows.append(dict(width=width,low=low,high=low+width,**calc(g)))
b=pd.DataFrame(rows);b.to_json(r/'results/exploratory_band_scan.json',orient='records',indent=2)
print(b[b.width==15].sort_values('net',ascending=False).to_string(index=False));print('OTHER WIDTH WINNERS');print(b.loc[b.groupby('width').net.idxmax()].to_string(index=False))
