from pathlib import Path
import csv,collections,json
R=Path(__file__).resolve().parent
# Reuse the verified case joins, original scores and weights from the plot script.
s=(R/'plot.py').read_text();exec(s[:s.index('plt.rcParams.update')])
datasets=np.array([r['dataset'] for r in rr])[ids];backbones=np.array([r['backbone'] for r in rr])[ids]
rows=[]
for ds in sorted(set(datasets)):
 for group,mask0 in [('all',np.ones(len(x),bool)),('labelled_similarity_yes',labels[:,0]==1),('important_contrast_yes',labels[:,1]==1)]:
  mask=(datasets==ds)&mask0;v={}
  for ei,name in [(0,'helpful'),(1,'unchanged'),(2,'harmful')]:
   m=mask&(ev==ei);v[name+'_mean_cosine']=float(w[m]@x[m]/w[m].sum()) if m.any() else float('nan');v[name+'_n']=int(m.sum())
  m=mask&(ev!=1);xx=x[m];yy=(ev[m]==0).astype(float);ww=w[m]/w[m].sum();mx=ww@xx;my=ww@yy;den=np.sqrt((ww@((xx-mx)**2))*(ww@((yy-my)**2)));v['cosine_helpful_among_flips_correlation']=float(ww@((xx-mx)*(yy-my))/den) if den else float('nan')
  rows.append(dict(dataset=ds,label_group=group,**v))
with open(O/'dataset_cosine.csv','w') as f:q=csv.DictWriter(f,fieldnames=list(rows[0]));q.writeheader();q.writerows(rows)
for r in rows:
 if r['label_group']=='all':print(r['dataset'],round(r['cosine_helpful_among_flips_correlation'],3),round(r['helpful_mean_cosine'],3),round(r['harmful_mean_cosine'],3))
