"""Frozen-score mechanism analysis; no inference, fitting of probes, or threshold reselection."""
from pathlib import Path
import json,gzip,hashlib,time
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
R=Path(__file__).resolve().parent; A=R.parent/'api_results_20261015'; O=R/'results';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'EXPORT_COMPLETED.json').read_text())
for name,h in manifest['files'].items():assert sha(R/name)==h,(name,'hash mismatch')
design=json.load(gzip.open(A/'frozen/design.json.gz','rt'))
cases=[dict(backbone=b,**r) for b in design['samples'] for r in design['samples'][b]]
canonical=pd.DataFrame(cases)
features=[]
for line in gzip.open(R/'features.jsonl.gz','rt'):
 r=json.loads(line);features.append({k:v for k,v in r.items() if k not in ['target_messages','neighbor_messages']})
f=pd.DataFrame(features).set_index(['backbone','sample_id']).loc[pd.MultiIndex.from_frame(canonical[['backbone','sample_id']])].reset_index()
assert len(f)==50000 and not f[['backbone','sample_id']].duplicated().any()
for k in ['dataset','split','true_label','probe_probability','inclusion_probability','group']:
 assert np.array_equal(f[k].to_numpy(),canonical[k].to_numpy()),k
scores=np.load(A/'results/scores.npz');models=scores.files;locals_=models[:6]
common=np.logical_and.reduce([(scores[m]>=0).all(axis=1) for m in models]);assert common.sum()==35897
f['common_valid']=common;f['weight']=1/f.inclusion_probability
f['cluster']=f.dataset+':'+f.target_content_hash
f['cosine_rank']=f.groupby(['backbone','dataset','split']).nearest_cosine.rank(method='average',pct=True)
f['cosine_bin']=np.minimum((f.cosine_rank*5).astype(int),4)+1
for nm in ['target','neighbor']:
 f[nm+'_length_rank']=f.groupby(['backbone','dataset','split'])[nm+'_chars'].rank(method='average',pct=True)
 f[nm+'_length_bin']=np.minimum((f[nm+'_length_rank']*4).astype(int),3)+1
f['label_aligned']=f.neighbor_target_label_agree.astype(int)
f['label_prediction_agree']=f.neighbor_label_prediction_agree.astype(int)
f['length_ratio']=(f.neighbor_chars+1)/(f.target_chars+1)
f.to_csv(O/'joined_features.csv.gz',index=False)
pol=pd.read_csv(A/'results/detailed_metrics.csv')
pol=pol[(pol.scope=='common_valid_all_judges') & (pol.policy=='joint')].set_index(['model','backbone','dataset','condition'])
def avg(v,w):return float(np.average(v,weights=w))
def met(y,p,w):
 tp=float(w[y.astype(bool)&p].sum());fp=float(w[~y.astype(bool)&p].sum());pos=float(w[y.astype(bool)].sum());n=float(w.sum())
 return dict(f1=2*tp/(pos+tp+fp) if pos+tp+fp else 0,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/pos if pos else 0,accuracy=(tp+n-pos-fp)/n,tp=tp,fp=fp)
def auc(y,p,w):return float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else np.nan
cellrows=[];long=[];retrieval=[];maxerr=0
for scope,mask,ms in [('common',common,models),('local_complete',np.ones(len(f),bool),locals_)]:
 for (bb,ds),ix in f[mask & (f.split=='test')].groupby(['backbone','dataset']).groups.items():
  z=f.loc[ix];y=z.true_label.to_numpy();w=z.weight.to_numpy();p=z.probe_probability.to_numpy()
  retrieval.append(dict(scope=scope,backbone=bb,dataset=ds,n=len(z),unique_texts=z.cluster.nunique(),ess=w.sum()**2/(w*w).sum(),prevalence=avg(y,w),nn_positive=avg(z.nearest_label,w),label_alignment=avg(z.label_aligned,w),label_prediction_agree=avg(z.label_prediction_agree,w),cosine=avg(z.nearest_cosine,w),lexical=avg(z.lexical_jaccard,w),chars=avg(z.target_chars,w),neighbor_chars=avg(z.neighbor_chars,w),exact_matches=int(z.exact_text_match.sum())))
  for m in ms:
   s=scores[m][ix].astype(float)/100;b=s[:,0];n=s[:,2]
   row=dict(scope=scope,model=m,backbone=bb,dataset=ds,n=len(z),ess=w.sum()**2/(w*w).sum(),base_auc=auc(y,b,w),nn_auc=auc(y,n,w),base_brier=avg((b-y)**2,w),nn_brier=avg((n-y)**2,w),score_delta=avg(n-b,w),toward_neighbor=avg((n-b)*(2*z.nearest_label.to_numpy()-1),w),toward_gold=avg((n-b)*(2*y-1),w),score_changed=avg(b!=n,w),base_zero=avg(b==0,w),nn_zero=avg(n==0,w),base_100=avg(b==1,w),nn_100=avg(n==1,w))
   for c,ss in [('base',b),('nn',n)]:
    row.update({c+'_judge50_'+k:v for k,v in met(y,ss>=.5,w).items()})
   row['auc_gain']=row['nn_auc']-row['base_auc'];row['brier_gain']=row['base_brier']-row['nn_brier']
   v=z[['backbone','dataset','sample_id','cluster','true_label','weight','probe_probability','nearest_label','nearest_prediction','nearest_cosine','cosine_rank','cosine_bin','label_aligned','label_prediction_agree','target_chars','neighbor_chars','target_length_bin','neighbor_length_bin','lexical_jaccard','length_ratio','nearest_id']].copy()
   v['scope']=scope;v['model']=m;v['base_score']=b;v['nn_score']=n;v['score_delta']=n-b;v['toward_neighbor']=(n-b)*(2*v.nearest_label-1);v['toward_gold']=(n-b)*(2*y-1);v['brier_gain']=(b-y)**2-(n-y)**2;v['correctness_gain']=(n>=.5).astype(int).__eq__(y).astype(int)-(b>=.5).astype(int).__eq__(y).astype(int)
   v['base_error']=np.abs(b-y);v['base_correct']=(b>=.5)==y;v['macro_weight']=w/w.sum()
   if scope=='common':
    pb=pol.loc[(m,bb,ds,'baseline')];pn=pol.loc[(m,bb,ds,'nearest')]
    tb,qb=float(pb.probe_threshold),float(pb.probability_threshold)/100;tn,qn=float(pn.probe_threshold),float(pn.probability_threshold)/100
    preds={'BB':(p>=tb)&(b>=qb),'NB':(p>=tb)&(n>=qb),'BN':(p>=tn)&(b>=qn),'NN':(p>=tn)&(n>=qn)}
    mets={k:met(y,pr,w) for k,pr in preds.items()}
    for k in ['f1','precision','recall']:
     maxerr=max(maxerr,abs(mets['BB'][k]-float(pb[k])),abs(mets['NN'][k]-float(pn[k])))
     row['base_'+k]=mets['BB'][k];row['nn_'+k]=mets['NN'][k];row[k+'_gain']=mets['NN'][k]-mets['BB'][k]
     row[k+'_score_effect']=.5*((mets['NB'][k]-mets['BB'][k])+(mets['NN'][k]-mets['BN'][k]))
     row[k+'_policy_effect']=.5*((mets['BN'][k]-mets['BB'][k])+(mets['NN'][k]-mets['NB'][k]))
     row[k+'_same_policy_gain']=mets['NB'][k]-mets['BB'][k]
     assert abs(row[k+'_score_effect']+row[k+'_policy_effect']-row[k+'_gain'])<1e-12
    row.update(base_gate=tb,nn_gate=tn,base_cutoff=qb,nn_cutoff=qn,base_bypass=int(qb==0),nn_bypass=int(qn==0))
    for name,pr in preds.items():v['decision_'+name]=pr.astype(int)
   cellrows.append(row);long.append(v)
 print('scope completed',scope,flush=True)
assert maxerr<1e-12,maxerr
cells=pd.DataFrame(cellrows);cells.to_csv(O/'cell_mechanisms.csv',index=False)
long=pd.concat(long,ignore_index=True);long.to_csv(O/'case_mechanisms.csv.gz',index=False)
retr=pd.DataFrame(retrieval);retr.to_csv(O/'retrieval_cells.csv',index=False)
for key in ['model','dataset','backbone']:
 cells.groupby(['scope',key]).mean(numeric_only=True).to_csv(O/(key+'_summary.csv'))
# Matched descriptive bins: each dataset/backbone/judge cell contributes one weighted subgroup mean.
rows=[]
for axis in ['label_aligned','label_prediction_agree','cosine_bin','target_length_bin','neighbor_length_bin','base_correct','true_label']:
 for keys,g in long.groupby(['scope','model','backbone','dataset',axis]):
  scope,m,bb,ds,level=keys;w=g.weight.to_numpy()
  row=dict(scope=scope,model=m,backbone=bb,dataset=ds,axis=axis,level=int(level),n=len(g),ess=w.sum()**2/(w*w).sum())
  for col in ['score_delta','toward_neighbor','toward_gold','brier_gain','correctness_gain','label_aligned','nearest_cosine']:
   row[col]=avg(g[col],w)
  rows.append(row)
bins=pd.DataFrame(rows);bins.to_csv(O/'factor_cells.csv',index=False)
bins.groupby(['scope','axis','level']).mean(numeric_only=True).to_csv(O/'factor_summary.csv')
# Dataset-level exploration, ten independent task units, not 500 independent datasets.
stats=pd.DataFrame(json.load(open(R/'dataset_stats.json')));stats['positive_fraction']=stats.positive/stats.n
stats.to_csv(O/'dataset_sizes.csv',index=False)
ds=cells[cells.scope=='common'].groupby('dataset').mean(numeric_only=True)
tr=stats[stats.split=='train'].groupby('dataset').mean(numeric_only=True)
for col in ['n','positive_fraction','unique_groups','mean_chars','median_chars']:ds['train_'+col]=tr[col]
rr=retr[retr.scope=='common'].groupby('dataset').mean(numeric_only=True)
for col in ['label_alignment','label_prediction_agree','cosine','lexical','chars','prevalence']:ds['retrieval_'+col]=rr[col]
ds.to_csv(O/'dataset_explanatory_table.csv')
cor=[]
for x in ['train_n','train_positive_fraction','train_mean_chars','retrieval_label_alignment','retrieval_cosine','retrieval_lexical','retrieval_prevalence','base_f1','base_auc']:
 for y in ['f1_gain','brier_gain','auc_gain']:
  v=ds[[x,y]].dropna();r,p=spearmanr(v[x],v[y]);loo=[spearmanr(v.drop(i)[x],v.drop(i)[y])[0] for i in v.index]
  cor.append(dict(factor=x,outcome=y,n=len(v),spearman=r,p_uncorrected=p,leave_one_out_min=min(loo),leave_one_out_max=max(loo)))
pd.DataFrame(cor).to_csv(O/'dataset_correlations_exploratory.csv',index=False)
# Target overlap, duplicate exposures and common-cohort missingness.
over=[]
for bb,g in f[f.split=='test'].groupby('backbone'):
 for bb2,g2 in f[f.split=='test'].groupby('backbone'):
  over.append(dict(backbone=bb,other=bb2,shared=len(set(g.cluster)&set(g2.cluster)),total=len(g)))
pd.DataFrame(over).to_csv(O/'target_overlap.csv',index=False)
missing=[]
for (bb,ds,y),g in f[f.split=='test'].groupby(['backbone','dataset','true_label']):missing.append(dict(backbone=bb,dataset=ds,true_label=y,n=len(g),common_valid=int(g.common_valid.sum()),fraction=float(g.common_valid.mean())))
pd.DataFrame(missing).to_csv(O/'common_cohort_coverage.csv',index=False)
receipt=dict(time=time.time(),n_features=len(f),common_val=int((common&(f.split=='val')).sum()),common_test=int((common&(f.split=='test')).sum()),common_unique_test_texts=int(f[common&(f.split=='test')].cluster.nunique()),common_test_judge_observations=int((long.scope=='common').sum()),max_previous_metric_error=maxerr,source_hashes={str(p):sha(p) for p in [R/'features.jsonl.gz',R/'dataset_stats.json',A/'frozen/design.json.gz',A/'results/scores.npz',A/'results/detailed_metrics.csv',Path(__file__)]})
(O/'CORE_VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
