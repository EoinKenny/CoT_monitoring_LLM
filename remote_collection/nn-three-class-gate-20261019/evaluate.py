from common import *
reserve();assert (R/'TRAINING_COMPLETED.json').exists()
receipts={};choices=[];cuts=[];training=[]
for b in BB:
 p=P/'representations'/b/'COMPLETED.json';receipts[str(p)]=sha(p)
 pr=read(p)
 for name,h in pr['outputs'].items():assert sha(p.parent/name)==h
 for rr in pr['records']:
  fitted=f.iloc[rr['fit_source_indices']];assert fitted.split.eq('val').all()
  assert fitted.backbone.eq(b).all() and fitted.dataset.isin(DS).all()
  if rr['fold']!='full':assert fitted.fold.ne(int(rr['fold'])).all()
 assert pr['shape'][0]==4000
 for m in scores.files:
  for pol in POL:
   folder=O/b/m/pol;rec=read(folder/'COMPLETED.json')
   for name,h in rec['files'].items():assert sha(folder/name)==h
   for name,h in rec['models'].items():assert sha(Path(name))==h
   receipts[str(folder/'COMPLETED.json')]=sha(folder/'COMPLETED.json');choices.append(pd.read_csv(folder/'choices.csv'));cuts.append(pd.read_csv(folder/'cutoffs.csv'));training.append(dict(backbone=b,model=m,policy=pol,validation_rows=rec['validation_rows'],training_disagreements=rec['training_disagreements'],training_rows=rec['training_rows'],helpful=rec['helpful'],harmful=rec['harmful'],unchanged=rec['unchanged']))
ch=pd.concat(choices,ignore_index=True);cu=pd.concat(cuts,ignore_index=True)
assert len(training)==200
vdiags=[]
for b in BB:
 for m in scores.files:
  ca=pd.read_csv(O/b/m/'fixed50'/'candidates.csv')
  for v in SECOND_VARIANTS:
   ee=ca[ca.candidate.map(lambda n:family(SPEC[n],v))].sort_values('validation_f1',ascending=False)
   vdiags.append(dict(backbone=b,model=m,variant=v,candidates=len(ee),best_validation_f1=ee.validation_f1.iloc[0],second_validation_f1=ee.validation_f1.iloc[1] if len(ee)>1 else np.nan,near_best_count=int((ee.validation_f1>=ee.validation_f1.max()-.001).sum())))
pd.DataFrame(vdiags).to_csv(O/'validation_search_diagnostics.csv',index=False)
freeze=O/'SELECTION_FROZEN.json'
if freeze.exists():assert read(freeze)['receipts']==receipts
else:write(freeze,dict(time=time.time(),receipts=receipts,test_evaluated=False))
ref=pd.read_csv(P/'analysis_inputs/reference_predictions.csv.gz');rows=[];predrows=[]
for b in BB:
 ga=cases(b);gt=ga[ga.split=='test'].reset_index(drop=True)
 for m in scores.files:
  g=gt[(scores[m][gt.source_index]>=0).all(axis=1)].reset_index(drop=True);tab=tabular(g);y=g.true_label.to_numpy();w=g.weight.to_numpy()
  for pol in POL:
   referred,base,nn=decisions(g,m,pol);cc=ch[(ch.backbone==b)&(ch.model==m)&(ch.policy==pol)];qs={};pred={'control':base,'always_nn':nn};shows={'control':np.zeros(len(g),bool),'always_nn':referred}
   rr=g[['source_index']].merge(ref[(ref.model==m)&(ref.policy==pol)],on='source_index',how='left',validate='one_to_one');assert not rr.isna().any().any()
   for v in ['cosine_gate','prior_pooled','prior_toxicity_pool']:pred[v]=rr[v].to_numpy(bool)
   for r in cc.itertuples():
    if r.candidate not in qs:
     mm=joblib.load(R/'models'/b/(m+'__'+pol+'__'+r.candidate+'.joblib'));s=mm['spec'];X=matrix(b,s,'full',g,tab)
     qs[r.candidate]=predict_log(mm['model'],X) if s['algorithm']=='logistic' else predict_knn(mm['model'],X,s['algorithm'].replace('knn_',''),[s['parameter']])[s['parameter']]
    cs=cu[(cu.backbone==b)&(cu.model==m)&(cu.policy==pol)&(cu.candidate==r.candidate)].set_index('dataset').cutoff
    show=(qs[r.candidate]>=g.dataset.map(cs).to_numpy())&referred;shows[r.variant]=show;pred[r.variant]=np.where(show,nn,base)
   pp=g[['source_index','backbone','dataset','cluster','true_label','weight']].copy();pp['model']=m;pp['policy']=pol
   for v,p in pred.items():pp[v]=p.astype(np.int8)
   predrows.append(pp)
   for ds,ii in g.groupby('dataset').indices.items():
    h=g.iloc[ii];po=pop(h,m)
    for v,p in pred.items():rows.append(dict(model=m,backbone=b,dataset=ds,policy=pol,method=v,test_n=len(h),population_n=po[0],population_positive=po[1],**metric(y[ii],p[ii],w[ii],po),nn_rate=float(w[ii]@shows[v][ii]/po[0]) if v in shows else np.nan))
  print('EVALUATED',b,m,flush=True)
a=pd.DataFrame(rows);p=pd.concat(predrows,ignore_index=True);a.to_csv(O/'cell_metrics.csv',index=False);p.to_csv(O/'test_predictions.csv.gz',index=False);ch.to_csv(O/'choices.csv',index=False);cu.merge(ch[['model','backbone','policy','candidate']].drop_duplicates(),on=['model','backbone','policy','candidate']).to_csv(O/'selected_cutoffs.csv',index=False);pd.DataFrame(training).to_csv(O/'training_counts.csv',index=False)
# Independently reproduce original comparison cells exactly.
old=pd.read_csv(I/'prior_cell_metrics.csv');old=old[old.dataset.isin(DS)&old.method.isin(['control','always_nn','cosine_gate'])];j=a.merge(old,on=['model','backbone','dataset','policy','method'],suffixes=('','_old'));assert len(j)==2400
err=max(float((j[k]-j[k+'_old']).abs().max()) for k in ['f1','precision','recall','accuracy']);assert err<1e-10,err
write(O/'EVALUATION_COMPLETED.json',dict(time=time.time(),cells=len(a),case_policy_rows=len(p),benchmark_replay_max_error=err,files={x.name:sha(x) for x in [O/'cell_metrics.csv',O/'test_predictions.csv.gz',O/'choices.csv',O/'selected_cutoffs.csv',O/'training_counts.csv']}))
print('EVALUATION COMPLETE',len(a),err,flush=True)
