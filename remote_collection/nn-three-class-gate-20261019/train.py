from common import *
import sys,collections
b=sys.argv[1];reserve();gall=cases(b);gval=gall[gall.split=='val'].reset_index(drop=True);taball=tabular(gval);out=O/b;out.mkdir(parents=True,exist_ok=True);modeldir=R/'models'/b;modeldir.mkdir(parents=True,exist_ok=True)
assert (P/'representations'/b/'COMPLETED.json').exists()
def train_one(m,pol,specs,variants):
 folder=out/m/pol;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETED.json').exists():
  rec=read(folder/'COMPLETED.json')
  for name,digest in rec['files'].items():assert sha(folder/name)==digest
  for name,digest in rec['models'].items():assert sha(Path(name))==digest
  return pd.read_csv(folder/'choices.csv').to_dict('records')
 st=time.time();valid=(scores[m][gval.source_index]>=0).all(axis=1);g=gval.loc[valid].reset_index(drop=True);tab=taball[valid];y=g.true_label.to_numpy();w=g.weight.to_numpy();norm=w/g.groupby('dataset').weight.transform('sum').to_numpy();folds=g.fold.to_numpy();ref,base,nn=decisions(g,m,pol);flip=base!=nn;target=np.where(flip,np.where(nn==y,1,-1),0)
 geometries=collections.defaultdict(list)
 for spec in specs:geometries[geometry_name(spec)].append(spec)
 oofs={spec_name(s):np.empty(len(g)) for s in specs};fitrecords=[]
 for geom,ss in geometries.items():
  for fold in range(3):
   tr=(folds!=fold);ho=folds==fold;assert not set(g.loc[tr,'cluster'])&set(g.loc[ho,'cluster']);X=matrix(b,ss[0],fold,g,tab);scaler=StandardScaler().fit(X[tr],sample_weight=norm[tr]) if len(np.unique(target[tr]))>=2 else None
   for spec in [s for s in ss if s['algorithm']=='logistic']:
    mm=fit_log(spec,X[tr],target[tr],norm[tr],scaler);oofs[spec_name(spec)][ho]=predict_log(mm,X[ho]);fitrecords.append(dict(candidate=spec_name(spec),fold=fold,training_rows=int(tr.sum()),training_disagreements=int((tr&flip).sum()),constant='constant' in mm,iterations=mm.get('iterations',0),max_iter=mm.get('max_iter',0),solver=mm.get('solver','constant')))
   for met in ['euclidean','cosine']:
    kk=[s['parameter'] for s in ss if s['algorithm']=='knn_'+met]
    if kk:
     mm=fit_knn(X[tr],target[tr],norm[tr],scaler);pr=predict_knn(mm,X[ho],met,kk)
     for spec in [s for s in ss if s['algorithm']=='knn_'+met]:oofs[spec_name(spec)][ho]=pr[spec['parameter']]
  if len(geometries)>5:print('geometry',b,m,pol,geom,round(time.time()-st,1),flush=True)
 labels=g[['source_index','dataset','fold','true_label']].copy();labels['utility_class']=target;labels['weight']=w;labels.to_csv(folder/'training_labels.csv',index=False)
 cuts=[];cand=[];groups=g.groupby('dataset').indices
 for spec in specs:
  name=spec_name(spec);q=oofs[name];assert np.isfinite(q).all();fs=[]
  for ds,ii in groups.items():
   h=g.iloc[ii];cut,f1,nshow=threshold(y[ii],base[ii],nn[ii],w[ii],q[ii],pop(h,m),ref[ii]);cuts.append(dict(model=m,backbone=b,policy=pol,candidate=name,dataset=ds,cutoff=cut,validation_f1=f1));fs.append(f1)
  cand.append(dict(model=m,backbone=b,policy=pol,candidate=name,validation_f1=float(np.mean(fs)),**spec))
 choices=[];models={}
 for variant in variants:
  eligible=[r for r in cand if family(SPEC[r['candidate']],variant)];best=max(r['validation_f1'] for r in eligible);winner=next(r for r in eligible if r['validation_f1']>=best-1e-12);choices.append(dict(model=m,backbone=b,policy=pol,variant=variant,candidate=winner['candidate'],validation_f1=winner['validation_f1']))
 for name in sorted({r['candidate'] for r in choices}):
  spec=SPEC[name];X=matrix(b,spec,'full',g,tab)
  mm=fit_log(spec,X,target,norm) if spec['algorithm']=='logistic' else fit_knn(X,target,norm);path=modeldir/(m+'__'+pol+'__'+name+'.joblib');joblib.dump(dict(spec=spec,model=mm),path,compress=3);models[str(path)]=sha(path)
 for name,rows in [('candidates',cand),('cutoffs',cuts),('choices',choices),('fit_diagnostics',fitrecords)]:pd.DataFrame(rows).to_csv(folder/(name+'.csv'),index=False)
 np.savez_compressed(folder/'selected_oof.npz',source_index=g.source_index.to_numpy(),**{name:oofs[name] for name in {r['candidate'] for r in choices}})
 write(folder/'COMPLETED.json',dict(seconds=time.time()-st,validation_rows=len(g),training_disagreements=int(flip.sum()),helpful=int((target[flip]==1).sum()),harmful=int((target==-1).sum()),unchanged=int((target==0).sum()),training_rows=len(g),candidate_count=len(specs),files={p.name:sha(p) for p in folder.iterdir() if p.suffix in ['.csv','.npz']},models=models,test_evaluated=False))
 print('TRAINED',b,m,pol,'seconds',round(time.time()-st,1),'flips',flip.sum(),'selected',choices[7]['candidate'] if len(choices)>7 else '',flush=True);return choices
for m in (scores.files if len(sys.argv)<3 else [sys.argv[2]]):
 for pol in POL:
  oldchoices=pd.read_csv(P/'results'/b/m/pol/'choices.csv')
  # Exactly the prior validation-selected learner/representation families;
  # retain all C or k values for each architecture, and reselect on validation.
  keep=oldchoices[oldchoices.variant.isin(['embedding_logistic','combined_logistic','embedding_knn','combined_knn','tabular'])]
  architectures={(geometry_name(SPEC[n]),SPEC[n]['algorithm']) for n in keep.candidate}
  specs=[s for s in SPECS if (geometry_name(s),s['algorithm']) in architectures]
  train_one(m,pol,specs,SECOND_VARIANTS)
if len(sys.argv)<3:
 write(out/'TRAINING_COMPLETED.json',dict(backbone=b,judge_count=len(scores.files),policy_count=4,test_evaluated=False));print('BACKBONE TRAINING COMPLETE',b,flush=True)
