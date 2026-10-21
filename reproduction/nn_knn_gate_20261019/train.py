import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,json,gzip,hashlib,time,joblib
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';S=P/'api_results_20261015';OLD=P/'nn_learned_gate_20261019';SEED=20261019
KS=[151,51,15,5];METRICS=['euclidean','cosine'];POL=['fixed50','judge50_direct','baseline_anchor','nearest_anchor','saliency_anchor'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122']
# Reuse exact audited functions without executing the previous training script.
source=OLD/'train.py';tree=ast.parse(source.read_text());wanted=['read','sha','features','pop','decisions','metric','threshold'];defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(defs)==len(wanted);exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),'exec'))
assert sha(source)==read(OLD/'results/COMPLETED.json')['script_sha256']
protocol=dict(extension='kNN selector using same feature whitelist, validation folds, targets, weights and evaluation policies as original logistic/tree study. Previous artifacts unchanged.',distance='Euclidean and cosine on StandardScaler-transformed 31-dimensional pre-judge feature vectors, not the original text embedding space. Scaler fitted with training-disagreement weights only, within each CV fold.',neighbors=KS,vote='Weighted mean include-label among k closest training disagreements, using same cell-normalized inverse-inclusion sample weights as logistic/tree. No additional inverse-distance weighting. k capped to available training count. Exact-distance ties follow the fixed sklearn neighbour implementation and frozen input order; labels do not break ties.',selection='Out-of-fold predictions for every validation case. Per-cell cutoff maximizes validation pipeline F1 with original endpoint/tie rules. Best k within each metric, best kNN overall, and expanded comparison including original logistic/tree candidates all selected using validation only. Ties prefer earlier candidate; old families precede kNN in expanded selection.',supervision='Only original validation control-vs-NN disagreements; target1 when NN correct,0 when control correct. No preference from unchanged predictions.',limits='Exploratory follow-up on previously inspected test set. No test-driven hyperparameter selection, no LLM calls. kNN retrieves validation examples of usefulness, a separate operation from retrieving the neighbour shown in the original prompt.')
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
src=[S/'results/scores.npz',S/'frozen/design.json.gz',P/'nn_mechanisms_20261015/results/joined_features.csv.gz',P/'anchor_transfer_20261018/results/thresholds.json']
rec=read(OLD/'results/COMPLETED.json')
for p in src:assert sha(p)==rec['inputs'][str(p)]
d=read(src[1]);scores=np.load(src[0]);f=pd.read_csv(src[2]);f['ix']=np.arange(len(f));f['cell']=f.backbone+'|'+f.dataset;f['fold']=f.cluster.map(lambda x:int(hashlib.sha256((str(SEED)+':'+x).encode()).hexdigest()[:12],16)%3);val=f[f.split=='val'].copy();test=f[f.split=='test'].copy();assert not set(val.cluster)&set(test.cluster)
oldfold=pd.read_csv(OLD/'results/validation_folds.csv.gz');assert np.array_equal(val.fold,oldfold.fold) and np.array_equal(val.ix,oldfold.ix)
ths=pd.DataFrame(read(src[3]));ths=ths[ths.scope!='common_valid'];ths={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in ths.itertuples()};BB=sorted(val.backbone.unique());DS=sorted(val.dataset.unique());Xv,fn=features(val);assert fn==read(OLD/'features.json');(R/'features.json').write_text(json.dumps(fn,indent=2))
oldc=pd.read_csv(OLD/'results/validation_candidates.csv');oldcuts=pd.read_csv(OLD/'results/validation_cutoffs.csv');cuts=[];cv=[];choices=[];runtimes=[];models={};start=time.time()

def fit_knn(X,y,w):
 if len(np.unique(y))<2:return dict(constant=float(y[0]) if len(y) else .5)
 sc=StandardScaler().fit(X,sample_weight=w);xx=sc.transform(X)
 return dict(scaler=sc,X=xx,y=y,weights=w,rows=len(y))
def predict_many(mm,X,metric_name,ks):
 if 'constant' in mm:return {k:np.full(len(X),mm['constant']) for k in ks}
 xx=mm['scaler'].transform(X);n=min(max(ks),len(mm['y']));nn=NearestNeighbors(n_neighbors=n,algorithm='brute',metric=metric_name,n_jobs=1).fit(mm['X']);out={k:np.empty(len(X)) for k in ks}
 for off in range(0,len(X),512):
  dd,ii=nn.kneighbors(xx[off:off+512]);ww=mm['weights'][ii];num=np.cumsum(ww*mm['y'][ii],axis=1);den=np.cumsum(ww,axis=1)
  for k in ks:out[k][off:off+512]=num[:,min(k,n)-1]/den[:,min(k,n)-1]
 return out
for m in scores.files:
 valid=(scores[m][val.ix]>=0).all(axis=1);g=val.loc[valid].reset_index(drop=True);X=Xv[valid];y=g.true_label.to_numpy();w=g.weight.to_numpy();norm=w/g.groupby('cell').weight.transform('sum').to_numpy();fold=g.fold.to_numpy();groups={c:np.array(ii) for c,ii in g.groupby('cell').indices.items()}
 for pol in POL:
  st=time.time();ref,base,nn=decisions(g,m,pol);flip=base!=nn;target=(nn==y).astype(int);oofs={(met,k):np.empty(len(g)) for met in METRICS for k in KS}
  for foldid in range(3):
   train=(fold!=foldid)&flip;hold=fold==foldid;assert not set(g.loc[train,'cluster'])&set(g.loc[hold,'cluster']);mm=fit_knn(X[train],target[train],norm[train])
   for met in METRICS:
    pred=predict_many(mm,X[hold],met,KS)
    for k in KS:oofs[met,k][hold]=pred[k]
  new=[]
  for met in METRICS:
   for k in KS:
    q=oofs[met,k];assert np.isfinite(q).all() and np.all((q>=0)&(q<=1));name='knn_'+met+'_k'+str(k);fs=[]
    for cell,ix in groups.items():
     gg=g.iloc[ix];cut,f1,nshow=threshold(y[ix],base[ix],nn[ix],w[ix],q[ix],pop(gg,m,pol),ref[ix]);assert abs(metric(y[ix],np.where(q[ix]>=cut,nn[ix],base[ix]),w[ix],pop(gg,m,pol))['f1']-f1)<1e-9
     cuts.append(dict(model=m,policy=pol,candidate=name,metric=met,k=k,backbone=gg.backbone.iloc[0],dataset=gg.dataset.iloc[0],cutoff=cut,validation_f1=f1));fs.append(f1)
    row=dict(model=m,policy=pol,candidate=name,metric=met,k=k,selection_validation_f1=float(np.mean(fs)),training_flips=int(flip.sum()));new.append(row);cv.append(row)
  old=oldc[(oldc.model==m)&(oldc.policy==pol)].to_dict('records')
  for variant,candidates in [('knn_euclidean',[r for r in new if r['metric']=='euclidean']),('knn_cosine',[r for r in new if r['metric']=='cosine']),('knn_selected',new),('expanded_selected',old+new)]:
   best=max(r['selection_validation_f1'] for r in candidates);win=next(r for r in candidates if r['selection_validation_f1']>=best-1e-12);choices.append(dict(model=m,policy=pol,variant=variant,candidate=win['candidate'],selection_validation_f1=best))
  mm=fit_knn(X[flip],target[flip],norm[flip]);models[m,pol]=mm;path=R/'models'/(m+'__'+pol+'.joblib');joblib.dump(dict(model=mm,features=fn),path);runtimes.append(dict(model=m,policy=pol,seconds=time.time()-st,flips=int(flip.sum())))
  print('trained',m,pol,round(time.time()-st,2),'s',choices[-2]['candidate'],flush=True)
for name,z in [('validation_cutoffs',cuts),('validation_candidates',cv),('model_choices',choices),('runtime',runtimes)]:pd.DataFrame(z).to_csv(O/(name+'.csv'),index=False)
freeze=dict(choices_sha256=sha(O/'model_choices.csv'),cutoffs_sha256=sha(O/'validation_cutoffs.csv'),elapsed_training_seconds=time.time()-start,models={p.name:sha(p) for p in (R/'models').glob('*.joblib')},test_evaluation_started=False);(O/'SELECTION_FROZEN.json').write_text(json.dumps(freeze,indent=2));print('FROZEN; TEST START',flush=True)
cm={(r['model'],r['policy'],r['candidate'],r['backbone'],r['dataset']):r['cutoff'] for r in cuts};cm.update({(r.model,r.policy,r.candidate,r.backbone,r.dataset):r.cutoff for r in oldcuts.itertuples()});choice={(r['model'],r['policy'],r['variant']):r['candidate'] for r in choices};Xt,_=features(test);rows=[];predrows=[]
for m in scores.files:
 valid=(scores[m][test.ix]>=0).all(axis=1);g=test.loc[valid].reset_index(drop=True);X=Xt[valid];y=g.true_label.to_numpy();w=g.weight.to_numpy();groups={c:np.array(ii) for c,ii in g.groupby('cell').indices.items()}
 for pol in POL:
  ref,base,nn=decisions(g,m,pol);names={choice[m,pol,v] for v in ['knn_euclidean','knn_cosine','knn_selected','expanded_selected']};pred={}
  for met in METRICS:
   ks=sorted({int(n.split('_k')[-1]) for n in names if n.startswith('knn_'+met+'_')})
   if ks:
    preds=predict_many(models[m,pol],X,met,ks)
    for k,q in preds.items():pred['knn_'+met+'_k'+str(k)]=q
  for name in names:
   if name.startswith('knn_'):continue
   mm=joblib.load(OLD/'models'/(m+'__'+pol+'__'+name+'.joblib'))['model']
   pred[name]=np.full(len(X),mm['constant']) if 'constant' in mm else mm['estimator'].predict_proba(mm['scaler'].transform(X) if mm['scaler'] is not None else X)[:,1]
  variants={}
  for variant in ['knn_euclidean','knn_cosine','knn_selected','expanded_selected']:
   name=choice[m,pol,variant];q=pred[name];cut=np.array([cm[m,pol,name,b,ds] for b,ds in zip(g.backbone,g.dataset)]);show=(q>=cut)&ref;pr=np.where(show,nn,base);variants[variant]=pr
   for cell,ix in groups.items():
    gg=g.iloc[ix];pp=pop(gg,m,pol);rows.append(dict(model=m,backbone=gg.backbone.iloc[0],dataset=gg.dataset.iloc[0],policy=pol,method=variant,**metric(y[ix],pr[ix],w[ix],pp),nn_rate=float(w[ix]@show[ix]/pp[0]),test_n=len(ix)))
  for i,row in enumerate(g.itertuples()):predrows.append(dict(model=m,policy=pol,source_index=row.ix,**{v:int(pr[i]) for v,pr in variants.items()}))
 print('tested',m,flush=True)
new=pd.DataFrame(rows);new.to_csv(O/'knn_cell_metrics.csv',index=False);pd.DataFrame(predrows).to_csv(O/'test_predictions.csv.gz',index=False);prior=pd.read_csv(OLD/'results/cell_metrics.csv');res=pd.concat([prior,new],ignore_index=True);res.to_csv(O/'all_cell_metrics.csv',index=False)
summary=[];delta=[]
for cohort in ['all','toxicity4']:
 rr=res if cohort=='all' else res[res.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])]
 for dim in ['overall','model','backbone','dataset']:
  gr=[('ALL',rr)] if dim=='overall' else rr.groupby(dim)
  for name,g in gr:
   for pol in POL+['anchor_average']:
    h=g[g.policy.isin(['baseline_anchor','nearest_anchor','saliency_anchor'])] if pol=='anchor_average' else g[g.policy==pol]
    for method,z in h.groupby('method'):summary.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,method=method,**{k:z[k].mean() for k in ['f1','precision','recall','accuracy','nn_rate']}))
    for method in ['knn_euclidean','knn_cosine','knn_selected','expanded_selected']:
     ga=h[h.method==method].set_index(['model','backbone','dataset','policy'])
     for refname in ['always_nn','cosine_gate','logistic_gate','selected_gate']:
      ba=h[h.method==refname].set_index(['model','backbone','dataset','policy']);dd=ga[['f1','precision','recall','accuracy']]-ba[['f1','precision','recall','accuracy']]
      if pol=='anchor_average':dd=dd.groupby(level=[0,1,2]).mean()
      delta.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,method=method,reference=refname,wins=int((dd.f1>1e-10).sum()),losses=int((dd.f1< -1e-10).sum()),ties=int((abs(dd.f1)<=1e-10).sum()),**{k:dd[k].mean() for k in dd}))
pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(delta).to_csv(O/'deltas.csv',index=False)
assert sha(O/'model_choices.csv')==freeze['choices_sha256'];assert sha(O/'validation_cutoffs.csv')==freeze['cutoffs_sha256'];assert len(new)==10000
(O/'COMPLETED.json').write_text(json.dumps(dict(new_model_calls=0,contexts_validation=len(val),contexts_test=len(test),target_overlap=0,new_cell_metrics=len(new),elapsed_seconds=time.time()-start,inputs={str(p):sha(p) for p in src+[source,OLD/'results/validation_candidates.csv',OLD/'results/validation_cutoffs.csv',OLD/'results/cell_metrics.csv']},script_sha256=sha(Path(__file__))),indent=2))
print(pd.DataFrame(summary).query("cohort=='all' and dimension=='overall' and policy in ['fixed50','anchor_average']").to_string(index=False))
