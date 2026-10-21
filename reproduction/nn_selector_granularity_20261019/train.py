import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
 os.environ[key]='1'
from pathlib import Path
import ast,json,gzip,hashlib,time,joblib,warnings,csv
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';OLD=P/'nn_learned_gate_20261019';S=P/'api_results_20261015';SEED=20261019
POL=['fixed50','baseline_anchor','nearest_anchor','saliency_anchor'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];TOX=['aegis','beavertails','toxicchat','toxigen']
KS=[151,51,15,5,3,1];METRICS=['euclidean','cosine'];SPECS=[('logistic_C0.1','logistic',.1),('logistic_C1','logistic',1.)]
source=OLD/'train.py';wanted=['read','sha','features','pop','decisions','metric','threshold','fit','predict'];tree=ast.parse(source.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted];assert len(defs)==len(wanted);exec(compile(ast.Module(body=defs,type_ignores=[]),str(source),'exec'))
assert sha(source)==read(OLD/'results/COMPLETED.json')['script_sha256']
ksource=P/'nn_knn_gate_20261019/train.py';tree=ast.parse(ksource.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fit_knn','predict_many']];exec(compile(ast.Module(body=defs,type_ignores=[]),str(ksource),'exec'));assert sha(ksource)==read(P/'nn_knn_gate_20261019/results/COMPLETED.json')['script_sha256']
protocol=dict(groups={'pool_all':'One per judge, pooling all ten datasets and five backbones','pool_toxicity':'One per judge, pooling four toxicity datasets and five backbones','dataset':'One per judge and dataset, pooling five backbones','cell':'One per judge, dataset and backbone'},policies=POL,features='Exact original 31-feature whitelist; no target gold, judge scores, or test mechanism annotations as inputs.',supervision='Validation control-vs-NN disagreements only; include=1 iff NN correct. Same target-cluster grouped three folds and equal-cell-mass inverse inclusion weights as original. Full validation support used for cutoff selection.',logistic_C=[.1,1],knn_k=KS,knn_distance=METRICS,knn_note='Same standardized feature-space distances and sample-weight voting as earlier kNN. Add k1 and k3 for sparse groups, equally for pooled and separate selectors. k is capped to available training rows.',selection='For each candidate, OOF gate scores select one inclusion cutoff per cell by original weighted F1 and tie rules. Candidate selected by macro validation F1 across cells in its training group. Logistic, each kNN metric, and kNN metric+k are selected separately. Refit on all group validation disagreements. Freeze all selections and models before any test evaluation.',sparse='No groups excluded. Empty training fit gives constant 0.5; one-class fit gives that class constant. Applies separately within CV and final fit; record counts. No test-guided fallback.',evaluation='Same original probe/judge thresholds across methods, same paired-valid cases and design weights. Macro across cells. Anchor average is mean of three paired evaluations, not an averaged threshold. Original pooled logistic replay required.',limits='Exploratory follow-up on previously examined test data; no new model calls. Group granularity comparison does not create new independent validation examples.')
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
src=[S/'results/scores.npz',S/'frozen/design.json.gz',P/'nn_mechanisms_20261015/results/joined_features.csv.gz',P/'anchor_transfer_20261018/results/thresholds.json'];receipt=read(OLD/'results/COMPLETED.json')
for path in src:assert sha(path)==receipt['inputs'][str(path)]
d=read(src[1]);scores=np.load(src[0]);f=pd.read_csv(src[2]);f['ix']=np.arange(len(f));f['cell']=f.backbone+'|'+f.dataset;f['fold']=f.cluster.map(lambda x:int(hashlib.sha256((str(SEED)+':'+x).encode()).hexdigest()[:12],16)%3)
val=f[f.split=='val'].copy();test=f[f.split=='test'].copy();assert not set(val.cluster)&set(test.cluster)
oldfold=pd.read_csv(OLD/'results/validation_folds.csv.gz');assert np.array_equal(val.fold,oldfold.fold) and np.array_equal(val.ix,oldfold.ix)
ths=pd.DataFrame(read(src[3]));ths=ths[ths.scope!='common_valid'];ths={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in ths.itertuples()};BB=sorted(val.backbone.unique());DS=sorted(val.dataset.unique());Xv,fn=features(val);assert fn==read(OLD/'features.json');(R/'features.json').write_text(json.dumps(fn,indent=2))
def subsets(g):
 yield 'pool_all','ALL',np.arange(len(g))
 yield 'pool_toxicity','TOX',np.flatnonzero(g.dataset.isin(TOX).to_numpy())
 for ds,ix in g.groupby('dataset').indices.items():yield 'dataset',ds,np.array(ix)
 for cell,ix in g.groupby('cell').indices.items():yield 'cell',cell.replace('|','__'),np.array(ix)
cuts=[];candidates=[];choices=[];counts=[];start=time.time();modelpaths={}
for m in scores.files:
 valid=(scores[m][val.ix]>=0).all(axis=1);gg=val.loc[valid].reset_index(drop=True);XX=Xv[valid]
 for pol in POL:
  st=time.time()
  for scope,group,ii in subsets(gg):
   g=gg.iloc[ii].reset_index(drop=True);X=XX[ii];y=g.true_label.to_numpy();w=g.weight.to_numpy();norm=w/g.groupby('cell').weight.transform('sum').to_numpy();fold=g.fold.to_numpy();ref,base,nn=decisions(g,m,pol);flip=base!=nn;target=(nn==y).astype(int)
   cells={c:np.array(ix) for c,ix in g.groupby('cell').indices.items()};meta=dict(model=m,policy=pol,scope=scope,group=group)
   oo={s[0]:np.empty(len(g)) for s in SPECS};oo.update({'knn_'+met+'_k'+str(k):np.empty(len(g)) for met in METRICS for k in KS});empty=oneclass=0
   for fid in range(3):
    tr=(fold!=fid)&flip;ho=fold==fid;assert not set(g.loc[tr,'cluster'])&set(g.loc[ho,'cluster']);empty+=int(not tr.any());oneclass+=int(tr.any() and len(np.unique(target[tr]))==1)
    if not ho.any():continue
    for spec in SPECS:oo[spec[0]][ho]=predict(fit(spec,X[tr],target[tr],norm[tr]),X[ho])
    mm=fit_knn(X[tr],target[tr],norm[tr])
    for met in METRICS:
     for k,q in predict_many(mm,X[ho],met,KS).items():oo['knn_'+met+'_k'+str(k)][ho]=q
   local=[]
   for name,q in oo.items():
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all();fs=[]
    for cell,ix in cells.items():
     h=g.iloc[ix];cut,f1,nshow=threshold(y[ix],base[ix],nn[ix],w[ix],q[ix],pop(h,m,pol),ref[ix]);assert abs(metric(y[ix],np.where(q[ix]>=cut,nn[ix],base[ix]),w[ix],pop(h,m,pol))['f1']-f1)<1e-9
     cuts.append(dict(**meta,candidate=name,backbone=h.backbone.iloc[0],dataset=h.dataset.iloc[0],cutoff=cut,validation_f1=f1));fs.append(f1)
    row=dict(**meta,candidate=name,validation_f1=float(np.mean(fs)));local.append(row);candidates.append(row)
   selected={}
   for variant,prefix in [('logistic','logistic_'),('knn_euclidean','knn_euclidean_'),('knn_cosine','knn_cosine_'),('knn','knn_')]:
    eligible=[r for r in local if r['candidate'].startswith(prefix)];best=max(r['validation_f1'] for r in eligible);win=next(r for r in eligible if r['validation_f1']>=best-1e-12);selected[variant]=win['candidate'];choices.append(dict(**meta,variant=variant,candidate=win['candidate'],validation_f1=win['validation_f1']))
   ls=next(s for s in SPECS if s[0]==selected['logistic']);fitted=dict(logistic=fit(ls,X[flip],target[flip],norm[flip]),knn=fit_knn(X[flip],target[flip],norm[flip]),features=fn,selected=selected)
   path=R/'models'/(m+'__'+pol+'__'+scope+'__'+group+'.joblib');joblib.dump(fitted,path,compress=3);modelpaths[m,pol,scope,group]=path
   counts.append(dict(**meta,validation_rows=len(g),training_flips=int(flip.sum()),include=int(target[flip].sum()),exclude=int(flip.sum()-target[flip].sum()),distinct_training_targets=g.loc[flip,'cluster'].nunique(),constant_final='constant' in fitted['logistic'],empty_final=not flip.any(),empty_cv_fits=empty,oneclass_cv_fits=oneclass))
  print('trained',m,pol,round(time.time()-st,1),'seconds',flush=True)
for name,rows in [('validation_cutoffs',cuts),('validation_candidates',candidates),('model_choices',choices),('training_counts',counts)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
freeze=dict(test_evaluation_started=False,elapsed_seconds=time.time()-start,files={str(p):sha(p) for p in [R/'PROTOCOL.json',O/'model_choices.csv',O/'validation_cutoffs.csv']},models={str(p):sha(p) for p in modelpaths.values()});(O/'SELECTION_FROZEN.json').write_text(json.dumps(freeze,indent=2));print('ALL SELECTIONS FROZEN; TEST START',flush=True)
cm={(r['model'],r['policy'],r['scope'],r['group'],r['candidate'],r['backbone'],r['dataset']):r['cutoff'] for r in cuts};Xt,_=features(test);rows=[]
with gzip.open(O/'test_predictions.csv.gz','wt') as handle:
 writer=csv.writer(handle);writer.writerow(['model','policy','scope','source_index','logistic','knn_euclidean','knn_cosine','knn'])
 for m in scores.files:
  valid=(scores[m][test.ix]>=0).all(axis=1);gg=test.loc[valid].reset_index(drop=True);XX=Xt[valid]
  for pol in POL:
   for scope,group,ii in subsets(gg):
    g=gg.iloc[ii].reset_index(drop=True);X=XX[ii];y=g.true_label.to_numpy();w=g.weight.to_numpy();ref,base,nn=decisions(g,m,pol);bank=joblib.load(modelpaths[m,pol,scope,group]);names=bank['selected'];qs={names['logistic']:predict(bank['logistic'],X)}
    for met in METRICS:
     kk=sorted({int(n.split('_k')[-1]) for n in names.values() if n.startswith('knn_'+met+'_')})
     if kk:
      for k,q in predict_many(bank['knn'],X,met,kk).items():qs['knn_'+met+'_k'+str(k)]=q
    preds={}
    for variant,name in names.items():
     cut=np.array([cm[m,pol,scope,group,name,b,ds] for b,ds in zip(g.backbone,g.dataset)]);show=(qs[name]>=cut)&ref;pr=np.where(show,nn,base);preds[variant]=pr
     for cell,ix in g.groupby('cell').indices.items():
      h=g.iloc[ix];pp=pop(h,m,pol);rows.append(dict(model=m,backbone=h.backbone.iloc[0],dataset=h.dataset.iloc[0],policy=pol,method=scope+'__'+variant,**metric(y[ix],pr[ix],w[ix],pp),nn_rate=float(w[ix]@show[ix]/pp[0]),test_n=len(ix)))
    writer.writerows((m,pol,scope,int(ix),*(int(preds[v][i]) for v in ['logistic','knn_euclidean','knn_cosine','knn'])) for i,ix in enumerate(g.ix))
  print('tested',m,flush=True)
new=pd.DataFrame(rows);new.to_csv(O/'new_cell_metrics.csv',index=False)
prior=pd.read_csv(OLD/'results/cell_metrics.csv');prior=prior[prior.policy.isin(POL)];check=new[new.method=='pool_all__logistic'].merge(prior[prior.method=='logistic_gate'],on=['model','backbone','dataset','policy'],suffixes=('_new','_old'),validate='one_to_one');assert len(check)==2000;error=max(abs(check[k+'_new']-check[k+'_old']).max() for k in ['f1','precision','recall','accuracy','nn_rate']);assert error<1e-10,error
bench=prior[prior.method.isin(['control','always_nn','cosine_gate'])];result=pd.concat([bench,new],ignore_index=True);assert not result.duplicated(['model','backbone','dataset','policy','method']).any();result.to_csv(O/'cell_metrics.csv',index=False)
summary=[];deltas=[]
for cohort in ['all','toxicity4']:
 rr=result[result.dataset.isin(TOX)] if cohort=='toxicity4' else result[~result.method.str.startswith('pool_toxicity')]
 for dim in ['overall','model','dataset','backbone']:
  for group,g in ([('ALL',rr)] if dim=='overall' else rr.groupby(dim)):
   for pol in POL+['anchor_average']:
    h=g[g.policy.isin(POL[1:])] if pol=='anchor_average' else g[g.policy==pol]
    for method,z in h.groupby('method'):
     summary.append(dict(cohort=cohort,dimension=dim,group=group,policy=pol,method=method,**{k:z[k].mean() for k in ['f1','precision','recall','accuracy','nn_rate']}))
     if '__' not in method:continue
     for reference in ['always_nn','cosine_gate','pool_all__logistic','pool_all__knn']:
      a=z.set_index(['model','backbone','dataset','policy']);b=h[h.method==reference].set_index(['model','backbone','dataset','policy']);dd=a[['f1','precision','recall','accuracy']]-b[['f1','precision','recall','accuracy']];assert dd.notna().all().all()
      if pol=='anchor_average':dd=dd.groupby(level=[0,1,2]).mean()
      deltas.append(dict(cohort=cohort,dimension=dim,group=group,policy=pol,method=method,reference=reference,wins=int((dd.f1>1e-10).sum()),losses=int((dd.f1< -1e-10).sum()),ties=int((abs(dd.f1)<=1e-10).sum()),**{k:dd[k].mean() for k in dd}))
pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(deltas).to_csv(O/'deltas.csv',index=False)
for p,digest in freeze['files'].items():assert sha(Path(p))==digest
for p,digest in freeze['models'].items():assert sha(Path(p))==digest
assert len(new)==27200
(O/'COMPLETED.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in src+[source,ksource]},script_sha256=sha(Path(__file__)),elapsed_seconds=time.time()-start,model_bundles=len(modelpaths),new_cell_metrics=len(new),pooled_logistic_replay_rows=len(check),pooled_logistic_max_error=error,validation_contexts=len(val),test_contexts=len(test),target_overlap=0,new_model_calls=0,outputs={p.name:sha(p) for p in O.glob('*.csv*')}),indent=2))
print(pd.DataFrame(summary).query("dimension=='overall' and policy in ['fixed50','anchor_average'] and (method.str.endswith('__logistic') or method.str.endswith('__knn') or method in ['always_nn','cosine_gate'])",engine='python').to_string(index=False))
