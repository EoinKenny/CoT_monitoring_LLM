import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json,gzip,hashlib,time,warnings,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';S=P/'api_results_20261015';SEED=20261019
MODELSPEC=[('logistic_C0.1','logistic',.1),('logistic_C1','logistic',1.),('tree_depth3','tree',3),('tree_depth5','tree',5)]
POL=['fixed50','judge50_direct','baseline_anchor','nearest_anchor','saliency_anchor'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122']
protocol=dict(primary='Fixed probe/judge50 pipeline; original paired-valid support, design weights and macro averaging. Supplement direct judge50 and three existing shared-anchor threshold policies.',learner='One pooled gate per outcome judge and policy. Regularized logistic C0.1 or1; decision tree depth3 or5 with min_samples_leaf50. No kNN: low-dimensional tabular pilot, mixed feature scales and sparsely observed disagreements.',supervision='Use validation cases where original control and NN binary decisions disagree. Target1 if NN correct,0 if control correct. Equal original cell mass with inverse-inclusion weights. Unchanged outcomes indifferent and excluded from fitting, included in F1 selection.',features='Explicit whitelist of pre-judge retrieval/probe statistics and dataset/backbone identity. No target gold, original judge outputs, mechanism annotations, target IDs or test-distribution ranks as inputs. See features.json.',selection='Three target-text-grouped folds, seeded deterministic hash, shared across all judges. Out-of-fold gate predictions for all validation cases. For each candidate and cell, choose gate threshold maximizing weighted validation pipeline F1, including always/never; ties fewer NN then higher cutoff. Choose candidate per judge/policy by mean cell validation F1, exact ties earlier simpler candidate. Also freeze best model within each family for comparison. Scaler fitted within training fold only. Refit selected models on all eligible validation disagreements. Persist frozen choices before test evaluation.',benchmarks='Original alwaysNN, control and original per-cell validation-selected cosine gates, identical probe/judge cutoffs and test support. Semantic-label gate not a model feature.',limits='Exploratory analysis on previously inspected test data. No inference/model calls or probe retraining. Gate uses known neighbour training label, not unknown target label. No claim of unbiased CV score after model/cutoff selection; held-out test reports actual performance.')
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src=[S/'results/scores.npz',S/'frozen/design.json.gz',P/'nn_mechanisms_20261015/results/joined_features.csv.gz',P/'anchor_transfer_20261018/results/thresholds.json',P/'nn_similarity_gate_20261019/results/selected_thresholds.csv']
d=read(src[1]);scores=np.load(src[0]);f=pd.read_csv(src[2]);case=pd.DataFrame([dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr]);assert list(zip(f.backbone,f.sample_id))==list(zip(case.backbone,case.sample_id));assert np.array_equal(f.true_label,case.true_label)
receipt=read(P/'nn_similarity_gate_20261019/results/COMPLETED.json')
for path in src[:4]:assert sha(path)==receipt['inputs'][str(path)],path
f['ix']=np.arange(len(f));f['cell']=f.backbone+'|'+f.dataset;f['fold']=f.cluster.map(lambda x:int(hashlib.sha256((str(SEED)+':'+x).encode()).hexdigest()[:12],16)%3)
val=f[f.split=='val'].copy();test=f[f.split=='test'].copy();assert not set(val.cluster)&set(test.cluster)
val[['ix','cluster','fold']].to_csv(O/'validation_folds.csv.gz',index=False)
ths=pd.DataFrame(read(src[3]));ths=ths[ths.scope!='common_valid'];ths={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in ths.itertuples()}
cos=pd.read_csv(src[4]);cos={(r.model,r.backbone,r.dataset,r.policy):r.cosine_threshold for r in cos.itertuples()}
# Feature encoders define only categories already observed in validation.
BB=sorted(val.backbone.unique());DS=sorted(val.dataset.unique())
def features(g):
 pp=g.probe_probability.to_numpy();nn=g.nearest_label.to_numpy();npred=g.nearest_prediction.to_numpy();sim=g.nearest_cosine.to_numpy()
 dd=dict(probe_probability=pp,probe_confidence=np.abs(pp-.5),nearest_cosine=sim,neighbor_training_label=nn,neighbor_probe_prediction=npred,neighbor_label_prediction_agree=(nn==npred).astype(float),probe_class_neighbor_label_agree=((pp>=.5)==nn).astype(float),probe_probability_x_neighbor_label=pp*nn,cosine_x_neighbor_label=sim*nn,cosine_x_probe_probability=sim*pp,log_target_chars=np.log1p(g.target_chars.to_numpy()),log_neighbor_chars=np.log1p(g.neighbor_chars.to_numpy()),log_length_ratio=np.log((1+g.neighbor_chars.to_numpy())/(1+g.target_chars.to_numpy())),lexical_jaccard=g.lexical_jaccard.to_numpy(),lexical_target_coverage=g.lexical_target_coverage.to_numpy(),exact_text_match=g.exact_text_match.astype(float).to_numpy())
 for x in BB:dd['backbone_'+x]=g.backbone.eq(x).astype(float).to_numpy()
 for x in DS:dd['dataset_'+x]=g.dataset.eq(x).astype(float).to_numpy()
 X=np.column_stack(list(dd.values())).astype(float);assert np.isfinite(X).all();return X,list(dd)
Xv,fn=features(val);(R/'features.json').write_text(json.dumps(fn,indent=2));assert not any(x in fn for x in ['true_label','control_score','nearest_score','label_aligned','target_content_hash','cluster'])

def pop(g,m,pol):
 b,ds=g.backbone.iloc[0],g.dataset.iloc[0]
 if m in LOCAL and pol!='judge50_direct':a=d['thresholds'][b][ds]['populations'][g.split.iloc[0]];return a['n'],a['positive']
 w=g.weight.to_numpy();return w.sum(),w@g.true_label.to_numpy()
def decisions(g,m,pol):
 pairs=np.array([(.5,50) if pol=='fixed50' else (0,50) if pol=='judge50_direct' else ths[m,b,ds,pol.replace('_anchor','')] for b,ds in zip(g.backbone,g.dataset)])
 ref=g.probe_probability.to_numpy()>=pairs[:,0];ss=scores[m][g.ix.to_numpy()];return ref,ref&(ss[:,0]>=pairs[:,1]),ref&(ss[:,2]>=pairs[:,1])
def metric(y,p,w,pop):
 n,pos=pop;tp=w@(p&(y==1));fp=w@(p&(y==0));den=pos+tp+fp
 return dict(f1=2*tp/den if den else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else 0.,accuracy=(tp+n-pos-fp)/n,tp=tp,fp=fp)
def threshold(y,base,nn,w,q,pop,ref):
 order=np.argsort(-q,kind='stable');qs=q[order];delta=nn.astype(float)-base.astype(float);mb=metric(y,base,w,pop)
 tp=mb['tp']+np.cumsum((w*delta*(y==1))[order]);fp=mb['fp']+np.cumsum((w*delta*(y==0))[order]);ends=np.r_[np.where(qs[:-1]!=qs[1:])[0],len(qs)-1]
 cuts=np.r_[1.0000001,qs[ends],-1.];tps=np.r_[mb['tp'],tp[ends],tp[-1]];fps=np.r_[mb['fp'],fp[ends],fp[-1]];den=pop[1]+tps+fps;fs=np.divide(2*tps,den,out=np.zeros(len(den)),where=den>0);nshow=np.r_[0,np.cumsum(ref[order])[ends],ref.sum()]
 best=np.flatnonzero(fs>=fs.max()-1e-12);k=sorted(best,key=lambda i:(nshow[i],-cuts[i]))[0]
 return float(cuts[k]),float(fs[k]),int(nshow[k])
def fit(spec,X,y,w):
 name,fam,param=spec
 if len(np.unique(y))<2:return dict(constant=float(y[0]) if len(y) else .5)
 w=w/w.mean()
 if fam=='logistic':
  scaler=StandardScaler().fit(X,sample_weight=w);xx=scaler.transform(X);est=LogisticRegression(C=param,solver='lbfgs',max_iter=1000,tol=1e-7,random_state=SEED)
 else:scaler=None;xx=X;est=DecisionTreeClassifier(max_depth=int(param),min_samples_leaf=50,random_state=SEED)
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);est.fit(xx,y,sample_weight=w)
 return dict(scaler=scaler,estimator=est)
def predict(model,X):
 if 'constant' in model:return np.full(len(X),model['constant'])
 return model['estimator'].predict_proba(model['scaler'].transform(X) if model['scaler'] is not None else X)[:,1]

allcuts=[];cvsummary=[];choices=[];training=[];fitted={};start=time.time()
# Do not evaluate test scores or select using test outcomes in this stage.
for m in scores.files:
 valid=(scores[m][val.ix]>=0).all(axis=1);g=val.loc[valid].reset_index(drop=True);X=Xv[valid];y=g.true_label.to_numpy();w=g.weight.to_numpy();norm=w/g.groupby('cell').weight.transform('sum').to_numpy();fold=g.fold.to_numpy();groups={c:np.array(ii) for c,ii in g.groupby('cell').indices.items()}
 for pol in POL:
  ref,base,nn=decisions(g,m,pol);flip=base!=nn;target=(nn==y).astype(int);perfs=[];stored={};startpol=time.time()
  for spec in MODELSPEC:
   name,family,param=spec;oof=np.empty(len(g))
   for k in range(3):
    train=(fold!=k)&flip;hold=fold==k
    assert not set(g.loc[train,'cluster'])&set(g.loc[hold,'cluster'])
    mm=fit(spec,X[train],target[train],norm[train]);oof[hold]=predict(mm,X[hold])
   assert np.isfinite(oof).all();cs=[]
   for cell,ix in groups.items():
    gg=g.iloc[ix];b,ds=gg.backbone.iloc[0],gg.dataset.iloc[0];cut,f1,nshow=threshold(y[ix],base[ix],nn[ix],w[ix],oof[ix],pop(gg,m,pol),ref[ix]);pred=np.where(oof[ix]>=cut,nn[ix],base[ix]);assert abs(metric(y[ix],pred,w[ix],pop(gg,m,pol))['f1']-f1)<1e-9
    row=dict(model=m,policy=pol,candidate=name,family=family,backbone=b,dataset=ds,cutoff=cut,validation_f1=f1,validation_nn_shown=nshow);cs.append(row);allcuts.append(row)
   perf=float(np.mean([r['validation_f1'] for r in cs]));perfs.append((perf,spec));stored[name]=cs;cvsummary.append(dict(model=m,policy=pol,candidate=name,family=family,selection_validation_f1=perf,training_flips=int(flip.sum()),training_helpful=int((flip&(target==1)).sum()),validation_rows=len(g)))
  selected={}
  for variant,family in [('selected',None),('logistic','logistic'),('tree','tree')]:
   eligible=[x for x in perfs if family is None or x[1][1]==family];best=max(x[0] for x in eligible);winner=next(x for x in eligible if x[0]>=best-1e-12)[1];selected[variant]=winner
   choices.append(dict(model=m,policy=pol,variant=variant,candidate=winner[0],selection_validation_f1=best,training_flips=int(flip.sum())))
  for name in {v[0] for v in selected.values()}:
   spec=next(x for x in MODELSPEC if x[0]==name);mm=fit(spec,X[flip],target[flip],norm[flip]);path=R/'models'/(m+'__'+pol+'__'+name+'.joblib');joblib.dump(dict(model=mm,features=fn,spec=spec),path);fitted[m,pol,name]=mm
  training.append(dict(model=m,policy=pol,seconds=time.time()-startpol,flips=int(flip.sum()),winner=selected['selected'][0]))
  print('trained',m,pol,'flips',flip.sum(),'winner',selected['selected'][0],round(time.time()-startpol,2),'s',flush=True)
# Durable choices before any test evaluation.
for name,data in [('validation_cutoffs',allcuts),('validation_candidates',cvsummary),('model_choices',choices),('training_runtime',training)]:pd.DataFrame(data).to_csv(O/(name+'.csv'),index=False)
freeze=dict(time=time.time(),elapsed_training_seconds=time.time()-start,choices_sha256=sha(O/'model_choices.csv'),cutoffs_sha256=sha(O/'validation_cutoffs.csv'),protocol_sha256=sha(R/'PROTOCOL.json'),features_sha256=sha(R/'features.json'),models={p.name:sha(p) for p in (R/'models').glob('*.joblib')},test_evaluation_started=False)
(O/'SELECTION_FROZEN.json').write_text(json.dumps(freeze,indent=2));print('SELECTION FROZEN; BEGIN TEST',flush=True)

Xt,_=features(test);cutmap={(r['model'],r['policy'],r['candidate'],r['backbone'],r['dataset']):r['cutoff'] for r in allcuts};choice={(r['model'],r['policy'],r['variant']):r['candidate'] for r in choices};rows=[];predrows=[];importance=[]
for (m,pol,name),mm in fitted.items():
 if 'constant' in mm:continue
 est=mm['estimator'];vv=est.coef_[0] if hasattr(est,'coef_') else est.feature_importances_
 for k,v in zip(fn,vv):importance.append(dict(model=m,policy=pol,candidate=name,feature=k,value=float(v),type='standardized_logistic_coefficient' if hasattr(est,'coef_') else 'tree_impurity_importance'))
for m in scores.files:
 valid=(scores[m][test.ix]>=0).all(axis=1);g=test.loc[valid].reset_index(drop=True);X=Xt[valid];y=g.true_label.to_numpy();w=g.weight.to_numpy();groups={c:np.array(ii) for c,ii in g.groupby('cell').indices.items()}
 for pol in POL:
  ref,base,nn=decisions(g,m,pol);gatepred={};shows={};qs={}
  for variant in ['selected','logistic','tree']:
   name=choice[m,pol,variant];q=predict(fitted[m,pol,name],X);cut=np.array([cutmap[m,pol,name,b,ds] for b,ds in zip(g.backbone,g.dataset)]);show=q>=cut;gatepred[variant]=np.where(show,nn,base);shows[variant]=show&ref;qs[variant]=q
  cc=np.array([cos[m,b,ds,pol] for b,ds in zip(g.backbone,g.dataset)]);cshow=(g.nearest_cosine.to_numpy()>=cc)&ref;cpred=np.where(cshow,nn,base)
  policies={'control':(base,np.zeros(len(g),bool)),'always_nn':(nn,ref),'cosine_gate':(cpred,cshow),**{v+'_gate':(gatepred[v],shows[v]) for v in gatepred}}
  for cell,ix in groups.items():
   gg=g.iloc[ix];b,ds=gg.backbone.iloc[0],gg.dataset.iloc[0];pp=pop(gg,m,pol)
   for method,(pr,show) in policies.items():rows.append(dict(model=m,backbone=b,dataset=ds,policy=pol,method=method,**metric(y[ix],pr[ix],w[ix],pp),nn_rate=float(w[ix]@show[ix]/pp[0]),test_n=len(ix)))
  for i,row in enumerate(g.itertuples()):predrows.append(dict(model=m,policy=pol,source_index=row.ix,backbone=row.backbone,dataset=row.dataset,cluster=row.cluster,true_label=int(y[i]),weight=float(w[i]),control=int(base[i]),always_nn=int(nn[i]),cosine_gate=int(cpred[i]),selected_gate=int(gatepred['selected'][i]),logistic_gate=int(gatepred['logistic'][i]),tree_gate=int(gatepred['tree'][i]),selected_show=bool(shows['selected'][i]),selected_score=float(qs['selected'][i])))
 print('tested',m,flush=True)
r=pd.DataFrame(rows);r.to_csv(O/'cell_metrics.csv',index=False);pd.DataFrame(predrows).to_csv(O/'test_predictions.csv.gz',index=False);pd.DataFrame(importance).to_csv(O/'feature_importance.csv',index=False)
summary=[];deltas=[]
for cohort in ['all','toxicity4']:
 rr=r if cohort=='all' else r[r.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])]
 for dim in ['overall','model','backbone','dataset']:
  gr=[('ALL',rr)] if dim=='overall' else rr.groupby(dim)
  for name,g in gr:
   for pol in POL+['anchor_average']:
    h=g[g.policy.isin(['baseline_anchor','nearest_anchor','saliency_anchor'])] if pol=='anchor_average' else g[g.policy==pol]
    for method,z in h.groupby('method'):summary.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,method=method,**{k:z[k].mean() for k in ['f1','precision','recall','accuracy','nn_rate']}))
    for method in ['selected_gate','logistic_gate','tree_gate']:
     ga=h[h.method==method].set_index(['model','backbone','dataset','policy'])
     for refname in ['always_nn','cosine_gate','control']:
      ba=h[h.method==refname].set_index(['model','backbone','dataset','policy']);dd=ga[['f1','precision','recall','accuracy']]-ba[['f1','precision','recall','accuracy']]
      if pol=='anchor_average':dd=dd.groupby(level=[0,1,2]).mean()
      deltas.append(dict(cohort=cohort,dimension=dim,group=name,policy=pol,method=method,reference=refname,wins=int((dd.f1>1e-10).sum()),losses=int((dd.f1< -1e-10).sum()),ties=int((abs(dd.f1)<=1e-10).sum()),**{k:dd[k].mean() for k in dd}))
pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(deltas).to_csv(O/'deltas.csv',index=False)
# Exact replay of every previous control, alwaysNN and cosine gate row.
prior=pd.read_csv(P/'nn_similarity_gate_20261019/results/cell_metrics.csv').replace({'method':{'similarity_gate':'cosine_gate'}});ck=r.merge(prior,on=['model','backbone','dataset','policy','method'],suffixes=('_new','_old'),validate='one_to_one');assert len(ck)==7500;err=max(abs(ck[k+'_new']-ck[k+'_old']).max() for k in ['f1','precision','recall','accuracy','nn_rate']);assert err<1e-10
assert sha(O/'model_choices.csv')==freeze['choices_sha256'] and sha(O/'validation_cutoffs.csv')==freeze['cutoffs_sha256']
(O/'COMPLETED.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in src},script_sha256=sha(Path(__file__)),model_count=len(fitted),validation_contexts=len(val),test_contexts=len(test),validation_unique_targets=val.cluster.nunique(),test_unique_targets=test.cluster.nunique(),target_overlap=0,benchmark_replay_rows=len(ck),benchmark_max_error=err,elapsed_seconds=time.time()-start,new_model_calls=0),indent=2))
print(pd.DataFrame(summary).query("cohort=='all' and dimension=='overall'").to_string(index=False));print(pd.DataFrame(deltas).query("cohort=='all' and dimension=='overall' and method=='selected_gate'").to_string(index=False))
