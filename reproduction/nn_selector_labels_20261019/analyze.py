import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['MPLCONFIGDIR']='/tmp/nn-selector-labels-mpl'
from pathlib import Path
import ast,json,gzip,hashlib,joblib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';OLD=P/'nn_learned_gate_20261019';G=P/'nn_selector_granularity_20261019'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src=OLD/'train.py';wanted=['features','predict'];tree=ast.parse(src.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted];exec(compile(ast.Module(body=defs,type_ignores=[]),str(src),'exec'))
fpath=P/'nn_mechanisms_20261015/results/joined_features.csv.gz';apath=P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz';spath=P/'api_results_20261015/results/scores.npz';tpath=P/'anchor_transfer_20261018/results/thresholds.json'
receipt=json.loads((G/'results/COMPLETED.json').read_text());freeze=json.loads((G/'results/SELECTION_FROZEN.json').read_text())
for path in [fpath,spath,tpath]:assert sha(path)==receipt['inputs'][str(path)]
f=pd.read_csv(fpath);f['source_index']=np.arange(len(f));BB=sorted(f[f.split=='val'].backbone.unique());DS=sorted(f[f.split=='val'].dataset.unique());f=f[(f.split=='test')&f.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])].copy()
a=pd.read_csv(apath);anncols=[x for x in a if any(x==p+'_'+z for p in ['majority','qwen122','gemma','qwen'] for z in ['labelled_similarity','important_contrast'])]
assert not a.duplicated(['backbone','sample_id']).any();f=f.merge(a[['backbone','sample_id']+anncols],on=['backbone','sample_id'],how='left',validate='one_to_one');assert f[anncols].notna().all().all();assert len(f)==10000
scores=np.load(spath);cuts=pd.read_csv(G/'results/validation_cutoffs.csv');choices=pd.read_csv(G/'results/model_choices.csv');ths=pd.read_json(tpath);ths=ths[ths.scope!='common_valid'];ths={(r.model,r.backbone,r.dataset,r.anchor):(r.probe_threshold,r.judge_threshold) for r in ths.itertuples()}
POL=['fixed50','baseline_anchor','nearest_anchor','saliency_anchor'];rows=[];modelhashes={}
for m in scores.files:
 valid=(scores[m][f.source_index]>=0).all(axis=1);gg=f.loc[valid].copy();X,names=features(gg)
 for scope,group in [('pool_all','ALL'),('pool_toxicity','TOX')]:
  for pol in POL:
   path=G/'models'/(m+'__'+pol+'__'+scope+'__'+group+'.joblib');modelhashes[str(path)]=sha(path);assert modelhashes[str(path)]==freeze['models'][str(path)]
   bank=joblib.load(path);assert names==bank['features'];name=bank['selected']['logistic'];q=predict(bank['logistic'],X)
   cc=cuts[(cuts.model==m)&(cuts.scope==scope)&(cuts.policy==pol)&(cuts.group==group)&(cuts.candidate==name)];cm={(r.backbone,r.dataset):r.cutoff for r in cc.itertuples()};threshold=np.array([cm[b,ds] for b,ds in zip(gg.backbone,gg.dataset)]);show=q>=threshold
   pairs=np.array([(.5,50) if pol=='fixed50' else ths[m,b,ds,pol.replace('_anchor','')] for b,ds in zip(gg.backbone,gg.dataset)]);ref=gg.probe_probability.to_numpy()>=pairs[:,0];ss=scores[m][gg.source_index];base=ref&(ss[:,0]>=pairs[:,1]);nn=ref&(ss[:,2]>=pairs[:,1])
   h=gg.copy();h['model']=m;h['scope']=scope;h['policy']=pol;h['referred']=ref;h['show_nn']=show;h['gate_score']=q;h['control']=base;h['nearest']=nn;h['gate_prediction']=np.where(show,nn,base);h['outcome']=np.where(base==nn,'unchanged',np.where(nn==h.true_label,'helpful','harmful'));rows.append(h)
z=pd.concat(rows,ignore_index=True);z.to_csv(O/'joined_gate_annotations.csv.gz',index=False)
# Replay final classifications; show decisions were reconstructed from frozen models,
# since final classification alone cannot identify a show decision on unchanged cases.
old=pd.read_csv(G/'results/test_predictions.csv.gz',usecols=['model','policy','scope','source_index','logistic']);old=old[old.scope.isin(['pool_all','pool_toxicity'])];check=z.merge(old,on=['model','policy','scope','source_index'],validate='one_to_one');assert len(check)==len(z);assert (check.gate_prediction==check.logistic).all()
counts=[];stats=[];joint=[]
for scope in ['pool_all','pool_toxicity']:
 for policy in POL+['anchor_average']:
  zz=z[(z.scope==scope)&(z.policy.isin(POL[1:]) if policy=='anchor_average' else z.policy.eq(policy))]
  for subset in ['referred','referred_flips']:
   zzs=zz[zz.referred & ((zz.control!=zz.nearest) if subset=='referred_flips' else True)].copy()
   # Equal cell/policy mass with original inclusion weights, conditional on referrals.
   zzs['analysis_weight']=zzs.weight/zzs.groupby(['model','backbone','dataset','policy']).weight.transform('sum')
   for dim in ['overall','model','dataset']:
    for group,g in ([('ALL',zzs)] if dim=='overall' else zzs.groupby(dim)):
     for annotator in ['majority','qwen122','gemma','qwen']:
      for feature in ['labelled_similarity','important_contrast']:
       col=annotator+'_'+feature;v=g[g[col].isin(['yes','no'])];meta=dict(scope=scope,policy=policy,subset=subset,dimension=dim,group=group,annotator=annotator,feature=feature)
       for weighting,ww in [('raw',np.ones(len(v))),('weighted',v.analysis_weight.to_numpy())]:
        tab=np.zeros((2,2));raw=np.zeros((2,2),int)
        for i,lab in enumerate(['no','yes']):
         for j,decision in enumerate([False,True]):
          mask=(v[col].to_numpy()==lab)&(v.show_nn.to_numpy()==decision);tab[i,j]=ww[mask].sum();raw[i,j]=int(mask.sum())
        denom=np.sqrt(tab.sum(axis=1).prod()*tab.sum(axis=0).prod());phi=(tab[0,0]*tab[1,1]-tab[0,1]*tab[1,0])/denom if denom else np.nan
        rates=np.divide(tab[:,1],tab.sum(axis=1),out=np.full(2,np.nan),where=tab.sum(axis=1)>0)
        stats.append(dict(**meta,weighting=weighting,n=len(v),excluded_unknown=len(g)-len(v),unique_contexts=v.source_index.nunique(),show_given_yes=rates[1],show_given_no=rates[0],difference=rates[1]-rates[0],phi=phi))
        for i,lab in enumerate(['no','yes']):
         for j,decision in enumerate(['suppress','include']):counts.append(dict(**meta,weighting=weighting,label=lab,decision=decision,n=int(raw[i,j]),mass=tab[i,j],row_fraction=tab[i,j]/tab[i,:].sum() if tab[i,:].sum() else np.nan))
     if dim=='overall':
      for (sim,con),h in g.groupby(['majority_labelled_similarity','majority_important_contrast']):joint.append(dict(scope=scope,policy=policy,subset=subset,similarity=sim,contrast=con,n=len(h),include_rate=h.show_nn.mean(),weighted_include=np.average(h.show_nn,weights=h.analysis_weight)))
pd.DataFrame(counts).to_csv(O/'matrices.csv',index=False);pd.DataFrame(stats).to_csv(O/'associations.csv',index=False);pd.DataFrame(joint).to_csv(O/'joint_labels.csv',index=False)
ct=pd.DataFrame(counts);st=pd.DataFrame(stats);main=ct[(ct.scope=='pool_all')&(ct.policy=='fixed50')&(ct.subset=='referred')&(ct.dimension=='overall')&(ct.annotator=='majority')&(ct.weighting=='raw')]
fig,axes=plt.subplots(1,2,figsize=(10,4.8))
for ax,feat,title in zip(axes,['labelled_similarity','important_contrast'],['Labelled similarity','Important contrast']):
 h=main[main.feature==feat];mat=h.pivot(index='label',columns='decision',values='row_fraction').reindex(index=['yes','no'],columns=['include','suppress']);ns=h.pivot(index='label',columns='decision',values='n').reindex(index=['yes','no'],columns=['include','suppress']);ax.imshow(mat.to_numpy(),vmin=0,vmax=1,cmap='Blues')
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{mat.iloc[i,j]*100:.1f}%\n(n={int(ns.iloc[i,j]):,})',ha='center',va='center',color='white' if mat.iloc[i,j]>.55 else '#172332',fontsize=13)
 ax.set_xticks([0,1]);ax.set_xticklabels(['Include NN','Suppress NN']);ax.set_yticks([0,1]);ax.set_yticklabels(['Label: yes','Label: no']);ax.set_title(title);ax.set_xlabel('Logistic selector decision')
fig.suptitle('Toxicity test referrals · pooled logistic · fixed 50/50\nMajority of three mechanism annotators; percentages within each row',fontsize=13);fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(O/'matrices.png',dpi=180);fig.savefig(O/'matrices.pdf');plt.close(fig)
parts=['<!doctype html><html><head><meta charset="utf-8"><title>Selector decisions and mechanism labels</title><style>body{font-family:system-ui;max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.5;color:#172332}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%}h2{margin-top:35px}</style></head><body><h1>What does the logistic selector choose to show?</h1><p>Toxicity datasets: Aegis, BeaverTails, ToxicChat and ToxiGen. Primary selector is the all-dataset pooled model per outcome judge, matching the preceding comparison. The earlier annotations come from Qwen122, Gemma and Qwen; majority means at least two agree. These features were not supplied to the selector.</p><p>These are cross-tabulations, not classification accuracy against a ground truth: similarity and contrast describe a neighbour, whereas the selector predicts whether showing it will improve the final decision. Both features can be present simultaneously. Cases below the probe referral threshold are excluded rather than counted as suppressed explanations.</p><img src="matrices.png">']
for scope,title in [('pool_all','Original pooled logistic'),('pool_toxicity','Toxicity-only pooled logistic')]:
 parts+=['<h2>'+title+'</h2>']
 for pol in ['fixed50','anchor_average']:
  h=st[(st.scope==scope)&(st.policy==pol)&(st.subset=='referred')&(st.dimension=='overall')].copy()
  for col in ['show_given_yes','show_given_no','difference']:h[col]*=100
  parts+=['<h3>'+pol+'</h3>',h[['annotator','feature','weighting','n','excluded_unknown','show_given_yes','show_given_no','difference','phi']].round(3).to_html(index=False,border=0)]
parts+=['<h2>By outcome judge: fixed 50/50, original pooled logistic</h2>']
h=st[(st.scope=='pool_all')&(st.policy=='fixed50')&(st.subset=='referred')&(st.dimension=='model')&(st.annotator=='majority')&(st.weighting=='raw')].copy()
for col in ['show_given_yes','show_given_no','difference']:h[col]*=100
parts+=[h[['group','feature','n','show_given_yes','show_given_no','difference','phi']].round(3).to_html(index=False,border=0),'<h2>Where control and NN actually disagree</h2>']
h=st[(st.scope=='pool_all')&(st.policy=='fixed50')&(st.subset=='referred_flips')&(st.dimension=='overall')&(st.annotator=='majority')].copy()
for col in ['show_given_yes','show_given_no','difference']:h[col]*=100
parts+=[h[['feature','weighting','n','show_given_yes','show_given_no','difference','phi']].round(3).to_html(index=False,border=0),'<h2>Joint annotation categories</h2>',pd.DataFrame(joint).query("scope=='pool_all' and policy=='fixed50' and subset=='referred'").to_html(index=False,border=0),'<p>Raw matrices count outcome-judge × backbone × target decisions, so the same annotated context can appear for multiple outcome judges. They are not independent examples. Weighted associations apply design weights with equal cell/policy mass among referred cases. Unknown/uncertain annotations are excluded separately for each feature and reported. Phi is a descriptive binary association; no independence-based significance test is claimed. The anchor average pools the three frozen policy evaluations, with each policy having equal mass in the weighted calculation.</p><p><a href="matrices.csv">All matrices</a> · <a href="associations.csv">All associations and dataset breakdowns</a> · <a href="joint_labels.csv">Joint labels</a> · <a href="matrices.pdf">Plot PDF</a></p></body></html>']
(O/'report.html').write_text('\n'.join(parts))
(O/'COMPLETED.json').write_text(json.dumps(dict(input_hashes={str(p):sha(p) for p in [fpath,apath,spath,tpath,G/'results/validation_cutoffs.csv',G/'results/model_choices.csv']},frozen_models=modelhashes,replayed_predictions=len(check),replay_mismatches=0,unique_toxicity_test_contexts=10000,joined_rows=len(z),script_sha256=sha(Path(__file__)),new_fits=0,new_llm_calls=0),indent=2))
print(main.to_string(index=False));print(st.query("scope=='pool_all' and policy=='fixed50' and subset=='referred' and dimension=='overall'").to_string(index=False))
