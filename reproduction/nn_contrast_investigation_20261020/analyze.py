"""Existing-data investigation of annotated contrast versus observed repulsion."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from pathlib import Path
import json,gzip,hashlib,time
import numpy as np,pandas as pd

R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';O.mkdir(exist_ok=True)
TOX=['aegis','beavertails','toxicchat','toxigen']
ANN=['majority','qwen122','gemma','qwen','unanimous']
F=P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz'
S=P/'api_results_20261015/results/scores.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=pd.read_csv(F);scores=np.load(S)
assert len(f)==25000 and f.split.eq('test').all()
with gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt') as h:des=json.load(h)
cases=pd.DataFrame([dict(backbone=b,**r) for b,rr in des['samples'].items() for r in rr])
assert list(zip(f.backbone,f.sample_id))==list(zip(cases.loc[f.source_index,'backbone'],cases.loc[f.source_index,'sample_id']))
def wrank(x,w):
    t=pd.DataFrame({'x':np.asarray(x),'w':np.asarray(w)});v=t.groupby('x').w.sum()
    return t.x.map((v.cumsum()-v/2)/v.sum()).to_numpy()
f['probe_saturated']=(f.probe_probability<=1e-12)|(f.probe_probability>=1-1e-12)
pp=np.clip(f.probe_probability,1e-12,1-1e-12)
f['probe_logit_gap']=np.abs(np.log(pp/(1-pp)))
for _,ix in f.groupby(['backbone','dataset']).groups.items():
    f.loc[ix,'probe_margin_rank']=wrank(f.loc[ix,'probe_logit_gap'],f.loc[ix,'weight'])
f['probe_bin']=np.minimum((f.probe_margin_rank*5).astype(int)+1,5)
f['cosine_bin5']=np.minimum((f.cosine_rank*5).astype(int)+1,5)
frames=[]
for m in scores.files:
    ix=f.source_index.to_numpy();v=(scores[m][ix]>=0).all(1);g=f.loc[v].copy();a=scores[m][g.source_index]
    g['model']=m;g['control_score']=a[:,0];g['nearest_score']=a[:,2];g['delta_score']=a[:,2]-a[:,0]
    g['cell']=m+'|'+g.backbone+'|'+g.dataset
    g['w']=g.weight/g.groupby('cell').weight.transform('sum')
    g['similarity']=g.majority_labelled_similarity.eq('yes')
    g['contrast']=g.majority_important_contrast.eq('yes')
    g['aligned']=g.nearest_label.eq(g.true_label)
    g['cue_agrees_control']=g.nearest_label.eq(g.control_score.ge(50).astype(int))
    g['attraction']=(2*g.nearest_label-1)*g.delta_score
    g['away']=g.attraction.lt(0);g['toward']=g.attraction.gt(0);g['score_stable']=g.attraction.eq(0)
    g['movement']=np.select([g.away,g.toward],['away','toward'],default='same_score')
    g['judge_margin']=abs(g.control_score-50)
    g['judge_margin_bin']=pd.cut(g.judge_margin,[-1,10,25,40,50],labels=['0–10','11–25','26–40','41–50']).astype(str)
    g['cue_margin']=(2*g.nearest_label-1)*(g.control_score-50)/50
    g['common_valid']=np.logical_and.reduce([(scores[j][g.source_index]>=0).all(1) for j in scores.files])
    for policy in ['judge50_direct','pipeline50']:
        z=g.copy();refer=np.ones(len(z),bool) if policy=='judge50_direct' else z.probe_probability.ge(.5).to_numpy()
        z['referred']=refer;bc=((z.control_score>=50)&refer)==z.true_label;nc=((z.nearest_score>=50)&refer)==z.true_label
        z['base_correct']=bc;z['helped']=~bc&nc;z['harmed']=bc&~nc;z['unchanged']=bc==nc
        z['gain']=z.helped.astype(int)-z.harmed.astype(int)
        z['flip']=~z.unchanged;z['away_flip']=z.flip&z.away;z['toward_flip']=z.flip&z.toward
        z['outcome']=np.select([z.helped,z.harmed],['helped','harmed'],default='unchanged');z['policy']=policy
        assert (z.away_flip|z.toward_flip).equals(z.flip)
        # In binary classification, these equalities are structural, not causal evidence.
        assert np.all(z.loc[z.away_flip,'helped']==~z.loc[z.away_flip,'aligned'])
        assert np.all(z.loc[z.toward_flip,'helped']==z.loc[z.toward_flip,'aligned'])
        frames.append(z)
d=pd.concat(frames,ignore_index=True)
if not (O/'cases.csv.gz').exists():d.to_csv(O/'cases.csv.gz',index=False)

MET=['helped','harmed','gain','unchanged','away','toward','score_stable','away_flip','toward_flip','attraction','aligned','base_correct','nearest_cosine','cosine_rank','probe_margin_rank','judge_margin','referred']
def stats(g):
    w=g.w.to_numpy();den=w.sum();out=dict(n=len(g),contexts=g.unit_id.nunique(),targets=g.cluster.nunique(),cells=g.cell.nunique(),mass=den)
    for k in MET:
        x=g[k].to_numpy(dtype=float);mu=w@x/den;out[k]=mu
        if k in ['gain','attraction','away_flip']:
            v=pd.Series(w*(x-mu)/den).groupby(g.cluster.to_numpy()).sum().to_numpy()
            se=np.sqrt(v@v*len(v)/max(1,len(v)-1));out[k+'_lo']=mu-1.96*se;out[k+'_hi']=mu+1.96*se
    out['away_share_of_moves']=out['away']/(out['away']+out['toward']) if out['away']+out['toward'] else np.nan
    out['help_share_of_flips']=out['helped']/(out['helped']+out['harmed']) if out['helped']+out['harmed'] else np.nan
    return out
summ=[];groups=[];bins=[];moves=[];sens=[]
for policy,z in d.groupby('policy'):
  for cohort,g0 in [('all10',z),('toxicity4',z[z.dataset.isin(TOX)])]:
    summ.append(dict(policy=policy,cohort=cohort,annotator='none',contrast='all',similarity='all',**stats(g0)))
    for ann in ANN:
      for val,g in g0.groupby(ann+'_important_contrast'):
        summ.append(dict(policy=policy,cohort=cohort,annotator=ann,contrast=val,similarity='all',**stats(g)))
    for (c,s),g in g0.groupby(['majority_important_contrast','majority_labelled_similarity']):
      summ.append(dict(policy=policy,cohort=cohort,annotator='majority',contrast=c,similarity=s,**stats(g)))
    for dim in ['aligned','cue_agrees_control','label_prediction_agree','dataset','model','backbone','base_correct']:
      for (value,c),g in g0.groupby([dim,'majority_important_contrast']):
        groups.append(dict(policy=policy,cohort=cohort,dimension=dim,value=str(value),contrast=c,**stats(g)))
    for axis in ['probe_bin','cosine_bin5','judge_margin_bin']:
      for (bn,c,al),g in g0.groupby([axis,'majority_important_contrast','aligned']):
        bins.append(dict(policy=policy,cohort=cohort,axis=axis,bin=str(bn),contrast=c,alignment=bool(al),**stats(g)))
    for (c,al,mv),g in g0.groupby(['majority_important_contrast','aligned','movement']):
      moves.append(dict(policy=policy,cohort=cohort,contrast=c,alignment=bool(al),movement=mv,**stats(g)))
    for scope,h in [('common_valid',g0[g0.common_valid]),('other7',g0[~g0.model.isin(['qwen122','gemma','qwen'])]),('local6',g0[g0.model.isin(list(scores.files)[:6])]),('api4',g0[g0.model.isin(list(scores.files)[6:])])]:
      h=h.copy();h['w']=h.weight/h.groupby('cell').weight.transform('sum')
      for c,g in h.groupby('majority_important_contrast'):sens.append(dict(policy=policy,cohort=cohort,scope=scope,contrast=c,**stats(g)))
for name,rows in [('summary',summ),('breakdowns',groups),('bins',bins),('movements',moves),('sensitivities',sens)]:pd.DataFrame(rows).to_csv(O/(name+'.csv'),index=False)
print('descriptives complete',flush=True)

def regress(g,ann,outcome,spec):
    g=g[g[ann+'_important_contrast'].isin(['yes','no'])&g[ann+'_labelled_similarity'].isin(['yes','no'])].copy()
    c=g[ann+'_important_contrast'].eq('yes').to_numpy(float);s=g[ann+'_labelled_similarity'].eq('yes').to_numpy(float)
    al=g.aligned.to_numpy(float);r=g.cosine_rank.to_numpy();p=g.probe_margin_rank.to_numpy();jm=g.cue_margin.to_numpy()
    cols=[c];names=['contrast']
    if spec!='cell_only':
        cols += [s,c*s,al,g.true_label.to_numpy(float),r,r*r,p,p*p,jm,jm*jm,jm**3,g.label_prediction_agree.to_numpy(float)]
        names += ['similarity','contrast_x_similarity','aligned','true_label','cosine_rank','cosine_rank2','probe_rank','probe_rank2','cue_margin','cue_margin2','cue_margin3','label_prediction_agree']
    if spec in ['alignment_interaction','same_target']:
        cols += [c*al];names+=['contrast_x_alignment']
    X=np.column_stack(cols);y=g[outcome].to_numpy(float);w=g.w.to_numpy();Z=np.column_stack([y,X])
    fe=[pd.factorize(g.cell)[0]]
    if spec=='same_target':fe.append(pd.factorize(g.model+'|'+g.cluster)[0])
    # Alternating weighted projections for cell and model-target fixed effects.
    converged=False
    for step in range(150):
        old=Z.copy()
        for ids in fe:
            den=np.bincount(ids,weights=w)
            for j in range(Z.shape[1]):Z[:,j]-=np.bincount(ids,weights=w*Z[:,j])[ids]/den[ids]
        if np.max(abs(Z-old))<1e-9:converged=True;break
        if len(fe)==1:converged=True;break
    y=Z[:,0];X=Z[:,1:];bread=np.linalg.pinv(X.T@(w[:,None]*X));b=bread@(X.T@(w*y));res=y-X@b
    ids=pd.factorize(g.cluster)[0];G=ids.max()+1
    u=np.column_stack([np.bincount(ids,weights=w*res*X[:,j],minlength=G) for j in range(X.shape[1])])
    cov=bread@(u.T@u)@bread*G/max(G-1,1)
    ans=[]
    targets=[('contrast',np.eye(len(names))[0])]
    if 'contrast_x_alignment' in names:
        for av in [0,1]:
            vec=np.zeros(len(names));vec[0]=1;vec[names.index('contrast_x_similarity')]=np.average(s,weights=w);vec[names.index('contrast_x_alignment')]=av
            targets.append(('contrast_marginal_aligned'+str(av),vec))
    elif 'contrast_x_similarity' in names:
        vec=np.zeros(len(names));vec[0]=1;vec[names.index('contrast_x_similarity')]=np.average(s,weights=w);targets.append(('contrast_marginal',vec))
    for name,vec in targets:
        est=float(vec@b);se=np.sqrt(max(float(vec@cov@vec),0));ans.append(dict(term=name,estimate=est,lo=est-1.96*se,hi=est+1.96*se,n=len(g),targets=G,fe_converged=converged,fe_iterations=step+1))
    return ans
reg=[]
for cohort,z in [('all10',d[d.policy.eq('judge50_direct')]),('toxicity4',d[d.policy.eq('judge50_direct')&d.dataset.isin(TOX)])]:
 for ann in ['majority','qwen122','gemma','qwen']:
  for outcome in ['attraction','gain','away_flip']:
   for spec in ['cell_only','adjusted','alignment_interaction']+(['same_target'] if ann=='majority' else []):
    for row in regress(z,ann,outcome,spec):reg.append(dict(cohort=cohort,annotator=ann,outcome=outcome,spec=spec,**row))
  print('regressions',cohort,ann,flush=True)
pd.DataFrame(reg).to_csv(O/'regressions.csv',index=False)
receipt=dict(time=time.time(),sources={str(p):sha(p) for p in [F,S,P/'api_results_20261015/frozen/design.json.gz']},contexts=len(f),targets=f.cluster.nunique(),judge_contexts=len(d)//2,probe_saturated_contexts=int(f.probe_saturated.sum()),new_model_calls=0,notes='Direct50 judge analysis primary. Probe score logit gap and within-probe rank are proxies, not geometric or semantic distance. CI clustered on target. Descriptive subgroup means retain original cell weights, so subgroup composition varies.')
(O/'ANALYSIS_COMPLETED.json').write_text(json.dumps(receipt,indent=2))
print(pd.DataFrame(summ).query("policy=='judge50_direct' and annotator=='majority' and similarity=='all'")[['cohort','contrast','n','gain','attraction','away_share_of_moves','help_share_of_flips']].to_string(index=False))
