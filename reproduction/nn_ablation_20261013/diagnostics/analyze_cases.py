import json,collections,itertools
from pathlib import Path
D=Path(__file__).resolve().parent
rows=json.loads((D/'cases.json').read_text())['rows'];pairs=json.loads((D.parent/'results/pairs.json').read_text());pol={(r['dataset'],r['condition'],r['policy']):r for r in pairs}
def decision(r,c,policy='joint',override=None):
 p=override or pol[r['dataset'],c,policy]
 return int(r['probe_probability']>=p['probe_threshold'] and r['scores'][c]>=p['probability_threshold'])
def measure(rr,c,p):
 n=sum(1/r['inclusion_probability'] for r in rr);positive=sum(r['true_label']/r['inclusion_probability'] for r in rr)
 tp=sum(r['true_label']*decision(r,c,override=p)/r['inclusion_probability'] for r in rr);fp=sum((1-r['true_label'])*decision(r,c,override=p)/r['inclusion_probability'] for r in rr)
 return dict(f1=2*tp/(positive+tp+fp),precision=tp/(tp+fp),recall=tp/positive,tp=tp,fp=fp,n=n)
tc=[r for r in rows if r['dataset']=='toxicchat' and r['split']=='test'];pn=pol['toxicchat','nearest','joint'];pt=pol['toxicchat','text_prediction','joint']
print('TOXIC_CROSS_POLICY')
for c in ['nearest','text_prediction']:
 for name,p in [('full_NN_policy',pn),('text_pred_policy',pt),('all_at50',dict(probe_threshold=0,probability_threshold=50))]:print(c,name,measure(tc,c,p))
changes=collections.Counter();helped=[];hurt=[]
for r in tc:
 a=decision(r,'nearest');b=decision(r,'text_prediction');y=r['true_label']
 if a!=b:
  kind='helped' if b==y else 'hurt';changes[kind]+=1
  changes[kind+('_routing' if r['probe_probability']<pn['probe_threshold'] else '_both_referred')]+=1
  (helped if b==y else hurt).append(r)
print('OWN_POLICY_CHANGES',dict(changes))
for rr,name in [(helped,'HELPED'),(hurt,'HURT')]:
 print(name,len(rr),'neighbour label pred disagreements',sum(r['nearest_label']!=r['nearest_prediction'] for r in rr),'scores same',sum(r['scores']['nearest']==r['scores']['text_prediction'] for r in rr))
 print('SCORE_PAIRS',collections.Counter((r['scores']['nearest'],r['scores']['text_prediction']) for r in rr))
for name,rr in [('helped',helped),('hurt',hurt)]:
 for r in sorted(rr,key=lambda r:(r['nearest_label']==r['nearest_prediction'],r['scores']['nearest']==r['scores']['text_prediction'],r['sample_id']))[:18]:
  print('EXAMPLE',name,json.dumps({k:r[k] for k in ['sample_id','true_label','probe_probability','scores','nearest_label','nearest_prediction','nearest_cosine','messages','nearest_messages']},ensure_ascii=False))
summary=[]
for scope in ['all','toxicchat']:
 test=[r for r in rows if r['split']=='test' and (scope=='all' or r['dataset']==scope)]
 for mode in ['joint_pipeline','judge_at50','same_baseline_gate_at50']:
  for group in ['aligned','diverged']:
   rr=[r for r in test if (r['random_label']==r['true_label'])==(group=='aligned')]
   data=dict(scope=scope,mode=mode,alignment=group,n=len(rr),helped=0,hurt=0,weighted_help=0.,weighted_hurt=0.,weighted_n=0.)
   for r in rr:
    if mode=='joint_pipeline': a=decision(r,'baseline');b=decision(r,'random_full')
    else:
     gate=0 if mode=='judge_at50' else pol[r['dataset'],'baseline','joint']['probe_threshold']
     p=dict(probe_threshold=gate,probability_threshold=50);a=decision(r,'baseline',override=p);b=decision(r,'random_full',override=p)
    w=1/r['inclusion_probability'];data['weighted_n']+=w
    if a!=b:
     k='help' if b==r['true_label'] else 'hurt';data['helped' if k=='help' else 'hurt']+=1;data['weighted_'+k]+=w
   data['weighted_net_accuracy_pp']=100*(data['weighted_help']-data['weighted_hurt'])/data['weighted_n'];summary.append(data)
print('RANDOM_ALIGNMENT',json.dumps(summary,indent=2))
# Keep target-positive and target-negative effects separate under a common judge cutoff.
for y in [0,1]:
 for lab in [0,1]:
  rr=[r for r in rows if r['split']=='test' and r['true_label']==y and r['random_label']==lab];h=w=0
  for r in rr:
   a=r['scores']['baseline']>=50;b=r['scores']['random_full']>=50
   if a!=b:
    if b==y:h+=1
    else:w+=1
  print('RANDOM_BY_TARGET',y,lab,len(rr),h,w)
for ds in sorted({r['dataset'] for r in rows}):
 for aligned in [True,False]:
  rr=[r for r in rows if r['split']=='test' and r['dataset']==ds and (r['random_label']==r['true_label'])==aligned];h=w=0
  for r in rr:
   a=r['scores']['baseline']>=50;b=r['scores']['random_full']>=50
   if a!=b:
    if b==r['true_label']:h+=1
    else:w+=1
  print('RANDOM_DATASET',ds,'aligned' if aligned else 'diverged',len(rr),h,w)
(D/'case_analysis.json').write_text(json.dumps(dict(toxic_helped=helped,toxic_hurt=hurt,random_alignment=summary),indent=2))
