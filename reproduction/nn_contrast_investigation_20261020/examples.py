from pathlib import Path
import gzip,json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results'
f=pd.read_csv(P/'nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz')
s=np.load(P/'api_results_20261015/results/scores.npz')
rows=[]
for m in s.files:
    ix=f.source_index.to_numpy();valid=(s[m][ix]>=0).all(1)
    g=f[valid].copy();a=s[m][g.source_index];bc=(a[:,0]>=50)==g.true_label;nc=(a[:,2]>=50)==g.true_label
    g['model']=m;g['control']=a[:,0];g['nearest']=a[:,2];g['attraction']=(2*g.nearest_label-1)*(a[:,2]-a[:,0])
    g['movement']=np.where(g.attraction<0,'away','toward')
    g['outcome']=np.where(nc,'helped','harmed')
    g=g[(bc!=nc)&g.majority_important_contrast.eq('yes')].copy()
    rows.append(g)
d=pd.concat(rows)
# Each context can show both helpful and harmful effects across judges; retain that.
context=d.groupby(['backbone','sample_id','movement','outcome'],as_index=False).agg(judges=('model','nunique'))
context=context.merge(f,on=['backbone','sample_id'],validate='many_to_one')
context['audit_order']=[hashlib.sha256(('contrast-audit-20261020|'+b+'|'+s+'|'+m+'|'+o).encode()).hexdigest() for b,s,m,o in zip(context.backbone,context.sample_id,context.movement,context.outcome)]
selected=[]
for (mv,ev),g in context.groupby(['movement','outcome']):
    # A deterministic sample across distinct datasets and target texts, max two/dataset.
    used=set();counts={}
    for _,r in g.sort_values('audit_order').iterrows():
        if r.cluster in used or counts.get(r.dataset,0)>=2:continue
        selected.append(r.to_dict());used.add(r.cluster);counts[r.dataset]=counts.get(r.dataset,0)+1
        if len(used)==8:break
want={(r['backbone'],r['sample_id']) for r in selected};texts={}
for line in gzip.open(P/'nn_mechanisms_20261015/features.jsonl.gz','rt'):
    r=json.loads(line);k=(r['backbone'],r['sample_id'])
    if k in want:texts[k]=r
with gzip.open(P/'nn_reason_flips_20261017/FINAL_ANALYSIS_SNAPSHOT.json.gz','rt') as h:rr=json.load(h)['rows']
traces={(r['judge'],r['backbone'],r['sample_id']):r for r in rr}
out=[]
for i,r in enumerate(selected):
    b,si=r['backbone'],r['sample_id'];g=d[(d.backbone==b)&(d.sample_id==si)]
    outcomes=[]
    for j in s.files:
        a=s[j][r['source_index']]
        if (a<0).any():continue
        bc=(a[0]>=50)==r['true_label'];nc=(a[2]>=50)==r['true_label']
        z=dict(judge=j,control=int(a[0]),nearest=int(a[2]),outcome='unchanged' if bc==nc else 'helped' if nc else 'harmed')
        t=traces.get((j,b,si))
        if t:
            z['reason_after_probability']=t['reason_probability'];z['reason_after_text']=t['reasoning'];z['finish_reason']=t['finish_reason']
            z['reason_keeps_original_decision']=bool(t['reason_probability'] is not None and (t['reason_probability']>=50)==(a[2]>=50))
        outcomes.append(z)
    out.append(dict(audit_id=i+1,selected_stratum=r['movement']+'_'+r['outcome'],source=texts[b,si],annotations={k:r[k] for k in f if k.endswith('_important_contrast') or k.endswith('_labelled_similarity')},judge_outcomes=outcomes))
(O/'audit_examples.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
(O/'AUDIT_SELECTION.json').write_text(json.dumps(dict(n=len(out),selection='8 deterministically hash-ranked distinct targets per away/toward x helpful/harmful stratum, maximum2 per dataset; majority important contrast yes; all judge outcomes retained. Cases may recur across strata. Exploratory review with outcomes visible, not blind annotation or prevalence estimation.',seed_text='contrast-audit-20261020',new_model_calls=0),indent=2))
for e in out:
    r=e['source'];text=lambda ms:' '.join(m['content'] for m in ms)
    print(f"\n{e['audit_id']} {e['selected_stratum']} {r['dataset']} {r['backbone']} GOLD {r['true_label']} NN LABEL {r['nearest_label']} PRED {r['nearest_prediction']} COS {r['nearest_cosine']:.3f}")
    print('TARGET:',text(r['target_messages'])[:1200])
    print('NN:',text(r['neighbor_messages'])[:1200])
    print('JUDGES:',[(z['judge'],z['control'],z['nearest'],z['outcome']) for z in e['judge_outcomes']])
