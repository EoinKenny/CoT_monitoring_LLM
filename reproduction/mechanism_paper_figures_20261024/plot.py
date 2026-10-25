import os
os.environ['MPLBACKEND']='Agg'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,gzip,hashlib,html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent;P=R.parent
O=R/'results';O.mkdir(exist_ok=True)
M=P/'mechanisms_all12_20261023';T=P/'nn_rationale_mechanisms_20261020'
LAB=['labelled_similarity','important_contrast'];OUT=['helpful','unchanged','harmful']
COL=['#247a59','#84909c','#c56b60']
NAMES={'qwen':'Qwen 27B','qwen122':'Qwen 122B','gemma':'Gemma','mistral':'Mistral','olmo':'OLMo','glm':'GLM','claude4.5-sonnet':'Sonnet 4.5','claude4.8-opus':'Opus 4.8','claude-opus-5.5':'Opus 5.5','gpt-4o':'GPT-4o','gpt-astra':'GPT Astra','gpt-5.5':'GPT-5.5'}
DS={'aegis':'Aegis','beavertails':'BeaverTails','toxicchat':'ToxicChat','toxigen':'ToxiGen','wildguard':'WildGuard','wildjailbreak':'WildJailbreak','fava':'FAVA','halueval':'HaluEval','ragtruth':'RAGTruth','high_stakes':'High Stakes'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})
inputs=[M/'annotations.npz',P/'main_performance_all12_20261023/scores.npz',P/'api_results_20261015/frozen/design.json.gz',P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl',T/'reasoning_first/results/paired_metadata.csv.gz',T/'results/coded_trace_metadata.jsonl.gz',T/'results/audited_examples.json']
a=np.load(inputs[0]);scores=np.load(inputs[1]);design=json.load(gzip.open(inputs[2],'rt'));mapping=[json.loads(l) for l in open(inputs[3])]
cases=[dict(backbone=b,**r) for b,rr in design['samples'].items() for r in rr];lookup={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)}
ix=np.array([lookup[r['backbone'],r['sample_id']] for r in mapping]);y=np.array([cases[i]['true_label'] for i in ix]);w=np.array([1/cases[i]['inclusion_probability'] for i in ix]);ds=np.array([r['dataset'] for r in mapping]);bb=np.array([r['backbone'] for r in mapping])
assert np.array_equal(a['unit_ids'],[r['unit_id'] for r in mapping])
cell_masks={(b,d):(bb==b)&(ds==d) for b in sorted(set(bb)) for d in DS}
rows=[];coverage=[]
for ji,j in enumerate(a['models']):
 s=scores[j][ix];valid=(s[:,0]>=0)&(s[:,2]>=0);bc=(s[:,0]>=50)==y;nc=(s[:,2]>=50)==y
 ev=np.where(~bc&nc,'helpful',np.where(bc&~nc,'harmful','unchanged'))
 for li,lab in enumerate(LAB):
  v=a['leave_one_out'][ji,:,li];known=valid&(v<2)
  coverage.append(dict(judge=j,label=lab,valid_pairs=int(valid.sum()),binary_label=int(known.sum()),excluded=int((valid&~known).sum())))
  for (b,d),mask in cell_masks.items():
   for out in OUT:
    q=mask&known&(ev==out);den=w[q].sum()
    rows.append(dict(judge=j,backbone=b,dataset=d,label=lab,outcome=out,n=int(q.sum()),yes_n=int((q&(v==1)).sum()),rate=100*w[q&(v==1)].sum()/den if den else np.nan))
c=pd.DataFrame(rows);c.to_csv(O/'label_cell_rates.csv',index=False);pd.DataFrame(coverage).to_csv(O/'label_coverage.csv',index=False)
# Explicit macro averaging: backbone -> dataset -> judge; no outcome-size weighting between cells.
lbd=c.groupby(['judge','dataset','label','outcome']).agg(rate=('rate','mean'),backbones=('rate','count')).reset_index()
lj=lbd.groupby(['judge','label','outcome']).agg(rate=('rate','mean'),datasets=('rate','count')).reset_index();lj.to_csv(O/'label_judge_rates.csv',index=False)
ls=lj.groupby(['label','outcome']).rate.mean().reset_index();ls.to_csv(O/'label_overall_rates.csv',index=False)
heat=lbd.pivot(index=['judge','dataset','label'],columns='outcome',values='rate');heat['delta']=heat.helpful-heat.harmful
heat=heat.groupby(['dataset','label']).delta.mean().unstack().reindex(list(DS))[LAB];heat.to_csv(O/'label_dataset_deltas.csv')
fig,axs=plt.subplots(1,3,figsize=(21,5.6),gridspec_kw={'width_ratios':[1,1,2.1]})
fig.subplots_adjust(left=.045,right=.99,bottom=.17,top=.87,wspace=.14)
for li,ax in enumerate(axs[:2]):
 for k,out in enumerate(['unchanged','helpful','harmful']):
  color=COL[OUT.index(out)]
  q=lj[(lj.label==LAB[li])&(lj.outcome==out)].sort_values('judge');vals=q.rate.to_numpy();jitter=np.linspace(-.20,.20,len(vals));ax.scatter(k+jitter,vals,s=45,c=color,alpha=.5)
  mean=vals.mean();ax.plot([k-.25,k+.25],[mean]*2,color=color,lw=4);ax.text(k,mean+3,f'{mean:.1f}%',ha='center',weight='bold',color=color)
 ax.set_xticks(range(3));ax.set_xticklabels(['Unchanged','Corrects\njudge','Misleads\njudge']);ax.set_ylim(0,105);ax.set_yticks(range(0,101,20));ax.set_title(['Similar NN','Contrasting NN'][li],pad=12);ax.set_box_aspect(1)
 ax.text(-.13,1.05,'('+chr(65+li)+')',transform=ax.transAxes,fontsize=16)
axs[0].set_ylabel('Mean label prevalence within outcome (%)');axs[1].set_yticklabels([])
# Complete original examples; join and check against the frozen scores and consensus.
from matplotlib.patches import FancyBboxPatch
import textwrap
example_path=R/'panel_examples.json';inputs.append(example_path)
examples_panel=json.loads(example_path.read_text())
ax=axs[2];ax.axis('off');ax.set_title('Examples and predictions',pad=12)
ax.text(-.035,1.05,'(C)',transform=ax.transAxes,fontsize=16)
map_lookup={(r['backbone'],r['sample_id']):i for i,r in enumerate(mapping)}
for index,(ex,title,color,bg,yy) in enumerate(zip(examples_panel,['Similar NN','Contrasting NN'],['#247a59','#8562aa'],['#edf5f1','#f2eef8'],[.53,.015])):
 row=lookup[ex['backbone'],ex['sample_id']];mi=map_lookup[ex['backbone'],ex['sample_id']];ji=list(a['models']).index(ex['judge'])
 assert scores[ex['judge']][row,0]==ex['control'] and scores[ex['judge']][row,2]==ex['nn']
 assert abs(cases[row]['probe_probability']-ex['probe_probability'])<1e-10
 assert list(a['leave_one_out'][ji,mi])==([1,0] if index==0 else [0,1])
 ctx=ex['context'];ex['independent_consensus']=a['leave_one_out'][ji,mi].tolist()
 ax.add_patch(FancyBboxPatch((.01,yy),.98,.455,boxstyle='round,pad=.01',facecolor=bg,edgecolor='none',transform=ax.transAxes))
 ax.text(.03,yy+.4,title,weight='bold',fontsize=13,color=color,transform=ax.transAxes)
 ax.text(.03,yy+.325,'Target: '+textwrap.fill(ctx['target_messages'][0]['content'].strip(),width=55),fontsize=10.5,va='top',linespacing=1.2,transform=ax.transAxes)
 ax.text(.03,yy+.185,'NN: '+textwrap.fill(ctx['neighbor_messages'][0]['content'].strip(),width=55),fontsize=10.5,va='top',linespacing=1.2,transform=ax.transAxes)
 ax.plot([.62,.62],[yy+.025,yy+.425],color='#ccd3d8',lw=1,transform=ax.transAxes)
 backbone={'gemma':'Gemma','qwen':'Qwen 4B','llama':'Llama 1B','llama70':'Llama 70B'}[ex['backbone']]
 ax.text(.64,yy+.405,{'claude-opus-5.5':'Claude Opus 5.5','claude4.8-opus':'Claude Opus 4.8','claude4.5-sonnet':'Claude Sonnet 4.5'}.get(ex['judge'],NAMES[ex['judge']])+' · '+backbone+' backbone',fontsize=10,weight='bold',transform=ax.transAxes)
 label=lambda v:('Unsafe' if v else 'Safe') if ex['dataset']=='aegis' else ('Toxic' if v else 'Non-toxic')
 info=[('Target label',label(ex['gold'])),('Target probe',label(ex['probe_probability']>=.5)+f" ({100*ex['probe_probability']:.2f}%)"),('NN label',label(ctx['nearest_label'])),('NN probe',label(ctx['nearest_prediction'])+(f" ({100*ex['neighbor_probe_probability']:.2f}%)" if 'neighbor_probe_probability' in ex else '')),('Judge before',label(ex['control']>=50)+f" ({ex['control']}%)"),('Judge + NN',label(ex['nn']>=50)+f" ({ex['nn']}%)")]
 for k,(name,value) in enumerate(info):
  ax.text(.64,yy+.34-k*.055,name+':',fontsize=9.5,color='#53616c',transform=ax.transAxes)
  ax.text(.79,yy+.34-k*.055,value,fontsize=9.5,weight='bold' if k>=4 else 'normal',transform=ax.transAxes)
ax.text(.5,-.065,'ToxiGen: P(toxic). Aegis: P(unsafe). Judge cutoff: 50%.\nNN probability shown here for analysis; the judge saw its binary probe prediction.',fontsize=9.5,ha='center',color='#63758b',transform=ax.transAxes)
for ext in ['png','pdf']:fig.savefig(O/('two_labels.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)
# Preserve the dataset comparison as a standalone supplementary plot.
fig,ax=plt.subplots(figsize=(6,6))
vmax=max(10,5*np.ceil(np.nanmax(np.abs(heat.values))/5));im=ax.imshow(heat.values,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
ax.set_yticks(range(10));ax.set_yticklabels(list(DS.values()));ax.set_xticks([0,1]);ax.set_xticklabels(['Similar NN','Contrasting NN']);ax.set_title('Corrects judge minus misleads judge')
for i in range(10):
 for j in range(2):
  v=heat.values[i,j];ax.text(j,i,f'{v:+.1f}',ha='center',va='center',color='white' if abs(v)>.6*vmax else '#202b35',fontsize=10)
fig.colorbar(im,ax=ax,label='Label prevalence difference (percentage points)',fraction=.05,pad=.04)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/('label_dataset_heatmap.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)
(O/'panel_c_sources.json').write_text(json.dumps(dict(description='Complete original target and neighbour texts; whitespace wrapped for display. Original probability-only judge scores at cutoff50, not later rationale responses. User-selected S7 and Aegis extreme contrast. Both pass probe50; Aegis judge65 to20 and neighbour probe99.59%. Neighbour numerical score is analysis-only. Illustrative successes, not causal evidence for a component.',examples=examples_panel),indent=2))
print('Label plots complete',flush=True)
# Probability-first versus explanation-first, same cases and per-cell denominators for each order.
pair=pd.read_csv(inputs[4]);assert len(pair)==35919 and pair.both_valid.sum()==30920
p=pair[pair.both_valid].copy();rr=[]
for (j,b,d,out),q in p.groupby(['judge','backbone','dataset','flip_type']):
 for order,col in [('Probability first','after_correct'),('Explanation first','first_correct')]:
  value=q[col].astype(float) if out=='harmed' else 1-q[col].astype(float)
  rr.append(dict(judge=j,backbone=b,dataset=d,original_outcome=out,order=order,n=len(q),rate=100*np.average(value,weights=q.w)))
r=pd.DataFrame(rr);r.to_csv(O/'trace_answer_cell_rates.csv',index=False)
rd=r.groupby(['judge','dataset','original_outcome','order']).rate.mean().reset_index()
rj=rd.groupby(['judge','original_outcome','order']).rate.mean().reset_index();rj.to_csv(O/'trace_answer_judge_rates.csv',index=False)
pc=pair.groupby('judge').agg(selected=('both_valid','size'),both_valid=('both_valid','sum'),probability_first_valid=('score_valid_after','sum'),explanation_first_valid=('score_valid','sum'));pc.to_csv(O/'trace_coverage.csv')
# Use only the completed contrast census. The other coded corpus is harm-enriched.
codes=pd.read_json(inputs[5],lines=True);codes=codes[codes.requested_census].copy();assert len(codes)==12219
codes=codes.merge(pair[['judge','backbone','sample_id','w']],on=['judge','backbone','sample_id'],validate='one_to_one')
cr=[]
for (j,b,d,out),q in codes.groupby(['judge','backbone','dataset','flip_type']):
 for feature,relations in [('Similarity stated',['S','B']),('Contrast stated',['C','B'])]:
  cr.append(dict(judge=j,backbone=b,dataset=d,outcome=out,feature=feature,n=len(q),rate=100*np.average(q.relation.isin(relations),weights=q.w)))
cd=pd.DataFrame(cr);cd.to_csv(O/'trace_code_cell_rates.csv',index=False)
cd=cd.groupby(['judge','dataset','outcome','feature']).rate.mean().reset_index().groupby(['judge','outcome','feature']).rate.mean().reset_index();cd.to_csv(O/'trace_code_judge_rates.csv',index=False)
judges=['qwen','gemma','mistral','olmo','glm','qwen122'];colors=['#8562aa','#247a59']
fig,axs=plt.subplots(1,3,figsize=(17,5.3),gridspec_kw={'width_ratios':[1.05,1.05,1]})
for k,out in enumerate(['harmed','helped']):
 ax=axs[k]
 for i,j in enumerate(judges):
  q=rj[(rj.judge==j)&(rj.original_outcome==out)].set_index('order').rate
  ax.plot([q['Probability first'],q['Explanation first']],[i,i],color='#bbc2c8',lw=1.5)
  for z,order in enumerate(['Probability first','Explanation first']):ax.scatter(q[order],i,color=colors[z],marker=['o','D'][z],s=50,zorder=3)
 ax.set_yticks(range(6));ax.set_yticklabels([NAMES[j] for j in judges] if k==0 else []);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Mean within-category rate (%)');ax.set_title(['Original harmful flips corrected','Original helpful flips undone'][k],pad=15);ax.set_box_aspect(1);ax.text(-.12,1.05,'('+chr(65+k)+')',transform=ax.transAxes,fontsize=16)
ax=axs[2]
for i,feature in enumerate(['Similarity stated','Contrast stated']):
 for k,out in enumerate(['helped','harmed']):
  vals=cd[(cd.feature==feature)&(cd.outcome==out)].sort_values('judge').rate.to_numpy();x=i+(-.17 if k==0 else .17)
  ax.scatter(x+np.linspace(-.06,.06,len(vals)),vals,color=COL[0 if k==0 else 2],alpha=.5,s=35)
  ax.scatter(x,vals.mean(),marker='D',s=65,color=COL[0 if k==0 else 2],edgecolor='white',zorder=5)
ax.set_xlim(-.5,1.5);ax.set_xticks([0,1]);ax.set_xticklabels(['Similarity\nstated','Contrast\nstated']);ax.set_ylim(0,100);ax.set_ylabel('Mean coded-trace prevalence (%)');ax.set_title('Among contrast-labelled flips',pad=15);ax.set_box_aspect(1);ax.text(-.18,1.05,'(C)',transform=ax.transAxes,fontsize=16)
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=COL[0],label='Originally helpful'),Line2D([],[],marker='o',ls='',color=COL[2],label='Originally harmful')],loc='upper left',frameon=False,fontsize=9)
fig.legend(handles=[Line2D([],[],marker=m,ls='',color=c,label=l) for m,c,l in zip(['o','D'],colors,['Probability first','Explanation first'])],loc='lower center',bbox_to_anchor=(.34,.005),ncol=2,frameon=False)
fig.tight_layout(rect=[0,.08,1,1],w_pad=3)
for ext in ['png','pdf']:fig.savefig(O/('explanation_traces.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)
print('Trace plots complete',flush=True)
# Explicitly selected illustrative examples; preserve full source records and quotations.
examples=json.load(open(inputs[6]));ids=['ctx_2ef7d342da6143923eee','ctx_ca937104271e7202a837','ctx_48183f1b9cb18e292127'];selected=[next(x for x in examples if x['audit_id']==i) for i in ids]
(O/'illustrative_examples.json').write_text(json.dumps(selected,indent=2))
body='''<h1>1-NN mechanisms: labels and written explanations</h1><p>Backbones are averaged throughout. The two-label figure covers all 12 judges. The explanation-trace figure uses the completed six-local-judge audit; API traces are not represented in that audit.</p><h2>1. Two-label analysis</h2><p>The two labels are displayed as <b>Similar NN</b> and <b>Contrasting NN</b> (the original annotation fields remain labelled_similarity and important_contrast).</p><p>Each dot in A/B is one outcome judge; bars show the mean across the 12 judges. Labels use an independent majority (6 of the other 11 annotators); uncertain/unresolved labels are excluded from binary percentages and retained in the coverage table. Corrects judge means control wrong → NN correct; Misleads judge means control correct → NN wrong; Unchanged includes both correct and both wrong. Decisions use a common judge cutoff of 50, without the probe gate.</p><img src="two_labels.png"><p>Panel C uses user-selected S7 (ToxiGen, Claude Sonnet 4.5, Llama 1B) and the Aegis boyfriend-versus-identity-theft example (GPT-5.5, Llama 1B). Complete texts, labels and scores are checked against frozen records. S7 changes non-toxic to toxic (15 to 75). The Aegis target is labelled safe; its neighbour is labelled unsafe with recorded probe probability 99.59%. GPT-5.5 changes from 65% unsafe to 20%, correcting the target decision. Both targets exceed the probe50 referral threshold. The numerical neighbour probability is displayed here for analysis, not part of the original judge prompt; the judge saw the neighbour text, label and binary prediction. These selected examples illustrate associations, not verified internal mechanisms. API names are collection aliases. <a href="panel_c_sources.json">Example provenance</a>. The labels can overlap; these examples illustrate the distinction rather than imply mutually exclusive classes. <a href="label_dataset_heatmap.png">Dataset heatmap (supplement)</a> · <a href="label_dataset_heatmap.pdf">Heatmap PDF</a>.</p><p><a href="two_labels.pdf">Vector PDF</a> · <a href="label_judge_rates.csv">Judge values</a> · <a href="label_coverage.csv">Label coverage</a></p><p>Within each judge/backbone/dataset/outcome, use inverse-inclusion weights. Then average backbones, datasets, and judges equally. Undefined outcome-category rates are omitted, not replaced by zero; available counts are in label_cell_rates.csv and label_judge_rates.csv. This explicit macro average differs from earlier plots that normalised cell mass then pooled conditional denominators.</p><h2>2. Explanation-trace analysis</h2><p>A: higher means more original NN errors corrected. B: lower means fewer original NN successes undone. Both orders use exactly the same 30,920 scoreable case–judge–backbone rows from 35,919 selected original flips. Average weighted rates within cell, then backbone and dataset means equally. C: probability-first explanations in the completed contrast-labelled census (12,219 traces), classified by the original NN outcome; dots are the six judges and diamonds are their means. This is a selected contrast subset, not all explanations. Expressed similarity and expressed contrast can coexist.</p><img src="explanation_traces.png"><p><a href="explanation_traces.pdf">Vector PDF</a> · <a href="trace_answer_judge_rates.csv">Answer changes</a> · <a href="trace_code_judge_rates.csv">Expressed comparisons</a> · <a href="trace_coverage.csv">Coverage</a></p><p>The probability-first run used a 256-token cap; most explanation-first responses allowed 512 tokens (640 inherited responses retained a 256-token cap). Order and token budget are therefore not isolated. Missing/unscored responses remain in coverage; the paired comparison does not assume they are random. Codes describe written explanations, not faithful internal reasoning. Original helpful/harmful groups stay fixed even when the new answer changes. The historical contrast-selection labels are not interchangeable with the newer 12-annotator consensus.</p><h2>3. Illustrative excerpts</h2><p>These examples were selected for readability from the existing contextual audit, not sampled to estimate prevalence. Full source text and audit notes are retained in <a href="illustrative_examples.json">this file</a>.</p>'''
for title,x in zip(['Helpful similarity: a labelled precedent','Helpful contrast: a relevant difference','Harmful comparison: a missed implication'],selected):
 ctx=x['context'];body+='<section><h3>'+title+'</h3><p>'+html.escape(DS[x['dataset']]+' · '+NAMES[x['judge']])+'</p>'
 for label,key in [('Target','target_messages'),('Neighbour','neighbor_messages')]:
  body+='<p><b>'+label+':</b> '+html.escape('\n'.join(m['content'] for m in ctx[key]))+'</p>'
 body+=f'<p>Neighbour label: {ctx["nearest_label"]}; neighbour prediction: {ctx["nearest_prediction"]}. Target gold: {x["gold"]}. Control → NN → probability-first: {x["control"]} → {x["nn"]} → {x["new_probability"]}.</p><blockquote>'+html.escape(x['rationale'])+'</blockquote><p><b>Audit note:</b> '+html.escape(x['readings'][0]['note'])+'</p></section>'
body+='<h2>Audit details</h2>'+pc.to_html()+'<p>All statistics are descriptive test-set associations; no new inference or threshold tuning was performed.</p>'
(O/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NN mechanism figures</title><style>body{font:17px/1.55 system-ui;max-width:1550px;margin:40px auto;padding:0 30px;color:#203040}img{width:100%}h2{margin-top:55px}section{border:1px solid #ddd;padding:20px;margin:20px 0;border-radius:10px}blockquote{border-left:4px solid #8562aa;padding-left:20px}td,th{padding:7px}a{color:#24698b}</style>'+body)
receipt=dict(no_new_calls=True,labels_judges=12,label_cell_rows=len(c),empty_label_outcome_cells=int(c.rate.isna().sum()),label_contexts=len(mapping),trace_judges=6,selected_traces=len(pair),paired_traces=len(p),coded_contrast_traces=len(codes),macro_averaging='Inverse inclusion within each outcome/cell; equal means over backbones then datasets then judges. Empty conditional cells omitted.',input_sha256={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},limitations=['Trace semantics cover six local judges only','Coded contrast subset is not all traces','Order and token budget confounded','New 12-annotator labels differ from historical trace-selection scheme'])
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2))
print(ls.to_string(index=False));print(rj.to_string(index=False));print(cd.groupby(['feature','outcome']).rate.mean().to_string())
