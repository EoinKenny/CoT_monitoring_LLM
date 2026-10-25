import os
os.environ['MPLCONFIGDIR']='/private/tmp/nn_paper_mpl'
from pathlib import Path
import json,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';s=pd.read_json(O/'associations.json');v=json.loads((O/'VERIFIED.json').read_text());j=json.loads((O/'joint_distributions.json').read_text());cov=pd.read_json(O/'coverage.json');within=pd.read_json(O/'within_cell_sensitivity.json');inter=pd.read_json(O/'outcome_interactions.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'similarity':'#8966ad','contrast':'#26876c'};names={'similarity':'Similar NN','contrast':'Contrasting NN'}
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/f'{name}.{ext}',dpi=190,bbox_inches='tight')
 plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4.7))
for ax,f in zip(axs,colors):
 row=s[(s.scope=='All original flips')&(s.feature==f)].iloc[0];joint=np.array(j['All original flips|'+f]);a=100*joint/joint.sum(1)[:,None];raw=np.array(row.raw_joint)
 ax.imshow(a,vmin=0,vmax=100,cmap='Purples' if f=='similarity' else 'Greens')
 for y in range(2):
  for x in range(2):ax.text(x,y,f'{a[y,x]:.1f}%\n(n={raw[y,x]:,})',ha='center',va='center',color='white' if a[y,x]>55 else '#203242',fontsize=13)
 ax.set_xticks([0,1]);ax.set_xticklabels(['Not stated','Stated']);ax.set_yticks([0,1]);ax.set_yticklabels(['Original label: no','Original label: yes']);ax.set_xlabel('Comparison in written explanation');ax.set_title(names[f]);ax.tick_params(length=0)
fig.tight_layout(w_pad=2);save(fig,'agreement_matrices')
fig,axs=plt.subplots(1,2,figsize=(10,4.7))
for ax,f in zip(axs,colors):
 for idx,scope in enumerate(['Corrects judge','Misleads judge']):
  row=s[(s.scope==scope)&(s.feature==f)].iloc[0]
  for offset,key,label,marker in [(-.12,'stated_if_no','Original label: no','o'),(.12,'stated_if_yes','Original label: yes','s')]:
   ax.errorbar(idx+offset,row[key],yerr=[[row[key]-row[key+'_lo']],[row[key+'_hi']-row[key]]],fmt=marker,color=colors[f] if offset>0 else '#697b8c',capsize=4,label=label if idx==0 else None)
   ax.annotate(f'{row[key]:.1f}',(idx+offset,row[key]),xytext=(0,13 if offset>0 else -19),textcoords='offset points',ha='center',fontsize=10)
 ax.set_xticks([0,1]);ax.set_xticklabels(['Corrects\njudge','Misleads\njudge']);ax.set_ylim(0,35);ax.set_xlim(-.4,1.4);ax.set_box_aspect(1);ax.set_title(names[f]);ax.set_ylabel('Comparison stated in explanation (%)' if f=='similarity' else '')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False);fig.tight_layout(rect=[0,.09,1,1]);save(fig,'outcomes')
for dim,title in [('judge','Source judge'),('genre','Dataset genre')]:
 groups=sorted(s[s.scope.str.startswith(dim+': ')].scope.unique());fig,axs=plt.subplots(1,2,figsize=(11,6.5 if dim=='judge' else 4.3),sharey=True)
 for ax,f in zip(axs,colors):
  rr=s[(s.feature==f)&s.scope.isin(groups)].set_index('scope').loc[groups];y=np.arange(len(groups));x=rr.difference_pp.to_numpy();lo=rr.difference_pp_lo.to_numpy();hi=rr.difference_pp_hi.to_numpy()
  ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt='o',color=colors[f],capsize=3);ax.axvline(0,color='#9a9a9a',ls='--',lw=1);ax.set_yticks(y);ax.set_yticklabels([g.split(': ',1)[1] for g in groups]);ax.set_xlabel('Increase when original label is yes (pp)');ax.set_title(names[f]);ax.invert_yaxis()
 axs[0].set_ylabel(title);fig.tight_layout();save(fig,dim+'_associations')
# Present concise tables with units and intervals.
def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.2f}',escape=True,border=0)
main=s[s.scope=='All original flips'][['feature','n','targets','stated_if_no','stated_if_yes','difference_pp','difference_pp_lo','difference_pp_hi','phi']].copy();main['feature']=main.feature.map(names)
out=s[s.scope.isin(['Corrects judge','Misleads judge','Corrects judge; answer retained','Misleads judge; answer retained'])][['scope','feature','n','stated_if_no','stated_if_yes','difference_pp','difference_pp_lo','difference_pp_hi']]
sens=s[s.scope.isin(['Original NN answer retained','Complete source explanation','Source answer probability first','Exclude Qwen122 source judge','Exclude Qwen122 from original consensus'])][['scope','feature','n','difference_pp','phi']]
sections=''
for dim in ['judge','genre','dataset']:
 rr=s[s.scope.str.startswith(dim+': ')][['scope','feature','n','stated_if_no','stated_if_yes','difference_pp','phi']]
 sections+=f'<h2>By {dim}</h2>'+(f'<img src="{dim}_associations.png"><p><a href="{dim}_associations.pdf">Download PDF</a></p>' if dim!='dataset' else '')+table(rr)
page=f'''<!doctype html><meta charset="utf-8"><title>Qwen trace labels versus original NN labels</title><style>body{{font:17px/1.55 system-ui;color:#203242;max-width:1250px;margin:36px auto;padding:0 24px}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px;display:block;overflow:auto}}td,th{{padding:8px;border-bottom:1px solid #e1e6eb;text-align:left}}h2{{margin-top:40px}}.lead{{background:#f0f5f7;padding:18px;border-radius:10px}}a{{color:#246c89}}</style>
<h1>Do the judges’ explanations match the original NN labels?</h1>
<div class="lead"><b>Yes, modestly: both labels predict the corresponding expressed comparison.</b> Similarity is stated in <b>24.6% versus 15.0%</b> of original yes/no cases (+9.6 pp; 95% interval 7.9–11.1). Contrast is stated in <b>14.9% versus 7.6%</b> (+7.3 pp; 5.8–8.7). Binary correlations are small (φ≈0.10 and 0.11). The similarity association is stronger on helpful flips; contrast does not reliably distinguish helpful from harmful flips.</div>
<p>Uses <b>your revised prompt</b>, with Qwen122 as the sole trace labeler, across saved probability-first explanations from all <b>12 source judges, five backbones, and ten datasets</b>. Only cases where the original no-explanation and 1-NN probability-only decisions differed at cutoff 50 are included. No probe-referral filter. This is a correlation analysis of expressed explanations, not an F1 evaluation.</p>
<h2>1. Original label versus expressed comparison</h2><img src="agreement_matrices.png"><p><a href="agreement_matrices.pdf">PDF</a> · <a href="agreement_matrices.svg">SVG</a>. Each row sums to 100%. The question is: when the original pair was labelled yes/no, how often does the later explanation explicitly express that comparison? Raw n counts differ from weighted percentages.</p>{table(main)}
<h2>2. Corrects judge versus misleads judge</h2><img src="outcomes.png"><p><a href="outcomes.pdf">PDF</a> · <a href="outcomes.svg">SVG</a>. Grey circles: original pair label no. Coloured squares: original pair label yes. Error bars are 95% target-cluster bootstrap intervals. Outcome categories use the <b>original probability-only</b> control and NN answers.</p>
<p>For similarity, the yes-versus-no association is +11.6 pp on helpful flips and +5.4 pp on harmful flips. Their difference is <b>+6.2 pp [1.9, 10.1]</b>. This remains positive when the requested explanation retains the original NN classification: +5.7 pp [0.7, 9.6]. Contrast’s corresponding difference is −2.2 pp [−4.7, 0.7], and +0.8 pp [−1.9, 3.9] in the retained-answer subset: <b>no clear helpful-versus-harmful distinction</b>.</p>{table(out)}<h3>Direct comparison of associations across outcomes</h3>{table(inter)}
<h2>3. Robustness checks</h2>{table(sens)}<p>Removing Qwen122 from the original consensus as well as excluding the source judge leaves a positive association for both labels. This reduces overlap between the original annotators and the trace labeler. Excluding Qwen122-generated explanations also preserves both associations. The matched-within-cell check below compares original yes/no cases within the same judge–dataset–backbone cell before averaging; the associations remain positive.</p>{table(within[within.scope=='All original flips'])}
{sections}
<h2>Coverage and exclusions</h2><p>{v['records']:,} source records map to {v['unique_annotation_calls']:,} unique prompts after exact-text deduplication; {v['distinct_targets']:,} distinct target IDs and {v['backbone_targets']:,} backbone–target contexts. There are {v['original_flip_counts']['Corrects judge']:,} original helpful flips and {v['original_flip_counts']['Misleads judge']:,} harmful flips. Qwen produced 43,560 valid and 529 invalid annotation outputs, mapping to 45,037 valid and 559 invalid source records. Another 42 source records were not sent because their explanations were missing, erroneous, or ambiguous. Among valid records, 44,552 are assessable, 430 unassessable, and 55 uncertain. Resolved original and trace labels are required separately for each feature; this leaves 44,083 similarity pairs and 43,441 contrast pairs. No invalid record is silently treated as “no”.</p>{table(cov)}
<h2>Methods and interpretation</h2><p>Original pair labels are the existing strict majority consensus excluding the source judge. The trace labeler saw only the saved written explanation, with edge answer-only integer lines removed; it did not see original mechanism labels, judge identity, gold answers, or helpful/harmful status. Both comparison labels can be yes. All analysed resolved pairs have a recorded first probability. “Answer retained” means that first probability gives the same ≥50 decision as the original NN probability-only response.</p>
<p>Inverse-inclusion weights are applied within source-judge × dataset × backbone cells. We average joint 2×2 proportions equally over available backbones, datasets within source judge, and source judges, then derive conditional rates and φ. Therefore the aggregate is not a raw pooled percentage and is not the arithmetic mean of per-judge conditional differences. Missing support can change subgroup composition. The additional matched-cell sensitivity controls this directly.</p>
<p>Intervals use 1,000 resamples of target IDs within dataset, sharing each target’s resample weight across every judge and backbone. They are pointwise, treat annotations as fixed, and do not include annotation-model or all sampling-design uncertainty. Helpful-minus-harmful differences use the same resamples. This analysis is exploratory and restricted to originally flipped cases; it does not estimate how mechanisms behave on unchanged decisions or their causal effects. Written explanations after a probability can be post-hoc justifications. An unmentioned comparison is not evidence that it was absent internally.</p>
<p><b>Paper conclusion:</b> Original similarity and contrast annotations have modest convergent evidence in explicitly expressed explanations. Similarity has the clearer link to successful corrections. Contrast appears in both successful and unsuccessful reasoning and remains an open question about when a comparison is useful, rather than a general explanation of gains.</p>
<p><a href="associations.json">All numerical results</a> · <a href="VERIFIED.json">Coverage, hashes, and verification</a></p>'''
(O/'report.html').write_text(page)
print(O/'report.html')
