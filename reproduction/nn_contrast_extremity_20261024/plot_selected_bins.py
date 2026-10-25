import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nn_paper_mpl')
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'results'
b=pd.read_json(R/'selected_spectrum_bins_ci.json');assert len(b)==10 and b.rows.sum()==3143
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
x=((b.left+b.right)/2).to_numpy();fig,ax=plt.subplots(1,2,figsize=(11,5))
for k,label,color in [('helped','Corrects judge','#258466'),('harmed','Misleads judge','#c66d66')]:ax[0].plot(x,b[k].to_numpy(),'-o',color=color,lw=2,label=label)

for k,color in [('helped','#258466'),('harmed','#c66d66')]:
 offset_x=x+(-1.8 if k=='helped' else 1.8)
 ax[0].vlines(offset_x,b[k+'_lo'].to_numpy(),b[k+'_hi'].to_numpy(),color=color,alpha=.7,lw=1.2)
 for suffix in ['_lo','_hi']:ax[0].plot(offset_x,b[k+suffix].to_numpy(),linestyle='none',marker='_',color=color,markersize=7)
ax[0].set_ylabel('Mean outcome rate (%)');ax[0].set_ylim(0,max(60,b[['helped_hi','harmed_hi']].max().max()+3));ax[0].legend(frameon=False)
ax[1].plot(x,b.net.to_numpy(),'-o',color='#5a72a1',lw=2);ax[1].axhline(0,color='#777',ls='--',lw=1);ax[1].set_ylabel('Net accuracy change (pp)')
ax[1].vlines(x,b.net_lo.to_numpy(),b.net_hi.to_numpy(),color='#5a72a1',alpha=.7,lw=1.2)
for suffix in ['_lo','_hi']:ax[1].plot(x,b['net'+suffix].to_numpy(),linestyle='none',marker='_',color='#5a72a1',markersize=7)
for a in ax:a.set_xlim(0,100);a.set_xticks(range(0,101,20));a.set_xlabel('Neighbour probe probability (%)');a.set_box_aspect(1)
fig.suptitle('Exploratory · Original judge probability 50–70',fontsize=15);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(R/f'exploratory_selected_bins.{ext}',dpi=180,bbox_inches='tight')
html=f'''<!doctype html><meta charset="utf-8"><title>Ten-point bins: contrasting neighbours</title><style>body{{font:17px/1.5 system-ui;max-width:1200px;margin:35px auto;padding:0 25px;color:#263547}}img{{width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}</style><h1>Contrasting neighbours: ten-point bins</h1><p>Back to the pooled view: original judge probability <b>50–70</b>, without separating neighbour sides. Same exploratory subset of 3,143 paired test judgments from the four toxicity/harmful-content datasets.</p><img src="exploratory_selected_bins.png"><p>Each point represents one non-overlapping neighbour-probability bin: [0,10), [10,20), …, [90,100]. Lines connect bin estimates; no smoothing is applied. Left: percentages of all cases corrected or misled, including unchanged cases in the denominator. Right: corrections minus harmful flips. Unchanged = 100 − corrected − misled.</p><p>Design weights within judge/backbone/dataset, then equal averaging over available backbones, datasets and judges. Group composition can differ across bins. The judge band was selected using test outcomes; these plots are exploratory. Green and red error bars are offset horizontally by 1.8 points for visibility; point estimates remain at bin centres. Error bars show 95% pointwise percentile intervals from 1,000 target-ID bootstrap resamples within dataset, keeping repeated judge/backbone observations together. Net-change intervals use paired corrections minus harms in each resample. They hold the selected 50–70 band fixed and do not correct for its selection using test outcomes.</p>{b.to_html(index=False,float_format=lambda x:f'{x:.2f}')}<p><a href="exploratory_selected_bins.pdf">PDF</a> · <a href="exploratory_selected.html">Smoothed pooled view</a></p>'''
(R/'exploratory_selected_bins.html').write_text(html)
