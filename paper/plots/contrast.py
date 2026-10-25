# Portable adapter of the archived figure source; numerical inputs are immutable.
import os, gzip
from pathlib import Path
DATA=Path(os.environ['NN_PLOT_DATA'])
OUTPUT=Path(os.environ['NN_PLOT_OUTPUT'])
OUTPUT.mkdir(parents=True,exist_ok=True)
import os

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=OUTPUT
b=pd.read_json(DATA/'contrast_bins.json');b=b[b.scope=='Overall'];assert len(b)==10
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig=plt.figure(figsize=(6,6));ax=fig.add_axes([.19,.18,.75,.75]);x=((b.left+b.right)/2).to_numpy()
ax.plot(x,b.net.to_numpy(),'-o',color='#5a72a1',lw=2,ms=5)
ax.vlines(x,b.net_lo.to_numpy(),b.net_hi.to_numpy(),color='#5a72a1',alpha=.7,lw=1.1)
for suf in ['_lo','_hi']:ax.plot(x,b['net'+suf].to_numpy(),linestyle='none',marker='_',color='#5a72a1',ms=6)
ax.axhline(0,color='#777',ls='--',lw=1);ax.set_xlim(0,100);ax.set_ylim(-60,100);ax.set_box_aspect(1)
ax.set_xlabel('Neighbour probe probability (%)');ax.set_ylabel('Net accuracy change (pp)')
for ext in ['png','pdf','svg']:fig.savefig(R/f'overall_contrast_net_square.{ext}',dpi=300)
