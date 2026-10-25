# Portable adapter of the archived figure source; numerical inputs are immutable.
import os, gzip
from pathlib import Path
DATA=Path(os.environ['NN_PLOT_DATA'])
OUTPUT=Path(os.environ['NN_PLOT_OUTPUT'])
OUTPUT.mkdir(parents=True,exist_ok=True)
import os

from pathlib import Path
import json,html
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=OUTPUT;s=pd.read_json(DATA/'trace_associations.json');j=json.loads((DATA/'trace_joints.json').read_text())
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
