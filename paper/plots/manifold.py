# Portable adapter of the archived figure source; numerical inputs are immutable.
import os, gzip
from pathlib import Path
DATA=Path(os.environ['NN_PLOT_DATA'])
OUTPUT=Path(os.environ['NN_PLOT_OUTPUT'])
OUTPUT.mkdir(parents=True,exist_ok=True)
"""Text-free schematic manifold for manual figure composition; not measured geometry."""
import os

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=OUTPUT
fig,ax=plt.subplots(figsize=(10,4))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(1,62),ylim=(16,45));ax.axis('off')
rng=np.random.default_rng(42)
t=np.linspace(0,1,240)
x=5+53*t;y=29+8*np.sin(1.45*np.pi*t-.7)
ax.plot(x,y,color='#dbe3e8',lw=48,alpha=.28,solid_capstyle='round')
xp=5+53*rng.uniform(0,1,140)
yp=29+8*np.sin(1.45*np.pi*(xp-5)/53-.7)+rng.normal(0,2.3,len(xp))
ax.scatter(xp,yp,s=36,c=np.where(xp<31,'#7e9fbb','#c7736b'),alpha=.68,edgecolors='white',linewidths=.35)
ax.plot([30,30],[18,42],ls=(0,(4,4)),color='#98a6b1',lw=1.4)
ax.text(30,45,'Probe boundary',ha='center',va='bottom',fontsize=16)
ax.set_ylim(16,48)
for ext in ('pdf','svg','png'):
 fig.savefig(R/f'manifold_background.{ext}',dpi=200,transparent=True)
plt.close(fig)
