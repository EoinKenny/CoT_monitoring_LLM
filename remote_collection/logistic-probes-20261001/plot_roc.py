"""Three held-out ROC curves from frozen logistic-probe predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

root=Path(__file__).resolve().parent
report=json.loads((root/'results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                     'axes.labelsize':11,'legend.fontsize':10,'savefig.facecolor':'white'})
names=[('wildguard','WildGuard response harmfulness','#2563eb'),
       ('high_stakes','High-Stakes · combined test','#0f766e'),
       ('toxicchat','ToxicChat toxicity','#9333ea')]


def draw(ax,name,title,color):
    saved=np.load(root/f'{name}_test_predictions.npz')
    y,p=saved['labels'],saved['probabilities']
    m=report['results'][name]['test']
    fpr,tpr,_=roc_curve(y,p)
    auc=roc_auc_score(y,p)
    assert abs(auc-m['auroc'])<1e-12
    pred=p>=m['threshold']
    assert np.array_equal(pred,saved['predictions'])
    assert int(np.sum(pred & (y==0)))==m['fp']
    assert int(np.sum(pred & (y==1)))==m['tp']
    ax.plot([0,1],[0,1],ls='--',color='#b0b6bf',lw=1.2,zorder=1)
    ax.plot(fpr,tpr,color=color,lw=2.3,label=f'AUROC = {auc:.3f}',zorder=2)
    op_fpr=m['fp']/(m['fp']+m['tn'])
    op_tpr=m['tp']/(m['tp']+m['fn'])
    ax.scatter([op_fpr],[op_tpr],s=65,color='#e4572e',edgecolors='white',linewidths=1,
               label=f'Val-selected threshold: {m["threshold"]:.3f}',zorder=3)
    ax.set_title(f'{title}\n',fontweight='bold',pad=8)
    ax.text(.5,1.015,f'Test n = {len(y):,} · positives = {int(y.sum()):,}',
            transform=ax.transAxes,ha='center',va='bottom',fontsize=10,color='#56616e')
    ax.set(xlim=(-.015,1.015),ylim=(-.015,1.015),xlabel='False positive rate',ylabel='True positive rate (recall)')
    ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
    ax.set_aspect('equal',adjustable='box')
    ax.grid(color='#e8ebef',lw=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor='#e5e7eb')


fig,axes=plt.subplots(1,3,figsize=(16,5.7))
for ax,(name,title,color) in zip(axes,names):
    draw(ax,name,title,color)
fig.suptitle('Mean-pooled logistic probes — held-out ROC curves',fontsize=17,fontweight='bold',y=.985)
fig.text(.5,.025,'Llama-3.2-1B-Instruct · block 11 of 16 · content tokens only · thresholds maximize validation F1',
         ha='center',fontsize=11,color='#56616e')
fig.subplots_adjust(left=.055,right=.99,bottom=.16,top=.79,wspace=.24)
fig.savefig(root/'roc_three_datasets.png',dpi=180)
fig.savefig(root/'roc_three_datasets.pdf')
plt.close(fig)
for name,title,color in names:
    fig,ax=plt.subplots(figsize=(6,6))
    draw(ax,name,title,color)
    fig.tight_layout()
    fig.savefig(root/f'{name}_roc.png',dpi=180)
    plt.close(fig)
print('Saved three verified ROC curves as individual PNGs, a combined PNG, and a vector PDF.')
