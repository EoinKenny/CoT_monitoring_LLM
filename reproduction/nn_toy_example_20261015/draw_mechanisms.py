from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
import numpy as np
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
blue='#216b9b';red='#b94e51';gold='#df9c16';dark='#203047'
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(R/(name+'.'+ext),dpi=170,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def card(fig,x,y,w,h,title,text,color=dark,fill='#f5f7fa'):
 fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',transform=fig.transFigure,facecolor=fill,edgecolor='#dce3e9',lw=1))
 fig.text(x+.012,y+h-.025,title,fontsize=11,weight='bold',color=color,va='top')
 fig.text(x+.012,y+h-.067,text,fontsize=12,color=dark,va='top',linespacing=1.5)
fig=plt.figure(figsize=(14,8.5),facecolor='white')
fig.text(.055,.95,'A close neighbour can also suggest the wrong class',fontsize=23,weight='bold',color=dark)
fig.text(.055,.904,'Mechanism 2: the label pulls toward a class; the example text can reveal whether that class really applies.',fontsize=12,color='#40556c')
T=(-.18,.87);N=(.28,1.01)
for i in range(2):
 ax=fig.add_axes([.075+.485*i,.425,.405,.36]);ax.set_xlim(-1.3,1.3);ax.set_ylim(0,1.4)
 boundary=-.4 if i==0 else 0
 ax.add_patch(Rectangle((-1.3,0),boundary+1.3,1.4,color='#eaf3fa'));ax.add_patch(Rectangle((boundary,0),1.3-boundary,1.4,color='#faeceb'))
 ax.axvline(boundary,color=red if i==0 else blue,lw=2)
 if i==0:ax.axvline(0,color='#6f7c87',lw=1.2,ls='--');ax.text(.035,.1,'Intended class boundary',rotation=90,fontsize=9,color='#657282')
 ax.text(-1.2,1.25,'Judges SAFE',color=blue,weight='bold',fontsize=10);ax.text(1.2,1.25,'Judges UNSAFE',ha='right',color=red,weight='bold',fontsize=10)
 ax.scatter(*T,s=340,marker='*',c=gold,edgecolors=dark,lw=1.1,zorder=5);ax.scatter(*N,s=110,marker='s',c=red,edgecolors='white',lw=1,zorder=5)
 ax.plot([T[0],N[0]],[T[1],N[1]],':',color=dark,lw=1.5)
 ax.annotate('T: actually SAFE',T,xytext=(-1.12,.54),fontsize=10,color=blue,arrowprops=dict(arrowstyle='-',color=blue))
 ax.annotate('N: actually UNSAFE',N,xytext=(.12,1.15),fontsize=10,color=red,arrowprops=dict(arrowstyle='-',color=red))
 ax.set_xticks([-.8,.8]);ax.set_xticklabels(['Detect phishing','Create phishing']);ax.set_xlabel('Decision-relevant intent',labelpad=8);ax.set_yticks([.1,1.3]);ax.set_yticklabels(['Low','High']);ax.set_ylabel('Shared wording / topic')
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 ax.set_title('A. Class cue alone: overgeneralisation' if i==0 else 'B. Full example: a useful contrast',loc='left',fontsize=14,weight='bold',pad=15,color=dark)
card(fig,.065,.235,.87,.09,'T — CURRENT REQUEST  ·  probe flags it; actual class: SAFE','“How can I detect a convincing phishing email?”')
card(fig,.065,.075,.39,.12,'NEIGHBOUR CLASS INFORMATION ONLY','Gold label: UNSAFE | Probe: UNSAFE\nPossible final answer: UNSAFE — wrong',red,'#fff5f4')
card(fig,.55,.075,.39,.12,'ADD THE NEIGHBOUR TEXT','“How can I create a convincing phishing email?”\nPossible final answer: SAFE — correct',blue,'#f1f7fc')
fig.text(.065,.035,'“Detect” versus “create” explains why the neighbour’s label should not transfer. Full text can help the judge resist a misleading class cue.',fontsize=11,color=dark)
fig.text(.065,.005,'Invented illustration, not a measured boundary. Full NN context does not always prevent this error; our ablation found it reduced the average harm.',fontsize=9,color='#536477')
save(fig,'misleading_neighbour_and_text')
# Same judge improvement, differing probe headroom: both conditions within each scenario have fixed thresholds.
fig=plt.figure(figsize=(14,7.7),facecolor='white')
fig.text(.055,.95,'A better judge decision does not always change the pipeline',fontsize=21,weight='bold',color=dark)
fig.text(.055,.901,'Mechanism 3: the probe decides which cases can benefit. Probe and judge thresholds are fixed at 50% in this toy.',fontsize=12,color='#40556c')
rowsA=[['A: safe','10%','60% → 10%','SAFE → SAFE'],['B: safe','20%','40% → 20%','SAFE → SAFE'],['C: unsafe','80%','90% → 95%','UNSAFE → UNSAFE'],['D: unsafe','90%','95% → 98%','UNSAFE → UNSAFE']]
rowsB=[r[:] for r in rowsA];rowsB[0]=['A: safe','80%','60% → 10%','UNSAFE → SAFE']
for i,rows in enumerate([rowsA,rowsB]):
 x=.055+.485*i;ax=fig.add_axes([x,.42,.43,.35]);ax.axis('off')
 ax.set_title('A. Probe already excludes the false alarm' if i==0 else 'B. Probe refers the false alarm',loc='left',fontsize=14,weight='bold',color=dark,pad=18)
 t=ax.table(cellText=rows,colLabels=['Case / truth','Probe score','Judge: none → NN','Final: none → NN'],cellLoc='left',colLoc='left',colWidths=[.19,.16,.29,.36],bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(9)
 for (r,c),cell in t.get_celld().items():
  cell.set_edgecolor('white');cell.set_text_props(color=dark)
  cell.set_facecolor('#e5edf4' if r==0 else ('#eaf5ef' if r==1 and i==1 else '#f2f5f8'))
  if r==0:cell.set_text_props(weight='bold')
 text='NN fixes the judge’s error on A.\nBut A never reaches the judge,\nso the final answer was already correct.' if i==0 else 'NN fixes exactly the same judge error.\nHere A reaches the judge,\nso NN removes a real false alarm.'
 card(fig,x+.01,.21,.40,.16,'SAME IMPROVEMENT, DIFFERENT CONSEQUENCE',text,blue)
 fig.text(x+.015,.155,'Pipeline F1: 100% → 100%' if i==0 else 'Pipeline F1: 80% → 100%',fontsize=16,weight='bold',color=blue)
fig.text(.065,.085,'This is why better NN-assisted judge scores can coexist with little F1 gain on a dataset where the probe is already very strong.',fontsize=12,color=dark)
fig.text(.065,.038,'All numbers are invented. A and B are two different hypothetical probe settings, each held fixed across explanation conditions.\nScores below the probe threshold are shown for comparison; a deployed pipeline would not need to query those cases.',fontsize=10,color='#536477',linespacing=1.5)
save(fig,'probe_headroom')
print('Saved both mechanism figures')
