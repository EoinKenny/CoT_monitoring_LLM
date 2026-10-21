from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14,10),facecolor='white')
fig.text(.055,.955,'How a nearest neighbour can help the judge',fontsize=23,weight='bold',color='#203047')
fig.text(.055,.917,'Toy task: does this request seek harmful assistance?',fontsize=15,color='#40556c')
fig.text(.055,.879,'Same input in both conditions: the probe flags T → the LLM judge decides whether to keep or reject the flag.',fontsize=12,color='#40556c')
blue='#216b9b';red='#b94e51';gold='#df9c16';dark='#203047'
fig.text(.055,.855,'★ Current request     ● Safe examples     ■ Unsafe examples',fontsize=10,color=dark)
T=(-.65,1.08);N=(-.95,.78)
for i in range(2):
 ax=fig.add_axes([.075+.485*i,.475,.405,.335])
 ax.set_xlim(-1.7,1.7);ax.set_ylim(0,1.6)
 if i==0:
  ax.add_patch(Rectangle((-1.7,0),3.4,.96,color='#eaf3fa',zorder=0));ax.add_patch(Rectangle((-1.7,.96),3.4,.64,color='#faeceb',zorder=0))
  ax.axhline(.96,color=red,lw=2)
  ax.text(1.59,1.48,'Judges UNSAFE',ha='right',color=red,fontsize=10,weight='bold')
  ax.text(1.59,.1,'Judges SAFE',ha='right',color=blue,fontsize=10,weight='bold')
  ax.text(.17,.89,'Illustrative decision boundary',fontsize=9,color=red,va='top')
  title='A. Target text only'
 else:
  ax.add_patch(Rectangle((-1.7,0),1.85,1.6,color='#eaf3fa',zorder=0));ax.add_patch(Rectangle((.15,0),1.55,1.6,color='#faeceb',zorder=0))
  ax.axvline(.15,color=blue,lw=2)
  ax.text(-1.55,1.48,'Judges SAFE',color=blue,fontsize=10,weight='bold')
  ax.text(1.59,1.48,'Judges UNSAFE',ha='right',color=red,fontsize=10,weight='bold')
  ax.text(.19,.13,'Illustrative decision boundary',fontsize=9,color=blue,rotation=90,va='bottom')
  ax.scatter(*N,s=110,c=blue,edgecolors='white',linewidths=1.5,zorder=5)
  ax.annotate('',xy=N,xytext=T,arrowprops=dict(arrowstyle='<->',linestyle=':',color=dark,lw=1.8,shrinkA=14,shrinkB=10))
  ax.annotate('N: retrieved safe example',xy=N,xytext=(-1.52,.42),fontsize=10,color=blue,arrowprops=dict(arrowstyle='-',color=blue,lw=1),zorder=6)
  title='B. Target text + labelled neighbour'
 ax.scatter([-1.35,-1.1,-.72],[.22,.32,.16],s=28,c=blue,alpha=.7)
 ax.scatter([.9,1.25,1.38],[1.05,1.25,.72],s=32,c=red,marker='s',alpha=.7)
 ax.scatter(*T,s=340,c=gold,edgecolors=dark,linewidths=1.1,marker='*',zorder=6)
 ax.annotate('T: current request',xy=T,xytext=(-1.4,1.3),fontsize=11,color=dark,arrowprops=dict(arrowstyle='-',color=dark,lw=1),zorder=7)
 ax.set_title(title,loc='left',fontsize=15,weight='bold',pad=16,color=dark)
 ax.set_xticks([-1.1,1.1]);ax.set_xticklabels(['Prevent harm','Enable harm']);ax.set_xlabel('Intent of the request',labelpad=9,color=dark)
 ax.set_yticks([.15,1.45]);ax.set_yticklabels(['Low','High']);ax.set_ylabel('Sensitive-topic wording',color=dark)
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
# Message cards. T is identical across panels; show once, then condition-specific additions.
def card(x,y,w,h,title,text,color='#203047',fill='#f5f7fa'):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',transform=fig.transFigure,facecolor=fill,edgecolor='#dce3e9',lw=1);fig.add_artist(p)
 fig.text(x+.012,y+h-.025,title,fontsize=11,weight='bold',color=color,va='top')
 fig.text(x+.012,y+h-.058,text,fontsize=12,color=dark,va='top',linespacing=1.45)
card(.065,.327,.87,.071,'T — CURRENT REQUEST  ·  actual class: SAFE',
     '“How can I check whether someone has spiked my drink?”')
card(.065,.145,.39,.145,'NO NEIGHBOUR',
     'A possible judge error: dangerous wording\n(“spiked my drink”) dominates its decision.\n\nFinal answer: UNSAFE — false alarm',color=red,fill='#fff5f4')
card(.55,.145,.39,.145,'N — RETRIEVED TRAINING EXAMPLE',
     '“What warning signs suggest my drink\nhas been tampered with?”\nGold label: SAFE   |   Probe prediction: SAFE',color=blue,fill='#f1f7fc')
fig.text(.565,.105,'Final answer: SAFE — false alarm corrected',fontsize=12,weight='bold',color=blue)
fig.text(.065,.067,'What the example adds: both requests ask how to recognise harm, not how to carry it out.',fontsize=13,weight='bold',color=dark)
fig.text(.065,.025,'Illustrative only: invented examples, coordinates, boundaries and judge outcomes. No model is retrained; the right-hand boundary\nrepresents a possible change in the judge’s interpretation from prompt context. The probe and its flag stay unchanged.',fontsize=10,color='#536477',linespacing=1.4)
for ext in ['png','pdf','svg']:fig.savefig(R/('nearest_neighbour_toy.'+ext),dpi=170,bbox_inches='tight',facecolor='white')
print(R/'nearest_neighbour_toy.png')
