import os
os.environ['MPLBACKEND']='Agg'
import csv,gzip,json,hashlib,collections,html
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'results'
ROOT=ROOT/'breakdowns'
ROOT.mkdir(exist_ok=True)
CONDS=['baseline','saliency','nearest','nearest2']
LABELS=['No explanation','Saliency','1-NN','2-NN']
COLORS=['#63758b','#cc9630','#8562aa','#247a59']
NAMES={'aegis':'Aegis','beavertails':'BeaverTails','fava':'FAVA','halueval':'HaluEval','high_stakes':'High-stakes','ragtruth':'RAGTruth','toxicchat':'ToxicChat','toxigen':'ToxiGen','wildguard':'WildGuard','wildjailbreak':'WildJailbreak','qwen':'Qwen 27B','qwen122':'Qwen 122B','gemma':'Gemma','mistral':'Mistral','olmo':'OLMo','glm':'GLM','claude4.5-sonnet':'Claude Sonnet 4.5','claude4.8-opus':'Claude Opus 4.8','claude-opus-5.5':'Claude Opus 5.5','gpt-4o':'GPT-4o','gpt-astra':'GPT Astra','gpt-5.5':'GPT-5.5'}
POLICIES={'fixed':('Probe fixed at 50; judge cutoff swept','50/50'), 'tuned':('Validation-selected probe + judge thresholds','Validation F1 optimum')}
aggregates=collections.defaultdict(lambda:np.zeros(3))
levels=collections.defaultdict(set)
for policy,name,index in [('fixed','fixed_probe_curves.csv.gz','judge_threshold'),('tuned','tradeoff_points.csv.gz','beta_index')]:
    with gzip.open(SRC/name,'rt') as f:
        for r in csv.DictReader(f):
            if policy=='fixed': assert float(r['probe_threshold'])==.5
            for dim,group in [('overall','ALL'),('dataset',r['dataset']),('model',r['model'])]:
                levels[dim].add(group)
                aggregates[policy,dim,group,r['condition'],int(r[index])]+=np.array([float(r['recall']),float(r['precision']),1])

# Verify exact marked operating points against the existing report, all groups.
checked=0
with (SRC/'summary.csv').open() as f:
    for r in csv.DictReader(f):
        if r['cohort']!='all10' or r['scope']!='combined' or r['dimension'] not in levels:continue
        if r['policy'] not in ['fixed50','average_of_four']:continue
        policy='fixed' if r['policy']=='fixed50' else 'tuned'
        marker=50 if policy=='fixed' else 20
        for c in CONDS:
            a=aggregates[policy,r['dimension'],r['group'],c,marker]
            assert np.allclose(a[:2]/a[2],[float(r[c+'_recall']),float(r[c+'_precision'])],atol=1e-12)
            checked+=1
assert checked==184,checked

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
def panel(ax,policy,dim,group):
    count=102 if policy=='fixed' else 41
    expected={'overall':600,'dataset':60,'model':50}[dim]*(4 if policy=='tuned' else 1)
    for c,label,color in zip(CONDS,LABELS,COLORS):
        rows=np.array([aggregates[policy,dim,group,c,i] for i in range(count)])
        assert np.all(rows[:,2]==expected),(policy,dim,group,c,rows[:,2])
        xy=100*rows[:,:2]/rows[:,2,None]
        ax.plot(xy[:,0],xy[:,1],color=color,lw=1.8,label=label)
        marker=50 if policy=='fixed' else 20
        ax.scatter(*xy[marker],color=color,s=42,edgecolors='white',linewidth=.7,zorder=5)
    ax.set_xlim(0,100);ax.set_ylim(0,100)
    ax.set_xticks(range(0,101,20));ax.set_yticks(range(0,101,20))
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    ax.set_xlabel('Mean test recall (%)');ax.set_ylabel('Mean test precision (%)')

def save(fig,slug):
    fig.savefig(ROOT/(slug+'.png'),bbox_inches='tight')
    fig.savefig(ROOT/(slug+'.pdf'),bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,6))
for ax,policy in zip(axes,POLICIES):
    panel(ax,policy,'overall','ALL');ax.set_title(POLICIES[policy][0],fontsize=11)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.04),frameon=False)
fig.suptitle('All 12 judges × 5 backbones × 10 datasets',fontsize=15)
fig.text(.5,.005,'Dots: exact 50/50 (left); validation F1 optimum averaged over four shared anchors (right).',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.10,1,.94]);save(fig,'overall')

for dim,ncols in [('dataset',5),('model',4)]:
    groups=sorted(levels[dim]);nrows=(len(groups)+ncols-1)//ncols
    for policy in POLICIES:
        fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*3.8,nrows*4.1+1))
        for ax,group in zip(axes.flat,groups):
            panel(ax,policy,dim,group);ax.set_title(NAMES.get(group,group),fontsize=12)
        fig.suptitle(('By dataset' if dim=='dataset' else 'By judge')+' · '+POLICIES[policy][0],fontsize=17)
        fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.025),frameon=False,fontsize=12)
        fig.text(.5,.007,'Dots: '+POLICIES[policy][1]+'. Equal-weight averages over backbones and '+('judges.' if dim=='dataset' else 'datasets.'),ha='center',fontsize=11)
        fig.tight_layout(rect=[0,.075,1,.95]);save(fig,dim+'_'+policy)
    for group in groups:
        fig,axes=plt.subplots(1,2,figsize=(12,6))
        for ax,policy in zip(axes,POLICIES):
            panel(ax,policy,dim,group);ax.set_title(POLICIES[policy][0],fontsize=11)
        fig.suptitle(NAMES.get(group,group),fontsize=16)
        fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.025),frameon=False)
        fig.tight_layout(rect=[0,.08,1,.94]);save(fig,dim+'_'+group)

with (ROOT/'curve_coordinates.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['policy','dimension','group','condition','operating_index','mean_test_recall','mean_test_precision','averaged_rows'])
    for key,a in sorted(aggregates.items()):w.writerow([*key,*(a[:2]/a[2]),int(a[2])])
receipt={'source':str(SRC),'source_sha256':{n:hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in ['tradeoff_points.csv.gz','fixed_probe_curves.csv.gz','summary.csv']},'operating_point_comparisons_passed':checked,'axes':[0,100],'new_model_calls':0,'notes':['Equal cell weighting; original design weights preserved within cells.','Tuned curves average four shared anchors: control, saliency, 1NN, 2NN; identical thresholds across all four conditions within each anchor.','41 validation-selected F-beta objectives, beta 0.1 to 10; beta=1 marked.','Fixed curve sweeps judge cutoff 0 through 101, probe held at 50; cutoff 50 marked.','Preserves existing precision=0 convention for no predicted positives, including reject-all endpoint.','No test optimisation, smoothing or extrapolation. API coverage remains judge-specific.']}
(ROOT/'VERIFIED.json').write_text(json.dumps(receipt,indent=2))
body='<h1>Experiment 1: full-range precision–recall curves</h1><p>All axes run from 0–100%. Backbones are averaged throughout. No scores, thresholds or source results have changed.</p><p><b>50/50 is one operating point, not a curve.</b> The fixed-probe curves hold the probe at 50 and sweep the judge cutoff; dots mark judge 50. Tuned curves select both cutoffs on validation for different F-beta objectives; dots mark the F1 objective. Each selected pair is shared across all four explanation conditions, then averaged across control-, saliency-, 1NN- and 2NN-based tuning anchors.</p><p>These are mean test operating curves, not test-optimised frontiers. No interpolation beyond the observed range. The fixed-probe recall ceiling can be below 100%. The reject-all endpoint uses the existing precision=0 convention (precision is mathematically undefined with no positive predictions), which explains the line towards the origin.</p>'
for slug,title in [('overall','Overall average'),('dataset_fixed','By dataset: fixed probe'),('dataset_tuned','By dataset: validation tuning'),('model_fixed','By judge: fixed probe'),('model_tuned','By judge: validation tuning')]:
    body+=f'<h2>{title}</h2><a href="{slug}.png"><img src="{slug}.png"></a><p><a href="{slug}.pdf">Vector PDF</a> · <a href="{slug}.png">Full-size PNG</a></p>'
body+='<h2>Individual side-by-side comparisons</h2>'
for dim in ['dataset','model']:
    body+='<p>'+ ' · '.join(f'<a href="{dim}_{g}.png">{html.escape(NAMES.get(g,g))}</a>' for g in sorted(levels[dim]))+'</p>'
body+='<p>Overall: 600 cells. Each dataset: 60 cells. Each judge: 50 cells. All are macro averages, with frozen sampling weights inside each cell. API judges retain their own available cases, matched across explanation conditions within judge.</p><p><a href="curve_coordinates.csv">Plot coordinates</a> · <a href="VERIFIED.json">Verification and source checksums</a></p>'
(ROOT/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Experiment 1 PR breakdowns</title><style>body{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#182738;background:#fafbfd}img{width:100%;background:white}a{color:#265e9d}h2{margin-top:40px}</style>'+body+'</html>')
print(json.dumps({'report':str(ROOT/'report.html'),'checked':checked,'figures':len(list(ROOT.glob('*.png')))}))
