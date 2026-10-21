"""Compare frozen class-weighted probes with the original unweighted probes."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

root=Path(__file__).resolve().parent
base=Path(sys.argv[1])
w=json.loads((root/'results.json').read_text())
b=json.loads((base/'results.json').read_text())
labels={'wildguard':'WildGuard','high_stakes':'High-Stakes','toxicchat':'ToxicChat'}
lines=['# Class-weighted logistic probe comparison','',
       'Same frozen feature matrices and split assignments as the unweighted baseline. Only the logistic loss class weights change. L2 regularization and threshold are independently selected on validation for each method, using the same C grid and F1 criterion. No oversampling or focal loss is used.','',
       '| Dataset | Unweighted test accuracy | Weighted test accuracy | Unweighted test F1 | Weighted test F1 | Weighted precision | Weighted recall | Unweighted AUROC | Weighted AUROC |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
fig,axes=plt.subplots(1,3,figsize=(15,5.2))
for ax,(name,title) in zip(axes,labels.items()):
    old=b['results'][name]['test'];new=w['results'][name]['test']
    frozen=base/f'{name}_splits.json'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==w['results'][name]['splits']['split_sha256']
    lines.append(f"| {title} | {old['accuracy']:.2%} | {new['accuracy']:.2%} | {old['f1']:.2%} | {new['f1']:.2%} | {new['precision']:.2%} | {new['recall']:.2%} | {old['auroc']:.4f} | {new['auroc']:.4f} |")
    baseline_preds=np.load(base/f'{name}_test_predictions.npz')
    weighted_preds=np.load(root/f'{name}_test_predictions.npz')
    assert np.array_equal(baseline_preds['indices'],weighted_preds['indices'])
    assert np.array_equal(baseline_preds['labels'],weighted_preds['labels'])
    for pred,m,label,color,style in [(baseline_preds,old,'Unweighted','#64748b','--'),
                                    (weighted_preds,new,'Weighted','#2563eb','-')]:
        fpr,tpr,_=roc_curve(pred['labels'],pred['probabilities'])
        assert abs(roc_auc_score(pred['labels'],pred['probabilities'])-m['auroc'])<1e-12
        ax.plot(fpr,tpr,color=color,ls=style,lw=2,label=f'{label}: {m["auroc"]:.3f}')
        ax.scatter(m['fp']/(m['fp']+m['tn']),m['recall'],color=color,s=40,zorder=3)
    ax.plot([0,1],[0,1],color='#d0d4da',ls=':',lw=1)
    ax.set(title=title+(' (combined)' if name=='high_stakes' else ''),xlabel='False positive rate',ylabel='True positive rate',xlim=(-.01,1.01),ylim=(-.01,1.01))
    ax.set_aspect('equal');ax.grid(color='#e8ebef',lw=.7);ax.legend(loc='lower right',title='Test AUROC')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Class weighting: held-out ROC comparison',fontsize=16)
fig.text(.5,.015,'Dots: thresholds selected separately to maximize validation F1. Test examples are identical across methods.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.10,1,.94))
fig.savefig(root/'weighted_roc_comparison.png',dpi=180)
fig.savefig(root/'weighted_roc_comparison.pdf')
plt.close(fig)
lines+=['','## Weighted train / validation / test performance','',
        '| Dataset | Split | Accuracy | F1 | Precision | Recall | AUROC |','|---|---|---:|---:|---:|---:|---:|']
for name,r in w['results'].items():
    for split,m in r['train_val_test'].items():
        lines.append(f"| {labels[name]} | {split} | {m['accuracy']:.2%} | {m['f1']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['auroc']:.4f} |")
lines+=['','## Class weights and selected parameters','',
        'Weights are n_train / (2 × n_class), calculated from training labels only. Weighted sigmoid outputs should be treated as scores, not automatically as calibrated probabilities under the original class prevalence.','',
        '| Dataset | Negative weight | Positive weight | Positive / negative weight ratio | Selected C | Selected threshold |',
        '|---|---:|---:|---:|---:|---:|']
for name,r in w['results'].items():
    weights=r['splits']['class_weights']
    lines.append(f"| {labels[name]} | {weights['0']:.6f} | {weights['1']:.6f} | {weights['1']/weights['0']:.3f} | {r['C']:g} | {r['threshold']:.6f} |")
lines+=['','## High-Stakes external test only','']
for method,report in [('Unweighted',b),('Weighted',w)]:
    m=report['results']['high_stakes']['external_test']
    lines.append(f"- {method}: accuracy {m['accuracy']:.2%}, F1 {m['f1']:.2%}, AUROC {m['auroc']:.4f}.")
lines+=['','No models were selected using test scores. This is an exploratory paired comparison on the same already-inspected test sets, not a new independent confirmation set. Baseline models and their earlier figures are preserved separately.','']
(root/'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines[:11]))
