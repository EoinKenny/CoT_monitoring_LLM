"""Select stronger regularization on train/validation, then audit held-out gap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='8'
os.environ['OMP_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
import hashlib
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,precision_recall_curve
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

threadpool_limits(limits=8)
ROOT=Path('/home/ubuntu/experiments/toxicchat-l2-final-20261001')
BASE=Path('/home/ubuntu/experiments/logistic-probes-20261001')
ROOT.mkdir(parents=True,exist_ok=True)
assert not (ROOT/'trials.json').exists(), 'An existing experiment must not be silently overwritten'
START=time.perf_counter()


def write(name,value):
    path=ROOT/name
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def log(event,**kwargs):
    value=dict(event=event,**kwargs); value.setdefault('elapsed_seconds',time.perf_counter()-START)
    print(json.dumps(value),flush=True)
    write('status.json',value)


def score(y,p,threshold):
    pred=p>=threshold
    return dict(n=len(y),positive=int(y.sum()),accuracy=float(accuracy_score(y,pred)),
        f1=float(f1_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred)),auroc=float(roc_auc_score(y,p)),threshold=float(threshold))


def threshold_for(y,p):
    precision,recall,ts=precision_recall_curve(y,p)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    ix=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
    return float(ts[ix])


splits_path=BASE/'toxicchat_splits.json'
assign=json.loads(splits_path.read_text())
X=np.load(BASE/'toxicchat_features.npy')
y=np.array([r['label'] for r in assign])
indices={s:np.array([r['index'] for r in assign if r['split']==s]) for s in ['train','val','test']}
Cs=[float(v)*1e-6 for v in range(10,31)]
config=dict(class_weight='balanced',C_grid=Cs,dimensions=[2048],seed=20261001,
    split_sha256=hashlib.sha256(splits_path.read_bytes()).hexdigest(),
    selection='Highest validation F1 among models with abs(train_F1 - val_F1) <= 0.01 and val_F1 >= 0.65; ties prefer smaller C.',
    test_gate='0 < train_F1 - test_F1 <= 0.05 (absolute F1, five percentage points)',
    threshold='maximize validation F1; ties prefer higher threshold',
    transforms='Training-only standardization; L2 only, all 2048 features retained',
    scope='ToxicChat only; frozen content pooling and official cleaned test set; exploratory follow-up after inspecting previous regularized test performance')
write('config.json',config)
scaler=StandardScaler().fit(X[indices['train']])
train=scaler.transform(X[indices['train']]).astype(np.float64)
val=scaler.transform(X[indices['val']]).astype(np.float64)
trials=[];models={}
for dim in config['dimensions']:
    A,B=train,val
    for C in Cs:
        if dim==2048 and C>0.001:
            continue # Already established as overfitting; focus on stronger L2.
        model=LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',
            max_iter=3000,tol=1e-7,random_state=20261001)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(A,y[indices['train']])
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught), 'Nonconvergence'
        pv=model.predict_proba(B)[:,1]
        threshold=threshold_for(y[indices['val']],pv)
        pt=model.predict_proba(A)[:,1]
        mt=score(y[indices['train']],pt,threshold)
        mv=score(y[indices['val']],pv,threshold)
        gap=mt['f1']-mv['f1']
        trial=dict(dim=dim,C=C,threshold=threshold,train=mt,val=mv,gap=gap,
                   eligible=bool(abs(gap)<=0.01 and mv['f1']>=0.65),iterations=int(model.n_iter_[0]))
        trials.append(trial);models[(dim,C)]=model
        write('trials.json',trials)
        log('trial',**trial)
eligible=[t for t in trials if t['eligible']]
if not eligible:
    log('no_eligible_model',candidates=len(trials))
    raise SystemExit(2)
chosen=sorted(eligible,key=lambda t:(-t['val']['f1'],t['dim'],t['C']))[0]
write('SELECTION_FROZEN.json',chosen)
model=models[(chosen['dim'],chosen['C'])]
dim=chosen['dim']
standardized_weight=model.coef_[0]
standardized_bias=float(model.intercept_[0])
raw_weight=standardized_weight/scaler.scale_
raw_bias=float(standardized_bias-raw_weight@scaler.mean_)
fitted=dict(model=model,scaler=scaler,dimension=dim,
    raw_weight=raw_weight,raw_bias=raw_bias,threshold=chosen['threshold'],config=config)
joblib.dump(fitted,ROOT/'toxicchat_probe.joblib')
metrics={}
for split,ix in indices.items():
    Z=scaler.transform(X[ix]).astype(np.float64)
    p=model.predict_proba(Z)[:,1]
    # Original-space linear coefficients remain usable for token-level scores.
    assert np.max(np.abs(p-expit(X[ix].astype(np.float64)@raw_weight+raw_bias)))<1e-5
    metrics[split]=score(y[ix],p,chosen['threshold'])
    np.savez(ROOT/f'{split}_predictions.npz',indices=ix,labels=y[ix],probabilities=p,
             predictions=p>=chosen['threshold'])
gap=metrics['train']['f1']-metrics['test']['f1']
results=dict(selected=chosen,metrics=metrics,train_test_f1_gap=gap,
    relative_gap_vs_test=gap/metrics['test']['f1'],goal_met=bool(0<gap<=0.05),
    elapsed_seconds=time.perf_counter()-START)
write('results.json',results)
write('COMPLETED',dict(goal_met=results['goal_met'],elapsed_seconds=results['elapsed_seconds']))
log('completed',**results)
