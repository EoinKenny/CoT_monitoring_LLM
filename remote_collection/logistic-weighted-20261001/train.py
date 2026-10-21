"""Balanced-class logistic regression on the frozen baseline splits and features."""
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import fcntl
import hashlib
import json
import re
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from safetensors import safe_open
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

ROOT = Path('/home/ubuntu/experiments/logistic-weighted-20261001')
CACHE = Path('/home/ubuntu/experiments/activation-full-20261001')
HUB = Path('/home/ubuntu/.cache/huggingface/hub')
ROOT.mkdir(exist_ok=True, parents=True)
LOCK = (ROOT/'run.lock').open('a')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (ROOT/'COMPLETED').exists():
    raise SystemExit('Already completed; refusing rerun.')
(ROOT/'pid.txt').write_text(str(os.getpid()))
START = time.perf_counter()
torch.set_num_threads(8)
threadpool_limits(limits=8)


def write(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def log(event, **data):
    item = dict(event=event, time=datetime.now(timezone.utc).isoformat(), **data)
    print(json.dumps(item), flush=True)
    with (ROOT/'journal.jsonl').open('a') as f:
        f.write(json.dumps(item)+'\n')
    write(ROOT/'status.json', dict(state=event, elapsed_seconds=time.perf_counter()-START, **data))


def failure(typ, exc, tb):
    write(ROOT/'FAILED.json', dict(error=str(exc), traceback=''.join(traceback.format_exception(typ,exc,tb))))
    sys.__excepthook__(typ,exc,tb)


sys.excepthook = failure
BASE = Path('/home/ubuntu/experiments/logistic-probes-20261001')
assert (BASE/'COMPLETED').exists()
config = json.loads((BASE/'config.json').read_text())
config.update(class_weight='balanced', class_weight_rule='n_train / (2 * n_class); counts from training only',
              baseline_directory=str(BASE), split_rule='reuse frozen baseline assignments exactly')
write(ROOT/'config.json',config)
data={}
for name in ['wildguard','high_stakes','toxicchat']:
    data[name]=(np.load(BASE/f'{name}_features.npy',mmap_mode='r'),
                json.loads((BASE/f'{name}_metadata.json').read_text()))


def make_splits(name,records):
    path=BASE/f'{name}_splits.json'
    saved=json.loads(path.read_text())
    assert len(saved)==len(records)
    for row,record in zip(saved,records):
        assert row['label']==record['label'] and row['input_hash']==record['input_hash']
    indices={s:np.array([r['index'] for r in saved if r['split']==s],dtype=int) for s in ['train','val','test']}
    summary={s:dict(n=len(ix),positive=sum(records[i]['label'] for i in ix)) for s,ix in indices.items()}
    summary['excluded']=dict(Counter(r['exclusion'] for r in saved if r['split']=='excluded'))
    summary['split_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    y=np.array([records[i]['label'] for i in indices['train']])
    summary['class_weights']={str(c):len(y)/(2*int((y==c).sum())) for c in [0,1]}
    write(ROOT/f'{name}_split_audit.json',summary)
    log('splits_ready',dataset=name,summary=summary)
    return indices,summary

def choose_threshold(y, probability):
    precision,recall,thresholds=precision_recall_curve(y,probability)
    scores=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    best=float(scores.max())
    winner=np.flatnonzero(np.isclose(scores,best,rtol=0,atol=1e-12))[-1]
    return float(thresholds[winner]),best


def metrics(y,p,threshold):
    pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y), positive=int(y.sum()), threshold=float(threshold),
        accuracy=float(accuracy_score(y,pred)), precision=float(precision_score(y,pred,zero_division=0)),
        recall=float(recall_score(y,pred,zero_division=0)), f1=float(f1_score(y,pred,zero_division=0)),
        auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,
        predicted_positive_fraction=float(pred.mean()), tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


selected={}
splits={}
for name,(X,records) in data.items():
    indices,summary=make_splits(name,records)
    splits[name]=(indices,summary)
    y=np.array([r['label'] for r in records],dtype=np.int64)
    scaler=StandardScaler().fit(X[indices['train']])
    train_X=scaler.transform(X[indices['train']]).astype(np.float64)
    val_X=scaler.transform(X[indices['val']]).astype(np.float64)
    trials=[]
    best=None
    for C in config['C_grid']:
        log('fitting',dataset=name,C=C)
        model=LogisticRegression(C=C,solver='lbfgs',fit_intercept=True,class_weight='balanced',
                                 max_iter=2000,tol=1e-6,random_state=20261001)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(train_X,y[indices['train']])
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught), 'Solver did not converge'
        prob=model.predict_proba(val_X)[:,1]
        threshold,score=choose_threshold(y[indices['val']],prob)
        trial=dict(C=C,threshold=threshold,validation=metrics(y[indices['val']],prob,threshold),
                   iterations=int(model.n_iter_[0]))
        trials.append(trial)
        log('validation_result',dataset=name,**trial)
        if best is None or score>best['score']+1e-12:
            best=dict(score=score,C=C,threshold=threshold,model=model)
    joblib.dump(dict(model=best['model'],scaler=scaler,threshold=best['threshold'],C=best['C'],
                    config=config),ROOT/f'{name}_probe.joblib')
    write(ROOT/f'{name}_selection.json',dict(C=best['C'],threshold=best['threshold'],trials=trials))
    selected[name]=dict(model=best['model'],scaler=scaler,C=best['C'],threshold=best['threshold'],trials=trials)

# All selections are frozen before any test-set metrics are computed.
write(ROOT/'SELECTIONS_FROZEN.json',{name:dict(C=v['C'],threshold=v['threshold']) for name,v in selected.items()})
results={}
for name,(X,records) in data.items():
    selection=selected[name]
    indices,summary=splits[name]
    ix=indices['test']
    y=np.array([records[i]['label'] for i in ix])
    p=selection['model'].predict_proba(selection['scaler'].transform(X[ix]))[:,1]
    results[name]=dict(splits=summary,C=selection['C'],threshold=selection['threshold'],
        test=metrics(y,p,selection['threshold']),test_at_05=metrics(y,p,0.5),
        validation=next(t['validation'] for t in selection['trials'] if t['C']==selection['C']),
        majority_baseline_accuracy=float(max(y.mean(),1-y.mean())))
    np.savez(ROOT/f'{name}_test_predictions.npz',indices=ix,labels=y,probabilities=p,
             predictions=p>=selection['threshold'])
    if name=='high_stakes':
        sources=np.array([records[i]['source'] for i in ix])
        results[name]['test_by_source']={s:metrics(y[sources==s],p[sources==s],selection['threshold']) for s in sorted(set(sources))}
        synthetic=np.array(['/training/' in s for s in sources])
        results[name]['synthetic_test']=metrics(y[synthetic],p[synthetic],selection['threshold'])
        results[name]['external_test']=metrics(y[~synthetic],p[~synthetic],selection['threshold'])
    results[name]['train_val_test']={}
    for split_name,split_ix in indices.items():
        split_y=np.array([records[i]['label'] for i in split_ix])
        split_p=selection['model'].predict_proba(selection['scaler'].transform(X[split_ix]))[:,1]
        results[name]['train_val_test'][split_name]=metrics(split_y,split_p,selection['threshold'])
    log('test_result',dataset=name,**results[name])
write(ROOT/'results.json',dict(configuration=config,results=results,wall_seconds=time.perf_counter()-START))
write(ROOT/'COMPLETED',dict(wall_seconds=time.perf_counter()-START,datasets=list(results)))
log('completed',wall_seconds=time.perf_counter()-START)
