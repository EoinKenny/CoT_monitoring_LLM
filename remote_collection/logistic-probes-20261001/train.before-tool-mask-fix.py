"""Pool cached activations; split, fit, validate thresholds, then evaluate once."""
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

ROOT = Path('/home/ubuntu/experiments/logistic-probes-20261001')
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
assert (CACHE/'COMPLETED').exists()
manifest = json.loads((CACHE/'manifest.json').read_text())
config = dict(activation_config=manifest['configuration'], sklearn=sklearn.__version__,
    seed=20261001, C_grid=[0.001,0.01,0.1,1.0,10.0], penalty='L2', class_weight=None,
    fit_intercept=True, scaler='StandardScaler fitted to training only', solver='lbfgs',
    max_iter=2000, tolerance=1e-6,
    threshold_rule='maximize validation F1 over all unique probabilities; ties use highest threshold',
    model_selection='maximize validation F1 after threshold tuning; ties choose smaller C',
    pooling='mean of content tokens; exclude all special IDs and template-only tokens',
    wildguard_tokens='assistant response only', toxicchat_tokens='user content only',
    high_stakes_tokens='all supplied message content, including supplied system context',
    split_rule='official test preserved except audited exclusions; high-stakes official validation; others 20% stratified grouped training holdout',
    grouping='identical complete inputs; WildGuard prompt; ToxicChat conversation; high-stakes source ID and synthetic pair ID',
    duplicate_rule='exact input conflicts excluded; exact duplicates keep test > validation > train; cross-split groups keep test > validation > train')
write(ROOT/'config.json', config)
tokenizer = AutoTokenizer.from_pretrained(HUB/'models--meta-llama--Llama-3.2-1B-Instruct'/'snapshots'/
    manifest['configuration']['revisions']['meta-llama/Llama-3.2-1B-Instruct'], local_files_only=True)
special_ids = set(tokenizer.all_special_ids)


def content_mask(messages, stored_ids, dataset):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == stored_ids.tolist(), 'Cached tokens do not match reconstruction'
    spans = []
    for j, message in enumerate(messages):
        if dataset == 'wildguard' and message['role'] != 'assistant':
            continue
        if dataset == 'toxicchat' and message['role'] != 'user':
            continue
        content = message['content'].strip()
        if not content:
            continue
        prefix = tokenizer.apply_chat_template(messages[:j+1], tokenize=False, add_generation_prompt=False)
        assert text.startswith(prefix), 'Chat template prefix changes with later messages'
        begin = prefix.rfind(content)
        assert begin >= 0, 'Message content not found verbatim in template'
        spans.append((begin, begin+len(content)))
    mask = np.array([token not in special_ids and end > begin and
        any(end > a and begin < b for a,b in spans)
        for token,(begin,end) in zip(encoded['input_ids'],encoded['offset_mapping'])])
    assert not any(int(t) in special_ids for t in stored_ids[mask])
    return mask


data = {}
for name in ['wildguard','high_stakes','toxicchat']:
    feature_path = ROOT/f'{name}_features.npy'
    metadata_path = ROOT/f'{name}_metadata.json'
    if feature_path.exists() or metadata_path.exists():
        assert feature_path.exists() and metadata_path.exists(), 'Incomplete pooled cache requires inspection'
        data[name] = (np.load(feature_path), json.loads(metadata_path.read_text()))
        log('pooled_cache_loaded', dataset=name, examples=len(data[name][1]))
        continue
    log('pooling', dataset=name)
    features, records = [], []
    shards = [s for s in manifest['shards'] if s['dataset']==name]
    for shard_idx, shard in enumerate(shards):
        path = CACHE/'activations'/shard['file']
        meta = json.loads(path.with_suffix('.json').read_text())
        assert path.stat().st_size == shard['bytes'] and meta['verified']
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''):
                h.update(block)
        assert h.hexdigest() == shard['sha256'], 'Activation file checksum mismatch'
        with safe_open(str(path), framework='pt') as f:
            acts = f.get_tensor('activations')
            ids = f.get_tensor('token_ids').numpy()
            offsets = f.get_tensor('offsets').numpy()
            for j,sample in enumerate(meta['samples']):
                a,b = int(offsets[j]),int(offsets[j+1])
                selected = content_mask(sample['messages'], ids[a:b], name)
                if selected.any():
                    vector = acts[a:b][torch.from_numpy(selected)].float().mean(0).numpy()
                    assert np.isfinite(vector).all()
                else:
                    vector = np.zeros(2048, dtype=np.float32)
                features.append(vector)
                label = int(sample['label'] == ('harmful' if name=='wildguard' else 'high-stakes')) if name!='toxicchat' else int(float(sample['label']))
                record = dict(sample, label=label, selected_tokens=int(selected.sum()),
                    all_tokens=b-a, input_hash=hashlib.sha256(ids[a:b].tobytes()).hexdigest(),
                    cache_shard=shard['file'], cache_row=j)
                records.append(record)
        if (shard_idx+1)%50==0:
            log('pooling', dataset=name, shards_done=shard_idx+1, shards_total=len(shards), examples=len(records))
    matrix = np.stack(features).astype(np.float32)
    assert len(records) == sum(s['examples'] for s in shards)
    with feature_path.with_suffix('.tmp').open('wb') as f:
        np.save(f, matrix)
    os.replace(feature_path.with_suffix('.tmp'), feature_path)
    write(metadata_path, records)
    data[name] = (matrix, records)
    log('pooling_complete', dataset=name, examples=len(records),
        content_tokens=sum(r['selected_tokens'] for r in records), empty=sum(r['selected_tokens']==0 for r in records))

source_pair = {}
rev = manifest['configuration']['revisions']['Arrrlex/models-under-pressure']
for split in ['train','test']:
    path=HUB/'datasets--Arrrlex--models-under-pressure'/'snapshots'/rev/'training'/f'{split}.jsonl'
    for line in path.read_text().splitlines():
        r=json.loads(line)
        source_pair[(f'high_stakes/training/{split}.jsonl',str(r['ids']))] = r.get('pair_id')


def native_split(record):
    source=record['source']
    if '/test' in source or source.endswith('_test.parquet'):
        return 'test'
    if '/validation' in source:
        return 'val'
    return 'train'


def norm(text):
    return re.sub(r'\s+',' ',text).strip().casefold()


def make_splits(name, records):
    n=len(records)
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(a,b):
        parent[root(b)]=root(a)
    seen={}
    for i,r in enumerate(records):
        keys=['input:'+r['input_hash']]
        if name=='wildguard':
            keys.append('prompt:'+norm(r['messages'][0]['content']))
        elif name=='toxicchat':
            keys.append('conversation:'+r['id'])
            keys.append('user:'+norm(r['messages'][0]['content']))
        else:
            domain=r['source'].split('/')[1]
            ident=re.sub(r'_(chosen|rejected)$','',r['id'])
            keys.append('source:'+domain+':'+ident)
            pair=source_pair.get((r['source'],r['id']))
            if pair:
                keys.append('synthetic_pair:'+str(pair))
            # Catch source duplicates whose system prompts or formatting differ.
            users=[norm(m['content']) for m in r['messages'] if m['role']=='user']
            if users:
                keys.append('user_content:'+json.dumps(users))
        for key in keys:
            if key in seen:
                union(i,seen[key])
            else:
                seen[key]=i
    groups=np.array([root(i) for i in range(n)])
    split=np.array([native_split(r) for r in records], dtype=object)
    excluded={i:'no_content_tokens' for i,r in enumerate(records) if not r['selected_tokens']}
    identical=defaultdict(list)
    for i,r in enumerate(records):
        identical[r['input_hash']].append(i)
    priority={'train':0,'val':1,'test':2}
    for inds in identical.values():
        if len({records[i]['label'] for i in inds})>1:
            for i in inds: excluded[i]='conflicting_labels_for_identical_input'
        elif len(inds)>1:
            keep=sorted(inds,key=lambda i:(-priority[split[i]], records[i]['dataset_index']))[0]
            for i in inds:
                if i!=keep: excluded[i]='exact_duplicate'
    members=defaultdict(list)
    for i,g in enumerate(groups):
        if i not in excluded: members[int(g)].append(i)
    for inds in members.values():
        highest=max(priority[split[i]] for i in inds)
        for i in inds:
            if priority[split[i]]<highest: excluded[i]='group_overlaps_higher_priority_split'
    if name!='high_stakes':
        available=np.array([i for i in range(n) if i not in excluded and split[i]=='train'])
        y=np.array([records[i]['label'] for i in available])
        folds=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261001)
        _,val=next(folds.split(np.zeros(len(available)),y,groups[available]))
        split[available[val]]='val'
    for i in excluded: split[i]='excluded'
    indices={s:np.array([i for i in range(n) if split[i]==s],dtype=int) for s in ['train','val','test']}
    for s in indices:
        assert len(indices[s]) and len({records[i]['label'] for i in indices[s]})==2
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(groups[indices[a]])&set(groups[indices[b]])
    assignments=[dict(index=i, original_index=r['dataset_index'], source=r['source'], id=r['id'],
        label=r['label'], input_hash=r['input_hash'], group=int(groups[i]), split=split[i],
        exclusion=excluded.get(i)) for i,r in enumerate(records)]
    write(ROOT/f'{name}_splits.json', assignments)
    summary={s:dict(n=len(ix),positive=sum(records[i]['label'] for i in ix)) for s,ix in indices.items()}
    summary['excluded']=dict(Counter(excluded.values()))
    log('splits_ready', dataset=name, summary=summary)
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
        model=LogisticRegression(C=C,solver='lbfgs',fit_intercept=True,
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
    log('test_result',dataset=name,**results[name])
write(ROOT/'results.json',dict(configuration=config,results=results,wall_seconds=time.perf_counter()-START))
write(ROOT/'COMPLETED',dict(wall_seconds=time.perf_counter()-START,datasets=list(results)))
log('completed',wall_seconds=time.perf_counter()-START)
