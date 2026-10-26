"""Full-train weighted logistic probes, stratified samples, exact cosine retrieval."""
import collections
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from .common import digest, validate_rows
from .prompt_transforms import random_index


def fit_probe(rows, features, dataset):
    validate_rows(rows)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features')
    y = np.array([r['label'] for r in rows])
    splits = np.array([r['split'] for r in rows])
    scaler = StandardScaler().fit(features[splits == 'train'])
    z = scaler.transform(features)
    model = LogisticRegression(C=.001 if dataset == 'high_stakes' else .01, class_weight='balanced', max_iter=2000, tol=1e-6, solver='lbfgs', random_state=20261004)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(z[splits == 'train'], y[splits == 'train'])
    probabilities = model.predict_proba(z)[:,1]
    precision, recall, thresholds = precision_recall_curve(y[splits == 'val'], probabilities[splits == 'val'])
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-15)
    threshold = float(thresholds[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]])
    return dict(model=model, scaler=scaler, threshold=threshold), probabilities


def select(rows, p, backbone, dataset, n=500):
    validation = np.array([r['split'] == 'val' for r in rows])
    edges = np.unique(np.quantile(p[validation], np.arange(.1,1,.1)))
    selected = []
    for split in ['val','test']:
        pools = collections.defaultdict(list)
        for i, row in enumerate(rows):
            if row['split'] == split:
                band = int(np.searchsorted(edges, p[i], side='right'))
                pools[band, row['label']].append(i)
        quota = {k:0 for k in pools}
        target = min(n, sum(map(len,pools.values())))
        while sum(quota.values()) < target:
            for key in sorted(pools):
                if sum(quota.values()) == target:
                    break
                if quota[key] < len(pools[key]):
                    quota[key] += 1
        for (band,y), pool in sorted(pools.items()):
            def rank(i):
                return digest('\0'.join(map(str,[20261009,backbone,dataset,split,band,y,rows[i]['immutable_id']])))
            pool.sort(key=lambda i: (rank(i), rows[i]['immutable_id']))
            k = quota[band,y]
            if not k:
                continue
            for i in pool[:k]:
                r = rows[i]
                selected.append(dict(index=i, sample_id=r['immutable_id'], backbone=backbone, dataset=dataset, split=split, true_label=r['label'], group=r['group'], probe_probability=float(p[i]), population_weight=len(pool)/k, inclusion_probability=k/len(pool), stratum=f'{band}:{y}'))
    return sorted(selected, key=lambda r:(r['dataset'],r['split'],r['sample_id']))


def top_two(query, training, chunk=8192):
    """Training is sorted by immutable ID. Exact-score ties keep the first ID."""
    if len(training) < 2:
        raise ValueError('At least two training neighbours required')
    best = np.full((len(query),2), -np.inf)
    ids = np.full((len(query),2), -1, dtype=int)
    for start in range(0,len(training),chunk):
        sim = query @ training[start:start+chunk].T
        if not np.isfinite(sim).all():
            raise ValueError('Nonfinite cosine')
        for _ in range(min(2,sim.shape[1])):
            ix = sim.argmax(axis=1)
            value = sim[np.arange(len(sim)),ix].copy()
            for j in range(len(query)):
                candidates = [(best[j,k],ids[j,k]) for k in range(2)] + [(value[j],start+ix[j])]
                candidates.sort(key=lambda x:(-x[0],x[1]))
                for k in range(2):
                    best[j,k],ids[j,k] = candidates[k]
            sim[np.arange(len(sim)),ix] = -np.inf
    if np.any(ids[:,0] == ids[:,1]):
        raise ValueError('Non-distinct nearest neighbours')
    return ids, best


def retrieve(rows, features, fitted, p, selected, dataset):
    train = np.array(sorted([i for i,r in enumerate(rows) if r['split']=='train'], key=lambda i:rows[i]['immutable_id']))
    scaler = fitted['scaler']
    t = scaler.transform(features[train]).astype(np.float64)
    t /= np.maximum(np.linalg.norm(t,axis=1)[:,None],1e-30)
    result = []
    def ref(i, cosine=None):
        r = rows[i]
        return dict(index=int(i), immutable_id=r['immutable_id'], messages=r['messages'], label=r['label'], prediction=int(p[i]>=fitted['threshold']), probe_probability=float(p[i]), cosine=cosine)
    for start in range(0,len(selected),64):
        group = selected[start:start+64]
        q = scaler.transform(features[[r['index'] for r in group]]).astype(np.float64)
        q /= np.maximum(np.linalg.norm(q,axis=1)[:,None],1e-30)
        ix, cosine = top_two(q,t)
        for j, r in enumerate(group):
            rand, _, _ = random_index(dataset,r['sample_id'],len(train))
            result.append(dict(r, messages=rows[r['index']]['messages'], nearest=ref(train[ix[j,0]],float(cosine[j,0])), second=ref(train[ix[j,1]],float(cosine[j,1])), random=ref(train[rand])))
    return result
