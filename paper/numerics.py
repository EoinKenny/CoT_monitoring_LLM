"""Frozen design-weighted metrics and validation threshold selection."""
import numpy as np

def metrics(tp, fp, n, pos):
    tp = float(tp)
    fp = float(fp)
    return dict(tp=tp, fp=fp, fn=float(pos - tp), tn=float(n - pos - fp), precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / pos if pos else 0.0, f1=2 * tp / (pos + tp + fp) if pos + tp + fp else 0.0, accuracy=(tp + n - pos - fp) / n)

def curve(rows, pred, ts, pop):
    score = np.array([r['probe_probability'] for r in rows])
    y = np.array([r['true_label'] for r in rows])
    w = np.array([1 / r['inclusion_probability'] for r in rows])
    ix = np.argsort(score, kind='stable')
    at = np.searchsorted(score[ix], ts, side='left')
    out = []
    for gold in [1, 0]:
        cumulative = np.r_[0.0, np.cumsum((w * (y == gold) * pred)[ix])]
        out.append(cumulative[-1] - cumulative[at])
    (tp, fp) = out
    den = pop['positive'] + tp + fp
    return dict(tp=tp, fp=fp, f1=np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0))

def grid(rows, prob, ts, pop):
    cc = [curve(rows, np.asarray(prob) >= cut, ts, pop) for cut in range(102)]
    return {name: np.stack([c[name] for c in cc]) for name in ['tp', 'fp', 'f1']}

def choose(f1, ts, fixed=None):
    if fixed is not None:
        k = int(np.flatnonzero(np.isclose(f1[fixed], f1[fixed].max(), rtol=0, atol=1e-12))[-1])
        return (fixed, k)
    positions = np.argwhere(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    (cut, k) = max(positions, key=lambda a: (ts[a[1]], a[0]))
    return (int(cut), int(k))
