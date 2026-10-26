"""One GPU-owning subprocess per backbone. No historical features are loaded."""
import argparse
import gc
import importlib.metadata
import os
from pathlib import Path

from .common import ROOT, commit, completed, digest, frozen, log, read, records, reserve, sha, smoke_rows, validate_rows, write, write_records


def run(out, backbone):
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    from transformers import AutoTokenizer
    from .models import batches, forward, load_backbone, model_path
    from .probes import fit_probe, retrieve, select
    from .tokenization import sentence_scores, tokenize_missing
    threadpool_limits(8)
    config = read(out/'RUN.json')
    root = out/'backbones'/backbone
    root.mkdir(parents=True, exist_ok=True)
    path, spec = model_path('backbone', backbone, config['model_cache'], config['model_paths'])
    provenance = dict(spec=spec, config_sha256=sha(path/'config.json'), packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','scipy','scikit-learn','safetensors']})
    frozen(root/'ENVIRONMENT.json', provenance)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 128009
    model = None
    for ds in config['datasets']:
        dest = root/ds
        dest.mkdir(parents=True, exist_ok=True)
        source = Path(config['inputs_dir'])/f'{backbone}__{ds}.jsonl.gz'
        inputs = dict(dataset_sha256=sha(source), environment_sha256=sha(root/'ENVIRONMENT.json'), smoke=config['smoke'], sample_n=config['sample_n'])
        if completed(dest/'COMPLETED.json', inputs):
            for name in ['FEATURES.json','PROBE.json','NEIGHBOURS.json']:
                marker = dest/name
                completed(marker,read(marker)['inputs'])
            log(out, 'backbone_dataset_already_complete', backbone=backbone, dataset=ds)
            continue
        rows = list(records(source))
        validate_rows(rows)
        if config['smoke']:
            rows = smoke_rows(rows)
        # Some original Llama records predate persisted input IDs. Reconstruct
        # their exact native content mask; never read the old activation caches.
        rows = tokenize_missing(tok, rows, backbone)
        features = dest/'features.npy'
        feature_inputs = dict(inputs, ids_sha256=digest('\n'.join(r['immutable_id'] for r in rows)))
        if not completed(dest/'FEATURES.json', feature_inputs):
            reserve(out, len(rows)*spec['hidden_size']*4)
            cache = dest/'feature_batches'
            cache.mkdir(exist_ok=True)
            chunks = list(batches(rows, spec['batch_size'], spec['batch_tokens']))
            for batch_id, ix in enumerate(chunks):
                f = cache/f'{batch_id:07d}.npz'
                mark = f.with_suffix('.json')
                batch_input = dict(feature_inputs, indices=ix)
                if completed(mark, batch_input):
                    continue
                reserve(out)
                if model is None:
                    log(out, 'loading_backbone', backbone=backbone)
                    model = load_backbone(backbone, path, spec, root)
                values, hidden = forward(model, [rows[i] for i in ix], pad)
                del hidden
                tmp = f.with_suffix('.tmp')
                with tmp.open('wb') as h:
                    np.savez(h, indices=np.array(ix), features=values)
                    h.flush()
                    os.fsync(h.fileno())
                tmp.replace(f)
                commit(mark, batch_input, [f])
                if batch_id % 100 == 0:
                    log(out, 'extracting_activations', backbone=backbone, dataset=ds, batch=batch_id+1, batches=len(chunks))
            tmp = features.with_suffix('.tmp.npy')
            x = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(len(rows),spec['hidden_size']))
            for batch_id, ix in enumerate(chunks):
                with np.load(cache/f'{batch_id:07d}.npz') as chunk:
                    if chunk['indices'].tolist() != ix:
                        raise ValueError('Feature-batch membership changed')
                    x[ix] = chunk['features']
            x.flush()
            del x
            tmp.replace(features)
            commit(dest/'FEATURES.json', feature_inputs, [features])
        x = np.load(features, mmap_mode='r')
        probe_inputs = dict(feature_inputs, features_sha256=sha(features))
        if not completed(dest/'PROBE.json', probe_inputs):
            log(out, 'fitting_probe', backbone=backbone, dataset=ds, rows=len(rows))
            fitted, p = fit_probe(rows, x, ds)
            joblib.dump(fitted, dest/'probe.joblib')
            np.savez(dest/'predictions.npz', probabilities=p, labels=np.array([r['label'] for r in rows]), splits=np.array([r['split'] for r in rows]))
            selected = select(rows,p,backbone,ds,config['sample_n'])
            write(dest/'selected.json', selected)
            counts = {s: {str(y):sum(r['split']==s and r['label']==y for r in rows) for y in [0,1]} for s in ['train','val','test']}
            write(dest/'population.json', counts)
            commit(dest/'PROBE.json', probe_inputs, [dest/'probe.joblib',dest/'predictions.npz',dest/'selected.json',dest/'population.json'], threshold=fitted['threshold'])
        fitted = joblib.load(dest/'probe.joblib')
        p = np.load(dest/'predictions.npz')['probabilities']
        selected = read(dest/'selected.json')
        retrieval_inputs = dict(probe_sha256=sha(dest/'probe.joblib'), selected_sha256=sha(dest/'selected.json'), features_sha256=sha(features))
        if not completed(dest/'NEIGHBOURS.json', retrieval_inputs):
            log(out, 'retrieving_neighbours', backbone=backbone, dataset=ds, targets=len(selected))
            refs = retrieve(rows,x,fitted,p,selected,ds)
            write_records(dest/'references.jsonl',refs)
            commit(dest/'NEIGHBOURS.json',retrieval_inputs,[dest/'references.jsonl'])
        refs = list(records(dest/'references.jsonl'))
        w = fitted['model'].coef_[0]/fitted['scaler'].scale_
        bias = float(fitted['model'].intercept_[0] - w @ fitted['scaler'].mean_)
        saliency_dir = dest/'saliency'
        saliency_dir.mkdir(exist_ok=True)
        for j,r in enumerate(refs):
            f = saliency_dir/(digest(r['sample_id'])+'.json')
            if f.exists():
                value = read(f)
                if value['inputs'] != retrieval_inputs or value['sample_id'] != r['sample_id']:
                    raise ValueError('Saliency provenance changed')
            else:
                reserve(out)
                if model is None:
                    log(out, 'loading_backbone', backbone=backbone)
                    model = load_backbone(backbone,path,spec,root)
                row = rows[r['index']]
                _, hidden = forward(model,[row],pad)
                scores = hidden[0,:row['n_tokens']].float().cpu().numpy().astype(np.float64) @ w + bias
                del hidden
                expected = float(fitted['model'].decision_function(fitted['scaler'].transform(x[r['index']:r['index']+1]))[0])
                error = abs(float(scores[row['mask']].mean())-expected)
                # Numerical replay differences are diagnostics, never agreement gates.
                if not np.isfinite(scores).all() or not np.isfinite(error):
                    raise ValueError('Nonfinite saliency')
                value = dict(inputs=retrieval_inputs,sample_id=r['sample_id'],saliency=sentence_scores(tok,row,backbone,scores),pooling_error=error)
                write(f,value)
            r.update(saliency=value['saliency'],pooling_error=value['pooling_error'])
            if j % 100 == 0:
                log(out,'saliency',backbone=backbone,dataset=ds,completed=j+1,total=len(refs))
        write_records(dest/'explanations.jsonl',refs)
        commit(dest/'COMPLETED.json',inputs,[dest/'FEATURES.json',dest/'PROBE.json',dest/'NEIGHBOURS.json',dest/'explanations.jsonl'],n=len(refs),max_pooling_error=max(r['pooling_error'] for r in refs))
        log(out,'backbone_dataset_complete',backbone=backbone,dataset=ds,targets=len(refs))
        del rows, x, refs, fitted, p
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backbone', required=True)
    args = parser.parse_args()
    run(args.out,args.backbone)
