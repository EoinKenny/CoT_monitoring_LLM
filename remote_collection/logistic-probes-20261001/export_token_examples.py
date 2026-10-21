"""Export authentic held-out token logits and verify mean-pooling equivalence."""
import ast
import json
from pathlib import Path
import joblib
import numpy as np
from safetensors import safe_open
from scipy.special import expit, logit
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent
cache=Path('/home/ubuntu/experiments/activation-full-20261001/activations')
report=json.loads((root/'results.json').read_text())
config=report['configuration']
tokenizer=AutoTokenizer.from_pretrained('/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',local_files_only=True)
special_ids=set(tokenizer.all_special_ids)
tree=ast.parse((root/'train.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='content_mask')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<verified-content-mask>','exec'))
examples=[]
for name in ['wildguard','high_stakes','toxicchat']:
    saved=np.load(root/f'{name}_test_predictions.npz')
    records=json.loads((root/f'{name}_metadata.json').read_text())
    candidates=[j for j,i in enumerate(saved['indices']) if saved['labels'][j]==1
                and 60<=records[i]['selected_tokens']<=180
                and (name!='high_stakes' or '/training/' not in records[i]['source'])]
    assert candidates
    median=float(np.median(saved['probabilities'][candidates]))
    j=min(candidates,key=lambda j:(abs(saved['probabilities'][j]-median),int(saved['indices'][j])))
    i=int(saved['indices'][j]);record=records[i]
    fitted=joblib.load(root/f'{name}_probe.joblib')
    scaler,model=fitted['scaler'],fitted['model']
    with safe_open(str(cache/record['cache_shard']),framework='pt') as f:
        offsets=f.get_tensor('offsets').numpy();row=record['cache_row']
        start,end=int(offsets[row]),int(offsets[row+1])
        ids=f.get_tensor('token_ids')[start:end].numpy()
        acts=f.get_tensor('activations')[start:end].float().numpy()
    mask=content_mask(record['messages'],ids,name)
    weights=model.coef_[0]/scaler.scale_
    bias=float(model.intercept_[0]-weights@scaler.mean_)
    scores=acts.astype(np.float64)@weights+bias
    pooled=np.load(root/f'{name}_features.npy',mmap_mode='r')[i:i+1]
    pooled_logit=float(model.decision_function(scaler.transform(pooled))[0])
    error=abs(float(scores[mask].mean())-pooled_logit)
    assert error<1e-4, (name,error)
    assert abs(float(expit(pooled_logit))-float(saved['probabilities'][j]))<1e-8
    text=tokenizer.apply_chat_template(record['messages'],tokenize=False,add_generation_prompt=False,
                                      date_string=config['template_date_string'])
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    positions=[dict(index=t,token_id=int(ids[t]),start=int(a),end=int(b),text=text[a:b],
                    logit=float(scores[t])) for t,(a,b) in enumerate(encoded['offset_mapping']) if mask[t]]
    examples.append(dict(dataset=name,source=record['source'],id=record['id'],index=i,
        true_label=1,prediction=int(saved['predictions'][j]),probability=float(saved['probabilities'][j]),
        threshold=fitted['threshold'],threshold_logit=float(logit(fitted['threshold'])),
        mean_logit=float(scores[mask].mean()),pooled_logit=pooled_logit,
        aggregation_absolute_error=error,selected_tokens=int(mask.sum()),
        messages=record['messages'],rendered_text=text,tokens=positions,
        candidate_count=len(candidates),candidate_median_probability=median))
output=dict(selection_rule='Within each held-out test set, positive examples with 60–180 selected content tokens; choose closest to median predicted probability, without requiring a correct prediction. High-Stakes uses external test examples only.',
    logit_definition='z_t = w^T ((h_t - training_mean) / training_scale) + b; average selected z_t equals pooled example logit.',
    examples=examples)
(root/'token_examples.json').write_text(json.dumps(output,indent=2))
for e in examples:
    print(json.dumps({k:e[k] for k in ['dataset','source','id','probability','threshold','prediction','selected_tokens','aggregation_absolute_error']}))
