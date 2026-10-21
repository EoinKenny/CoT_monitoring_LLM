"""User-authorized restart following cross-dataset numerical replay audit."""
import json,hashlib,time,shutil
from pathlib import Path
R=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009');A=R/'repair_v3';A.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=Path('/home/ubuntu/experiments/replay-diagnosis-20261009/broad');summary=json.loads((audit/'SUMMARY.json').read_text());assert summary['n']==100 and summary['same_sentence']>=95 and summary['min_cosine']>=.999 and summary['label_changes']==0
assert sha(audit/'ROWS.json')==summary['rows_sha256']
for name in ['CODE_FROZEN.json','STATUS.json','PROTOCOL.json']:shutil.copyfile(R/name,A/name)
shutil.copyfile(audit/'SUMMARY.json',A/'AUDIT_SUMMARY.json');shutil.copyfile(audit/'ROWS.json',A/'AUDIT_ROWS.json')
verified=[]
for mark in R.glob('*/judges/*/run/COMPLETED'):
 d=json.loads(mark.read_text());p=mark.parent/'judgments.jsonl';assert sha(p)==d['judgments_sha256'];verified.append(dict(path=str(p),sha256=sha(p),n=d['n']))
assert len(verified)==6 and sum(v['n'] for v in verified)==60000
for b in ['llama','qwen','gemma']:
 p=R/b/'explanations.py';s=p.read_text();(A/(b+'_explanations.py')).write_text(s);needle="assert err<.25 and proof['cosine']>=.999";assert s.count(needle)==1;s=s.replace(needle,"assert (err<.25 or proof.get('saturation_exception',False)) and proof['cosine']>=.999");compile(s,str(p),'exec');p.write_text(s)
p=R/'llama70/saliency.py';s=p.read_text();(A/'llama70_saliency.py').write_text(s)
needle="assert err<.25,(ds,r['immutable_id'],err)"
replacement="original_p=float(1/(1+np.exp(-np.clip(expected,-700,700)))); replay_p=float(1/(1+np.exp(-np.clip(float(score[r['mask']].mean()),-700,700)))); saturated=abs(original_p-replay_p)<=1e-4 and (min(original_p,replay_p)>=.999 or max(original_p,replay_p)<=.001);assert err<.25 or saturated,(ds,r['immutable_id'],err)"
assert s.count(needle)==1;s=s.replace(needle,replacement).replace("pooling_error=err))","pooling_error=err,original_probability=original_p,replayed_probability=replay_p,saturation_exception=bool(saturated)))");compile(s,str(p),'exec');p.write_text(s)
protocol=json.loads((R/'PROTOCOL.json').read_text());protocol['saliency_numerical_reproduction']['saturation_amendment']={'guard':'Retain cosine >=0.999 and original abs-logit-error <0.25. Permit larger finite logit errors ONLY if original and replay probabilities are in the same extreme tail (both >=0.999 or both <=0.001), and absolute probability difference <=0.0001 (0.01 percentage points).','audit':summary,'limitations':'99/100 audited cached Qwen examples retained the same top sentence, not perfect identity; zero decisions changed in this diagnostic sample. Replay-based saliency is numerical reproduction, not exact artifact replay. Audit is diagnostic, not a universal guarantee for other backbones. Frozen routing scores and metrics unchanged.','authorization':'User explicitly requested restart and continuation after diagnosis.'}
(R/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));(R/'REPAIR_v3.json').write_text(json.dumps(dict(time=time.time(),audit=summary,verified_judgments=verified,protocol_sha256=sha(R/'PROTOCOL.json'),preserved_failed_attempts=True),indent=2));(R/'CODE_FROZEN.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in R.rglob('*.py') if not p.is_symlink()},indent=2))
print(json.dumps(dict(verified_judgments=60000,audit=summary)),flush=True)
