"""Verify saved collection and repair only the failed completion-marker write."""
from pathlib import Path
import sys,fcntl,shutil,json,time,importlib.util
ROOT=Path('/home/ubuntu/experiments/nn-reason-flips-20261017')
sys.path.insert(0,str(ROOT/'code'))
from common import read,sha,write,ALL,cells,reserve

lock=(ROOT/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'COMPLETED.json').exists(), 'Already finalized; no action needed'
reserve()
for name,h in read(ROOT/'CODE_FROZEN.json').items():assert sha(ROOT/name)==h,name
for name,h in read(ROOT/'ARTIFACTS_FROZEN.json').items():assert sha(ROOT/name)==h,name
for j in ALL:assert read(ROOT/'models'/j/'STATUS.json')['state']=='completed'
stamp=str(time.time_ns())
audit=ROOT/('finalization_repair_'+stamp);audit.mkdir()
for n in ['STATUS.json','COVERAGE.json']:shutil.copyfile(ROOT/n,audit/n)
spec=importlib.util.spec_from_file_location('verify_coverage',ROOT/'code/coverage.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.main()
c=read(ROOT/'COVERAGE.json')
assert c['planned']==c['accounted']==35919 and len(c['cells'])==30
for j in ALL:assert read(ROOT/'models'/j/'COMPLETED.json')['n']==sum(x['accounted'] for x in c['cells'] if x['judge']==j)
write(audit/'REPAIR.json',dict(reason='Duplicate time keyword in original controller completion-marker construction after all judges and coverage finished',verified_frozen_inputs_and_code=True,verified_full_saved_output_coverage=True,inference_calls=0,judgments_modified=False,old_failure_preserved=True,script_sha256=sha(__file__),time=time.time()))
write(ROOT/'COMPLETED.json',dict(c,coverage_sha256=sha(ROOT/'COVERAGE.json'),finalized_at=time.time(),finalization_repair=str(audit)))
write(ROOT/'STATUS.json',dict(state='completed',time=time.time(),coverage=c,finalization_repair=str(audit)))
print(json.dumps({k:v for k,v in c.items() if k!='cells'},indent=2))
