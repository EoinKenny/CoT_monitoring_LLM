from pathlib import Path
import subprocess,json,hashlib,os,time
cpu='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for b in ['qwen','gemma']:
 root=Path(f'/home/ubuntu/experiments/nn-ablation-{b}-{b}-20261013')
 assert not (root/'FROZEN.json').exists()
 setup=json.loads((root/'SETUP_COMPLETED.json').read_text())
 assert setup['n']==70000 and setup['random_references']==10000
 assert setup['canonical_sha256']==sha(root/'canonical/cases.jsonl')
 logs={}
 for script in ['test_contract.py','test_evaluation.py']:
  result=subprocess.run([cpu,str(root/'code'/script)],env=dict(os.environ,EXPANSION_ROOT=str(root)),capture_output=True,text=True)
  logs[script]=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
  print(b,script,result.returncode,result.stdout,result.stderr,flush=True)
  assert result.returncode==0
 # Canonical cases: unique prompt ids, exact original ordered membership in all seven conditions.
 rows=[json.loads(l) for l in (root/'canonical/cases.jsonl').open()]
 expected=[r['sample_id'] for r in json.loads((root/f'score/{b}/selected_flags.json').read_text())]
 assert len(set(r['prompt_id'] for r in rows))==70000
 for c in sorted({r['condition'] for r in rows}):assert [r['sample_id'] for r in rows if r['condition']==c]==expected
 (root/'TESTS_PASSED.json').write_text(json.dumps(dict(time=time.time(),tests=logs,coverage=70000),indent=2))
 files=[*sorted((root/'code').glob('*.py')),*sorted(root.glob('*.json')),root/'canonical/PREPARED.json',*sorted((root/f'score/{b}').glob('*.json'))]
 frozen={str(p.relative_to(root)):sha(p) for p in files}
 (root/'FROZEN.json').write_text(json.dumps(frozen,indent=2))
 print('FROZEN',root,len(frozen),flush=True)
