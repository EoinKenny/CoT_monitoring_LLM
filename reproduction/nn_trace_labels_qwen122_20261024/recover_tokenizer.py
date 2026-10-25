from pathlib import Path
import json,time,subprocess,fcntl,os
R=Path('/home/ubuntu/experiments/nn-trace-labels-qwen122-20261024')
with (R/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'RECOVERY_LAUNCH.json').exists()
 pid=json.loads((R/'LAUNCH.json').read_text())['pid']
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:
  state=Path(f'/proc/{pid}/stat').read_text().split()[2];assert state=='Z',state
 failures=[json.loads(p.read_text()) for p in R.glob('FAILED_*.json')];assert any("min_input_id < 0" in x['traceback'] and "'str' and 'int'" in x['traceback'] for x in failures)
 assert not list((R/'smoke').glob('*.raw.json')) and not (R/'batches').exists()
 a=R/'smoke/000000.attempt.json';assert a.exists()
 archive=R/'failed_prevalidation';archive.mkdir();a.rename(archive/a.name)
 (R/'RUN_CONFIG.json').rename(archive/'RUN_CONFIG.json')
 (archive/'AUDIT.json').write_text(json.dumps(dict(reason='Transformers apply_chat_template returned mapping; vLLM rejected string keys as token IDs in input validation before scheduling generation. Use rendered string plus encode to produce integer list. Original attempt retained; no real units attempted.',time=time.time()),indent=2))
 log=(R/'run_v2.log').open('ab');p=subprocess.Popen(['/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/bin/python','-u',str(R/'run_v2.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=p.pid,time=time.time(),reason='Known pre-inference tokenizer input rejection repaired; original logs and attempt archived');(R/'RECOVERY_LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
