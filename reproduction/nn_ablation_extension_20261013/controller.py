from pathlib import Path
import os,json,time,subprocess,fcntl,traceback,hashlib,csv,html,shutil
ROOT=Path('/home/ubuntu/experiments/nn-ablation-extension-20261013')
PREDECESSOR=ROOT.parent/'nn-ablation-replication-20261013'
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (ROOT/'COMPLETED.json').exists()
 while not (PREDECESSOR/'COMPLETED.json').exists():
  prior=read(PREDECESSOR/'STATUS.json')
  assert prior['state']!='failed','Predecessor failed; stop queue for inspection'
  write(ROOT/'STATUS.json',dict(state='waiting_for_predecessor',predecessor=str(PREDECESSOR),time=time.time()))
  time.sleep(60)
 done=read(PREDECESSOR/'COMPLETED.json');assert done['new_judgments']==140000
 for filename,h in done['analysis'].items():assert sha(PREDECESSOR/'analysis'/filename)==h
 for b,j in [('llama','qwen'),('gemma','qwen122')]:
  root=ROOT.parent/f'nn-ablation-{b}-{j}-20261013'
  assert shutil.disk_usage(ROOT).free>20*2**30
  write(ROOT/'STATUS.json',dict(state='running',pair=b+'_'+j,time=time.time()))
  with (ROOT/(b+'.log')).open('a') as log:
   p=subprocess.Popen([CPU,str(root/'code/controller.py')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,EXPANSION_ROOT=str(root)))
   write(ROOT/'CHILD.json',dict(pid=p.pid,pair=b+'_'+j,time=time.time()))
   assert p.wait()==0,f'{b} controller failed; preserved logs and outputs'
  assert read(root/'COMPLETED.json')['new_judgments']==70000
 combined=[];sources={}
 for b,j,name in [('llama70','qwen122','nn-ablation-20261013'),('qwen','qwen','nn-ablation-qwen-qwen-20261013'),('gemma','gemma','nn-ablation-gemma-gemma-20261013'),('llama','qwen','nn-ablation-llama-qwen-20261013'),('gemma','qwen122','nn-ablation-gemma-qwen122-20261013')]:
  a=ROOT.parent/name/'analysis';done=read(a/'COMPLETED.json')
  for file,h in done['files'].items():assert sha(a/file)==h
  sources[name]=sha(a/'COMPLETED.json')
  for row in read(a/'summary.json'):combined.append(dict(backbone=b,judge=j,**row))
 out=ROOT/'analysis';out.mkdir(exist_ok=True);write(out/'summary.json',combined)
 keys=['backbone','judge','condition','policy','datasets','precision','recall','f1','routing_rate','input_tokens_per_population_example']
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in combined)
 table='<table><tr>'+''.join('<th>'+k+'</th>' for k in ['Backbone','Judge','Condition','Policy','Precision','Recall','F1'])+'</tr>'
 for r in combined:table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['backbone'],r['judge'],r['condition'],r['policy'],*[f'{100*r[k]:.2f}' for k in ['precision','recall','f1']]])+'</tr>'
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NN ablation replication</title><style>body{font:16px system-ui;margin:40px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}</style><h1>NN ablations across five backbone-judge pairs</h1><p>Equal averages across ten datasets; weighted test estimates. Thresholds selected using validation only. Pairs selected for observed NN benefit: this is an exploratory mechanism study. Existing and new pair results are shown separately; cases and neighbours differ between backbones.</p>'+table+'</table>')
 write(ROOT/'COMPLETED.json',dict(time=time.time(),new_judgments=140000,reused_judgments=40000,sources=sources,analysis={p.name:sha(p) for p in out.iterdir()}))
 write(ROOT/'STATUS.json',dict(state='completed',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  x=dict(time=time.time(),state='failed',traceback=traceback.format_exc());write(ROOT/f'FAILED_{time.time_ns()}.json',x);write(ROOT/'STATUS.json',x);raise
