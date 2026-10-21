from common import *
import ast,numpy as np,csv,html
source=ROOT/'code/numerics_source.py';tree=ast.parse(source.read_text());ns={'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','curve','grid','choose','bootstrap','boot_metrics']],type_ignores=[]),str(source),'exec'),ns)
ALL_CS=['baseline','nearest']+CS
FACTORS={'baseline':(0,0,0),'nearest':(1,1,1),'text_label':(1,1,0),'text_prediction':(1,0,1),'text_only':(1,0,0),'label_only':(0,1,0),'prediction_only':(0,0,1),'label_prediction':(0,1,1)}
def main(b,j):
 reserve();out=ROOT/'analysis'/f'{b}_{j}';out.mkdir(parents=True,exist_ok=True)
 rows=read(ROOT/f'score/{b}/selected_flags.json');ids=[r['sample_id'] for r in rows];spec=read(ROOT/f'score/{b}/THRESHOLDS.json');answers={};audit={};runtime=[]
 for c in ALL_CS:
  base=PARENT if c in ['baseline','nearest'] else ROOT
  run=base/f'score/{b}/judges'/f'{j}_{c}'/'run';done=read(run/'COMPLETED');config=read(run/'config.json');jp=run/'judgments.jsonl'
  assert done['n']==10000 and sha(jp)==done['judgments_sha256'] and sha(run/'config.json')==done['config_sha256']
  rr=list(jl(jp));assert [x['sample_id'] for x in rr]==ids
  committed=[];seconds=0
  for p in sorted((run/'batches').glob('*.json')):
   if p.name.endswith(('.raw.json','.attempt.json')):continue
   x=read(p);assert sha(p.with_suffix('.raw.json'))==x['raw_sha256'];assert x['config_sha256']==done['config_sha256'];assert digest(json.dumps(x['records'],sort_keys=True))==x['records_sha256'];committed.extend(x['records']);seconds+=x['records'][0]['batch_seconds']
  assert committed==rr
  assert all(type(x['positive_probability']) is int and 0<=x['positive_probability']<=100 and x['prediction']==int(x['positive_probability']>=50) for x in rr)
  answers[c]={x['sample_id']:x for x in rr};audit[c]=dict(path=str(run),n=len(rr),judgments_sha256=sha(jp),config_sha256=sha(run/'config.json'),reused=base==PARENT)
  runtime.append(dict(condition=c,n=10000,input_tokens=sum(x['input_tokens'] for x in rr),inference_seconds=seconds,reused=base==PARENT))
 write(out/'INPUT_VERIFIED.json',audit);write(out/'runtime.json',runtime)
 result=[];draws={};fullcurves={};validation_selected={}
 for ds,s in spec.items():
  vr=[r for r in rows if r['dataset']==ds and r['split']=='val'];tr=[r for r in rows if r['dataset']==ds and r['split']=='test'];assert len(vr)==len(tr)==500
  ts=np.array(s['thresholds']);pop=s['populations']['test'];vp=s['populations']['val'];z=np.load(ROOT/f'score/{b}'/f'{ds}_predictions.npz');fulltest=z['probabilities'][z['splits']=='test'];mult=ns['bootstrap'](tr,np.random.default_rng(int(digest('20261013:'+ds)[:16],16)),2000);scores=np.array([r['probe_probability'] for r in tr]);dsresult={};arrays={}
  for c in ALL_CS:
   pv=[answers[c][r['sample_id']]['positive_probability'] for r in vr];pt=np.array([answers[c][r['sample_id']]['positive_probability'] for r in tr]);vg=ns['grid'](vr,pv,ts,vp);tg=ns['grid'](tr,pt,ts,pop)
   arrays[c+'_validation_f1']=vg['f1'];arrays[c+'_test_f1']=tg['f1']
   for policy in ['fixed50','joint']:
    cut,k=ns['choose'](vg['f1'],ts,50 if policy=='fixed50' else None);yes=(scores>=ts[k])&(pt>=cut)
    m=ns['metrics'](tg['tp'][cut,k],tg['fp'][cut,k],pop['n'],pop['positive']);boot=ns['boot_metrics'](mult,tr,yes,pop);draws[ds,c,policy]=boot
    m.update(condition=c,policy=policy,dataset=ds,probe_threshold=float(ts[k]),probability_threshold=cut,validation_f1=float(vg['f1'][cut,k]),routing_rate=float(np.mean(fulltest>=ts[k])),input_tokens_per_population_example=sum(answers[c][r['sample_id']]['input_tokens']/r['inclusion_probability'] for r in tr if r['probe_probability']>=ts[k])/pop['n'],validation_sample=500,test_sample=500,test_estimated=True,ci95={n:np.quantile(a,[.025,.975]).tolist() for n,a in boot.items()})
    result.append(m);dsresult[c+'_'+policy]=m
  candidates=list(dsresult.values());winner=max(candidates,key=lambda x:(x['validation_f1'],x['probe_threshold'],x['probability_threshold'],x['condition'],x['policy']));validation_selected[ds]=winner
  np.savez_compressed(out/f'{ds}_curves.npz',probe_thresholds=ts,**arrays)
  write(out/'STATUS.json',dict(state='evaluating',dataset=ds,completed=len(result),total=60))
 write(out/'pairs.json',result);write(out/'validation_selected.json',validation_selected)
 macro=[];comparisons=[];datasets=sorted(spec)
 for policy in ['fixed50','joint']:
  mb={}
  for c in ALL_CS:
   rr=[r for r in result if r['condition']==c and r['policy']==policy];m=dict(condition=c,policy=policy,datasets=10)
   for name in ['f1','precision','recall','routing_rate','input_tokens_per_population_example']:m[name]=float(np.mean([r[name] for r in rr]))
   mb[c]={name:np.mean([draws[ds,c,policy][name] for ds in datasets],axis=0) for name in ['f1','precision','recall']}
   m['ci95']={name:np.quantile(v,[.025,.975]).tolist() for name,v in mb[c].items()};macro.append(m)
  for a,reference in [(c,'baseline') for c in ALL_CS if c!='baseline']+[(c,'nearest') for c in CS]:
   rr_a=sorted([r for r in result if r['condition']==a and r['policy']==policy],key=lambda r:r['dataset']);rr_b=sorted([r for r in result if r['condition']==reference and r['policy']==policy],key=lambda r:r['dataset'])
   effect=dict(condition=a,reference=reference,policy=policy,metrics={})
   ix=np.random.default_rng(20261013).integers(0,10,size=(10000,10))
   for name in ['f1','precision','recall']:
    delta=np.array([x[name]-y[name] for x,y in zip(rr_a,rr_b)]);effect['metrics'][name]=dict(delta=float(delta.mean()),paired_test_ci95=np.quantile(mb[a][name]-mb[reference][name],[.025,.975]).tolist(),dataset_block_ci95=np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist())
   comparisons.append(effect)
 write(out/'summary.json',macro);write(out/'comparisons.json',comparisons)
 with (out/'summary.csv').open('w') as f:
  keys=[k for k in macro[0] if k!='ci95'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in macro)
 table='<table><tr><th>Condition</th><th>Policy</th><th>F1</th><th>Precision</th><th>Recall</th><th>Referral</th></tr>'
 for r in macro:table+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['condition'],r['policy'],*[f"{100*r[k]:.2f}%" for k in ['f1','precision','recall','routing_rate']]])+'</tr>'
 report='<!doctype html><meta charset="utf-8"><title>NN component ablation</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}</style><h1>NN component ablation</h1><p>BACKBONE backbone; JUDGE judge; 10 datasets; 500 validation and 500 test cases per dataset. One new condition (10,000 calls) and two reused parent conditions (20,000 calls).</p><p>Fixed50 tunes the probe gate on validation; joint tunes both thresholds on validation. Test metrics are weighted population estimates. Equal dataset macro averages. This study includes the complete existing backbone-judge grid.</p>'+table+'</table><p>summary.json and comparisons.json contain uncertainty and paired contrasts. Paired test bootstrap intervals condition on validation-selected thresholds; dataset-block intervals capture between-dataset variation. Confidence thresholds include zero, allowing the judge to accept every referral. Scores are not assumed calibrated. Parent results are reused with provenance; no fresh model calls were made for controls.</p>'
 (out/'report.html').write_text(report.replace('BACKBONE',b).replace('JUDGE',j))
 write(out/'COMPLETED.json',dict(time=time.time(),new_judgments=10000,reused=20000,cells=30,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETED.json'}))
if __name__=='__main__':
 import sys
 main(sys.argv[1],sys.argv[2])
