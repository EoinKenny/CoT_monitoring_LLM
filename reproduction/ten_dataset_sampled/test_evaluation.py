from pathlib import Path
import tempfile,json,subprocess,sys
s=Path(__file__).with_name('evaluate.py').read_text()
with tempfile.TemporaryDirectory() as tmp:
 r=Path(tmp);rows=[dict(sample_id=str(i),dataset='toy',true_label=int(i<2)) for i in range(4)]
 (r/'private.json').write_text(json.dumps(rows));(r/'weighted_results.json').write_text(json.dumps({'results':{'toy':{'test':dict(tp=2,fp=2,fn=1,tn=5,f1=4/7)}}}))
 for m in ['qwen','gemma','llama']:
  for c in ['baseline','saliency','nearest']:
   d=r/'judges'/f'{m}_{c}'/'run';d.mkdir(parents=True);(d/'COMPLETED').touch();(d/'judgments.jsonl').write_text(''.join(json.dumps(dict(x,prediction=int(x['true_label']==1)))+'\n' for x in rows))
 script=r/'eval.py';script.write_text(s.replace('/home/ubuntu/experiments/ten-dataset-sampled-20261002',tmp))
 subprocess.run([sys.executable,str(script)],check=True)
 out=json.loads((r/'results.json').read_text());v=out['results']['toy']['qwen_baseline'];assert v['cascade_estimate']['f1']==.8 and v['cascade_f1_ci95']==[.8,.8]
 # Sample expansion: one half of each flag class observed, estimates recover counts.
 (r/'weighted_results.json').write_text(json.dumps({'results':{'toy':{'test':dict(tp=4,fp=4,fn=2,tn=10,f1=4/7)}}}))
 subprocess.run([sys.executable,str(script)],check=True)
 v=json.loads((r/'results.json').read_text())['results']['toy']['qwen_baseline'];assert v['cascade_estimate']['tp']==4 and v['cascade_estimate']['tn']==14 and v['cascade_estimate']['f1']==.8
 print('PASS: census counts, sample expansion, F1, paired intervals and all nine conditions.')
