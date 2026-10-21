from pathlib import Path
import json
r=Path(__file__).resolve().parent;src=r.parent/'nn_ablation_20261013'
selection=json.loads((r.parent/'nn_ablation_replication_20261013/SELECTION.json').read_text())
selection['rule']='Next two remaining backbone-judge combinations by equal-dataset macro test F1 gain, using parent validation-joint-tuned nearest versus baseline; exclude existing llama70/qwen122, qwen/qwen and gemma/gemma ablations. No distinct-judge constraint. Exploratory mechanism follow-up.'
selection['selected']=[x for x in selection['ranking'] if (x['backbone'],x['judge']) in [('llama','qwen'),('gemma','qwen122')]]
(r/'SELECTION.json').write_text(json.dumps(selection,indent=2))
for b,j,title in [('llama','qwen','Llama 3.2 1B backbone; Qwen3.6-27B judge'),('gemma','qwen122','Gemma4-31B backbone; Qwen3.5-122B judge')]:
 d=r/f'{b}_{j}';(d/'code').mkdir(parents=True,exist_ok=True)
 remote=f'/home/ubuntu/experiments/nn-ablation-{b}-{j}-20261013'
 for p in (src/'code').glob('*.py'):
  s=p.read_text()
  if p.name!='numerics_source.py':
   s=s.replace('/home/ubuntu/experiments/nn-ablation-20261013',remote).replace('llama70',b).replace('qwen122',j)
   s=s.replace('Llama 3.3 70B backbone; Qwen 3.5 122B judge',title)
   s=s.replace('The pair was chosen for the largest observed parent NN improvement','The pair was chosen among the next largest observed parent NN improvements')
  if p.name=='setup.py':
   s=s.replace("expl={r['sample_id']:r for r in read(SOURCE/'judge_inputs'/f'{ds}_explanations.json')['records']}","ef=SOURCE/'judge_inputs/explanations.jsonl'; proof=read(SOURCE/'judge_inputs/EXTRACTION_COMPLETED'); assert sha(ef)==proof['explanations_sha256']; source_hashes[str(ef)]=sha(ef)\n  expl={r['sample_id']:r for r in jl(ef) if r['dataset']==ds}")
   s=s.replace("assert meta[nn['index']]['immutable_id']==nn['immutable_id'];", "assert meta[nn['index']]['id']==nn['id']; assert meta[nn['index']]['messages']==nn['messages']; nn['immutable_id']=meta[nn['index']]['immutable_id'];")
  (d/'code'/p.name).write_text(s)
 protocol=json.loads((src/'PROTOCOL.json').read_text());protocol.update(study='NN component ablation extension',backbone=b,judge=j,selection_provenance=selection['rule'],inference=f'Reuse pinned parent weights, tokenizer and engine settings, batch{16 if j=="qwen" else 4}, greedy seed20261001, max_tokens8, thinking disabled, scalar regex. Retain parent context capacity unless random text requires larger capacity within original context limit.')
 protocol['analysis']=protocol['analysis'].replace('Model pair chosen for largest observed parent NN gain','Model pair chosen among next largest observed parent NN gains')
 (d/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');(d/'SELECTION.json').write_text(json.dumps(selection,indent=2)+'\n')
# Reuse the tested verifier with pair-specific roots.
s=(r.parent/'nn_ablation_replication_20261013/validate_and_freeze.py').read_text().replace("for b in ['qwen','gemma']:","for b,j in [('llama','qwen'),('gemma','qwen122')]:").replace("nn-ablation-{b}-{b}","nn-ablation-{b}-{j}")
(r/'validate_and_freeze.py').write_text(s)
# Separate controller waits for predecessor completion; running code is not modified.
s=(r.parent/'nn_ablation_replication_20261013/controller.py').read_text().replace("ROOT=Path('/home/ubuntu/experiments/nn-ablation-replication-20261013')","ROOT=Path('/home/ubuntu/experiments/nn-ablation-extension-20261013')\nPREDECESSOR=ROOT.parent/'nn-ablation-replication-20261013'")
s=s.replace(" for b in ['qwen','gemma']:\n", " while not (PREDECESSOR/'COMPLETED.json').exists():\n  prior=read(PREDECESSOR/'STATUS.json')\n  assert prior['state']!='failed','Predecessor failed; stop queue for inspection'\n  write(ROOT/'STATUS.json',dict(state='waiting_for_predecessor',predecessor=str(PREDECESSOR),time=time.time()))\n  time.sleep(60)\n done=read(PREDECESSOR/'COMPLETED.json');assert done['new_judgments']==140000\n for filename,h in done['analysis'].items():assert sha(PREDECESSOR/'analysis'/filename)==h\n for b,j in [('llama','qwen'),('gemma','qwen122')]:\n")
s=s.replace("nn-ablation-{b}-{b}","nn-ablation-{b}-{j}").replace("pair=b+'_'+b","pair=b+'_'+j")
s=s.replace("('gemma','gemma','nn-ablation-gemma-gemma-20261013')]:","('gemma','gemma','nn-ablation-gemma-gemma-20261013'),('llama','qwen','nn-ablation-llama-qwen-20261013'),('gemma','qwen122','nn-ablation-gemma-qwen122-20261013')]:")
s=s.replace('across three backbone-judge pairs','across five backbone-judge pairs')
(r/'controller.py').write_text(s)
