from common import *
import tempfile,subprocess,sys
# Exercise the full evaluator with identical 1-NN and 2-NN score streams in an isolated temporary root.
# No judge calls are made, and no production results are altered.
with tempfile.TemporaryDirectory(prefix='nn2-evaluation-test-') as tmp:
 root=Path(tmp);(root/'code').mkdir();(root/'score/llama/judges').mkdir(parents=True)
 for p in (ROOT/'code').glob('*.py'):(root/'code'/p.name).symlink_to(p)
 for p in (PARENT/'score/llama').iterdir():
  if p.name in ['selected_flags.json','THRESHOLDS.json'] or p.name.endswith('_predictions.npz'):(root/'score/llama'/p.name).symlink_to(p)
 (root/'score/llama/judges/qwen_nearest2').symlink_to(PARENT/'score/llama/judges/qwen_nearest',target_is_directory=True)
 subprocess.run([sys.executable,str(root/'code/evaluate.py'),'llama','qwen'],env=dict(os.environ,EXPANSION_ROOT=str(root)),check=True)
 comparisons=read(root/'analysis/llama_qwen/comparisons.json')
 for row in comparisons:
  if row['condition']=='nearest2' and row['reference']=='nearest':
   for metric in row['metrics'].values():assert abs(metric['delta'])<1e-12 and max(abs(x) for x in metric['paired_test_ci95'])<1e-12
 print('Identical-stream evaluation produces zero paired differences under both tuning policies')
