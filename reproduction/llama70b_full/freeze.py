"""Freeze local source, tokenizer, model identities and additive disk budget before launch."""
from common import *
import subprocess
assert not (ROOT/'SOURCE_MANIFEST.json').exists(),'Existing frozen manifest must not be overwritten'
free=shutil.disk_usage(ROOT).free;feature_bytes=456008*8192*4;budget=feature_bytes+30*2**30
assert free-budget>20*2**30,'Projected artifacts do not fit above reserve'
write(ROOT/'DISK_BUDGET.json',dict(initial_free_gib=free/2**30,pooled_feature_gib=feature_bytes/2**30,other_artifact_budget_gib=30,reserve_gib=20,projected_remaining_gib=(free-budget)/2**30,cleanup='none'))
plan=read(PILOT/'PLAN.json');weights={n:dict(bytes=(MODEL/n).stat().st_size,blob=(MODEL/n).resolve().name) for n in plan['shards']}
tokenizer={p.name:sha(p) for p in MODEL.iterdir() if p.is_file() and p.suffix in ['.json','.jinja'] and 'safetensors' not in p.name}
write(B/'backbone.json',dict(repo='meta-llama/Llama-3.3-70B-Instruct',revision=MODEL.name,path=str(MODEL),block_index=54,original_blocks=80,executed_blocks=55,hidden_size=8192,dtype='bfloat16',quantization=None,gpu_blocks=17,tokenizer_files=tokenizer,weight_blobs=weights,template_date_string='01 Oct 2026',seed=SEED))
write(ROOT/'SOURCE_MANIFEST.json',dict(code={p.name:sha(p) for p in (ROOT/'code').glob('*.py')},readme_sha256=sha(ROOT/'code/README.md'),prompt_builder_sha256=sha(B/'prepare_judges.py'),rubrics_sha256=sha(B/'judge_rubrics.json'),policy_sha256=sha(ROOT/'BATCH_POLICY.json'),prior_metadata={ds:sha(PRIOR/'llama'/f'{ds}_metadata.json') for ds in DATASETS},backbone_sha256=sha(B/'backbone.json'),time=time.time()))
print(json.dumps(read(ROOT/'DISK_BUDGET.json')),flush=True)
