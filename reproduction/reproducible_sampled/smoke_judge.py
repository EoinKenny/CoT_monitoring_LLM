from common import *
source=E/'ten-dataset-sampled-20261002/judges/qwen_baseline';dest=ROOT/'smoke/qwen_baseline';dest.mkdir(parents=True,exist_ok=True)
assert not (dest/'prompts.jsonl').exists()
lines=[]
with (source/'prompts.jsonl').open() as f:
 for _ in range(2):lines.append(next(f))
(dest/'prompts.jsonl').write_text(''.join(lines));(dest/'model_path.txt').write_text((source/'model_path.txt').read_text());cfg=read(source/'approved_config.json');cfg.update(n=2,prompt_sha256=sha(dest/'prompts.jsonl'),purpose='Runner validation only; excluded from experiment results');write(dest/'approved_config.json',cfg)
