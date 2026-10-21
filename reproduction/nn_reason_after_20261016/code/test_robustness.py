import json,importlib.util
from pathlib import Path
from output_policy import outcome,missing
D=Path(__file__).parent

def module(name):
 s=importlib.util.spec_from_file_location(name,D/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v3=module('run_judge_v3');v4=module('run_judge_v4')
p=dict(sample_id='x',prompt_id='p',dataset='ds',backbone='llama',condition='nearest_reason_after',token_ids=[1,2],input_tokens=2,instruction_sha256='hash')
for text in ['0\nNegative.','50\nUncertain.','100\nPositive.','1\n   ','33','0\nActually 100','12\n'+('Repeat '*100)]:
 for finish in ['stop','length']:
  raw=[dict(prompt_token_ids=[1,2],outputs=[dict(text=text,token_ids=[8],finish_reason=finish)])]
  assert v3.parse(raw,[p],0,1.0)==v4.parse(raw,[p],0,1.0)
for text in ['', 'nonsense\n50','101\nreason','-1\nreason','0.5\nreason','{"probability":50}', 'thinking first\n80']:
 raw=[dict(prompt_token_ids=[1,2],outputs=[dict(text=text,token_ids=[8],finish_reason='length')])]
 r=v4.parse(raw,[p],0,1.0)[0]
 assert r['positive_probability'] is None and r['prediction'] is None and r['generated_text']==text
 assert r['output_status']=='invalid_probability'
r=missing([p],4,0.)[0]
assert r['positive_probability'] is None and r['output_status']=='ambiguous_no_raw' and r['prompt_id']=='p'
assert missing([p],4,0.)==json.loads(json.dumps(missing([p],4,0.)))
try:v4.parse([dict(prompt_token_ids=[9],outputs=[])],[p],0,0.)
except AssertionError:pass
else:raise AssertionError('Prompt-integrity gate missing')
print('PASS: legacy parser compatibility; malformed outputs retained; first probability authoritative; ambiguous attempts explicit; integrity gate retained')
