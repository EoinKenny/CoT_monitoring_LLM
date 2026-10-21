import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from setup import transform
from run_judge import parse,make_plan
root=Path(__file__).resolve().parents[1];old=(root/'OLD_OUTPUT.txt').read_text();new=(root/'NEW_OUTPUT.txt').read_text()
r=dict(condition='nearest',backbone='llama',sample_id='a',prompt_id='old',messages=[dict(role='system',content='TASK AND REFERENCE INSTRUCTIONS\n'+old),dict(role='user',content='TARGET AND NEIGHBOUR\nPositive-class probability (0-100):')])
x=transform(r,old,new);assert x['messages'][1]==r['messages'][1]
assert x['messages'][0]['content'].split('First, return')[0]==r['messages'][0]['content'].split('Return only')[0]
assert 'and VERY briefly explain the reasoning' in new
prompts=[dict(dataset='d'+str(i//1000),input_tokens=i%1000+30,sample_id=str(i),prompt_id='p'+str(i)) for i in range(10000)]
for bs in [4,8,16]:
 plan=make_plan(prompts,bs,'llama');ix=[i for b in plan for i in b['indices']]
 assert len(ix)==len(set(ix))==10000 and set(ix)==set(range(10000))
 assert sum(b['smoke'] for b in plan)==2
 assert make_plan(prompts,bs,'llama')==plan
p=dict(sample_id='a',prompt_id='p',dataset='d',backbone='llama',token_ids=[1],input_tokens=1,instruction_sha256='h')
def raw(txt,finish='stop'):return [dict(prompt_token_ids=[1],outputs=[dict(text=txt,token_ids=[2],finish_reason=finish)])]
assert parse(raw('85\nA brief assessment.'),[p],0,1)[0]['positive_probability']==85
assert not parse(raw('85\nA partial assessment','length'),[p],0,1)[0]['reasoning_complete']
for text in ['Reasoning first\n85','101\nNo','85','85\n ']:
 try:parse(raw(text),[p],0,1)
 except AssertionError:pass
 else:raise AssertionError(text)
print('PASS: exact prompt preservation, deterministic smoke/full coverage, probability-first parsing, retained truncation status')
