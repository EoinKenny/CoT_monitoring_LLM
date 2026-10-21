from common import *
import sys

def main(b,j):
 rows=read(ROOT/'score'/b/'selected_flags.json')
 old={x['sample_id']:x for x in jl(PARENT/'score'/b/'judges'/f'{j}_nearest/run/judgments.jsonl')}
 new={x['sample_id']:x for x in jl(ROOT/'score'/b/'judges'/f'{j}_{CS[0]}/run/judgments.jsonl')}
 out=[]
 for ds in sorted({x['dataset'] for x in rows}):
  for split in ['val','test']:
   rr=[x for x in rows if x['dataset']==ds and x['split']==split];den=sum(1/x['inclusion_probability'] for x in rr)
   def mean(fn):return sum(fn(x)/x['inclusion_probability'] for x in rr)/den
   def p(x,d):return d[x['sample_id']]['positive_probability']
   out.append(dict(dataset=ds,split=split,n=len(rr),score_agreement=mean(lambda x:p(x,old)==p(x,new)),decision_agreement_at50=mean(lambda x:(p(x,old)>=50)==(p(x,new)>=50)),mean_probability_change=mean(lambda x:p(x,new)-p(x,old)),mean_absolute_probability_change=mean(lambda x:abs(p(x,new)-p(x,old))),corrected=mean(lambda x:(p(x,old)>=50)!=x['true_label'] and (p(x,new)>=50)==x['true_label']),harmed=mean(lambda x:(p(x,old)>=50)==x['true_label'] and (p(x,new)>=50)!=x['true_label']),truncated=sum(not new[x['sample_id']]['reasoning_complete'] for x in rr)))
 d=ROOT/'analysis'/f'{b}_{j}';write(d/'agreement.json',out);done=read(d/'COMPLETED.json');done['files']['agreement.json']=sha(d/'agreement.json');write(d/'COMPLETED.json',done)
if __name__=='__main__':main(*sys.argv[1:3])
