from common import *
import numpy as np,sys
# Conditional descriptions of matched sampled cases, not population-stratum F1 estimates.
def main(b,j):
 out=ROOT/'analysis'/f'{b}_{j}';metrics=read(out/'pairs.json');rows=read(ROOT/'score'/b/'selected_flags.json');answers={}
 for c in ['nearest','nearest2']:
  base=PARENT if c=='nearest' else ROOT
  answers[c]={r['sample_id']:r for r in jl(base/'score'/b/'judges'/f'{j}_{c}'/'run/judgments.jsonl')}
 summaries=[];cross=[]
 for ds in sorted({r['dataset'] for r in rows}):
  tr=[r for r in rows if r['dataset']==ds and r['split']=='test'];reference={r['sample_id']:r for r in read(ROOT/'retrieval'/b/f'{ds}.json')['records']}
  pop=read(ROOT/'score'/b/'THRESHOLDS.json')[ds]['populations']['test']
  def decision(r,c,cut):return r['probe_probability']>=cut['probe_threshold'] and answers[c][r['sample_id']]['positive_probability']>=cut['probability_threshold']
  own={c:next(x for x in metrics if x['dataset']==ds and x['condition']==c and x['policy']=='joint') for c in answers}
  for policy,cut in own.items():
   for c in answers:
    tp=sum(1/r['inclusion_probability'] for r in tr if r['true_label']==1 and decision(r,c,cut));fp=sum(1/r['inclusion_probability'] for r in tr if r['true_label']==0 and decision(r,c,cut));cross.append(dict(dataset=ds,threshold_source=policy,score_source=c,probe_threshold=cut['probe_threshold'],probability_threshold=cut['probability_threshold'],f1=2*tp/(pop['positive']+tp+fp) if pop['positive']+tp+fp else 0,description='Apply each validation-selected threshold pair to both score conditions; diagnostic, no test tuning'))
  for dimension in ['labels_agree','predictions_agree']:
   for value in [True,False]:
    subset=[r for r in tr if reference[r['sample_id']][dimension]==value]
    for policy in ['judge_fixed50','joint_own_thresholds']:
     helped=harmed=0;weighted_help=weighted_harm=0.;same=0
     for r in subset:
      sid=r['sample_id'];gold=r['true_label'];same+=answers['nearest'][sid]['positive_probability']==answers['nearest2'][sid]['positive_probability']
      pred={c:(answers[c][sid]['positive_probability']>=50 if policy=='judge_fixed50' else decision(r,c,own[c])) for c in answers}
      help=pred['nearest']!=gold and pred['nearest2']==gold;harm=pred['nearest']==gold and pred['nearest2']!=gold
      helped+=help;harmed+=harm;weighted_help+=help/r['inclusion_probability'];weighted_harm+=harm/r['inclusion_probability']
     summaries.append(dict(dataset=ds,dimension=dimension,value=value,policy=policy,n=len(subset),unchanged_scores=same,corrected=int(helped),introduced=int(harmed),weighted_corrected=weighted_help,weighted_introduced=weighted_harm))
 write(out/'alignment_diagnostics.json',summaries);write(out/'crossed_threshold_diagnostics.json',cross)
 done=read(out/'COMPLETED.json')
 for name in ['alignment_diagnostics.json','crossed_threshold_diagnostics.json']:done['files'][name]=sha(out/name)
 write(out/'COMPLETED.json',done)
if __name__=='__main__':main(*sys.argv[1:3])
