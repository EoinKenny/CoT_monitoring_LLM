"""Exact historical ablation and 2-NN prompt transforms, without host paths."""
import json
from .common import digest
REFERENCE_START='You will also receive the most similar training example, retrieved using the activation representation used by the probe.'
USE='Use the reference as supporting context, and independently classify the current conversation.'
FOOTER='Positive-class probability (0-100):'
FIELDS={'text_label':(1,1,0),'text_prediction':(1,0,1),'text_only':(1,0,0),'label_only':(0,1,0),'prediction_only':(0,0,1),'label_prediction':(0,1,1),'random_full':(1,1,1)}
def random_index(ds,target,n):
 limit=2**256-(2**256 % n)
 for counter in range(100):
  key=json.dumps([20261013,ds,target,counter],separators=(',',':'))
  h=digest(key); number=int(h,16)
  if number<limit:return number%n,h,counter
 raise RuntimeError('Hash rejection failed')

def ablation(messages,condition,random_ref=None,names=None):
 text,label,pred=FIELDS[condition];system,user=[m['content'] for m in messages]
 assert system.count(REFERENCE_START)==1 and system.count(USE)==1
 prefix,rest=system.split(REFERENCE_START);_,suffix=rest.split(USE)
 random=condition=='random_full'
 intro=('You will also receive a randomly selected training example from the same dataset.' if random else REFERENCE_START if text else 'You will also receive information about the most similar training example, retrieved using the activation representation used by the probe.')
 details=[]
 if label and pred:details.append("The reference example includes its dataset label and the probe's prediction. The probe may disagree with the dataset label.")
 elif label:details.append('The reference information includes its dataset label.')
 elif pred:details.append("The reference information includes the probe's prediction. The probe can be wrong.")
 details.append('The reference and the current conversation may belong to different classes.' if random else 'Similarity does not guarantee that the reference and the current conversation belong to the same class.')
 new_system=prefix+intro+'\n\n'+' '.join(details)+'\n\n'+USE+suffix
 current,ref=user.rsplit('\n\nNearest-neighbour explanation:\n',1)
 body,ending=ref.rsplit('\n\n'+FOOTER,1);assert ending==''
 raw_text,raw_labels=body.removeprefix('The most similar training example was:\n').rsplit('\n\nReference example:\n',1)
 assert raw_labels.count('\nProbe prediction: ')==1
 label_line,pred_line=raw_labels.split(' |\n');assert label_line.startswith('Dataset label: ') and pred_line.startswith('Probe prediction: ')
 if random:
  raw_text=json.dumps(random_ref['messages'],ensure_ascii=True)
  label_line=f"Dataset label: {names[random_ref['label']]} ({random_ref['label']})"
  pred_line=f"Probe prediction: {names[random_ref['prediction']]} ({random_ref['prediction']})"
 parts=[current, 'Random training-reference control:' if random else 'Nearest-neighbour explanation:']
 if text:parts.append(('The randomly selected training example was:\n' if random else 'The most similar training example was:\n')+raw_text)
 if label or pred:
  ls=([label_line] if label else [])+([pred_line] if pred else [])
  parts.append('Reference example:\n'+' |\n'.join(ls))
 parts.append(FOOTER)
 return [{'role':'system','content':new_system},{'role':'user','content':parts[0]+'\n\n'+parts[1]+'\n'+'\n\n'.join(parts[2:])}]

INTRO='You will also receive the most similar training example, retrieved using the activation representation used by the probe.'
def two_nn(messages,second,names):
 system,user=[m['content'] for m in messages]
 assert system.count(INTRO)==1 and user.endswith(FOOTER)
 system=system.replace(INTRO,'You will also receive the two most similar distinct training examples, retrieved using the activation representation used by the probe.',1)
 system=system.replace("The reference example includes its dataset label and the probe's prediction.","Each reference example includes its dataset label and the probe's prediction.")
 system=system.replace('the reference and the current conversation','either reference and the current conversation').replace('Use the reference as supporting context','Use the references as supporting context')
 original=user[:-len(FOOTER)]
 extra='The second most similar distinct training example was:\n'+json.dumps(second['messages'],ensure_ascii=True)+'\n\nReference example 2:\n'+f"Dataset label: {names[second['label']]} ({second['label']}) |\nProbe prediction: {names[second['prediction']]} ({second['prediction']})\n\n"
 result=[dict(role='system',content=system),dict(role='user',content=original+extra+FOOTER)]
 assert result[1]['content'].startswith(original)
 assert system.split('Return only one integer',1)[1]==messages[0]['content'].split('Return only one integer',1)[1]
 return result
