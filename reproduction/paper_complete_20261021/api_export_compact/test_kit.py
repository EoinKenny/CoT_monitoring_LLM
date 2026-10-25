"""Offline tests. No API keys, provider SDKs, network calls or inference."""
import csv,json,sqlite3,tempfile
from pathlib import Path
import kit

def main():
 assert kit.parse('main','50','stop')==(50,'ok')
 for x in ['50%','-1','101','50 because','50\n20','null','5.0']:assert kit.parse('main',x,'stop')[1]=='invalid'
 assert kit.parse('reason_after','80\nUnfinished because','length')==(80,'ok')
 assert kit.parse('reason_after','80\nActually 20.','stop')==(80,'ok')
 assert kit.parse('reason_first','Explanation\n80','stop')==(80,'ok')
 assert kit.parse('reason_first','Explanation\n80','length')[1]=='invalid'
 assert kit.parse('reason_first','80','stop')[1]=='invalid'
 assert kit.parse('mechanism','{"labelled_similarity":"yes","important_contrast":"no"}','stop')[1]=='ok'
 assert kit.parse('mechanism','{"labelled_similarity":"yes","labelled_similarity":"no","important_contrast":"no"}','stop')[1]=='invalid'
 # The selector must keep flips, exclude unchanged and unresolved/invalid pairs,
 # and assign identical underlying cases to both reasoning orders.
 original=kit.ROOT
 with tempfile.TemporaryDirectory() as d:
  kit.ROOT=Path(d);db=sqlite3.connect(kit.ROOT/'prompts.sqlite');db.executescript('CREATE TABLE prompts(prompt_id TEXT,study TEXT);CREATE TABLE pairs(case_key TEXT,baseline TEXT,nearest TEXT,reason_after TEXT,reason_first TEXT);')
  rs=[];events=[]
  for i,(a,b,status) in enumerate([(20,80,'ok'),(80,20,'ok'),(20,30,'ok'),(20,80,'ambiguous')]):
   db.execute('INSERT INTO pairs VALUES(?,?,?,?,?)',(str(i),f'b{i}',f'n{i}',f'a{i}',f'f{i}'))
   for prefix,value in [('b',a),('n',b)]:
    pid=f'{prefix}{i}';db.execute('INSERT INTO prompts VALUES (?,?)',(pid,'main'));st=status if prefix=='n' else 'ok';rs.append(dict(zip(kit.FIELDS,[pid,'gpt-5.5',str(value),st,'stop','','',''])))
    events.extend([dict(event='dispatch',model='gpt-5.5',prompt_id=pid),dict(event='result',model='gpt-5.5',prompt_id=pid,status=st)])
  db.commit();db.close()
  with (kit.ROOT/'responses.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=kit.FIELDS);w.writeheader();w.writerows(rs)
  (kit.ROOT/'journal.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events));kit.select_traces('gpt-5.5',kit.ROOT/'responses.csv',kit.ROOT/'journal.jsonl',kit.ROOT/'selection.json');sel=json.loads((kit.ROOT/'selection.json').read_text());assert sel['prompt_ids']=={'reason_after':['a0','a1'],'reason_first':['f0','f1']}
 kit.ROOT=original
 print('Offline parsing and dynamic flip-selection tests passed')
if __name__=='__main__':main()
