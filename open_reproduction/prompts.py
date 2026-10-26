"""Fresh prompt construction using the approved historical wording."""
import json
import re
import sqlite3

from .common import ROOT, commit, completed, digest, read, records, sha
from .prompt_transforms import FIELDS, ablation, two_nn

TEMPLATES = ROOT/'open_reproduction/templates'
TEXT = read(TEMPLATES/'classification.json')
RUBRICS = read(TEMPLATES/'rubrics.json')
OUTPUT = TEXT['INSTRUCTION']
FOOTER = TEXT['FOOTER']


def classification(row, condition):
    ds = row['dataset']
    user = 'Classify this supplied conversation JSON:\n'+json.dumps(row['messages'],ensure_ascii=False)
    if condition == 'baseline':
        system = RUBRICS['common']+'\n\n'+RUBRICS['rubrics'][ds]
    else:
        system = TEXT['common'].format(section='explanation' if condition=='saliency' else 'reference example', rubric=RUBRICS['rubrics'][ds], explanation_instruction=TEXT['instructions'][condition])
        if condition == 'saliency':
            user += '\n\nProbe saliency explanation:\nThe sentence with the highest mean activation toward '+TEXT['positive'][ds]+' was:\n'+json.dumps(row['saliency']['sentence'],ensure_ascii=False)
        else:
            n = row['nearest']
            names = TEXT['names'][ds]
            user += '\n\nNearest-neighbour explanation:\nThe most similar training example was:\n'+json.dumps(n['messages'],ensure_ascii=False)
            user += f"\n\nReference example:\nDataset label: {names[n['label']]} ({n['label']}) |\nProbe prediction: {names[n['prediction']]} ({n['prediction']})"
    if system.count(TEXT['OLD_INSTRUCTION']) != 1:
        raise ValueError('Output instruction changed')
    system = system.replace(TEXT['OLD_INSTRUCTION'],OUTPUT,1)
    return [dict(role='system',content=system),dict(role='user',content=user+'\n\n'+FOOTER)]


def reasoning(messages, order):
    messages = [dict(m) for m in messages]
    instruction = (TEMPLATES/(order+'.txt')).read_text().rstrip('\n')
    if messages[0]['content'].count(OUTPUT) != 1:
        raise ValueError('Missing probability instruction')
    messages[0]['content'] = messages[0]['content'].replace(OUTPUT,instruction,1)
    if order == 'reason_first':
        # Exact cue from the original order comparison.
        messages[1]['content'] = messages[1]['content'].removesuffix(FOOTER)+'Assessment (VERY brief reasoning, then positive-class probability):'
    return messages


def mechanism(row):
    n = row['nearest']
    names = TEXT['names'][row['dataset']]
    values = dict(task_definition=RUBRICS['rubrics'][row['dataset']].removeprefix('Task: '),target_text=json.dumps(row['messages'],ensure_ascii=False),target_probe_prediction='Positive-class probability: '+repr(row['probe_probability']*100)+'%',neighbour_text=json.dumps(n['messages'],ensure_ascii=False),neighbour_label=f"{names[n['label']]} ({n['label']})",neighbour_probe_prediction=f"{names[n['prediction']]} ({n['prediction']})")
    template = (TEMPLATES/'mechanism.txt').read_text()
    text = re.sub(r'\{(task_definition|target_text|target_probe_prediction|neighbour_text|neighbour_label|neighbour_probe_prediction)\}',lambda m:values[m[1]],template)
    return [dict(role='user',content=text)]


def connect(path):
    con = sqlite3.connect(path)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')
    con.execute('CREATE TABLE IF NOT EXISTS prompts(prompt_id TEXT PRIMARY KEY,study TEXT,case_key TEXT,condition TEXT,split TEXT,messages TEXT,messages_hash TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS cases(case_key TEXT PRIMARY KEY,backbone TEXT,dataset TEXT,metadata TEXT)')
    con.execute('CREATE INDEX IF NOT EXISTS by_study ON prompts(study)')
    return con


def insert(con, row, study, condition, messages):
    case = digest(row['backbone']+':'+row['sample_id'])
    encoded = json.dumps(messages,ensure_ascii=False,separators=(',',':'))
    mh = digest(encoded)
    pid = 'fresh_'+digest(json.dumps([case,study,condition,mh],separators=(',',':')))
    con.execute('INSERT INTO prompts VALUES (?,?,?,?,?,?,?)',(pid,study,case,condition,row['split'],encoded,mh))


def prepare(out):
    config = read(out/'RUN.json')
    files = [out/'backbones'/bb/ds/'explanations.jsonl' for bb in config['backbones'] for ds in config['datasets']]
    inputs = {str(p.relative_to(out)):sha(p) for p in files}
    mark = out/'PROMPTS.json'
    if completed(mark,inputs):
        return
    tmp = out/'prompts.building.sqlite'
    # Preparation has no model calls; an interrupted deterministic build is safe
    # to reconstruct. The committed catalogue is never silently replaced.
    if (out/'prompts.sqlite').exists():
        raise RuntimeError('Uncommitted prompt database requires inspection')
    for suffix in ['', '-wal', '-shm']:
        p = tmp.with_name(tmp.name+suffix)
        if p.exists():
            p.unlink()
    con = connect(tmp)
    for file in files:
        for row in records(file):
            case = digest(row['backbone']+':'+row['sample_id'])
            # Gold labels and sampling weights stay in this private analysis
            # table; insert() sends ONLY the two approved message contents.
            metadata = {k:v for k,v in row.items() if k not in ['messages','saliency','random','nearest','second']}
            con.execute('INSERT INTO cases VALUES (?,?,?,?)',(case,row['backbone'],row['dataset'],json.dumps(metadata)))
            for condition in ['baseline','saliency','nearest']:
                insert(con,row,'main',condition,classification(row,condition))
            nn = classification(row,'nearest')
            insert(con,row,'two_nn','nearest2',two_nn(nn,row['second'],TEXT['names'][row['dataset']]))
            for condition in FIELDS:
                insert(con,row,'ablation',condition,ablation(nn,condition,row['random'],TEXT['names'][row['dataset']]))
            if row['split'] == 'test':
                for study in ['reason_after','reason_first']:
                    insert(con,row,study,study,reasoning(nn,study))
                insert(con,row,'mechanism','mechanism',mechanism(row))
        con.commit()
    counts = dict(con.execute('SELECT study,count(*) FROM prompts GROUP BY study'))
    con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    con.close()
    tmp.replace(out/'prompts.sqlite')
    commit(mark,inputs,[out/'prompts.sqlite'],counts=counts)


def jobs(out, study):
    con = sqlite3.connect('file:'+str(out/'prompts.sqlite')+'?mode=ro',uri=True)
    try:
        for pid,case,condition,split,messages,mh in con.execute('SELECT prompt_id,case_key,condition,split,messages,messages_hash FROM prompts WHERE study=? ORDER BY rowid',(study,)):
            if digest(messages) != mh:
                raise ValueError('Prompt content hash mismatch')
            yield dict(prompt_id=pid,study=study,case_key=case,condition=condition,split=split,messages=json.loads(messages))
    finally:
        con.close()
