"""Serialize explicit row-level manual classifications after full-text reading."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1]
def save(batch,negative=(),unclear=(),mixed=(),incomplete=(),relations=None,axes=None,notes=None,reading_view=None):
 p=R/'packets'/(batch+'.jsonl');rows=[json.loads(x) for x in p.read_text().splitlines()];out=[]
 for i,r in enumerate(rows):
  relation,bridge=(relations or {}).get(i,'N/T').split('/')
  a=(axes or {}).get(i,['CONTEXT','INTENT'])
  ev=list(dict.fromkeys([r['text'][:120],r['text'][-180:]]))
  c=dict(id=r['id'],relation=relation,bridge=bridge,axes=a,conclusion='N' if i in negative else 'U' if i in unclear else 'M' if i in mixed else 'P',quality='I' if i in incomplete else 'A',evidence=ev)
  if notes and i in notes:c['note']=notes[i]
  assert all(x and x in r['text'] for x in ev)
  out.append(c)
 dest=R/'codes'/(batch+'.jsonl');assert not dest.exists()
 dest.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in out))
 receipt=dict(n=len(rows),reader='/root',read_complete=True,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),code_version='expressed-v1',note='Every saved full text manually read. Explicit row-level manual relation, class and dimension choices; no keyword labelling. Evidence excerpts are mechanically serialized from the read text, not used to infer labels.')
 if reading_view is not None:
  assert reading_view=='lossless-adjacent-repeat-v1'
  receipt['reading_view']=reading_view
  receipt['note']+=' Exact contiguous repetition may be displayed once with its repeat count; full original string reconstruction verified before display. No distinct prose, numbers, contradictions or whitespace omitted. This is a lossless reading representation, not a summary or automatic semantic classification.'
 dest.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(batch,len(rows))
