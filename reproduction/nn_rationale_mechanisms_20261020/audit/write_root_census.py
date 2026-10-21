"""Serialize explicit human-readable manual coding decisions after full reading."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
def save(batch,positive=(),mixed=(),unclear=(),incomplete=(),overrides=None):
    p=R/'packets'/(batch+'.jsonl')
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    out=[]
    for i,row in enumerate(rows):
        # Default is a manually reviewed target-only factuality assertion.
        c=dict(id=row['id'],relation='N',axes=['FACT'],bridge='T',
               conclusion='P' if i in positive else 'M' if i in mixed else 'U' if i in unclear else 'N',
               quality='I' if i in incomplete else 'A',evidence=[row['text'][:120]])
        if overrides and i in overrides:c.update(overrides[i])
        assert all(q in row['text'] and q for q in c['evidence']),(i,c)
        out.append(c)
    dest=R/'codes'/(batch+'.jsonl')
    assert not dest.exists(),dest
    dest.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in out))
    dest.with_suffix('.receipt.json').write_text(json.dumps(dict(n=len(rows),reader='/root',read_complete=True,
        input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),code_version='expressed-v1',
        note='Every complete saved rationale manually read. Individual code exceptions explicitly enumerated; no keyword classifier. These are expressed semantic claims, not verified factual correctness or private reasoning.'),indent=2)+'\n')
    print('Saved full read',batch,len(rows))
