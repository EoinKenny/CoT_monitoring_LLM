from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parents[2];batch=sys.argv[1]
packet=R/'packets'/(batch+'.jsonl');rs=[json.loads(x) for x in packet.read_text().splitlines()]
lines=(Path(__file__).parent/(batch+'.manual')).read_text().splitlines();assert len(lines)==len(rs),(len(lines),len(rs))
out=[]
for i,(r,l) in enumerate(zip(rs,lines)):
 a=l.split('|');assert len(a) in [6,7],(i,a)
 relation,axes,bridge,conclusion,quality,evidence=a[:6];assert relation in 'NSCBU' and bridge in 'TLDRQU' and conclusion in 'PNMU' and quality in 'AIU'
 ev=evidence.split('~~');assert all(e and e in r['text'] for e in ev),(i,ev,r['text'])
 o=dict(id=r['id'],relation=relation,axes=axes.split(','),bridge=bridge,conclusion=conclusion,quality=quality,evidence=ev)
 if len(a)==7:o['note']=a[6]
 out.append(o)
assert {x['id'] for x in out}=={x['id'] for x in rs};assert len({x['id'] for x in out})==len(out)
(R/'codes'/(batch+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
receipt=dict(batch_id=batch,reader='trace_inventory',read_complete=True,n=len(out),input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),notes='All full texts manually read and semantically coded outcome-masked; no keyword classifier. Evidence substrings and exact IDs validated.',code_version='expressed-v1')
(R/'codes'/(batch+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(batch,len(out),'validated')
