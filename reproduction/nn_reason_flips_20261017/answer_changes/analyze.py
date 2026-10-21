from pathlib import Path
import json,gzip,html,collections
r=Path(__file__).parent
with gzip.open(r/'snapshot.json.gz','rt') as f:x=json.load(f)
rows=x['rows'];summary=[];datasets=[]
for j in ['qwen','gemma','mistral','olmo','glm','qwen122']:
 a=[z for z in rows if z['judge']==j and type(z['reason_probability']) is int]
 assert all((z['control_probability']>=50)!=(z['nn_probability']>=50) for z in a)
 for z in a:
  if (z['reason_probability']>=50)!=(z['nn_probability']>=50):assert (z['reason_probability']>=50)==(z['control_probability']>=50)
 h=[z for z in a if z['flip_type']=='helped'];bad=[z for z in a if z['flip_type']=='harmed']
 fix=sum(int(z['reason_probability']>=50)==z['true_label'] for z in bad);undo=sum(int(z['reason_probability']>=50)!=z['true_label'] for z in h)
 prev=next(s for s in x['summary'] if s['judge']==j);assert fix==prev['correct_harmful'] and undo==prev['undo_helpful']
 summary.append(dict(prev,share_changes_corrective=fix/(fix+undo) if fix+undo else None,repair_rate=fix/len(bad) if bad else None,undo_rate=undo/len(h) if h else None,accuracy_change_pp=100*(fix-undo)/len(a)))
 for ds in sorted({z['dataset'] for z in a}):
  b=[z for z in a if z['dataset']==ds];f=sum(z['flip_type']=='harmed' and int(z['reason_probability']>=50)==z['true_label'] for z in b);u=sum(z['flip_type']=='helped' and int(z['reason_probability']>=50)!=z['true_label'] for z in b)
  datasets.append(dict(judge=j,dataset=ds,n=len(b),corrected=f,undone=u,net=f-u))
(r/'summary.json').write_text(json.dumps(dict(time=x['time'],summary=summary,datasets=datasets,scope='Raw unweighted counts on each judge own original fixed50 test-flip subset; no full-test F1 claim. Invalid/missing outputs retained in snapshot but excluded from paired valid-score denominators.'),indent=2))
trs=''
for s in summary:
 good='—' if s['share_changes_corrective'] is None else f"{100*s['share_changes_corrective']:.1f}%"
 trs+=f"<tr><td>{s['judge']}{'' if s['complete'] else ' (partial)'}</td><td>{s['valid']:,}</td><td>{s['correct_harmful']:,} / {s['original_harmful']:,}</td><td>{s['undo_helpful']:,} / {s['original_helpful']:,}</td><td>{good}</td><td>{s['accuracy_change_pp']:+.2f}</td></tr>"
examples=json.loads((r/'examples.json').read_text());cards=''
for e in examples:
 cards+=f"<details><summary>{html.escape(e['judge']+' / '+e['dataset'])}: control {e['control_probability']} → NN {e['nn_probability']} → explanation prompt {e['reason_probability']}; gold {e['true_label']}</summary><pre>{html.escape(e['body'])}</pre><blockquote>{html.escape(e['reasoning'])}</blockquote><code>{e['sample_id']}</code></details>"
page='''<!doctype html><meta charset="utf-8"><title>Changes after requesting an explanation</title><style>body{font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#20303e}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #dbe0e6}th{background:#eef3f7}details{margin:20px 0;border:1px solid #dbe0e6;padding:16px}pre{white-space:pre-wrap;font:inherit}blockquote{border-left:4px solid #7496ae;padding-left:16px}code{overflow-wrap:anywhere}</style><h1>Does the explanation prompt correct or undo NN flips?</h1><p>Original control and NN predictions differed at cutoff50. The new prompt requests probability first and a very brief explanation afterwards. Every changed binary answer therefore returns to the original control classification. This is a diagnostic, outcome-selected subset, not a full-test performance estimate. Counts are unweighted and cases can recur across backbones. Qwen has two missing interrupted attempts excluded from valid-score denominators; other sampled models have valid probabilities for all collected outputs.</p><table><tr><th>Judge</th><th>Valid n</th><th>Corrected / original harmful</th><th>Undone / original helpful</th><th>Corrective share of changes</th><th>Accuracy change (pp)</th></tr>'''+trs+'''</table><p>OLMo has 1,004 net lost correct answers; FAVA contributes474 and HaluEval400. Explanations below are illustrative, not a systematic coding of rationale consistency. Some contradict their first probability. Temperature and seeds were fixed, but the prompt and batching differ from the original inference; results do not isolate a private reasoning mechanism.</p><h2>Illustrative changed-answer traces</h2>'''+cards
(r/'report.html').write_text(page)
for s in summary:print(s['judge'],s['valid'],s['correct_harmful'],s['undo_helpful'],round(100*s['share_changes_corrective'],1) if s['share_changes_corrective'] is not None else None,round(s['accuracy_change_pp'],2))
