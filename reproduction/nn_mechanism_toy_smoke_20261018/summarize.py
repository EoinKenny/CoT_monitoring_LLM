import json,html,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(R/'COMPLETED.json');assert sha(R/'results.json')==receipt['results_sha256']
for name,digest in receipt['raw_sha256'].items():assert sha(R/'batches'/name)==digest
frozen=read(R/'PREFROZEN.json')
for n,digest in frozen['files'].items():assert sha(R/n)==digest
cases=read(R/'toy_cases_with_expectations.json');results={r['id']:r for r in read(R/'results.json')};assert set(results)=={r['id'] for r in cases}
fields=['labelled_similarity','important_contrast'];out=[]
for c in cases:
 r=results[c['id']];got=[r['parsed'].get(k) for k in fields] if r['valid'] else [None,None]
 out.append(dict(id=c['id'],target=c['target_text'],neighbour=c['neighbour_text'],expected=c['expected'],observed=got,expected_note=c['note'],matches=[x==y for x,y in zip(c['expected'],got)],raw_text=r['raw_text'],valid=r['valid']))
summary=dict(n=len(out),valid_json=sum(r['valid'] for r in out),exact_pairs=sum(all(r['matches']) for r in out),individual_matches=sum(sum(r['matches']) for r in out),individual_labels=len(out)*2,loading_seconds=receipt['loading_seconds'],generation_seconds=receipt['generation_seconds'],cases=out,limits='Manual toy expectations are not independently adjudicated truth. This tests prompt comprehension and format, not reliability on natural data or a causal mechanism.')
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Qwen122 mechanism annotation toy test</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:35px auto;padding:0 24px;color:#172335}table{border-collapse:collapse;width:100%}td,th{padding:12px;vertical-align:top;border-bottom:1px solid #d5dce4;text-align:left}.miss{background:#fff1e1}pre{white-space:pre-wrap;background:#f3f6fa;padding:15px}small{color:#526375}</style><h1>Qwen122: eight-case annotation smoke test</h1>']
parts.append(f"<p>{summary['valid_json']}/8 valid JSON responses; {summary['exact_pairs']}/8 pairs and {summary['individual_matches']}/16 individual labels match expectations recorded before calls. Native thinking disabled; greedy temperature 0; max output 128 tokens; no constrained decoding or retries.</p>")
parts.append('<p>Manual expectations are a toy rubric check, not independent ground truth or a real-data accuracy estimate. No mechanism annotation of research cases was launched.</p><table><tr><th>Case</th><th>Target</th><th>Neighbour</th><th>Expected<br>similarity / contrast</th><th>Qwen<br>similarity / contrast</th></tr>')
for r in out:
 parts.append('<tr'+(' class="miss"' if not all(r['matches']) else '')+'>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['id'],r['target'],r['neighbour'],' / '.join(r['expected']),' / '.join(str(x) for x in r['observed'])])+'</tr>')
parts.append('</table><h2>Raw outputs</h2>')
for r in out:parts.append('<h3>'+html.escape(r['id'])+'</h3><pre>'+html.escape(r['raw_text'])+'</pre><small>Pre-call expectation: '+html.escape(r['expected_note'])+'</small>')
parts.append('<details><summary>Approved prompt template</summary><pre>'+html.escape((R/'prompt_template.txt').read_text())+'</pre></details></html>')
(R/'report.html').write_text(''.join(parts))
print(json.dumps(summary,indent=2))
