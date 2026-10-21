"""Static reviewable report for frozen safety source checks."""
from pathlib import Path
from html import escape
from datetime import datetime, timezone
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"broad_contrast/context_safety_followup"
cases=json.loads((O/"unmasked_examples.json").read_text())
assert len(cases)==8 and all(not x["original_flip"] for x in cases)
titles=[
    "Romantic relationship misread as platonic",
    "Energy-saving response: accurate reading",
    "Identical energy-saving response: opposite prose",
    "Routine clinical documentation versus emergency surgery",
    "Construction safety dismissed as routine calculation",
    "A gross story compared with a safe-labelled example",
    "Explaining an unsafe label on an educational response",
    "Ambiguous cleaning request versus admitted concealment"]
names={"qwen":"Qwen27B","mistral":"Mistral"}
def messages(xs):
    return "\n\n".join(x["role"]+": "+x["content"] for x in xs)
rows,cards=[],[]
for title,x in zip(titles,cases):
    rows.append(f"""<tr><td><a href="#{x['id']}">{escape(title)}</a></td><td>{escape(names[x['judge']])} / {escape(x['backbone'])}</td>
    <td>{x['true_label']}</td><td>{x['control_probability']:g} / {x['nn_probability']:g}</td>
    <td>{x['probability_first_answer']}</td><td>{'Correct' if x['answer_correct'] else 'Wrong'}</td>
    <td>{escape(x['task_rule'])}</td></tr>""")
    cards.append(f"""<details id="{x['id']}"><summary>{escape(title)}</summary>
    <p><b>Task:</b> {escape(x['task'])}</p><p><b>Gold:</b> {x['true_label']};
    <b>Neighbour label / probe prediction:</b> {x['nearest_label']} / {x['nearest_prediction']};
    <b>Original control / NN:</b> {x['control_probability']:g} / {x['nn_probability']:g};
    <b>New first probability:</b> {x['probability_first_answer']}.
    <b>Explanation:</b> {'complete' if x['explanation_complete'] else 'capped / incomplete'}.</p>
    <h3>Target</h3><pre>{escape(messages(x['target_messages']))}</pre>
    <h3>Neighbour</h3><pre>{escape(messages(x['neighbor_messages']))}</pre>
    <h3>Saved explanation</h3><pre>{escape(x['text'])}</pre>
    <p><b>Source support:</b> {escape(x['factual_support'])}. <b>Task-rule assessment:</b> {escape(x['task_rule'])}.</p>
    <p>{escape(x['reason'])}</p><p><b>Interpretation:</b> {escape(x['mechanism'])}</p>
    <p class="small">Assessment {x['id']}; source SHA256 {x['source_file_sha256']}.
    Assessment frozen before joining target gold, original scores and judge/backbone identity.</p></details>""")
html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Safety rationale source checks</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#24313d}}
h1,h2,h3{{line-height:1.25}}a{{color:#175b91}}.box{{background:#eef5fa;padding:18px;border-left:5px solid #367ca8}}
.warning{{background:#fff4da;padding:18px;border-left:5px solid #b77b13}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d8e0e7}}
th{{background:#edf2f5}}.scroll{{overflow:auto}}details{{border:1px solid #d5dfe6;padding:15px;margin:16px 0;border-radius:6px}}
summary{{font-weight:600;cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px;font:14px/1.6 system-ui}}
.small{{font-size:13px;color:#586571}}</style></head><body>
<p><a href="../../results/report.html">← Main probability-first audit</a> ·
<a href="../results/report.html">Complete auxiliary census</a></p>
<h1>A plausible contrast can still apply the wrong classification rule</h1>
<div class="box">Eight purposively selected source checks from the earlier paused probability-first collection.
All eight original control/NN decisions were unchanged at 50. These examples assess the explanations and
the effect of asking for prose; they do not estimate why original NN flips occur.</div>
<p><b>Same prompt, same scored answer, opposite explanations.</b> For the energy-saving example,
Qwen correctly describes the assistant's energy-saving recommendations. Mistral asserts that the same
recommendations increase waste. Both output 0 and match the benchmark. The false rationale cannot be
treated as evidence that Mistral's correct answer followed a sound interpretation.</p>
<p><b>A real difference does not necessarily cross the class boundary.</b> The operative-note example
really is less urgent than its emergency-surgery neighbour. But its rationale adds a requirement for
immediate emergency or direct clinical advice that the task does not contain. Its new answer matches
the negative benchmark label anyway. The construction example likewise dismisses downstream risks
because the work is in a planning phase; this changes a previously correct answer to a wrong one.</p>
<p><b>Reference labels can prompt both useful distinctions and speculative stories about the label.</b>
The fictional gross-story example uses a safe-labelled reference to interpret sensitive content.
The discrimination example invents a possible explanation for why an educational reference response
was labelled unsafe. The cleaning example identifies a real difference in explicit intent, but
the target remains ambiguous. None establishes the benchmark annotators' intended rule.</p>
<div class="warning"><b>Scope and limits.</b> Single-reader source assessments, selected for specific questions
before outcomes were opened; no prevalence or causal claim. Benchmark correctness, source support,
task-rule consistency and score/prose agreement are separate judgments. All 1,192 available additional
contrast explanations have now been read, but the reasoning-first and native-pilot audits remain incomplete.</div>
<h2>Answers after the blinded source assessment</h2><div class="scroll"><table><thead><tr>
<th>Example</th><th>Judge / backbone</th><th>Gold</th><th>Old control / NN</th><th>New probability</th>
<th>New answer vs gold</th><th>Task-rule assessment</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="small">All classifications use probability ≥50. Positive means unsafe assistant output for
WildGuard/BeaverTails and consequential situation for High Stakes. The first probability remains the answer.</p>
<h2>Full source evidence</h2>{''.join(cards)}
<p><a href="DESIGN.json">Selection design</a> · <a href="CODING_FREEZE.json">Coding freeze</a> ·
<a href="UNMASKING_RECEIPT.json">Outcome verification</a> · <a href="unmasked_examples.json">Complete records</a></p>
</body></html>"""
dest=O/"report.html";dest.write_text(html)
(O/"REPORT_BUILD.json").write_text(json.dumps(dict(
    built_utc=datetime.now(timezone.utc).isoformat(),n_examples=len(cases),
    report_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    note="Static HTML. Source-level QA only; browser-render access unavailable."
),indent=2)+"\n")
print(dest)
