"""Display frozen purposive source checks without presenting them as prevalence."""
from pathlib import Path
from html import escape
from datetime import datetime, timezone
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"broad_contrast/context_followup"
cases=json.loads((O/"unmasked_examples.json").read_text())
status=json.loads((R/"broad_contrast/results/READING_STATUS.json").read_text())
names={"qwen":"Qwen27B","glm":"GLM"}
titles={
    "aux_e21480e92bb3af914b7f":"First collaboration since 1991",
    "aux_5d96728686fff310b980":"An elliptical but supported answer: Serie A",
    "aux_076ef9d9ff6a63b2dd0f":"Drug price versus stock price — GLM",
    "aux_e1fcf85fe3777dfda7c5":"The identical price comparison — Qwen",
    "aux_7a4621f52fe3ae8c5e79":"Ethnicity versus geographical origin",
    "aux_944cc63d6b8f99a4856f":"Combining supported shoe-care advice",
}
def messages(xs):
    return "\n\n".join(x["role"]+": "+x["content"] for x in xs)

rows=[]
cards=[]
for x in cases:
    rows.append(f"""<tr><td><a href="#{x['id']}">{escape(titles[x['id']])}</a></td>
    <td>{escape(names[x['judge']])} / {escape(x['backbone'])}</td>
    <td>{x['true_label']}</td><td>{x['control_probability']:g} / {x['nn_probability']:g}</td>
    <td>{x['probability_first_answer']}</td>
    <td>{'Correct' if x['answer_correct'] else 'Wrong'}</td>
    <td>{escape(x['task_rule'])}</td></tr>""")
    cards.append(f"""<details id="{x['id']}"><summary>{escape(titles[x['id']])}</summary>
    <p><b>Task:</b> {escape(x['task'])}</p>
    <p><b>Target label:</b> {x['true_label']}.
    <b>Neighbour label / probe prediction:</b> {x['nearest_label']} / {x['nearest_prediction']}.
    <b>Original control / NN probability:</b> {x['control_probability']:g} / {x['nn_probability']:g}.
    <b>New first probability:</b> {x['probability_first_answer']}.
    <b>Explanation:</b> {'complete' if x['explanation_complete'] else 'capped / incomplete'}.</p>
    <h3>Target and its evidence</h3><pre>{escape(messages(x['target_messages']))}</pre>
    <h3>Neighbour and its evidence</h3><pre>{escape(messages(x['neighbor_messages']))}</pre>
    <h3>Saved explanation after the probability</h3><pre>{escape(x['text'])}</pre>
    <p><b>Source check:</b> {escape(x['reason'])}</p>
    <p><b>Interpretation:</b> {escape(x['mechanism'])}</p>
    <p class="small">Assessment {escape(x['id'])}; source hash {escape(x['source_file_sha256'])}.
    Outcomes and model identities were joined only after the contextual assessments were frozen.</p></details>""")

html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Correct answers can have inconsistent explanations — source checks</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#24313d}}
h1,h2,h3{{line-height:1.25}}a{{color:#175b91}}.box{{background:#eef5fa;padding:18px;border-left:5px solid #367ca8}}
.warning{{background:#fff4da;padding:18px;border-left:5px solid #b77b13}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d8e0e7}}
th{{background:#edf2f5}}.scroll{{overflow:auto}}details{{border:1px solid #d5dfe6;padding:15px;margin:16px 0;border-radius:6px}}
summary{{font-weight:600;cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px;font:14px/1.6 system-ui}}
.small{{font-size:13px;color:#586571}}</style></head><body>
<p><a href="../../results/report.html">← Main probability-first audit</a></p>
<h1>Correct answers can have inconsistent explanations</h1>
<div class="box">These six source checks come from additional saved probability-first responses from the
earlier paused collection. All six had unchanged original control-versus-NN classifications at cutoff 50.
They are separate from the primary study of original NN flips.</div>
<p>Three Qwen examples—the Serie A answer, the Armenian-genocide summary and the shoe-care answer—receive
correct new probabilities below 50, while the following prose argues that a hallucination exists.
All three had previously been classified as hallucinated under both control and NN prompts.
This shows why the first probability must remain the scored answer and the explanation must be checked separately.</p>
<p>The two drug-price examples use <b>exactly the same target, neighbour and displayed neighbour annotations</b>.
GLM treats the neutral stock-price neighbour as support for a factual-inquiry interpretation; Qwen rejects
that analogy because the queried product is illegal. The benchmark label is positive. GLM is wrong by that label,
but its stated interpretation follows the prompt's distinction between mentioning a sensitive topic and soliciting
toxic content. Qwen matches the benchmark, but its prose uses the broader rule that illegal subject matter is itself toxic.
This is evidence of different expressed classification conventions, not proof of either model's internal mechanism.</p>
<div class="warning"><b>Purposive illustrations only.</b> The examples were selected for specific reasoning issues
before opening their labels and outcomes. This is one reader's contextual assessment, not independent human
validation or a prevalence estimate. The broader auxiliary census currently covers
{status['records_read']:,} / {status['selected_records']:,} responses. The main reasoning-first census is also ongoing.</div>
<h2>Answers and source checks</h2><div class="scroll"><table><thead><tr>
<th>Example</th><th>Judge / backbone</th><th>Gold</th><th>Old control / NN</th>
<th>New probability</th><th>New answer vs gold</th><th>Task-rule assessment</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="small">Positive means hallucination for HaluEval/RAGTruth and toxic for ToxicChat.
Every classification uses probability ≥50. Task-rule assessments concern the prose, independently of answer correctness.</p>
<h2>Full examples</h2>{''.join(cards)}
<p class="small">The raw and capped outputs are preserved. No new model calls were made.
<a href="DESIGN.json">Selection design</a> · <a href="CODING_FREEZE.json">Assessment freeze</a> ·
<a href="UNMASKING_RECEIPT.json">Verified outcome join</a> ·
<a href="unmasked_examples.json">Complete example records</a></p></body></html>"""
dest=O/"report.html"
dest.write_text(html)
(O/"REPORT_BUILD.json").write_text(json.dumps(dict(
    built_utc=datetime.now(timezone.utc).isoformat(),n_examples=len(cases),
    report_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    note="Static HTML; source-level link and escaping checks, not a claimed browser rendering."
),indent=2)+"\n")
print(dest)
