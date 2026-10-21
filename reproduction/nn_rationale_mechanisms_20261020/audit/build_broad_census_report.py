"""Report auxiliary unchanged cases without generalising imbalanced counts."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import csv
import hashlib
import json

R=Path(__file__).resolve().parents[1]
O=R/"broad_contrast/results"
status=json.loads((O/"READING_STATUS.json").read_text())
assert status["all_available_read"]
rows=list(csv.DictReader((O/"descriptive_summary.csv").open()))
total=next(x for x in rows if x["dimension"]=="all")
def table(dimension):
    out=[]
    for x in rows:
        if x["dimension"]!=dimension:continue
        out.append("<tr>"+"".join(f"<td>{escape(x[k])}</td>" for k in [
            "group","n","corrected","undone","unchanged","capped","prose_score_conflict"])+"</tr>")
    return """<div class="scroll"><table><thead><tr><th>Group</th><th>Responses</th>
    <th>Original mistakes corrected</th><th>Original correct answers lost</th><th>Same answer</th>
    <th>Capped explanations</th><th>Score/prose conflict</th></tr></thead><tbody>"""+''.join(out)+"</tbody></table></div>"

html=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Additional unchanged contrast cases</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#24313d}}
h1,h2{{line-height:1.25}}a{{color:#175b91}}.box{{background:#eef5fa;padding:18px;border-left:5px solid #367ca8}}
.warning{{background:#fff4da;padding:18px;border-left:5px solid #b77b13}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d8e0e7}}
th{{background:#edf2f5}}.scroll{{overflow:auto}}.small{{font-size:13px;color:#586571}}</style></head><body>
<p><a href="../../results/report.html">← Main probability-first audit</a></p>
<h1>Additional contrast cases where the original prediction did not change</h1>
<div class="box"><b>Complete reading coverage: 1,192 / 1,192 available explanations.</b>
These are additional test responses saved by the earlier paused broader collection.
Every case has majority-labelled important contrast, and every original control/NN decision agrees at cutoff 50.
They supplement the primary audit of original helpful and harmful flips.</div>
<p>In the later probability-first/explanation-after response, <b>1,114 answers stayed the same</b>,
<b>26 original mistakes were corrected</b>, and <b>52 originally correct answers became wrong</b>.
All first probabilities are valid; 16 explanations hit the output cap. These are raw descriptive counts
in this saved subset, not estimates of deployment performance or a causal effect of asking for an explanation.</p>
<p><b>85 / 1,189 explanations with a clear verbal class disagree with their own first probability</b>.
Three explanations do not settle on a clear verbal class. This is why the numeric answer and the prose
must be analysed separately. A correct scored answer can be followed by an incorrect argument.</p>
<div class="warning"><b>Strong selection imbalance:</b> 1,146 / 1,192 responses come from Qwen27B.
The remaining judges contribute only 3–23 responses each. Cases were available because the earlier
collection stopped, not because they form a representative random sample of unchanged cases.
Do not pool these counts with the primary flip study, rank judges using them, or infer population rates.
No validation cases are included. Target repetitions across contexts/judges are not independent.</div>
<h2>By dataset family</h2>{table("family")}
<p class="small">Toxicity: Aegis, BeaverTails, ToxicChat, ToxiGen.
Hallucination: FAVA, HaluEval, RAGTruth. Other safety: High Stakes, WildGuard, WildJailbreak.
Corrected/lost compares the new first probability with the original NN prediction, not control versus NN.</p>
<h2>By dataset</h2>{table("dataset")}
<h2>Judge coverage</h2>{table("judge")}
<h2>What the full-source checks add</h2>
<p>The recurring distinctions include harmful requested intent versus actual safe output,
general information versus consequential advice, and offensive wording versus the dataset's actual positive class.
Some rationales use the neighbour to apply those distinctions. Others simply describe the target;
310 / 1,192 explicitly express resemblance, difference or both. Absence of an explicit comparison is not
evidence that the neighbour had no effect.</p>
<p>Source checks also expose failures that a class label alone cannot detect: conflating the request with
the delivered response, inventing an emergency-only rule for high stakes, explaining away reference labels
with unstated context, and treating a supported answer as hallucinated because it is short or synthesised.</p>
<ul><li><a href="../context_followup/report.html">Six earlier source checks: identical drug-price prompts and correct answers with inconsistent prose</a></li>
<li><a href="../context_safety_followup/report.html">Eight safety source checks: identical energy-saving prompts, clinical documentation and real versus inferred intent</a></li></ul>
<p>These 14 contextual assessments were selected for specific questions and frozen before original outcomes
were opened. They are one reader's purposive illustrations, not a prevalence estimate or independent human
validation. The main balanced 200-case audit remains the separately defined contextual comparison.</p>
<p class="small">The primary probability-first census is complete. The reasoning-first and native-pilot
contrast censuses are still in progress. This auxiliary completion does not imply the whole study is finished.</p>
<p><a href="DESCRIPTIVE_METHOD.json">Method and hashes</a> ·
<a href="READING_STATUS.json">Reading verification</a> ·
<a href="descriptive_summary.csv">Summary counts</a> ·
<a href="answer_prose_records.csv">Per-response comparison</a></p>
</body></html>"""
dest=O/"report.html";dest.write_text(html)
(O/"REPORT_BUILD.json").write_text(json.dumps(dict(
    built_utc=datetime.now(timezone.utc).isoformat(),report_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    full_read_records=1192,context_assessments=14,
    note="Static HTML, source-level validation; no browser-render claim."
),indent=2)+"\n")
print(dest)
