"""Integrate completed saved-rationale censuses; no inference or outcome changes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from aggregate_semantics import compare, weighted_rate

R = Path(__file__).resolve().parent
O = R / "results"
F = R / "reasoning_first/results"
NAMES = {"all10": "All ten", "toxicity4": "Toxicity / unsafe content",
         "hallucination3": "Hallucination", "other_safety3": "Other safety / stakes"}
JUDGES = {"gemma": "Gemma", "glm": "GLM", "mistral": "Mistral",
          "olmo": "OLMo", "qwen": "Qwen 27B", "qwen122": "Qwen 122B"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(x):
    return x.to_html(index=False, border=0, escape=True)


def main():
    primary = json.loads((O / "READING_STATUS.json").read_text())
    first = json.loads((F / "READING_STATUS.json").read_text())
    assert primary["all_semantic_work_complete"] and first["contrast_census_complete"]
    assert primary["contrast_records_read"] == 12219
    assert primary["similarity_harm_records_read"] == 6413
    assert first["contrast_records_read"] == 12220

    c = pd.read_json(F / "coded_trace_metadata.jsonl.gz", lines=True)
    c = c[c.requested_census].copy()
    m = pd.read_csv(R / "corpus/metadata.csv.gz", low_memory=False)
    keys = ["backbone", "sample_id", "judge"]
    checks = ["true_label", "control_probability", "nn_probability", "dataset", "split"]
    c = c.merge(m[keys + ["w", "cell"] + checks], on=keys, how="left",
                suffixes=("", "_old"), validate="one_to_one")
    assert len(c) == 12220 and c.w.notna().all()
    for k in checks:
        assert c[k].eq(c[k + "_old"]).all(), k
    c["uses_reference"] = c.relation.isin(["S", "C", "B"])
    c["explicit_contrast"] = c.relation.isin(["C", "B"])
    c["same_class_transfer"] = c.bridge.eq("L")
    c["different_class_from_difference"] = c.bridge.eq("D")
    c["criterion_with_reference"] = c.bridge.eq("R")
    features = ["uses_reference", "explicit_contrast", "same_class_transfer",
                "different_class_from_difference", "criterion_with_reference"]
    distributions, comparisons, conflicts = [], [], []
    scopes = [("all10", c)] + list(c.groupby("family")) + list(c.groupby("dataset"))
    for scope, x in scopes:
        for judge in ["all"] + sorted(x.judge.unique()):
            z = x if judge == "all" else x[x.judge.eq(judge)]
            for retention in ["all", "retained", "changed"]:
                q = z if retention == "all" else z[
                    z.score_valid & z.retains_original_nn.eq(retention == "retained")]
                if not len(q):
                    continue
                for feature in features:
                    distributions.append(dict(scope=scope, judge=judge, retention=retention,
                        feature=feature, n=len(q), marked=int(q[feature].sum()),
                        weighted_rate=weighted_rate(q, feature)))
                    comparisons.append(dict(scope=scope, judge=judge, retention=retention,
                        feature=feature, **compare(q, feature)))
                for exclude50 in [False, True]:
                    a = q[q.score_valid & q.prose_assessable]
                    if exclude50:
                        a = a[a.reason_probability.ne(50)]
                    conflicts.append(dict(scope=scope, judge=judge, retention=retention,
                        exclude50=exclude50, all_n=len(q), valid_clear_n=len(a),
                        conflicts=int(a.score_prose_conflict.sum()),
                        weighted_rate=weighted_rate(a, "score_prose_conflict")))
    dist = pd.DataFrame(distributions)
    cmp = pd.DataFrame(comparisons)
    conf = pd.DataFrame(conflicts)
    dist.to_csv(F / "completed_census_distributions.csv", index=False)
    cmp.to_csv(F / "completed_census_helpful_harmful.csv", index=False)
    conf.to_csv(F / "completed_census_score_prose.csv", index=False)
    (F / "COMPLETED_CENSUS_METHOD.json").write_text(json.dumps(dict(
        read_scope="All 12220 saved contrast-labelled reasoning-first records, including unscored prose.",
        join="One-to-one backbone/sample_id/judge; gold, old control/NN scores, dataset and split verified.",
        weights="Original population inverse-inclusion weights normalized to equal evaluation-cell mass; condition on subgroup.",
        uncertainty="Target-cluster delta-method; conditional on frozen outcomes and semantic codes, not coding uncertainty.",
        retention="Retained/changed require valid new numeric answer; unscored outputs remain in all-prose census.",
        interpretation="Expressed comparisons, not evidence of internal computation or contextual correctness.",
        readers="Assistant-agent readings; exact-text code reuse does not assert equal contextual validity.",
        new_calls=0,
    ), indent=2) + "\n")

    den = pd.read_csv(O / "original_outcome_denominators.csv")
    rows = []
    for family in NAMES:
        q = den[(den.policy == "judge50_direct") & (den.judge == "all") & (den.scope == family)]
        row = {"Family": NAMES[family]}
        for subset, name in [("similarity_yes", "Similarity"), ("contrast_yes", "Contrast")]:
            a = q[q.subset == subset].iloc[0]
            row[name + ": helpful among flips"] = f"{100*a.weighted_helped_among_flips:.1f}%"
            row[name + ": unchanged"] = f"{100*a.weighted_unchanged_fraction:.1f}%"
        rows.append(row)
    outcome_table = table(pd.DataFrame(rows))

    audit = pd.read_csv(O / "context_audit/summary.csv")
    rows = []
    for arm, label in [("contrast", "Contrast"), ("similarity_only", "Similarity without contrast")]:
        row = {"Audit stratum": label}
        for outcome, label2 in [("helped", "Originally helpful"), ("harmed", "Originally harmful")]:
            a = audit[(audit.arm == arm) & (audit.scope == "all10") &
                      (audit.retention == "all") & (audit.original_outcome == outcome) &
                      (audit.feature == "any_context_or_rule_error")].set_index("coding")
            lo, hi, n = int(a.loc["both", "marked"]), int(a.loc["either", "marked"]), int(a.loc["both", "n"])
            row[label2] = f"{lo}–{hi} / {n} ({lo/n:.1%}–{hi/n:.1%})"
        rows.append(row)
    audit_table = table(pd.DataFrame(rows))

    pair = pd.read_csv(F / "paired_answer_order_comparison.csv")
    q = pair[(pair.subset == "contrast_yes") & (pair.family == "all10") &
             (pair.original_outcome == "all") & (pair.judge != "all")].copy()
    rows = []
    for judge, name in JUDGES.items():
        a = q[q.judge == judge].iloc[0]
        rows.append({"Judge": name, "Valid paired cases": int(a.both_valid_rows),
            "Probability first": f"{100*a.weighted_after_accuracy:.2f}%",
            "Explanation first": f"{100*a.weighted_first_accuracy:.2f}%",
            "Difference (pp)": f"{100*a.first_minus_after:+.2f}",
            "Approx. 95% interval": f"{100*a.lower:+.2f} to {100*a.upper:+.2f}"})
    order_table = table(pd.DataFrame(rows))

    rows = []
    for family in NAMES:
        q = dist[(dist.scope == family) & (dist.judge == "all") & (dist.retention == "all")].set_index("feature")
        a = conf[(conf.scope == family) & (conf.judge == "all") &
                 (conf.retention == "all") & (~conf.exclude50)].iloc[0]
        rows.append({"Family": NAMES[family], "All saved contrast prose": int(q.iloc[0]["n"]),
            "Explicit neighbour difference": f"{100*q.loc['explicit_contrast','weighted_rate']:.1f}%",
            "Same-class transfer": f"{100*q.loc['same_class_transfer','weighted_rate']:.1f}%",
            "Probability–prose conflict": f"{100*a.weighted_rate:.1f}% ({int(a.conflicts)}/{int(a.valid_clear_n)} raw)"})
    census_table = table(pd.DataFrame(rows))
    rows = []
    for family in NAMES:
        a = cmp[(cmp.scope == family) & (cmp.judge == "all") &
                (cmp.retention == "all") & (cmp.feature == "explicit_contrast")].iloc[0]
        rows.append({"Family": NAMES[family],
            "Originally helpful flips": f"{100*a.helped_rate:.1f}% (n={int(a.helped_n):,})",
            "Originally harmful flips": f"{100*a.harmed_rate:.1f}% (n={int(a.harmed_n):,})"})
    contrast_mentions_table = table(pd.DataFrame(rows))
    families = c.groupby("family").dataset.unique().map(lambda a: ", ".join(sorted(a))).to_dict()
    family_text = "; ".join(f"{NAMES[k]}: {v}" for k, v in families.items())

    content = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>What the complete NN rationale audit tells us</title>
<style>body{{font:17px/1.6 system-ui,sans-serif;color:#203448;background:#f5f8fb;margin:0}}
main{{max-width:1180px;margin:auto;padding:44px 28px}}h1{{font-size:38px;line-height:1.2}}h2{{margin-top:42px;line-height:1.3}}
.lead{{font-size:22px}}.box,figure{{padding:22px;background:white;border:1px solid #dbe4ec;border-radius:10px;margin:20px 0}}
.small,figcaption{{font-size:14px;color:#536778}}table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:11px;border-bottom:1px solid #dbe4ec;vertical-align:top}}th{{background:#eaf1f6}}
.scroll{{overflow-x:auto}}img{{width:100%;height:auto}}a{{color:#176397}}li{{margin:9px 0}}strong{{color:#122d44}}</style>
</head><body><main>
<p class="small">Interpretable probe · completed saved-response investigation · 20 October 2026</p>
<h1>Nearest neighbours help when the analogy preserves the relevant rule</h1>
<p class="lead">The strongest explanation is local clarification of what should count as the positive class.
A neighbour can supply a useful precedent or a useful exception. It can also supply the wrong intent,
facts, or standard. The audit supports these possibilities, but does not establish a clean causal
division between “good contrast” and “bad contrast.”</p>
<div class="box"><strong>The requested reading is complete.</strong>
<ul><li>Probability first: all <strong>12,219 available contrast responses</strong> and all
<strong>6,413 available similarity-labelled harmful responses</strong>; 16,880 distinct texts after deduplication.
One selected contrast response and one selected similarity-harm response have no saved prose.</li>
<li>Explanation first: all <strong>12,220 contrast responses</strong>, including capped/unscored prose;
11,854 newly read distinct texts plus 26 exact prior-text code reuses.</li>
<li>Auxiliary collections: all <strong>1,192 additional originally unchanged test contrast records</strong>
and all <strong>448 native-reasoning contrast records</strong>. These separate interventions are not pooled.</li>
<li>Context checks: 200 balanced primary cases read twice by separate assistant agents; another
61 reasoning-first trace/context pairs, 14 broad-collection traces, and 28 native-pilot traces
were checked against full saved inputs. The latter are purposive illustrations, not prevalence estimates.</li></ul>
<p>No new model calls, retries, inference restarts, score changes or relabelling were performed.</p></div>

<h2>The original outcome pattern</h2>
<p>The remembered “80% help” refers to the helpful share <em>among decisions that change</em> in
similarity-labelled cases. Most decisions do not change. Contrast is a more difficult and heterogeneous subset.</p>
<div class="scroll">{outcome_table}</div>
<p class="small">Original probability-only answers, ten judges, common judge cutoff 50, before adding the rationale request.
Design-weighted, equal evaluation-cell population mass; conditional percentages. Similarity and contrast can overlap.
These are direct-judge decisions on the sampled cases, not whole-population pipeline F1.</p>
<figure><img src="original_flip_outcomes.png" alt="Helpful share of original NN flips by family and annotation">
<figcaption>The original outcome comparison, with target-cluster intervals. <a href="original_flip_outcomes.pdf">PDF</a>.</figcaption></figure>

<h2>What appears to distinguish helpful and harmful comparisons</h2>
<div class="scroll"><table><thead><tr><th>Proposed mechanism</th><th>Useful application</th><th>Failure observed in saved text</th></tr></thead><tbody>
<tr><td>Class convention / ambiguity</td><td>A similar labelled example clarifies whether an indirect stereotype,
sexual context, or adversarial request belongs to this dataset’s positive class.</td><td>The judge imports a sexual,
malicious, or manipulative intention absent from the target. Following the label can improve benchmark agreement
without establishing that the interpretation is uniquely justified.</td></tr>
<tr><td>Decision-relevant contrast</td><td>Both examples concern a sensitive topic, but differ in endorsement,
consent, victim, actionability, or what the assistant actually provides.</td><td>A shared topic overrides the
difference; alternatively, “less severe than the neighbour” is treated as “negative,” although both can satisfy the rule.</td></tr>
<tr><td>Keeping target and neighbour separate</td><td>The reference illustrates a rule while the target is assessed
on its own content.</td><td>Neighbour-only facts, intentions or instructions are attributed to the target. Some
dataset-correct answers still have this faulty explanation.</td></tr>
<tr><td>Applying the correct task</td><td>Judge the supplied assistant response, or its factual support, or its
potential consequences, as specified.</td><td>Judge the user’s request instead; treat a disclaimer as a refusal;
require an immediate emergency for high stakes; or excuse a factual error because most of an answer is correct.</td></tr>
</tbody></table></div>

<h2>Toxicity: the class boundary is often interpretive</h2>
<p>Here your original intuition fits best: a reference can clarify the intended category for a joke, quotation,
roleplay, indirect stereotype, or ambiguous request. A useful contrast separates sensitive subject matter from
harmful intent or endorsement. But the same process can overinterpret a benign target using its neighbour’s
more explicit content.</p>
<p>In a source-checked ToxicChat case, OLMo distinguishes an educational singing question from harmful intent
and retains a helpful <strong>98 → 10</strong> flip. In another, GLM attributes the neighbour’s sexual character
traits and safety-bypass instructions to a romantic-roleplay target, retaining a harmful <strong>0 → 100</strong>
flip. These are inspectable examples, not estimates of how frequently each mechanism occurs.</p>
<p>The native Aegis and BeaverTails checks add an important qualification: a judge sometimes explicitly chooses
what it infers to be the dataset convention over its own benign reading. Another judge prioritises the written
rule and disagrees with the benchmark. Thus “closer to the dataset labels” and “better semantic reasoning”
are related but distinct outcomes.</p>

<h2>Hallucination: the neighbour must not replace source verification</h2>
<p>Source evidence is already supplied in these tasks. Retrieval does not itself establish whether a response
is faithful to that evidence. A neighbour may demonstrate an error type—wrong entity, wrong relation,
unsupported addition—but the current answer still needs checking against its own source.</p>
<p>The failures include mixing the neighbour’s facts into the current example, inventing a stricter or looser
standard, and letting one salient comparison distract from another contradiction. In a checked RAGTruth
trace, a judge correctly distinguishes the target’s free Wi-Fi from a neighbour’s Wi-Fi error, then overlooks
a contradiction in the target’s opening hours. That trace hits the cap and has no valid final score; it is
evidence about the expressed argument, not a measured corrected or undone answer.</p>
<p>A separate RAGTruth case receives the dataset-correct negative answer while its prose discusses the
neighbour’s FBI/horseback story rather than the target’s NYPD/Uber story. This is why correctness alone cannot
validate a proposed mechanism. Some checked cases also expose label/rubric tensions; their original labels
remain unchanged.</p>

<h2>Other safety and high stakes: scope and severity matter</h2>
<p>Many disagreements concern which object is being classified. A harmful user request followed by a safe
assistant response is different from a response that supplies harmful instructions after a warning.
High stakes concerns potential substantial consequences, not only emergencies or malicious intent.</p>
<p>For the same clinical-record target and the same dinner-related neighbour, one judge treats passive
medical discussion as low stakes, while another recognises consequential clinical information processing.
In the Halloween cases, some traces dismiss risk because the setting is recreational; others notice unsafe
advice in the assistant’s response. These examples implicate criterion selection and attention to the
whole conversation, not simply distance in embedding space.</p>
<p>One repeatedly used high-stakes neighbour itself contains mixed drafts and editing chatter. Its source
defect is documented rather than silently cleaned. It occurs in many selected traces, so those repetitions
are not independent evidence for a general mechanism.</p>

<h2>Does the audit cleanly confirm the hypothesis?</h2>
<p><strong>No: it supports concrete failure modes more strongly than a clean two-way taxonomy.</strong>
The balanced, outcome-masked primary source audit found the following text or task-rule problems:</p>
<div class="scroll">{audit_table}</div>
<p class="small">Range = flagged by both assistant readers through flagged by either reader; not a confidence
interval. Contrast has six helpful and six harmful cases per dataset; similarity-only has four of each.
These small balanced strata are not population-prevalence estimates. Original outcomes precede the rationale prompt;
the linked audit also restricts to cases where the new answer retained the original NN decision.</p>
<p>Similarity failures show a clearer separation: harmful cases more often contain factual, attribution,
stance or criterion problems. For contrast, such problems also appear frequently in helpful cases.
Conversely, some benchmark-wrong answers have a defensible reading. Simply labelling “important contrast”
therefore misses the direction, validity and applicability of the comparison.</p>
<figure><img src="context_audit_comparison.png" alt="Balanced source audit of helpful and harmful contrast and similarity cases">
<figcaption>Filled endpoints: both readers. Open endpoints: either reader. <a href="context_audit_comparison.pdf">PDF</a>.</figcaption></figure>

<h2>The completed explanation-first census</h2>
<div class="scroll">{census_table}</div>
<p class="small">Feature percentages use all available contrast prose, including unscored outputs.
Probability–prose disagreement conditions on a valid numeric score and a clear positive/negative prose conclusion;
the parentheses give unweighted counts. A stated comparison is not certified as correct, and a target-only
rationale does not establish that the neighbour had no influence. Mixed/unclear prose is retained separately.</p>
<p>Only a minority explicitly states a neighbour difference even though every row was independently
labelled contrast. Many explanations simply restate a target classification. More reasoning text is not
automatically more faithful evidence of why an earlier prediction changed.</p>
<p>In fact, explicit neighbour differences are stated more often in the originally harmful group in
each family:</p><div class="scroll">{contrast_mentions_table}</div>
<p class="small">Weighted prevalence of an expressed difference, not its correctness. Group membership
uses the original control-versus-NN answers; later explanation-first answers can change. The downloads
also restrict to valid answers retaining the original decision. This association does not show that
stating a contrast causes harm.</p>
<p><a href="../reasoning_first/results/completed_census_helpful_harmful.csv">All helpful/harmful expressed-code comparisons</a> ·
<a href="../reasoning_first/results/completed_census_distributions.csv">Distributions by dataset, family and judge</a> ·
<a href="../reasoning_first/results/completed_census_score_prose.csv">Score–prose consistency, including exclusion of exact 50</a>.</p>

<h2>Putting the explanation first often repairs answers, but changes the experiment</h2>
<p>Among the <strong>10,410 contrast cases with valid answers in both orders</strong>, weighted accuracy
rises from <strong>53.97% to 67.11%</strong>: +13.14 percentage points, approximate target-cluster interval
<strong>+11.31 to +14.97</strong>. In raw counts, 2,794 wrong probability-first answers become correct and
1,542 correct answers become wrong. These compare the two explanation prompts, not control versus NN.</p>
<div class="scroll">{order_table}</div>
<p>The conditional gain is +6.80 points in toxicity, +5.53 in hallucination and +21.92 in other safety.
However, 1,809 contrast explanation-first outputs are unscored, versus one probability-first output.
The later collection generally allowed 512 rather than 256 tokens, retaining some earlier capped settings.
This is not a clean isolation of output order, and the percentages are not deployment accuracy.
The <a href="../reasoning_first/results/report.html">paired report</a> gives missing-answer bounds and all full-source illustrations.</p>

<h2>How the traces fit the numerical evidence</h2>
<ul><li><strong>Contrast does not simply push the answer away from the neighbour.</strong> In the earlier
toxicity analysis, 83.2% of harmful contrast flips moved toward the neighbour’s class. Contrast cases
also had substantially more target–neighbour label disagreement.</li>
<li><strong>Direction and opportunity matter.</strong> When neighbour and target labels differed,
contrast-labelled cases showed less harmful following and more useful resistance. When labels aligned,
they also showed less useful following. The label alone compresses these distinct situations.</li>
<li><strong>Cosine is not an LLM decision-boundary distance.</strong> The smaller probe margins and lower
cosine similarities in contrast cases describe correlations. Within-target comparisons attenuated
the association, and the threshold policy changed the net result.</li>
<li><strong>The policy sensitivity remains.</strong> Original toxicity contrast net accuracy gain was
about +0.99 points for direct judge decisions at 50, +0.45 for the 50/50 pipeline, and −0.12 averaged
over the three shared tuned anchors, with the latter interval crossing zero. Those are different estimands;
the fixed-50 rationale selection does not resolve the tuned-policy question.</li>
<li><strong>Always showing NN remains a strong practical baseline.</strong> The small selector improvements
do not establish whether training data or missing mechanisms are the limiting factor. A learning curve
or controlled feature intervention would be needed to distinguish them.</li></ul>
<p><a href="../../nn_contrast_investigation_20261020/results/report.html">Earlier quantitative investigation:
opportunities, cosine, margins, ablations, 2-NN and policy sensitivity</a>.</p>

<h2>What to take into the paper</h2>
<p>A defensible interpretation is: <strong>nearest-neighbour explanations act as local labelled precedents.
They help when they clarify the applicable category or a relevant exception, and can hurt through
misapplied analogy, source confusion, or criterion drift.</strong> The evidence for dataset conventions is
especially clear in toxicity. Hallucination and high-stakes tasks add evidence-verification and task-scope failures.</p>
<p>The transcripts illustrate these mechanisms but do not establish private reasoning or causal mediation.
They were generated under a changed prompt, can contradict their own numeric answer, and sometimes
rationalise an existing label. Source-checked examples are stronger than unverified prose, while the
balanced audit does not yield a clean contrast-success classifier.</p>
<p>The most informative next experiment would keep the target fixed and alter one relevant neighbour
property—endorsement, actor, consent, severity or factual error—while holding wording as constant as possible.
Include irrelevant and label/prediction-conflict references, score numeric decisions separately from prose,
and evaluate a preregistered rule on fresh examples. This is a proposal only; no new collection was launched.</p>

<h2>Audit trail and full examples</h2>
<ul>
<li><a href="report.html">Primary complete census, balanced 200-case source audit and searchable full examples</a></li>
<li><a href="../reasoning_first/results/report.html">Complete explanation-first contrast census and 61 full-context cards</a></li>
<li><a href="../broad_contrast/results/report.html">Complete auxiliary unchanged-case census</a>:
not representative; overwhelmingly Qwen 27B and distinct sampling.</li>
<li><a href="../broad_contrast/context_followup/report.html">Auxiliary factuality and task-convention examples</a> ·
<a href="../broad_contrast/context_safety_followup/report.html">auxiliary safety source checks</a></li>
<li>Native pilot: <a href="../native_contrast/context_aegis_followup/report.html">Aegis</a> ·
<a href="../native_contrast/context_beavertails_followup/report.html">BeaverTails</a> ·
<a href="../native_contrast/context_wildguard_followup/report.html">WildGuard</a> ·
<a href="../native_contrast/context_wildjailbreak_followup/report.html">WildJailbreak</a>.
Separate modes and repeated timing observations are not independent replicas of the main experiment.</li>
<li><a href="READING_STATUS.json">Primary coverage</a> ·
<a href="../reasoning_first/results/READING_STATUS.json">Explanation-first coverage</a> ·
<a href="../reasoning_first/results/COMPLETED_CENSUS_METHOD.json">New aggregation method</a> ·
<a href="SYNTHESIS_BUILD.json">Synthesis source manifest</a></li></ul>
<p class="small">{family_text}. Dataset groupings are analytic labels, not claims that all tasks measure the same construct.
All quoted correctness judgments refer to the frozen benchmark labels unless explicitly qualified.
Static artifact checks do not claim browser rendering.</p>
</main></body></html>"""
    (O / "synthesis.html").write_text(content)
    sources = [
        O / "READING_STATUS.json", F / "READING_STATUS.json",
        O / "original_outcome_denominators.csv", O / "context_audit/summary.csv",
        F / "paired_answer_order_comparison.csv", F / "coded_trace_metadata.jsonl.gz",
        R / "corpus/metadata.csv.gz",
    ]
    (O / "SYNTHESIS_BUILD.json").write_text(json.dumps(dict(
        built_utc=datetime.now(timezone.utc).isoformat(),
        probability_first=primary, reasoning_first=first,
        source_sha256={str(p.relative_to(R)): sha(p) for p in sources},
        source_preserved=True, no_new_inference=True,
        final_verification_pending=True,
        output_sha256=sha(O / "synthesis.html"),
    ), indent=2) + "\n")
    print(O / "synthesis.html")


if __name__ == "__main__":
    main()
