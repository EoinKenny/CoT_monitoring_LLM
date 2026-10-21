"""Reproducible figures and report from completed local census and context audit."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nn_rationale_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
O = R / "results"
S = O / "semantic_census"
C = O / "context_audit"
FAMILIES = ["toxicity4", "hallucination3", "other_safety3"]
FAMILY_NAMES = {"toxicity4": "Toxicity / unsafe content", "hallucination3": "Hallucination",
                "other_safety3": "Other safety / stakes", "all10": "All ten datasets"}
JUDGES = {"gemma": "Gemma", "glm": "GLM", "mistral": "Mistral",
          "olmo": "OLMo", "qwen": "Qwen 27B", "qwen122": "Qwen 122B"}
BLUE, RED, GRAY, TEAL = "#2876a8", "#bc584e", "#a4adb5", "#278879"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(O / (name + ".png"))
    fig.savefig(O / (name + ".pdf"))
    plt.close(fig)


def table(df, columns=None):
    return df[columns].to_html(index=False, escape=True, border=0) if columns else df.to_html(index=False, escape=True, border=0)


def pct(x):
    return f"{100*x:.1f}%"


def main():
    status = json.load(open(O / "READING_STATUS.json"))
    assert status["all_semantic_work_complete"], "Report waits for all local assigned readings"
    assert (C / "CODING_FREEZE.json").exists()
    den = pd.read_csv(O / "original_outcome_denominators.csv")
    cmp = pd.read_csv(S / "helpful_harmful_codes.csv")
    dist = pd.read_csv(S / "code_distributions.csv")
    conflict = pd.read_csv(S / "score_prose_conflicts.csv")
    audit = pd.read_csv(C / "summary.csv")
    retain = pd.read_csv(O / "rationale_answer_retention.csv")
    inventory = json.load(open(R / "inventory.json"))
    original = den[(den.policy == "judge50_direct") & (den.judge == "all") &
                   den.scope.isin(FAMILIES) & den.subset.isin(["similarity_yes", "contrast_yes"])]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    yy = np.arange(3)
    for subset, offset, color, label in [
        ("similarity_yes", -.16, TEAL, "Labelled similarity"),
        ("contrast_yes", .16, BLUE, "Important contrast"),
    ]:
        q = original[original.subset == subset].set_index("scope").loc[FAMILIES]
        x = q.weighted_helped_among_flips.to_numpy() * 100
        err = np.array([x - q.lower.to_numpy()*100, q.upper.to_numpy()*100-x])
        ax.errorbar(x, yy+offset, xerr=err, fmt="o", capsize=3, color=color, label=label)
        for i, v in enumerate(x):
            ax.text(v+.9, i+offset-.05, f"{v:.1f}%", color=color, fontsize=10)
    ax.axvline(50, color=GRAY, ls="--", lw=1)
    ax.set_yticks(yy); ax.set_yticklabels([FAMILY_NAMES[f] for f in FAMILIES])
    ax.invert_yaxis(); ax.set_xlim(35, 97)
    ax.set_xlabel("Helpful share of changed decisions (%)")
    ax.set_title("Similarity and contrast have different outcome distributions\nOriginal probability-only answers · 10 judges · common 50 cutoff")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); save(fig, "original_flip_outcomes")

    selected_features = ["explicit_contrast", "bridge_L", "bridge_R", "score_prose_conflict"]
    feature_names = ["Explicit neighbour difference", "Same-class transfer",
                     "Criterion applied with neighbour", "Probability–prose disagreement"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        q = cmp[(cmp.scope == family) & (cmp.judge == "all") & (cmp.retention == "all")].set_index("feature")
        for f, y in zip(selected_features, range(len(selected_features))):
            row = q.loc[f]
            ax.plot([100*row.helped_rate,100*row.harmed_rate], [y,y], color="#d9e0e6", lw=3)
            ax.scatter(100*row.helped_rate, y, color=BLUE, s=45)
            ax.scatter(100*row.harmed_rate, y, color=RED, s=45)
        ax.set_title(FAMILY_NAMES[family]); ax.set_xlabel("Rationales expressing feature (%)")
        ax.set_xlim(0, 55); ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(np.arange(len(feature_names))); axes[0].set_yticklabels(feature_names)
    axes[0].invert_yaxis()
    axes[-1].scatter([],[],color=BLUE,label="Original NN helped")
    axes[-1].scatter([],[],color=RED,label="Original NN harmed")
    axes[-1].legend(loc="lower right",frameon=False)
    fig.suptitle("What the saved explanations actually say in contrast-labelled cases\nSix local judges; original outcomes, before adding the explanation request", y=1.03)
    fig.tight_layout(); save(fig, "expressed_mechanism_patterns")

    fig, ax = plt.subplots(figsize=(10.5,5))
    q = conflict[(conflict.subset=="contrast") & (conflict.scope=="all10") &
                 (conflict.retention=="all") & (conflict.original_outcome=="all") &
                 (conflict.judge!="all")]
    for exclude, offset, color, label in [(False,-.16,RED,"All clear prose"),
                                          (True,.16,BLUE,"Exclude probability 50")]:
        z=q[q.exclude50==exclude].set_index("judge").reindex(JUDGES)
        ax.barh(np.arange(6)+offset, 100*z.conflict_rate.to_numpy(), height=.29, color=color, label=label)
    ax.set_yticks(np.arange(6)); ax.set_yticklabels(list(JUDGES.values())); ax.invert_yaxis()
    ax.set_xlabel("First probability disagrees with the prose class (%)")
    ax.set_title("A written explanation does not always explain the recorded answer\nContrast cases with an assessable prose conclusion")
    ax.legend(frameon=False); fig.tight_layout(); save(fig, "probability_prose_disagreement")

    audit_features = ["any_context_or_rule_error", "imports_neighbour_facts_or_intent",
                      "misses_stance_negation_or_role", "changes_task_criterion"]
    audit_names = ["Text or task-rule problem", "Neighbour details imported",
                   "Stance / negation / role missed", "Task criterion changed"]
    fig, axes=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
    for ax,family in zip(axes,FAMILIES):
        for outcome,offset,color,label in [("helped",-.14,BLUE,"Original NN helped"),
                                           ("harmed",.14,RED,"Original NN harmed")]:
            z=audit[(audit.arm=="contrast")&(audit.scope==family)&
                    (audit.retention=="all")&(audit.original_outcome==outcome)]
            for i,f in enumerate(audit_features):
                t=z[z.feature==f].set_index("coding")
                lo=float(t.loc["both","balanced_sample_rate"])*100
                hi=float(t.loc["either","balanced_sample_rate"])*100
                ax.plot([lo,hi],[i+offset,i+offset],color=color,lw=4,alpha=.65)
                ax.scatter(hi,i+offset,facecolors="white",edgecolors=color,s=32)
                ax.scatter(lo,i+offset,color=color,s=18,zorder=4)
            ax.plot([],[],color=color,lw=3,label=label)
        ax.set_title(FAMILY_NAMES[family]);ax.set_xlabel("Balanced audit cases (%)");ax.set_xlim(-3,103)
    axes[0].set_yticks(np.arange(len(audit_names)));axes[0].set_yticklabels(audit_names);axes[0].invert_yaxis()
    axes[-1].legend(frameon=False,loc="lower right")
    fig.suptitle("Full-context checks: helpful and harmful contrast cases\nFilled = both readers flag it; open = either reader · ranges are coding sensitivity, not confidence intervals",y=1.04)
    fig.tight_layout();save(fig,"context_audit_patterns")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, arm, title in zip(axes, ["contrast", "similarity_only"],
                              ["Important contrast", "Similarity without contrast"]):
        for outcome, offset, color in [("helped", -.14, BLUE), ("harmed", .14, RED)]:
            for i, family in enumerate(FAMILIES):
                z = audit[(audit.arm == arm) & (audit.scope == family) &
                          (audit.retention == "all") & (audit.original_outcome == outcome) &
                          (audit.feature == "any_context_or_rule_error")].set_index("coding")
                lo, hi = [100 * float(z.loc[k, "balanced_sample_rate"]) for k in ["both", "either"]]
                ax.plot([lo, hi], [i + offset, i + offset], color=color, lw=4, alpha=.6)
                ax.scatter(hi, i + offset, facecolors="white", edgecolors=color, s=38)
                ax.scatter(lo, i + offset, color=color, s=22, zorder=4)
            ax.plot([], [], color=color, label=f"Original NN {outcome}")
        ax.set_title(title); ax.set_xlabel("Cases with text or task-rule problems (%)")
        ax.set_xlim(-4, 104); ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(np.arange(3))
    axes[0].set_yticklabels([FAMILY_NAMES[f] for f in FAMILIES])
    axes[0].invert_yaxis(); axes[-1].legend(frameon=False)
    fig.suptitle("The context audit separates similarity failures more clearly than contrast failures\n"
                 "Balanced 200-case audit · filled = both readers; open = either reader", y=1.03)
    fig.tight_layout(); save(fig, "context_audit_comparison")

    audit_table_rows = []
    for arm, name in [("contrast", "Contrast"), ("similarity_only", "Similarity without contrast")]:
        row = {"Case group": name}
        for outcome, label in [("helped", "Helpful flips"), ("harmed", "Harmful flips")]:
            z = audit[(audit.arm == arm) & (audit.scope == "all10") &
                      (audit.retention == "all") & (audit.original_outcome == outcome) &
                      (audit.feature == "any_context_or_rule_error")].set_index("coding")
            lo, hi, n = int(z.loc["both", "marked"]), int(z.loc["either", "marked"]), int(z.loc["both", "n"])
            row[label] = f"{lo}–{hi} of {n} ({lo/n:.1%}–{hi/n:.1%})"
        audit_table_rows.append(row)
    audit_error_table = table(pd.DataFrame(audit_table_rows))

    # Inspectable private research examples, including all independently read contexts.
    packets={}
    for p in (R/"context_audit/packets").glob("*.jsonl"):
        for line in p.read_text().splitlines():
            x=json.loads(line);packets[x["audit_id"]]=x
    cases=pd.read_csv(C/"paired_reader_cases.csv")
    records=pd.read_json(C/"individual_reader_records.jsonl",lines=True)
    browser=[]
    for x in cases.to_dict("records"):
        p=packets[x["audit_id"]]
        cc=records[records.audit_id==x["audit_id"]].to_dict("records")
        browser.append(dict(
            audit_id=x["audit_id"],dataset=x["dataset"],family=x["family"],
            arm=x["arm"],outcome=x["original_outcome"],judge=x["judge"],
            backbone=x["backbone"],control=x["control"],nn=x["original_nn"],
            new_probability=x["new_probability"],gold=x["gold"],retained=bool(x["retains"]),
            task=p["task"],context=p["context"],rationale=p["rationale"],
            readings=[{k:c[k] for k in ["reader","context_support","task_rule","flags","note","evidence"]} for c in cc],
        ))
    (O/"audited_examples.json").write_text(json.dumps(browser,ensure_ascii=False,indent=2)+"\n")
    examples_js=json.dumps(browser,ensure_ascii=False).replace("</","<\\/")

    coverage=pd.read_csv(S/"coverage.csv")
    q=conflict[(conflict.subset=="contrast")&(conflict.scope=="all10")&
               (conflict.retention=="all")&(conflict.original_outcome=="all")&
               (~conflict.exclude50)&(conflict.judge!="all")].copy()
    q["Judge"]=q.judge.map(JUDGES);q["Disagree"]=q.conflict_rate.map(pct)
    scoretable=table(q,["Judge","all_n","assessable_n","conflict_n","Disagree"])
    r=retain[(retain.scope=="all10")&(retain.subset=="contrast_yes")&(retain.judge=="all")]
    rt=r[["original_outcome","rows","retained","reversed","weighted_retained_fraction"]].copy()
    rt["weighted_retained_fraction"]=rt.weighted_retained_fraction.map(pct)
    d=original[["scope","subset","rows","helped","harmed","unchanged","weighted_helped_among_flips","weighted_unchanged_fraction"]].copy()
    for col in ["weighted_helped_among_flips","weighted_unchanged_fraction"]:d[col]=d[col].map(pct)
    agreements=pd.read_csv(C/"agreement.csv")
    base=R.parents[0]/"nn_contrast_investigation_20261020/results/report.html"
    content=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Why nearest neighbours help or hurt</title>
<style>
:root{{--ink:#173047;--muted:#536779;--line:#dce5ed}}body{{margin:0;background:#f5f8fb;color:var(--ink);font:17px/1.55 system-ui,sans-serif}}
main{{max-width:1220px;margin:auto;padding:48px 28px}}h1{{font-size:40px;line-height:1.13}}h2{{margin-top:48px;font-size:28px}}h3{{margin-top:28px}}
.lead{{font-size:22px;max-width:980px}}.box,article{{background:white;border:1px solid var(--line);border-radius:12px;padding:23px;margin:20px 0}}
.muted,figcaption{{color:var(--muted);font-size:14px}}img{{max-width:100%;height:auto}}figure{{margin:24px 0;background:white;padding:18px;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}}th{{background:#edf3f7}}
.scroll{{overflow-x:auto}}a{{color:#146da4}}pre{{white-space:pre-wrap;font:14px/1.5 system-ui;max-height:460px;overflow:auto;background:#f6f8fa;padding:14px}}
select,input{{padding:10px;margin:5px;border:1px solid #bccbd7;border-radius:6px;font:inherit}}.tag{{display:inline-block;background:#e8eff5;padding:3px 8px;border-radius:6px;font-size:13px;margin-right:6px}}
summary{{cursor:pointer}}.good{{color:#146b94}}.bad{{color:#a94137}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="muted">Interpretable probe · saved-response audit · {datetime.now(timezone.utc).strftime('%d %B %Y')}</p>
<p><a href="synthesis.html">Read the completed integrated investigation →</a></p>
<h1>Why nearest neighbours help—and why they sometimes hurt</h1>
<p class="lead">The traces suggest that neighbours help when they clarify the applicable class rule, and hurt when an analogy changes the target’s meaning or the rule itself. But the full-context audit does <em>not</em> split contrast cases neatly into sound explanations that help and unsound explanations that hurt. The clearest quality difference appears in similarity-only failures.</p>
<div class="box"><b>What was checked</b><p>Every one of the <b>12,219 available contrast-labelled responses</b> in the saved six-local-judge probability-first collection, plus every available similarity-labelled harmful response. The census contains {status['unique_texts_read']:,} distinct full texts after exact dataset-and-text deduplication. One selected contrast response is missing. A separate balanced sample of <b>200 complete task–target–neighbour–rationale cases</b> was read independently twice, with original outcomes hidden.</p>
<p>This is an audit of expressed explanations. The judge produced its probability before the prose. We do not treat the prose as a verified account of its internal computation. All 12,220 reasoning-first contrast responses have now been read and their receipts validated. All 448 contrast records from the separate native-reasoning pilot and timing collections have also been read; these are a different intervention and are not pooled into the primary results. See the <a href="../reasoning_first/results/report.html">completed contrast census, paired answer-order comparison and contextual examples</a>, the <a href="../broad_contrast/results/report.html">complete auxiliary census of 1,192 originally unchanged contrast cases</a>, <a href="../broad_contrast/context_followup/report.html">additional source checks where correct answers have inconsistent explanations</a>, <a href="../native_contrast/context_wildguard_followup/report.html">native-pilot examples where identical neighbours support different readings</a>, <a href="../native_contrast/context_beavertails_followup/report.html">BeaverTails examples separating dataset-label matching from the written safety definition</a>, <a href="../native_contrast/context_aegis_followup/report.html">Aegis examples where the same distinction leads to different use of the neighbour label</a>, and <a href="../native_contrast/context_wildjailbreak_followup/report.html">WildJailbreak examples where a neighbour supports either literal reading or inferred hidden intent</a>. This page retains the primary probability-first analysis; the integrated report combines all completed corpora.</p></div>

<h2>First, the “80% help” figure needs a denominator</h2>
<p>The approximate 80/20 split applies to <b>changed decisions within similarity-labelled cases</b>, pooled across the original ten judges. Most labelled cases do not change the decision at all. Toxicity, hallucination, and other safety tasks also differ substantially. Contrast-labelled flips are much closer to an even split overall.</p>
<figure><img src="original_flip_outcomes.png" alt="Helpful fraction among similarity and contrast flips by dataset family"><figcaption>Original probability-only control versus 1-NN, judge cutoff 50. Design-weighted, equal-cell population weights; target-cluster approximate 95% intervals. Labels can overlap. This panel includes the ten original judges, whereas the rationale census covers six local judges.</figcaption></figure>
<details><summary>Exact denominators, including unchanged cases</summary><div class="scroll">{table(d)}</div></details>

<h2>What separates a useful comparison from a misleading one?</h2>
<div class="scroll"><table><thead><tr><th>Situation</th><th>Useful explanation</th><th>Failure seen in the texts</th></tr></thead><tbody>
<tr><td>Toxicity and unsafe-content conventions</td><td>Shows that an apparently mild joke, stereotype, request, or sexual context belongs to the dataset’s positive class—or that a sensitive topic alone is insufficient.</td><td>Treats shared vocabulary as shared intent; loses negation, quotation, endorsement, the victim, or who is acting. A labelled neighbour supplies a meaning that is absent from the target.</td></tr>
<tr><td>Important contrast</td><td>Identifies a decision-relevant difference and explains how the task criterion applies to the target.</td><td>“Less extreme than the neighbour” becomes “negative,” even when both can meet the positive-class rule. Or the judge notices a difference but follows the neighbour’s label anyway.</td></tr>
<tr><td>Hallucination detection</td><td>Uses the neighbour to illustrate the error type while checking the target against its own evidence.</td><td>Changes the standard for what counts as unsupported, confuses the neighbour’s details with the target’s source, or treats faithful paraphrase/omission as an error. Conversely, “mostly accurate” can excuse a real false detail.</td></tr>
<tr><td>Other safety and high stakes</td><td>Distinguishes request from response, depiction from facilitation, or severe potential consequences from benign discussion.</td><td>Evaluates the harmful user request instead of the final assistant response; treats a disclaimer as a refusal; or requires an actual emergency where the task asks about substantial potential consequences.</td></tr>
</tbody></table></div>
<p>These are recurring expressed patterns and context-checked examples. They are not a validated causal taxonomy, and a defensible reading can still disagree with a dataset label.</p>

<h2>Full-rationale census</h2>
<figure><img src="expressed_mechanism_patterns.png" alt="Expressed features in helpful and harmful contrast rationales"><figcaption>Features are coded from the complete saved prose, with judge identity, first score, and original outcome masked during the reading. “Criterion applied” means an explicit stated rule is used; it does not certify that the rule or factual premise is correct. A target-only explanation can still have been influenced by the neighbour.</figcaption></figure>
<p>Merely mentioning a contrast is not a success criterion. The more useful distinction is whether the comparison preserves the target’s facts and applies the actual class rule. The context audit checks those two properties directly.</p>
<p>Downloads: <a href="semantic_census/helpful_harmful_codes.csv">helpful versus harmful code rates</a> · <a href="semantic_census/code_distributions.csv">all code distributions</a> · <a href="semantic_census/coverage.csv">dataset coverage</a>. Tables include analyses restricted to responses whose new answer retained the original NN decision.</p>

<h2>The probability and the prose sometimes disagree</h2>
<figure><img src="probability_prose_disagreement.png" alt="Probability-prose disagreement by judge"><figcaption>The first number remains the scored answer. A prose claim of “not toxic” with a probability of 50 or more is a disagreement under the agreed positive-at-50 rule. The blue bars exclude exact-50 answers; mixed and unclear prose is reported separately, not counted as agreement.</figcaption></figure>
<div class="scroll">{scoretable}</div>
<p>Adding the explanation request also changed some first probabilities. The following table separates retention of the original NN answer from the wording of the later explanation:</p>
<div class="scroll">{table(rt)}</div>
<p>This is why an eloquent correct explanation attached to an incorrect first number cannot, by itself, explain the original harmful flip.</p>

<h2>Checking the explanations against the actual inputs</h2>
<p><b>The result is mixed, rather than a clean confirmation of our initial hypothesis.</b> In the balanced audit, text or task-rule problems occur at similar rates in helpful and harmful contrast cases. In similarity-only cases, they occur much more often in harmful flips. A correct class can have a faulty explanation, and an incorrect class can have a defensible explanation that disagrees with the dataset’s convention.</p>
<div class="scroll">{audit_error_table}</div>
<p class="muted">Counts span agreement by both readers to a flag by either reader. These are coding-sensitivity bounds, not confidence intervals, and they are not prevalence estimates for all test examples.</p>
<figure><img src="context_audit_comparison.png" alt="Context and task-rule problems in helpful and harmful contrast and similarity-only cases"><figcaption>Contrast: 24 helpful and 24 harmful toxicity cases, and 18 of each in each other family. Similarity-only: 16 of each in toxicity and 12 of each in each other family. Small subgroup counts make these exploratory comparisons. The downloadable tables also restrict to responses retaining the original NN classification.</figcaption></figure>
<figure><img src="context_audit_patterns.png" alt="Independent contextual error coding in helpful and harmful cases"><figcaption>Contrast audit: 6 helpful and 6 harmful cases per dataset. Similarity-only audit: 4 of each per dataset. Cases were randomly sampled by context within these strata, then one saved judge response per context. The shown bars are balanced audit frequencies; they are not population mechanism prevalence estimates.</figcaption></figure>
<p>The two readings are preserved. A filled endpoint means both readers flagged the property; the open endpoint includes cases flagged by either reader. These spans expose coding disagreement and are not statistical confidence intervals. The readers are separate assistant agents, not independent human annotators.</p>
<p><a href="context_audit/summary.csv">All contrast and similarity-only audit comparisons</a> · <a href="context_audit/binary_agreement.csv">binary coding agreement</a> · <a href="context_audit/agreement.csv">categorical agreement</a> · <a href="context_audit/paired_reader_cases.csv">paired case data</a></p>
<details><summary>Reader agreement by coding field</summary><div class="scroll">{table(agreements)}</div></details>

<h2>Concrete differences visible in the saved explanations</h2>
<p>These examples were inspected with their full task, target and neighbour. Both readers identified the stated distinction or error. The new probability-first response retained the original NN classification in each example; that improves comparability but does not make the later rationale a causal explanation of the earlier answer.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Original control → NN</th><th>What the explanation shows</th></tr></thead><tbody>
<tr><td><a href="#ctx_ca937104271e7202a837">ToxicChat · OLMo · helpful contrast</a></td><td>98 → 10; gold 0</td><td>Recognizes a technical singing question as educational, explicitly distinguishing it from the neighbour’s harmful intent. Shared surface wording does not determine the class.</td></tr>
<tr><td><a href="#ctx_bdc56a0a614a028244ac">ToxicChat · GLM · harmful similarity</a></td><td>0 → 100; gold 0</td><td>Quotes sexual character traits and safety-bypass instructions that appear only in the neighbour as if the target romantic roleplay contained them.</td></tr>
<tr><td><a href="#ctx_6b76c6513cac015cfc48">WildJailbreak · Mistral · harmful contrast</a></td><td>0 → 95; gold 0</td><td>Assesses the neighbour’s asset-exploitation instructions instead of the target’s caregiver situation. This is a source mix-up, not simply excessive sensitivity to cosine distance.</td></tr>
<tr><td><a href="#ctx_39beba28f51791560ceb">High stakes · Qwen 27B · harmful contrast</a></td><td>95 → 10; gold 1</td><td>Acknowledges dangerous advice but calls the situation low-stakes because its topic is recreational and harm has not occurred. The task asks about potential substantial consequences.</td></tr>
<tr><td><a href="#ctx_7e5068e602449bcada40">WildGuard · Qwen 122B · helpful contrast</a></td><td>50 → 0; gold 0</td><td>Evaluates the safe assistant response rather than the adversarial user wrapper. The prose does not explicitly use the neighbour, so it supports a sound target assessment rather than proving a neighbour mechanism.</td></tr>
<tr><td><a href="#ctx_4bce2e4351552a9bafb8">WildGuard · Mistral · harmful similarity</a></td><td>50 → 5; gold 1</td><td>Calls a disclaimer followed by substantive harmful instructions a refusal. The decision depends on what the response actually supplies, not just its cautionary language.</td></tr>
<tr><td><a href="#ctx_33d08a6bd571d4ff7064">FAVA · harmful contrast</a></td><td>0 → 50; gold 0</td><td>The expressed factual-error diagnosis concerns the neighbour instead of the target. Checking which passage a claim belongs to is essential.</td></tr>
<tr><td><a href="#ctx_dae6f69a480111ce3669">HaluEval · harmful similarity</a></td><td>100 → 0; gold 1</td><td>Notices an unsupported uniqueness claim but discounts it because the main answer is correct. That adds a “major error” requirement absent from the task.</td></tr>
<tr><td><a href="#ctx_69e094688c815d4bf758">RAGTruth · helpful contrast, faulty explanation</a></td><td>90 → 0; gold 0</td><td>Returns the dataset-correct class, but its entire supporting account concerns the neighbour’s FBI/horseback story rather than the target NYPD/Uber story. A correct label does not certify correct use of the evidence.</td></tr>
</tbody></table></div>

<h2>How this fits the earlier contrast investigation</h2>
<p>The <a href="{html.escape(str(base))}">earlier quantitative report</a> found that most harmful toxicity contrast flips move <em>toward</em> the neighbour’s class, not away from it. Contrast cases also have much more neighbour–target label disagreement, lower cosine similarity, and smaller probe margins. Those differences create different opportunities for helping and harming; the label alone cannot establish a mechanism.</p>
<p>Within the opportunities where the neighbour has the wrong class for the target, contrast-labelled cases show less harmful following and more useful resistance. When their correct classes align, contrast-labelled cases also show less useful following. These associations make “contrast is bad” too coarse a conclusion. The earlier within-target comparisons attenuate the association, and the threshold-policy sensitivity remains important.</p>
<p>Cosine similarity is not a measure of distance from an LLM’s decision boundary. Nor does a lower relative severity establish that the target falls below the task’s positive threshold. The present context audit supplies concrete examples of these distinctions, rather than treating a curve as proof of the judge’s reasoning.</p>

<h2>What we can—and cannot—conclude</h2>
<ul>
<li>The evidence supports useful local examples as a strong default, with context-sensitive failure modes. It does not yet identify a reliable deployment rule for excluding all harmful neighbours.</li>
<li>The two annotation labels omit crucial distinctions: who is acting, what is endorsed, which object is classified, whether a difference crosses the actual class rule, and whether the neighbour’s label is appropriate for this target.</li>
<li>Small selector gains are compatible with limited informative validation disagreements and incomplete features. This audit does not establish which is the limiting factor; that requires a learning curve or a feature intervention.</li>
<li>For hallucination tasks, the evidence passages are already supplied. Retrieval alone does not settle whether a judge faithfully checks a response against them. Here the neighbour can change the interpretation of the task, but it should not substitute for that source check.</li>
<li>These are selected original fixed-50 flips. There is no full-rationale unchanged-decision control in this corpus. We cannot infer how frequently an expressed mechanism occurs across all test cases from this sample alone.</li>
</ul>
<p>A focused next experiment would hold the target fixed and manipulate one decision-relevant property of its neighbour—stance, actor, criterion, or factual error—while preserving as much wording as possible. That would test the proposed distinctions more directly than another undifferentiated round of explanations.</p>

<h2>Inspect all 200 audited examples</h2>
<p>Each record shows both independent readings, the original control/NN probabilities, the new probability, and the full saved rationale. Sensitive benchmark content is quoted only inside the expandable source views.</p>
<label>Family <select id="family"><option value="">All</option>{''.join(f'<option value="{f}">{FAMILY_NAMES[f]}</option>' for f in FAMILIES)}</select></label>
<label>Outcome <select id="outcome"><option value="">Both</option><option value="helped">Helpful</option><option value="harmed">Harmful</option></select></label>
<label>Annotation <select id="arm"><option value="">Both</option><option value="contrast">Contrast</option><option value="similarity_only">Similarity only</option></select></label>
<input id="search" placeholder="Search dataset, judge, text or note"><p id="count" class="muted"></p><div id="examples"></div>
<h2>Provenance and reproducibility</h2>
<p>Primary snapshot SHA-256: <code>{inventory['source_integrity']['final_snapshot_sha256']}</code>. Frozen source data and inference settings were not changed. No additional judge calls were made for this audit. Exact-text deduplication applies only to expressed-word codes; contextual correctness was judged separately for each sampled full input.</p>
<p><a href="READING_STATUS.json">Primary reading coverage</a> · <a href="../CORPUS_MANIFEST.json">primary corpus manifest</a> · <a href="../inventory.json">original source inventory</a> · <a href="../reasoning_first/SOURCE_VERIFICATION.json">retrieved reasoning-first source verification</a> · <a href="../reasoning_first/results/READING_STATUS.json">completed reasoning-first reading coverage</a> · <a href="semantic_census/METHOD.json">census methods</a> · <a href="context_audit/METHOD.json">context-audit methods</a> · <a href="audited_examples.json">all audited examples</a></p>
</main><script>
const data={examples_js};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function render(){{
const family=document.querySelector('#family').value,outcome=document.querySelector('#outcome').value,arm=document.querySelector('#arm').value,query=document.querySelector('#search').value.toLowerCase();
const rows=data.filter(x=>(!family||x.family===family)&&(!outcome||x.outcome===outcome)&&(!arm||x.arm===arm)&&(!query||JSON.stringify(x).toLowerCase().includes(query)));
document.querySelector('#count').textContent=rows.length+' of 200 cases';
document.querySelector('#examples').innerHTML=rows.map(x=>`<article id="${{esc(x.audit_id)}}"><h3>${{esc(x.dataset)}} · ${{esc(x.judge)}} · ${{esc(x.backbone)}}</h3><p><span class="tag">${{esc(x.arm)}}</span><span class="tag">${{esc(x.outcome)}}</span><span class="tag">Gold ${{x.gold}}</span></p><p>Original control <b>${{x.control}}</b> → NN <b>${{x.nn}}</b>; new probability <b>${{x.new_probability}}</b>. ${{x.retained?'Original NN decision retained.':'Original NN decision changed.'}}</p><details><summary>Saved rationale</summary><pre>${{esc(x.rationale)}}</pre></details><div class="grid">${{x.readings.map(r=>`<div><b>${{esc(r.reader)}}</b><p>${{esc(r.context_support)}} · ${{esc(r.task_rule)}}</p><p>${{esc(r.note)}}</p><small>${{r.flags.map(esc).join(' · ')}}</small><details><summary>Exact evidence</summary><pre>${{esc(JSON.stringify(r.evidence,null,2))}}</pre></details></div>`).join('')}}</div><details><summary>Full task, target and neighbour</summary><pre>${{esc(x.task)}}</pre><pre>${{esc(JSON.stringify(x.context,null,2))}}</pre></details><small>${{esc(x.audit_id)}}</small></article>`).join('');
}}
document.querySelectorAll('select,input').forEach(e=>e.addEventListener('input',render));render();
</script></body></html>"""
    (O/"report.html").write_text(content)
    generated = sorted(p for p in O.iterdir() if p.suffix in [".png", ".pdf", ".html"])
    (O/"REPORT_BUILD.json").write_text(json.dumps(dict(
        built_utc=datetime.now(timezone.utc).isoformat(),
        local_census_complete=True, context_double_read_complete=True,
        remote_collections_claimed_read=True,
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in generated},
    ),indent=2)+"\n")
    print(O/"report.html")


if __name__ == "__main__":
    main()
