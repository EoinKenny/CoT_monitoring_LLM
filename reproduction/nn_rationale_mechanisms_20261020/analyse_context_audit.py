"""Join outcomes only after both independent full-context readings are frozen."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
A = R / "context_audit"
O = R / "results" / "context_audit"
TOX = ["aegis", "beavertails", "toxicchat", "toxigen"]
HALL = ["fava", "halueval", "ragtruth"]
FLAGS = [
    "imports_neighbour_facts_or_intent", "topic_equated_to_class",
    "relative_severity_crossing_without_rule", "misses_stance_negation_or_role",
    "neighbour_substitutes_for_source_check", "changes_task_criterion",
    "reference_label_prediction_conflict", "unsupported_target_inference",
    "contradictory_prose", "uncertain",
]


def main():
    status = json.load(open(A / "STATUS.json"))
    assert status["all_double_read"], "Do not unmask before all 200 are independently double-read"
    codes = pd.read_json(A / "validated_records.jsonl", lines=True)
    assert len(codes) == 400 and codes.groupby("audit_id").size().eq(2).all()
    frozen = {}
    for p in sorted((A / "codes").glob("*/*")):
        frozen[str(p.relative_to(A))] = hashlib.sha256(p.read_bytes()).hexdigest()
    O.mkdir(exist_ok=True)
    receipt = O / "CODING_FREEZE.json"
    if receipt.exists():
        assert json.load(open(receipt))["sha256"] == frozen, "Independent coding changed after unmasking"
    else:
        receipt.write_text(json.dumps(dict(
            utc=datetime.now(timezone.utc).isoformat(), sha256=frozen,
            n_codes=400, n_contexts=200,
            outcome_masked=True,
            reader_type="Separate assistant readers; not independent human raters.",
        ), indent=2) + "\n")
    selections = pd.read_json(A / "selection_private.json")
    meta = pd.read_csv(R / "corpus/metadata.csv.gz", usecols=["trace_id", "w", "cluster", "family"])
    selections = selections.merge(meta, on="trace_id", validate="one_to_one")
    assert selections.w.notna().all()
    selections["audit_weight"] = selections.w / selections.audit_selection_probability
    joined = codes.merge(selections, on=["audit_id", "dataset"], validate="many_to_one")
    joined["support_error"] = joined.context_support.isin(["unsupported_inference", "misreads_text"])
    joined["rule_error"] = joined.task_rule.isin(["criterion_shift", "wrong_object"])
    joined["any_context_or_rule_error"] = joined.support_error | joined.rule_error
    joined["supported_consistent"] = (joined.context_support == "supported") & (joined.task_rule == "consistent")
    joined["uncertain_context_or_rule"] = (
        joined.context_support.isin(["insufficient_context", "no_checkable_claim"]) |
        (joined.task_rule == "unclear")
    )
    joined["clear_prose"] = joined.conclusion.isin(["P", "N"])
    joined["score_prose_conflict"] = joined.clear_prose & (
        (joined.conclusion == "P") != (joined.new_probability >= 50)
    )
    features = [
        "support_error", "rule_error", "any_context_or_rule_error",
        "supported_consistent", "uncertain_context_or_rule", "score_prose_conflict",
    ]
    for f in FLAGS:
        joined[f] = joined["flags"].map(lambda z: f in z)
        features.append(f)
    joined.to_json(O / "individual_reader_records.jsonl", orient="records", lines=True, force_ascii=False)
    # Both/either preserve uncertainty without silently substituting an adjudication.
    bounds = selections.copy()
    for f in features:
        s = joined.groupby("audit_id")[f].agg(["min", "max", "mean"])
        for mode, col in [("both", "min"), ("either", "max"), ("reader_average", "mean")]:
            bounds[f + "__" + mode] = bounds.audit_id.map(s[col]).astype(float)
    bounds.to_csv(O / "paired_reader_cases.csv", index=False)
    summaries = []
    for arm in ["contrast", "similarity_only", "both_arms"]:
        aa = bounds if arm == "both_arms" else bounds[bounds.arm == arm]
        for scope in ["all10", "toxicity4", "hallucination3", "other_safety3"] + sorted(aa.dataset.unique()):
            ss = aa if scope == "all10" else aa[aa.family == scope] if scope.endswith(("4", "3")) else aa[aa.dataset == scope]
            for retained in ["all", "retained", "changed"]:
                qq = ss if retained == "all" else ss[ss.retains == (retained == "retained")]
                for outcome in ["helped", "harmed"]:
                    z = qq[qq.original_outcome == outcome]
                    if not len(z):
                        continue
                    for f in features:
                        for mode in ["both", "either", "reader_average"]:
                            col = f + "__" + mode
                            summaries.append(dict(
                                arm=arm, scope=scope, retention=retained,
                                original_outcome=outcome, feature=f, coding=mode,
                                n=len(z), marked=float(z[col].sum()),
                                balanced_sample_rate=float(z[col].mean()),
                                weighted_sensitivity=float(np.average(z[col], weights=z.audit_weight)),
                                effective_n=float(z.audit_weight.sum() ** 2 / (z.audit_weight ** 2).sum()),
                            ))
    pd.DataFrame(summaries).to_csv(O / "summary.csv", index=False)
    # The original two-reader agreement is reported, not replaced by outcome-aware consensus.
    agreement = json.load(open(A / "agreement.json"))
    pd.DataFrame(agreement).to_csv(O / "agreement.csv", index=False)
    count_agreement = []
    for f in features:
        s = joined.groupby("audit_id")[f].agg(["min", "max"])
        count_agreement.append(dict(feature=f, n=len(s), agree=int((s["min"] == s["max"]).sum()),
                                    both_yes=int(s["min"].sum()), either_yes=int(s["max"].sum())))
    pd.DataFrame(count_agreement).to_csv(O / "binary_agreement.csv", index=False)
    (O / "METHOD.json").write_text(json.dumps(dict(
        selected=200, distinct_contexts=199, reader_records=400,
        sample="Per dataset: 6 helpful and 6 harmful contrast; 4 helpful and 4 harmful similarity-only.",
        primary="Balanced audit sample counts, with agreement bounds from both/either reader coding.",
        sensitivity="Population design weight divided by two-stage audit selection probability; very noisy at 6 or 4 cases per stratum.",
        interpretation="Exploratory associations between expressed-rationale quality and original fixed50 outcome. No causal or private reasoning inference.",
        adjudication="None: independent original records preserved. Both/either are coding sensitivity bounds, not confidence intervals.",
        uncertainty="No significance tests for small audit. Independent assistant readers are not human validation.",
    ), indent=2) + "\n")
    print("Unmasked and analysed 200 double-read full-context cases.")


if __name__ == "__main__":
    main()
