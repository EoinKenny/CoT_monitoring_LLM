"""Check native-pilot task, target and reference against annotated frozen cases."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import json
import gzip
import csv
import hashlib
import re

R = Path(__file__).resolve().parent
PROJECT = R.parents[2]
BODY = R / "native_prompt_bodies"
NATIVE = R / "native_saved_records.jsonl.gz"
FEATURES = PROJECT / "reproduction/nn_mechanisms_20261015/features.jsonl.gz"
ANNOTATIONS = PROJECT / "reproduction/nn_mechanism_agreement_20261019/results/joined_annotations.csv.gz"


def parse(messages):
    assert len(messages) == 2
    system, user = messages
    assert system["role"] == "system" and user["role"] == "user"
    head = "Classify this supplied conversation JSON:\n"
    assert user["content"].startswith(head)
    decoder = json.JSONDecoder()
    target, end = decoder.raw_decode(user["content"][len(head):])
    tail = user["content"][len(head)+end:]
    sep = "\n\nNearest-neighbour explanation:\nThe most similar training example was:\n"
    assert tail.startswith(sep)
    neighbour, end = decoder.raw_decode(tail[len(sep):])
    tail = tail[len(sep)+end:]
    match = re.match(r"\n\nReference example:\nDataset label: [^\n]+ \(([01])\) \|\n"
                     r"Probe prediction: [^\n]+ \(([01])\)\n\n", tail)
    assert match, repr(tail[:200])
    task = "Task:" + system["content"].split("Task:", 1)[1].split("\n\n", 1)[0]
    return dict(task=task, target_messages=target, neighbor_messages=neighbour,
                nearest_label=int(match[1]), nearest_prediction=int(match[2]))


def main():
    native = list(map(json.loads, gzip.open(NATIVE, "rt")))
    needed = {(x["backbone"], x["sample_id"]) for x in native if x["condition"] == "nearest"}
    features = {}
    for x in map(json.loads, gzip.open(FEATURES, "rt")):
        key = x["backbone"], x["sample_id"]
        if key in needed:
            assert key not in features
            features[key] = {k: x[k] for k in [
                "target_messages", "neighbor_messages", "nearest_label", "nearest_prediction", "dataset", "split"]}
    annotations = {(x["backbone"], x["sample_id"]): x
                   for x in csv.DictReader(gzip.open(ANNOTATIONS, "rt"))}
    tasks = json.loads((R.parent/"TASK_RUBRICS.json").read_text())
    prompts = {}
    manifest = json.loads((BODY/"NATIVE_PROMPT_MANIFEST.json").read_text())
    for f in manifest["files"]:
        path = BODY/f["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == f["sha256"]
        run = Path(f["path"]).parts[0]
        for x in map(json.loads, path.read_text().splitlines()):
            key = run, x["id"]
            assert key not in prompts
            prompts[key] = x
    records = []
    counts = defaultdict(Counter)
    for i, x in enumerate(native):
        record = dict(native_index=i, run=x["run"], judge=x["judge"],
                      backbone=x["backbone"], sample_id=x["sample_id"], dataset=x["dataset"],
                      condition=x["condition"], mode=x["mode"], source_raw_file=x["source_raw_file"],
                      source_row=x["source_row"], eligible_contrast=False)
        if x["condition"] != "nearest":
            record["status"] = "not_nearest_condition"
        else:
            key = x["backbone"], x["sample_id"]
            prompt = prompts.get((x["run"], x["raw"]["id"]))
            feature, ann = features.get(key), annotations.get(key)
            record["annotation_by_id_available"] = ann is not None
            if not prompt:
                record["status"] = "missing_saved_prompt"
            elif not feature:
                record["status"] = "outside_current_frozen_features"
            else:
                try:
                    parsed = parse(prompt["messages"])
                    checks = {k: parsed[k] == feature[k] for k in [
                        "target_messages", "neighbor_messages", "nearest_label", "nearest_prediction"]}
                    checks["task"] = parsed["task"] == tasks[x["dataset"]]["task"]
                    record["checks"] = checks
                    record["parsed_context_sha256"] = hashlib.sha256(
                        json.dumps(parsed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    record["status"] = "exact_context_verified" if all(checks.values()) else "context_differs"
                    record["eligible_contrast"] = bool(all(checks.values()) and ann and
                        ann["majority_important_contrast"] == "yes")
                    if ann:
                        record["contrast_annotation"] = ann["majority_important_contrast"]
                        record["similarity_annotation"] = ann["majority_labelled_similarity"]
                except (AssertionError, KeyError, ValueError, IndexError) as error:
                    record["status"] = "prompt_parse_failed"
                    record["parse_error"] = str(error)
        counts[x["run"]][record["status"]] += 1
        counts[x["run"]]["eligible_contrast"] += record["eligible_contrast"]
        records.append(record)
    ledger = R/"native_context_verification.jsonl.gz"
    with gzip.open(ledger, "wt") as out:
        for x in records:
            out.write(json.dumps(x, ensure_ascii=False)+"\n")
    summary = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),
        native_rows=len(records), by_run={k: dict(v) for k, v in counts.items()},
        eligible_contrast=sum(x["eligible_contrast"] for x in records),
        ledger_sha256=hashlib.sha256(ledger.read_bytes()).hexdigest(),
        semantics="Exact task string, decoded target and neighbour message strings, and displayed neighbour label/prediction must all match.",
        scope="Separate native-thinking interventions. Eligibility does not mean semantic reading completed; repeated timing inputs remain explicit.",
        no_inference=True, source_read_only=True)
    (R/"NATIVE_CONTEXT_VERIFICATION.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
