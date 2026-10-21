"""Preserve separate auxiliary contrast corpora, with exact context checks."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import hashlib
import csv
from verify_native_contexts import parse

R = Path(__file__).resolve().parent
ROOT = R.parent
PROJECT = R.parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()


def readgz(p):
    return list(map(json.loads, gzip.open(p, "rt")))


def make_corpus(name, rows, audit, prior):
    dest = ROOT/name
    assert not (dest/"CORPUS_MANIFEST.json").exists()
    for child in ["corpus", "packets", "codes", "results"]:
        (dest/child).mkdir(parents=True, exist_ok=True)
    texts, reuse = {}, {}
    for row in rows:
        tid = sha((row["dataset"]+"\n"+row["text"]).encode())
        row["text_id"] = tid
        if row["text"].strip():
            if tid in prior:
                reuse[tid] = prior[tid]
            else:
                texts[tid] = dict(id=tid, text=row["text"], dataset=row["dataset"])
    with gzip.open(dest/"corpus/rows.jsonl.gz", "wt") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
    batches = []
    for dataset in sorted({t["dataset"] for t in texts.values()}):
        ts = sorted((t for t in texts.values() if t["dataset"] == dataset), key=lambda x:x["id"])
        for i in range(0, len(ts), 50):
            bid = f"contrast_{dataset}_{i//50:03d}"
            path = dest/"packets"/(bid+".jsonl")
            path.write_text("".join(json.dumps({k:t[k] for k in ["id", "text"]}, ensure_ascii=False)+"\n"
                                    for t in ts[i:i+50]))
            batches.append(dict(batch_id=bid, dataset=dataset, role="contrast", n=len(ts[i:i+50]),
                                path=str(path.resolve()), sha256=sha(path.read_bytes())))
    with (dest/"BATCHES.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=["batch_id","dataset","role","n","path","sha256"])
        writer.writeheader(); writer.writerows(batches)
    (dest/"REUSED_EXPRESSED_CODES.json").write_text(json.dumps(list(reuse.values()), indent=2)+"\n")
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_records=len(rows),
        nonempty_texts=sum(bool(r["text"].strip()) for r in rows),
        unique_new_texts=len(texts), exact_previously_read_texts=len(reuse),
        records_with_reused_expressed_codes=sum(r["text_id"] in reuse for r in rows),
        batches=len(batches), by_dataset=dict(Counter(x["dataset"] for x in rows)),
        dedup="Exact dataset plus full text, for expressed-word codes only; not contextual validity.",
        original_outcomes_masked=True, semantic_census_complete=False,
        no_inference=True, **audit)
    (dest/"CORPUS_MANIFEST.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(name, json.dumps({k:summary[k] for k in ["selected_records","unique_new_texts",
        "exact_previously_read_texts","records_with_reused_expressed_codes","batches","by_dataset"]}))


def main():
    prior = {}
    for path in [ROOT/"results/validated_codes.jsonl",
                 ROOT/"reasoning_first/results/validated_codes.jsonl"]:
        for code in map(json.loads, path.read_text().splitlines()):
            # Save source-specific exact-text codes; do not resolve any coding disagreement by outcome.
            prior.setdefault(code["id"], dict(dataset=code["dataset"], code=code,
                                              source=str(path.resolve())))
    native = readgz(R/"native_saved_records.jsonl.gz")
    verified = readgz(R/"native_context_verification.jsonl.gz")
    selected_native = []
    for check in verified:
        if not check["eligible_contrast"]:
            continue
        x = native[check["native_index"]]
        selected_native.append(dict(
            dataset=x["dataset"], text=x["raw"].get("text", ""), run=x["run"], judge=x["judge"],
            backbone=x["backbone"], sample_id=x["sample_id"], mode=x["mode"],
            source_raw_file=x["source_raw_file"], source_raw_sha256=x["source_raw_sha256"],
            source_row=x["source_row"], final=x["result"], context_check=check))
    native_audit = dict(
        scope="Earlier native-thinking pilot and timing tests; distinct from requested output-rationale runs.",
        text_contract="Entire raw generated text retained, including delimiters and terminal answers. Original control/NN outcomes and identities masked in packets, but generated terminal values are not masked.",
        selection_limit="Includes repeated timing inputs and both hard/probability contracts. Do not pool as independent cases or as an unbiased deployment sample.",
        source_verification=str((R/"NATIVE_CONTEXT_VERIFICATION.json").resolve()))
    native_frozen = ROOT/"native_contrast/corpus/rows.jsonl.gz"
    if (ROOT/"native_contrast/CORPUS_MANIFEST.json").exists():
        frozen = readgz(native_frozen)
        assert len(frozen) == len(selected_native)
        for old, new in zip(frozen, selected_native):
            assert {k:v for k,v in old.items() if k != "text_id"} == new
        print("native_contrast already frozen; all source records verified identical")
    else:
        make_corpus("native_contrast", selected_native, native_audit, prior)

    broad = readgz(R/"broad_saved_records.jsonl.gz")
    wanted = [x for x in broad if x["split"] == "test" and x["nonempty_rationale"]
              and not x["exact_output_reused_in_primary"] and x["majority_important_contrast"] == "yes"]
    keys = {(x["backbone"], x["sample_id"]) for x in wanted}
    features = {}
    for x in map(json.loads, gzip.open(PROJECT/"reproduction/nn_mechanisms_20261015/features.jsonl.gz","rt")):
        key = x["backbone"], x["sample_id"]
        if key in keys:
            features[key] = x
    prompts = {}
    for x in map(json.loads, gzip.open(R/"nn-reason-after-20261016.attempted_prompts.jsonl.gz","rt")):
        if "row" in x:
            prompts[x["source_file"], x["source_line"]] = x
    tasks = json.loads((ROOT/"TASK_RUBRICS.json").read_text())
    selected_broad, checks = [], []
    for x in wanted:
        source = prompts[x["source_prompt_file"], x["source_prompt_line"]]
        assert source["line_sha256"] == x["source_prompt_line_sha256"]
        rendered = source["row"]["prompt"]
        marker = "Classify this supplied conversation JSON:\n"
        # Parse JSON lengths, so any apparent prompt delimiters inside the texts remain data.
        before, after = rendered.split(marker, 1)
        parsed = parse([dict(role="system", content=before),
                        dict(role="user", content=marker+after)])
        f = features[x["backbone"], x["sample_id"]]
        equality = {k: parsed[k] == f[k] for k in [
            "target_messages", "neighbor_messages", "nearest_label", "nearest_prediction"]}
        equality["task"] = parsed["task"] == tasks[x["dataset"]]["task"]
        checks.append(dict(judge=x["judge"], backbone=x["backbone"], sample_id=x["sample_id"], checks=equality))
        assert all(equality.values()), checks[-1]
        selected_broad.append(dict(dataset=x["dataset"], text=x["source_record"]["reasoning"],
            judge=x["judge"], backbone=x["backbone"], sample_id=x["sample_id"],
            original_source=x, true_label=f["true_label"],
            context={k:f[k] for k in ["target_messages", "neighbor_messages", "nearest_label", "nearest_prediction"]}))
    checkpath = R/"BROAD_SELECTED_CONTEXT_CHECKS.json"
    checkpath.write_text(json.dumps(checks, indent=2)+"\n")
    make_corpus("broad_contrast", selected_broad, dict(
        scope="Additional nonempty test contrast responses from intentionally paused probability-first run, excluding exact source outputs already in primary corpus.",
        text_contract="Saved parsed explanation after first probability; original outcomes and identities masked in packets.",
        selection_limit="Sequential partial production dominated by Qwen27 on Llama backbone plus length/dataset smoke samples. Validation responses have no approved mechanism labels and remain explicitly unclassified in original inventory.",
        context_checks_sha256=sha(checkpath.read_bytes()), source_rows=len(broad),
        all_source_rows_preserved=True), prior)


if __name__ == "__main__":
    main()
