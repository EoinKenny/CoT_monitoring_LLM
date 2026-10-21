"""Final coverage, numeric, source-link and static document verification."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import pandas as pd

R = Path(__file__).resolve().parents[1]
O = R / "results"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Doc(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links.extend(attrs[k] for k in ["href", "src"] if k in attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])


def main():
    p = json.loads((O / "READING_STATUS.json").read_text())
    f = json.loads((R / "reasoning_first/results/READING_STATUS.json").read_text())
    aux = {name: json.loads((R / name / "results/READING_STATUS.json").read_text())
           for name in ["broad_contrast", "native_contrast"]}
    assert p["all_semantic_work_complete"] and f["contrast_census_complete"]
    assert p["remaining_contrast"] == p["remaining_similarity_harm"] == f["remaining_contrast"] == 0
    assert all(x["all_available_read"] for x in aux.values())
    assert aux["broad_contrast"]["records_read"] == 1192
    assert aux["native_contrast"]["records_read"] == 448
    assert f["contrast_batches_read"] == f["contrast_batches_total"] == 122
    build = json.loads((O / "SYNTHESIS_BUILD.json").read_text())
    for path, expected in build["source_sha256"].items():
        assert digest(R / path) == expected, path
    assert build["output_sha256"] == digest(O / "synthesis.html")
    pairs = pd.read_csv(R / "reasoning_first/results/paired_answer_order_comparison.csv")
    a = pairs[(pairs.subset == "contrast_yes") & (pairs.family == "all10") &
              (pairs.judge == "all") & (pairs.original_outcome == "all")].iloc[0]
    assert a.both_valid_rows == 10410
    assert a.first_correct_after_wrong == 2794 and a.first_wrong_after_correct == 1542
    assert round(100*a.weighted_after_accuracy, 2) == 53.97
    assert round(100*a.weighted_first_accuracy, 2) == 67.11
    examples = json.loads((O / "audited_examples.json").read_text())
    assert len(examples) == 200
    dynamic_ids = {x["audit_id"] for x in examples}
    link_counts = {}
    for file in [O / "synthesis.html", O / "report.html", R / "reasoning_first/results/report.html"]:
        source = file.read_text()
        doc = Doc()
        doc.feed(source)
        count = 0
        for link in doc.links:
            u = urlsplit(link)
            if u.scheme or u.netloc:
                continue
            target = (file.parent / unquote(u.path)).resolve() if u.path else file
            assert target.is_file(), (file, link)
            if u.fragment:
                target_doc = Doc()
                target_doc.feed(target.read_text())
                assert unquote(u.fragment) in target_doc.ids or (
                    target == O / "report.html" and unquote(u.fragment) in dynamic_ids), link
            count += 1
        assert "</html>" in source and "<body>" in source
        assert not re.search(r"semantic census in progress|reading audit is ongoing", source)
        link_counts[str(file.relative_to(R))] = count
    scripts = re.findall(r"<script>(.*?)</script>", (O / "report.html").read_text(), flags=re.S)
    node = shutil.which("node")
    syntax_checked = False
    if node:
        for script in scripts:
            with tempfile.NamedTemporaryFile(suffix=".js", mode="w") as temp:
                temp.write(script)
                temp.flush()
                subprocess.run([node, "--check", temp.name], check=True, capture_output=True)
        syntax_checked = True
    figures = {}
    from PIL import Image
    for filename in ["original_flip_outcomes.png", "context_audit_comparison.png"]:
        with Image.open(O / filename) as image:
            assert image.width > 500 and image.height > 300
            image.verify()
        figures[filename] = digest(O / filename)
    receipt = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(),
        required_available_reading_complete=True,
        primary_contrast_available_read=p["contrast_records_read"],
        primary_similarity_harm_available_read=p["similarity_harm_records_read"],
        reasoning_first_contrast_available_read=f["contrast_records_read"],
        auxiliary_status=aux, primary_context_cases=200,
        local_links_checked=link_counts, primary_script_syntax_checked=syntax_checked,
        embedded_png_integrity=figures, browser_rendered=False,
        numeric_paired_checks=True, synthesis_inputs_checksums_verified=True,
        output_sha256={str(x.relative_to(R)): digest(x) for x in
                       [O / "synthesis.html", O / "report.html", R / "reasoning_first/results/report.html"]},
        no_new_inference=True,
        inference_scope="Expressed explanations and observational associations; no private-reasoning or causal claim.",
        limitations="Primary similarity census is failure-enriched; auxiliary unchanged and native pilots separate; source checks partly purposive; missing/unscored answers retained.",
    )
    (O / "COMPLETED_AUDIT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    build["final_verification_pending"] = False
    build["verification_receipt"] = "COMPLETED_AUDIT.json"
    (O / "SYNTHESIS_BUILD.json").write_text(json.dumps(build, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
