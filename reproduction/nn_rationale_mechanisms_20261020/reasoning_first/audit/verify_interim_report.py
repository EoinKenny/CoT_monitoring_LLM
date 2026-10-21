"""Static artifact QA; deliberately makes no browser-rendering claim."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import hashlib
import json

R = Path(__file__).resolve().parents[1]
O = R/"results"
report = O/"report.html"
html = report.read_text()


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.cards = [], set(), 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.links.extend(attrs[k] for k in ["href", "src"] if k in attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        self.cards += tag == "details"


doc = Document()
doc.feed(html)
checked = 0
for link in doc.links:
    url = urlsplit(link)
    if url.scheme or url.netloc:
        continue
    path = (report.parent/unquote(url.path)).resolve() if url.path else report
    assert path.is_file(), (link, path)
    if url.fragment:
        target = Document()
        target.feed(path.read_text())
        assert unquote(url.fragment) in target.ids, link
    checked += 1

examples = []
for directory in [
    "context_followup", "context_wildjailbreak_followup", "context_convention_followup",
    "context_beavertails_followup", "context_high_stakes_followup",
    "context_high_stakes_followup2", "context_high_stakes_followup3",
    "context_high_stakes_followup4", "context_high_stakes_followup5",
    "context_ragtruth_followup",
]:
    examples.extend(json.loads((R/directory/"unmasked_examples.json").read_text()))
for x in examples:
    assert escape(x["audit_id"]) in html
    for field in ["target_messages", "neighbor_messages"]:
        content = "\n\n".join(f"{m['role']}: {m['content']}" for m in x["context"][field])
        assert f"<pre>{escape(content)}</pre>" in html, (x["audit_id"], field)
    assert f"<pre>{escape(x['rationale'])}</pre>" in html, x["audit_id"]
build = json.loads((O/"INTERIM_REPORT_BUILD.json").read_text())
status = json.loads((O/"READING_STATUS.json").read_text())
assert build["reading_status"] == status
assert doc.cards == len(examples) == build["qualitative_trace_context_pairs"]
assert len({x["read_text_id"] for x in examples}) == build["qualitative_texts"]
assert not build["complete_goal"]
assert status["contrast_census_complete"]
assert "contrast semantic census complete" in html
receipt = dict(
    checked_utc=datetime.now(timezone.utc).isoformat(),
    report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
    local_links_checked=checked,
    full_source_and_rationale_escaping_checked=len(examples),
    cards=doc.cards, texts=build["qualitative_texts"],
    reading_records=status["contrast_records_read"], browser_rendered=False,
)
(O/"INTERIM_STATIC_QA.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt, indent=2))
