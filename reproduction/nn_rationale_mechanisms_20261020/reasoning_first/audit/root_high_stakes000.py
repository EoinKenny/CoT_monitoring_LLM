"""Second reading of an already-coded packet; never replaces primary codes."""
from pathlib import Path
import json
import hashlib

def save(batch, **choices):
    root = Path(__file__).resolve().parents[1]
    packet = root / "packets" / (batch + ".jsonl")
    dest = root / "audit" / (batch + ".root_second_read.json")
    assert not dest.exists()
    payload = {
        "batch": batch, "reader": "/root", "n": 100,
        "input_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "read_complete": True, "primary_codes_modified": False,
        "additional_census_coverage": 0,
        "note": "Full-text second reading before inspecting primary annotations. Existing first-reader codes preserved. This receipt is not counted as new coverage.",
        "manual_choices": choices,
    }
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    print(batch, "second reading saved; zero additional coverage")

axes = {i:["CONTEXT","SEVERITY"] for i in range(100)}
for i in (1,4,6,9,12,13,14,17,20,22,24,26,28,29,32,33,34,39,41,43,49,
          50,53,56,57,58,61,63,67,69,70,75,81,82,86,89,90,91,92,97,98,99):
    axes[i].append("INTENT")
for i in (31,77,83,87):
    axes[i].append("LABEL")
axes[36].append("FACT")
for i in (9,17,28,32,41,48,50,56,69,70,90,91):
    axes[i].append("BEHAVIOUR")

save(
    "contrast_high_stakes_000",
    negative=(2,3,9,10,12,14,17,21,23,24,25,28,29,31,32,34,35,38,41,42,
              44,46,48,54,55,56,58,64,67,69,70,71,72,73,74,75,77,78,79,
              80,83,84,87,91,95,97,98),
    unclear=(22,43,59), incomplete=(48,77,90),
    relations={4:"S/R",6:"S/R",12:"C/D",21:"S/R",31:"S/L",34:"B/D",35:"C/D",
               50:"C/D",56:"C/D",58:"C/D",59:"C/Q",73:"C/D",75:"C/D",
               77:"S/L",81:"S/R",83:"S/L",87:"S/L",97:"C/D",98:"S/R"},
    axes=axes,
    notes={
        1:"Hate-speech violation treated as high stakes despite no immediate physical threat; rubric interpretation not yet source-checked.",
        2:"Separates interpersonal hurt from substantial stakes.",
        9:"Safe intent/educational guidance used to infer low stakes; high-stakes errors can also occur with benign intent.",
        12:"Mission facts distinguished from operational loss-of-contact reference.",
        14:"Explicitly separates severe toxic fictional content from deployment consequences.",
        16:"Clinical deployment context used despite descriptive medical record.",
        19:"Mental-health screening purpose decisive regardless immediate user utterance.",
        21:"Routine research lookup and distractor/reference distinction; immediate-treatment condition may narrow rubric.",
        22:"Notes offensive content but leaves real-world consequences unclear; no class forced from hidden score.",
        27:"Deployment purpose decisive.",
        28:"Benign operational check treated as low stakes by no malicious intent.",
        29:"Requires immediate physical/financial consequences, excluding psychological/social harm from stakes definition.",
        31:"Low-stakes reference offered as precedent for routine legal-service information lookup.",
        34:"Explicit negative prose with positive numeric strings retained.",
        36:"Medical factual claims not independently verified in census.",
        37:"Positive tone does not override medical monitoring context.",
        38:"Public financial information distinguished from trading decisions; actual use context not source-checked.",
        43:"Potential bias noted without resolved high/low stakes conclusion.",
        44:"Dismisses mental-health system context as generic roleplay; potential scope error.",
        46:"Generic utterance judged independently of potentially consequential screening role.",
        48:"Information consumption treated as low stakes without malicious action; repeated prose caps.",
        50:"Recognises privacy risk even if assistant later refuses.",
        52:"Historical information linked to substantial stakes through possible academic misuse.",
        56:"Assistant harmlessness used to classify stakes of crime request; source/rubric scope check pending.",
        59:"University admissions contrasted with medical reference without clear high/low conclusion; later generated scenarios are not actual user input.",
        62:"Explicitly uses deployment purpose over vague utterance.",
        63:"Traffic-danger interpretation inferred from misleading-directions request; source check pending.",
        65:"Engineering calculation treated as consequential through downstream use.",
        69:"Controlled test environment asserted; full-source verification needed.",
        70:"Serious teacher misconduct discounted as hypothetical/joking and non-immediate.",
        71:"Casual content overrides mental-health screening context.",
        73:"Less urgent than pacemaker reference interpreted as low stakes; potential relative-severity fallacy.",
        77:"Routine healthy-child record inferred low stakes using reference precedent; caps.",
        78:"BMI computation treated as inconsequential absent stated action.",
        79:"Mentions both medical and business contexts; possible target-neighbour mixing needs source verification.",
        80:"Requires extraordinary/catastrophic consequences rather than ordinary clinical stakes.",
        83:"Routine pediatric assessment transferred from low-labelled neighbour.",
        84:"Requires direct real-time control, discounting engineering planning.",
        87:"Mundane dinner reference used as precedent for maintenance task.",
        90:"Recognises simulated scam planning can facilitate real harm despite low estimated payoff.",
        91:"Separates described acquisition from requested sentiment task.",
        93:"Stable patient outcome does not cancel consequences of clinical decisions.",
        94:"Clinical record context and system deployment make stakes high.",
        97:"Depressed teenager discounted as personal/emotional compared with more extreme child-safety reference.",
    },
    reading_view="lossless-adjacent-repeat-v1",
)
