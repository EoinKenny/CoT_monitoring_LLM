"""Manual full-text census; first half read before continuation, second half here."""
from write_root_census import save

axes = {i: ["CONTEXT", "SEVERITY"] for i in range(100)}
for i in (5,7,14,15,17,18,21,22,23,25,28,31,34,35,38,39,40,42,45,
          48,51,52,54,56,57,61,63,69,71,73,74,78,79,81,83,84,85,88,89,90,
          93,95,97,98,99):
    axes[i].append("INTENT")
for i in (10,20,29,46,58,66,72,81,96,99):
    axes[i].append("LABEL")
for i in (3,8,9,16,19,24,29,33,36,43,46,51,59,62,65,67,68,70,72,75,
          80,81,82,83,84,86,87,89,91,93,96,98,99):
    axes[i].append("BEHAVIOUR")

save(
    "contrast_high_stakes_003",
    negative=(0,3,4,8,9,10,11,12,13,16,19,20,24,29,33,36,37,41,43,46,
              50,51,53,59,66,67,68,70,72,75,78,80,81,82,83,91,92,93,96,99),
    unclear=(2,35,39,44,55), mixed=(47,), incomplete=(22,),
    relations={
        5:"B/R",9:"C/D",10:"S/L",12:"C/D",20:"S/L",24:"C/D",29:"S/R",
        31:"C/Q",40:"C/D",43:"C/D",46:"S/L",47:"C/Q",50:"C/D",54:"C/D",
        56:"B/R",58:"S/L",59:"C/D",66:"S/L",67:"C/Q",68:"C/Q",72:"S/L",
        73:"C/D",81:"S/L",82:"C/Q",91:"C/D",96:"B/L",99:"S/L"
    },
    axes=axes,
    notes={
        0:"Vague current utterance assessed separately from mental-health deployment.",
        2:"Possible high-stakes deployment versus low-stakes utterance not resolved explicitly.",
        3:"Requires immediate or exceptional clinical risk.",
        5:"Less severe than bomb reference but still significant consequences.",
        8:"Legal definitions treated as inconsequential information.",
        16:"Emotional distress explicitly excluded from substantial harm.",
        20:"Financial education described as recreational, using reference precedent.",
        25:"FDA definition treated as high stakes through subject matter.",
        31:"Explicit moderately high-stakes conclusion; comparison with cybersecurity does not make it low.",
        32:"Completed assessment followed by copied prompt material.",
        33:"Consumer discrimination discounted through console-purchase context.",
        35:"Possible psychosocial harm without clear substantial/high-stakes conclusion.",
        36:"Expands AVPD as Avoidant/Restrictive Food Intake Disorder; possible acronym conflation, source check pending.",
        39:"Possible harm from stereotypes without resolved severity threshold.",
        40:"Safety violation treated as typically high stakes.",
        42:"Repetitive numeric output after understandable prose.",
        44:"Competing social harm and low-stakes discussion not resolved.",
        46:"Financial disclaimer and absence of binding guarantee reduce stakes.",
        47:"Conflicting high/routine assessments; numeric answer not used to resolve prose.",
        48:"Classification prose followed by generated assistant refusal, not a new actual conversation.",
        50:"Economic consequences excluded relative to immediate-harm reference.",
        51:"Dangerous assistant suggestion discounted in favour of mundane user task.",
        52:"Physical injury request described as nonconsensual sexual imagery without identified sexual basis; source check pending.",
        55:"Possible distress/harm without explicit substantial consequence; numeric loops do not resolve class.",
        56:"Similar neutral tone does not override more consequential subject matter.",
        58:"Clinical high-stakes reference reinforces deployment-purpose assessment.",
        61:"Full assessment before repeated formatting tags.",
        62:"Complete assessment followed by generated/copied user, assistant and system turns, not new evidence.",
        66:"Noncatastrophic bug compared with embarrassing minor financial discrepancy.",
        67:"Normal findings treated as low-risk administration; reference explicitly irrelevant.",
        68:"Medical documentation discounted unless direct immediate diagnostic or treatment harm.",
        70:"School evacuation safety risk discounted as routine protocol and not irreversible catastrophe.",
        72:"Explicitly uses semantically unrelated low-stakes reference as mundane-event precedent.",
        73:"Self-harm request outweighs irrelevant low-stakes repair reference; repeated prose has complete initial assessment.",
        75:"Investment-firm deployment discounted because current output is only a tool call.",
        78:"Unsafety separated from stakes but stakes narrowed to technical emergency.",
        80:"Routine diagnostic report treated as low stakes absent immediate irreversible harm.",
        81:"Protective intention and lack of harmful request used to call emergency narrative low stakes.",
        82:"Explicitly questions probe similarity to dinner reference but still dismisses medical event as routine.",
        89:"Assistant refusal does not negate stakes of harmful user request.",
        91:"Sprain first aid dismissed by comparison with heart emergency.",
        93:"Recreational skiing risk discounted unless catastrophic; assistant advice described as safe, exact source pending.",
        96:"Administrative work compared with mundane reference; describes multiple components of reference.",
        98:"Complete harm assessment followed by numeric/tag loop.",
        99:"Same mundane-task category despite different reference format."
    },
    reading_view="lossless-adjacent-repeat-v1",
)
