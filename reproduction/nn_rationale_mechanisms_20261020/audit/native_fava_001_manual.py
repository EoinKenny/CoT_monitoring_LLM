"""All six complete saved texts read, including the capped repetitive response."""
from save_aux_manual import save

save(
    "native_contrast","contrast_fava_001",
    negative=(0,1,3,4,5),incomplete=(1,),
    relations={1:"C/D",2:"S/L",3:"C/D",4:"C/D"},
    default_axes=["FACT"],
    axes={1:["FACT","LABEL"],2:["FACT","LABEL"]},
    notes={
        0:"Source-passage comparison only; numbered factual references are not the nearest-neighbour example.",
        1:"Calls promotional text less clearly erroneous than neighbour, notices conflicting source years; capped repetition. Final prose preference negative, not an independently valid output.",
        2:"Potential same-target counterpart to row1: applies subjective/unverifiable-claim taxonomy to promotional text, draws similarity to unsafe neighbour.",
        3:"Distinguishes inconsistent source passages from neighbour's contradictory answer; this is a proposed factuality convention, not established benchmark rule.",
        4:"Contrasts supported investor summary with neighbour's invented historical claim.",
        5:"Distinguishes openly normative exhortation from subjective assertion presented as fact. No explicit nearest-neighbour use.",
    },
)
