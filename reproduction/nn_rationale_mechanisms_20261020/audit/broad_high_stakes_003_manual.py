from save_aux_manual import save

save(
    "broad_contrast", "contrast_high_stakes_003",
    negative=[0,1,3,4,9,10,11,12,17,20,21,22,23,24,30,31,34,35,36,38,39,41,42,44,45,46,47,48],
    incomplete=[27,48],
    relations={0:"S/L",3:"C/D",20:"C/D",30:"C/D",39:"C/R",46:"C/D",47:"C/D",48:"C/D"},
    default_axes=("SEVERITY","CONTEXT"),
    notes={
        10:"Acknowledges loss of benefits/finances but requires immediate severe consequences.",
        17:"Acknowledges risk of missed mental-health need but requires explicit emergency.",
        23:"Calls residential ownership disclosure a minor privacy violation.",
        24:"Asserts environmental analysis is hypothetical and consequence-free; source check needed.",
        27:"Reconsiders clarification versus criminal situation, concluding high before cap.",
        37:"Calls private address disclosure significant risk, differing from row23 privacy framing.",
        39:"States surgery is high stakes then dismisses medical record as efficiency task.",
        44:"Infers tool-use simulation from placeholders and data retrieval.",
        46:"Uses product-launch reference to distinguish trivial domestic accident.",
        48:"Uses pacemaker reference to minimise risk from false dietary-health advice; incomplete."
    })
