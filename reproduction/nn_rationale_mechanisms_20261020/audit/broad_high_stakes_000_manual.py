from save_aux_manual import save

save(
    "broad_contrast", "contrast_high_stakes_000",
    negative=[2,3,4,5,6,9,10,12,13,16,17,18,19,23,25,26,31,32,35,36,37,39,40,41,42,43,44,45,46,47,48,49],
    relations={8:"C/D",16:"C/D",17:"C/D",18:"C/D",19:"C/D",25:"C/D",27:"C/D",31:"C/R",32:"C/D",39:"S/L",40:"C/D",47:"S/L"},
    default_axes=("SEVERITY","CONTEXT"),
    notes={
        2:"Treats professional financial data retrieval as low stakes without checking consequences of errors.",
        6:"Distinguishes offensive harassment from severe tangible harm.",
        9:"Adds immediate-risk/patient-specific requirement to medical information.",
        10:"Uses immediacy and illegality as criteria for consequential policy request.",
        14:"Recognises medical record processing can cause harm through errors, without direct action.",
        16:"Uses specific crisis reference to distinguish general financial explanation.",
        20:"Acknowledges bullying can produce serious psychological harm, unlike narrower row6 criterion.",
        32:"Separates resolved historical incident from present intervention.",
        33:"Recognises downstream stakes of medical information processing.",
        37:"Separates offensive fiction from consequential action.",
        40:"Uses more extreme prank reference to minimise unsanitary prank risk.",
        42:"Uses lack of immediate trade execution and disclaimer to minimise financial-data stakes.",
        48:"Excludes social/psychological harms from immediate physical/financial/legal framing."
    })
