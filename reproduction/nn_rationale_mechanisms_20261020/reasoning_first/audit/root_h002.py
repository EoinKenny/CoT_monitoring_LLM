from write_root_census import save

save(
    "contrast_halueval_002",
    negative=[
        0,2,4,5,6,7,8,9,12,13,14,15,16,20,21,24,27,30,32,33,35,41,
        45,49,50,51,52,53,54,57,58,60,63,64,66,67,75,78,79,81,82,83,
        85,86,87,90,91,92,95,96,97,
    ],
    unclear=[1,25],
    mixed=[98],
    incomplete=[20,23,95],
    relations={
        2:"C/D",5:"C/D",6:"C/R",10:"S/L",13:"C/D",23:"S/L",
        52:"C/D",60:"C/D",64:"C/D",71:"S/R",79:"C/R",81:"C/R",95:"C/R",97:"C/D",
    },
    axes={i:["FACT"] for i in range(100)},
    notes={
        1:"Incomplete/possibly misleading answer described without settled hallucination class.",
        2:"Explicitly distinguishes added witches content in reference from current supported answer.",
        13:"Distinguishes incomplete true response from reference's fabricated identifier.",
        15:"Calls the question multiple-choice while treating A as supported; original source not checked here.",
        23:"Explicit reference/probe support for positive judgment; potentially misattributed entity claims need source check.",
        25:"Says no factual claim can be evaluated, rather than giving a resolved class.",
        27:"Initially misstates single ordering but corrects it and distinguishes band question from ordering.",
        54:"Separates erroneous question premise from supported answer entity.",
        56:"Biographical death attribution unverified; do not treat it as a factual finding.",
        66:"Says born1947/died1950 but then says death precedes birth; conclusion negative despite faulty arithmetic.",
        71:"Transfers evidence-support criterion from a reference with a different alleged error type.",
        72:"Quotes15versus7seasons then denies that comparative tenure follows; source check needed.",
        77:"Names distinct subjects yet calls their different statements contradictory; no source validity inferred.",
        81:"Opening describes reference error; final current-target judgment is correct/non-hallucinated.",
        85:"Acknowledges unsupported answer but adopts false-only criterion and resolves negative.",
        91:"Reference dismissed as irrelevant without specifying relation; coded target-only.",
        95:"Before truncation explicitly resolves supported generalization negative, then reopens dataset conventions.",
        98:"Calls the same attribution incorrect and answer correct without distinguishing reference from target; mixed prose.",
    },
    reading_view="lossless-adjacent-repeat-v1",
)

save(
    "contrast_halueval_003",
    negative=[2,6,8],
    incomplete=[3],
    relations={0:"S/L",7:"S/R"},
    axes={i:["FACT"] for i in range(10)},
    notes={
        0:"Uses wrong-entity reference as precedent for current non-answer.",
        3:"Calls supported broader denomination unsupported due lack of specificity.",
        5:"Acknowledges possible question defect but ultimately resolves category mismatch positive.",
        7:"Uses wrong-number precedent to support unsupported-subjective-detail classification.",
        8:"Entity/character attribution is a model claim, not source-verified census evidence.",
    },
    reading_view="lossless-adjacent-repeat-v1",
)
