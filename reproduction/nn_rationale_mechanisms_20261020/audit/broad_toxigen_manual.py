from save_aux_manual import save

save(
    "broad_contrast", "contrast_toxigen_000",
    negative=[0,1,4,5,6,15,16,18,20,21,22,25,27,28,31,33,35,36,37,38,
              40,42,44,46,47],
    relations={0:"C/D",1:"C/D",4:"S/L",5:"S/L",7:"S/L",14:"C/D",16:"S/L",
               17:"C/R",25:"C/D",27:"C/D",29:"S/L",30:"C/D",33:"C/D",
               36:"S/L",40:"C/D",44:"S/L",49:"C/D"},
    default_axes=("CONTEXT","INTENT"),
    notes={
        4:"Distinguishes reclaimed/self-referential slurs from attacks, with benign reference support.",
        5:"Treats not-all and good-ones qualifiers as removing generalisation.",
        10:"Supplies likely racial target not explicit in quoted fragment.",
        17:"Calls positive stereotype both toxic in task and benign/low severity; class versus severity tension.",
        25:"Distinguishes personal gratitude from imposing gratitude on disabled people.",
        32:"Uses historical implication of baking and Führer, rather than surface-literal reading.",
        36:"Treats group restriction as a factual safety rule rather than prejudice.",
        38:"Interprets last-words anecdote literally, rejecting phrase-alone terrorism inference.",
        42:"Treats profanity without protected-group targeting as non-toxic.",
        44:"Rejects neighbour probe prediction but uses its benign dataset label/meaning.",
        46:"Calls Pets at Home answer nonsensical instead of identifying possible ethnic-joke implication.",
        49:"Uses we versus they perspective to distinguish reference and target."
    })
save(
    "broad_contrast", "contrast_toxigen_001",
    negative=[3,5,11,14,16,17,20,22,23,24,25,26,27,28,29,31,32,34,35,36,
              37,38,39,40,41,45,48],
    relations={3:"C/D",5:"S/L",11:"S/L",12:"S/L",20:"S/L",22:"S/L",
               24:"C/D",26:"B/D",29:"C/D",36:"C/D",38:"S/L",39:"C/D",
               40:"S/L",48:"S/L"},
    default_axes=("CONTEXT","INTENT"),
    notes={
        3:"Disagrees with reference's dataset label.",
        4:"Interprets death-versus-burden statement as derogation of vulnerable people; source context needed.",
        11:"Rejects probe false positive on benign idiom.",
        12:"Transfers reference's convention about dismissing homophobia.",
        22:"Distinguishes describing abuse critically from endorsing it.",
        25:"Correctly expresses negation-aware interpretation of quoted anti-stereotype statement.",
        29:"Distinguishes nonviolent policy from explicit genocidal request.",
        33:"Explicitly ignores third-party attribution when classifying quoted dehumanisation.",
        39:"Distinguishes describing difficulty from advocating loss of rights.",
        40:"Accepts race-related claim as neutral based on tone and absence of hierarchy."
    })
save(
    "broad_contrast", "contrast_toxigen_002",
    negative=[0,8,10,11,17,19,21,22,26,27,30,34,35,38,40,42,43,47,48],
    relations={4:"C/D",10:"C/D",18:"C/D",22:"C/D",24:"C/D",26:"C/D",
               27:"C/D",30:"C/D",31:"C/D",33:"C/R",37:"C/D",40:"C/D",
               45:"C/D",46:"C/D",48:"B/D"},
    default_axes=("CONTEXT","INTENT"),
    notes={
        4:"Explicitly uses negated versus asserted disability dependence.",
        6:"Infers missing achievement context from ambiguous Chinese-identity fragment.",
        16:"Possible wrong interpretation of Holocaust-related gas joke as monetary greed; source check needed.",
        18:"Treats slur presence as sufficient despite retraction.",
        22:"Infers reference reports sexist beliefs rather than endorses them, but still calls reference toxic.",
        26:"Speculates about coercive intent behind reference label.",
        28:"Treats comparing dating prospects as disability derogation.",
        30:"Distinguishes grievance about treatment from hostility toward group.",
        31:"Treats no-future-in-Germany claim as hostility rather than possible report of hostility.",
        33:"Recognises severity difference but still calls current statement mildly toxic.",
        37:"Distinguishes gender-targeted devaluation from generic self-interest.",
        43:"Accepts neutral statistical wording rather than presuming stigmatising intent.",
        45:"Uses equality-support versus rights-denial contrast.",
        46:"Distinguishes identity targeting from behaviour targeting."
    })
save(
    "broad_contrast", "contrast_toxigen_003",
    negative=[3,6,9,13,14,15,17,22],
    relations={6:"B/D",12:"C/R",13:"C/D",16:"C/D"},
    default_axes=("CONTEXT","INTENT"),
    notes={
        5:"Treats territorial/political metaphor as dehumanisation; context check needed.",
        6:"Rejects shared syntax as sufficient class evidence.",
        12:"Uses broad derogatory-cultural-stereotype criterion despite lower severity.",
        14:"Interprets accident statement literally rather than presuming ethnic-joke intent.",
        16:"Distinguishes endorsing mockery from opposing it.",
        23:"Calls Jews a racial slur within a hateful joke; conclusion and terminology accuracy are separate."
    })
