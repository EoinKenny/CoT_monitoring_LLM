from save_aux_manual import save

save(
    "broad_contrast", "contrast_ragtruth_000",
    negative=[9,11,15,19,29,31,37,43,44],
    incomplete=[13,49],
    relations={44:"C/D"},
    notes={
        0:"Combines unsupported additions with failure to incorporate all passages; omission alone is not a hallucination.",
        2:"Identifies wrong-topic source merging between paint rollers and skin treatment.",
        6:"Treats friendly/cozy descriptions as unsupported against source reviews.",
        7:"Uses external circuit-length claim in addition to absent source superlative.",
        10:"Infers only three reviews exist from three supplied reviews; source-context check needed.",
        27:"Requires explicit source categorisation for examples even where physical inference is straightforward.",
        30:"Explicitly distinguishes null attributes from absence of a service.",
        31:"Accepts source-faithful content despite scientifically unusual generalisation.",
        34:"Distinguishes speculative future event from confirmed occurrence.",
        44:"Explicitly distinguishes current correct source binding from reference's reversed speaker attribution.",
        46:"Adds specific senator identity requiring source verification."
    })
save(
    "broad_contrast", "contrast_ragtruth_001",
    negative=[22,30,31,41,43,45],
    incomplete=[46],
    relations={34:"C/R"},
    notes={
        0:"Identifies context mismatch when combining instructions from different material types.",
        4:"Equates Armenian ethnicity with not originating from Turkey; source-context check needed.",
        13:"Penalises verbatim source statement because source passages conflict.",
        16:"Calls category-supported bubble tea unconfirmed while identifying separate clear errors.",
        19:"Explicitly distinguishes omission from factual contradiction.",
        24:"Distinguishes word-count instruction failure from source-attribution hallucination.",
        33:"Treats unit conversion as unsupported alongside a genuinely incorrect average calculation.",
        34:"Suggests reference-phase contamination; requires checking both full examples.",
        35:"Treats falsely denying that evidence answers question as hallucination of insufficiency.",
        41:"Acknowledges search-input versus result-field conflation but dismisses it as minor.",
        45:"Accepts a reasonable implication rather than requiring verbatim support.",
        46:"Acknowledges water-staining support, then objects to synthesis of a guide; unfinished."
    })
