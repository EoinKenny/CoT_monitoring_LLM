from save_aux_manual import save

save(
    "broad_contrast", "contrast_halueval_000",
    negative=[1,2,3,5,6,8,9,11,13,15,16,20,22,23,24,28,39,43,45,46],
    incomplete=[0,19,37],
    relations={5:"C/D",11:"C/D",15:"C/D",39:"C/D",45:"C/D"},
    notes={
        0:"Unfinished reconsideration of which person made an announcement.",
        10:"Interprets first film together since 1991 as implying no collaboration in 1991; verify full source.",
        15:"Calls the answer factually correct but incomplete, then assigns 50; distinguish completeness from falsity.",
        19:"Questions pronoun-to-entity binding rather than merely comparing dates.",
        26:"Separates founding from ownership, with possible external factual assumptions.",
        31:"Counts accepting an unsupported question premise as a false factual claim.",
        35:"Treats an answer about origin rather than operation location as a factual error.",
        37:"Treats elliptical A answering in what Serie as incomplete and hallucinated; requires task-context check.",
        46:"Uses birth/death dates to support no romantic involvement."
    })
save("broad_contrast", "contrast_halueval_001", negative=[0,2,3])
