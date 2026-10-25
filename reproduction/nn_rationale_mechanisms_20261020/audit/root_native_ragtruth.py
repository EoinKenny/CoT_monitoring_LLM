"""All22 full native RAGTruth texts read; source passages are not neighbours."""
from save_aux_manual import save

save(
    "native_contrast","contrast_ragtruth_000",
    negative=(5,8,9,16,20),
    incomplete=(8,),
    relations={8:"S/L",9:"C/D",16:"C/D",20:"C/D"},
    axes={8:["FACT","LABEL"],9:["FACT"],16:["FACT"],20:["FACT"]},
    notes={
        0:"Checks target against source. Calls detective-apology attribution a contradiction; absent versus exclusive source claims require context verification.",
        1:"Treats divided/united as factual contradiction and former allies as unsupported; no explicit neighbour comparison.",
        2:"Tracks invented restaurant praise, waiting time, and burnt-pizza/chicken conflation.",
        3:"Marks dry/burnt chicken statement true despite own source summary saying burnt pizza; other unsupported details still identified.",
        4:"Hours/payment errors identified; quantity complaint conflated with quality while checking reviews.",
        5:"Correctly states omission is not hallucination; numbered source passages are not nearest-neighbour comparisons.",
        6:"Target-source review, not neighbour comparison. Quantity/quality conflation appears in otherwise detailed check.",
        7:"Debates phrasing but ultimately treats divided/united and allies as factual errors.",
        8:"Acknowledges contradiction then excuses it as minor wording; late invocation of non-hallucinated summary neighbour; output ends unfinished.",
        9:"Explicitly distinguishes supported target extraction from neighbour combining passages and adding outside exercises.",
        10:"Direct source checking of hours and payments; no explicit neighbour reference.",
        11:"Direct source checking; treats quantity complaint as quality complaint without distinguishing them.",
        12:"Notices quantity versus quality mismatch but excuses it; already finds separate hours/payment hallucinations.",
        13:"Source checking only; not labelled contrast prose despite contrast-context selection.",
        14:"Direct source check notices burnt chicken versus burnt pizza and invented positive review content.",
        15:"Misses chicken/pizza binding while correctly identifying invented praise.",
        16:"Explicitly uses factual binding and added-content distinction between target and reference, rather than degree of topical similarity.",
        17:"Checks target-source factual support; no explicit neighbour comparison.",
        18:"Detailed repeated target-source verification; repetition resolves to a complete valid final response.",
        19:"Treats missing parking/anonymity detail as hallucination; unlike row0 calls detective apology supported. Exact source audit needed.",
        20:"Explicit contrast between source-faithful extraction and reference combining unrelated exercise facts.",
        21:"Divided/united contradiction preserved, but claimed98-word count conflicts with row8's explicit108 count; word count is not hallucination criterion.",
    },
)
