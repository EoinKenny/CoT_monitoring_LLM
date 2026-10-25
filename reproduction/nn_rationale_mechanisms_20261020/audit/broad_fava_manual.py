from save_aux_manual import save

save(
    "broad_contrast", "contrast_fava_000",
    negative=[2,3,8,10,15,17,20,22,26,32,35,39,45,47,49],
    relations={30:"S/L"},
    notes={
        0:"Separates contributions to film from musical and treats a superlative as factual assertion.",
        5:"Adds a questionable inference that living in New York earlier rules out current residence.",
        20:"Distinguishes attributing disputed claims from asserting their truth.",
        27:"Uses references and external Olympic knowledge; comments on incompletely specified thickness.",
        29:"Treats unverifiable qualification and evaluative description as factual errors.",
        32:"Explicitly accepts subjective praise as style rather than factual error.",
        43:"Infers satisfaction reports cannot be true from the general definition of a scam.",
        45:"Explicitly accepts flowery language as stylistic.",
        48:"External geographical correction; full reference and target verification required."
    })
save(
    "broad_contrast", "contrast_fava_001",
    negative=[1,2,4,7,8,11,12,13,24,25,26,27,30,34,36,38,39,42,43,46,49],
    relations={13:"C/D",34:"C/D",38:"C/D"},
    notes={
        0:"Distinguishes geographical and historical-period claims.",
        6:"Infers diplomat nationality from envoy affiliation.",
        13:"Accepts general subjective praise while distinguishing specific invented neighbour details.",
        27:"Accepts fictional narrative frame as stylistic presentation of factual data.",
        36:"Accepts informal praise without treating it as invented fact.",
        38:"Contrasts a factual generic summary with neighbour's subjective judgement.",
        44:"Treats first-person narrative frame and subjective praise as fabricated content.",
        49:"Infers not active from currently on display."
    })
save("broad_contrast", "contrast_fava_002")
