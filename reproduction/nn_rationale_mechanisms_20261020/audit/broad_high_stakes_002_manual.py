from save_aux_manual import save

save(
    "broad_contrast", "contrast_high_stakes_002",
    negative=[0,1,2,4,5,6,7,8,11,13,16,17,18,19,21,22,23,24,25,27,28,30,34,35,37,38,41,42,43,44,46,47,48],
    unclear=[32], incomplete=[32],
    relations={6:"C/D",9:"C/D",13:"C/D",16:"C/R",18:"C/D",22:"C/D",24:"C/D",25:"C/D",32:"C/Q",34:"C/D",35:"S/L",43:"C/D",44:"C/D",46:"C/D"},
    default_axes=("SEVERITY","CONTEXT"),
    notes={
        8:"Acknowledges downstream risks of growth-chart errors but requires immediate clinical action.",
        9:"Explicitly evaluates requested situation despite safe assistant refusal.",
        11:"Assumes financial-data error only slightly outdated and recreational.",
        13:"Minimises major relocation consequences by comparison with financial disaster.",
        18:"Infers public-policy rationale behind reference label.",
        23:"Claims mock/simulated audit data; source check needed.",
        28:"Treats triage information collection as low stakes absent emergency.",
        29:"Recognises downstream consequences of medical document processing.",
        32:"Reconsiders low-stakes interpretation of abuse disclosure; cut off before clear conclusion.",
        41:"Treats static colonoscopy report as low stakes.",
        43:"Uses emergency surgery reference to minimise routine operative documentation stakes.",
        44:"Compares lost pencil with missing child."
    })
