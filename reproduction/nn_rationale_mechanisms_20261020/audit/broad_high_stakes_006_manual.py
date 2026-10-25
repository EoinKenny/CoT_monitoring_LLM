from save_aux_manual import save

save(
    "broad_contrast", "contrast_high_stakes_006",
    negative=[0,1,3,5,6,9,10],
    relations={5:"S/L",9:"C/D"},
    default_axes=("SEVERITY","CONTEXT"),
    notes={
        3:"Assumes inaccuracies in business-proposal inputs have no significant consequence.",
        5:"Uses benign drill reference to support historical-information classification.",
        6:"Accepts movie-script framing as sufficient to minimise crime-instruction risk.",
        9:"Distinguishes policy explanation from compliance decision."
    })
