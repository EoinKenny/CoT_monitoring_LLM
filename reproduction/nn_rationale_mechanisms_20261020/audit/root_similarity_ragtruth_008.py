from write_root_census import save
save('similarity_harm_only_ragtruth_008',positive=[0,2,6,10,12,15,19,20,22,26,29,30,31,35,36,51,52,63,68,70,74,81,84,86,89,90,93,97,98],unclear=[9,27],incomplete=[7,15,27,36,61],overrides={
4:dict(relation='C',axes=['FACT','TOPIC'],bridge='Q',evidence=['The reference example is about a different restaurant (Costa Kitchen & Bar) and is not relevant to the classification of the current conversation.']),
40:dict(relation='C',bridge='R',evidence=['unlike the reference example where it was explicitly `True`']),
76:dict(relation='C',bridge='Q',evidence=['The reference example is not relevant to the current task of classifying the assistant\'s response based on the source material.'])})
