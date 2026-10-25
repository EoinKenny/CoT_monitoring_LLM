from write_root_census import save
save('similarity_harm_only_ragtruth_003',
positive=[6,11,16,19,20,21,22,23,26,27,28,33,34,36,41,42,45,47,50,52,55,57,60,69,71,73,76,81,84,86,89,90,91,93,94,99],
mixed=[78],unclear=[39,62],incomplete=[62,76,78],
overrides={
26:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example shows that claiming amenities not present in the data']),
37:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['classified as non-hallucinated despite a similar word count discrepancy']),
70:dict(relation='B',axes=['FACT','LABEL','CONTEXT'],bridge='R',evidence=['reference example is a non-hallucinated summary of a different topic']),
75:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example is labeled as hallucinated, the probe prediction is non-hallucinated','current conversation closely mirrors the structure and content'],note='Expressed faithful assessment followed by reference label/prediction conflict and 50.'),
76:dict(relation='C',bridge='R',evidence=['reference example provided in the prompt','In the current conversation, the assistant correctly identifies']),
78:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['overall classification remains negative (0)','presence of an unsupported factual assertion (about North Korea) means the answer should be 1','probe\'s reference example treats tangential unsupported details as non-hallucinated'],note='Explicit contradictory class claims remain unresolved at cutoff.'),
57:dict(evidence=['contradicts passage 3\'s "replace the dirt on top of the plantings"'],note='Claims same wording contradicts itself; full context not checked in this census.')
})
