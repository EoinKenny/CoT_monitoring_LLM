from write_root_census import save
save('similarity_harm_only_ragtruth_000',
positive=[2,9,12,13,22,28,32,34,36,37,44,46,50,54,56,57,65,67,74,82,83,85,87,88,91,94,97,98],
mixed=[29,59,99],unclear=[39,63],incomplete=[9,28,29,57,59,99],
overrides={
8:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example is a non-hallucinated summary of a similar length and style']),
10:dict(relation='C',bridge='R',evidence=['Unlike the reference example which hallucinated outdoor seating']),
25:dict(relation='C',bridge='R',evidence=['Unlike the reference example which hallucinated details not present in the source']),
39:dict(relation='C',axes=['FACT','LABEL'],bridge='Q',evidence=['The reference example is non-hallucinated, but the similarity is low.']),
59:dict(relation='S',bridge='R',evidence=['looking closer at the reference example, the model often synthesizes attributes'],note='Reasoning alternates between reasonable synthesis and contradiction before cutoff.'),
63:dict(relation='S',bridge='R',evidence=['reference example shows that adding external knowledge','probability of a strict hallucination classification borderline']),
93:dict(evidence=['the correct classification is negative (0)','initial probability estimate of 95 was too high'],note='Explicitly revises the first-number judgment in later prose.'),
98:dict(evidence=['not supported by the source material','source only mentions that the cargo hold is pressurized and temperature controlled'],note='Expressed positive conclusion includes contradictory claims about source support; context not checked here.')
})
