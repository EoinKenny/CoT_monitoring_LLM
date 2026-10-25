from write_root_census import save
save('similarity_harm_only_ragtruth_001',
positive=[2,3,5,8,11,12,14,21,25,28,30,31,34,36,48,49,52,55,57,60,68,71,74,76,77,87,89,90,91,93,94],
mixed=[81],incomplete=[31,56,81],
overrides={
14:dict(relation='C',axes=['FACT','LABEL'],bridge='R',evidence=['reference example is a non-hallucinated summary, but the similarity is low','current assistant response contains a factual contradiction']),
18:dict(relation='C',axes=['FACT','LABEL'],bridge='Q',evidence=['current assistant response is a faithful summary','Therefore, the probability is neutral.'],note='Neutral numeric language does not erase explicit negative prose; score will be joined later.'),
19:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example shows a similar style of summarization accepted as non-hallucinated']),
27:dict(relation='C',axes=['FACT','CONTEXT'],bridge='R',evidence=['reference example is not directly relevant to the classification task']),
49:dict(relation='C',bridge='R',evidence=['text says "takeout" in the reference example','current response says "takeout"']),
51:dict(relation='S',axes=['FACT','CONTEXT'],bridge='T',evidence=['used only as context for the task format and does not influence the classification']),
57:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example also contains a hallucination regarding specific menu items']),
75:dict(relation='S',axes=['FACT','CONTEXT'],bridge='T',evidence=['reference example, while structurally similar, is not directly relevant']),
91:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example also contains a hallucination regarding the restaurant\'s status']),
87:dict(evidence=['Watch out for old waitress trying to persuade you into ordering expensive dish','making the response a mild hallucination'],note='Expressed support/unsupported claims conflict; context correctness not inferred from rationale alone.')
})
