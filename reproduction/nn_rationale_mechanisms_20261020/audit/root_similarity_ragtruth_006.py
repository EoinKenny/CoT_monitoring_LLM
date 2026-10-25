from write_root_census import save
save('similarity_harm_only_ragtruth_006',positive=[2,5,7,21,22,23,24,25,26,29,30,33,37,38,46,49,51,63,67,70,73,74,77,83,84,87,91,92],unclear=[44,50],incomplete=[44,47,76,77,84,94],overrides={
4:dict(relation='C',axes=['FACT','TOPIC'],bridge='Q',evidence=['the lack of strong similarity to the reference example prevents a higher confidence score.']),
22:dict(relation='S',bridge='R',evidence=['The reference example also shows a similar violation of the word count constraint.']),
23:dict(relation='S',bridge='R',evidence=['the assistant\'s response (and the reference example it mimics) claims the business has "valet service available."']),
35:dict(relation='S',axes=['FACT','LABEL'],bridge='Q',evidence=['The reference example is also non-hallucinated, supporting a neutral classification.']),
36:dict(relation='S',axes=['FACT','LABEL'],bridge='Q',evidence=['The reference example is also non-hallucinated, making the classification ambiguous.']),
71:dict(relation='C',bridge='Q',evidence=['The reference example is not relevant to the current conversation.']),
85:dict(relation='C',bridge='R',evidence=['Unlike the reference example, which hallucinated specific details (like "unexpected data usage") not present in the source text, this response remains grounded in the provided material.'])})
