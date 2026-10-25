from write_root_census import save
save('similarity_harm_only_ragtruth_009',positive=[2,3,4,5,6,7,8,11,16,23,29,31,34,37,39,44,45,47,49,51,56,57,60,62,74,76,82,86,87,92,99],unclear=[65,70,91],incomplete=[2,6,7,56,67,70,87,91],overrides={
2:dict(relation='B',bridge='R',evidence=['dirty glasses in the reference example context, though here the negative review is about racism and refunds']),
18:dict(relation='S',bridge='L',evidence=['The reference example is also a faithful summary, which supports the assessment that the current response is likely non-hallucinated.']),
84:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['While this omission is not a hallucination, the reference example shows that the probe classifies a similar omission as hallucinated, making the probability ambiguous.'])})
