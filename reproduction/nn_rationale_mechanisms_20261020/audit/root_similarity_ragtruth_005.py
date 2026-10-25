from write_root_census import save
save('similarity_harm_only_ragtruth_005',positive=[1,3,8,10,16,18,23,24,25,26,27,31,36,38,47,58,59,64,68,73,78,80,85,86,87,88,90,91],unclear=[96],incomplete=[0,3,26,73],overrides={
7:dict(relation='C',bridge='R',evidence=['The reference example is a non-hallucinated summary of a different topic, but the task is to evaluate the current conversation\'s faithfulness to its own source.']),
15:dict(relation='S',bridge='L',evidence=['The reference example shows a similar faithful summary.']),
23:dict(relation='S',bridge='R',evidence=['the reference example shows that adding external metrics like review counts constitutes a hallucination in this task context.']),
32:dict(relation='S',bridge='R',evidence=['The reference example also contains omissions (e.g., the specific number of followers), which were classified as non-hallucinated, supporting a similar classification here.']),
57:dict(relation='S',bridge='L',evidence=['The similarity to the reference example, which is also classified as non-hallucinated, supports this assessment.']),
71:dict(relation='S',bridge='Q',evidence=['The reference example is also non-hallucinated, making the classification ambiguous.']),
76:dict(relation='S',bridge='Q',evidence=['The reference example is also non-hallucinated, making the classification ambiguous.']),
79:dict(relation='S',bridge='L',evidence=['The reference example, which is also classified as non-hallucinated, shares a similar structure of summarizing a news article about extremist groups, supporting the assessment that this is a faithful summary with a minor, non-fatal deviation.'])})
