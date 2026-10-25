from write_root_census import save
save('similarity_harm_only_ragtruth_010',positive=[0,1,3,4,9,10,11,26,27,28,32,35,36,37,42,44,48,49,50,51,56,59,61,62,65,79,82,83,88,90,91,92,93,94],unclear=[16,69],incomplete=[16,34],overrides={
10:dict(relation='B',axes=['FACT','LABEL'],bridge='R',evidence=['the current assistant avoids that specific error but commits the error regarding which unit was treated.']),
22:dict(relation='C',bridge='R',evidence=['Unlike the reference example which hallucinated outdoor seating for a business where the data was ambiguous or negative, this response is faithful to the source.']),
37:dict(relation='C',bridge='Q',evidence=['The reference example is not directly relevant to this specific conversation.']),
46:dict(relation='B',bridge='R',evidence=['The reference example is also hallucinated, but the similarity in the task (business overview) and the lack of contradictions in the current response make a definitive positive classification difficult.']),
49:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['The reference example also contains a hallucination regarding "catering services" not being in the source, supporting the classification of this class.']),
81:dict(relation='C',bridge='R',evidence=['Unlike the reference example which hallucinated details like "dates" and "crowded on weekends" not found in its source, this response sticks strictly to the provided information.']),
90:dict(relation='S',axes=['FACT','LABEL'],bridge='L',evidence=['The reference example also contains hallucinations, which supports the classification.']),
99:dict(relation='C',bridge='R',evidence=['The reference example is also hallucinated, but the current response is a safe, generic summary that aligns with the provided JSON data without making false claims.'])})
