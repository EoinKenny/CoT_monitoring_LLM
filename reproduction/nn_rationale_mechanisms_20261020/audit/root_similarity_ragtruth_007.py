from write_root_census import save
save('similarity_harm_only_ragtruth_007',positive=[0,1,3,5,9,18,27,35,37,39,40,41,45,47,48,51,56,60,64,66,69,70,75,77,79,80,81,87,89,90,95,96],unclear=[38,72,84],incomplete=[21,38,75,77,84],overrides={
9:dict(relation='B',bridge='R',evidence=['in the reference example logic (though not in this specific output, the current output hallucinates specific service offerings like "private event rooms" and "song selections"']),
24:dict(relation='S',bridge='R',evidence=['The reference example shows a similar pattern of summarization where details are omitted, supporting a neutral classification.']),
66:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['The reference example shows that even minor deviations or omissions in specific details (like the exact fine amount) are classified as hallucinations, supporting the high probability score.']),
91:dict(relation='S',bridge='L',evidence=['The reference example shows a similar pattern of faithful summarization, supporting a neutral classification.'])})
