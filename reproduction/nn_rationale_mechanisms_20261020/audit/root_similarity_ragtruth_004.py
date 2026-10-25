from write_root_census import save
save('similarity_harm_only_ragtruth_004',
positive=[2,6,11,18,19,20,23,25,28,30,32,33,38,39,42,46,50,51,56,58,61,69,73,85,86,90,92,96],
unclear=[63,68],incomplete=[11,32,63,68],
overrides={
46:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example was labeled hallucinated likely due to similar synthesis issues','tendency in this dataset to penalize any inference beyond strict literal extraction']),
80:dict(relation='U',axes=['FACT','LABEL'],bridge='U',evidence=['reference example is labeled as hallucinated, the probe prediction is non-hallucinated','current response does not contain any false or unsupported factual assertions'],note='Explicit reference label/probe conflict and 50, without a specified resemblance or difference linking target to neighbour.'),
85:dict(relation='S',axes=['FACT','LABEL'],bridge='L',evidence=['probe prediction for the reference example was hallucinated (1), which supports the likelihood of the current conversation being hallucinated']),
92:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example shows that hallucinating specific details like capacity or drink types constitutes a positive classification']),
39:dict(evidence=['incorrectly states that the restaurant has a rating of 3.5 stars, when the source material specifies a rating of 3.5 stars'],note='One expressed purported contradiction repeats the same rating; source not audited here.')
})
