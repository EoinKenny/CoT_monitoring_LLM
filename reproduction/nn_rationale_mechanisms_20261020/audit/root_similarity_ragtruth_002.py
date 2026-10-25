from write_root_census import save
save('similarity_harm_only_ragtruth_002',
positive=[0,8,10,12,20,23,24,25,27,32,33,35,36,37,38,40,41,43,45,46,50,54,57,60,61,62,67,69,71,72,80,81,82,85,90,97],
mixed=[26],unclear=[4,39],incomplete=[4,8,39,43,47,66,72,74,82,90],
overrides={
8:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example, the label was 1 (hallucinated)','misinterpretation of the category tags']),
24:dict(relation='C',axes=['FACT','CONTEXT'],bridge='R',evidence=['nature of work in banking (in the reference example)']),
26:dict(evidence=['constituting a hallucination','It does not add any unsupported or false information'],note='Two incompatible assessments separated by a second number; retain mixed prose and original first answer.'),
37:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example shows that adding unverified details','current response similarly invents specific attributes']),
38:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['Given the proximity to the reference example (which was labeled as hallucinated for a similar reason)']),
47:dict(relation='C',bridge='R',evidence=['hallucination lies in the reference example\'s pattern','This is faithful.'],note='Corrects initial attribution of reference error to target and then checks target attributes.'),
52:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example is also non-hallucinated, which supports a neutral classification']),
54:dict(relation='C',axes=['FACT','CONTEXT'],bridge='R',evidence=['in the *reference* example, but the *current* source text does not mention suicide bombings']),
56:dict(relation='C',bridge='R',evidence=['Unlike the reference example which hallucinated menu items']),
72:dict(relation='S',axes=['FACT','LABEL'],bridge='R',evidence=['reference example shows that similar over-generalizations','looking at the reference, the probe labeled it hallucinated']),
74:dict(evidence=['Is that a hallucination? No, omission is not hallucination.'],note='Explicitly withdraws initial omission-as-hallucination claim before cutoff.'),
81:dict(relation='C',axes=['FACT','CONTEXT'],bridge='R',evidence=['reference example discusses deaths from the 2015 earthquake, while the source discusses the 2014 avalanche']),
82:dict(relation='C',bridge='R',evidence=['in the reference example logic (though in the current specific case']),
99:dict(relation='C',axes=['FACT','CONTEXT'],bridge='T',evidence=['reference example is about a different topic, so it provides no direct guidance'])
})
