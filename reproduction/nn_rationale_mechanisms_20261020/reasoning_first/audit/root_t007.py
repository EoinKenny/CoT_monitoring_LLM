from write_root_census import save
relations={2:'B/R',3:'S/R',10:'S/R',11:'S/R',12:'S/R',13:'C/R',15:'C/R',16:'S/R',17:'C/R',24:'B/R',25:'B/R',27:'C/R',33:'B/R',35:'C/R',43:'C/R',44:'C/R',45:'S/R',52:'C/R',56:'S/R',57:'C/R',65:'S/R',67:'C/R',68:'C/R',71:'C/R',72:'C/R',74:'S/R',75:'C/R',78:'S/R',81:'C/R',82:'C/R',84:'B/R',85:'S/R',87:'B/R',92:'S/R',93:'C/R'}
axes={i:['TOPIC'] for i in range(100)}
for i in [0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,24,25,26,27,28,29,30,32,33,34,35,36,39,40,42,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,63,64,65,66,67,68,70,71,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,91,92,93,95,96,97,98]:axes[i].append('CONTEXT')
for i in [0,1,3,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,24,25,26,27,28,29,30,32,34,35,36,39,40,42,43,44,45,47,48,49,50,51,52,53,54,55,56,57,58,59,61,63,64,65,66,67,68,70,71,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,91,92,93,95,96,97]:axes[i].append('INTENT')
for i in [2,3,11,12,13,15,16,24,27,33,35,43,45,52,56,57,65,68,71,74,75,78,81,82,84,87,93]:axes[i].append('LABEL')
for i in [2,11,12,14,21,22,26,28,32,34,39,40,42,49,50,51,54,55,56,63,66,67,72,75,86,89,96]:axes[i].append('FACT')
for i in [2,17,24,27,33,44,64,72,85,87]:axes[i].append('SEVERITY')
save('contrast_toxigen_007',
 negative=[3,7,11,12,13,14,21,22,25,26,27,28,29,30,32,34,36,39,40,42,44,45,47,49,50,51,52,54,55,56,57,58,59,63,66,68,71,72,75,76,79,80,81,82,86,88,92,96],
 unclear=[24,48,69],
 incomplete=[0,1,3,11,16,25,46,54,55,56,65,67,70,71,74,75,76,78,79,85,93],
 relations=relations,axes=axes,notes={
 2:"Elaborate invented interpretation of a metal-door joke may miss historical context; verify against input.",
 10:"Who-cares death joke explained as physical superiority rather than indifference to deaths.",
 22:"Names a specific Diego Hernandez incident and calls it factual; exact source verification needed.",
 24:"Explicitly less clearly toxic but no clear prose class; negative attitude is not necessarily the negative class.",
 27:"Requires explicit hate despite implicit-hate task; dark joke's stereotype may be missed.",
 35:"Speculates a reference labelling error rather than simply copying the label.",
 44:"Less severe border policy contrasted with mass violence, but explicit-toxicity requirement may be too narrow.",
 48:"Both possible implicit toxicity and absence of direct hate without a firm class.",
 52:"Educational inclusion versus segregation supplies a decision-relevant reversal.",
 57:"Calls Holocaust silence unrelated to anti-Semitism despite clear thematic relationship; context audit candidate.",
 65:"Acknowledges explicit denial of racial inferiority but overrides it with shared sentence structure.",
 67:"Appears to confuse the cop as a label of the groups with the racialized joke's punchline.",
 69:"Only target-like text and a number; no interpretable rationale conclusion.",
 73:"Describing an episode as racist is itself treated as expressing hate.",
 78:"Negated self-identification is explicitly treated as toxic identity; possible scope error.",
 82:"Explicitly denies that the reference influences classification; this is a stated rationale, not verified internal causation.",
 84:"Preserves harmful-structure interpretation despite neighbour non-toxic label.",
 88:"Complimentary generalization distinguished from derogatory stereotype.",
 93:"Neutral neighbour label explicitly rejected based on target dehumanization."})
