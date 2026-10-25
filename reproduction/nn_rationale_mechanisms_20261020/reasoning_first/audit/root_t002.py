from write_root_census import save
relations={1:'S/L',2:'S/L',3:'S/R',4:'C/R',8:'U/Q',9:'B/Q',10:'C/R',11:'C/R',16:'C/R',19:'C/R',21:'S/R',23:'S/R',24:'C/R',25:'S/L',28:'S/L',33:'C/R',34:'C/R',35:'S/R',39:'S/R',40:'B/R',41:'B/R',47:'S/R',
53:'C/R',56:'S/R',58:'B/R',59:'C/R',62:'C/R',64:'C/R',65:'C/R',66:'C/R',67:'S/L',70:'S/R',72:'C/R',73:'S/R',74:'S/L',81:'S/R',84:'S/R',88:'C/R',89:'S/R',90:'C/R',92:'S/L',97:'B/R',98:'S/R'}
axes={i:['TOPIC'] for i in range(100)}
for i in [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,22,24,25,26,28,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,46,47,49,51,52,53,54,56,58,59,62,63,64,65,66,67,68,69,70,71,72,73,74,77,78,80,81,83,84,85,87,88,89,90,92,93,95,96,97,98,99]:
 axes[i]=['CONTEXT','TOPIC']
for i in [1,2,3,8,9,21,23,25,28,35,39,40,47,56,59,65,67,70,73,74,81,84,89,92,97,98]: axes[i].append('LABEL')
for i in [5,7,13,15,16,17,19,25,33,34,41,43,44,46,47,49,53,58,64,70,72,73,83,84,88,89,90,95,96,98]: axes[i].append('INTENT')
for i in [12,32,54,66,69]: axes[i].append('FACT')
for i in [34,40,44,59]: axes[i].append('SEVERITY')
save('contrast_toxigen_002',
 negative=[3,12,13,16,17,18,22,24,26,33,34,41,42,43,45,46,47,49,53,62,64,66,68,72,73,80,83,85,88,89,90,93,99],
 unclear=[44],mixed=[1],
 incomplete=[1,2,4,8,19,23,25,28,31,34,38,39,41,51,56,62,64,67,70,71,72,73,81,83,85,98],
 relations=relations,axes=axes,
 notes={1:"Derogatory/toxic prose but calls its supporting label negative; recorded mixed class wording.",
 10:"Potential literal idiom versus slur error requires target context.",
 19:"Anger about stereotyping potentially mistaken for endorsement; context needed.",
 25:"Concern about discrimination treated as toxic sentiment toward group.",
 28:"Generated text invents additional examples/prompt material after initial interpretation; preserved.",
 32:"Unsupported claim of statistical generalization and stigma; no external verification.",
 34:"Interprets GF supply as gift/marketing; compare with exact target.",
 39:"Naming-complexity statement reinterpreted as intellectual inferiority.",
 64:"Potential pun treated as respectful social exchange; context required.",
 70:"Report of innocents being grouped with terrorists treated as endorsing grouping.",
 89:"Switches to judging invented assistant reply; confirm supplied conversation.",
 96:"Accusation of threat treated as itself a direct violent threat."})
