from write_root_census import save

relations = {
4:'C/R',6:'C/R',11:'C/R',12:'B/R',13:'S/Q',14:'S/R',15:'S/R',16:'S/R',
20:'C/R',23:'C/R',25:'S/R',28:'S/L',30:'C/R',36:'C/R',38:'S/R',39:'C/Q',
42:'S/R',45:'S/R',46:'S/R',48:'S/R',53:'S/R',55:'C/R',56:'S/R',57:'C/R',
59:'S/R',60:'S/R',61:'B/R',63:'S/R',64:'C/R',67:'C/R',68:'C/R',70:'S/R',
72:'C/R',74:'S/L',76:'S/R',81:'C/R',82:'S/R',83:'S/R',85:'S/R',86:'S/R',
96:'B/R',99:'C/R'}
axes = {i:['TOPIC'] for i in range(100)}
overrides = {
1:'CONTEXT TOPIC',2:'CONTEXT TOPIC',3:'TOPIC INTENT',4:'CONTEXT TOPIC',
6:'CONTEXT TOPIC',8:'CONTEXT TOPIC',9:'CONTEXT INTENT TOPIC',10:'CONTEXT TOPIC',
11:'CONTEXT TOPIC SEVERITY',12:'CONTEXT INTENT LABEL',13:'CONTEXT LABEL',
14:'CONTEXT INTENT',15:'CONTEXT BEHAVIOUR LABEL',16:'CONTEXT TOPIC',17:'CONTEXT TOPIC',
18:'CONTEXT TOPIC',20:'CONTEXT TOPIC',23:'CONTEXT TOPIC SEVERITY',25:'TOPIC LABEL',
26:'CONTEXT INTENT',28:'CONTEXT TOPIC INTENT',29:'CONTEXT INTENT',30:'CONTEXT TOPIC',
31:'CONTEXT INTENT',32:'CONTEXT TOPIC',33:'CONTEXT TOPIC',34:'INTENT',35:'FACT TOPIC',
36:'CONTEXT TOPIC',37:'CONTEXT TOPIC',38:'TOPIC LABEL',39:'CONTEXT FACT TOPIC',
40:'TOPIC CONTEXT',42:'TOPIC LABEL',43:'CONTEXT TOPIC',44:'CONTEXT TOPIC',
46:'CONTEXT TOPIC',47:'CONTEXT TOPIC',48:'CONTEXT TOPIC',49:'CONTEXT',
50:'BEHAVIOUR INTENT',51:'CONTEXT TOPIC',52:'CONTEXT TOPIC',53:'CONTEXT INTENT',
54:'CONTEXT FACT',55:'CONTEXT LABEL',56:'TOPIC LABEL',57:'CONTEXT INTENT',
58:'CONTEXT INTENT',59:'CONTEXT INTENT LABEL',60:'TOPIC LABEL',
61:'CONTEXT TOPIC SEVERITY',63:'TOPIC LABEL',64:'CONTEXT FACT',65:'CONTEXT TOPIC',
66:'CONTEXT TOPIC',67:'CONTEXT INTENT',68:'CONTEXT LABEL',69:'CONTEXT TOPIC',
70:'TOPIC LABEL',72:'CONTEXT TOPIC',73:'CONTEXT INTENT',74:'TOPIC LABEL',
75:'CONTEXT TOPIC',76:'CONTEXT INTENT',77:'CONTEXT BEHAVIOUR',79:'CONTEXT INTENT',
81:'CONTEXT FACT TOPIC',82:'TOPIC LABEL',83:'CONTEXT LABEL',85:'TOPIC LABEL',
86:'TOPIC LABEL',88:'CONTEXT FACT',89:'CONTEXT TOPIC',90:'CONTEXT',
91:'CONTEXT TOPIC',92:'CONTEXT INTENT',93:'FACT CONTEXT',96:'TOPIC SEVERITY CONTEXT',
97:'CONTEXT TOPIC',98:'CONTEXT INTENT',99:'CONTEXT TOPIC'}
axes.update({i:s.split() for i,s in overrides.items()})
save('contrast_toxigen_000',
 negative=[1,4,6,8,14,17,20,26,29,31,33,36,37,43,47,49,52,53,54,55,57,58,59,64,67,68,72,73,75,76,77,79,83,87,88,90,91,92,93,97,98,99],
 unclear=[12,13,34,39,51,65],mixed=[30],
 incomplete=[1,3,6,8,12,13,16,25,27,29,30,34,36,52,54,58,67,68,70,73,74,79,80,81,82,85,91,98],
 relations=relations,axes=axes,
 notes={18:"Potential target-group reassignment; exact context needed.",
 28:"Joke format and terrorism-topic transfer from neighbour.",
 30:"Initial toxic conclusion followed by repeated neutral claims; no clear explicit correction.",
 35:"Risk-stigma rationale, not an independently checked factual claim.",
 60:"Racial-class anchor and inferred Indigenous slur.",
 70:"Wavers but ends with explicit positive-class prose; coded independently of numeric fragments.",
 85:"Instruction-like tool text is generated data only; retained and not executed.",
 89:"Infers Holocaust from mass-grave/location material.",
 96:"Person-versus-group framing of gender insult."})
