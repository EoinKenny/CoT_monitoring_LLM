from write_root_census import save
relations={3:'S/R',4:'B/R',13:'C/R',15:'S/R',16:'S/L',17:'C/R',29:'C/R',31:'C/R',34:'C/R',35:'B/R',36:'S/R',37:'S/R',42:'B/R',43:'C/R',47:'C/R',48:'C/R',50:'C/R',52:'C/R',53:'C/R',54:'C/R',55:'S/R',57:'S/R',63:'C/R',64:'S/R',66:'C/R',68:'S/R',72:'S/R',74:'B/R',75:'B/R',77:'C/R',78:'S/R',84:'B/R',85:'S/R',88:'S/R',90:'S/R',93:'S/R'}
axes={i:['TOPIC'] for i in range(95)}
for i in [0,1,2,3,4,5,8,9,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,72,73,74,75,76,77,78,79,80,81,82,83,84,85,87,88,89,90,91,92,93,94]:axes[i].append('CONTEXT')
for i in [0,1,2,3,4,5,8,9,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,72,74,75,76,77,78,79,80,81,82,83,84,85,87,88,89,90,91,92,93,94]:axes[i].append('INTENT')
for i in [3,4,13,15,16,17,29,31,34,35,36,37,42,43,47,48,50,52,53,54,55,57,63,64,66,68,72,74,75,77,78,84,85,88,90,93]:axes[i].append('LABEL')
for i in [8,9,12,13,16,17,18,22,24,25,29,30,32,33,34,35,40,41,44,46,47,48,51,56,57,62,63,67,73,81,84,87,88,90,91,92]:axes[i].append('FACT')
for i in [20,26,31,43,47,52,54,59,64,68,74,77,79]:axes[i].append('SEVERITY')
save('contrast_toxigen_012',
 negative=[0,2,3,5,8,9,11,12,13,16,17,18,20,22,23,24,25,28,29,30,32,33,34,35,40,41,43,44,46,47,48,53,54,56,58,61,62,63,67,70,75,77,79,81,84,87,91,92,93],
 unclear=[6],mixed=[76],
 incomplete=[4,15,19,29,33,35,36,37,54,57,70,74,76,80,84,86,90,92],
 relations=relations,axes=axes,notes={
 3:"Explicitly tolerates mocking tone with reference non-toxic convention.",
 4:"Shared religion topic separated from hostile versus neutral sentiment.",
 6:"Target-like echo plus numeric fragment, without a rationale.",
 12:"Reporting or retweeting hate distinguished from expressing it.",
 16:"Near-identical reference used to support historical-reporting interpretation.",
 17:"Preserves attacks-against-Islam direction rather than treating Muslims as perpetrators.",
 19:"Context-removal remark read as admission of hateful intent; verify scope.",
 25:"Mockery aimed at political discourse distinguished from mockery of Muslims.",
 31:"Family-utility rationale distinguished from specific vegetative-state context.",
 35:"Shared criticism structure separated from supportive discussion of biphobia.",
 37:"Explicitly invokes hate-speech dataset patterns and reference reverse-racism narrative to classify terminology criticism.",
 42:"Similarity in stereotype structure preserved despite non-toxic reference label and probe.",
 47:"Reference probe used to support meme/game interpretation despite weak content similarity.",
 51:"Statistical description itself called dehumanizing because unsupported; task criterion may broaden.",
 52:"Less overtly hateful than reference but negative cultural generalization still judged sufficient.",
 64:"Negative assessment means adverse sentiment, not negative class; prose explicitly toxic.",
 68:"Anti-joke/slur mechanism compared with direct-hate reference rather than same stereotype.",
 75:"Acknowledges reference's implied toxic trope but ends with explicit non-toxic target conclusion.",
 76:"Cycles between positive and negative prose classes without a coherent resolution.",
 77:"Requires unambiguous explicit toxicity despite implicit-hate task.",
 80:"Numbers fluctuate while prose consistently claims toxic; numeric variation does not make prose mixed.",
 84:"Distinguishes identical grammatical structure from completely different semantic content.",
 86:"Prose alleges inferiority despite repeated tiny numeric fragments.",
 90:"Infers anti-trans meaning of duck idiom, reinforced by probe despite non-toxic reference label."})
