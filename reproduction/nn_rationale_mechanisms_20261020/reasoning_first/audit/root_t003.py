from write_root_census import save
relations={0:'S/R',1:'B/R',3:'S/R',4:'S/R',5:'S/R',8:'S/R',9:'S/L',12:'S/R',14:'B/R',17:'C/R',18:'S/R',22:'S/R',24:'C/R',27:'C/R',28:'B/R',30:'S/R',31:'C/R',32:'C/R',39:'S/L',40:'C/R',41:'S/R',42:'S/R',44:'B/R',45:'S/R',49:'C/R',
50:'S/R',51:'S/R',52:'S/R',53:'S/R',54:'S/R',56:'B/R',57:'S/R',60:'S/R',62:'S/L',71:'S/R',72:'S/R',76:'S/R',78:'B/R',80:'S/R',82:'S/R',83:'C/R',87:'C/R',92:'S/R',96:'S/R',98:'B/R'}
axes={i:['CONTEXT','TOPIC'] for i in range(100)}
for i in [2,16,20,21,29,32,36,47,58,63,69,70,74,75,84,85,88,89,93,94,95]: axes[i]=['TOPIC']
for i in [0,1,3,4,5,8,9,12,14,17,18,22,28,30,31,32,39,41,42,44,45,49,50,51,52,53,54,56,57,60,62,71,72,76,78,80,82,92,96,98]: axes[i].append('LABEL')
for i in [0,1,2,6,7,10,13,19,22,24,25,26,27,28,34,40,43,46,48,51,53,55,56,60,61,64,65,66,67,68,71,73,76,81,82,83,86,87,90,91,92,96,97,99]: axes[i].append('INTENT')
for i in [5,13,17,35,42,43,48,53,61,65,68,72]: axes[i].append('FACT')
for i in [14,28,64]: axes[i].append('SEVERITY')
save('contrast_toxigen_003',
 negative=[8,10,11,12,13,15,17,19,22,26,27,34,37,40,42,43,46,48,49,50,51,55,59,60,61,65,66,67,68,72,73,83,86,87,90,91,97,99],
 unclear=[9,98],mixed=[76],
 incomplete=[4,6,7,8,24,28,29,31,32,35,38,45,49,52,54,57,60,63,72,76,79,80,84,86,87,91,93,96,97,98],
 relations=relations,axes=axes,
 notes={9:"Gratitude for disability reinterpreted negatively via neighbour toxic label; no firm prose class.",
 25:"Frames GIRL insult as demeaning men; compare exact semantic target.",
 35:"Generated rationale calls historical persecution trope false; not accepted as factual evidence.",
 53:"Factual prison-population claim assigned dog-whistle intent using neighbour analogy.",
 64:"Explicitly changes from insensitive joke to dehumanization criterion, yielding positive.",
 66:"Narrow group-directed criterion used to dismiss a personal insult.",
 71:"Explicitly imports implied group from reference context.",
 76:"Repeated negative class alongside positive leaning; mixed wording independent of score.",
 86:"Initial target rationale negative; subsequent invented user/assistant examples not treated as actual target conclusions.",
 93:"Reassigns comparative joke's implied dehumanized group.",
 96:"Neutral readiness question read as inherent unsuitability via negative reference.",
 98:"Ambivalent prose with threshold explanation; no unambiguous independent class conclusion."})
