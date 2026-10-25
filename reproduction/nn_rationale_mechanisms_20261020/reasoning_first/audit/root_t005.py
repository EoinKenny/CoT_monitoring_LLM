from write_root_census import save
relations={4:'S/L',9:'S/L',10:'C/R',11:'S/R',15:'S/R',18:'C/R',19:'S/R',21:'B/R',23:'C/R',24:'S/R',28:'S/R',29:'S/R',30:'S/R',31:'C/R',32:'S/R',33:'S/R',34:'C/R',36:'S/R',37:'S/R',38:'C/R',40:'S/R',45:'B/R',46:'B/R',49:'B/R',50:'C/R',52:'C/R',53:'C/R',54:'S/R',57:'B/R',58:'S/L',61:'S/R',62:'C/R',64:'C/R',65:'C/R',69:'C/R',72:'B/R',76:'B/R',77:'C/R',78:'S/R',81:'C/R',83:'S/R',85:'S/R',86:'C/R',88:'C/R',89:'S/R',90:'S/L',92:'B/R',94:'C/R',95:'B/R',98:'C/R'}
axes={i:['TOPIC'] for i in range(100)}
for i in [0,1,4,5,8,9,10,14,15,16,17,18,19,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,69,70,72,75,76,77,78,79,80,81,83,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]:axes[i].append('CONTEXT')
for i in [0,5,8,9,14,17,18,19,21,22,23,24,26,27,28,31,32,33,34,35,37,38,40,45,46,47,48,49,50,52,53,54,55,57,60,61,62,64,65,69,70,72,75,76,77,78,79,80,81,83,85,87,88,89,90,91,92,95,96,97,98,99]:axes[i].append('INTENT')
for i in [4,9,10,11,15,19,21,23,24,28,29,30,33,34,36,37,40,45,49,52,53,54,57,58,61,64,65,72,76,78,85,89,90,92,94,95,98]:axes[i].append('LABEL')
for i in [5,19,26,28,33,42,44,53,69,70,78,81,89,96]:axes[i].append('FACT')
for i in [18,21,23,34,35,52,57,81,88,92]:axes[i].append('SEVERITY')
save('contrast_toxigen_005',
 negative=[0,5,17,19,21,22,24,26,27,28,30,31,33,35,37,38,42,43,46,47,48,49,50,60,61,62,63,64,65,66,69,70,72,75,80,83,87,88,89,91,96,97,98,99],
 unclear=[55],
 incomplete=[9,10,11,15,16,20,22,24,25,28,30,31,40,43,45,46,49,54,57,65,69,71,72,75,76,78,79,80,85,89,90,92,95],
 relations=relations,axes=axes,
 notes={4:"Possible victim/perpetrator reversal in attacks against Islam; full context needed.",
 16:"Invented prompt continuation is not observed neighbour evidence.",
 21:"Toxicity discussion appears to require explicit hate despite a task that includes implicit hate.",
 22:"Invented extra user/assistant records are generated continuations, not actual source passages.",
 31:"Dog-shelter joke interpreted as benign food access; context audit candidate.",
 34:"Explicitly reconsiders anti-joke interpretation and ends with a positive-class claim.",
 54:"Take-our-jobs language is interpreted as hostility towards AI classifiers; possible role contamination.",
 55:"Disability gratitude interpreted as identity reduction without a firm class.",
 70:"Victim and attacker roles appear confused although prose concludes non-toxic.",
 81:"Quran inspiration claim associated with Protocols conspiracy; unsupported association requires context audit.",
 83:"Unlike is a wording slip; described target and neighbour are both benign.",
 86:"Profanity itself used as a toxicity criterion.",
 90:"Whites-only statement associated with misogyny and neighbour prediction.",
 92:"Verify whether the direct violent claim belongs to the target or neighbour.",
 95:"Proverb interpreted as tribal dehumanization through an analogy of format.",
 96:"Generated environment correctness feedback is saved generated content, not actual validation."})
