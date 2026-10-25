from write_root_census import save
relations={0:'S/R',1:'C/R',2:'C/R',4:'S/R',7:'B/R',8:'S/R',10:'S/R',11:'S/L',12:'S/R',15:'C/R',26:'S/R',27:'B/R',28:'S/R',29:'S/L',33:'S/R',34:'C/R',37:'B/R',38:'C/R',41:'C/R',42:'S/R',44:'S/R',45:'S/R',50:'S/R',51:'C/R',53:'B/R',59:'S/R',62:'B/R',63:'S/R',68:'S/R',69:'C/R',71:'S/R',73:'C/R',75:'S/R',77:'C/R',80:'C/R',82:'C/R',91:'S/R',95:'S/R'}
axes={i:['TOPIC'] for i in range(100)}
for i in [0,1,2,3,4,7,8,9,10,11,12,13,14,15,17,18,19,20,21,24,26,27,28,29,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,68,69,70,71,72,73,74,75,77,78,80,81,82,83,84,85,86,87,88,89,90,91,92,95,96,97,98,99]:axes[i].append('CONTEXT')
for i in [0,1,2,4,5,7,9,11,12,15,17,18,21,22,24,26,27,28,29,31,32,33,34,35,36,37,38,39,41,42,43,45,49,50,51,54,55,56,57,60,61,62,64,65,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90,91,92,95,96,97,98,99]:axes[i].append('INTENT')
for i in [1,2,7,11,12,15,26,27,29,33,42,62,68,71,77,80,82,95]:axes[i].append('LABEL')
for i in [13,14,21,24,26,33,42,45,46,49,51,54,60,61,65,68,69,78,88,89,90,99]:axes[i].append('FACT')
for i in [5,7,34,37,45,53,91]:axes[i].append('SEVERITY')
save('contrast_toxigen_006',
 negative=[1,2,9,13,14,15,21,24,26,33,34,35,36,42,49,51,56,57,60,61,64,65,68,69,71,74,78,80,82,85,87,88,89,90,92],
 mixed=[27],incomplete=[4,8,12,20,24,27,33,37,39,44,50,51,58,59,62,64,76,82,91,92,95,96,97],
 relations=relations,axes=axes,notes={
 7:"Codes-of-conduct demand treated as toxic through shared demand structure, despite explicitly less hateful content.",
 11:"Reference is explicitly said to map a linguistic trope to the toxic class.",
 27:"Repeated alternative classifications are contradictory without an explicit reconsideration bridge.",
 29:"Cannot all be Chinese interpreted as denying group identity; inspect target context.",
 33:"Reclaiming slurs distinguished from endorsing them; reference label preferred to probe disagreement.",
 34:"Explicit derogatory intent treated as required despite the task including implicit stereotypes.",
 37:"Shared ideology preserved despite different severity.",
 41:"Neighbour explicitly rejected as irrelevant; target judged from slurs and derogatory terms.",
 46:"Historical gassing reference described as historically inaccurate; factual and stance audit candidate.",
 50:"Invented additional conversational turns are generated content, not actual input evidence.",
 51:"Very high death toll in one accident treated literally as news; joke-context audit candidate.",
 54:"Reported speaker belief may be attributed to current speaker; inspect stance.",
 62:"Shared stereotype despite opposing probe classification.",
 73:"The literal word toxic in a criticism of racism is used to classify the utterance itself as toxic.",
 75:"First-person disability experience interpreted as an essentializing hostile generalization.",
 82:"Neighbour explicitly dismissed as not semantically relevant.",
 92:"Complimentary ethnic generalization judged benign, whereas another saved rationale judges a related claim positive.",
 97:"Invented subsequent examples are not the actual target or neighbour.",
 98:"Positive stereotype is described as implicit derogatory solely because it reduces complexity."})
