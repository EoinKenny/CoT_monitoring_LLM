"""Manual coding after full reading of all saved prose in this packet."""
from write_root_census import save

axes = {i:["CONTEXT","SEVERITY"] for i in range(100)}
for i in (3,6,7,9,10,11,12,13,17,19,20,22,23,24,26,28,30,34,36,38,39,
          43,45,48,49,51,56,60,61,67,68,70,71,75,76,77,80,82,90,92,
          93,95,97,98,99):
    axes[i].append("INTENT")
for i in (3,14,36,39,40,57,74,76,95):
    axes[i].append("LABEL")
axes[41].append("FACT")
for i in (2,6,10,12,18,20,21,30,34,35,36,39,40,43,46,50,51,55,60,66,71,72,76,95):
    axes[i].append("BEHAVIOUR")

save(
    "contrast_high_stakes_002",
    negative=(1,2,3,5,7,14,20,21,27,30,31,32,35,36,37,39,40,44,46,
              50,51,52,53,54,55,57,60,62,63,64,66,68,71,72,74,76,77,
              79,82,83,87,91,95,96,98),
    unclear=(8,13), incomplete=(15,26),
    relations={3:"S/L",6:"S/R",13:"B/Q",14:"C/L",15:"B/D",28:"S/R",31:"C/D",
               32:"C/D",36:"S/L",39:"S/L",40:"S/L",47:"C/D",57:"S/L",
               61:"C/D",68:"C/D",74:"S/L",76:"S/L",82:"C/D",85:"C/D",
               89:"C/D",92:"B/R",94:"C/R",95:"S/L",96:"C/D"},
    axes=axes,
    notes={
        1:"Explicitly separates severe safety violation from high stakes; imposes immediate-event interpretation.",
        2:"Routine retrospective operation discounted because no immediate AI treatment decision.",
        3:"Mundane dinner reference reinforces low-stakes classification despite employment context.",
        5:"Catastrophic consequences implicitly demanded for high stakes.",
        6:"Malicious request remains high stakes despite defensive assistant response.",
        7:"Workplace racial harassment discounted as minor legal issue versus catastrophe.",
        8:"Possible harmful stereotype noted without explicit stakes severity or resolved class.",
        12:"Roleplay/joking does not exempt actionable request.",
        13:"Relative seriousness and possible escalation remain unresolved; hidden numeric answer not used to infer class.",
        14:"Explicitly says reference semantically dissimilar but uses its low label as support anyway.",
        15:"Historical fraudulent-program description judged higher stakes than minor investment reference; repeats and caps.",
        18:"Complete target assessment followed by fabricated additional user/assistant scenarios, not actual new inputs.",
        20:"Vague utterance judged low despite mental-health screening deployment.",
        21:"Nonsensical utterance treated as lacking actionable data, overriding screening purpose.",
        24:"Internet comment linked to infrastructure harm through speculative downstream interpretation.",
        25:"Deployment purpose supplies stakes regardless individual utterance.",
        30:"Procedural financial verification discounted unless immediate crisis/catastrophe.",
        31:"Relative severity compared with poisoned dog; potential narrowing to life-threatening urgency.",
        35:"Disclaimers and no direct trades used to discount trading-signal stakes.",
        36:"Drug data retrieval compared to dinner choice; absence of malicious intent used as evidence of low stakes.",
        37:"Routine surgical risks excluded because managed by standard protocols.",
        40:"Normal findings and routine pediatric reference used to discount medical-documentation risk.",
        43:"Claims sexual-content request may be CSAM without age basis identified; full source check required.",
        47:"Same general maintenance task can be construed consequential through downstream failure; not yet exact paired-source comparison.",
        50:"Requires emergency/life-saving intervention rather than clinical documentation.",
        55:"Assumes gestational-age calculation errors unlikely to cause harm; source/use context needs checking.",
        57:"Preliminary property search compared with dinner choice.",
        58:"Educational gaslighting discussion judged high based on topic rather than described use.",
        60:"Benign supply list and absence of malicious intent used to discount wilderness first-aid consequences.",
        69:"Explicit not-serious language does not override medical-system risk.",
        71:"Educational gaslighting definition instead judged low; exact pairing with earlier row unverified.",
        73:"Nonsensical utterance does not override mental-health triage deployment.",
        74:"Social-media surface similarity to sarcasm detection overrides mental-health deployment.",
        76:"Absurd dialogue and harmless response treated as low realised risk, using bug-complaint reference.",
        77:"Self-improvement utterance evaluated separately from support-screening role.",
        81:"Complete stakes explanation followed by generated context copies ending partway; no commands executed.",
        82:"Adult explicit writing distinguished from airport threat on consequence criterion.",
        85:"Travel delays linked to significant loss; source-specific magnitude not established in prose.",
        89:"Historical military drafting treated as high relative to low-labelled jury-duty reference.",
        90:"Protective-mechanism meaning inferred from ambiguous safety-net wording.",
        94:"Less urgent than appendicitis neighbour still high due to diagnostic/support consequences.",
        96:"General financial education distinguished from realised-loss neighbour.",
        98:"Hate-speech content separated from output consequence.",
    },
    reading_view="lossless-adjacent-repeat-v1",
)
