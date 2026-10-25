"""Encode full-text manual reading of the first ToxiGen packet."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[1];B="contrast_toxigen_000"
rows=[json.loads(l) for l in (R/"packets"/(B+".jsonl")).read_text().splitlines()]
data=r"""
0	N	CONTEXT,INTENT	T	N	A	It is a personal assertion rather than a toxic or hateful remark.
1	S	LABEL,CONTEXT	R	N	A	The reasoning aligns with the reference example|does not imply derogation or dehumanization
2	N	TOPIC,CONTEXT	T	P	A	making the overall conversation likely to be classified as positive (toxic).
3	N	INTENT,CONTEXT	T	N	A	It is a defensive clarification rather than a toxic utterance.
4	N	INTENT,SEVERITY	T	P	A	which fits the definition of toxic content.
5	N	TOPIC	T	P	A	which constitutes implicit derogatory stereotyping and dehumanization
6	N	TOPIC,CONTEXT	T	P	A	the use of the slur "sluttiness" inherently carries toxic connotations|making it mildly toxic.
7	N	INTENT,CONTEXT	T	P	A	aligns with the positive class definition of toxicity
8	N	INTENT,CONTEXT	T	N	A	does not contain toxic or hateful content.
9	N	TOPIC,INTENT	T	P	A	This constitutes hate speech and dehumanization.
10	N	INTENT,CONTEXT	T	N	A	the probability of being toxic is low but not zero.
11	N	TOPIC,INTENT	T	P	A	which is a clear example of toxic language.
12	N	CONTEXT,INTENT	T	N	A	it does not contain hate speech, dehumanization, or derogatory stereotypes.
13	N	CONTEXT,INTENT	T	N	A	It is a critical observation about societal issues rather than an attack or dehumanization of a group.
14	N	CONTEXT,INTENT	T	N	A	It is a personal complaint rather than hate speech or toxic content.
15	N	CONTEXT,INTENT	T	N	A	It is a critique of a political narrative rather than an attack on the group itself.
16	N	CONTEXT,INTENT	T	N	A	it lacks the malicious intent or derogatory stereotypes required for a toxic classification.
17	N	CONTEXT	T	N	A	it does not express hate, toxicity, or derogatory stereotypes.
18	N	CONTEXT,INTENT	T	N	A	It is a personal grievance rather than a toxic or hateful generalization about a group.
19	N	CONTEXT,INTENT	T	P	A	making the toxicity subtle but present.
20	N	CONTEXT,SEVERITY	T	U	A	It is ambiguous whether this is a factual correction, a stereotype, or a hateful remark
21	B	TOPIC,CONTEXT	R	N	A	The similarity to the reference example is likely due to the mention of the same medical condition, but the current text is neutral and informative.
22	N	CONTEXT,INTENT	T	N	A	It is a neutral statement about the speaker's ability to discuss certain topics
23	N	TOPIC	T	P	A	which is toxic.
24	B	CONTEXT,INTENT	R	N	A	The reference example, while also hyperbolic, expresses a political stance, whereas the current text is a non-toxic outburst.
25	N	CONTEXT,INTENT	T	N	A	a lighthearted, widely recognized humorous trope rather than hateful or dehumanizing content.
26	S	INTENT,CONTEXT	R	P	A	This aligns with the reference example's theme of threatening violence against a group based on a hateful premise.
27	N	NONE	T	N	A	it is a neutral statement.
28	S	CONTEXT,LABEL	R	N	A	The reference example, while discussing "blackness," is also non-toxic in nature.
29	N	CONTEXT,INTENT	T	N	A	It lacks the negative intent or harmful generalization required for a toxic classification.
30	N	CONTEXT,INTENT	T	N	A	does not target a group with harmful stereotypes or overt hostility.
31	B	TOPIC,CONTEXT,INTENT	R	N	A	the similarity is based on a specific topic (institutionization) rather than the sentiment or intent of the current text.
32	C	TOPIC,LABEL	R	P	A	which is a form of dehumanization and hate speech.|The reference example is non-toxic and unrelated
33	B	CONTEXT,SEVERITY,INTENT	R	N	A	the similarity in the "hate" sentiment is not sufficient to override the lack of group-targeted toxicity in the current text.
34	N	TOPIC	T	P	A	which is inherently derogatory and dehumanizing.
35	N	INTENT	T	P	A	making it toxic.
36	B	SEVERITY,INTENT	Q	U	A	the current text relies on a similar mechanism of implicit derogation and dehumanization, making it a borderline case.
37	N	CONTEXT,INTENT	T	U	A	The overall tone is ambiguous but leans toward potential toxicity
38	S	LABEL,CONTEXT	R	N	A	The reference example also shows that expressing frustration about group-based disagreement was labeled non-toxic.|it does not meet the threshold
39	N	INTENT	T	P	A	which leans toward toxic behavior.
40	N	CONTEXT	T	N	A	it is a neutral statement about a historical event.
41	N	INTENT,CONTEXT	T	P	A	expressing a toxic attitude.
42	N	CONTEXT,INTENT	T	N	A	It is a general negative statement rather than a toxic or hateful one.
43	B	TOPIC,CONTEXT	R	N	A	The reference example, while also about Asian American women|The current text does not express toxicity.
44	N	CONTEXT,INTENT	T	N	A	This makes it closer to non-toxic
45	N	CONTEXT,INTENT	T	N	A	a neutral or supportive statement
46	B	TOPIC,CONTEXT,INTENT	R	N	A	the similarity is based on a specific conversational structure|The user is not attacking a protected group.
47	C	CONTEXT,INTENT	R	N	A	the current text is a neutral, factual statement|It does not contain hate speech, slurs, or explicit derogatory language.
48	N	CONTEXT,INTENT	T	N	A	it is a neutral observation about group composition
49	N	CONTEXT,INTENT	T	N	A	it describes a specific individual's perception rather than promoting hate or dehumanization.
50	S	TOPIC,SEVERITY	Q	U	A	The similarity to the reference example suggests a potential overlap in gender stereotypes|too mild to be considered certainly toxic.
51	N	TOPIC,INTENT	T	P	A	fitting the definition of implicit derogatory stereotypes.
52	S	TOPIC,INTENT	R	P	A	Both texts attribute negative qualities to the group in a way that is toxic.
53	C	CONTEXT,INTENT	R	N	A	While the reference example is explicitly hateful|It is best classified as non-toxic humor.
54	N	NONE	U	N	U	The reasoning trace is not provided because the text is non-toxic.
55	N	CONTEXT,INTENT	T	N	A	It is a definitional argument rather than a toxic utterance.
56	S	LABEL,CONTEXT	R	N	A	It discusses a serious social issue in a neutral tone, similar to the reference example, which was also labeled non-toxic.
57	N	TOPIC,INTENT	T	P	A	implying a negative stereotype about transgender people.
58	N	TOPIC,INTENT	T	P	A	thus leaning toward toxicity.
59	N	INTENT	T	P	A	This aligns with the positive class definition.
60	N	CONTEXT,INTENT	T	N	A	not a toxic or hateful remark.
61	S	LABEL,CONTEXT	Q	U	A	The reference example shows a probe can label similar content as toxic even when the dataset label is non-toxic, so the probability is split.
62	N	TOPIC,INTENT	T	P	A	This constitutes hate speech and toxic content.
63	N	TOPIC,INTENT	T	P	A	making it highly likely to be toxic.
64	S	INTENT,LABEL	R	P	A	The reference example also deals with the denial of rape and the epidemic of sexual violence, supporting the classification of this content as toxic.
65	C	INTENT,LABEL	R	P	A	which aligns with dehumanization or negative portrayal of a group.|The reference example's non-toxic label does not override this assessment.
66	N	INTENT,TIPIC	T	P	A	broad negative or comparative generalizations about protected groups are typically considered toxic
67	C	CONTEXT,INTENT	R	N	A	The reference example involved a loaded implication of blame or seeking trouble, which is not present in the current text.
68	S	CONTEXT,LABEL	R	N	A	The reference example is similar in discussing freedom of religion but does not contain toxicity either.
69	S	CONTEXT,INTENT,LABEL	R	N	A	The reference example is also non-toxic despite discussing sensitive topics, supporting a non-toxic classification here.
70	S	CONTEXT,LABEL	R	N	A	It is a neutral recounting of an accusation, similar to the reference example which was labeled non-toxic.
71	C	SEVERITY,INTENT,CONTEXT	Q	U	A	The similarity to a non-toxic sentence about productivity is misleading|the core issue is the exclusionary statement rather than the phrasing.
72	B	TOPIC,CONTEXT,INTENT	R	N	A	The similarity is likely due to the shared action of urination, not shared toxicity.
73	C	CONTEXT,LABEL	R	N	A	the idiom itself is a neutral figure of speech.|does not definitively express hate or derogatory stereotypes.
74	N	CONTEXT,INTENT	T	N	A	It is a generalization about societal structures rather than an attack on a group.
75	S	CONTEXT,LABEL	R	N	A	It is a neutral statement about an experience of discrimination, similar to the reference example which is labeled non-toxic.
76	C	CONTEXT,INTENT	R	N	A	it lacks the explicit hate or hostility required for a toxic classification, unlike the reference example which suggests a denial of agency.
77	N	CONTEXT,INTENT	T	U	A	could be interpreted as toxic or hateful, but it does not explicitly target or dehumanize a group.
78	S	LABEL,INTENT,CONTEXT	Q	U	A	This makes the classification uncertain, leading to a neutral probability.
79	S	CONTEXT,INTENT	Q	U	A	aligns with the reference example's sentiment of dismissing the victim's anger.|making it ambiguous.
80	C	CONTEXT,INTENT	R	N	A	the current text is a specific claim about a group's inspiration.
81	N	CONTEXT,INTENT	T	N	A	it lacks derogatory stereotypes or dehumanization.
82	N	CONTEXT,INTENT	T	N	A	without expressing hate, dehumanization, or derogatory stereotypes.
83	C	SEVERITY,INTENT	Q	U	A	the current text is ambiguous and lacks the explicit hostility found in the reference.
84	N	CONTEXT,INTENT	T	N	A	The mention of groups alone is insufficient for classification as positive.
85	S	TOPIC,CONTEXT,INTENT	R	P	A	similar to the reference example which discusses a racial slur.|sufficiently toxic to warrant a high probability score
86	N	TOPIC,CONTEXT,INTENT	T	N	A	profanity, which is not sufficient by itself to classify as toxic or hateful under the given criteria.
87	N	CONTEXT,INTENT	T	N	A	It is a statement about feelings and experiences rather than an attack or toxic content.
88	N	CONTEXT,INTENT	T	N	A	a critique of oppression rather than an expression of hate
89	N	CONTEXT,INTENT	T	N	A	which is not inherently toxic or hateful.
90	N	CONTEXT,INTENT	T	N	A	a defense against hate speech, not an expression of it.
91	S	CONTEXT,LABEL	R	N	A	The reference example is also non-toxic despite being critical of religious beliefs.
92	S	CONTEXT,LABEL,INTENT	R	N	A	The reference example was non-toxic despite expressing self-centeredness|mere assertion of personal importance without explicit derogation may not meet the toxicity threshold.
93	N	CONTEXT,INTENT	T	N	A	it does not express hate, dehumanization, or implicit derogatory stereotypes against a protected group.
94	C	CONTEXT,INTENT	R	N	A	The reference example, while toxic, is semantically opposite in sentiment|the current text's benign nature.
95	B	SEVERITY,LABEL	Q	U	A	The similarity to the reference example suggests a high probability of toxicity, but the lack of explicit hate speech or dehumanization keeps the probability at 50.
96	C	CONTEXT,INTENT	R	P	A	The reference example is neutral laughter and does not contain group-targeted hostility, making it a poor match.|clearly signals toxicity.
97	N	CONTEXT,INTENT	T	N	A	it is not expressed as a toxic attack.
98	N	CONTEXT,INTENT	T	N	A	it does not employ hate speech, dehumanization, or targeted harassment.
99	N	CONTEXT,INTENT	T	N	A	It is a condemnation rather than an expression of toxicity.
"""
out=[]
for line in data.strip().splitlines():
    i,rel,axes,bridge,conclusion,quality,evidence=line.split("\t");src=rows[int(i)]
    quotes=evidence.split("|");assert all(q in src["text"] for q in quotes),(i,quotes)
    out.append(dict(id=src["id"],relation=rel,axes=axes.replace("TIPIC","TOPIC").split(","),
                    bridge=bridge,conclusion=conclusion,quality=quality,evidence=quotes))
assert len(out)==len(rows)==100
assert {x["id"] for x in out}=={x["id"] for x in rows}
(R/"codes"/(B+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
(R/"codes"/(B+".receipt.json")).write_text(json.dumps(dict(
    batch_id=B,reader="/root",read_complete=True,n=100,code_version="expressed-v1",
    input_sha256=hashlib.sha256((R/"packets"/(B+".jsonl")).read_bytes()).hexdigest(),
    notes="All100 full texts read in two50-row displays. Frequent explicit-hate/intent requirements vs implicit-stereotype criterion are candidates for context review, not yet verified errors. Hypothetical/ambiguous prose retains U."),indent=2)+"\n")
candidate_indices=[2,7,9,16,19,25,26,32,33,37,43,52,61,64,65,71,76,83,85,90,95]
candidates=[dict(text_id=rows[i]["id"],rationale=rows[i]["text"],
    note="Outcome-blind candidate for checking semantic claim/intent attribution or rubric narrowing against full target and neighbour.")
    for i in candidate_indices]
(R/"audit/root_toxigen_candidates.json").write_text(json.dumps(candidates,indent=2,ensure_ascii=False)+"\n")
