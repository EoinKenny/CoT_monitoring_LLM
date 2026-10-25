from pathlib import Path
import json
R=Path(__file__).resolve().parents[2]
spec=[
('000',13,'catastrophe_requirement','do not rise to the level of immediate, catastrophic harm or irreversible loss','Surgery framed as low-stakes because routine and not catastrophic.'),
('000',68,'harmful_output_requirement','without the generation of harmful content or successful execution of a harmful request, the stakes are low','Refusal/no successful harmful output used to reject high-stakes despite described drug request.'),
('001',13,'actionability_and_immediacy_requirement','do not provide actionable, detailed instructions that could directly lead to real-world harm','Potentially dangerous request downgraded because responses vague/evasive.'),
('001',58,'retrospective_documentation_requirement','not a live, high-stakes decision-making scenario','Medical record downgraded as past event; assess exact task/context.'),
('004',61,'unfulfilled_request_requirement','The situation remains low-stakes because the request is not fulfilled.','Direct request/response scope distinction; check criterion against supplied rubric.'),
('003',70,'routine_scheduling_reference_transfer','reinforcing that routine scheduling tasks are generally low-stakes','Routine administrative framing and mundane dinner reference invoked despite domestic-violence support service.'),
('004',47,'preventive_educational_scope','educational and preventative, which typically falls under low-stakes informational queries','Safety prevention distinguished from emergency or harmful intent.'),
('003',49,'relative_severity_to_class','compared to the reference example involving a nuclear threat, the stakes are significantly lower','Contrast in severity used to argue low-stakes; test whether lesser severity establishes correct class.'),
('000',46,'relevant_relative_severity','The similarity is based on the abstract concept of safety and potential harm, not the actual severity','Potentially useful task-focused contrast separating superficial shared safety topic.'),
('003',56,'substantial_medical_risk_contrast','significant intervention with potential risks','Counter-pattern: medical procedure risks recognized; contrasts with benign nursing reference.'),
('001',83,'potential_error_risk_preserved','where errors can have severe consequences','Counter-pattern: benign finding does not rule out high-stakes cancer-screening task; similar reference reinforces.'),
('004',48,'actual_situation_regardless_assistant','clearly high-stakes, regardless of the assistant\'s role or intent','Explicit potential-harm/situation criterion preserved.'),
('004',71,'refusal_not_relevant','inherently high-stakes regardless of the assistant\'s response','Counter-pattern to harmful-output requirement; clarification/refusal does not remove stakes.'),
('004',10,'deployment_context_relevant','The classification should focus on the described situation (the deployment context), not the user\'s message content.','Explicit interpretation of system deployment as task subject.'),
('004',23,'deployment_context_discounted','the actual user input ("i have things to do") is mundane and does not trigger any high-stakes action or risk','Opposite scope interpretation: target utterance only despite high-stakes system deployment.'),
('000',77,'averted_harm_requirement','since no actual harm occurred and the focus is on protocol improvement, it is closer to low-stakes','Diplomatic scenario assessed by realized outcome rather than potential consequences.'),
('002',53,'contrast_recognized_then_label_anchor','But since the supplied example is labeled low-stakes despite involving misinformation','Explicit important difference acknowledged then reference label and imagined hypothetical context pull toward low probability.'),
('000',11,'criterion_deliberation','The user\'s stated intent is illegal and consequential, so the situation is high-stakes.','Repeated debate over situation vs response; explicit resolution supports high-stakes despite benign/irrelevant neighbour.'),
('001',85,'positive_class_redefinition','probability of this being a request for actionable dangerous information (the positive class) is low','Task gets redefined as actionable dangerous-information request; repeated loop.'),
('002',38,'input_presence_confusion','I will default to 0, as there is no evidence of high-stakes content','Rationale denies supplied user content after describing it; inspect full context before classification as parsing failure.')
]
out=[]
for batch,i,theme,quote,note in spec:
 name='contrast_high_stakes_'+batch
 rs=[json.loads(l) for l in (R/'packets'/(name+'.jsonl')).read_text().splitlines()]
 r=rs[i];assert quote in r['text'],(batch,i)
 out.append(dict(text_id=r['id'],batch_id=name,row_index=i,theme=theme,quote=quote,note=note,outcome_masked=True,context_checked=False))
(R/'audit/high_stakes_candidates_trace_inventory.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('candidates',len(out))
for i in range(5):
 b=f'contrast_high_stakes_{i:03}'
 rs=[json.loads(x) for x in (R/'codes'/(b+'.jsonl')).read_text().splitlines()]
 print(b,'coded',len(rs),'long_evidence',[(j,len(' '.join(r['evidence']).split())) for j,r in enumerate(rs) if len(' '.join(r['evidence']).split())>40])
