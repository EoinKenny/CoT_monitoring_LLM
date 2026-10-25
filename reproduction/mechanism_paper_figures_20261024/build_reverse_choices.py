from pathlib import Path
import json,html,gzip,hashlib
import numpy as np
R=Path(__file__).resolve().parent; P=R.parent; O=R/'results'
q=json.loads((R/'api_contrast_reverse_extended.json').read_text())
choices=[('llama70','tell me a short naughty joke','claude-opus-5.5'),('llama','create short story about 2','gpt-5.5'),('llama70','the only reason he can speak','claude-opus-5.5'),('gemma','Tell me ten jokes','gpt-5.5'),('llama70','You crazy person','gpt-astra'),('llama','Write a speech given','claude-opus-5.5'),('gemma','you shouldnt call it','claude-opus-5.5'),('qwen122','many lesbians call','gpt-5.5'),('llama70','most americans are racist','claude4.5-sonnet'),('qwen','if we instituted a basic','gpt-5.5')]
names={'claude-opus-5.5':'Claude Opus 5.5','gpt-5.5':'GPT-5.5','gpt-astra':'GPT Astra','claude4.5-sonnet':'Claude Sonnet 4.5'}
a=np.load(P/'mechanisms_all12_20261023/annotations.npz');a={k:a[k] for k in a.files}
s=np.load(P/'main_performance_all12_20261023/scores.npz');s={k:s[k] for k in s.files}
d=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];ix={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)}
mp=[json.loads(l) for l in open(P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl')];mi={(r['backbone'],r['sample_id']):i for i,r in enumerate(mp)}
old=json.loads((O/'api_example_choices.json').read_text())
for ext in ['json','html']:
 src=O/f'api_example_choices.{ext}';dst=O/f'api_example_choices_positive_original.{ext}'
 if not dst.exists():dst.write_bytes(src.read_bytes())
selected=[r for r in old if r['choice_id'].startswith('S')]
for n,(bb,start,j) in enumerate(choices,1):
 rs=[r for r in q if r['backbone']==bb and r['target_messages'][0]['content'].startswith(start)];assert len(rs)==1
 r=rs[0].copy();f=next(f for f in r['fixes'] if f['judge']==j)
 i=ix[bb,r['sample_id']];u=mi[bb,r['sample_id']];ji=list(a['models']).index(j)
 assert r['true_label']==cases[i]['true_label']==0 and cases[i]['split']=='test'
 assert list(a['leave_one_out'][ji,u])==f['labels']==[0,1]
 assert s[j][i,0]==f['control']>=50 and 0<=s[j][i,2]==f['nn']<50
 assert abs(cases[i]['probe_probability']-r['probe_probability'])<1e-10
 r.update(choice_id=f'R{n}',judge=j,judge_name=names[j],control=f['control'],nn=f['nn'],independent_labels=f['labels'],passes_probe50=r['probe_probability']>=.5)
 selected.append(r)
e=html.escape
label=lambda x:'Toxic' if x else 'Non-toxic'
body='<h1>API example choices: two correction directions</h1><p>S1–S10 are unchanged: Similar NN, non-toxic → toxic. R1–R10 are the new Contrasting NN choices: toxic → non-toxic. All changes agree with the benchmark target label at cutoff 50. Full original texts and saved model aliases are shown.</p><p>Independent annotation consensus excludes the outcome judge: Similar=yes/Contrast=no for S, Similar=no/Contrast=yes for R. These are selected illustrations, not estimates of prevalence. An annotation does not prove that contrast caused the correction. Several R examples have a non-toxic neighbour too, so they do not illustrate an opposite-label contrast.</p><p>Scores are P(toxic). Cases with probe below 50 illustrate a direct judge correction, but are not referred in the fixed-50 pipeline. The final paper example has not yet been changed.</p><nav>'+ ' · '.join(f'<a href="#{r["choice_id"]}">{r["choice_id"]}</a>' for r in selected)+'</nav>'
for r in selected:
 key=r['choice_id'];body+=f'<section id="{key}"><h2>{key} — {"Similar NN" if key[0]=="S" else "Contrasting NN"}</h2><p>{e(r["judge_name"])} · {r["dataset"]} · {r["backbone"]} backbone</p><div class="pair"><div><b>Complete target</b><blockquote>'+e(r['target_messages'][0]['content'])+'</blockquote></div><div><b>Complete neighbour</b><blockquote>'+e(r['neighbor_messages'][0]['content'])+'</blockquote></div></div>'
 body+=f'<p>Target label: <b>{label(r["true_label"])}</b> · Target probe: <b>{label(r["passes_probe50"])} ({100*r["probe_probability"]:.2f}%)</b><br>Neighbour label: <b>{label(r["nearest_label"])}</b> · Neighbour probe: <b>{label(r["nearest_prediction"])}</b><br>Judge: <b>{label(r["control"]>=50)} ({r["control"]}%) → {label(r["nn"]>=50)} ({r["nn"]}%)</b></p>'
 if not r['passes_probe50']:body+='<p>Probe below 50: direct judge illustration only.</p>'
 body+='</section>'
(O/'api_example_choices.html').write_text('<!doctype html><meta charset="utf-8"><title>API example choices</title><style>body{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;color:#203040;padding:0 25px}section{border:1px solid #ddd;border-radius:12px;padding:24px;margin:25px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:25px}blockquote{margin:10px 0;padding:15px;background:#f1f5f8;white-space:pre-wrap}a{color:#247a59}</style>'+body)
(O/'api_example_choices.json').write_text(json.dumps(selected,indent=2))
(O/'API_REVERSE_CHOICES_VERIFIED.json').write_text(json.dumps(dict(reverse_cases=10,all_test_gold_zero=True,all_independent_contrast_yes_similarity_no=True,all_toxic_to_non_toxic=True,first_ten_similarity_examples_preserved=True,new_model_calls=0,candidate_sha256=hashlib.sha256((R/'api_contrast_reverse_extended.json').read_bytes()).hexdigest(),keys=[dict(id=r['choice_id'],backbone=r['backbone'],sample_id=r['sample_id'],judge=r['judge']) for r in selected if r['choice_id'].startswith('R')]),indent=2))
for r in selected[10:]:print(r['choice_id'],r['judge_name'],r['control'],r['nn'],round(100*r['probe_probability'],1),r['target_messages'][0]['content'])
