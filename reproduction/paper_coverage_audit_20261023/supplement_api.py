import pathlib,json,csv,gzip,sqlite3,collections,re,importlib.util
R=pathlib.Path(__file__).resolve().parent;P=R.parent.parent;C=P/'reproduction/paper_complete_20261021/api_export_compact'
s=importlib.util.spec_from_file_location('kit',C/'kit.py');k=importlib.util.module_from_spec(s);s.loader.exec_module(k)
db=k.connect();meta={p:(s,c) for p,s,c in db.execute('select prompt_id,study,condition from prompts')};count=collections.Counter();samples=collections.defaultdict(list)
with gzip.open(R/'api_recovered_mechanism_labels.jsonl.gz','wt') as out, open('/Users/eoinkenny/Desktop/six_judges_responses_5col.csv',newline='') as f:
 for r in csv.DictReader(f):
  study,cond=meta[r['prompt_id']];text=r['response'];v,st=k.parse(study,text,r['finish_reason'])
  if study=='mechanism' and r['status']=='invalid' and st!='ok':
   m=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',text.strip(),re.S)
   if m:
    v,st=k.parse(study,m.group(1),r['finish_reason'])
    if st=='ok':out.write(json.dumps(dict(model=r['model'],prompt_id=r['prompt_id'],labels=v,original_status=r['status'],raw_response=text,finish_reason=r['finish_reason'],repair='single_complete_markdown_json_fence'),ensure_ascii=False)+'\n');count['recovered_mechanism_rows']+=1
  if r['model']=='claude-opus-5.5' and r['status']=='invalid' and r['finish_reason'] in k.NORMAL and study in ['main','ablation','two_nn']:
   value,p=k.parse('reason_first',text,r['finish_reason'])
   if p=='ok':count['opus55_unsolicited_explanation_with_final_integer']+=1;count[study+'_unsolicited_explanation_with_final_integer']+=1
   if text.strip() and len(samples[study])<3:samples[study].append(dict(prompt_id=r['prompt_id'],condition=cond,response=text[:1500]))
(R/'api_format_findings.json').write_text(json.dumps(dict(counts=dict(count),samples=dict(samples),note='Unsolicited explanation outputs remain excluded from strict direct-output analyses. They are not repaired into equivalent direct-output trials.'),indent=2))
print(dict(count))
