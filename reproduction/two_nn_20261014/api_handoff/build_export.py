"""Export existing frozen 2-NN prompts; never retrieve or generate new neighbours."""
import argparse, collections, csv, hashlib, json, shutil, zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    src=a.root/'canonical/cases.jsonl'
    proof=json.loads((a.root/'SETUP_COMPLETED.json').read_text())
    assert sha(src)==proof['canonical_sha256']
    seen=set(); counts=collections.Counter(); rows=[]
    for line in src.open():
        r=json.loads(line); msgs=r['messages']
        assert r['condition']=='nearest2' and r['mode']=='score'
        assert [m['role'] for m in msgs]==['system','user']
        mh=hashlib.sha256(json.dumps(msgs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        assert mh==r['messages_sha256']
        pid='nn2_'+hashlib.sha256(json.dumps([r['backbone'],r['sample_id'],mh],separators=(',',':')).encode()).hexdigest()
        assert pid==r['prompt_id'] and pid not in seen
        seen.add(pid); counts[r['backbone'],r['dataset'],r['split']]+=1
        rows.append([pid,'probability',msgs[0]['content'],msgs[1]['content']])
    assert len(rows)==50000 and len(counts)==100 and set(counts.values())=={500}
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'prompts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['prompt_id','output_type','system_prompt','user_prompt']); w.writerows(rows)
    manifest={'total':len(rows),'condition':'nearest2','canonical_sha256':sha(src),'prompts_csv_sha256':sha(a.output/'prompts.csv'),'cases_per_backbone_dataset_split':500}
    (a.output/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    base=Path(__file__).resolve().parent
    for name in ['README.txt','validate_results.py','model_manifest.template.json','responses_template.csv']:
        shutil.copyfile(base/name,a.output/name)
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(a.output.iterdir()): z.write(p,p.name)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
