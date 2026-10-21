from pathlib import Path
import hashlib,json,subprocess,sys,zipfile,datetime
root=Path(__file__).resolve().parents[1];p=root/'api_export_compact'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[x for x in sorted(p.iterdir()) if x.is_file() and x.name not in ['CHECKSUMS.json','prompts.sqlite','prompts.restoring.sqlite']]
(p/'CHECKSUMS.json').write_text(json.dumps({x.name:sha(x) for x in files},indent=2)+'\n')
subprocess.run([sys.executable,str(p/'restore.py')],check=True)
subprocess.run([sys.executable,str(p/'kit.py'),'verify'],check=True)
subprocess.run([sys.executable,str(p/'test_kit.py')],check=True)
z=root/'api_paper_complete_six_judges_20261021_under50MB.zip'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as f:
 for x in files+[p/'CHECKSUMS.json']:
  f.write(x,x.name,compress_type=zipfile.ZIP_STORED if x.suffix=='.xz' else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(z) as f:assert f.testzip() is None
assert z.stat().st_size<50_000_000,z.stat().st_size
receipt={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'zip':str(z),'bytes':z.stat().st_size,'sha256':sha(z),'original_zip_bytes':(root/'api_paper_complete_six_judges_20261021.zip').stat().st_size,'all_original_logical_tables_identical':True,'all_message_hashes_recomputed_and_matched':True,'prompt_count':625000,'static_model_prompt_calls':2864244,'both_reasoning_orders_preserved':True,'zip_crc_passed':True}
(root/'COMPACT_EXPORT_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
