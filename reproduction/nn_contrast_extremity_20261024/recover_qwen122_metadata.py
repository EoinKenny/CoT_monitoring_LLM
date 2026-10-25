exec(open(__file__.replace('recover_qwen122_metadata.py','recover.py')).read().split('archives=')[0])
registry=json.load(open(B/'DEPENDENCIES_VERIFIED.json'))['files'];wanted={r['path']:r for r in registry if '/qwen122b-fp8-full-20261007/' in r['source'] and r['source'].endswith('/metadata.json')};manifest=json.load(open(R/'RECOVERED.json'))
with tarfile.open(B/'external_dependencies.tar.gz','r|gz') as t:
 for m in t:
  if m.name not in wanted:continue
  reg=wanted[m.name];ds=reg['source'].split('/')[-2];raw=t.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==reg['sha256'];meta=json.loads(raw);kept={}
  for i,x in enumerate(meta):
   sid=x.get('immutable_id',ds+':'+str(x.get('input_hash',x.get('id'))))
   if sid in need['qwen122',ds]:kept[sid]={'index':i,'label':x['label'],'split':x['split']}
  assert len(kept)==len(need['qwen122',ds]),(ds,len(kept),len(need['qwen122',ds]))
  dest=O/f'qwen122__{ds}.json';dest.write_text(json.dumps(kept));manifest.append(dict(archive=str(B/'external_dependencies.tar.gz'),member=m.name,source_sha256=reg['sha256'],local=dest.name,local_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(ds,flush=True)
  if len(manifest)==100:break
(R/'RECOVERED.json').write_text(json.dumps(manifest,indent=2));assert len(manifest)==100
