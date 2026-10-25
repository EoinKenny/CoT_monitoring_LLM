from pathlib import Path
import tarfile,json,gzip,hashlib
R=Path(__file__).resolve().parent;P=R.parent;B=P.parent/'paper_data_backup_20261021';O=R/'inputs';O.mkdir(exist_ok=True)
f=[json.loads(l) for l in gzip.open(P/'nn_mechanisms_20261015/features.jsonl.gz','rt')];need={}
for r in f:need.setdefault((r['backbone'],r['dataset']),set()).update([r['sample_id'],r['nearest_id']])
archives=[(B/'external_dependencies.tar.gz','remote_experiments/reproducible-sampled-20261004/',{'gemma','qwen','llama'}),(B/'remote_archives/grid-completion-20261008.tar.gz','grid-completion-20261008/',{'qwen122'}),(B/'remote_archives/llama70b-full-20261005.tar.gz','llama70b-full-20261005/',{'llama70'})]
manifest=[]
for archive,prefix,bbs in archives:
 with tarfile.open(archive,'r|gz') as t:
  for m in t:
   if not m.isfile() or not m.name.startswith(prefix):continue
   parts=m.name[len(prefix):].split('/')
   if len(parts)!=2 or parts[0] not in bbs:continue
   b,name=parts
   for suffix in ['_predictions.npz','_metadata.json']:
    if not name.endswith(suffix):continue
    ds=name[:-len(suffix)]
    if (b,ds) not in need:continue
    raw=t.extractfile(m).read();h=hashlib.sha256(raw).hexdigest()
    if suffix=='_predictions.npz':dest=O/f'{b}__{ds}.npz';dest.write_bytes(raw)
    else:
     meta=json.loads(raw);kept={}
     for i,x in enumerate(meta):
      sid=x.get('immutable_id',ds+':'+str(x.get('input_hash',x.get('id'))))
      if sid in need[b,ds]:kept[sid]={'index':i,'label':x['label'],'split':x['split']}
     assert len(kept)==len(need[b,ds]),(b,ds,len(kept),len(need[b,ds]))
     dest=O/f'{b}__{ds}.json';dest.write_text(json.dumps(kept))
    manifest.append(dict(archive=str(archive),member=m.name,source_sha256=h,local=dest.name,local_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(b,ds,suffix,flush=True)
(R/'RECOVERED.json').write_text(json.dumps(manifest,indent=2));assert len(manifest)==90,len(manifest) # Qwen122 metadata is recovered by recover_qwen122_metadata.py.
