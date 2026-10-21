"""Bounded transport repair; pinned source, selected cases and inference unchanged."""
from prepare_download import ROOT, CACHE, REPO, REV, sha, save
from pathlib import Path
import concurrent.futures, json, shutil, subprocess, time, hashlib

def main():
    info=json.loads((ROOT/'DOWNLOAD_MANIFEST.json').read_text())
    assert info['repo']==REPO and info['revision']==REV
    target=Path(CACHE)/('models--'+REPO.replace('/','--'))/'snapshots'/REV
    target.mkdir(parents=True,exist_ok=True)
    start=time.time()
    def get(f):
        assert shutil.disk_usage(CACHE).free > 20*2**30
        name=f['rfilename']; p=target/name; partial=p.with_name(p.name+'.partial')
        assert not p.exists(), 'Unexpected existing file; inspect before replacing'
        url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}'
        with (ROOT/(name+'.download.log')).open('x') as log:
            subprocess.run(['curl','-4','-L','--fail','--show-error','--silent','--connect-timeout','15','--max-time','1800','--speed-limit','1024','--speed-time','60','--output',str(partial),url],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert partial.stat().st_size==f['size']
        h=hashlib.sha256()
        with partial.open('rb') as stream:
            for b in iter(lambda:stream.read(16*2**20),b''):h.update(b)
        digest=h.hexdigest()
        if f.get('lfs',{}).get('sha256'):assert digest==f['lfs']['sha256']
        partial.rename(p)
        print('Downloaded and verified',name,f['size'],round(time.time()-start,1),flush=True)
        return name,digest
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        hashes=dict(pool.map(get,info['files']))
    (ROOT/'model_path.txt').write_text(str(target))
    save(ROOT/'WEIGHTS_SHA256.json',hashes)
    save(ROOT/'DOWNLOAD_COMPLETED.json',dict(path=str(target),seconds=time.time()-start,transport='curl IPv4',weights_sha256=sha(ROOT/'WEIGHTS_SHA256.json')))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        save(ROOT/'DOWNLOAD_CURL_FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
