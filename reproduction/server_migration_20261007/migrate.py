"""Copy, checksum-verify and validate the project without disrupting source jobs."""
import os, json, time, subprocess, pathlib, shutil, traceback, fcntl
R=pathlib.Path('/home/ubuntu/project-migration-20261007')
HOME=pathlib.Path('/home/ubuntu')
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o',f'UserKnownHostsFile={R}/known_hosts','-o','StrictHostKeyChecking=yes','-i',str(R/'transfer_key')]
import shlex
TRANSPORT=shlex.join(SSH)
def write(name,obj):
    p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def status(stage,**kw):
    write('STATUS.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
def run(name,args):
    with (R/(name+'.log')).open('a') as log:
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        write('CHILD.json',dict(stage=name,pid=p.pid,args=args,time=time.time()))
        while p.poll() is None:
            if shutil.disk_usage(HOME).free<20*2**30:
                p.terminate();p.wait();raise RuntimeError('Persistent disk below 20 GiB reserve; transfer stopped')
            time.sleep(10)
        if p.returncode not in [0]:raise RuntimeError(f'{name}: exit {p.returncode}; see preserved log')
def sync(name,src,dst,verify=False):
    status(name,source=src,destination=str(dst),verification=verify)
    cmd=['rsync','-aH','--no-owner','--no-group','--partial','--partial-dir=.migration-partial','--timeout=180','--stats','-e',TRANSPORT]
    if verify:cmd+=['--checksum','--dry-run','--itemize-changes','--out-format=CHECK %i %n%L']
    else:cmd+=['--info=progress2']
    cmd+=['ubuntu@10.0.14.197:'+src,str(dst)]
    run(name,cmd)
    if verify:
        changes=[s for s in (R/(name+'.log')).read_text().splitlines() if s.startswith('CHECK ') and not s.startswith('CHECK .d')]
        assert not changes, f'{name}: {len(changes)} checksum/metadata differences; inspect log'
    write(name+'_COMPLETED.json',dict(time=time.time(),source=src,destination=str(dst),checksum_verified=verify))
def once(name,src,dst,verify=False):
    if not (R/(name+'_COMPLETED.json')).exists():sync(name,src,dst,verify)
def main():
    lock=(R/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    nvme=pathlib.Path('/opt/dlami/nvme/interpretable-probe-models')
    assert os.path.ismount('/opt/dlami/nvme')
    nvme.mkdir(exist_ok=True);hub=HOME/'.cache/huggingface/hub'
    if not hub.exists():hub.symlink_to(nvme,target_is_directory=True)
    assert hub.is_symlink() and hub.resolve()==nvme
    pairs=[('experiments','experiments/',HOME/'experiments/'),('repository','misalignment-indicators-repro/',HOME/'misalignment-indicators-repro/'),('model_cache','.cache/huggingface/hub/',nvme),('datasets','.cache/huggingface/datasets/',HOME/'.cache/huggingface/datasets/')]
    for name,src,dst in pairs:
        pathlib.Path(dst).mkdir(parents=True,exist_ok=True)
        once('copy_'+name,src,dst)
    # Finish the already-running experiment on its original GPU; do not mix hosts mid-study.
    while True:
        sync('poll_source','experiments/llama70b-full-20261005/STATUS.json',R/'source_status.json')
        s=json.loads((R/'source_status.json').read_text())
        if s.get('state')=='completed':break
        if s.get('state')=='failed':raise RuntimeError('Source experiment halted; artifacts copied, inspect source before handover')
        status('waiting_for_source_experiment',source_status=s)
        time.sleep(60)
    sync('final_experiments','experiments/',HOME/'experiments/')
    for name,src,dst in pairs:once('verify_'+name,src,dst,True)
    # Existing environments used a system Python absent on the new host.
    runtime=HOME/'.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12'
    assert runtime.exists();changes=[]
    for root in [HOME/'experiments',HOME/'misalignment-indicators-repro']:
        for path,dirs,files in os.walk(root):
            for name in dirs+files:
                p=pathlib.Path(path)/name
                if p.is_symlink() and os.readlink(p)=='/usr/bin/python3.12':
                    changes.append(dict(path=str(p),old='/usr/bin/python3.12',new=str(runtime)));p.unlink();p.symlink_to(runtime)
    write('RUNTIME_ADAPTATION.json',dict(changes=changes,reason='Source Python 3.12.3 system runtime unavailable on destination; use existing managed CPython 3.12.14, packages unchanged. Original artifacts remain authoritative; cross-host inference is not bitwise reproduction.'))
    cpu=HOME/'misalignment-indicators-repro/.venv-uv/bin/python';gpu=HOME/'experiments/judge-pilot-20261001/.venv-vllm/bin/python'
    for name,exe in [('cpu',cpu),('judge',gpu)]:
        run('validate_'+name,[str(exe),str(R/'validate_runtime.py'),name])
    write('COMPLETED.json',dict(time=time.time(),source_preserved=True,source_experiment_complete=True,persistent_artifacts=str(HOME/'experiments'),redownloadable_weights=str(nvme),source_host='ec2-3-147-62-119.us-east-2.compute.amazonaws.com',destination_host='ec2-52-14-51-0.us-east-2.compute.amazonaws.com',free_persistent_gib=shutil.disk_usage(HOME).free/2**30))
    status('completed')
if __name__=='__main__':
    try:main()
    except Exception as e:
        write('FAILED_'+str(time.time_ns())+'.json',dict(error=str(e),traceback=traceback.format_exc()));status('failed',error=str(e));raise
