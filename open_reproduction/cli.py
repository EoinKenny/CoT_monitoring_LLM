import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import urllib.parse
import venv

from .common import ROOT, BACKBONES, DATASETS, JUDGES, frozen, lock, log, read, reserve, sha, write

STUDIES = 'main,two_nn,ablation,mechanism,reason_after,reason_first,trace_labels'


def choices(value, allowed):
    result = value.split(',')
    if not result or len(set(result)) != len(result) or set(result)-set(allowed):
        raise argparse.ArgumentTypeError('Choose comma-separated entries from '+','.join(allowed))
    return result


def code_hashes():
    paths = [ROOT/'reproduce_open_source.py',ROOT/'paper/collect.py']
    paths += list((ROOT/'open_reproduction').rglob('*.py'))
    paths += [p for p in (ROOT/'open_reproduction/templates').iterdir() if p.is_file()]
    paths += [ROOT/'configs/backbones.json',ROOT/'configs/local_judges.json',ROOT/'configs/rebuild_inputs.json',ROOT/'requirements-backbones.txt',ROOT/'requirements-backbone-qwen122.txt',ROOT/'requirements-judges.txt',ROOT/'data/prompts/trace_label_prompt.txt']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def environment(out, name, override=None):
    if override:
        path = Path(override).expanduser().absolute()
        if not path.exists():
            raise ValueError(f'Missing Python runtime: {path}')
        return path
    folder = out/'.envs'/name
    python = folder/'bin/python'
    requirement = ROOT/{'backbones':'requirements-backbones.txt','backbone-qwen122':'requirements-backbone-qwen122.txt','judges':'requirements-judges.txt'}[name]
    receipt = folder/'READY.json'
    fingerprint = dict(requirements_sha256=sha(requirement),python='3.12')
    if receipt.exists():
        frozen(receipt,fingerprint)
        return python
    reserve(out,12*2**30)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(folder)
    subprocess.run([str(python),'-m','pip','install','--upgrade','pip'],check=True)
    if name == 'backbones':
        subprocess.run([str(python),'-m','pip','install','torch==2.9.0','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(requirement)],check=True)
    write(receipt,fingerprint)
    return python


def inputs(config):
    manifest = read(ROOT/'configs/rebuild_inputs.json')
    dest = Path(config['inputs_dir'])
    dest.mkdir(parents=True,exist_ok=True)
    for name, entry in manifest['files'].items():
        if entry['backbone'] not in config['backbones'] or entry['dataset'] not in config['datasets']:
            continue
        target = dest/name
        if target.exists():
            if sha(target) != entry['sha256']:
                raise ValueError(f'Input checksum mismatch: {target}')
            continue
        reserve(dest,entry['bytes'])
        tmp = target.with_suffix('.download')
        # Private releases require GH_TOKEN/GITHUB_TOKEN or an authenticated gh.
        # No credentials enter RUN.json, command arguments, or saved manifests.
        token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
        if shutil.which('gh') and not token:
            with tmp.open('wb') as f:
                subprocess.run(['gh','api',f'repos/{manifest["repository"]}/releases/assets/{entry["asset_id"]}','-H','Accept: application/octet-stream'],stdout=f,check=True)
        else:
            url = entry['api_url'] if token else entry['url']
            headers = {'Accept':'application/octet-stream'}
            if token:
                headers['Authorization'] = 'Bearer '+token
            class SafeRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, response_headers, newurl):
                    redirected = super().redirect_request(req,fp,code,msg,response_headers,newurl)
                    if redirected and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
                        redirected.remove_header('Authorization')
                    return redirected
            opener = urllib.request.build_opener(SafeRedirect())
            with opener.open(urllib.request.Request(url,headers=headers),timeout=120) as response, tmp.open('wb') as f:
                shutil.copyfileobj(response,f,2**20)
        if sha(tmp) != entry['sha256']:
            raise ValueError(f'Download checksum mismatch: {name}')
        tmp.replace(target)


def child(python, module, out, *args, extra_env=None):
    env = dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1')
    env.pop('HF_HUB_OFFLINE',None)
    env.update(extra_env or {})
    subprocess.run([str(python),'-m',module,'--out',str(out),*args],cwd=ROOT,env=env,check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/open_source_rebuild')
    parser.add_argument('--backbones',type=lambda s:choices(s,BACKBONES),default=BACKBONES)
    parser.add_argument('--judges',type=lambda s:choices(s,JUDGES),default=JUDGES)
    parser.add_argument('--datasets',type=lambda s:choices(s,DATASETS),default=DATASETS)
    parser.add_argument('--studies',type=lambda s:choices(s,STUDIES.split(',')),default=STUDIES.split(','))
    parser.add_argument('--mechanism-judges',type=lambda s:choices(s,JUDGES),default=['qwen122','qwen','gemma'])
    parser.add_argument('--trace-judge',choices=JUDGES,default='qwen122')
    parser.add_argument('--model-cache',type=Path,default=Path(os.environ.get('HF_HUB_CACHE',Path.home()/'.cache/huggingface/hub')))
    parser.add_argument('--model-paths',type=Path,help='Optional JSON with backbone/judge alias -> existing pinned snapshot path')
    parser.add_argument('--inputs-dir',type=Path)
    parser.add_argument('--backbone-python',type=Path,help='Use a compatible existing extraction runtime instead of installing one')
    parser.add_argument('--qwen-backbone-python',type=Path,help='Qwen122 FP8 extraction runtime: Transformers 5.18 / PyTorch 2.13')
    parser.add_argument('--judge-python',type=Path,help='Use a compatible existing vLLM runtime instead of installing one')
    parser.add_argument('--mistral-overlay',type=Path,help='Existing original Transformers 5.4 / tokenizers 0.22.2 overlay')
    parser.add_argument('--smoke',action='store_true',help='48 short records per dataset, four held-out examples per split; separate output required')
    parser.add_argument('--dry-run',action='store_true',help='Show stages and resource requirements; no downloads or inference')
    parser.add_argument('--stop-after',choices=['inputs','backbones','prompts','judges','all'],default='all')
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if any(s in args.studies for s in ['reason_after','reason_first']) and 'main' not in args.studies:
        parser.error('Reasoning selection requires main in --studies')
    if 'trace_labels' in args.studies and 'reason_after' not in args.studies:
        parser.error('Trace labelling requires reason_after in --studies')
    config = dict(protocol='fresh-open-source-v1',backbones=args.backbones,judges=args.judges,datasets=args.datasets,studies=args.studies,mechanism_judges=[j for j in args.mechanism_judges if j in args.judges],trace_judge=args.trace_judge,smoke=args.smoke,sample_n=4 if args.smoke else 500,model_cache=str(args.model_cache.expanduser().resolve()),inputs_dir=str((args.inputs_dir or out/'inputs').expanduser().resolve()),model_paths=read(args.model_paths) if args.model_paths else {},code_sha256=code_hashes())
    plan = dict(backbones=config['backbones'],judges=config['judges'],datasets=config['datasets'],studies=config['studies'],smoke=config['smoke'],stages=['verify/download dataset records','download pinned backbone weights','extract fresh activations','fit train-only scaler and logistic probe','select validation/test samples','retrieve exact activation-space 1-NN/2-NN and compute saliency','construct approved prompts','collect local judges with durable journal','select own-judge flips and collect both explanation orders','Qwen122 labels written probability-first explanations','export raw responses and coverage'],requirements='Linux, Python 3.12, CUDA-capable NVIDIA driver. Full grid: 96 GiB GPU, at least 128 GiB host RAM, approximately 1 TB free storage recommended. Gated Hugging Face models require accepted licences and HF_TOKEN. No API calls. Full run takes days.',output=str(out))
    if args.dry_run:
        print(json.dumps(plan,indent=2))
        return
    if platform.system() != 'Linux' or sys.version_info[:2] != (3,12):
        parser.error('Run collection on Linux with Python 3.12; --dry-run works elsewhere.')
    subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],check=True)
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'controller.lock'):
        frozen(out/'RUN.json',config)
        write(out/'PLAN.json',plan)
        log(out,'verifying_dataset_inputs')
        inputs(config)
        if args.stop_after == 'inputs':
            return
        bp = environment(out,'backbones',args.backbone_python) if any(b != 'qwen122' for b in config['backbones']) else None
        qp = environment(out,'backbone-qwen122',args.qwen_backbone_python) if 'qwen122' in config['backbones'] else None
        for bb in config['backbones']:
            child(qp if bb == 'qwen122' else bp,'open_reproduction.backbone_worker',out,'--backbone',bb)
        if args.stop_after == 'backbones':
            return
        from .prompts import prepare
        log(out,'building_fresh_prompts')
        prepare(out)
        if args.stop_after == 'prompts':
            return
        jp = environment(out,'judges',args.judge_python)
        overlay = args.mistral_overlay
        if 'mistral' in config['judges'] and overlay is None:
            overlay = out/'.envs/mistral_overlay'
            receipt = overlay/'READY.json'
            if not receipt.exists():
                subprocess.run([str(jp),'-m','pip','install','--no-deps','--target',str(overlay),'transformers==5.4.0','tokenizers==0.22.2'],check=True)
                write(receipt,dict(transformers='5.4.0',tokenizers='0.22.2'))
        for judge in config['judges']:
            env = {'PYTHONPATH':str(overlay)} if judge == 'mistral' else {}
            child(jp,'open_reproduction.judge_worker',out,'--judge',judge,extra_env=env)
        if args.stop_after == 'judges':
            return
        if 'trace_labels' in config['studies']:
            child(jp,'open_reproduction.judge_worker',out,'--judge',config['trace_judge'],'--traces-only')
        from .export import export
        report = export(out)
        log(out,'collection_finished',**report)


if __name__ == '__main__':
    main()
