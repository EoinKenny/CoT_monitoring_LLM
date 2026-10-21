"""Two batches only. Preserve all raw outputs and never retry ambiguous calls."""
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['CUDA_HOME']='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13'
import json, time, re, subprocess, importlib.metadata, threading, fcntl
from pathlib import Path
from prepare_download import ROOT, REPO, REV, sha, save
ROOT=Path(os.environ.get('MISTRAL_SMOKE_ATTEMPT',str(ROOT)))

def main():
    import shutil
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.tokenizers.mistral import MistralTokenizer
    lock=(ROOT/'smoke.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STARTED.json').exists(), 'Existing attempt: inspect and preserve'
    save(ROOT/'STARTED.json',dict(time=time.time(),pid=os.getpid()))
    assert shutil.disk_usage(ROOT).free > 20*2**30
    assert (ROOT/'DOWNLOAD_COMPLETED.json').exists()
    prep=json.loads((ROOT/'PREPARED.json').read_text()); assert sha(ROOT/'cases.jsonl')==prep['cases_sha256']
    rows=[json.loads(x) for x in (ROOT/'cases.jsonl').open()]
    path=(ROOT/'model_path.txt').read_text().strip()
    tok=MistralTokenizer.from_pretrained(path)
    for r in rows:
        r['token_ids']=tok.apply_chat_template(r['messages'],add_generation_prompt=True,reasoning_effort='none')
        r['input_tokens']=len(r['token_ids'])
        r['rendered_prompt']=tok.decode(r['token_ids'],skip_special_tokens=False)
    (ROOT/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    label_ids=[tok.encode(s,add_special_tokens=False) for s in ['0','1']]
    assert all(len(x)==1 for x in label_ids)
    maxlen=((max(r['input_tokens'] for r in rows)+8+255)//256)*256
    engine=dict(model=path,tokenizer_mode='mistral',config_format='mistral',load_format='mistral',dtype='auto',max_model_len=maxlen,max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.90,enable_prefix_caching=False,enforce_eager=True,seed=20261001,attention_backend='TRITON_MLA',limit_mm_per_prompt={'image':0},structured_outputs_config={'backend':'xgrammar'})
    save(ROOT/'CONFIG.json',dict(engine=engine,repo=REPO,revision=REV,reasoning_effort='none',batch_size=16,prompts_sha256=sha(ROOT/'prompts.jsonl'),source_sha256=sha(Path(__file__)),versions={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','mistral-common','compressed-tensors']},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),sampling=dict(temperature=0,seed=20261001,hard=dict(max_tokens=1,allowed_token_ids=[x[0] for x in label_ids]),probability=dict(max_tokens=8,choice=[str(i) for i in range(101)])),note='Feasibility and rough timing only. Identical 16 cases across output modes; all ten datasets, all five backbones, all three explanations. No prefix caching and eager execution; includes first-use overhead. Gold labels are never passed to generation.'))
    done=threading.Event()
    def monitor():
        with (ROOT/'gpu_samples.jsonl').open('x') as f:
            while not done.is_set():
                s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                f.write(json.dumps(dict(time=time.time(),gpu=s))+'\n'); f.flush(); done.wait(1)
    threading.Thread(target=monitor,daemon=True).start()
    save(ROOT/'STATUS.json',dict(state='loading',time=time.time(),max_input_tokens=max(r['input_tokens'] for r in rows)))
    start=time.perf_counter(); llm=LLM(**engine); load_seconds=time.perf_counter()-start
    save(ROOT/'LOADED.json',dict(seconds=load_seconds,time=time.time()))
    results=[]
    for mode in ['hard','probability']:
        assert shutil.disk_usage(ROOT).free > 20*2**30
        chunk=[r for r in rows if r['mode']==mode]; assert len(chunk)==16
        p=ROOT/(mode+'.attempt.json'); assert not p.exists()
        save(p,dict(time=time.time(),ids=[r['id'] for r in chunk],config_sha256=sha(ROOT/'CONFIG.json')))
        if mode=='hard': params=SamplingParams(temperature=0,seed=20261001,max_tokens=1,allowed_token_ids=[x[0] for x in label_ids])
        else: params=SamplingParams(temperature=0,seed=20261001,max_tokens=8,structured_outputs=StructuredOutputsParams(choice=[str(i) for i in range(101)]))
        save(ROOT/'STATUS.json',dict(state='generating',mode=mode,time=time.time()))
        start=time.perf_counter(); outputs=llm.generate([dict(prompt_token_ids=r['token_ids']) for r in chunk],params,use_tqdm=False); secs=time.perf_counter()-start
        raw=[dict(id=r['id'],prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for r,o in zip(chunk,outputs)]
        save(ROOT/(mode+'.raw.json'),raw)
        assert len(outputs)==len(chunk)
        records=[]
        for r,o in zip(chunk,outputs):
            assert o.prompt_token_ids==r['token_ids'] and len(o.outputs)==1
            g=o.outputs[0]
            valid=bool(re.fullmatch(r'[01]' if mode=='hard' else r'(?:0|[1-9][0-9]?|100)',g.text.strip()))
            if mode=='probability': valid=valid and g.finish_reason=='stop'
            else: valid=valid and len(g.token_ids)==1 and g.token_ids[0] in [x[0] for x in label_ids]
            records.append(dict(id=r['id'],dataset=r['dataset'],backbone=r['backbone'],condition=r['condition'],input_tokens=r['input_tokens'],generated_tokens=len(g.token_ids),text=g.text,valid=valid,value=int(g.text.strip()) if valid else None,finish_reason=g.finish_reason))
        report=dict(mode=mode,n=16,seconds=secs,input_tokens=sum(r['input_tokens'] for r in chunk),generated_tokens=sum(r['generated_tokens'] for r in records),valid=sum(r['valid'] for r in records),records=records,raw_sha256=sha(ROOT/(mode+'.raw.json')))
        save(ROOT/(mode+'.result.json'),report); results.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    done.set()
    save(ROOT/'RESULTS.json',dict(load_seconds=load_seconds,batches=results,rough_300k_hours=sum(r['seconds'] for r in results)/32*300000/3600,note='32 calls are only a smoke test. Runtime extrapolation is noisy, includes first-use overhead and is not a validated full-run ETA.'))
    assert sum(r['valid'] for r in results)==32,'Invalid outputs retained; smoke did not pass'
    save(ROOT/'SMOKE_PASSED.json',dict(time=time.time(),n=32,results_sha256=sha(ROOT/'RESULTS.json'),config_sha256=sha(ROOT/'CONFIG.json')))
    save(ROOT/'STATUS.json',dict(state='completed',time=time.time(),n=32))

if __name__=='__main__':
    try: main()
    except BaseException:
        import traceback
        save(ROOT/('FAILED_'+str(time.time_ns())+'.json'),dict(time=time.time(),traceback=traceback.format_exc())); raise
