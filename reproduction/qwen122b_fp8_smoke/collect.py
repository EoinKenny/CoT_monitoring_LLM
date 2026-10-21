"""Pinned FP8 Qwen prefix extraction. Fail closed; never truncate or alter splits."""
import os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
                  TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1', OMP_NUM_THREADS='4')
import pathlib, json, time, hashlib, shutil, sys, re, copy, traceback, random, subprocess, fcntl
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, FineGrainedFP8Config
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel, Qwen3_5MoeTextRotaryEmbedding
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.integrations.finegrained_fp8 import replace_with_fp8_linear

SMOKE=pathlib.Path('/home/ubuntu/qwen122b-fp8-smoke-20261007')
ROOT=pathlib.Path('/home/ubuntu/qwen122b-fp8-full-20261007')
SOURCE=pathlib.Path('/home/ubuntu/experiments/reproducible-sampled-20261004/llama')
DATASETS=['wildguard','high_stakes','toxicchat','beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']
SEED=20261004
ROOT.mkdir(exist_ok=True)

def read(p):return json.loads(p.read_text())
def write(p,value):
    tmp=p.with_name(p.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2)); os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def reserve(extra=0):
    if shutil.disk_usage(ROOT).free<20*2**30+extra:raise RuntimeError('Disk reserve below 20 GiB plus required output; paused safely')
def status(state,**kw):
    d=dict(state=state,time=time.time(),**kw);write(ROOT/'STATUS.json',d);print(json.dumps(d),flush=True)
def setup():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4);torch.set_default_dtype(torch.bfloat16)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # A single frozen SDPA backend avoids automatic numerical dispatch changes.
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # Load the exact previously downloaded official kernel source, without moving Hub refs.
    import importlib.util
    from transformers.integrations.hub_kernels import _KERNEL_MODULE_MAPPING
    kernel=SMOKE/'frozen_fp8_kernel'
    manifest=read(SMOKE/'KERNEL_MANIFEST.json')
    for name,digest in manifest['files'].items():assert sha(kernel/name)==digest
    spec=importlib.util.spec_from_file_location('probe_frozen_fp8',kernel/'__init__.py',submodule_search_locations=[str(kernel)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    _KERNEL_MODULE_MAPPING['finegrained-fp8']=module

def tokenize(tok,r):
    missing_user=not any(m['role']=='user' for m in r['messages'])
    def render(mm):
        # Qwen3.5 requires a user turn, including for assistant-only source records.
        # An empty user turn adds no content and is excluded from the pooling mask.
        if missing_user:
            mm=copy.deepcopy(mm);mm.insert(1 if mm[0]['role']=='system' else 0,dict(role='user',content=''))
        return tok.apply_chat_template(mm,tokenize=False,add_generation_prompt=False,enable_thinking=False,date_string='01 Oct 2026')
    text=render(r['messages']);spans=[]
    for mi,m in enumerate(r['messages']):
        if r['role']!='all' and m['role']!=r['role']:continue
        content=m['content'];assert isinstance(content,str)
        if not content.strip():continue
        marker='PROBE_CONTENT_SENTINEL_'+hashlib.sha256((str(mi)+content).encode()).hexdigest()
        assert marker not in text
        changed=copy.deepcopy(r['messages']);changed[mi]['content']=marker;marked=render(changed)
        assert marked.count(marker)==1,(r['id'],mi,'missing content sentinel')
        before,after=marked.split(marker)
        if text==before+content+after:body=content
        elif text==before+content.strip()+after:body=content.strip()
        else:raise ValueError(f'Nonliteral content rendering for {r["immutable_id"]}: explicit review required')
        spans.append(dict(start=len(before),end=len(before)+len(body),message_index=mi,role=m['role']))
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);special=set(tok.all_special_ids)
    mask=[i for i,(tid,(a,b)) in enumerate(zip(enc['input_ids'],enc['offset_mapping'])) if tid not in special and b>a and any(a<s['end'] and b>s['start'] for s in spans)]
    assert mask,(r['id'],'empty content mask')
    result={k:r[k] for k in ['id','messages','label','split','group','input_hash','role','immutable_id']}
    result.update(token_ids=enc['input_ids'],mask=mask,n_tokens=len(enc['input_ids']),content_spans=spans,rendered_sha256=hashlib.sha256(text.encode()).hexdigest(),empty_user_template_adapter=missing_user)
    return result

def prepare(model_path):
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    summary={}
    for ds in DATASETS:
        reserve(10*2**30);out=ROOT/ds;out.mkdir(exist_ok=True);mark=out/'PREPARED.json';src=SOURCE/f'{ds}_metadata.json'
        if mark.exists():
            d=read(mark);assert d['source_sha256']==sha(src) and d['metadata_sha256']==sha(out/'metadata.json');summary[ds]=d;continue
        original=read(src);rows=[]
        for i,r in enumerate(original):
            rows.append(tokenize(tok,r))
            if i%10000==0:status('preparing',dataset=ds,rows=i,total=len(original))
        assert len({r['immutable_id'] for r in rows})==len(rows)
        for key in ['group','input_hash']:
            sets={s:{r[key] for r in rows if r['split']==s} for s in ['train','val','test']}
            assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']),(ds,key,'split leakage')
        write(out/'metadata.json',rows)
        d=dict(rows=len(rows),source_sha256=sha(src),metadata_sha256=sha(out/'metadata.json'),min_tokens=min(r['n_tokens'] for r in rows),max_tokens=max(r['n_tokens'] for r in rows),total_tokens=sum(r['n_tokens'] for r in rows))
        write(mark,d);summary[ds]=d;status('prepared',dataset=ds,**d)
    assert sum(d['rows'] for d in summary.values())==456008
    write(ROOT/'PREPARED.json',summary)
    return tok

def load_model(plan):
    cfgraw=read(pathlib.Path(plan['model_path'])/'config.json');cfg=Qwen3_5MoeTextConfig(**cfgraw['text_config']);cfg._attn_implementation='sdpa';cfg._experts_implementation='grouped_mm'
    assert cfg.num_hidden_layers==48 and cfg.hidden_size==3072
    with torch.device('meta'):
        m=Qwen3_5MoeTextModel(cfg)
        m.layers=torch.nn.ModuleList(list(m.layers[:32]));m.norm=torch.nn.Identity()
        quant=FineGrainedFP8Config(**cfgraw['quantization_config'])
        skip=[n.removeprefix('model.language_model.') for n in quant.modules_to_not_convert]
        m=replace_with_fp8_linear(m,modules_to_not_convert=skip,quantization_config=quant,pre_quantized=True)
    m.to_empty(device='cuda');m.rotary_emb=Qwen3_5MoeTextRotaryEmbedding(cfg,device='cuda')
    params=dict(m.named_parameters());coverage={k:0 for k in params};seen=set();scale_promotions=0;index=read(pathlib.Path(plan['model_path'])/'model.safetensors.index.json')['weight_map']
    needed=set(plan['needed_weight_keys']);files=sorted({index[k] for k in needed})
    with torch.no_grad():
        for fi,name in enumerate(files):
            with safe_open(str(pathlib.Path(plan['model_path'])/name),framework='pt',device='cpu') as f:
                for key in f.keys():
                    if key not in needed:continue
                    assert key not in seen;seen.add(key);short=key.removeprefix('model.language_model.');value=f.get_tensor(key)
                    mt=re.fullmatch(r'(layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)',short)
                    if mt:
                        prefix,ei,projection,kind=mt.groups();suffix='' if kind=='weight' else '_scale_inv'
                        target=prefix+('.down_proj' if projection=='down_proj' else '.gate_up_proj')+suffix
                        dest=params[target][int(ei)]
                        if projection!='down_proj':dest=dest[:dest.shape[0]//2] if projection=='gate_proj' else dest[dest.shape[0]//2:]
                        assert dest.shape==value.shape,(key,dest.shape,value.shape)
                        if dest.dtype!=value.dtype:
                            assert kind=='weight_scale_inv' and dest.dtype==torch.float32 and value.dtype in (torch.bfloat16,torch.float16),(key,dest.dtype,value.dtype)
                            scale_promotions+=1  # BF16/FP16 -> FP32 is an exact value-preserving promotion.
                        dest.copy_(value);coverage[target]+=1
                    else:
                        assert short in params,(key,'unexpected key');dest=params[short];assert dest.shape==value.shape
                        # Preserve checkpoint dtype, including FP32 state parameters.
                        if short.endswith('weight_scale_inv') and dest.dtype!=value.dtype:
                            assert dest.dtype==torch.float32 and value.dtype in (torch.bfloat16,torch.float16)
                            dest.copy_(value);scale_promotions+=1
                        elif dest.dtype!=value.dtype:
                            module_name,attr=short.rsplit('.',1);mod=m.get_submodule(module_name);p=torch.nn.Parameter(value.to('cuda'),requires_grad=False);setattr(mod,attr,p);params[short]=p
                        else:dest.copy_(value)
                        coverage[short]+=1
                    del value
            status('loading',shards=fi+1,total_shards=len(files),keys=len(seen))
    assert seen==needed
    for key,count in coverage.items():
        expected=512 if '.experts.gate_up_proj' in key else 256 if '.experts.down_proj' in key else 1
        assert count==expected,(key,count,expected)
    assert all(not p.is_meta for p in m.parameters()) and all(not b.is_meta for b in m.buffers())
    m.eval();m.requires_grad_(False)
    write(ROOT/'LOADER_VERIFIED.json',dict(source_keys=len(seen),target_parameters=len(coverage),coverage=coverage,exact_scale_promotions_to_fp32=scale_promotions,parameter_bytes=sum(p.numel()*p.element_size() for p in m.parameters()),revision=plan['revision'],block_index=31,remaining_layers_omitted=16,final_norm_omitted=True))
    return m

@torch.inference_mode()
def forward(m,rows,pad):
    torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    length=max(r['n_tokens'] for r in rows);ids=torch.full((len(rows),length),pad,dtype=torch.long,device='cuda');attn=torch.zeros_like(ids)
    for j,r in enumerate(rows):ids[j,:r['n_tokens']]=torch.tensor(r['token_ids'],device='cuda');attn[j,:r['n_tokens']]=1
    hidden=m(input_ids=ids,attention_mask=attn,use_cache=False).last_hidden_state
    assert hidden.shape==(len(rows),length,3072) and torch.isfinite(hidden).all()
    pooled=torch.stack([hidden[j,torch.tensor(r['mask'],device='cuda')].float().mean(0) for j,r in enumerate(rows)])
    values=pooled.cpu().numpy();torch.cuda.synchronize()
    return values,dict(seconds=time.perf_counter()-start,rows=len(rows),padded_tokens=len(rows)*length,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)

def smoke(m,tok):
    trials=[];shorts=[];longest=None;batch_candidates=[]
    for ds in DATASETS:
        rows=read(ROOT/ds/'metadata.json');short=min(rows,key=lambda r:r['n_tokens']);long=max(rows,key=lambda r:r['n_tokens']);shorts.append(short)
        if longest is None or long['n_tokens']>longest['n_tokens']:longest=long
        batch_candidates.extend(sorted((r for r in rows if r['n_tokens']<=1024),key=lambda r:-r['n_tokens'])[:4])
        # Smoke covers every dataset and every role policy, using frozen cases.
        values,stats=forward(m,[short],tok.pad_token_id);assert np.isfinite(values).all() and np.linalg.norm(values)>0
        trials.append(dict(dataset=ds,ids=[short['immutable_id']],**stats))
    v,a=forward(m,[longest],tok.pad_token_id);repeat,b=forward(m,[longest],tok.pad_token_id)
    assert np.array_equal(v,repeat),'Repeated identical batch differs: stop for review'
    trials.append(dict(dataset='global_longest_repeat',ids=[longest['immutable_id']],tokens=longest['n_tokens'],bitwise_equal=True,**b))
    # Batched padding must preserve a close representation of the same text.
    grouped,stats=forward(m,shorts[:4],tok.pad_token_id);cosines=[]
    for i,r in enumerate(shorts[:4]):
        single,_=forward(m,[r],tok.pad_token_id);cos=float(np.dot(grouped[i],single[0])/(np.linalg.norm(grouped[i])*np.linalg.norm(single[0])));cosines.append(cos)
    assert min(cosines)>0.999,'Batch/padding changed representation materially'
    trials.append(dict(dataset='mixed_batch',cosines=cosines,**stats))
    stress=sorted(batch_candidates,key=lambda r:(-r['n_tokens'],r['immutable_id']))[:4]
    batch_values,stats=forward(m,stress,tok.pad_token_id);batch_repeat,_=forward(m,stress,tok.pad_token_id)
    assert np.array_equal(batch_values,batch_repeat),'Repeated padded batch differs: stop for review'
    trials.append(dict(dataset='batch_budget_repeat',ids=[r['immutable_id'] for r in stress],bitwise_equal=True,**stats))
    result=dict(passed=True,trials=trials,max_batch=4,max_padded_tokens=4096,explanation='Frozen length-sorted batches, right padding, longest examples singly, no truncation.',max_tokens=longest['n_tokens'],time=time.time())
    write(ROOT/'SMOKE_PASSED.json',result);return result

def extract(m,tok):
    start=time.time();completed_this_run=0
    for ds in DATASETS:
        out=ROOT/ds;rows=read(out/'metadata.json');path=out/'features.npy';proofdir=out/'commits';proofdir.mkdir(exist_ok=True)
        if (out/'EXTRACTED.json').exists():assert sha(path)==read(out/'EXTRACTED.json')['sha256'];continue
        reserve(len(rows)*3072*4)
        x=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),3072))
        assert x.shape==(len(rows),3072) and x.dtype==np.float32
        # Freeze length-sorted batches, retaining original row indices in the feature matrix.
        order=sorted(range(len(rows)),key=lambda i:(rows[i]['n_tokens'],rows[i]['immutable_id']))
        schedule=[];batch=[]
        for i in order:
            if batch and (len(batch)>=4 or (len(batch)+1)*rows[i]['n_tokens']>4096):schedule.append(batch);batch=[]
            batch.append(i)
        if batch:schedule.append(batch)
        assert sorted(i for batch in schedule for i in batch)==list(range(len(rows)))
        schedule_path=out/'BATCHES.json'
        if schedule_path.exists():assert read(schedule_path)==schedule
        else:write(schedule_path,schedule)
        for begin in range(0,len(schedule),25):
            reserve();end=min(begin+25,len(schedule));ix=[i for batch in schedule[begin:end] for i in batch];proof=proofdir/f'{begin:07d}.json';ids=[rows[i]['immutable_id'] for i in ix]
            h=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            if proof.exists():
                p=read(proof);assert p['ids']==ids and p['sha256']==h(x[ix]);continue
            pending=proof.with_suffix('.pending.json')
            if pending.exists():raise RuntimeError(f'Uncommitted prior attempt {pending}: preserve and review before resuming')
            write(pending,dict(ids=ids,started=time.time(),begin=begin,end=end))
            timings=[]
            for bi in range(begin,end):
                indices=schedule[bi];reserve();val,stats=forward(m,[rows[i] for i in indices],tok.pad_token_id);assert np.isfinite(val).all();x[indices]=val;timings.append(stats);completed_this_run+=len(indices)
                if bi%10==0:status('extracting',dataset=ds,dataset_batches=bi+1,total_batches=len(schedule),dataset_total=len(rows),run_rows=completed_this_run,elapsed_seconds=time.time()-start,seconds_per_row=(time.time()-start)/completed_this_run)
            x.flush()
            with path.open('rb') as f:os.fsync(f.fileno())
            write(proof,dict(ids=ids,indices=ix,begin_batch=begin,end_batch=end,sha256=h(x[ix]),timings=timings,metadata_sha256=read(out/'PREPARED.json')['metadata_sha256'],schedule_sha256=sha(schedule_path)));pending.rename(proof.with_suffix('.attempt.json'))
        assert np.isfinite(x).all();write(out/'EXTRACTED.json',dict(rows=len(rows),sha256=sha(path),metadata_sha256=sha(out/'metadata.json')));del x
    write(ROOT/'COMPLETED.json',dict(time=time.time(),datasets=DATASETS,rows=456008));status('completed',rows=456008)

def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('waiting_for_download')
    while not (SMOKE/'DOWNLOAD_COMPLETED.json').exists():
        failures=list(SMOKE.glob('DOWNLOAD_FAILED_*.json'))
        if failures:raise RuntimeError(f'Download failed: {failures}')
        time.sleep(30)
    setup();plan=read(SMOKE/'DOWNLOAD_PLAN.json');reserve(15*2**30)
    provenance=dict(seed=SEED,model=plan['repo'],revision=plan['revision'],torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),script_sha256=sha(pathlib.Path(__file__)),source_root=str(SOURCE),feature_dtype='float32',pooling='mean content tokens only',block_index=31,quantization=read(pathlib.Path(plan['model_path'])/'config.json')['quantization_config'],deterministic_algorithms=True,sdpa_backend='math',experts_implementation='grouped_mm',deepgemm_disabled=True)
    write(ROOT/'PROVENANCE.json',provenance)
    (ROOT/'packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    tok=prepare(plan['model_path']);m=load_model(plan);result=smoke(m,tok)
    reserve(15*2**30);status('smoke_passed_starting_collection',tests=len(result['trials']));extract(m,tok)

if __name__=='__main__':
    try:main()
    except Exception as e:
        failure=dict(error=str(e),traceback=traceback.format_exc(),time=time.time());write(ROOT/f'FAILED_{time.time_ns()}.json',failure);status('failed',error=str(e));raise
