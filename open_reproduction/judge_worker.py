"""Local vLLM inference with immutable prompts and a durable no-retry journal."""
import argparse
import collections
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sysconfig
import time

from paper.collect import CAPS, INT, digest as message_digest, initialise, parse, trace_jobs
from .common import ROOT, digest, frozen, log, read, reserve, sha, write
from .models import model_path
from .prompts import jobs


def flip_cases(con, alias):
    scores = collections.defaultdict(dict)
    for case, condition, parsed in con.execute('SELECT case_key,condition,parsed FROM requests WHERE model=? AND study="main" AND split="test" AND status="valid"',(alias,)):
        scores[case][condition] = json.loads(parsed)
    return {k for k,v in scores.items() if 'baseline' in v and 'nearest' in v and (v['baseline']>=50) != (v['nearest']>=50)}


def save_results(con, alias, payload):
    for row in payload:
        parsed, status = parse(row['study'],row['text'],row['finish_reason'])
        con.execute('UPDATE requests SET status=?,result_time=?,response=?,parsed=?,finish_reason=?,returned_model=?,raw_json=?,latency_ms=?,input_tokens=?,output_tokens=? WHERE model=? AND prompt_id=?',
                    (status,row['time'],row['text'],json.dumps(parsed),row['finish_reason'],row['model'],json.dumps(row),row['seconds']*1000,len(row['input_token_ids']),len(row['output_token_ids']),alias,row['prompt_id']))
    con.commit()


def dispatch(con, alias, chunk):
    # Commit BEFORE calling the backend. Resume must never repeat an attempted ID.
    with con:
        for job in chunk:
            con.execute('INSERT INTO requests(model,prompt_id,study,case_key,condition,split,messages_hash,status,dispatch_time) VALUES (?,?,?,?,?,?,?,?,?)',
                        (alias,job['prompt_id'],job['study'],job.get('case_key'),job.get('condition'),job.get('split'),message_digest(job['messages']),'dispatched',time.time()))


def run(out, alias, traces_only=False):
    cuda = Path(sysconfig.get_paths()['purelib'])/'nvidia/cu13'
    if cuda.exists():
        os.environ['CUDA_HOME'] = str(cuda)
    config = read(out/'RUN.json')
    root = out/'judges'/alias
    root.mkdir(parents=True, exist_ok=True)
    con = initialise(out/'responses.sqlite',config)
    for raw in sorted((root/'batches').glob('*.raw.json')):
        payload = read(raw)
        # Recover only from a durable raw result; never regenerate the request.
        pending = [r for r in payload if con.execute('SELECT status FROM requests WHERE model=? AND prompt_id=?',(alias,r['prompt_id'])).fetchone() == ('dispatched',)]
        if pending:
            save_results(con,alias,pending)
    path, spec = model_path('judge',alias,config['model_cache'],config['model_paths'])
    versions = {p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','tokenizers','mistral-common','compressed-tensors']}
    expected = dict(spec.get('versions',read(ROOT/'configs/local_judges.json')['olmo']['versions']))
    if versions != expected:
        raise RuntimeError(f'Judge runtime differs from pinned versions: observed={versions}, expected={expected}')
    if alias == 'mistral':
        from vllm.tokenizers.mistral import MistralTokenizer
        tok = MistralTokenizer.from_pretrained(str(path))
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    frozen(root/'ENVIRONMENT.json',dict(spec=spec,versions=versions,tokenizer_hashes={p.name:sha(p) for p in path.iterdir() if p.is_file() and p.suffix in ['.json','.model','.jinja'] and 'safetensors' not in p.name}))
    queue = sqlite3.connect(root/'tokens.sqlite')
    queue.execute('CREATE TABLE IF NOT EXISTS tokens(prompt_id TEXT PRIMARY KEY,job TEXT,token_ids TEXT,n INTEGER)')
    known = {r[0] for r in queue.execute('SELECT prompt_id FROM tokens')}
    studies = ['trace_labels'] if traces_only else [s for s in config['studies'] if s != 'trace_labels' and (s != 'mechanism' or alias in config['mechanism_judges'])]
    for study in studies:
        iterator = trace_jobs(con,alias) if study == 'trace_labels' else jobs(out,study)
        for i, job in enumerate(iterator):
            if job['prompt_id'] in known:
                continue
            kwargs = spec['chat_template_kwargs']
            if alias == 'mistral':
                ids = tok.apply_chat_template(job['messages'],**kwargs)
            else:
                text = tok.apply_chat_template(job['messages'],**kwargs)
                ids = tok.encode(text,add_special_tokens=False)
            if not ids or len(ids)+CAPS[study] > spec['context_limit']:
                raise ValueError(f'Prompt exceeds context; no truncation allowed: {job["prompt_id"]}')
            queue.execute('INSERT INTO tokens VALUES (?,?,?,?)',(job['prompt_id'],json.dumps(job,ensure_ascii=False),json.dumps(ids),len(ids)))
            if i % 1000 == 0:
                queue.commit()
                reserve(out)
                log(out,'tokenizing_judge',judge=alias,study=study,completed=i+1)
        queue.commit()
        con.commit()  # trace-source mappings
    attempted = {r[0] for r in con.execute('SELECT prompt_id FROM requests WHERE model=?',(alias,))}
    engine = dict(spec['engine'],model=str(path))
    maximum = queue.execute('SELECT max(n) FROM tokens').fetchone()[0] or 0
    engine['max_model_len'] = max(2048,((maximum+max(CAPS[s] for s in studies)+255)//256)*256) if studies else 2048
    phase = 'traces' if traces_only else 'responses'
    frozen(root/(phase+'_ENGINE.json'),dict(engine=engine,versions=versions))
    llm = None
    batch_dir = root/'batches'
    batch_dir.mkdir(exist_ok=True)
    for study in studies:
        flips = flip_cases(con,alias) if study in ['reason_after','reason_first'] else None
        selected = []
        for encoded, ids in queue.execute('SELECT job,token_ids FROM tokens ORDER BY rowid'):
            job = json.loads(encoded)
            if job['study'] != study or (flips is not None and job['case_key'] not in flips):
                continue
            selected.append(job['prompt_id'])
        frozen(root/(study+'_PLAN.json'),dict(prompt_ids=selected,selection='own valid control versus 1-NN test flips at 50' if flips is not None else 'all scheduled cases',cap=CAPS[study]))
        pending = [pid for pid in selected if pid not in attempted]
        if not pending:
            continue
        if llm is None:
            reserve(out)
            log(out,'loading_judge',judge=alias,pending=len(pending))
            from vllm import LLM
            llm = LLM(**engine)
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        options = dict(temperature=0,seed=20261001,max_tokens=CAPS[study])
        if study in ['main','two_nn','ablation']:
            options['structured_outputs'] = StructuredOutputsParams(regex=r'(?:0|[1-9][0-9]?|100)')
        sampling = SamplingParams(**options)
        for start in range(0,len(pending),spec['batch_size']):
            reserve(out)
            ids = pending[start:start+spec['batch_size']]
            chunk = []
            for pid in ids:
                job, tokens = queue.execute('SELECT job,token_ids FROM tokens WHERE prompt_id=?',(pid,)).fetchone()
                chunk.append(dict(json.loads(job),token_ids=json.loads(tokens)))
            key = digest(json.dumps(ids,separators=(',',':')))
            dispatch(con,alias,chunk)
            attempted.update(ids)
            write(batch_dir/(key+'.attempt.json'),dict(prompt_ids=ids,study=study,engine_sha256=sha(root/(phase+'_ENGINE.json')),time=time.time()))
            before = time.perf_counter()
            try:
                generated = llm.generate([dict(prompt_token_ids=j['token_ids']) for j in chunk],sampling,use_tqdm=False)
                seconds = time.perf_counter()-before
                if len(generated) != len(chunk):
                    raise ValueError('Backend returned an incomplete batch')
                payload = []
                for job, response in zip(chunk,generated):
                    if len(response.outputs) != 1:
                        raise ValueError('Expected one response per request')
                    g = response.outputs[0]
                    payload.append(dict(prompt_id=job['prompt_id'],study=study,model=spec['repo'],revision=spec['revision'],text=g.text,input_token_ids=job['token_ids'],output_token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason,request_id=response.request_id,seconds=seconds,time=time.time()))
                write(batch_dir/(key+'.raw.json'),payload)
                save_results(con,alias,payload)
            except Exception as error:
                write(batch_dir/(key+'.error.json'),dict(type=type(error).__name__,error=str(error),prompt_ids=ids,time=time.time()))
                # A fatal engine error stops this process. These IDs remain
                # explicitly ambiguous and are skipped, not retried, on resume.
                raise
            if start % (spec['batch_size']*25) == 0:
                log(out,'collecting',judge=alias,study=study,new_completed=start+len(chunk),new_total=len(pending))
        counts = dict(con.execute('SELECT status,count(*) FROM requests WHERE model=? AND study=? GROUP BY status',(alias,study)))
        write(root/(study+'_COVERAGE.json'),dict(expected=len(selected),status_counts=counts))
    queue.close()
    con.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--judge',required=True)
    p.add_argument('--traces-only',action='store_true')
    a = p.parse_args()
    run(a.out,a.judge,a.traces_only)
