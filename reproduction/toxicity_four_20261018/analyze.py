"""Read-only four-dataset reaggregation of verified, frozen evaluations."""
import json, hashlib, collections, statistics, html
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
DATASETS = ['toxigen', 'toxicchat', 'beavertails', 'aegis']
CONDS = ['baseline', 'saliency', 'nearest']
METRICS = ['f1', 'precision', 'recall', 'accuracy', 'referral_rate']
LABELS = {'baseline':'Control','saliency':'Saliency','nearest':'NN',
          'qwen':'Qwen 27B','qwen122':'Qwen 122B','gemma':'Gemma 31B',
          'glm':'GLM','olmo':'OLMo','mistral':'Mistral',
          'claude4.5-sonnet':'Claude Sonnet','claude4.8-opus':'Claude Opus',
          'gpt-4o':'GPT-4o','gpt-astra':'GPT Astra',
          'toxigen':'ToxiGen','toxicchat':'ToxicChat','beavertails':'BeaverTails','aegis':'Aegis'}
BB = {'qwen':'Qwen','qwen122':'Qwen 122B','gemma':'Gemma','llama':'Llama','llama70':'Llama 70B'}
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, x): (OUT/name).write_text(json.dumps(x, indent=2, allow_nan=False))
def key(r): return tuple(r[k] for k in ['scope','model','backbone','dataset'])
anchor_root = P/'anchor_transfer_20261018/results'
receipt = read(anchor_root/'COMPLETED.json')
oldpath = P/'two_policy_evaluation_20261018/results/cells.json'
assert sha(oldpath) == receipt['inputs'][str(oldpath)]
assert sha(anchor_root/'cells.json') == receipt['outputs']['cells.json']
assert sha(anchor_root/'thresholds.json') == receipt['outputs']['thresholds.json']
for path, expected in receipt['inputs'].items():
    if path.endswith(('scores.npz', 'design.json.gz')): assert sha(Path(path)) == expected
oldfixed = [r for r in read(oldpath) if r['policy']=='fixed_both50']
previous = {(key(r),r['condition']):r for r in read(P/'both_fixed50_20261018/cells.json')}
for r in oldfixed:
    v = previous[key(r),r['condition']]
    assert r['test_n'] == v['test_n']
    assert r['probe_threshold']==v['probe_threshold']==0.5
    assert r['judge_threshold']==v['judge_threshold']==50
    for metric in METRICS[:4]: assert abs(r[metric]-v[metric]) < 1e-12
rows=[]
for r in oldfixed:
    if r['dataset'] in DATASETS: rows.append(dict(r,anchor='fixed50',analysis='fixed50'))
for r in read(anchor_root/'cells.json'):
    if r['dataset'] in DATASETS: rows.append(dict(r,analysis='tuned'))
assert len(rows)==4800
paired=collections.defaultdict(list)
for r in rows: paired[key(r),r['anchor']].append(r)
for rr in paired.values():
    assert {r['condition'] for r in rr}==set(CONDS)
    assert len({(r['probe_threshold'],r['judge_threshold'],r['test_n'],r['test_referred']) for r in rr})==1
    if rr[0]['judge_threshold']==0:
        for m in METRICS: assert max(r[m] for r in rr)-min(r[m] for r in rr)<1e-12
summary=[]
for scope in ['combined','local_full','api_matched','common_valid']:
    ss=[r for r in rows if (scope=='combined' and r['scope']!='common_valid') or r['scope']==scope]
    for policy in ['fixed50','baseline','nearest','saliency','average_of_three']:
        pp=[r for r in ss if (policy=='average_of_three' and r['analysis']=='tuned') or r['anchor']==policy]
        for dim in ['overall','model','dataset','backbone']:
            names=['ALL'] if dim=='overall' else sorted({r[dim] for r in pp})
            for name in names:
                gg=[r for r in pp if dim=='overall' or r[dim]==name]
                result=dict(scope=scope,policy=policy,dimension=dim,group=name,cells=len({key(r) for r in gg}))
                for cond in CONDS:
                    cc=[r for r in gg if r['condition']==cond]
                    result[cond]={m:statistics.mean(r[m] for r in cc) for m in METRICS}
                    result[cond]['zero_judge_fraction']=statistics.mean(r['judge_threshold']==0 for r in cc)
                    result[cond]['no_test_veto_fraction']=statistics.mean(r['test_rejected']==0 for r in cc)
                for a,b in [('nearest','baseline'),('saliency','baseline'),('nearest','saliency')]:
                    result[a+'-'+b]={m:result[a][m]-result[b][m] for m in METRICS}
                    indexed={(key(r),r['anchor'],r['condition']):r for r in gg}
                    for m in METRICS:
                        paired_delta=statistics.mean(r[m]-indexed[key(r),r['anchor'],b][m] for r in gg if r['condition']==a)
                        assert abs(paired_delta-result[a+'-'+b][m])<1e-12
                summary.append(result)
save('summary.json',summary)
save('cells.json',rows)
thresholds=[r for r in read(anchor_root/'thresholds.json') if r['dataset'] in DATASETS]
save('thresholds.json',thresholds)
coverage=[]
for (scope,model),rr in __import__('itertools').groupby(sorted([r for r in rows if r['anchor']=='fixed50' and r['condition']=='baseline'],key=lambda r:(r['scope'],r['model'])),key=lambda r:(r['scope'],r['model'])):
    rr=list(rr);coverage.append(dict(scope=scope,model=model,cells=len(rr),test_case_judge_records=sum(r['test_n'] for r in rr),minimum_per_cell=min(r['test_n'] for r in rr),maximum_per_cell=max(r['test_n'] for r in rr)))
save('coverage.json',coverage)
protocol=dict(datasets=DATASETS,reason='Four content-safety/toxicity datasets discussed with the user; excludes WildGuard and WildJailbreak from this specified subset.',
    estimand='Equal-weight macro means over 200 judge-backbone-dataset cells (10 judges x 5 backbones x 4 datasets). Dataset and judge selection unchanged within subset.',
    fixed='Probe probability >=0.5 AND judge probability >=50. No threshold fitting.',
    tuned='Reuse original per-cell validation-only full-grid F1 selections separately for control, NN and saliency. Freeze BOTH selected cutoffs across all three explanations. Average paired deltas equally across the three sources, then across cells. No new fitting or test selection is needed because original fitting was per cell.',
    scope='Primary combines 120 local full-test design-weighted cells with 80 API condition-matched valid-case cells. Common-valid-all-judges sensitivity is separate. Same masks within paired comparisons. The average across sources is not a deployable threshold.',
    limits='Previously examined datasets; exploratory subset, not an independent confirmation. Point estimates, no new significance claim. Zero-cutoff and no-veto cells retained; repeated targets and threshold sources are not independent samples.',new_model_calls=0)
save('PROTOCOL.json',protocol)
parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Toxicity and content safety: four-dataset analysis</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172335}table{border-collapse:collapse;width:100%;margin:18px 0 35px;font-variant-numeric:tabular-nums}td,th{padding:9px 12px;border-bottom:1px solid #dbe1e8;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f8}u{text-decoration-thickness:2px;text-underline-offset:4px}p{line-height:1.5}details{margin:25px 0}h2{margin-top:40px}small{color:#536174}</style><h1>Toxicity and content safety: four datasets</h1>']
parts.append('<p>ToxiGen, ToxicChat, BeaverTails and Aegis. Ten judges × five backbones × four datasets = 200 combinations. All values are percentages; differences are percentage points. Highest displayed performance in each row is underlined. No model calls or retraining.</p>')
parts.append('<p>Fixed: probe ≥0.5 and judge ≥50. Tuned: choose both cutoffs on validation for one explanation condition, apply that same pair to all conditions, repeat for all three selection sources, then average. Thresholds remain separate for each judge–backbone–dataset combination.</p>')
parts.append('<p>Primary means combine local full-test estimates and API matched-valid estimates. Results on the same valid cases across every judge are supplied separately below. Design weights are preserved. These are exploratory subset point estimates. Zero judge cutoffs remain included; three-source means are sensitivity summaries, not three independent experiments.</p>')
def table(rs,metric='f1'):
    out=['<table><tr><th>Group</th><th>Cells</th><th>Control</th><th>Saliency</th><th>NN</th><th>Saliency − control</th><th>NN − control</th><th>NN − saliency</th></tr>']
    for r in rs:
        name='Overall' if r['dimension']=='overall' else (BB if r['dimension']=='backbone' else LABELS).get(r['group'],r['group'])
        vals=[100*r[c][metric] for c in CONDS];best=max(round(v,2) for v in vals)
        vstr=[f'<u>{v:.2f}</u>' if round(v,2)==best else f'{v:.2f}' for v in vals]
        out.append('<tr><td>'+html.escape(name)+'</td><td>'+str(r['cells'])+'</td>'+''.join('<td>'+v+'</td>' for v in vstr)+''.join(f'<td>{100*r[c][metric]:+.2f}</td>' for c in ['saliency-baseline','nearest-baseline','nearest-saliency'])+'</tr>')
    return ''.join(out)+'</table>'
for scope in ['combined','common_valid']:
    if scope=='common_valid':parts.append('<details><summary>Common-valid-all-judges sensitivity</summary>')
    for policy,title in [('fixed50','Fixed thresholds: 50/50'),('average_of_three','Controlled tuning: average across three selection sources')]:
        parts.append('<h2>'+title+'</h2>')
        for dim,title2 in [('overall','Overall'),('model','By judge'),('dataset','By dataset'),('backbone','By backbone')]:
            rr=[r for r in summary if r['scope']==scope and r['policy']==policy and r['dimension']==dim]
            rr.sort(key=lambda r:r['nearest-baseline']['f1'],reverse=True)
            parts.append('<h3>'+title2+' — F1</h3>'+table(rr))
            parts.append('<details><summary>Precision, recall and accuracy</summary>')
            for metric in ['precision','recall','accuracy']:parts.append('<h4>'+metric.title()+'</h4>'+table(rr,metric))
            parts.append('</details>')
    parts.append('<h2>Overall result for each tuning source</h2>')
    for anchor in ['baseline','nearest','saliency']:
        parts.append('<h3>Thresholds selected using '+LABELS[anchor]+'</h3>'+table([r for r in summary if r['scope']==scope and r['policy']==anchor and r['dimension']=='overall']))
    if scope=='common_valid':parts.append('</details>')
parts.append('<details><summary>Exact primary thresholds: 600 selected pairs</summary><table><tr><th>Judge</th><th>Backbone</th><th>Dataset</th><th>Selection source</th><th>Probe cutoff</th><th>Judge cutoff</th></tr>')
for r in thresholds:
    if r['scope']=='common_valid':continue
    parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [LABELS.get(r['model'],r['model']),BB[r['backbone']],LABELS[r['dataset']],LABELS[r['anchor']],f"{r['probe_threshold']:.8g}",r['judge_threshold']])+'</tr>')
parts.append('</table></details><h2>Coverage</h2><pre>'+html.escape(json.dumps(coverage,indent=2))+'</pre><p>Source identifiers, checksums and machine-readable results are saved alongside this report.</p></html>')
(OUT/'report.html').write_text(''.join(parts))
save('VERIFIED.json',dict(fixed_replays=len(oldfixed),rows=len(rows),primary_cells=200,paired_thresholds_masks_and_referrals_verified=True,paired_delta_aggregation_verified=True,new_model_calls=0,inputs={str(p):sha(p) for p in [oldpath,anchor_root/'cells.json',anchor_root/'thresholds.json',P/'both_fixed50_20261018/cells.json',Path(__file__)]},outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='VERIFIED.json'}))
for policy in ['fixed50','baseline','nearest','saliency','average_of_three']:
    print('\nPOLICY',policy)
    for r in summary:
        if r['scope']=='combined' and r['policy']==policy and (policy in ['fixed50','average_of_three'] or r['dimension']=='overall'):
            print(r['dimension'],r['group'],*[round(100*r[c]['f1'],3) for c in CONDS], 'delta',round(100*r['nearest-baseline']['f1'],3))
