"""Native backbone tokenization and exact content/sentence span alignment."""
import copy
import json

from .common import digest


def render(tok, messages, backbone):
    kwargs = dict(tokenize=False, add_generation_prompt=False, date_string='01 Oct 2026')
    if backbone not in ['llama', 'llama70']:
        kwargs['enable_thinking'] = False
    if backbone == 'qwen122' and not any(m['role'] == 'user' for m in messages):
        messages = copy.deepcopy(messages)
        messages.insert(1 if messages[0]['role'] == 'system' else 0, dict(role='user', content=''))
    if backbone == 'gemma':
        template = tok.chat_template
        for old, new in {
            "{%- if message['role'] != 'tool' -%}": "{%- if true -%}",
            "{%- if loop_messages[j]['role'] != 'tool' -%}": "{%- if true -%}",
            "strip_thinking(message['content'])": "message['content'] | trim",
            "strip_thinking(item['text'])": "item['text'] | trim",
        }.items():
            if template.count(old) != 1:
                raise ValueError('Pinned Gemma template changed')
            template = template.replace(old, new)
        kwargs['chat_template'] = template
    return tok.apply_chat_template(messages, **kwargs)


def escaped(text, ascii, html):
    value = json.dumps(text, ensure_ascii=ascii)[1:-1]
    if html:
        for a, b in [('<','\\u003c'),('>','\\u003e'),('&','\\u0026'),("'",'\\u0027')]:
            value = value.replace(a,b)
    return value


def align(tok, row, backbone):
    text = render(tok, row['messages'], backbone)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if 'token_ids' in row and enc['input_ids'] != row['token_ids']:
        raise ValueError(f'Native tokenization differs from frozen input: {row["immutable_id"]}')
    spans = []
    for mi, message in enumerate(row['messages']):
        if row['role'] != 'all' and message['role'] != row['role']:
            continue
        content = message['content']
        if not content.strip():
            continue
        if backbone in ['llama', 'llama70']:
            prefix = render(tok, row['messages'][:mi+1], backbone)
            if not text.startswith(prefix):
                raise ValueError('Llama chat prefix changed')
            if message['role'] in ['tool','ipython']:
                previous = render(tok, row['messages'][:mi], backbone) if mi else ''
                header = '<|start_header_id|>ipython<|end_header_id|>\n\n'
                start = prefix.find(header, len(previous)) + len(header)
                body = prefix[start:-len('<|eot_id|>')]
                if json.loads(body) != content:
                    raise ValueError('Tool content did not round-trip')
                mode = next(( (a,h) for a in [True,False] for h in [True,False] if '"'+escaped(content,a,h)+'"' == body), None)
                if mode is None:
                    raise ValueError('Unknown tool JSON escaping')
                spans.append(dict(start=start+1, end=len(prefix)-len('<|eot_id|>')-1, text=content, message_index=mi, escaped=mode))
            else:
                body = content.strip()
                start = prefix.rfind(body)
                if start < 0:
                    raise ValueError('Nonliteral Llama content')
                spans.append(dict(start=start, end=start+len(body), text=body, message_index=mi))
        else:
            marker = 'PROBE_CONTENT_SENTINEL_'+digest(str(mi)+content)
            changed = copy.deepcopy(row['messages'])
            changed[mi]['content'] = marker
            marked = render(tok, changed, backbone)
            if marked.count(marker) != 1:
                raise ValueError('Missing content sentinel')
            before, after = marked.split(marker)
            body = next((x for x in [content, content.strip()] if text == before+x+after), None)
            if body is None:
                raise ValueError('Nonliteral content rendering requires review')
            spans.append(dict(start=len(before), end=len(before)+len(body), text=body, message_index=mi))
    special = set(tok.all_special_ids)
    mask = [i for i, (tid, (a,b)) in enumerate(zip(enc['input_ids'], enc['offset_mapping'])) if tid not in special and b > a and any(b>s['start'] and a<s['end'] for s in spans)]
    if not mask:
        raise ValueError('Empty content mask')
    if 'mask' in row and mask != row['mask']:
        raise ValueError(f'Content mask differs from frozen input: {row["immutable_id"]}')
    return enc['input_ids'], mask, enc['offset_mapping'], spans


def tokenize_missing(tok, rows, backbone):
    for row in rows:
        if 'token_ids' not in row:
            ids, mask, _, _ = align(tok, row, backbone)
            row.update(token_ids=ids, mask=mask, n_tokens=len(ids))
    return rows


def sentence_scores(tok, row, backbone, scores):
    import numpy as np
    import pysbd
    _, mask_indices, offsets, spans = align(tok, row, backbone)
    mask = np.isin(np.arange(len(scores)), mask_indices)
    offsets = np.array(offsets)
    segmenter = pysbd.Segmenter(language='en', clean=False, char_span=True)
    candidates = []
    for content in spans:
        def charpos(x):
            if 'escaped' in content:
                return content['start'] + len(escaped(content['text'][:x], *content['escaped']))
            return content['start']+x
        for span in segmenter.segment(content['text']):
            a, b = charpos(span.start), charpos(span.end)
            selected = mask & (offsets[:,1] > a) & (offsets[:,0] < b)
            if selected.any():
                mi = content['message_index']
                candidates.append(dict(sentence=span.sent.strip(), message_index=mi, role=row['messages'][mi]['role'], mean_logit=float(scores[selected].mean()), tokens=int(selected.sum())))
    if not candidates:
        raise ValueError('No content sentences')
    return max(candidates, key=lambda x: x['mean_logit'])
