"""Reasoning-first: only a completed, final standalone probability is scored."""
import re
def outcome(text,finish_reason):
    reason,sep,last=text.strip().rpartition('\n')
    valid=bool(sep and reason.strip()) and finish_reason=='stop' and re.fullmatch(r'(?:0|[1-9][0-9]?|100)',last.strip()) is not None
    return dict(probability=int(last) if valid else None,reasoning=reason.strip() if valid else text.strip(),reasoning_complete=valid,probability_valid=valid)
def missing(chunk,number,seconds):
    return [dict(sample_id=p['sample_id'],prompt_id=p['prompt_id'],dataset=p['dataset'],backbone=p['backbone'],condition=p['condition'],mode='score',prediction=None,positive_probability=None,prediction_at_50=None,generated_text=None,reasoning='',reasoning_complete=False,token_ids=[],generated_tokens=0,finish_reason='ambiguous_no_raw',input_tokens=p['input_tokens'],batch_number=number,batch_seconds=seconds,instruction_sha256=p['instruction_sha256'],output_status='ambiguous_no_raw') for p in chunk]
