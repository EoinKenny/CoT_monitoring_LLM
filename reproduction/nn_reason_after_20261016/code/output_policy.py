"""Outcome policy: model text is data; never search later text for a replacement score."""
import re

def outcome(text,finish_reason):
 first,sep,reason=text.partition('\n')
 valid=re.fullmatch(r'(?:0|[1-9][0-9]?|100)',first.strip()) is not None
 return dict(probability=int(first) if valid else None,reasoning=reason.strip(),reasoning_complete=bool(reason.strip()) and finish_reason=='stop',probability_valid=valid)

def missing(chunk,number,seconds):
 return [dict(sample_id=p['sample_id'],prompt_id=p['prompt_id'],dataset=p['dataset'],backbone=p['backbone'],condition=p['condition'],mode='score',prediction=None,positive_probability=None,prediction_at_50=None,generated_text=None,reasoning='',reasoning_complete=False,token_ids=[],generated_tokens=0,finish_reason='ambiguous_no_raw',input_tokens=p['input_tokens'],batch_number=number,batch_seconds=seconds,instruction_sha256=p['instruction_sha256'],output_status='ambiguous_no_raw') for p in chunk]
