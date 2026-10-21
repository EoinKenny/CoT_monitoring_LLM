"""Gemma's native turn delimiters, retaining every frozen dataset message.

The dataset has role/content pairs without structured tool-call metadata. The
canonical template silently drops role:tool in this case. Render those turns
with their original tool role and literal content. Do not strip channel-like
strings from supplied assistant data. Judge prompt templates are unaffected.
"""
def render_messages(tokenizer,messages):
 template=tokenizer.chat_template
 replacements={
  "{%- if message['role'] != 'tool' -%}":"{%- if true -%}",
  "{%- if loop_messages[j]['role'] != 'tool' -%}":"{%- if true -%}",
  "strip_thinking(message['content'])":"message['content'] | trim",
  "strip_thinking(item['text'])":"item['text'] | trim",
 }
 for old,new in replacements.items():
  assert template.count(old)==1,('Unexpected pinned template',old)
  template=template.replace(old,new)
 return tokenizer.apply_chat_template(messages,chat_template=template,tokenize=False,add_generation_prompt=False,enable_thinking=False,date_string='01 Oct 2026')
