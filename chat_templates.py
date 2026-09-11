"""Resolve one template for both local token counting and vLLM serving."""
import argparse,json
from pathlib import Path

# Restricted to the single-user, text-only requests made by this pipeline.
# Mistral Small 3.1 token names verified against its official tokenizer_config.json.
MISTRAL_SINGLE_USER = """{%- if messages|length != 1 or messages[0]['role'] != 'user' or messages[0]['content'] is not string -%}
{{- raise_exception('FutureScope Mistral fallback supports exactly one text user message') -}}
{%- endif -%}
{{- bos_token + '[INST]' + messages[0]['content'] + '[/INST]' -}}"""

def resolve(tokenizer,model,model_dir,explicit=None):
    if explicit:
        template=Path(explicit).read_text()
        if not template.strip(): raise ValueError('Explicit chat template is empty')
        return template,'explicit_file'
    native=getattr(tokenizer,'chat_template',None)
    if native:
        if isinstance(native,str): return native,'tokenizer'
        # Let Transformers select a default from its named-template collection.
        return tokenizer.get_chat_template(),'tokenizer_named_default'
    root=Path(model_dir)
    for name in ['chat_template.jinja','chat_template.json','processor_config.json','tokenizer_config.json']:
        p=root/name
        if not p.exists(): continue
        if p.suffix=='.jinja': template=p.read_text()
        else:
            obj=json.loads(p.read_text());template=obj.get('chat_template') if isinstance(obj,dict) else None
        if isinstance(template,str) and template.strip(): return template,name
    if model=='mistral':
        vocab=tokenizer.get_vocab()
        if not tokenizer.bos_token or any(t not in vocab for t in ['[INST]','[/INST]',tokenizer.bos_token]):
            raise ValueError('Mistral fallback tokens absent. Set CHAT_TEMPLATE to a verified local template file.')
        return MISTRAL_SINGLE_USER,'mistral_single_user_text_fallback'
    raise ValueError('No chat template found. Set CHAT_TEMPLATE to a verified local template file.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--model-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--template',default=None);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    template,source=resolve(tok,a.model,a.model_dir,a.template)
    tok.apply_chat_template([{'role':'user','content':'Select future work.'}],chat_template=template,tokenize=True,add_generation_prompt=True)
    Path(a.output).write_text(template)
    from common import write_json,digest
    write_json(Path(a.output).with_suffix('.meta.json'),dict(source=source,sha256=digest(template)))
    print('[chat template] '+source,flush=True)

if __name__=='__main__': main()
