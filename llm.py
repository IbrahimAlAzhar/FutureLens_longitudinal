"""Local vLLM HTTP client, full request cache and tokenizer-bounded sentence batches."""
import json, time, urllib.request, urllib.error
from pathlib import Path
from common import digest, write_json


def response_schema(prompt):
    if prompt.startswith("Label this research topic"):
        return {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        }
    return None

class Client:
    def __init__(self, args):
        self.url=args.vllm_url.rstrip('/'); self.model=args.model_dir
        self.seed=args.seed; self.cache=Path(args.out_dir)/'requests'; self.cache.mkdir(parents=True,exist_ok=True)
        self.max_tokens=args.max_tokens; self.context=args.context_length; self.qwen='qwen' in args.model.lower()
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(self.model,local_files_only=True)
        from chat_templates import resolve
        self.chat_template,self.template_source=resolve(self.tokenizer,args.model,self.model,getattr(args,'chat_template',None))
    def ask(self, prompt):
        body=dict(model=self.model,messages=[{'role':'user','content':prompt}],temperature=0,
                  seed=self.seed,max_tokens=self.max_tokens)
        schema = response_schema(prompt)
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "futurescope_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        if self.qwen: body['chat_template_kwargs']={'enable_thinking':False}
        # Count the actual chat template, including control tokens.
        ids=self.tokenizer.apply_chat_template(body['messages'],chat_template=self.chat_template,tokenize=True,add_generation_prompt=True,
                                               **({'enable_thinking':False} if self.qwen else {}))
        if len(ids)+self.max_tokens+32>self.context: raise ValueError('Prompt exceeds context; reduce --chunk-tokens')
        key=digest({'request':body,'chat_template':self.chat_template}); path=self.cache/f'{key}.json'
        if path.exists(): return json.loads(path.read_text())['text']
        for attempt in range(4):
            try:
                req=urllib.request.Request(self.url+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=600) as f: response=json.load(f)
                if not isinstance(response,dict) or not isinstance(response.get('choices'),list) or not response['choices']:
                    raise ValueError('Invalid vLLM response envelope: expected object containing choices')
                choice=response['choices'][0]
                if not isinstance(choice,dict) or not isinstance(choice.get('message'),dict):
                    raise ValueError('Invalid vLLM choice/message object')
                if choice.get('finish_reason')=='length':
                    write_json(self.cache/f'{key}.truncated.json',dict(request=body,response=response))
                    raise ValueError('LLM output truncated; increase --max-tokens or decrease --chunk-tokens')
                text=choice['message'].get('content') or ''
                if not isinstance(text,str): raise ValueError('vLLM message content must be a text string')
                write_json(path,dict(request=body,response=response,text=text,prompt_tokens_local=len(ids),template_source=self.template_source))
                return text
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code in (400, 422):
                    raise RuntimeError(f"vLLM rejected the request (HTTP {e.code}): {detail}") from e
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
            except (urllib.error.URLError,TimeoutError):
                if attempt==3: raise
                time.sleep(2**attempt)
def parse_json(text):
    import re
    text=re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if text.startswith('```'): text='\n'.join(text.splitlines()[1:-1])
    try: return json.loads(text)
    except json.JSONDecodeError:
        start=text.find('{'); end=text.rfind('}')
        if start>=0 and end>start: return json.loads(text[start:end+1])
        raise


def topic_label(text):
    obj=parse_json(text)
    if isinstance(obj,list) and len(obj)==1: obj=obj[0]
    label=obj.get('label') if isinstance(obj,dict) else None
    if not isinstance(label,str) or not label.strip(): raise ValueError('Expected {"label":"nonempty title"}')
    return label.strip()

def ask_validated(client,prompt,validator):
    errors=[]
    for attempt in range(3):
        correction='' if not attempt else '\nYour previous response had an invalid JSON shape. Return only the exact JSON object specified above. Do not use prose or code fences. Correction attempt '+str(attempt)+'.'
        text=client.ask(prompt+correction)
        try: return validator(text)
        except (ValueError,TypeError) as e: errors.append(str(e))
    raise ValueError('Invalid model output after 3 attempts; raw responses remain in requests/: '+errors[-1])
